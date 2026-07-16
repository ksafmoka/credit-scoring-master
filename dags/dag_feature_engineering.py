"""
Feature engineering DAG — batched for 2.2M + cold-start aware sampling.

Architecture for portfolio:
- Raw 2.2M always full (data lake)
- Ingestion: 300k random payments (PAYMENT_HISTORY_MAX_APPS)
- FE: 400k final = 300k with history (from payment_history) + 100k without (cold start 25%)
  => 75% coverage for aggregation, model learns to handle thin-file clients
- Tasks sequential + batched writes to avoid Postgres DNS overload
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="feature_engineering",
    default_args=default_args,
    description="Batched FE 400k = 300k with payments (75%) + 100k cold-start (25%)",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features", "ml"],
)


def _count_raw(engine) -> int:
    from sqlalchemy import text

    with engine.connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM raw.applications")).scalar() or 0)


def _write_df(df, table: str, engine, batch_size: int | None = None) -> None:
    from sqlalchemy import text

    from src.config import FeatureEngineeringConfig

    batch_size = batch_size or FeatureEngineeringConfig.STAGING_WRITE_BATCH_SIZE

    if df is None:
        raise ValueError(f"Refusing to write None to {table}")
    if df.empty:
        print(f"Writing features.{table}: EMPTY")
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS features.{table}"))
            conn.execute(text(f"CREATE TABLE features.{table} (application_id BIGINT)"))
        return

    total = len(df)
    print(f"Writing features.{table}: rows={total}, batch_size={batch_size}")
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS features.{table}"))

    for start in range(0, total, batch_size):
        chunk = df.iloc[start : start + batch_size]
        with engine.begin() as conn:
            chunk.to_sql(
                table,
                conn,
                schema="features",
                if_exists="append",
                index=False,
                method="multi",
                chunksize=10_000,
            )
        print(f"  -> {table} batch {start // batch_size + 1}: {len(chunk)} ({min(start + batch_size, total)}/{total})")


def select_fe_ids(**context):
    """
    Create features.fe_ids table with cold-start aware sampling:
    FE = payment_ids (300k) + additional random without history (100k) = 400k total = 75% coverage
    If payment_history empty, fallback to pure random FE_MAX_APPS.
    This table is the single source of truth for all downstream FE tasks -> consistent ids.
    """
    from sqlalchemy import text

    from src.config import FeatureEngineeringConfig, IngestionConfig, get_db_engine
    import pandas as pd

    engine = get_db_engine()
    fe_max = FeatureEngineeringConfig.MAX_APPS or 400000
    payment_max = IngestionConfig.PAYMENT_HISTORY_MAX_APPS or 300000
    strategy = IngestionConfig.SAMPLE_STRATEGY

    # Get distinct payment ids
    try:
        with engine.connect() as conn:
            payment_df = pd.read_sql(
                text("SELECT DISTINCT application_id FROM raw.payment_history"), conn
            )
            payment_ids = payment_df["application_id"].tolist()
    except Exception as exc:
        print(f"[select_fe_ids] payment_history read failed {exc}, fallback random")
        payment_ids = []

    print(f"[select_fe_ids] payment_history distinct ids: {len(payment_ids)} (expected ~{payment_max})")

    # If no payment history (first run), pure random FE
    if not payment_ids:
        print(f"[select_fe_ids] No payment ids, selecting {fe_max} random from raw")
        with engine.connect() as conn:
            fe_df = pd.read_sql(
                text(f"SELECT application_id FROM raw.applications ORDER BY random() LIMIT :limit"),
                conn,
                params={"limit": int(fe_max)},
            )
        fe_ids = fe_df["application_id"].tolist()
    else:
        if len(payment_ids) >= fe_max:
            # 100% coverage case: FE is subset of payment ids
            import random

            random.seed(42)
            fe_ids = random.sample(payment_ids, fe_max)
            print(f"[select_fe_ids] FE subset of payments: {fe_max} / {len(payment_ids)} -> 100% coverage")
        else:
            # 75% case: 300k with history + 100k without
            needed = fe_max - len(payment_ids)
            print(f"[select_fe_ids] Need additional {needed} without history for cold-start")
            with engine.connect() as conn:
                additional_df = pd.read_sql(
                    text(
                        """
                        SELECT application_id FROM raw.applications
                        WHERE application_id NOT IN (SELECT DISTINCT application_id FROM raw.payment_history)
                        ORDER BY random() LIMIT :limit
                        """
                    ),
                    conn,
                    params={"limit": int(needed)},
                )
            additional_ids = additional_df["application_id"].tolist()
            fe_ids = payment_ids + additional_ids
            print(
                f"[select_fe_ids] FE = {len(payment_ids)} with history + {len(additional_ids)} cold-start = {len(fe_ids)} total, coverage {len(payment_ids)/len(fe_ids):.1%}"
            )

    # Write to features.fe_ids
    fe_df = pd.DataFrame({"application_id": fe_ids})
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features.fe_ids"))
        fe_df.to_sql(
            "fe_ids",
            conn,
            schema="features",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=20_000,
        )
    print(f"[select_fe_ids] Wrote features.fe_ids rows={len(fe_df)}")
    context["ti"].xcom_push(key="fe_ids_count", value=len(fe_df))
    context["ti"].xcom_push(key="coverage", value=len(payment_ids) / len(fe_ids) if fe_ids else 0)


def _get_fe_ids_filter(engine) -> str | None:
    """Return SQL filter for fe_ids if table exists, else None."""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            exists = conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='features' AND table_name='fe_ids')"
                )
            ).scalar()
            if exists:
                return "application_id IN (SELECT application_id FROM features.fe_ids)"
    except Exception:
        pass
    return None


def compute_numerical_features(**context):
    from sqlalchemy import text

    from src.config import FeatureEngineeringConfig, get_db_engine
    from src.features.numerical import NumericalFeatureComputer
    import pandas as pd

    engine = get_db_engine()
    n_raw = _count_raw(engine)
    print(f"raw.applications COUNT(*) = {n_raw}")

    fe_filter = _get_fe_ids_filter(engine)
    if fe_filter:
        print(f"[numerical] Using fe_ids filter: {fe_filter}")
        count_q = f"SELECT COUNT(*) FROM features.fe_ids"
        with engine.connect() as conn:
            total = int(conn.execute(text(count_q)).scalar() or 0)
    else:
        max_apps = FeatureEngineeringConfig.MAX_APPS
        total = min(n_raw, max_apps) if max_apps else n_raw

    batch_size = FeatureEngineeringConfig.NUMERICAL_BATCH_SIZE
    print(f"Numerical: total={total}, batch_size={batch_size}")

    computer = NumericalFeatureComputer()
    all_features = []

    for offset in range(0, total, batch_size):
        limit = min(batch_size, total - offset)

        if fe_filter:
            query = f"""
                SELECT a.* FROM raw.applications a
                WHERE {fe_filter}
                ORDER BY a.application_id
                LIMIT :limit OFFSET :offset
            """
            params = {"limit": limit, "offset": offset}
        else:
            query = """
                SELECT * FROM raw.applications
                ORDER BY application_id
                LIMIT :limit OFFSET :offset
            """
            params = {"limit": limit, "offset": offset}

        with engine.connect() as conn:
            chunk = pd.read_sql(text(query), conn, params=params)

        if chunk.empty:
            continue

        print(f"Numerical batch {offset // batch_size + 1}: {len(chunk)} ({offset + len(chunk)}/{total})")
        feats = computer.compute(chunk)
        all_features.append(feats)

    if not all_features:
        raise ValueError("Numerical 0 rows")

    features = pd.concat(all_features, ignore_index=True)
    print(f"numerical shape: {features.shape}")
    _write_df(features, "numerical_staging", engine)


def compute_aggregation_features(**context):
    from src.config import FeatureEngineeringConfig, get_db_engine
    from src.features.aggregations import AggregationFeatureComputer

    engine = get_db_engine()
    computer = AggregationFeatureComputer(reference_date=None)

    try:
        from src.config import FeatureConfig

        computer.windows = FeatureConfig.PAYMENT_WINDOWS
    except Exception:
        pass

    # Aggregation now respects fe_ids table via its own batch logic, but we also pass filter hint
    features = computer.compute(
        engine,
        batch_size=FeatureEngineeringConfig.AGGREGATION_BATCH_SIZE,
        max_apps=FeatureEngineeringConfig.MAX_APPS,
    )
    print(f"aggregation shape: {features.shape}")

    # Fallback: ensure fe_ids present even if no payments
    if features.empty or "application_id" not in features.columns:
        from sqlalchemy import text
        import pandas as pd

        with engine.connect() as conn:
            # Try fe_ids first
            try:
                features = pd.read_sql(text("SELECT application_id FROM features.fe_ids"), conn)
                print(f"aggregation fallback fe_ids: {len(features)}")
            except Exception:
                features = pd.read_sql(text("SELECT application_id FROM raw.applications LIMIT 1000"), conn)

    _write_df(features, "aggregation_staging", engine)


def compute_target_encoding(**context):
    from src.config import ARTIFACTS_DIR, FeatureEngineeringConfig, TrainingConfig, get_db_engine
    from src.features.target_encoding import RegularizedTargetEncoder

    engine = get_db_engine()
    encoder = RegularizedTargetEncoder(cols=["purpose", "home_ownership"], smoothing=20, noise_level=0.0)

    result = encoder.fit_transform(
        engine,
        train_cutoff=TrainingConfig.TRAIN_END_DATE,
        execution_date=None,
        batch_size=FeatureEngineeringConfig.TARGET_ENCODING_BATCH_SIZE,
        max_apps=FeatureEngineeringConfig.MAX_APPS,
    )
    if result.empty:
        raise ValueError("Target encoding 0 rows")
    print(f"target encoding shape: {result.shape}")
    encoder.save(ARTIFACTS_DIR / "target_encoding.json")
    _write_df(result, "target_enc_staging", engine)


def compute_bureau_features(**context):
    from src.config import FeatureEngineeringConfig, get_db_engine
    from src.features.bureau import BureauFeatureComputer

    engine = get_db_engine()
    features = BureauFeatureComputer().compute(
        engine,
        reference_date=None,
        batch_size=FeatureEngineeringConfig.BUREAU_BATCH_SIZE,
        max_apps=FeatureEngineeringConfig.MAX_APPS,
    )
    print(f"bureau shape: {features.shape}")
    if features.empty or "application_id" not in features.columns:
        from sqlalchemy import text
        import pandas as pd

        with engine.connect() as conn:
            try:
                features = pd.read_sql(text("SELECT application_id FROM features.fe_ids"), conn)
            except Exception:
                features = pd.read_sql(text("SELECT application_id FROM raw.applications LIMIT 1000"), conn)
    _write_df(features, "bureau_staging", engine)


def merge_and_validate(**context):
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    from src.config import FeatureConfig, FeatureEngineeringConfig, get_db_engine
    from src.data.validation import validate_feature_table

    engine = get_db_engine()
    n_raw = _count_raw(engine)
    print(f"[merge] raw COUNT={n_raw}")

    def _read(name):
        try:
            with engine.connect() as conn:
                return pd.read_sql(text(f"SELECT * FROM features.{name}"), conn)
        except Exception as exc:
            print(f"[merge] no {name}: {exc}")
            return pd.DataFrame()

    num = _read("numerical_staging")
    agg = _read("aggregation_staging")
    te = _read("target_enc_staging")
    bureau = _read("bureau_staging")
    print(f"[merge] staging: num={len(num)} agg={len(agg)} te={len(te)} bureau={len(bureau)}")

    if num.empty:
        raise ValueError("numerical_staging empty")

    for frame in (num, agg, te, bureau):
        if not frame.empty and "application_id" in frame.columns:
            frame["application_id"] = pd.to_numeric(frame["application_id"], errors="coerce")

    merged = num.copy()
    if not agg.empty and "application_id" in agg.columns:
        merged = merged.merge(agg, on="application_id", how="left")
    if not te.empty and "application_id" in te.columns:
        merged = merged.merge(te, on="application_id", how="left")
    if not bureau.empty and "application_id" in bureau.columns:
        merged = merged.merge(bureau, on="application_id", how="left")

    print(f"[merge] merged shape: {merged.shape}, columns with NaN agg: {merged['avg_days_overdue_90d'].isna().sum() if 'avg_days_overdue_90d' in merged else 'N/A'}")

    for col in FeatureConfig.ALL_FEATURES:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    for col in FeatureConfig.CRITICAL_NO_NULL:
        if col in merged.columns and merged[col].isna().any():
            med = merged[col].median()
            merged[col] = merged[col].fillna(0.0 if pd.isna(med) else med)

    for col in list(FeatureConfig.ALL_FEATURES) + ["avg_account_age_months"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    merged = merged.dropna(subset=["application_id"])
    merged["application_id"] = merged["application_id"].astype(np.int64)

    if len(merged) == 0:
        raise ValueError("Merged 0 rows")

    report = validate_feature_table(
        merged,
        checks={
            "no_nulls_in_critical": FeatureConfig.CRITICAL_NO_NULL,
            "value_ranges": {"loan_to_income": (0, 1000)},
            "row_count_min": 1,
        },
    )
    if not report.success:
        raise ValueError(f"Validation failed: {report.errors}")

    merged["feature_version"] = FeatureConfig.VERSION
    if "feature_date" not in merged.columns:
        merged["feature_date"] = pd.Timestamp.utcnow().normalize()
    merged["feature_date"] = pd.to_datetime(merged["feature_date"], errors="coerce").dt.date
    if merged["feature_date"].isna().any():
        merged["feature_date"] = merged["feature_date"].fillna(pd.Timestamp.utcnow().date())

    keep = (
        ["application_id", "feature_date", "feature_version"]
        + FeatureConfig.ALL_FEATURES
        + ["dti_bucket", "credit_score_bucket", "avg_account_age_months"]
    )
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep].copy()

    create_sql = """
    CREATE TABLE features.application_features (
        application_id              BIGINT PRIMARY KEY,
        feature_date                DATE NOT NULL,
        loan_to_income              DOUBLE PRECISION,
        credit_utilization          DOUBLE PRECISION,
        income_log                  DOUBLE PRECISION,
        loan_amount_log             DOUBLE PRECISION,
        dti_ratio_clipped           DOUBLE PRECISION,
        employment_years            DOUBLE PRECISION,
        credit_score_norm           DOUBLE PRECISION,
        num_open_accounts           DOUBLE PRECISION,
        num_delinquencies           DOUBLE PRECISION,
        interest_rate               DOUBLE PRECISION,
        loan_amount_x_dti           DOUBLE PRECISION,
        income_x_credit_score       DOUBLE PRECISION,
        dti_bucket                  VARCHAR(20),
        credit_score_bucket         VARCHAR(20),
        avg_days_overdue_30d        DOUBLE PRECISION,
        avg_days_overdue_90d        DOUBLE PRECISION,
        avg_days_overdue_180d       DOUBLE PRECISION,
        max_days_overdue_90d        DOUBLE PRECISION,
        pct_late_payments_90d       DOUBLE PRECISION,
        total_paid_90d              DOUBLE PRECISION,
        payment_consistency_90d     DOUBLE PRECISION,
        bureau_balance_to_income    DOUBLE PRECISION,
        inquiries_per_account       DOUBLE PRECISION,
        avg_account_age_months      DOUBLE PRECISION,
        purpose_target_enc          DOUBLE PRECISION,
        home_ownership_target_enc   DOUBLE PRECISION,
        feature_version             VARCHAR(50) NOT NULL,
        computed_at                 TIMESTAMP DEFAULT NOW()
    )
    """

    batch_size = FeatureEngineeringConfig.STAGING_WRITE_BATCH_SIZE
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features.application_features"))
        conn.execute(text(create_sql))

    total = len(out)
    print(f"[merge] Writing final {total} rows batch {batch_size}")
    # Cold-start stats for portfolio
    if "avg_days_overdue_90d" in out.columns:
        has_history = (out["avg_days_overdue_90d"] != 0).sum()
        print(f"[merge] Cold-start analysis: {has_history}/{total} with history ({has_history/total:.1%}), {total-has_history} without (cold-start)")

    for start in range(0, total, batch_size):
        chunk = out.iloc[start : start + batch_size]
        with engine.begin() as conn:
            chunk.to_sql(
                "application_features",
                conn,
                schema="features",
                if_exists="append",
                index=False,
                method="multi",
                chunksize=10_000,
            )
        print(f"  -> final batch {start // batch_size + 1}: {len(chunk)}")

    print(f"[merge] Wrote {len(out)} rows")


with dag:
    t_select = PythonOperator(task_id="select_fe_ids", python_callable=select_fe_ids)
    t_num = PythonOperator(task_id="compute.numerical", python_callable=compute_numerical_features)
    t_agg = PythonOperator(task_id="compute.aggregations", python_callable=compute_aggregation_features)
    t_te = PythonOperator(task_id="compute.target_encoding", python_callable=compute_target_encoding)
    t_bureau = PythonOperator(task_id="compute.bureau", python_callable=compute_bureau_features)
    t_merge = PythonOperator(task_id="merge_and_validate", python_callable=merge_and_validate)

    # First select consistent ids, then sequential FE to avoid DB overload
    t_select >> t_num >> t_agg >> t_te >> t_bureau >> t_merge
