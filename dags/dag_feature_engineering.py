"""Airflow DAG: feature engineering — full historical snapshot (no Airflow ds filter).

Fix for 2.2M rows: old code loaded 2.2M into RAM + single to_sql transaction (4min+)
-> Postgres overload -> DNS resolution Temporary failure + heartbeat timeout + zombie.

New: batched processing (100k apps per batch), batched writes (50k rows/commit),
sequential tasks to avoid 4 parallel heavy writes.
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
    description="Compute features for ALL raw.applications (batched for 2.2M safe)",
    schedule_interval=None,  # manual — avoid surprise empty runs
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features", "ml"],
)


def _count_raw(engine) -> int:
    from sqlalchemy import text

    with engine.connect() as conn:
        return int(
            conn.execute(text("SELECT COUNT(*) FROM raw.applications")).scalar() or 0
        )


def _write_df(df, table: str, engine, batch_size: int | None = None) -> None:
    """Batched write to avoid long transaction holding Postgres."""
    from sqlalchemy import text

    from src.config import FeatureEngineeringConfig

    batch_size = batch_size or FeatureEngineeringConfig.STAGING_WRITE_BATCH_SIZE

    if df is None:
        raise ValueError(f"Refusing to write None to features.{table}")
    if df.empty:
        print(f"Writing features.{table}: EMPTY, creating empty table")
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS features.{table}"))
            # Create empty with at least application_id
            conn.execute(text(f"CREATE TABLE features.{table} (application_id BIGINT)"))
        return

    total = len(df)
    print(f"Writing features.{table}: rows={total}, batch_size={batch_size}")

    # Drop once
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS features.{table}"))

    # Write in batches to keep transactions short
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
        print(
            f"  -> {table} batch {start // batch_size + 1}: "
            f"wrote {len(chunk)} rows ({min(start + batch_size, total)}/{total})"
        )


def compute_numerical_features(**context):
    from sqlalchemy import text

    from src.config import FeatureEngineeringConfig, IngestionConfig, get_db_engine
    from src.features.numerical import NumericalFeatureComputer

    engine = get_db_engine()
    n_raw = _count_raw(engine)
    print(f"raw.applications COUNT(*) = {n_raw}")
    if n_raw == 0:
        raise ValueError(
            "raw.applications is EMPTY. Run data_ingestion successfully first."
        )

    max_apps = FeatureEngineeringConfig.MAX_APPS
    strategy = IngestionConfig.SAMPLE_STRATEGY
    total = min(n_raw, max_apps) if max_apps else n_raw
    batch_size = FeatureEngineeringConfig.NUMERICAL_BATCH_SIZE
    print(
        f"Numerical: total={total}, batch_size={batch_size}, max_apps={max_apps}, strategy={strategy}"
    )

    computer = NumericalFeatureComputer()
    all_features = []

    # Pre-select random ids if needed for consistent sampling (variant A)
    random_ids = None
    if strategy == "random" and max_apps:
        import pandas as pd

        with engine.connect() as conn:
            id_df = pd.read_sql(
                text("SELECT application_id FROM raw.applications ORDER BY random() LIMIT :limit"),
                conn,
                params={"limit": int(max_apps)},
            )
            random_ids = id_df["application_id"].tolist()
            total = len(random_ids)
            print(f"Numerical random_ids selected: {total}")

    for offset in range(0, total, batch_size):
        limit = min(batch_size, total - offset)
        import pandas as pd

        if random_ids is not None:
            batch_ids = random_ids[offset : offset + limit]
            if not batch_ids:
                continue
            query = "SELECT * FROM raw.applications WHERE application_id = ANY(:ids)"
            params = {"ids": batch_ids}
        else:
            order = "application_date DESC, application_id DESC" if max_apps else "application_date, application_id"
            query = f"""
                SELECT * FROM raw.applications
                ORDER BY {order}
                LIMIT :limit OFFSET :offset
            """
            params = {"limit": limit, "offset": offset}

        with engine.connect() as conn:
            chunk = pd.read_sql(text(query), conn, params=params)

        if chunk.empty:
            continue

        print(
            f"Numerical batch {offset // batch_size + 1}: loaded {len(chunk)} apps "
            f"({offset + len(chunk)}/{total})"
        )
        feats = computer.compute(chunk)
        all_features.append(feats)
        del chunk

    if not all_features:
        raise ValueError("NumericalFeatureComputer returned 0 rows")

    import pandas as pd

    features = pd.concat(all_features, ignore_index=True)
    print(f"numerical features shape: {features.shape}")
    _write_df(features, "numerical_staging", engine)


def compute_aggregation_features(**context):
    from src.config import FeatureEngineeringConfig, get_db_engine
    from src.features.aggregations import AggregationFeatureComputer

    engine = get_db_engine()
    computer = AggregationFeatureComputer(
        windows=FeatureEngineeringConfig.PAYMENT_WINDOWS
        if hasattr(FeatureEngineeringConfig, "PAYMENT_WINDOWS")
        else [30, 90, 180],
        reference_date=None,
    )
    # Use FeatureConfig for windows if above misses
    try:
        from src.config import FeatureConfig

        computer.windows = FeatureConfig.PAYMENT_WINDOWS
    except Exception:
        pass

    features = computer.compute(
        engine,
        batch_size=FeatureEngineeringConfig.AGGREGATION_BATCH_SIZE,
        max_apps=FeatureEngineeringConfig.MAX_APPS,
    )
    print(f"aggregation features shape: {features.shape}")
    if features.empty or "application_id" not in features.columns:
        from sqlalchemy import text
        import pandas as pd

        with engine.connect() as conn:
            features = pd.read_sql(
                text("SELECT application_id FROM raw.applications"), conn
            )
        print(f"aggregation fallback application_ids only: {len(features)}")
    _write_df(features, "aggregation_staging", engine)


def compute_target_encoding(**context):
    from src.config import (
        ARTIFACTS_DIR,
        FeatureEngineeringConfig,
        TrainingConfig,
        get_db_engine,
    )
    from src.features.target_encoding import RegularizedTargetEncoder

    engine = get_db_engine()
    encoder = RegularizedTargetEncoder(
        cols=["purpose", "home_ownership"],
        smoothing=20,
        noise_level=0.0,
    )
    result = encoder.fit_transform(
        engine,
        train_cutoff=TrainingConfig.TRAIN_END_DATE,
        execution_date=None,
        batch_size=FeatureEngineeringConfig.TARGET_ENCODING_BATCH_SIZE,
        max_apps=FeatureEngineeringConfig.MAX_APPS,
    )
    if result.empty:
        raise ValueError("Target encoding produced 0 rows")
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
    print(f"bureau features shape: {features.shape}")
    if features.empty or "application_id" not in features.columns:
        from sqlalchemy import text
        import pandas as pd

        with engine.connect() as conn:
            features = pd.read_sql(
                text("SELECT application_id FROM raw.applications"), conn
            )
    _write_df(features, "bureau_staging", engine)


def merge_and_validate(**context):
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    from src.config import FeatureConfig, FeatureEngineeringConfig, get_db_engine
    from src.data.validation import validate_feature_table

    engine = get_db_engine()
    n_raw = _count_raw(engine)
    print(f"[merge] raw.applications COUNT(*) = {n_raw}")

    def _read_staging(name: str) -> pd.DataFrame:
        try:
            with engine.connect() as conn:
                return pd.read_sql(text(f"SELECT * FROM features.{name}"), conn)
        except Exception as exc:
            print(f"[merge] could not read features.{name}: {exc}")
            return pd.DataFrame()

    num = _read_staging("numerical_staging")
    agg = _read_staging("aggregation_staging")
    te = _read_staging("target_enc_staging")
    bureau = _read_staging("bureau_staging")

    print(
        f"[merge] staging sizes: num={len(num)} agg={len(agg)} te={len(te)} bureau={len(bureau)}"
    )

    if num.empty and n_raw > 0:
        print("[merge] numerical_staging empty but raw has data — abort")
        raise ValueError(
            f"features.numerical_staging empty raw={n_raw}. Check compute.numerical logs."
        )

    for frame in (num, agg, te, bureau):
        if not frame.empty and "application_id" in frame.columns:
            frame["application_id"] = pd.to_numeric(
                frame["application_id"], errors="coerce"
            )

    merged = num.copy()
    if not agg.empty and "application_id" in agg.columns:
        merged = merged.merge(agg, on="application_id", how="left")
    if not te.empty and "application_id" in te.columns:
        merged = merged.merge(te, on="application_id", how="left")
    if not bureau.empty and "application_id" in bureau.columns:
        merged = merged.merge(bureau, on="application_id", how="left")

    print(f"[merge] merged shape: {merged.shape}")

    for col in FeatureConfig.ALL_FEATURES:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    for col in FeatureConfig.CRITICAL_NO_NULL:
        if col in merged.columns and merged[col].isna().any():
            med = merged[col].median()
            merged[col] = merged[col].fillna(0.0 if pd.isna(med) else med)

    for col in list(FeatureConfig.ALL_FEATURES) + ["avg_account_age_months"]:
        if col in merged.columns:
            merged[col] = (
                pd.to_numeric(merged[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    merged = merged.dropna(subset=["application_id"])
    merged["application_id"] = merged["application_id"].astype(np.int64)

    if len(merged) == 0:
        raise ValueError(f"Merged 0 rows. num={len(num)} raw={n_raw}")

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
    merged["feature_date"] = pd.to_datetime(
        merged["feature_date"], errors="coerce"
    ).dt.date
    if merged["feature_date"].isna().any():
        merged["feature_date"] = merged["feature_date"].fillna(
            pd.Timestamp.utcnow().date()
        )

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

    # Batched write for final table as well
    batch_size = FeatureEngineeringConfig.STAGING_WRITE_BATCH_SIZE
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features.application_features"))
        conn.execute(text(create_sql))

    total = len(out)
    print(f"[merge] Writing application_features rows={total} batch_size={batch_size}")
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
        print(f"  -> final batch {start // batch_size + 1}: {len(chunk)} rows")

    print(f"[merge] Wrote features.application_features rows={len(out)}")


with dag:
    # Sequential to avoid Postgres overload (4 parallel 2.2M writes caused DNS timeout)
    t_numerical = PythonOperator(
        task_id="compute.numerical",
        python_callable=compute_numerical_features,
    )
    t_agg = PythonOperator(
        task_id="compute.aggregations",
        python_callable=compute_aggregation_features,
    )
    t_te = PythonOperator(
        task_id="compute.target_encoding",
        python_callable=compute_target_encoding,
    )
    t_bureau = PythonOperator(
        task_id="compute.bureau",
        python_callable=compute_bureau_features,
    )
    t_merge = PythonOperator(
        task_id="merge_and_validate",
        python_callable=merge_and_validate,
    )

    # Sequential chain reduces DB contention, still clear lineage for portfolio
    t_numerical >> t_agg >> t_te >> t_bureau >> t_merge
