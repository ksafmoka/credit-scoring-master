"""Airflow DAG: feature engineering — full historical snapshot (no Airflow ds filter)."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="feature_engineering",
    default_args=default_args,
    description="Compute features for ALL raw.applications (Lending Club safe)",
    schedule_interval=None,  # manual — avoid surprise empty runs on schedule
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features", "ml"],
)


def _count_raw(engine) -> int:
    from sqlalchemy import text

    with engine.connect() as conn:
        return int(
            conn.execute(text("SELECT COUNT(*) FROM raw.applications")).scalar()
            or 0
        )


def _write_df(df, table: str, engine) -> None:
    print(f"Writing features.{table}: rows={len(df)}")
    if df is None:
        raise ValueError(f"Refusing to write None to features.{table}")
    with engine.begin() as conn:
        df.to_sql(
            table,
            conn,
            schema="features",
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=5_000,
        )


def compute_numerical_features(**context):
    from src.config import get_db_engine
    from src.data.queries import get_raw_applications
    from src.features.numerical import NumericalFeatureComputer

    engine = get_db_engine()
    n_raw = _count_raw(engine)
    print(f"raw.applications COUNT(*) = {n_raw}")
    if n_raw == 0:
        raise ValueError(
            "raw.applications is EMPTY. Run data_ingestion successfully first, "
            "then re-run feature_engineering."
        )

    # CRITICAL: never filter by context['ds'] — LC dates are historical
    df = get_raw_applications(engine, date=None)
    print(
        f"loaded applications: {len(df)}; "
        f"date range: {df['application_date'].min()} .. {df['application_date'].max()}"
    )
    features = NumericalFeatureComputer().compute(df)
    if features.empty:
        raise ValueError(
            "NumericalFeatureComputer returned 0 rows despite non-empty raw data."
        )
    print(f"numerical features shape: {features.shape}")
    _write_df(features, "numerical_staging", engine)


def compute_aggregation_features(**context):
    from src.config import FeatureConfig, get_db_engine
    from src.features.aggregations import AggregationFeatureComputer

    engine = get_db_engine()
    computer = AggregationFeatureComputer(
        windows=FeatureConfig.PAYMENT_WINDOWS,
        reference_date=None,  # all applications
    )
    features = computer.compute(engine)
    print(f"aggregation features shape: {features.shape}")
    # Always at least application_id list so merge doesn't die
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
    from src.config import ARTIFACTS_DIR, TrainingConfig, get_db_engine
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
        execution_date=None,  # ALL applications
    )
    if result.empty:
        raise ValueError(
            "Target encoding produced 0 rows — raw.applications empty?"
        )
    print(f"target encoding shape: {result.shape}")
    encoder.save(ARTIFACTS_DIR / "target_encoding.json")
    _write_df(result, "target_enc_staging", engine)


def compute_bureau_features(**context):
    from src.config import get_db_engine
    from src.features.bureau import BureauFeatureComputer

    engine = get_db_engine()
    features = BureauFeatureComputer().compute(engine, reference_date=None)
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

    from src.config import FeatureConfig, get_db_engine
    from src.data.validation import validate_feature_table
    from src.features.numerical import NumericalFeatureComputer
    from src.data.queries import get_raw_applications

    engine = get_db_engine()
    n_raw = _count_raw(engine)
    print(f"[merge] raw.applications COUNT(*) = {n_raw}")

    def _read_staging(name: str) -> pd.DataFrame:
        try:
            with engine.connect() as conn:
                return pd.read_sql(
                    text(f"SELECT * FROM features.{name}"), conn
                )
        except Exception as exc:
            print(f"[merge] could not read features.{name}: {exc}")
            return pd.DataFrame()

    num = _read_staging("numerical_staging")
    agg = _read_staging("aggregation_staging")
    te = _read_staging("target_enc_staging")
    bureau = _read_staging("bureau_staging")

    print(
        f"[merge] staging sizes: num={len(num)} agg={len(agg)} "
        f"te={len(te)} bureau={len(bureau)}"
    )

    # Self-heal: recompute numerical if staging empty but raw has data
    if num.empty and n_raw > 0:
        print(
            "[merge] numerical_staging empty but raw has data — "
            "recomputing numerical features inline"
        )
        apps = get_raw_applications(engine, date=None)
        num = NumericalFeatureComputer().compute(apps)
        _write_df(num, "numerical_staging", engine)

    if num.empty:
        raise ValueError(
            "features.numerical_staging is empty and raw.applications has "
            f"{n_raw} rows. If raw=0, re-run data_ingestion. "
            "If raw>0, check compute.numerical task logs."
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
        raise ValueError(
            "Merged feature table has 0 rows after joins. "
            f"num={len(num)} raw={n_raw}. "
            "Inspect features.numerical_staging and raw.applications."
        )

    report = validate_feature_table(
        merged,
        checks={
            "no_nulls_in_critical": FeatureConfig.CRITICAL_NO_NULL,
            "value_ranges": {"loan_to_income": (0, 1000)},
            "row_count_min": 1,
        },
    )
    if not report.success:
        raise ValueError(
            f"Validation failed: {report.errors} (merged_rows={len(merged)})"
        )

    merged["feature_version"] = FeatureConfig.VERSION
    if "feature_date" not in merged.columns:
        merged["feature_date"] = pd.Timestamp.utcnow().normalize()
    merged["feature_date"] = pd.to_datetime(
        merged["feature_date"], errors="coerce"
    ).dt.date
    # fill any bad feature_date from today
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

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features.application_features"))
        conn.execute(text(create_sql))
        out.to_sql(
            "application_features",
            conn,
            schema="features",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5_000,
        )
    print(f"[merge] Wrote features.application_features rows={len(out)}")


with dag:
    with TaskGroup("compute") as compute_group:
        t1 = PythonOperator(
            task_id="numerical",
            python_callable=compute_numerical_features,
        )
        t2 = PythonOperator(
            task_id="aggregations",
            python_callable=compute_aggregation_features,
        )
        t3 = PythonOperator(
            task_id="target_encoding",
            python_callable=compute_target_encoding,
        )
        t4 = PythonOperator(
            task_id="bureau",
            python_callable=compute_bureau_features,
        )
        [t1, t2, t3, t4]

    t_merge = PythonOperator(
        task_id="merge_and_validate",
        python_callable=merge_and_validate,
    )
    compute_group >> t_merge
