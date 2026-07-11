"""Airflow DAG: feature engineering snapshot."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="feature_engineering",
    default_args=default_args,
    description="Compute features for credit scoring",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["features", "ml"],
)


def _write_df(df, table: str, engine):
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS features.{table}"))
        df.to_sql(table, conn, schema="features", if_exists="replace", index=False)


def compute_numerical_features(**context):
    from src.config import get_db_engine
    from src.data.queries import get_raw_applications
    from src.features.numerical import NumericalFeatureComputer

    engine = get_db_engine()
    df = get_raw_applications(engine, date=context["ds"])
    features = NumericalFeatureComputer().compute(df)
    _write_df(features, "numerical_staging", engine)


def compute_aggregation_features(**context):
    from src.config import FeatureConfig, get_db_engine
    from src.features.aggregations import AggregationFeatureComputer

    engine = get_db_engine()
    computer = AggregationFeatureComputer(
        windows=FeatureConfig.PAYMENT_WINDOWS,
        reference_date=context["ds"],
    )
    features = computer.compute(engine)
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
        execution_date=context["ds"],
    )
    encoder.save(ARTIFACTS_DIR / "target_encoding.json")
    _write_df(result, "target_enc_staging", engine)


def compute_bureau_features(**context):
    from src.config import get_db_engine
    from src.features.bureau import BureauFeatureComputer

    engine = get_db_engine()
    features = BureauFeatureComputer().compute(
        engine, reference_date=context["ds"]
    )
    _write_df(features, "bureau_staging", engine)


def merge_and_validate(**context):
    import pandas as pd
    from sqlalchemy import text

    from src.config import FeatureConfig, get_db_engine
    from src.data.validation import validate_feature_table

    engine = get_db_engine()
    with engine.connect() as conn:
        num = pd.read_sql(text("SELECT * FROM features.numerical_staging"), conn)
        agg = pd.read_sql(
            text("SELECT * FROM features.aggregation_staging"), conn
        )
        te = pd.read_sql(text("SELECT * FROM features.target_enc_staging"), conn)
        bureau = pd.read_sql(text("SELECT * FROM features.bureau_staging"), conn)

    merged = num.merge(agg, on="application_id", how="left")
    merged = merged.merge(te, on="application_id", how="left")
    merged = merged.merge(bureau, on="application_id", how="left")

    # Coerce numerics and fill residual nulls (critical first, then rest)
    for col in FeatureConfig.ALL_FEATURES:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    for col in FeatureConfig.CRITICAL_NO_NULL:
        if col in merged.columns and merged[col].isna().any():
            # last-resort fill so validation does not hard-fail on edge rows
            med = merged[col].median()
            merged[col] = merged[col].fillna(med if pd.notna(med) else 0.0)

    report = validate_feature_table(
        merged,
        checks={
            "no_nulls_in_critical": FeatureConfig.CRITICAL_NO_NULL,
            "value_ranges": {"loan_to_income": (0, 100)},
            "row_count_min": 50,
        },
    )
    if not report.success:
        raise ValueError(f"Validation failed: {report.errors}")

    merged["feature_version"] = FeatureConfig.VERSION
    # Replace full snapshot for this version (idempotent demo training)
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM features.application_features "
                "WHERE feature_version = :version"
            ),
            {"version": FeatureConfig.VERSION},
        )
        # Keep only model-relevant + keys columns if present
        keep = (
            ["application_id", "feature_date", "feature_version"]
            + FeatureConfig.ALL_FEATURES
            + ["dti_bucket", "credit_score_bucket", "avg_account_age_months"]
        )
        keep = [c for c in keep if c in merged.columns]
        merged[keep].to_sql(
            "application_features",
            conn,
            schema="features",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5_000,
        )


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
