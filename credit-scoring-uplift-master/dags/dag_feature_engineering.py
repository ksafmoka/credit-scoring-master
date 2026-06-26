# dags/dag_feature_engineering.py

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


def compute_numerical_features(**context):
    from src.features.numerical import NumericalFeatureComputer
    from src.data.queries import get_raw_applications
    from src.config import get_db_engine

    engine = get_db_engine()
    df = get_raw_applications(engine, date=context["ds"])
    computer = NumericalFeatureComputer()
    features = computer.compute(df)
    features.to_sql(
        "numerical_staging",
        engine,
        schema="features",
        if_exists="replace",
        index=False,
    )


def compute_aggregation_features(**context):
    from src.features.aggregations import AggregationFeatureComputer
    from src.config import get_db_engine

    engine = get_db_engine()
    computer = AggregationFeatureComputer(
        windows=[30, 90, 180],
        reference_date=context["ds"],
    )
    features = computer.compute(engine)
    features.to_sql(
        "aggregation_staging",
        engine,
        schema="features",
        if_exists="replace",
        index=False,
    )


def compute_target_encoding(**context):
    from src.features.target_encoding import RegularizedTargetEncoder
    from src.config import get_db_engine, TrainingConfig

    engine = get_db_engine()
    encoder = RegularizedTargetEncoder(
        cols=["purpose", "home_ownership"],
        smoothing=20,
    )
    result = encoder.fit_transform(
        engine,
        train_cutoff=TrainingConfig.TRAIN_END_DATE,
        execution_date=context["ds"],
    )
    result.to_sql(
        "target_enc_staging",
        engine,
        schema="features",
        if_exists="replace",
        index=False,
    )


def merge_and_validate(**context):
    import pandas as pd
    from src.data.validation import validate_feature_table
    from src.config import get_db_engine, FeatureConfig

    engine = get_db_engine()

    num = pd.read_sql(
        "SELECT * FROM features.numerical_staging", engine
    )
    agg = pd.read_sql(
        "SELECT * FROM features.aggregation_staging", engine
    )
    te = pd.read_sql(
        "SELECT * FROM features.target_enc_staging", engine
    )

    merged = num.merge(agg, on="application_id", how="left")
    merged = merged.merge(te, on="application_id", how="left")

    report = validate_feature_table(
        merged,
        checks={
            "no_nulls_in_critical": ["loan_to_income"],
            "value_ranges": {"loan_to_income": (0, 100)},
            "row_count_min": 100,
        },
    )

    if not report.success:
        raise ValueError(f"Validation failed: {report.errors}")

    merged["feature_version"] = FeatureConfig.VERSION
    merged.to_sql(
        "application_features",
        engine,
        schema="features",
        if_exists="append",
        index=False,
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
        [t1, t2, t3]  # параллельно

    t_merge = PythonOperator(
        task_id="merge_and_validate",
        python_callable=merge_and_validate,
    )

    compute_group >> t_merge