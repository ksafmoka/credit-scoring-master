"""Airflow DAG: raw data ingestion + synthetic histories."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="data_ingestion",
    default_args=default_args,
    description="Load raw applications, payment history, bureau data",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "raw"],
)


def load_raw_data(**context):
    from src.config import get_db_engine
    from src.data.ingestion import load_lending_club_data
    from src.data.paths import resolve_applications_csv
    from sqlalchemy import text

    engine = get_db_engine()
    filepath = resolve_applications_csv()
    print(f"Resolved CSV path: {filepath} exists={filepath.is_file()}")

    n = load_lending_club_data(
        filepath=filepath, engine=engine, replace=True
    )

    with engine.connect() as conn:
        count = int(
            conn.execute(text("SELECT COUNT(*) FROM raw.applications")).scalar()
            or 0
        )
        bounds = conn.execute(
            text(
                "SELECT MIN(application_date), MAX(application_date) "
                "FROM raw.applications"
            )
        ).one()

    print(
        f"Loaded {n} rows from {filepath}; "
        f"DB count={count}; date range={bounds[0]} .. {bounds[1]}"
    )
    if count == 0:
        raise ValueError(
            "data_ingestion finished but raw.applications is still empty!"
        )

    context["ti"].xcom_push(key="rows_loaded", value=n)
    context["ti"].xcom_push(key="db_count", value=count)
    context["ti"].xcom_push(key="source_file", value=str(filepath))


def generate_payment_history(**context):
    from src.config import get_db_engine
    from src.data.ingestion import generate_synthetic_payment_history

    engine = get_db_engine()
    n = generate_synthetic_payment_history(engine)
    context["ti"].xcom_push(key="payments_generated", value=n)


def generate_bureau(**context):
    from src.config import get_db_engine
    from src.data.ingestion import generate_synthetic_bureau

    engine = get_db_engine()
    n = generate_synthetic_bureau(engine)
    context["ti"].xcom_push(key="bureau_generated", value=n)


def validate_raw_data(**context):
    from src.config import get_db_engine
    from src.data.queries import get_raw_applications
    from src.data.validation import validate_raw_applications

    engine = get_db_engine()
    df = get_raw_applications(engine, limit=50_000)
    report = validate_raw_applications(df)
    if not report.success:
        raise ValueError(f"Raw data validation failed: {report.errors}")
    context["ti"].xcom_push(key="validation_stats", value=str(report.stats))


with dag:
    t_load = PythonOperator(
        task_id="load_raw_data",
        python_callable=load_raw_data,
    )
    t_payments = PythonOperator(
        task_id="generate_payment_history",
        python_callable=generate_payment_history,
    )
    t_bureau = PythonOperator(
        task_id="generate_bureau",
        python_callable=generate_bureau,
    )
    t_validate = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
    )

    t_load >> [t_payments, t_bureau] >> t_validate
