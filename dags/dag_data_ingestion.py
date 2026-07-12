"""Airflow DAG: raw data ingestion + synthetic histories."""

from __future__ import annotations

import os
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

    engine = get_db_engine()
    filepath = os.getenv(
        "LENDING_CLUB_CSV_PATH",
        "/opt/airflow/data/sample_applications.csv",
    )
    # Prefer full lending_club.csv when present
    alt = "/opt/airflow/data/lending_club.csv"
    if filepath.endswith("sample_applications.csv") and os.path.exists(alt):
        filepath = alt
    n = load_lending_club_data(filepath=filepath, engine=engine, replace=True)
    context["ti"].xcom_push(key="rows_loaded", value=n)


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
