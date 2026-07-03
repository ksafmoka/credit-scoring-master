# dags/dag_data_ingestion.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os

default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="data_ingestion",
    default_args=default_args,
    description="Load raw data into PostgreSQL",
    schedule_interval="@daily",
    start_date=datetime(2026, 6, 26),
    catchup=False,
    tags=["ingestion", "raw"],
)


def load_raw_data(**context):
    from src.data.ingestion import load_lending_club_data
    from src.config import get_db_engine

    engine = get_db_engine()
    filepath = os.getenv(
        "LENDING_CLUB_CSV_PATH",
        "/opt/airflow/data/lending_club.csv",
    )
    load_lending_club_data(filepath=filepath, engine=engine)


def generate_payment_history(**context):
    from src.data.ingestion import generate_synthetic_payment_history
    from src.config import get_db_engine

    engine = get_db_engine()
    generate_synthetic_payment_history(engine)


def validate_raw_data(**context):
    from src.data.validation import validate_raw_applications
    from src.data.queries import get_raw_applications
    from src.config import get_db_engine

    engine = get_db_engine()
    df = get_raw_applications(engine, limit=10_000)
    report = validate_raw_applications(df)

    if not report.success:
        raise ValueError(
            f"Raw data validation failed: {report.errors}"
        )

    context["ti"].xcom_push(
        key="validation_stats", value=str(report.stats)
    )


with dag:
    t_load = PythonOperator(
        task_id="load_raw_data",
        python_callable=load_raw_data,
    )

    t_payments = PythonOperator(
        task_id="generate_payment_history",
        python_callable=generate_payment_history,
    )

    t_validate = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
    )

    t_load >> t_payments >> t_validate