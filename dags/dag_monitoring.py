"""Airflow DAG: feature drift + prediction distribution checks."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="monitoring",
    default_args=default_args,
    description="Monitor data drift and prediction health",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["monitoring"],
)


def check_feature_drift(**context):
    from src.config import FeatureConfig, get_db_engine
    from src.monitoring.data_drift import run_drift_check

    engine = get_db_engine()
    execution_date = context["ds"]
    exec_dt = datetime.strptime(execution_date, "%Y-%m-%d")

    reference_end = (exec_dt - timedelta(days=90)).strftime("%Y-%m-%d")
    reference_start = (exec_dt - timedelta(days=180)).strftime("%Y-%m-%d")
    current_start = (exec_dt - timedelta(days=7)).strftime("%Y-%m-%d")

    results = run_drift_check(
        engine=engine,
        reference_start=reference_start,
        reference_end=reference_end,
        current_start=current_start,
        current_end=execution_date,
        features=FeatureConfig.NUMERICAL_FEATURES
        + FeatureConfig.AGGREGATION_FEATURES,
        check_date=execution_date,
    )
    drifted = [r["feature_name"] for r in results if r["is_drifted"]]
    if drifted:
        raise ValueError(
            f"Drift detected in features: {drifted}. Consider retraining."
        )


def check_prediction_distribution(**context):
    import pandas as pd
    from sqlalchemy import text

    from src.config import get_db_engine

    engine = get_db_engine()
    execution_date = context["ds"]
    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT pd_calibrated, risk_bucket
                FROM predictions.scoring_predictions
                WHERE DATE(predicted_at) = :execution_date
                """
            ),
            conn,
            params={"execution_date": execution_date},
        )

    if len(df) < 10:
        return

    mean_pd = float(df["pd_calibrated"].mean())
    std_pd = float(df["pd_calibrated"].std())
    if mean_pd > 0.5 or mean_pd < 0.01:
        raise ValueError(f"Suspicious mean PD: {mean_pd:.4f}")
    if std_pd < 0.001:
        raise ValueError(f"Model predictions degenerated (std={std_pd:.6f})")


with dag:
    t_drift = PythonOperator(
        task_id="check_feature_drift",
        python_callable=check_feature_drift,
    )
    t_preds = PythonOperator(
        task_id="check_prediction_distribution",
        python_callable=check_prediction_distribution,
    )
    [t_drift, t_preds]
