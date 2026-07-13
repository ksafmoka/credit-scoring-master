"""Airflow DAG: batch scoring of feature snapshot."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import text

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="batch_prediction",
    default_args=default_args,
    description="Batch PD scoring into predictions.scoring_predictions",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["prediction", "batch"],
)


def run_batch_scoring(**context):
    from src.config import ARTIFACTS_DIR, FeatureConfig, get_db_engine
    from src.models.scoring.artifacts import ScoringArtifact
    from src.models.scoring.evaluate import get_risk_bucket

    engine = get_db_engine()
    artifact = ScoringArtifact.load(ARTIFACTS_DIR)

    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT *
                FROM features.application_features
                WHERE feature_version = :version
                """
            ),
            conn,
            params={"version": FeatureConfig.VERSION},
        )

    if df.empty:
        raise ValueError("No features available for batch scoring")

    proba = artifact.predict_proba(df)
    rows = []
    for app_id, score in zip(df["application_id"].tolist(), proba.tolist()):
        score = float(score)
        rows.append(
            {
                "application_id": int(app_id),
                "model_version": f"batch:{artifact.model_type}",
                "pd_score": score,
                "pd_calibrated": score,
                "risk_bucket": get_risk_bucket(score),
                "shap_top3": None,
            }
        )

    out = pd.DataFrame(rows)
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM predictions.scoring_predictions "
                "WHERE model_version LIKE 'batch:%'"
            )
        )
        out.to_sql(
            "scoring_predictions",
            conn,
            schema="predictions",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5_000,
        )
    context["ti"].xcom_push(key="n_scored", value=len(out))


with dag:
    PythonOperator(
        task_id="batch_score",
        python_callable=run_batch_scoring,
    )
