"""Airflow DAG: dual-model batch scoring of feature snapshot."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
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
    description="Dual-model batch PD scoring (with-history + cold-start)",
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

    # Load features
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

    # Get payment history IDs to split segments
    with engine.connect() as conn:
        hist_ids = set(
            pd.read_sql(
                text("SELECT DISTINCT application_id FROM raw.payment_history"),
                conn,
            )["application_id"].tolist()
        )

    df["has_history"] = df["application_id"].isin(hist_ids)
    df_history = df[df["has_history"]].copy()
    df_cold = df[~df["has_history"]].copy()

    print(f"Batch scoring: {len(df_history)} with history, {len(df_cold)} cold-start")

    all_rows = []

    # Score with-history segment
    if not df_history.empty:
        seg_dir = ARTIFACTS_DIR / "with_history"
        if seg_dir.exists() and (seg_dir / "model.pkl").exists():
            artifact = ScoringArtifact.load(seg_dir)
            proba = artifact.predict_proba(df_history)
            for app_id, score in zip(df_history["application_id"].tolist(), proba.tolist()):
                score = float(score)
                all_rows.append({
                    "application_id": int(app_id),
                    "model_version": f"batch:with_history:{artifact.model_type}",
                    "pd_score": score,
                    "pd_calibrated": score,
                    "risk_bucket": get_risk_bucket(score),
                    "shap_top3": None,
                })
            print(f"With-history: scored {len(df_history)} rows")
        else:
            print("WARNING: with_history model not found, skipping segment")

    # Score cold-start segment
    if not df_cold.empty:
        seg_dir = ARTIFACTS_DIR / "cold_start"
        if seg_dir.exists() and (seg_dir / "model.pkl").exists():
            artifact = ScoringArtifact.load(seg_dir)
            # Use only cold-start features
            cold_cols = [c for c in artifact.feature_names if c in df_cold.columns]
            df_cold_pred = df_cold[cold_cols].copy()
            proba = artifact.predict_proba(df_cold_pred)
            for app_id, score in zip(df_cold["application_id"].tolist(), proba.tolist()):
                score = float(score)
                all_rows.append({
                    "application_id": int(app_id),
                    "model_version": f"batch:cold_start:{artifact.model_type}",
                    "pd_score": score,
                    "pd_calibrated": score,
                    "risk_bucket": get_risk_bucket(score),
                    "shap_top3": None,
                })
            print(f"Cold-start: scored {len(df_cold)} rows")
        else:
            print("WARNING: cold_start model not found, skipping segment")

    if not all_rows:
        raise ValueError("No models available for batch scoring")

    out = pd.DataFrame(all_rows)

    # Write to predictions table
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

    print(f"Total scored: {len(out)} rows")
    context["ti"].xcom_push(key="n_scored", value=len(out))


with dag:
    PythonOperator(
        task_id="batch_score",
        python_callable=run_batch_scoring,
    )
