"""Airflow DAG: train PD models, ensemble, register in MLflow."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.task_group import TaskGroup

from src.config import (
    ARTIFACTS_DIR,
    MLflowConfig,
    TrainingConfig,
)

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    dag_id="model_training",
    default_args=default_args,
    description="Train PD scoring models + stacking ensemble",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "ml"],
)


def prepare_datasets(**context):
    from src.config import get_db_engine
    from src.models.scoring.train import prepare_time_split

    engine = get_db_engine()
    info = prepare_time_split(engine)
    context["ti"].xcom_push(key="split_info", value=json.dumps(info))


def train_single_model(model_type: str, **context):
    import mlflow
    import mlflow.sklearn

    from src.config import get_db_engine
    from src.models.scoring.artifacts import ScoringArtifact
    from src.models.scoring.hyperopt import optimize_hyperparams
    from src.models.scoring.train import train_model
    from src.features.target_encoding import RegularizedTargetEncoder

    engine = get_db_engine()
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    mlflow.set_experiment(MLflowConfig.EXPERIMENT_SCORING)

    n_trials = int(context.get("n_trials", TrainingConfig.N_OPTUNA_TRIALS))
    # Keep CI/demo lighter unless overridden
    n_trials = min(n_trials, 15)

    with mlflow.start_run(run_name=f"{model_type}_{context['ds']}") as run:
        best_params = optimize_hyperparams(
            model_type=model_type,
            engine=engine,
            n_trials=n_trials,
        )
        model, metrics, feature_cols = train_model(
            model_type=model_type,
            params=best_params,
            engine=engine,
            calibrate=True,
        )
        mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", len(feature_cols))

        medians = getattr(model, "_feature_medians", {})
        te_path = ARTIFACTS_DIR / "target_encoding.json"
        te_payload = {}
        if te_path.exists():
            te_payload = json.loads(te_path.read_text())

        artifact = ScoringArtifact(
            model=model,
            feature_names=feature_cols,
            feature_medians=medians,
            target_encoding=te_payload,
            model_type=model_type,
            metrics=metrics,
            global_default_rate=float(
                te_payload.get("global_mean", 0.15) if te_payload else 0.15
            ),
        )
        out_dir = ARTIFACTS_DIR / model_type
        artifact.save(out_dir)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifacts(str(out_dir), artifact_path="scoring_bundle")

        context["ti"].xcom_push(key=f"{model_type}_auc", value=metrics["val_auc_roc"])
        context["ti"].xcom_push(key=f"{model_type}_run_id", value=run.info.run_id)


def train_ensemble(**context):
    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd

    from src.config import get_db_engine
    from src.models.scoring.artifacts import ScoringArtifact
    from src.models.scoring.ensemble import StackingEnsemble

    engine = get_db_engine()
    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    mlflow.set_experiment(MLflowConfig.EXPERIMENT_SCORING)

    class EnsembleWrapper:
        """Sklearn-like wrapper so MLflow can pickle predict_proba."""

        def __init__(self, ensemble: StackingEnsemble):
            self.ensemble = ensemble

        def predict_proba(self, X):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.ensemble.feature_names)
            p1 = self.ensemble.predict_proba(X)
            return np.column_stack([1 - p1, p1])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    with mlflow.start_run(run_name=f"ensemble_{context['ds']}") as run:
        ensemble = StackingEnsemble(
            base_models=["catboost", "lightgbm", "xgboost"],
        )
        model, metrics = ensemble.fit(engine)
        wrapper = EnsembleWrapper(model)

        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", "stacking_ensemble")

        te_path = ARTIFACTS_DIR / "target_encoding.json"
        te_payload = json.loads(te_path.read_text()) if te_path.exists() else {}
        artifact = ScoringArtifact(
            model=wrapper,
            feature_names=model.feature_names or [],
            feature_medians=model.feature_medians or {},
            target_encoding=te_payload,
            model_type="stacking_ensemble",
            metrics=metrics,
            global_default_rate=float(te_payload.get("global_mean", 0.15)),
        )
        out_dir = ARTIFACTS_DIR / "ensemble"
        artifact.save(out_dir)
        # Also publish as default local serving artifact
        artifact.save(ARTIFACTS_DIR)

        mlflow.sklearn.log_model(wrapper, artifact_path="model")
        mlflow.log_artifacts(str(out_dir), artifact_path="scoring_bundle")

        context["ti"].xcom_push(key="ensemble_auc", value=metrics["val_auc_roc"])
        context["ti"].xcom_push(key="ensemble_run_id", value=run.info.run_id)


def run_leakage_check(**context):
    from src.config import get_db_engine
    from src.models.scoring.evaluate import leakage_check

    engine = get_db_engine()
    result = leakage_check(
        engine, checks=["train_test_overlap", "future_payments"]
    )
    if not result.passed:
        raise ValueError(f"LEAKAGE: {result.details}")


def decide_register(**context):
    auc = context["ti"].xcom_pull(task_ids="ensemble", key="ensemble_auc")
    threshold = TrainingConfig.MIN_AUC_FOR_REGISTRATION
    if auc and float(auc) >= threshold:
        return "register"
    return "skip"


def register_model(**context):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
    client = MlflowClient()
    run_id = context["ti"].xcom_pull(task_ids="ensemble", key="ensemble_run_id")
    auc = context["ti"].xcom_pull(task_ids="ensemble", key="ensemble_auc")

    if not run_id:
        # fallback: best run in experiment
        experiment = client.get_experiment_by_name(MLflowConfig.EXPERIMENT_SCORING)
        if experiment is None:
            return
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_auc_roc DESC"],
            max_results=1,
        )
        if not runs:
            return
        run_id = runs[0].info.run_id
        auc = runs[0].data.metrics.get("val_auc_roc", 0)

    if auc is not None and float(auc) < TrainingConfig.MIN_AUC_FOR_REGISTRATION:
        return

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MLflowConfig.REGISTERED_MODEL_NAME)

    # Prefer modern aliases; keep stage transition as best-effort fallback
    try:
        client.set_registered_model_alias(
            name=MLflowConfig.REGISTERED_MODEL_NAME,
            alias=MLflowConfig.MODEL_ALIAS,
            version=result.version,
        )
    except Exception:
        try:
            client.transition_model_version_stage(
                name=MLflowConfig.REGISTERED_MODEL_NAME,
                version=result.version,
                stage="Production",
                archive_existing_versions=True,
            )
        except Exception:
            pass


with dag:
    t_prepare = PythonOperator(
        task_id="prepare_datasets",
        python_callable=prepare_datasets,
    )

    with TaskGroup("base_models") as base_group:
        for m in ["catboost", "lightgbm", "xgboost"]:
            PythonOperator(
                task_id=f"train_{m}",
                python_callable=train_single_model,
                op_kwargs={"model_type": m},
            )

    t_ensemble = PythonOperator(
        task_id="ensemble",
        python_callable=train_ensemble,
    )
    t_leakage = PythonOperator(
        task_id="leakage_check",
        python_callable=run_leakage_check,
    )
    t_decide = BranchPythonOperator(
        task_id="decide",
        python_callable=decide_register,
    )
    t_register = PythonOperator(
        task_id="register",
        python_callable=register_model,
    )
    t_skip = PythonOperator(
        task_id="skip",
        python_callable=lambda: print("Model below AUC threshold; not registered"),
    )

    t_prepare >> base_group >> t_ensemble >> t_leakage >> t_decide >> [
        t_register,
        t_skip,
    ]
