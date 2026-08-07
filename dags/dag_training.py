"""Airflow DAG: train dual PD models (with-history + cold-start), register best of each."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

from src.config import ARTIFACTS_DIR, FeatureConfig, MLflowConfig, TrainingConfig

default_args = {
    "owner": "ml-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="model_training",
    default_args=default_args,
    description="Train dual PD models: with-history + cold-start, register best of each",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "ml"],
    max_active_runs=1,
)


def prepare_datasets(**context):
    from src.config import get_db_engine
    from src.mlflow_utils import ensure_experiment
    from src.models.scoring.train import prepare_time_split

    exp_id = ensure_experiment()
    context["ti"].xcom_push(key="mlflow_experiment_id", value=exp_id)

    engine = get_db_engine()
    info = prepare_time_split(engine)
    context["ti"].xcom_push(key="split_info", value=json.dumps(info, default=str))


def _load_segment_data(engine, segment: str):
    """Load train/val split for a segment.

    segment='history': clients WITH records in payment_history, ALL features
    segment='cold_start': clients WITHOUT records in payment_history, non-aggregation features only
    """
    import pandas as pd
    from sqlalchemy import text

    from src.config import TrainingConfig
    from src.data.queries import get_feature_dataset
    from src.models.scoring.train import _auto_date_cuts, get_feature_matrix

    train_end = TrainingConfig.TRAIN_END_DATE
    val_end = TrainingConfig.VAL_END_DATE
    train, val, test = get_feature_dataset(engine, train_end, val_end)

    if TrainingConfig.AUTO_DATE_SPLIT and (train.empty or val.empty):
        cuts = _auto_date_cuts(engine)
        if cuts:
            train_end, val_end = cuts
            train, val, test = get_feature_dataset(engine, train_end, val_end)

    if train.empty or val.empty:
        raise ValueError(f"Insufficient data for {segment} segment")

    # Get list of application_ids that have payment history records
    with engine.connect() as conn:
        hist_ids = pd.read_sql(
            text("SELECT DISTINCT application_id FROM raw.payment_history"), conn
        )["application_id"].tolist()
    hist_ids_set = set(hist_ids)
    print(f"[data] payment_history distinct ids: {len(hist_ids_set)}")

    if segment == "history":
        # Filter to clients that HAVE payment history records
        train = train[train["application_id"].isin(hist_ids_set)]
        val = val[val["application_id"].isin(hist_ids_set)]
        feature_cols = FeatureConfig.ALL_FEATURES
        seg_label = "with_history"
    else:
        # Cold-start: clients WITHOUT payment history records
        train = train[~train["application_id"].isin(hist_ids_set)]
        val = val[~val["application_id"].isin(hist_ids_set)]
        feature_cols = FeatureConfig.COLD_START_FEATURES
        seg_label = "cold_start"

    if train.empty or val.empty:
        raise ValueError(f"Empty {segment} segment after filtering (train={len(train)}, val={len(val)})")

    X_train, feature_cols = get_feature_matrix(train, feature_cols)
    X_val, _ = get_feature_matrix(val, feature_cols)

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    y_train = train[TrainingConfig.TARGET_COL].astype(int)
    y_val = val[TrainingConfig.TARGET_COL].astype(int)

    print(f"[{seg_label}] train={len(X_train)}, val={len(X_val)}, "
          f"features={len(feature_cols)}, default_rate_train={y_train.mean():.3%}")

    return X_train, X_val, y_train, y_val, medians, feature_cols, seg_label


def _train_and_compare(segment: str, engine, n_trials: int = 10):
    """Train all model types for a segment, return best (model, metrics, model_type, feature_cols)."""
    import mlflow
    from src.config import get_db_engine
    from src.models.scoring.artifacts import ScoringArtifact
    from src.models.scoring.train import _fit_boosting, get_feature_matrix, get_model
    from src.models.scoring.evaluate import compute_metrics

    X_train, X_val, y_train, y_val, medians, feature_cols, seg_label = _load_segment_data(
        engine, segment
    )

    # Imbalance handling
    pos_rate = float(y_train.mean()) if len(y_train) else 0.15
    scale_w = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    results = {}
    run_date = datetime.utcnow().strftime("%Y-%m-%d")

    for model_type in ["catboost", "lightgbm", "xgboost"]:
        with mlflow.start_run(run_name=f"{seg_label}_{model_type}_{run_date}") as run:
            params = {}
            if model_type in ("lightgbm", "xgboost"):
                params["scale_pos_weight"] = scale_w

            # Use Optuna for hyperparameter tuning on SEGMENT data
            from src.models.scoring.hyperopt import optimize_hyperparams

            best_params = optimize_hyperparams(
                model_type=model_type,
                engine=engine,
                n_trials=min(n_trials, int(TrainingConfig.N_OPTUNA_TRIALS)),
                preloaded_data={
                    "X_train": X_train,
                    "X_val": X_val,
                    "y_train": y_train,
                    "y_val": y_val,
                    "feature_cols": feature_cols,
                },
            )

            # Merge imbalance params
            if model_type in ("lightgbm", "xgboost"):
                best_params.setdefault("scale_pos_weight", scale_w)

            model = get_model(model_type, best_params)
            model = _fit_boosting(model, model_type, X_train, y_train, X_val, y_val)

            # Compare by RAW AUC (before calibration) — calibration on small val
            # can degrade AUC; we calibrate only the winning model afterward
            raw_proba = model.predict_proba(X_val)[:, 1]
            raw_metrics = compute_metrics(y_val, raw_proba, prefix="val")

            mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})
            mlflow.log_metrics(raw_metrics)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("segment", seg_label)
            mlflow.log_param("n_features", len(feature_cols))

            # Save artifact
            te_path = ARTIFACTS_DIR / "target_encoding.json"
            te_payload = json.loads(te_path.read_text()) if te_path.exists() else {}

            model._feature_medians = medians.to_dict()
            model._feature_names_in_custom = feature_cols

            artifact = ScoringArtifact(
                model=model,
                feature_names=feature_cols,
                feature_medians=medians.to_dict(),
                target_encoding=te_payload,
                model_type=model_type,
                metrics=raw_metrics,
                global_default_rate=float(te_payload.get("global_mean", 0.15)),
            )
            out_dir = ARTIFACTS_DIR / seg_label / model_type
            artifact.save(out_dir)

            results[model_type] = {
                "auc": raw_metrics["val_auc_roc"],
                "metrics": raw_metrics,
                "run_id": run.info.run_id,
                "artifact": artifact,
                "out_dir": out_dir,
                "model": model,
            }

            print(f"[{seg_label}] {model_type}: val AUC={raw_metrics['val_auc_roc']:.4f} (raw, pre-calibration)")

    # Pick best by raw AUC
    best_type = max(results, key=lambda k: results[k]["auc"])
    best = results[best_type]

    # Calibrate ONLY the winning model — but only if it improves AUC
    from src.models.scoring.train import calibrate_model
    calibrated_model = calibrate_model(best["model"], X_val, y_val)
    cal_proba = calibrated_model.predict_proba(X_val)[:, 1]
    cal_metrics = compute_metrics(y_val, cal_proba, prefix="val")

    if cal_metrics["val_auc_roc"] >= best["auc"]:
        # Calibration helped — use calibrated model
        print(f"[{seg_label}] Best: {best_type} raw AUC={best['auc']:.4f} → "
              f"calibrated AUC={cal_metrics['val_auc_roc']:.4f} (calibration KEPT)")
        best["artifact"].model = calibrated_model
        best["artifact"].metrics = cal_metrics
        best["auc"] = cal_metrics["val_auc_roc"]
        best["metrics"] = cal_metrics
    else:
        # Calibration hurt — keep raw model
        print(f"[{seg_label}] Best: {best_type} raw AUC={best['auc']:.4f} → "
              f"calibrated AUC={cal_metrics['val_auc_roc']:.4f} (calibration SKIPPED, raw model kept)")

    # Re-save artifact with final model
    best["out_dir"].mkdir(parents=True, exist_ok=True)
    best["artifact"].save(best["out_dir"])

    # Copy best artifact to segment root
    import shutil
    seg_dir = ARTIFACTS_DIR / seg_label
    seg_dir.mkdir(parents=True, exist_ok=True)
    for f in best["out_dir"].iterdir():
        if f.is_file():
            dst = seg_dir / f.name
            if dst.exists():
                dst.unlink()
            shutil.copyfile(f, dst)

    return best_type, best


def train_with_history(**context):
    """Train best model for clients WITH payment history (all features)."""
    import mlflow
    from src.config import get_db_engine
    from src.mlflow_utils import ensure_experiment

    engine = get_db_engine()
    ensure_experiment()

    best_type, best = _train_and_compare("history", engine)

    context["ti"].xcom_push(key="history_best_model", value=best_type)
    context["ti"].xcom_push(key="history_best_auc", value=best["auc"])
    context["ti"].xcom_push(key="history_run_id", value=best["run_id"])
    context["ti"].xcom_push(key="history_metrics", value=json.dumps(best["metrics"], default=str))


def train_cold_start(**context):
    """Train best model for cold-start clients (no aggregation features)."""
    import mlflow
    from src.config import get_db_engine
    from src.mlflow_utils import ensure_experiment

    engine = get_db_engine()
    ensure_experiment()

    best_type, best = _train_and_compare("cold_start", engine)

    context["ti"].xcom_push(key="cold_start_best_model", value=best_type)
    context["ti"].xcom_push(key="cold_start_best_auc", value=best["auc"])
    context["ti"].xcom_push(key="cold_start_run_id", value=best["run_id"])
    context["ti"].xcom_push(key="cold_start_metrics", value=json.dumps(best["metrics"], default=str))


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
    """Check if both segment models meet AUC threshold."""
    threshold = TrainingConfig.MIN_AUC_FOR_REGISTRATION

    history_auc = context["ti"].xcom_pull(task_ids="train_with_history", key="history_best_auc")
    cold_auc = context["ti"].xcom_pull(task_ids="train_cold_start", key="cold_start_best_auc")

    history_model = context["ti"].xcom_pull(task_ids="train_with_history", key="history_best_model")
    cold_model = context["ti"].xcom_pull(task_ids="train_cold_start", key="cold_start_best_model")

    print(f"With-history: {history_model} AUC={history_auc:.4f}")
    print(f"Cold-start:   {cold_model} AUC={cold_auc:.4f}")
    print(f"Threshold: {threshold}")

    # Register if at least one model meets threshold
    history_ok = history_auc and float(history_auc) >= threshold
    cold_ok = cold_auc and float(cold_auc) >= threshold

    if history_ok or cold_ok:
        return "register"
    return "skip"


def register_models(**context):
    """Register both segment models in MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.mlflow_utils import ensure_experiment

    ensure_experiment()
    client = MlflowClient(tracking_uri=MLflowConfig.TRACKING_URI)
    threshold = TrainingConfig.MIN_AUC_FOR_REGISTRATION

    # Register with-history model
    history_auc = context["ti"].xcom_pull(task_ids="train_with_history", key="history_best_auc")
    history_run_id = context["ti"].xcom_pull(task_ids="train_with_history", key="history_run_id")
    history_model = context["ti"].xcom_pull(task_ids="train_with_history", key="history_best_model")

    if history_auc and float(history_auc) >= threshold and history_run_id:
        model_name = f"{MLflowConfig.REGISTERED_MODEL_NAME}_with_history"
        model_uri = f"runs:/{history_run_id}/model"
        try:
            result = mlflow.register_model(model_uri, model_name)
            client.set_registered_model_alias(name=model_name, alias="champion", version=result.version)
            print(f"Registered {model_name}: {history_model} AUC={history_auc:.4f}")
        except Exception as exc:
            print(f"MLflow register failed for with_history: {exc}")

    # Register cold-start model
    cold_auc = context["ti"].xcom_pull(task_ids="train_cold_start", key="cold_start_best_auc")
    cold_run_id = context["ti"].xcom_pull(task_ids="train_cold_start", key="cold_start_run_id")
    cold_model = context["ti"].xcom_pull(task_ids="train_cold_start", key="cold_start_best_model")

    if cold_auc and float(cold_auc) >= threshold and cold_run_id:
        model_name = f"{MLflowConfig.REGISTERED_MODEL_NAME}_cold_start"
        model_uri = f"runs:/{cold_run_id}/model"
        try:
            result = mlflow.register_model(model_uri, model_name)
            client.set_registered_model_alias(name=model_name, alias="champion", version=result.version)
            print(f"Registered {model_name}: {cold_model} AUC={cold_auc:.4f}")
        except Exception as exc:
            print(f"MLflow register failed for cold_start: {exc}")

    # Copy best artifacts to serving location (with_history takes priority)
    import shutil
    for seg in ["with_history", "cold_start"]:
        seg_dir = ARTIFACTS_DIR / seg
        if seg_dir.exists():
            for f in seg_dir.iterdir():
                if f.is_file():
                    dst = ARTIFACTS_DIR / f"{seg}_{f.name}"
                    if dst.exists():
                        dst.unlink()
                    shutil.copyfile(f, dst)
            print(f"Serving artifacts copied from {seg}")


with dag:
    t_prepare = PythonOperator(
        task_id="prepare_datasets",
        python_callable=prepare_datasets,
    )

    t_history = PythonOperator(
        task_id="train_with_history",
        python_callable=train_with_history,
    )

    t_cold = PythonOperator(
        task_id="train_cold_start",
        python_callable=train_cold_start,
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
        python_callable=register_models,
    )

    t_skip = PythonOperator(
        task_id="skip",
        python_callable=lambda: print("Both models below AUC threshold; not registered"),
    )

    (
        t_prepare
        >> t_history
        >> t_cold
        >> t_leakage
        >> t_decide
        >> [t_register, t_skip]
    )
