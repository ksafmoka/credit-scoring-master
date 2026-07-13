"""Optuna hyperparameter search for boosting models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import TrainingConfig
from src.logging_utils import get_logger
from src.models.scoring.train import (
    _fit_boosting,
    get_feature_matrix,
    get_model,
    load_training_frames,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = get_logger(__name__)


def optimize_hyperparams(
    model_type: str,
    engine: Engine,
    n_trials: int = 30,
) -> dict:
    """Search hyperparameters; reuses one DB load for all trials."""
    import optuna
    from src.models.scoring.evaluate import compute_metrics

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train, val, _, feature_cols, medians = load_training_frames(engine)
    if train.empty or val.empty:
        raise ValueError("No data for hyperparameter optimization")

    X_train, feature_cols = get_feature_matrix(train, feature_cols)
    X_val, _ = get_feature_matrix(val, feature_cols)
    y_train = train[TrainingConfig.TARGET_COL].astype(int)
    y_val = val[TrainingConfig.TARGET_COL].astype(int)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_type)
        try:
            model = get_model(model_type, params)
            model = _fit_boosting(
                model, model_type, X_train, y_train, X_val, y_val
            )
            proba = model.predict_proba(X_val)[:, 1]
            metrics = compute_metrics(y_val, proba, prefix="val")
            return float(metrics["val_auc_roc"])
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Trial failed: {exc}")
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TrainingConfig.RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, timeout=3600)
    logger.info(f"Best params for {model_type}: {study.best_params}")
    logger.info(f"Best AUC: {study.best_value:.4f}")
    return study.best_params


def _suggest_params(trial, model_type: str) -> dict:
    if model_type == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1e-2, 10.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.6, 1.0
            ),
        }

    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 1.0, log=True
            ),
        }

    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "gamma": trial.suggest_float("gamma", 0, 3),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        }

    return {}
