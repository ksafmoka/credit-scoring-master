# src/models/scoring/hyperopt.py

import optuna
import numpy as np
from loguru import logger
from sqlalchemy.engine import Engine

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparams(
    model_type: str,
    engine: Engine,
    n_trials: int = 50,
) -> dict:
    """Поиск гиперпараметров с Optuna."""

    def objective(trial: optuna.Trial) -> float:
        from src.models.scoring.train import train_model

        params = _suggest_params(trial, model_type)

        try:
            _, metrics = train_model(
                model_type=model_type,
                params=params,
                engine=engine,
            )
            return metrics["val_auc_roc"]
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=3600)

    logger.info(
        f"Best params for {model_type}: {study.best_params}"
    )
    logger.info(f"Best AUC: {study.best_value:.4f}")

    return study.best_params


def _suggest_params(trial: optuna.Trial, model_type: str) -> dict:
    """Пространство поиска для каждой модели."""

    if model_type == "catboost":
        return {
            "iterations": trial.suggest_int(
                "iterations", 300, 1500
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1e-3, 10.0, log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.6, 1.0
            ),
        }

    elif model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 300, 1500
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 20, 150
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 100
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-4, 1.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 1.0, log=True
            ),
        }

    elif model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 300, 1500
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 20
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "gamma": trial.suggest_float(
                "gamma", 0, 5
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-4, 1.0, log=True
            ),
        }

    return {}