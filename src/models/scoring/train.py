"""Model training utilities for PD scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.config import FeatureConfig, TrainingConfig
from src.data.queries import get_feature_dataset
from src.logging_utils import get_logger

logger = get_logger(__name__)


def prepare_time_split(
    engine: Engine,
    train_end: str = TrainingConfig.TRAIN_END_DATE,
    val_end: str = TrainingConfig.VAL_END_DATE,
) -> dict:
    """Time-based split summary — no random split."""
    train, val, test = get_feature_dataset(engine, train_end, val_end)

    def _rate(frame: pd.DataFrame) -> float | None:
        if frame.empty or TrainingConfig.TARGET_COL not in frame.columns:
            return None
        return float(frame[TrainingConfig.TARGET_COL].astype(float).mean())

    info = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "target_rate_train": _rate(train),
        "target_rate_val": _rate(val),
        "target_rate_test": _rate(test),
        "train_date_range": (
            (
                str(train[TrainingConfig.DATE_COL].min()),
                str(train[TrainingConfig.DATE_COL].max()),
            )
            if len(train)
            else None
        ),
    }
    logger.info(
        f"Split: train={info['train_size']}, "
        f"val={info['val_size']}, test={info['test_size']}"
    )
    return info


def get_feature_matrix(
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    cols = feature_cols or FeatureConfig.ALL_FEATURES
    cols = [c for c in cols if c in frame.columns]
    X = frame[cols].apply(pd.to_numeric, errors="coerce")
    return X, cols


def get_model(model_type: str, params: dict | None = None):
    """Model factory with lazy imports (install only what you use)."""
    params = dict(params or {})
    seed = TrainingConfig.RANDOM_SEED

    if model_type == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            verbose=False,
            random_seed=seed,
            loss_function="Logloss",
            eval_metric="AUC",
            **params,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            random_state=seed,
            verbose=-1,
            objective="binary",
            **params,
        )
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=seed,
            eval_metric="auc",
            objective="binary:logistic",
            tree_method="hist",
            **params,
        )
    raise ValueError(
        f"Unknown model type: {model_type}. "
        "Choose from catboost, lightgbm, xgboost"
    )


def _fit_boosting(model, model_type: str, X_train, y_train, X_val, y_val):
    if model_type == "lightgbm":
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        return model

    if model_type == "xgboost":
        try:
            from xgboost.callback import EarlyStopping

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=[EarlyStopping(rounds=50, save_best=True)],
            )
        except Exception:
            model.set_params(
                n_estimators=min(getattr(model, "n_estimators", 300), 300)
            )
            model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
            )
        return model

    if model_type == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        return model

    model.fit(X_train, y_train)
    return model


def calibrate_model(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = TrainingConfig.CALIBRATION_METHOD,
):
    if method in (None, "none", ""):
        return model
    if len(y_val) < 50 or y_val.nunique() < 2:
        logger.warning("Not enough validation data for calibration; skipping")
        return model

    try:
        calibrated = CalibratedClassifierCV(
            estimator=model, method=method, cv="prefit"
        )
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=model, method=method, cv="prefit"
        )
    calibrated.fit(X_val, y_val)
    logger.info(f"Applied {method} calibration on validation set")
    return calibrated


def train_model(
    model_type: str,
    params: dict | None,
    engine: Engine,
    train_end: str = TrainingConfig.TRAIN_END_DATE,
    val_end: str = TrainingConfig.VAL_END_DATE,
    calibrate: bool = True,
) -> tuple[Any, dict, list[str]]:
    """Train one model; returns (model, metrics, feature_cols)."""
    from src.models.scoring.evaluate import compute_metrics

    train, val, test = get_feature_dataset(engine, train_end, val_end)
    if train.empty or val.empty:
        raise ValueError(
            f"Insufficient data for training: train={len(train)}, val={len(val)}"
        )

    X_train, feature_cols = get_feature_matrix(train)
    X_val, _ = get_feature_matrix(val, feature_cols)
    y_train = train[TrainingConfig.TARGET_COL].astype(int)
    y_val = val[TrainingConfig.TARGET_COL].astype(int)

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    model = get_model(model_type, params or {})
    model = _fit_boosting(model, model_type, X_train, y_train, X_val, y_val)

    if calibrate:
        model = calibrate_model(model, X_val, y_val)

    val_proba = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, val_proba, prefix="val")

    if not test.empty:
        X_test, _ = get_feature_matrix(test, feature_cols)
        X_test = X_test.fillna(medians)
        y_test = test[TrainingConfig.TARGET_COL].astype(int)
        if y_test.nunique() > 1:
            test_proba = model.predict_proba(X_test)[:, 1]
            metrics.update(compute_metrics(y_test, test_proba, prefix="test"))

    model._feature_names_in_custom = feature_cols  # type: ignore[attr-defined]
    model._feature_medians = medians.to_dict()  # type: ignore[attr-defined]

    logger.info(
        f"{model_type} trained. "
        f"Val AUC: {metrics.get('val_auc_roc', float('nan')):.4f}"
    )
    return model, metrics, feature_cols


def load_training_frames(
    engine: Engine,
    train_end: str = TrainingConfig.TRAIN_END_DATE,
    val_end: str = TrainingConfig.VAL_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Load once for hyperopt loops."""
    train, val, test = get_feature_dataset(engine, train_end, val_end)
    X_train, feature_cols = get_feature_matrix(train)
    medians = X_train.median(numeric_only=True).to_dict()
    return train, val, test, feature_cols, medians
