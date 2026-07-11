# src/models/scoring/train.py

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from loguru import logger
from sqlalchemy.engine import Engine

from src.config import TrainingConfig, FeatureConfig
from src.data.queries import get_feature_dataset


def prepare_time_split(
    engine: Engine,
    train_end: str = TrainingConfig.TRAIN_END_DATE,
    val_end: str = TrainingConfig.VAL_END_DATE,
) -> dict:
    """Time-based split — никакого random split!"""
    train, val, test = get_feature_dataset(engine, train_end, val_end)

    info = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "target_rate_train": float(
            train[TrainingConfig.TARGET_COL].mean()
        ),
        "target_rate_val": float(
            val[TrainingConfig.TARGET_COL].mean()
        ),
        "target_rate_test": float(
            test[TrainingConfig.TARGET_COL].mean()
        ),
        "train_date_range": (
            str(train["application_date"].min()),
            str(train["application_date"].max()),
        ),
    }

    logger.info(
        f"Split: train={info['train_size']}, "
        f"val={info['val_size']}, test={info['test_size']}"
    )
    logger.info(
        f"Target rates: train={info['target_rate_train']:.3f}, "
        f"val={info['target_rate_val']:.3f}"
    )

    return info


def get_model(model_type: str, params: dict):
    """Фабрика моделей."""
    models = {
        "catboost": CatBoostClassifier(
            verbose=100,
            random_seed=TrainingConfig.RANDOM_SEED,
            **params,
        ),
        "lightgbm": LGBMClassifier(
            random_state=TrainingConfig.RANDOM_SEED,
            verbose=-1,
            **params,
        ),
        "xgboost": XGBClassifier(
            random_state=TrainingConfig.RANDOM_SEED,
            eval_metric="auc",
            **params,
        ),
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from {list(models.keys())}"
        )

    return models[model_type]


def train_model(
    model_type: str,
    params: dict,
    engine: Engine,
    train_end: str = TrainingConfig.TRAIN_END_DATE,
    val_end: str = TrainingConfig.VAL_END_DATE,
) -> tuple:
    """Обучение одной модели."""
    from src.models.scoring.evaluate import compute_metrics

    train, val, test = get_feature_dataset(engine, train_end, val_end)

    feature_cols = FeatureConfig.ALL_FEATURES
    feature_cols = [c for c in feature_cols if c in train.columns]
    target_col = TrainingConfig.TARGET_COL

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]

    model = get_model(model_type, params)

    # early stopping для gradient boosting
    if model_type in ["lightgbm", "xgboost"]:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb_early_stopping(50)
            ] if model_type == "lightgbm" else None,
            early_stopping_rounds=50
            if model_type == "xgboost"
            else None,
        )
    elif model_type == "catboost":
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
        )
    else:
        model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, val_proba, prefix="val")

    logger.info(
        f"{model_type} trained. "
        f"Val AUC: {metrics['val_auc_roc']:.4f}"
    )

    return model, metrics


def lgb_early_stopping(stopping_rounds):
    """Helper для LightGBM early stopping."""
    import lightgbm as lgb
    return lgb.early_stopping(stopping_rounds)