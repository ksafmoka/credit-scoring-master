"""Stacking ensemble over boosting base learners."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from src.config import FeatureConfig, TrainingConfig
from src.data.queries import get_feature_dataset
from src.models.scoring.evaluate import compute_metrics
from src.models.scoring.train import get_feature_matrix, get_model


class StackingEnsemble:
    """
    Level-1: CatBoost / LightGBM / XGBoost (OOF via TimeSeriesSplit)
    Level-2: Logistic Regression meta-model
    """

    def __init__(
        self,
        base_models: list[str] | None = None,
        cv_folds: int = 5,
    ):
        self.base_models = base_models or ["catboost", "lightgbm", "xgboost"]
        self.cv_folds = cv_folds
        self.fitted_base_models: list = []
        self.fitted_meta_model = None
        self.feature_names: list[str] | None = None
        self.feature_medians: dict | None = None

    def fit(self, engine: Engine) -> tuple["StackingEnsemble", dict]:
        train, val, _ = get_feature_dataset(
            engine,
            TrainingConfig.TRAIN_END_DATE,
            TrainingConfig.VAL_END_DATE,
        )
        if train.empty or val.empty:
            raise ValueError("Insufficient data for ensemble training")

        # Keep chronological order for TimeSeriesSplit
        train = train.sort_values(TrainingConfig.DATE_COL)
        val = val.sort_values(TrainingConfig.DATE_COL)

        X_train_df, feature_cols = get_feature_matrix(train)
        X_val_df, _ = get_feature_matrix(val, feature_cols)
        medians = X_train_df.median(numeric_only=True)
        X_train_df = X_train_df.fillna(medians)
        X_val_df = X_val_df.fillna(medians)

        self.feature_names = feature_cols
        self.feature_medians = medians.to_dict()

        X_train = X_train_df.values
        y_train = train[TrainingConfig.TARGET_COL].astype(int).values
        X_val = X_val_df.values
        y_val = val[TrainingConfig.TARGET_COL].astype(int).values

        # Imbalance handling for stacking (same as train_model)
        pos_rate = float(y_train.mean()) if len(y_train) else 0.15
        scale_w = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        logger.info(f"Ensemble imbalance: default_rate={pos_rate:.3%}, scale_pos_weight={scale_w:.2f}")

        oof = np.zeros((len(X_train), len(self.base_models)))
        val_preds = np.zeros((len(X_val), len(self.base_models)))
        self.fitted_base_models = []

        n_splits = min(self.cv_folds, max(2, len(X_train) // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for i, model_type in enumerate(self.base_models):
            logger.info(f"Training base model: {model_type} (scale_pos_weight={scale_w:.2f})")
            oof_col = np.zeros(len(X_train))

            for train_idx, val_idx in tscv.split(X_train):
                base_params = {"scale_pos_weight": scale_w} if model_type in ("lightgbm", "xgboost") else {}
                base_params.update(
                    {"iterations": 200} if model_type == "catboost" else {"n_estimators": 200}
                )
                fold_model = get_model(model_type, base_params)
                fold_model.fit(X_train[train_idx], y_train[train_idx])
                oof_col[val_idx] = fold_model.predict_proba(X_train[val_idx])[
                    :, 1
                ]

            missing = oof_col == 0
            if missing.any() and (~missing).any():
                oof_col[missing] = oof_col[~missing].mean()

            oof[:, i] = oof_col

            full_params = {"scale_pos_weight": scale_w} if model_type in ("lightgbm", "xgboost") else {}
            full_params.update(
                {"iterations": 300} if model_type == "catboost" else {"n_estimators": 300}
            )
            full_model = get_model(model_type, full_params)
            full_model.fit(X_train, y_train)
            val_preds[:, i] = full_model.predict_proba(X_val)[:, 1]
            self.fitted_base_models.append(full_model)

        meta = LogisticRegression(
            C=1.0, max_iter=1000, random_state=TrainingConfig.RANDOM_SEED
        )
        meta.fit(oof, y_train)
        self.fitted_meta_model = meta

        final_val = meta.predict_proba(val_preds)[:, 1]
        metrics = compute_metrics(y_val, final_val, "val")
        logger.info(f"Ensemble val AUC: {metrics['val_auc_roc']:.4f}")
        return self, metrics

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                X = X.reindex(columns=self.feature_names)
            if self.feature_medians:
                X = X.fillna(self.feature_medians)
            X_mat = X.values
        else:
            X_mat = X

        base = np.column_stack(
            [
                model.predict_proba(X_mat)[:, 1]
                for model in self.fitted_base_models
            ]
        )
        return self.fitted_meta_model.predict_proba(base)[:, 1]

    def predict_proba_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Sklearn-like (n, 2) probabilities for calibrated wrappers / MLflow."""
        p1 = self.predict_proba(X)
        return np.column_stack([1 - p1, p1])
