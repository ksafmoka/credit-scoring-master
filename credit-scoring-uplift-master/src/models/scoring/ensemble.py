# src/models/scoring/ensemble.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from loguru import logger
from sqlalchemy.engine import Engine

from src.config import TrainingConfig, FeatureConfig
from src.data.queries import get_feature_dataset


class StackingEnsemble:
    """
    Двухуровневый стэкинг:
    Level 1: CatBoost + LightGBM + XGBoost
    Level 2: Logistic Regression как мета-модель
    """

    def __init__(
        self,
        base_models: list[str],
        meta_model: str = "logistic_regression",
        cv_folds: int = 5,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.fitted_base_models = []
        self.fitted_meta_model = None
        self.feature_names = None

    def fit(self, engine: Engine) -> tuple:
        from src.models.scoring.train import get_model
        from src.models.scoring.evaluate import compute_metrics

        train, val, _ = get_feature_dataset(
            engine,
            TrainingConfig.TRAIN_END_DATE,
            TrainingConfig.VAL_END_DATE,
        )

        feature_cols = [
            c for c in FeatureConfig.ALL_FEATURES
            if c in train.columns
        ]
        self.feature_names = feature_cols
        target_col = TrainingConfig.TARGET_COL

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_val = val[feature_cols].values
        y_val = val[target_col].values

        # Level 1: OOF предсказания для обучения мета-модели
        oof_predictions = np.zeros((len(X_train), len(self.base_models)))
        val_predictions = np.zeros((len(X_val), len(self.base_models)))

        kf = KFold(
            n_splits=self.cv_folds,
            shuffle=False,  # не shuffle! time-based логика
        )

        for i, model_type in enumerate(self.base_models):
            logger.info(f"Training base model: {model_type}")

            model = get_model(model_type, {})

            oof_preds = np.zeros(len(X_train))
            for fold, (train_idx, val_idx) in enumerate(
                kf.split(X_train)
            ):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]

                model.fit(X_fold_train, y_fold_train)
                oof_preds[val_idx] = model.predict_proba(
                    X_fold_val
                )[:, 1]

            oof_predictions[:, i] = oof_preds

            # retrain on full train
            model.fit(X_train, y_train)
            val_predictions[:, i] = model.predict_proba(X_val)[:, 1]
            self.fitted_base_models.append(model)

        # Level 2: мета-модель
        meta = LogisticRegression(C=1.0, random_state=42)
        meta.fit(oof_predictions, y_train)
        self.fitted_meta_model = meta

        final_val_preds = meta.predict_proba(val_predictions)[:, 1]
        metrics = compute_metrics(y_val, final_val_preds, "val")

        logger.info(
            f"Ensemble val AUC: {metrics['val_auc_roc']:.4f}"
        )

        return self, metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.fitted_base_models
        ])
        return self.fitted_meta_model.predict_proba(base_preds)[:, 1]