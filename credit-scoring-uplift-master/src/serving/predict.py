# src/serving/predict.py

import numpy as np
import pandas as pd
import shap
import mlflow
from loguru import logger

from src.serving.schemas import ScoringRequest, ScoringResponse, ReasonCode
from src.models.scoring.evaluate import get_risk_bucket
from src.models.scoring.explain import get_top_reasons
from src.config import MLflowConfig, FeatureConfig


class Predictor:
    """Загрузка модели и инференс."""

    def __init__(self):
        self.model = None
        self.explainer = None
        self.model_version = None
        self.feature_names = FeatureConfig.ALL_FEATURES

    def load(self, stage: str = "Production") -> None:
        """Загрузка из MLflow Model Registry."""
        try:
            self.model = mlflow.pyfunc.load_model(
                f"models:/credit_scoring/{stage}"
            )
            self.model_version = "production_v1"
            logger.info(
                f"Model loaded from MLflow: {self.model_version}"
            )
        except Exception as e:
            logger.warning(
                f"MLflow model not found ({e}), "
                f"trying local fallback..."
            )
            self._load_local_fallback()

    def _load_local_fallback(self) -> None:
        """Загрузка из локального файла (для разработки)."""
        import pickle
        from pathlib import Path

        model_path = Path("artifacts/model.pkl")
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_version = "local_dev"
            logger.info("Loaded local model fallback")
        else:
            raise RuntimeError("No model found!")

    def _compute_features(
        self, request: ScoringRequest
    ) -> pd.DataFrame:
        """Online feature computation из raw запроса."""
        features = {}

        # числовые трансформации
        features["loan_to_income"] = (
            request.loan_amount / request.income
        )
        features["credit_utilization"] = (
            request.loan_amount / max(request.total_credit_limit, 1)
        )
        features["income_log"] = np.log1p(request.income)
        features["loan_amount_log"] = np.log1p(request.loan_amount)
        features["dti_ratio_clipped"] = np.clip(
            request.dti_ratio, 0, 100
        )

        # кросс-фичи
        features["loan_amount_x_dti"] = (
            request.loan_amount * request.dti_ratio
        )
        features["income_x_credit_score"] = (
            request.income * request.credit_score
        )

        # target encoded (используем global_mean как fallback)
        features["purpose_target_enc"] = 0.15
        features["home_ownership_target_enc"] = 0.15

        # агрегаты из payment_history недоступны онлайн
        # → используем нули / средние значения
        agg_features = [
            "avg_days_overdue_30d",
            "avg_days_overdue_90d",
            "avg_days_overdue_180d",
            "max_days_overdue_90d",
            "pct_late_payments_90d",
            "total_paid_90d",
            "payment_consistency_90d",
            "bureau_balance_to_income",
            "inquiries_per_account",
        ]
        for feat in agg_features:
            features[feat] = 0.0

        return pd.DataFrame([features])

    def predict(self, request: ScoringRequest) -> ScoringResponse:
        features_df = self._compute_features(request)

        # инференс
        if hasattr(self.model, "predict_proba"):
            pd_score = float(
                self.model.predict_proba(features_df)[:, 1][0]
            )
        else:
            pd_score = float(self.model.predict(features_df)[0])

        # калибровка (простая линейная — заменить на реальную)
        pd_calibrated = pd_score

        risk_bucket = get_risk_bucket(pd_calibrated)

        # SHAP (если модель поддерживает)
        top_reasons = []
        try:
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(
                    self.model._model_impl
                    if hasattr(self.model, "_model_impl")
                    else self.model
                )
            shap_vals = self.explainer.shap_values(features_df)[0]
            raw_reasons = get_top_reasons(
                shap_vals, features_df.columns.tolist()
            )
            top_reasons = [
                ReasonCode(**r) for r in raw_reasons
            ]
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

        return ScoringResponse(
            application_id=request.application_id,
            pd_score=pd_score,
            pd_calibrated=pd_calibrated,
            risk_bucket=risk_bucket,
            top_reasons=top_reasons,
            model_version=self.model_version or "unknown",
        )

    def get_model_info(self) -> dict:
        return {
            "model_version": self.model_version,
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
        }