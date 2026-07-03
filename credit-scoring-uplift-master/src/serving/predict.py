import numpy as np
import pandas as pd
import mlflow
from loguru import logger

from src.serving.schemas import ScoringRequest, ScoringResponse, ReasonCode
from src.models.scoring.evaluate import get_risk_bucket
from src.models.scoring.explain import get_top_reasons
from src.config import MLflowConfig, FeatureConfig


class Predictor:

    def __init__(self):
        self.model = None
        self.explainer = None
        self.model_version = None
        self.feature_names = FeatureConfig.ALL_FEATURES

    def load(self, stage: str = "Production") -> None:
        """
        Загрузка из MLflow Model Registry.
        НЕ падает если модели нет — просто логирует warning.
        """
        mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)

        # Попытка 1: MLflow Registry
        try:
            self._load_from_mlflow(stage)
            return
        except Exception as e:
            logger.warning(f"MLflow model not found: {e}")

        # Попытка 2: локальный файл
        try:
            self._load_local_fallback()
            return
        except Exception as e:
            logger.warning(f"Local fallback not found: {e}")

        # Попытка 3: заглушка для разработки
        logger.warning(
            "No model found. API will start without a model. "
            "/predict will return 503 until a model is loaded."
        )
        self.model = None
        self.model_version = None

    def _load_from_mlflow(self, stage: str) -> None:
        """Загрузка из MLflow по alias (новый API) или stage (старый)."""
        try:
            # Новый API MLflow 2.9+: aliases вместо stages
            model_uri = f"models:/credit_scoring@champion"
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_version = "champion"
            logger.info("Model loaded from MLflow (alias: champion)")
            return
        except Exception:
            pass

        # Старый API: stage
        model_uri = f"models:/credit_scoring/{stage}"
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.model_version = f"mlflow_{stage.lower()}"
        logger.info(f"Model loaded from MLflow (stage: {stage})")

    def _load_local_fallback(self) -> None:
        """Загрузка из локального файла."""
        import pickle
        from pathlib import Path

        model_path = Path("artifacts/model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(f"No model at {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.model_version = "local_dev"
        logger.info("Model loaded from local fallback")

    def is_ready(self) -> bool:
        return self.model is not None

    def reload(self) -> None:
        """Перезагрузка модели без рестарта API."""
        logger.info("Reloading model...")
        self.model = None
        self.model_version = None
        self.explainer = None
        self.load()

    def _compute_features(self, request: ScoringRequest) -> pd.DataFrame:
        features = {}

        features["loan_to_income"] = (
            request.loan_amount / request.income
            if request.income > 0 else 0.0
        )
        features["credit_utilization"] = (
            request.loan_amount / request.total_credit_limit
            if request.total_credit_limit > 0 else 0.0
        )
        features["income_log"] = np.log1p(request.income)
        features["loan_amount_log"] = np.log1p(request.loan_amount)
        features["dti_ratio_clipped"] = np.clip(request.dti_ratio, 0, 100)
        features["loan_amount_x_dti"] = request.loan_amount * request.dti_ratio
        features["income_x_credit_score"] = request.income * request.credit_score

        # target encoding: используем global_mean как fallback
        features["purpose_target_enc"] = 0.15
        features["home_ownership_target_enc"] = 0.15

        # aggregation: недоступны онлайн
        for feat in FeatureConfig.AGGREGATION_FEATURES:
            features[feat] = 0.0
        features["bureau_balance_to_income"] = 0.0
        features["inquiries_per_account"] = 0.0

        df = pd.DataFrame([features])

        # гарантируем порядок и полноту колонок
        for col in FeatureConfig.ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        return df[FeatureConfig.ALL_FEATURES]

    def predict(self, request: ScoringRequest) -> ScoringResponse:
        features_df = self._compute_features(request)

        if hasattr(self.model, "predict_proba"):
            pd_score = float(self.model.predict_proba(features_df)[:, 1][0])
        else:
            pd_score = float(self.model.predict(features_df)[0])

        pd_calibrated = pd_score
        risk_bucket = get_risk_bucket(pd_calibrated)

        top_reasons = []
        try:
            if self.explainer is None:
                import shap
                underlying = (
                    self.model._model_impl
                    if hasattr(self.model, "_model_impl")
                    else self.model
                )
                self.explainer = shap.TreeExplainer(underlying)
            shap_vals = self.explainer.shap_values(features_df)[0]
            raw_reasons = get_top_reasons(
                shap_vals, features_df.columns.tolist()
            )
            top_reasons = [ReasonCode(**r) for r in raw_reasons]
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
            "model_loaded": self.is_ready(),
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
        }