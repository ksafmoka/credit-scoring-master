"""Online scoring predictor with artifact / MLflow loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)

from src.config import ARTIFACTS_DIR, FeatureConfig, MLflowConfig
from src.models.scoring.artifacts import ScoringArtifact
from src.models.scoring.evaluate import get_risk_bucket
from src.models.scoring.explain import get_top_reasons
from src.serving.schemas import ReasonCode, ScoringRequest, ScoringResponse


class Predictor:
    def __init__(self):
        self.artifact: ScoringArtifact | None = None
        self.model = None
        self.explainer = None
        self.model_version: str | None = None
        self.feature_names = list(FeatureConfig.ALL_FEATURES)
        self.feature_medians: dict[str, float] = {}
        self.target_encoding: dict = {}
        self.global_default_rate: float = 0.15

    def load(self, stage: str = "Production") -> None:
        try:
            self._load_from_mlflow(stage)
            return
        except Exception as exc:
            logger.warning(f"MLflow model not found: {exc}")

        try:
            self._load_local_fallback()
            return
        except Exception as exc:
            logger.warning(f"Local fallback not found: {exc}")

        logger.warning(
            "No model found. API will start without a model. "
            "/predict returns 503 until a model is loaded."
        )
        self.model = None
        self.artifact = None
        self.model_version = None

    def _load_from_mlflow(self, stage: str) -> None:
        import mlflow

        mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
        name = MLflowConfig.REGISTERED_MODEL_NAME
        alias = MLflowConfig.MODEL_ALIAS
        try:
            model_uri = f"models:/{name}@{alias}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_version = f"mlflow@{alias}"
            logger.info(f"Model loaded from MLflow alias: {alias}")
            self._try_load_sidecar_meta()
            return
        except Exception:
            pass

        model_uri = f"models:/{name}/{stage}"
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.model_version = f"mlflow_{stage.lower()}"
        logger.info(f"Model loaded from MLflow stage: {stage}")
        self._try_load_sidecar_meta()

    def _try_load_sidecar_meta(self) -> None:
        meta_path = ARTIFACTS_DIR / "artifact_meta.json"
        te_path = ARTIFACTS_DIR / "target_encoding.json"
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text())
            self.feature_names = meta.get("feature_names", self.feature_names)
            self.feature_medians = meta.get("feature_medians", {})
            self.global_default_rate = float(
                meta.get("global_default_rate", 0.15)
            )
        if te_path.exists():
            import json

            self.target_encoding = json.loads(te_path.read_text())

    def _load_local_fallback(self) -> None:
        artifact = ScoringArtifact.load(ARTIFACTS_DIR)
        self.artifact = artifact
        self.model = artifact.model
        self.feature_names = artifact.feature_names
        self.feature_medians = artifact.feature_medians
        self.target_encoding = artifact.target_encoding
        self.global_default_rate = artifact.global_default_rate
        self.model_version = f"local:{artifact.model_type}"
        logger.info("Model loaded from local artifact bundle")

    def is_ready(self) -> bool:
        return self.model is not None

    def reload(self) -> None:
        logger.info("Reloading model...")
        self.model = None
        self.artifact = None
        self.model_version = None
        self.explainer = None
        self.load()

    def _encode_category(self, col: str, value: str) -> float:
        te = self.target_encoding or {}
        mapping = te.get("encoding_map", {}).get(col) or te.get(col) or {}
        if isinstance(mapping, dict) and value in mapping:
            return float(mapping[value])
        # nested structure from RegularizedTargetEncoder.to_dict()
        if "encoding_map" in te and col in te["encoding_map"]:
            return float(
                te["encoding_map"][col].get(
                    value, te.get("global_mean", self.global_default_rate)
                )
            )
        return float(te.get("global_mean", self.global_default_rate))

    def _compute_features(self, request: ScoringRequest) -> pd.DataFrame:
        features: dict[str, float] = {}

        features["loan_to_income"] = (
            request.loan_amount / request.income if request.income > 0 else 0.0
        )
        features["credit_utilization"] = (
            request.loan_amount / request.total_credit_limit
            if request.total_credit_limit > 0
            else 0.0
        )
        features["income_log"] = float(np.log1p(request.income))
        features["loan_amount_log"] = float(np.log1p(request.loan_amount))
        features["dti_ratio_clipped"] = float(np.clip(request.dti_ratio, 0, 100))
        features["employment_years"] = float(request.employment_years)
        features["credit_score_norm"] = float(
            np.clip((request.credit_score - 300) / 550.0, 0, 1)
        )
        features["num_open_accounts"] = float(request.num_open_accounts)
        features["num_delinquencies"] = float(request.num_delinquencies)
        features["interest_rate"] = float(request.interest_rate)
        features["loan_amount_x_dti"] = float(
            request.loan_amount * request.dti_ratio
        )
        features["income_x_credit_score"] = float(
            request.income * request.credit_score
        )

        features["purpose_target_enc"] = self._encode_category(
            "purpose", request.purpose
        )
        features["home_ownership_target_enc"] = self._encode_category(
            "home_ownership", request.home_ownership
        )

        optional_map = {
            "avg_days_overdue_30d": request.avg_days_overdue_30d,
            "avg_days_overdue_90d": request.avg_days_overdue_90d,
            "avg_days_overdue_180d": request.avg_days_overdue_180d,
            "max_days_overdue_90d": request.max_days_overdue_90d,
            "pct_late_payments_90d": request.pct_late_payments_90d,
            "total_paid_90d": request.total_paid_90d,
            "payment_consistency_90d": request.payment_consistency_90d,
            "bureau_balance_to_income": request.bureau_balance_to_income,
            "inquiries_per_account": request.inquiries_per_account,
        }
        for key, value in optional_map.items():
            if value is None:
                features[key] = float(self.feature_medians.get(key, 0.0))
            else:
                features[key] = float(value)

        df = pd.DataFrame([features])
        ordered = self.feature_names or FeatureConfig.ALL_FEATURES
        for col in ordered:
            if col not in df.columns:
                df[col] = float(self.feature_medians.get(col, 0.0))
        df = df[ordered].apply(pd.to_numeric, errors="coerce")
        if self.feature_medians:
            df = df.fillna(self.feature_medians)
        return df.fillna(0.0)

    def predict(self, request: ScoringRequest) -> ScoringResponse:
        features_df = self._compute_features(request)

        if self.artifact is not None:
            pd_score = float(self.artifact.predict_proba(features_df)[0])
        elif hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features_df)
            pd_score = float(
                proba[:, 1][0] if getattr(proba, "ndim", 1) == 2 else proba[0]
            )
        else:
            # mlflow.pyfunc
            pred = self.model.predict(features_df)
            pd_score = float(np.asarray(pred).reshape(-1)[0])

        # Model is already calibrated offline when training used CalibratedClassifierCV
        pd_calibrated = float(np.clip(pd_score, 0.0, 1.0))
        risk_bucket = get_risk_bucket(pd_calibrated)

        top_reasons: list[ReasonCode] = []
        try:
            from src.models.scoring.explain import _unwrap_model
            import shap

            if self.explainer is None:
                underlying = _unwrap_model(self.model)
                self.explainer = shap.TreeExplainer(underlying)
            shap_vals = self.explainer.shap_values(features_df)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            row = np.asarray(shap_vals)[0]
            raw = get_top_reasons(row, list(features_df.columns))
            top_reasons = [ReasonCode(**r) for r in raw]
        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}")

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
