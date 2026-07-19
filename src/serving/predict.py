"""Online scoring predictor with dual-model routing (with-history / cold-start)."""

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


# Payment history feature names — if any of these are provided, use history model
_HISTORY_FEATURE_NAMES = {
    "avg_days_overdue_30d",
    "avg_days_overdue_90d",
    "avg_days_overdue_180d",
    "max_days_overdue_90d",
    "pct_late_payments_90d",
    "total_paid_90d",
    "payment_consistency_90d",
}


class _SegmentModel:
    """Container for one segment's model + metadata."""

    def __init__(self):
        self.artifact: ScoringArtifact | None = None
        self.model = None
        self.explainer = None
        self.model_version: str | None = None
        self.feature_names: list[str] = []
        self.feature_medians: dict[str, float] = {}
        self.target_encoding: dict = {}
        self.global_default_rate: float = 0.15

    def is_ready(self) -> bool:
        return self.model is not None


class Predictor:
    """Dual-model predictor: routes to with_history or cold_start model."""

    def __init__(self):
        self.history_model = _SegmentModel()
        self.cold_start_model = _SegmentModel()
        # Backward compatibility: single model mode
        self._legacy = _SegmentModel()
        self.model_version: str | None = None

    @property
    def model(self):
        """Backward compat — returns history model if available."""
        if self.history_model.is_ready():
            return self.history_model.model
        if self._legacy.is_ready():
            return self._legacy.model
        return None

    @property
    def artifact(self):
        if self.history_model.artifact:
            return self.history_model.artifact
        return self._legacy.artifact

    @property
    def feature_names(self):
        if self.history_model.feature_names:
            return self.history_model.feature_names
        return self._legacy.feature_names or list(FeatureConfig.ALL_FEATURES)

    @property
    def feature_medians(self):
        if self.history_model.feature_medians:
            return self.history_model.feature_medians
        return self._legacy.feature_medians

    def load(self, stage: str = "Production") -> None:
        # Try dual-model loading first
        loaded_any = False

        # Load with_history model
        try:
            self._load_segment("with_history", self.history_model, stage)
            loaded_any = True
            logger.info("✅ with_history model loaded")
        except Exception as exc:
            logger.warning(f"with_history model not found: {exc}")

        # Load cold_start model
        try:
            self._load_segment("cold_start", self.cold_start_model, stage)
            loaded_any = True
            logger.info("✅ cold_start model loaded")
        except Exception as exc:
            logger.warning(f"cold_start model not found: {exc}")

        # Fallback to legacy single model
        if not loaded_any:
            try:
                self._load_legacy(stage)
                logger.info("Loaded legacy single model")
            except Exception as exc:
                logger.warning(f"No models found: {exc}")
                logger.warning(
                    "API started WITHOUT any model. "
                    "/predict returns 503 until a model is loaded."
                )

    def _load_segment(self, segment: str, seg_model: _SegmentModel, stage: str) -> None:
        """Load a segment model from MLflow or local artifacts."""
        # Try MLflow first
        try:
            import mlflow
            mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
            model_name = f"{MLflowConfig.REGISTERED_MODEL_NAME}_{segment}"
            alias = MLflowConfig.MODEL_ALIAS
            model_uri = f"models:/{model_name}@{alias}"
            seg_model.model = mlflow.pyfunc.load_model(model_uri)
            seg_model.model_version = f"mlflow:{model_name}@{alias}"
            logger.info(f"MLflow model loaded: {model_name}@{alias}")
        except Exception:
            # Try local artifact fallback
            seg_dir = ARTIFACTS_DIR / segment
            if seg_dir.exists() and (seg_dir / "model.pkl").exists():
                artifact = ScoringArtifact.load(seg_dir)
                seg_model.artifact = artifact
                seg_model.model = artifact.model
                seg_model.feature_names = artifact.feature_names
                seg_model.feature_medians = artifact.feature_medians
                seg_model.target_encoding = artifact.target_encoding
                seg_model.global_default_rate = artifact.global_default_rate
                seg_model.model_version = f"local:{segment}:{artifact.model_type}"
            else:
                raise FileNotFoundError(f"No model for segment '{segment}'")

        # Load sidecar meta if available
        seg_dir = ARTIFACTS_DIR / segment
        meta_path = seg_dir / "artifact_meta.json"
        te_path = seg_dir / "target_encoding.json"
        if not seg_model.feature_names and meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            seg_model.feature_names = meta.get("feature_names", [])
            seg_model.feature_medians = meta.get("feature_medians", {})
            seg_model.global_default_rate = float(meta.get("global_default_rate", 0.15))
        if not seg_model.target_encoding and te_path.exists():
            import json
            seg_model.target_encoding = json.loads(te_path.read_text())

    def _load_legacy(self, stage: str) -> None:
        """Fallback: load single model from default artifact location."""
        try:
            import mlflow
            mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
            name = MLflowConfig.REGISTERED_MODEL_NAME
            alias = MLflowConfig.MODEL_ALIAS
            model_uri = f"models:/{name}@{alias}"
            self._legacy.model = mlflow.pyfunc.load_model(model_uri)
            self._legacy.model_version = f"mlflow@{alias}"
            return
        except Exception:
            pass

        artifact = ScoringArtifact.load(ARTIFACTS_DIR)
        self._legacy.artifact = artifact
        self._legacy.model = artifact.model
        self._legacy.feature_names = artifact.feature_names
        self._legacy.feature_medians = artifact.feature_medians
        self._legacy.target_encoding = artifact.target_encoding
        self._legacy.global_default_rate = artifact.global_default_rate
        self._legacy.model_version = f"local:{artifact.model_type}"

    def is_ready(self) -> bool:
        return (
            self.history_model.is_ready()
            or self.cold_start_model.is_ready()
            or self._legacy.is_ready()
        )

    def reload(self) -> None:
        logger.info("Reloading models...")
        self.history_model = _SegmentModel()
        self.cold_start_model = _SegmentModel()
        self._legacy = _SegmentModel()
        self.load()

    def _has_payment_history(self, request: ScoringRequest) -> bool:
        """Check if request includes any payment history features."""
        history_fields = [
            request.avg_days_overdue_30d,
            request.avg_days_overdue_90d,
            request.avg_days_overdue_180d,
            request.max_days_overdue_90d,
            request.pct_late_payments_90d,
            request.total_paid_90d,
            request.payment_consistency_90d,
        ]
        return any(v is not None for v in history_fields)

    def _select_model(self, request: ScoringRequest) -> tuple[_SegmentModel, str]:
        """Select the appropriate model based on request features."""
        has_history = self._has_payment_history(request)

        if has_history and self.history_model.is_ready():
            return self.history_model, "with_history"
        if not has_history and self.cold_start_model.is_ready():
            return self.cold_start_model, "cold_start"

        # Fallback to any available model
        if self.history_model.is_ready():
            return self.history_model, "with_history"
        if self.cold_start_model.is_ready():
            return self.cold_start_model, "cold_start"
        if self._legacy.is_ready():
            return self._legacy, "legacy"

        raise RuntimeError("No model available")

    def _encode_category(self, seg_model: _SegmentModel, col: str, value: str) -> float:
        te = seg_model.target_encoding or {}
        mapping = te.get("encoding_map", {}).get(col) or te.get(col) or {}
        if isinstance(mapping, dict) and value in mapping:
            return float(mapping[value])
        if "encoding_map" in te and col in te["encoding_map"]:
            return float(
                te["encoding_map"][col].get(
                    value, te.get("global_mean", seg_model.global_default_rate)
                )
            )
        return float(te.get("global_mean", seg_model.global_default_rate))

    def _compute_features(self, request: ScoringRequest, seg_model: _SegmentModel) -> pd.DataFrame:
        features: dict[str, float] = {}

        # Numerical features
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
        features["loan_amount_x_dti"] = float(request.loan_amount * request.dti_ratio)
        features["income_x_credit_score"] = float(request.income * request.credit_score)

        # Target encoded
        features["purpose_target_enc"] = self._encode_category(seg_model, "purpose", request.purpose)
        features["home_ownership_target_enc"] = self._encode_category(
            seg_model, "home_ownership", request.home_ownership
        )

        # Optional features (history + bureau)
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
                features[key] = float(seg_model.feature_medians.get(key, 0.0))
            else:
                features[key] = float(value)

        # Build DataFrame with correct column order for this model
        df = pd.DataFrame([features])
        ordered = seg_model.feature_names or FeatureConfig.ALL_FEATURES
        for col in ordered:
            if col not in df.columns:
                df[col] = float(seg_model.feature_medians.get(col, 0.0))
        df = df[ordered].apply(pd.to_numeric, errors="coerce")
        if seg_model.feature_medians:
            df = df.fillna(seg_model.feature_medians)
        return df.fillna(0.0)

    def predict(self, request: ScoringRequest) -> ScoringResponse:
        seg_model, segment = self._select_model(request)
        features_df = self._compute_features(request, seg_model)

        # Predict
        if seg_model.artifact is not None:
            pd_score = float(seg_model.artifact.predict_proba(features_df)[0])
        elif hasattr(seg_model.model, "predict_proba"):
            proba = seg_model.model.predict_proba(features_df)
            pd_score = float(
                proba[:, 1][0] if getattr(proba, "ndim", 1) == 2 else proba[0]
            )
        else:
            pred = seg_model.model.predict(features_df)
            pd_score = float(np.asarray(pred).reshape(-1)[0])

        pd_calibrated = float(np.clip(pd_score, 0.0, 1.0))
        risk_bucket = get_risk_bucket(pd_calibrated)

        # SHAP explanations
        top_reasons: list[ReasonCode] = []
        try:
            from src.models.scoring.explain import _unwrap_model
            import shap

            if seg_model.explainer is None:
                underlying = _unwrap_model(seg_model.model)
                seg_model.explainer = shap.TreeExplainer(underlying)
            shap_vals = seg_model.explainer.shap_values(features_df)
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
            model_version=f"{segment}:{seg_model.model_version or 'unknown'}",
        )

    def get_model_info(self) -> dict:
        return {
            "model_version": self.model_version,
            "history_model_loaded": self.history_model.is_ready(),
            "cold_start_model_loaded": self.cold_start_model.is_ready(),
            "legacy_model_loaded": self._legacy.is_ready(),
            "model_loaded": self.is_ready(),
            "history_features": list(self.history_model.feature_names) if self.history_model.is_ready() else [],
            "cold_start_features": list(self.cold_start_model.feature_names) if self.cold_start_model.is_ready() else [],
        }
