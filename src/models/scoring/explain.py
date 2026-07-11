"""SHAP explanations and model card helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)


def _unwrap_model(model):
    """Best-effort unwrap of CalibratedClassifierCV / pyfunc wrappers."""
    if hasattr(model, "calibrated_classifiers_"):
        try:
            return model.calibrated_classifiers_[0].estimator
        except Exception:
            try:
                return model.calibrated_classifiers_[0].base_estimator
            except Exception:
                return model
    if hasattr(model, "estimator"):
        return model.estimator
    if hasattr(model, "_model_impl"):
        impl = model._model_impl
        return getattr(impl, "python_model", impl)
    return model


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_display: int = 20,
) -> tuple[np.ndarray, object]:
    import shap

    logger.info("Computing SHAP values...")
    base = _unwrap_model(model)
    explainer = shap.TreeExplainer(base)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values, explainer


def get_top_reasons(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
    positive_only: bool = False,
) -> list[dict]:
    if shap_values.ndim > 1:
        shap_values = shap_values.reshape(-1)
    feature_shap = list(zip(feature_names, shap_values.tolist()))
    if positive_only:
        feature_shap = [(f, v) for f, v in feature_shap if v > 0]
    sorted_reasons = sorted(
        feature_shap, key=lambda x: abs(x[1]), reverse=True
    )[:top_n]
    return [
        {
            "feature": feature,
            "shap_value": float(value),
            "direction": "increases_risk" if value > 0 else "decreases_risk",
        }
        for feature, value in sorted_reasons
    ]


def generate_model_card(
    model,
    evaluation: dict,
    features_used: list[str],
) -> dict:
    return {
        "model_type": type(model).__name__,
        "intended_use": "Credit default probability prediction (PD)",
        "training_data": {
            "description": "Lending Club / synthetic credit applications",
            "time_period": "time-based split via TrainingConfig",
            "n_features": len(features_used),
            "features": features_used,
        },
        "evaluation_metrics": evaluation,
        "limitations": [
            "Trained primarily on US consumer lending distributions",
            "Synthetic payment / bureau history may be used for demo",
            "Aggregation features may be sparse for thin-file applicants",
            "Requires monitoring and periodic recalibration",
        ],
        "ethical_considerations": [
            "Protected attributes (race, gender) are not used as features",
            "Regular fairness audits recommended",
            "Human review required for borderline / high-risk cases",
        ],
    }
