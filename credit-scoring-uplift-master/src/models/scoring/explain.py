# src/models/scoring/explain.py

import pandas as pd
import numpy as np
import shap
from loguru import logger


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_display: int = 20,
) -> tuple[np.ndarray, shap.Explainer]:
    """Вычисление SHAP values."""
    logger.info("Computing SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # для бинарной классификации берём класс 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, explainer


def get_top_reasons(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
    positive_only: bool = False,
) -> list[dict]:
    """Топ-N причин решения для одного объекта."""
    feature_shap = list(zip(feature_names, shap_values))

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
    """Model Card для governance и документации."""
    return {
        "model_type": type(model).__name__,
        "intended_use": "Credit default probability prediction",
        "training_data": {
            "description": "Lending Club loan data",
            "time_period": "2018-2022",
            "n_features": len(features_used),
            "features": features_used,
        },
        "evaluation_metrics": evaluation,
        "limitations": [
            "Model trained on US lending data",
            "May not generalize to different economic conditions",
            "Requires calibration for regulatory compliance",
        ],
        "ethical_considerations": [
            "Protected attributes (race, gender) not used",
            "Regular fairness audits recommended",
            "Human review required for borderline cases",
        ],
    }