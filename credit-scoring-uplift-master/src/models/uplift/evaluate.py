# src/models/uplift/evaluate.py

import numpy as np
import pandas as pd
from loguru import logger


def qini_curve(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Qini curve — основная метрика для uplift моделей.
    Аналог ROC AUC, но для uplift.
    """
    n = len(y_true)
    order = np.argsort(uplift)[::-1]

    y_sorted = y_true[order]
    t_sorted = treatment[order]

    # накопленные treated с конверсией и без
    cumulative_treated = np.cumsum(t_sorted)
    cumulative_control = np.cumsum(1 - t_sorted)
    cumulative_conversions_treated = np.cumsum(
        y_sorted * t_sorted
    )
    cumulative_conversions_control = np.cumsum(
        y_sorted * (1 - t_sorted)
    )

    n_treated = t_sorted.sum()
    n_control = (1 - t_sorted).sum()

    # Qini values
    with np.errstate(divide="ignore", invalid="ignore"):
        qini_values = np.where(
            cumulative_treated > 0,
            cumulative_conversions_treated
            - cumulative_conversions_control
            * cumulative_treated / np.maximum(cumulative_control, 1),
            0,
        )

    # AUC under qini curve
    qini_auc = np.trapz(
        qini_values / (n_treated + n_control),
        np.arange(n) / n,
    )

    return qini_values, float(qini_auc)


def uplift_at_k(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float = 0.3,
) -> float:
    """Uplift в топ-K% по скору."""
    n = len(y_true)
    top_k_idx = np.argsort(uplift)[::-1][:int(n * k)]

    t_topk = treatment[top_k_idx]
    y_topk = y_true[top_k_idx]

    if t_topk.sum() == 0 or (1 - t_topk).sum() == 0:
        return 0.0

    cr_treated = y_topk[t_topk == 1].mean()
    cr_control = y_topk[t_topk == 0].mean()

    return float(cr_treated - cr_control)