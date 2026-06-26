# src/models/scoring/evaluate.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve
from scipy import stats
from loguru import logger


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    prefix: str = "val",
) -> dict:
    """Полный набор метрик для скоринговой модели."""
    metrics = {}

    metrics[f"{prefix}_auc_roc"] = float(
        roc_auc_score(y_true, y_pred)
    )
    metrics[f"{prefix}_auc_pr"] = float(
        average_precision_score(y_true, y_pred)
    )
    metrics[f"{prefix}_brier_score"] = float(
        brier_score_loss(y_true, y_pred)
    )
    metrics[f"{prefix}_log_loss"] = float(
        log_loss(y_true, y_pred)
    )

    # KS-статистика (важна для скоринга)
    ks_stat, _ = stats.ks_2samp(
        y_pred[y_true == 1],
        y_pred[y_true == 0],
    )
    metrics[f"{prefix}_ks_statistic"] = float(ks_stat)

    # Gini coefficient
    metrics[f"{prefix}_gini"] = float(
        2 * metrics[f"{prefix}_auc_roc"] - 1
    )

    logger.info(
        f"{prefix} metrics: "
        f"AUC={metrics[f'{prefix}_auc_roc']:.4f}, "
        f"KS={metrics[f'{prefix}_ks_statistic']:.4f}, "
        f"Gini={metrics[f'{prefix}_gini']:.4f}"
    )

    return metrics


def get_risk_bucket(pd_score: float) -> str:
    """Разбивка по риск-бакетам."""
    if pd_score < 0.05:
        return "LOW"
    elif pd_score < 0.15:
        return "MEDIUM"
    elif pd_score < 0.30:
        return "HIGH"
    else:
        return "VERY_HIGH"


def leakage_check(engine, checks: list[str]) -> object:
    """Базовые проверки на data leakage."""
    from dataclasses import dataclass

    @dataclass
    class LeakageResult:
        passed: bool
        details: dict

    issues = {}

    if "train_test_overlap" in checks:
        # проверяем, что application_id не пересекаются
        overlap_query = """
            SELECT COUNT(*) as overlap_count
            FROM (
                SELECT application_id FROM raw.applications
                WHERE application_date <= '2022-12-31'
                INTERSECT
                SELECT application_id FROM raw.applications
                WHERE application_date > '2023-06-30'
            ) t
        """
        result = pd.read_sql(overlap_query, engine)
        overlap = result["overlap_count"].iloc[0]
        if overlap > 0:
            issues["train_test_overlap"] = (
                f"Found {overlap} overlapping IDs!"
            )

    passed = len(issues) == 0
    if not passed:
        logger.error(f"Leakage check FAILED: {issues}")
    else:
        logger.info("Leakage check PASSED")

    return LeakageResult(passed=passed, details=issues)