"""Scoring metrics and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sqlalchemy import text
from sqlalchemy.engine import Engine


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "val",
) -> dict:
    """Full metric set for a PD model."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    metrics: dict[str, float] = {}
    metrics[f"{prefix}_auc_roc"] = float(roc_auc_score(y_true, y_pred))
    metrics[f"{prefix}_auc_pr"] = float(average_precision_score(y_true, y_pred))
    metrics[f"{prefix}_brier_score"] = float(brier_score_loss(y_true, y_pred))
    metrics[f"{prefix}_log_loss"] = float(log_loss(y_true, y_pred))

    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    if len(pos) and len(neg):
        ks_stat, _ = stats.ks_2samp(pos, neg)
        metrics[f"{prefix}_ks_statistic"] = float(ks_stat)
    else:
        metrics[f"{prefix}_ks_statistic"] = 0.0

    metrics[f"{prefix}_gini"] = float(2 * metrics[f"{prefix}_auc_roc"] - 1)

    logger.info(
        f"{prefix} metrics: "
        f"AUC={metrics[f'{prefix}_auc_roc']:.4f}, "
        f"KS={metrics[f'{prefix}_ks_statistic']:.4f}, "
        f"Gini={metrics[f'{prefix}_gini']:.4f}"
    )
    return metrics


def get_risk_bucket(pd_score: float) -> str:
    if pd_score < 0.05:
        return "LOW"
    if pd_score < 0.15:
        return "MEDIUM"
    if pd_score < 0.30:
        return "HIGH"
    return "VERY_HIGH"


@dataclass
class LeakageResult:
    passed: bool
    details: dict


def leakage_check(engine: Engine, checks: list[str]) -> LeakageResult:
    """Basic temporal leakage checks against the database."""
    issues: dict[str, str] = {}

    if "train_test_overlap" in checks:
        overlap_query = text(
            """
            SELECT COUNT(*) AS overlap_count
            FROM (
                SELECT application_id FROM raw.applications
                WHERE application_date <= '2022-12-31'
                INTERSECT
                SELECT application_id FROM raw.applications
                WHERE application_date > '2023-06-30'
            ) t
            """
        )
        with engine.connect() as conn:
            result = pd.read_sql(overlap_query, conn)
        overlap = int(result["overlap_count"].iloc[0])
        if overlap > 0:
            issues["train_test_overlap"] = f"Found {overlap} overlapping IDs!"

    if "future_payments" in checks:
        q = text(
            """
            SELECT COUNT(*) AS cnt
            FROM raw.payment_history ph
            JOIN raw.applications a ON a.application_id = ph.application_id
            WHERE ph.payment_date >= a.application_date
            """
        )
        with engine.connect() as conn:
            result = pd.read_sql(q, conn)
        cnt = int(result["cnt"].iloc[0])
        if cnt > 0:
            issues["future_payments"] = (
                f"Found {cnt} payments on/after application_date"
            )

    passed = len(issues) == 0
    if not passed:
        logger.error(f"Leakage check FAILED: {issues}")
    else:
        logger.info("Leakage check PASSED")
    return LeakageResult(passed=passed, details=issues)
