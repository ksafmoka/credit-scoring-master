"""Feature drift monitoring (PSI / KS)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)
from scipy import stats

from src.config import MonitoringConfig


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """Population Stability Index."""
    ref_clean = pd.to_numeric(reference, errors="coerce").dropna()
    cur_clean = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return 0.0

    breakpoints = np.unique(
        np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    )
    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def run_drift_check(
    engine: Engine,
    reference_start: str,
    reference_end: str,
    current_start: str,
    current_end: str,
    features: list[str],
    check_date: str,
) -> list[dict]:
    from sqlalchemy import text

    safe_features = [f for f in features if f.isidentifier()]
    if not safe_features:
        return []

    cols = ", ".join(safe_features)
    with engine.connect() as conn:
        ref_df = pd.read_sql(
            text(
                f"""
                SELECT {cols}
                FROM features.application_features
                WHERE feature_date BETWEEN :reference_start AND :reference_end
                """
            ),
            conn,
            params={
                "reference_start": reference_start,
                "reference_end": reference_end,
            },
        )
        cur_df = pd.read_sql(
            text(
                f"""
                SELECT {cols}
                FROM features.application_features
                WHERE feature_date BETWEEN :current_start AND :current_end
                """
            ),
            conn,
            params={
                "current_start": current_start,
                "current_end": current_end,
            },
        )

    if len(ref_df) < 100 or len(cur_df) < 100:
        logger.warning("Not enough data for drift check. Skipping.")
        return []

    results: list[dict] = []
    for feature in safe_features:
        if feature not in ref_df.columns:
            continue
        psi = compute_psi(ref_df[feature], cur_df[feature])
        ks_stat, ks_pval = stats.ks_2samp(
            pd.to_numeric(ref_df[feature], errors="coerce").dropna(),
            pd.to_numeric(cur_df[feature], errors="coerce").dropna(),
        )
        is_drifted = psi > MonitoringConfig.PSI_THRESHOLD
        is_warning = psi > MonitoringConfig.PSI_WARNING and not is_drifted

        if is_drifted:
            logger.error(f"DRIFT: {feature}, PSI={psi:.4f}")
            _send_alert(
                f"Feature drift detected!\nFeature: {feature}\n"
                f"PSI: {psi:.4f} (threshold: {MonitoringConfig.PSI_THRESHOLD})"
            )
        elif is_warning:
            logger.warning(f"WARNING drift: {feature}, PSI={psi:.4f}")

        results.append(
            {
                "feature_name": feature,
                "check_date": check_date,
                "psi_value": psi,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "is_drifted": bool(is_drifted),
                "reference_period": f"{reference_start}/{reference_end}",
                "current_period": f"{current_start}/{current_end}",
            }
        )

    if results:
        with engine.begin() as conn:
            pd.DataFrame(results).to_sql(
                "feature_drift",
                conn,
                schema="monitoring",
                if_exists="append",
                index=False,
            )
    return results


def _send_alert(message: str) -> None:
    import requests

    token = MonitoringConfig.TELEGRAM_BOT_TOKEN
    chat_id = MonitoringConfig.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        logger.warning("Telegram not configured, skipping alert")
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=15,
        )
    except Exception as exc:
        logger.error(f"Failed to send alert: {exc}")
