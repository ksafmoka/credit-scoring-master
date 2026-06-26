# src/monitoring/data_drift.py

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger
from sqlalchemy.engine import Engine

from src.config import MonitoringConfig


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """Population Stability Index."""
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    breakpoints = np.percentile(
        ref_clean, np.linspace(0, 100, n_bins + 1)
    )
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def run_drift_check(
    engine: Engine,
    reference_start: str,
    reference_end: str,
    current_start: str,
    current_end: str,
    features: list[str],
    check_date: str,
) -> list[dict]:
    """Полная проверка дрифта с записью в monitoring схему."""

    ref_df = pd.read_sql(
        f"""
        SELECT {', '.join(features)}
        FROM features.application_features
        WHERE feature_date BETWEEN '{reference_start}'
              AND '{reference_end}'
        """,
        engine,
    )

    cur_df = pd.read_sql(
        f"""
        SELECT {', '.join(features)}
        FROM features.application_features
        WHERE feature_date BETWEEN '{current_start}'
              AND '{current_end}'
        """,
        engine,
    )

    if len(ref_df) < 100 or len(cur_df) < 100:
        logger.warning(
            "Not enough data for drift check. Skipping."
        )
        return []

    results = []
    for feature in features:
        if feature not in ref_df.columns:
            continue

        psi = compute_psi(ref_df[feature], cur_df[feature])

        ks_stat, ks_pval = stats.ks_2samp(
            ref_df[feature].dropna(),
            cur_df[feature].dropna(),
        )

        is_drifted = psi > MonitoringConfig.PSI_THRESHOLD
        is_warning = (
            psi > MonitoringConfig.PSI_WARNING and not is_drifted
        )

        if is_drifted:
            logger.error(
                f"DRIFT: {feature}, PSI={psi:.4f}"
            )
            _send_alert(
                f"⚠️ Feature drift detected!\n"
                f"Feature: {feature}\n"
                f"PSI: {psi:.4f} (threshold: "
                f"{MonitoringConfig.PSI_THRESHOLD})"
            )
        elif is_warning:
            logger.warning(
                f"WARNING drift: {feature}, PSI={psi:.4f}"
            )

        results.append({
            "feature_name": feature,
            "check_date": check_date,
            "psi_value": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "is_drifted": is_drifted,
            "reference_period": f"{reference_start}/{reference_end}",
            "current_period": f"{current_start}/{current_end}",
        })

    # записываем результаты
    if results:
        pd.DataFrame(results).to_sql(
            "feature_drift",
            engine,
            schema="monitoring",
            if_exists="append",
            index=False,
        )

    return results


def _send_alert(message: str) -> None:
    """Отправка алерта в Telegram."""
    import requests
    from src.config import MonitoringConfig

    token = MonitoringConfig.TELEGRAM_BOT_TOKEN
    chat_id = MonitoringConfig.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        logger.warning(
            "Telegram not configured, skipping alert"
        )
        return

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=5,
        )
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")