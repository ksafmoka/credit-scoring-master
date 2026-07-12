"""Feature drift monitoring (PSI / KS) + Telegram alerts on drift."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

import numpy as np
import pandas as pd
from scipy import stats

from src.config import MonitoringConfig
from src.logging_utils import get_logger

logger = get_logger(__name__)


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
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _feature_date_bounds(engine: Engine):
    from sqlalchemy import text

    with engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT MIN(feature_date) AS dmin, MAX(feature_date) AS dmax "
                    "FROM features.application_features"
                )
            )
            .mappings()
            .first()
        )
    if not row or row["dmin"] is None or row["dmax"] is None:
        return None
    return pd.Timestamp(row["dmin"]), pd.Timestamp(row["dmax"])


def resolve_drift_windows(engine: Engine) -> dict[str, str] | None:
    """
    Windows from actual feature_date range in DB.

    Using Airflow "today" against historical loan dates often yields empty sets.
    """
    bounds = _feature_date_bounds(engine)
    if bounds is None:
        return None
    dmin, dmax = bounds
    span_days = max(int((dmax - dmin).days), 1)
    cur_days = max(min(span_days // 6, 120), min(30, span_days))
    current_start = (dmax - pd.Timedelta(days=cur_days)).strftime("%Y-%m-%d")
    current_end = dmax.strftime("%Y-%m-%d")
    ref_end_ts = dmax - pd.Timedelta(days=cur_days + max(span_days // 10, 7))
    if ref_end_ts <= dmin:
        ref_end_ts = dmin + pd.Timedelta(days=max(span_days // 3, 1))
    ref_start_ts = max(dmin, ref_end_ts - pd.Timedelta(days=cur_days))
    return {
        "reference_start": ref_start_ts.strftime("%Y-%m-%d"),
        "reference_end": ref_end_ts.strftime("%Y-%m-%d"),
        "current_start": current_start,
        "current_end": current_end,
    }


def ensure_alert_queue(engine: Engine) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS monitoring.alert_queue (
                    alert_id   BIGSERIAL PRIMARY KEY,
                    message    TEXT NOT NULL,
                    status     VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    sent_at    TIMESTAMP,
                    error      TEXT
                )
                """
            )
        )


def enqueue_alert(message: str, engine: Engine | None = None) -> bool:
    """Store alert for host-side telegram_notifier.py (Docker often blocks TG)."""
    from sqlalchemy import text

    try:
        if engine is None:
            from src.config import get_db_engine

            engine = get_db_engine()
        ensure_alert_queue(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO monitoring.alert_queue (message, status) "
                    "VALUES (:message, 'pending')"
                ),
                {"message": message[:4000]},
            )
        logger.info("Drift alert queued → run: make telegram-once (on host)")
        return True
    except Exception as exc:
        logger.error(f"Failed to queue alert: {exc}")
        return False


def send_telegram_alert(
    message: str,
    engine: Engine | None = None,
) -> bool:
    """
    Send drift alert to Telegram.

    1) Try Bot API from this process (works on host / open networks).
    2) If Docker cannot reach api.telegram.org → queue for host notifier.
    """
    import requests

    token = MonitoringConfig.telegram_bot_token()
    chat_id = MonitoringConfig.telegram_chat_id()
    if not token or not chat_id:
        logger.warning(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — queueing alert only"
        )
        return enqueue_alert(message, engine=engine)

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message[:4000],
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        data = resp.json() if resp.content else {}
        if resp.ok and data.get("ok"):
            logger.info("Telegram drift alert sent (direct)")
            return True
        logger.error(f"Telegram API error: {resp.status_code} {data}")
    except Exception as exc:
        logger.warning(
            f"Direct Telegram failed ({exc}); queuing for host notifier"
        )

    return enqueue_alert(message, engine=engine)


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

    logger.info(
        f"Drift windows ref={reference_start}..{reference_end} n={len(ref_df)} | "
        f"current={current_start}..{current_end} n={len(cur_df)}"
    )

    if len(ref_df) < 50 or len(cur_df) < 50:
        logger.warning(
            f"Not enough data for drift (ref={len(ref_df)}, cur={len(cur_df)}). Skip."
        )
        return []

    results: list[dict] = []
    for feature in safe_features:
        if feature not in ref_df.columns or feature not in cur_df.columns:
            continue
        psi = compute_psi(ref_df[feature], cur_df[feature])
        ks_stat, ks_pval = stats.ks_2samp(
            pd.to_numeric(ref_df[feature], errors="coerce").dropna(),
            pd.to_numeric(cur_df[feature], errors="coerce").dropna(),
        )
        is_drifted = psi > MonitoringConfig.PSI_THRESHOLD
        is_warning = psi > MonitoringConfig.PSI_WARNING and not is_drifted

        # As designed: notify on feature drift (and soft warning)
        if is_drifted:
            logger.error(f"DRIFT: {feature}, PSI={psi:.4f}")
            send_telegram_alert(
                "🚨 Feature drift detected\n"
                f"Feature: {feature}\n"
                f"PSI: {psi:.4f} (threshold: {MonitoringConfig.PSI_THRESHOLD})\n"
                f"Current: {current_start} → {current_end}\n"
                f"Reference: {reference_start} → {reference_end}",
                engine=engine,
            )
        elif is_warning:
            logger.warning(f"WARNING drift: {feature}, PSI={psi:.4f}")
            send_telegram_alert(
                "⚠️ Feature drift warning\n"
                f"Feature: {feature}\n"
                f"PSI: {psi:.4f} (warning: {MonitoringConfig.PSI_WARNING})",
                engine=engine,
            )

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
