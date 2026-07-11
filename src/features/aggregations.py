"""Payment-history aggregation features (strictly pre-application)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)
from sqlalchemy import text
from sqlalchemy.engine import Engine


class AggregationFeatureComputer:
    """
    Aggregates from payment_history over rolling windows.
    Only payments with payment_date < application_date are used (no leakage).
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        date_col: str = "payment_date",
        reference_date: str | None = None,
    ):
        self.windows = windows or [30, 90, 180]
        self.date_col = date_col
        self.reference_date = reference_date

    def compute(self, engine: Engine) -> pd.DataFrame:
        logger.info("Computing aggregation features...")

        query = """
            SELECT
                ph.application_id,
                ph.payment_date,
                ph.amount_due,
                ph.amount_paid,
                ph.days_overdue,
                a.application_date
            FROM raw.payment_history ph
            JOIN raw.applications a
                ON ph.application_id = a.application_id
        """
        params = {}
        if self.reference_date:
            query += " WHERE a.application_date <= :ref_date"
            params["ref_date"] = self.reference_date

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params or None)

        if df.empty:
            logger.warning("No payment history rows; returning empty aggregations")
            return pd.DataFrame(columns=["application_id"])

        df["payment_date"] = pd.to_datetime(df["payment_date"])
        df["application_date"] = pd.to_datetime(df["application_date"])

        # Global pre-application filter
        df = df[df["payment_date"] < df["application_date"]].copy()
        if df.empty:
            logger.warning("All payments are on/after application_date")
            with engine.connect() as conn:
                apps = pd.read_sql(
                    text("SELECT application_id FROM raw.applications"),
                    conn,
                )
            return apps

        parts = [
            self._compute_window_features(df, window_days)
            for window_days in self.windows
        ]

        result = parts[0]
        for part in parts[1:]:
            result = result.merge(part, on="application_id", how="outer")

        # Canonical columns expected by FeatureConfig / SQL
        result = self._ensure_canonical_columns(result)
        logger.info(f"Aggregation features computed: {result.shape}")
        return result

    def compute_from_frames(
        self,
        payments: pd.DataFrame,
        applications: pd.DataFrame,
    ) -> pd.DataFrame:
        """Pure-pandas path for unit tests (no DB)."""
        df = payments.merge(
            applications[["application_id", "application_date"]],
            on="application_id",
            how="inner",
        )
        df["payment_date"] = pd.to_datetime(df["payment_date"])
        df["application_date"] = pd.to_datetime(df["application_date"])
        df = df[df["payment_date"] < df["application_date"]].copy()

        if df.empty:
            return pd.DataFrame({"application_id": applications["application_id"].unique()})

        parts = [
            self._compute_window_features(df, window_days)
            for window_days in self.windows
        ]
        result = parts[0]
        for part in parts[1:]:
            result = result.merge(part, on="application_id", how="outer")
        return self._ensure_canonical_columns(result)

    def _compute_window_features(
        self,
        df: pd.DataFrame,
        window_days: int,
    ) -> pd.DataFrame:
        suffix = f"_{window_days}d"
        window_start = df["application_date"] - pd.Timedelta(days=window_days)
        df_window = df[
            (df["payment_date"] >= window_start)
            & (df["payment_date"] < df["application_date"])
        ].copy()

        if df_window.empty:
            return pd.DataFrame(columns=["application_id"])

        agg = (
            df_window.groupby("application_id", as_index=False)
            .agg(
                **{
                    f"avg_days_overdue{suffix}": ("days_overdue", "mean"),
                    f"max_days_overdue{suffix}": ("days_overdue", "max"),
                    f"total_paid{suffix}": ("amount_paid", "sum"),
                    f"num_payments{suffix}": ("amount_paid", "count"),
                    f"std_days_overdue{suffix}": ("days_overdue", "std"),
                }
            )
        )

        late = df_window[df_window["days_overdue"] > 0]
        late_counts = late.groupby("application_id").size()
        total_counts = df_window.groupby("application_id").size()
        pct = (late_counts / total_counts).rename(f"pct_late_payments{suffix}")
        pct = pct.reset_index()
        pct.columns = ["application_id", f"pct_late_payments{suffix}"]

        agg = agg.merge(pct, on="application_id", how="left")
        agg[f"pct_late_payments{suffix}"] = agg[f"pct_late_payments{suffix}"].fillna(0.0)

        # Consistency: 1 - normalized overdue volatility (clipped)
        std_col = f"std_days_overdue{suffix}"
        if std_col in agg.columns:
            agg[f"payment_consistency{suffix}"] = (
                1.0 - (agg[std_col].fillna(0.0) / 90.0)
            ).clip(0.0, 1.0)
        else:
            agg[f"payment_consistency{suffix}"] = 1.0

        return agg

    def _ensure_canonical_columns(self, result: pd.DataFrame) -> pd.DataFrame:
        required = [
            "avg_days_overdue_30d",
            "avg_days_overdue_90d",
            "avg_days_overdue_180d",
            "max_days_overdue_90d",
            "pct_late_payments_90d",
            "total_paid_90d",
            "payment_consistency_90d",
        ]
        for col in required:
            if col not in result.columns:
                result[col] = np.nan
        # Prefer explicit 90d consistency if window produced it
        if "payment_consistency_90d" not in result.columns and "payment_consistency_90d" in required:
            result["payment_consistency_90d"] = result.get(
                "payment_consistency_90d", np.nan
            )
        return result
