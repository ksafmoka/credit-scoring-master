# src/features/aggregations.py

import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy.engine import Engine


class AggregationFeatureComputer:
    """
    Агрегаты из payment_history по временным окнам.
    Важно: все агрегаты считаются ТОЛЬКО по данным
    до даты заявки (no leakage!).
    """

    def __init__(
        self,
        windows: list[int],
        date_col: str = "payment_date",
        reference_date: str | None = None,
    ):
        self.windows = windows
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
        if self.reference_date:
            query += (
                f" WHERE a.application_date <= '{self.reference_date}'"
            )

        df = pd.read_sql(query, engine)
        df["payment_date"] = pd.to_datetime(df["payment_date"])
        df["application_date"] = pd.to_datetime(
            df["application_date"]
        )

        results = []
        for window in self.windows:
            agg = self._compute_window_features(df, window)
            results.append(agg)

        # merge all windows
        final = results[0]
        for r in results[1:]:
            final = final.merge(r, on="application_id", how="outer")

        logger.info(f"Aggregation features computed: {final.shape}")
        return final

    def _compute_window_features(
        self,
        df: pd.DataFrame,
        window_days: int,
    ) -> pd.DataFrame:
        """Агрегаты для одного окна."""
        suffix = f"_{window_days}d"

        # оставляем только платежи в окне [app_date - window, app_date]
        df_window = df[
            (df["payment_date"] >= df["application_date"]
             - pd.Timedelta(days=window_days))
            & (df["payment_date"] < df["application_date"])
        ].copy()

        agg = df_window.groupby("application_id").agg(
            **{
                f"avg_days_overdue{suffix}": (
                    "days_overdue", "mean"
                ),
                f"max_days_overdue{suffix}": (
                    "days_overdue", "max"
                ),
                f"total_paid{suffix}": (
                    "amount_paid", "sum"
                ),
                f"num_payments{suffix}": (
                    "payment_id", "count"
                ) if "payment_id" in df_window.columns else (
                    "amount_paid", "count"
                ),
            }
        ).reset_index()

        # pct late payments
        late = df_window[df_window["days_overdue"] > 0]
        late_counts = late.groupby("application_id").size().rename(
            f"late_count{suffix}"
        )
        total_counts = df_window.groupby("application_id").size().rename(
            f"total_count{suffix}"
        )
        pct_late = (late_counts / total_counts).rename(
            f"pct_late_payments{suffix}"
        ).reset_index()
        pct_late.columns = ["application_id", f"pct_late_payments{suffix}"]

        agg = agg.merge(pct_late, on="application_id", how="left")
        agg[f"pct_late_payments{suffix}"] = agg[
            f"pct_late_payments{suffix}"
        ].fillna(0)

        return agg