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

    Batched implementation: old version joined 1.8M payments * 2.2M apps in one query,
    causing long Postgres query and Airflow heartbeat timeout. Now loads payments
    once (1.8M) and processes application batches in pandas using compute_from_frames.
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

    def compute(
        self,
        engine: Engine,
        batch_size: int | None = None,
        max_apps: int | None = None,
        sample_strategy: str | None = None,
    ) -> pd.DataFrame:
        from src.config import FeatureEngineeringConfig, IngestionConfig

        batch_size = batch_size or FeatureEngineeringConfig.AGGREGATION_BATCH_SIZE
        if max_apps is None:
            max_apps = FeatureEngineeringConfig.MAX_APPS
        sample_strategy = (sample_strategy or IngestionConfig.SAMPLE_STRATEGY).lower()

        logger.info(
            f"Computing aggregation features batched: batch_size={batch_size}, "
            f"max_apps={max_apps}, strategy={sample_strategy}"
        )

        # Load payment history once (with optional reference_date filter)
        ph_query = """
            SELECT
                ph.application_id,
                ph.payment_date,
                ph.amount_due,
                ph.amount_paid,
                ph.days_overdue
            FROM raw.payment_history ph
        """
        ph_params: dict = {}
        if self.reference_date:
            # Need application_date for pre-filter, join only for reference_date
            ph_query = """
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
                WHERE a.application_date <= :ref_date
            """
            ph_params["ref_date"] = self.reference_date

        with engine.connect() as conn:
            payments = pd.read_sql(text(ph_query), conn, params=ph_params or None)

        if payments.empty:
            logger.warning("No payment history rows; returning empty aggregations")
            return pd.DataFrame(columns=["application_id"])

        logger.info(f"Loaded payment history: {len(payments)} rows")

        # Load applications in batches for merging
        count_q = "SELECT COUNT(*) FROM raw.applications"
        count_params: dict = {}
        if self.reference_date:
            count_q += " WHERE application_date <= :ref_date"
            count_params["ref_date"] = self.reference_date

        with engine.connect() as conn:
            total = int(conn.execute(text(count_q), count_params or None).scalar() or 0)
        if max_apps is not None:
            total = min(total, int(max_apps))
        if total == 0:
            return pd.DataFrame(columns=["application_id"])

        logger.info(f"Aggregation total apps to process: {total}")

        # Random pre-selection for uniform train coverage (variant A)
        random_ids = None
        if sample_strategy == "random" and max_apps is not None:
            with engine.connect() as conn:
                id_df = pd.read_sql(
                    text(
                        f"SELECT application_id, application_date FROM raw.applications "
                        f"{'WHERE ' + count_q.split('WHERE',1)[1] if 'WHERE' in count_q else ''} "
                        f"ORDER BY random() LIMIT :limit"
                    ),
                    conn,
                    params={**count_params, "limit": int(max_apps)},
                )
                random_ids = id_df[["application_id", "application_date"]]
                total = len(random_ids)
                logger.info(f"Aggregation random_ids selected: {total}")

        results = []
        for offset in range(0, total, batch_size):
            limit = min(batch_size, total - offset)

            if random_ids is not None:
                batch_apps = random_ids.iloc[offset : offset + limit]
                apps = batch_apps
            else:
                order = "application_date DESC, application_id DESC" if max_apps else "application_id"
                app_query = f"""
                    SELECT application_id, application_date
                    FROM raw.applications
                    {f'WHERE application_date <= :ref_date' if self.reference_date else ''}
                    ORDER BY {order}
                    LIMIT :limit OFFSET :offset
                """
                app_params = dict(count_params)
                app_params["limit"] = limit
                app_params["offset"] = offset

                with engine.connect() as conn:
                    apps = pd.read_sql(text(app_query), conn, params=app_params)

            if apps.empty:
                continue

            batch_result = self.compute_from_frames(payments, apps)
            # Ensure application_id list for apps without payments
            if batch_result.empty:
                batch_result = pd.DataFrame(
                    {"application_id": apps["application_id"].unique()}
                )
            else:
                # Add missing app_ids that had no payments
                missing_ids = set(apps["application_id"]) - set(
                    batch_result["application_id"]
                )
                if missing_ids:
                    missing_df = pd.DataFrame(
                        {"application_id": list(missing_ids)}
                    )
                    batch_result = pd.concat(
                        [batch_result, missing_df], ignore_index=True
                    )

            results.append(batch_result)
            if (offset // batch_size) % 5 == 0:
                logger.info(
                    f"Aggregation batch {offset // batch_size + 1}: "
                    f"{len(batch_result)} rows, processed {offset + len(apps)}/{total}"
                )

        if not results:
            return pd.DataFrame(columns=["application_id"])

        final = pd.concat(results, ignore_index=True)
        # Deduplicate by application_id (keep first)
        final = final.drop_duplicates(subset=["application_id"])
        final = self._ensure_canonical_columns(final)
        logger.info(f"Aggregation features computed: {final.shape}")
        return final

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
