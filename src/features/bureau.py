"""Credit bureau derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)
from sqlalchemy import text
from sqlalchemy.engine import Engine


class BureauFeatureComputer:
    """Join latest pre-application bureau report and derive ratios.

    Batched implementation to avoid 2.2M lateral join in one query (was 6min+ and caused
    Airflow heartbeat timeout + DNS resolution failure under Postgres overload).
    Processes in configurable batches (default 100k apps per query).
    """

    def compute(
        self,
        engine: Engine,
        reference_date: str | None = None,
        batch_size: int | None = None,
        max_apps: int | None = None,
        sample_strategy: str | None = None,
    ) -> pd.DataFrame:
        from src.config import FeatureEngineeringConfig, IngestionConfig

        batch_size = batch_size or FeatureEngineeringConfig.BUREAU_BATCH_SIZE
        if max_apps is None:
            max_apps = FeatureEngineeringConfig.MAX_APPS
        sample_strategy = (sample_strategy or IngestionConfig.SAMPLE_STRATEGY).lower()

        logger.info(
            f"Computing bureau features batched: batch_size={batch_size}, "
            f"max_apps={max_apps}, strategy={sample_strategy}"
        )

        params: dict = {}
        base_filter = ""
        if reference_date:
            base_filter = "AND a.application_date <= :ref_date"
            params["ref_date"] = reference_date

        # Count total to process
        count_q = f"SELECT COUNT(*) FROM raw.applications a WHERE 1=1 {base_filter}"
        with engine.connect() as conn:
            total = int(
                conn.execute(text(count_q), params or None).scalar() or 0
            )
        if max_apps is not None:
            total = min(total, int(max_apps))
        if total == 0:
            return pd.DataFrame(columns=["application_id"])

        logger.info(f"Bureau total apps to process: {total}")

        # If random sampling with max_apps, pre-select random ids once for consistency
        random_ids: list[int] | None = None
        if sample_strategy == "random" and max_apps is not None:
            with engine.connect() as conn:
                id_rows = pd.read_sql(
                    text(
                        f"SELECT application_id FROM raw.applications "
                        f"{'WHERE ' + base_filter.replace('AND','',1) if base_filter else ''} "
                        f"ORDER BY random() LIMIT :limit"
                    ),
                    conn,
                    params={**params, "limit": int(max_apps)},
                )
                random_ids = id_rows["application_id"].tolist()
                total = len(random_ids)
                logger.info(f"Bureau random_ids selected: {total}")

        results = []
        # Iterate by application_id ordering for stable pagination
        for offset in range(0, total, batch_size):
            limit = min(batch_size, total - offset)

            if random_ids is not None:
                batch_ids = random_ids[offset : offset + limit]
                if not batch_ids:
                    continue
                query = f"""
                    SELECT
                        a.application_id,
                        a.income,
                        a.application_date,
                        b.num_inquiries_6m,
                        b.num_active_loans,
                        b.total_balance,
                        b.num_defaults_hist,
                        b.oldest_account_months,
                        b.report_date
                    FROM raw.applications a
                    LEFT JOIN LATERAL (
                        SELECT *
                        FROM raw.credit_bureau b
                        WHERE b.client_id = a.client_id
                          AND b.report_date < a.application_date
                        ORDER BY b.report_date DESC
                        LIMIT 1
                    ) b ON TRUE
                    WHERE a.application_id = ANY(:ids)
                """
                batch_params = {"ids": batch_ids}
            else:
                # If max_apps, we order by application_date DESC to match ingestion recent bias,
                # otherwise by application_id for determinism
                order = "a.application_date DESC, a.application_id DESC" if max_apps else "a.application_id"
                query = f"""
                    SELECT
                        a.application_id,
                        a.income,
                        a.application_date,
                        b.num_inquiries_6m,
                        b.num_active_loans,
                        b.total_balance,
                        b.num_defaults_hist,
                        b.oldest_account_months,
                        b.report_date
                    FROM raw.applications a
                    LEFT JOIN LATERAL (
                        SELECT *
                        FROM raw.credit_bureau b
                        WHERE b.client_id = a.client_id
                          AND b.report_date < a.application_date
                        ORDER BY b.report_date DESC
                        LIMIT 1
                    ) b ON TRUE
                    WHERE 1=1
                    {base_filter}
                    ORDER BY {order}
                    LIMIT :limit OFFSET :offset
                """
                batch_params = dict(params)
                batch_params["limit"] = limit
                batch_params["offset"] = offset

            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=batch_params)

            if df.empty:
                continue

            income = pd.to_numeric(df["income"], errors="coerce").replace(0, np.nan)
            balance = pd.to_numeric(df["total_balance"], errors="coerce")
            inquiries = pd.to_numeric(df["num_inquiries_6m"], errors="coerce")
            active = pd.to_numeric(df["num_active_loans"], errors="coerce").replace(
                0, np.nan
            )

            out = pd.DataFrame(
                {
                    "application_id": df["application_id"],
                    "bureau_balance_to_income": (balance / income).replace(
                        [np.inf, -np.inf], np.nan
                    ),
                    "inquiries_per_account": (inquiries / active).replace(
                        [np.inf, -np.inf], np.nan
                    ),
                    "avg_account_age_months": pd.to_numeric(
                        df["oldest_account_months"], errors="coerce"
                    ),
                }
            )
            results.append(out)
            if (offset // batch_size) % 5 == 0:
                logger.info(
                    f"Bureau batch {offset // batch_size + 1}: "
                    f"{len(out)} rows, processed {offset + len(out)}/{total}"
                )

        if not results:
            return pd.DataFrame(columns=["application_id"])

        final = pd.concat(results, ignore_index=True)
        logger.info(f"Bureau features computed: {final.shape}")
        return final
