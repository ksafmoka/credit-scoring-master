"""Credit bureau derived features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)
from sqlalchemy import text
from sqlalchemy.engine import Engine


class BureauFeatureComputer:
    """Join latest pre-application bureau report and derive ratios."""

    def compute(self, engine: Engine, reference_date: str | None = None) -> pd.DataFrame:
        logger.info("Computing bureau features...")
        params: dict = {}
        app_filter = ""
        if reference_date:
            app_filter = "AND a.application_date <= :ref_date"
            params["ref_date"] = reference_date

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
            {app_filter}
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params or None)

        if df.empty:
            return pd.DataFrame(columns=["application_id"])

        income = pd.to_numeric(df["income"], errors="coerce").replace(0, np.nan)
        balance = pd.to_numeric(df["total_balance"], errors="coerce")
        inquiries = pd.to_numeric(df["num_inquiries_6m"], errors="coerce")
        active = pd.to_numeric(df["num_active_loans"], errors="coerce").replace(0, np.nan)

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
        logger.info(f"Bureau features computed: {out.shape}")
        return out
