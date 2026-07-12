"""Numerical and cross features from raw applications."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)


class NumericalFeatureComputer:
    """Application-level numerical transforms and cross features."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing numerical features...")
        result = pd.DataFrame(index=df.index)
        result["application_id"] = df["application_id"].values
        result["feature_date"] = pd.to_datetime(df["application_date"]).values

        result["loan_to_income"] = self._loan_to_income(df)
        result["credit_utilization"] = self._credit_utilization(df)
        result["income_log"] = self._log_transform(df["income"])
        result["loan_amount_log"] = self._log_transform(df["loan_amount"])
        dti = pd.to_numeric(df.get("dti_ratio"), errors="coerce")
        result["dti_ratio_clipped"] = dti.clip(0, 100)

        result["employment_years"] = pd.to_numeric(
            df.get("employment_years"), errors="coerce"
        ).fillna(0.0)
        credit = pd.to_numeric(df.get("credit_score"), errors="coerce")
        # Normalize FICO-like scores to ~[0, 1]
        result["credit_score_norm"] = ((credit - 300.0) / 550.0).clip(0, 1)
        result["num_open_accounts"] = pd.to_numeric(
            df.get("num_open_accounts"), errors="coerce"
        ).fillna(0)
        result["num_delinquencies"] = pd.to_numeric(
            df.get("num_delinquencies"), errors="coerce"
        ).fillna(0)
        result["interest_rate"] = pd.to_numeric(
            df.get("interest_rate"), errors="coerce"
        )

        loan_amount = pd.to_numeric(df["loan_amount"], errors="coerce")
        income = pd.to_numeric(df["income"], errors="coerce")
        result["loan_amount_x_dti"] = loan_amount * dti.fillna(0)
        result["income_x_credit_score"] = income * credit.fillna(0)

        # Optional descriptive buckets (not in model feature list)
        result["dti_bucket"] = self._dti_bucket(dti)
        result["credit_score_bucket"] = self._credit_score_bucket(credit)

        logger.info(f"Numerical features computed: {result.shape}")
        return result

    def _loan_to_income(self, df: pd.DataFrame) -> pd.Series:
        income = pd.to_numeric(df["income"], errors="coerce")
        loan = pd.to_numeric(df["loan_amount"], errors="coerce")
        return np.where(income > 0, loan / income, np.nan)

    def _credit_utilization(self, df: pd.DataFrame) -> pd.Series:
        limit = pd.to_numeric(df.get("total_credit_limit"), errors="coerce")
        loan = pd.to_numeric(df["loan_amount"], errors="coerce")
        return np.where(limit > 0, loan / limit, np.nan)

    def _log_transform(self, series: pd.Series) -> pd.Series:
        return np.log1p(pd.to_numeric(series, errors="coerce").clip(lower=0))

    def _dti_bucket(self, dti: pd.Series) -> pd.Series:
        return pd.cut(
            dti,
            bins=[-np.inf, 10, 20, 35, 50, np.inf],
            labels=["very_low", "low", "medium", "high", "very_high"],
        ).astype(str)

    def _credit_score_bucket(self, score: pd.Series) -> pd.Series:
        return pd.cut(
            score,
            bins=[0, 580, 670, 740, 800, 900],
            labels=["poor", "fair", "good", "very_good", "exceptional"],
        ).astype(str)
