# src/features/numerical.py

import numpy as np
import pandas as pd
from loguru import logger


class NumericalFeatureComputer:
    """Числовые трансформации и производные фичи."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing numerical features...")
        result = pd.DataFrame()
        result["application_id"] = df["application_id"]
        result["feature_date"] = df["application_date"]

        result["loan_to_income"] = self._loan_to_income(df)
        result["credit_utilization"] = self._credit_utilization(df)
        result["income_log"] = self._log_transform(df["income"])
        result["loan_amount_log"] = self._log_transform(
            df["loan_amount"]
        )
        result["dti_ratio_clipped"] = df["dti_ratio"].clip(0, 100)
        result["dti_bucket"] = self._dti_bucket(df["dti_ratio"])
        result["credit_score_bucket"] = self._credit_score_bucket(
            df["credit_score"]
        )
        result["loan_amount_x_dti"] = (
            df["loan_amount"] * df["dti_ratio"]
        )
        result["income_x_credit_score"] = (
            df["income"] * df["credit_score"]
        )

        logger.info(
            f"Numerical features computed: {result.shape}"
        )
        return result

    def _loan_to_income(self, df: pd.DataFrame) -> pd.Series:
        return np.where(
            df["income"] > 0,
            df["loan_amount"] / df["income"],
            np.nan,
        )

    def _credit_utilization(self, df: pd.DataFrame) -> pd.Series:
        return np.where(
            df["total_credit_limit"] > 0,
            df["loan_amount"] / df["total_credit_limit"],
            np.nan,
        )

    def _log_transform(self, series: pd.Series) -> pd.Series:
        return np.log1p(series.clip(lower=0))

    def _dti_bucket(self, dti: pd.Series) -> pd.Series:
        return pd.cut(
            dti,
            bins=[-np.inf, 10, 20, 35, 50, np.inf],
            labels=["very_low", "low", "medium", "high", "very_high"],
        ).astype(str)

    def _credit_score_bucket(self, score: pd.Series) -> pd.Series:
        return pd.cut(
            score,
            bins=[0, 580, 670, 740, 800, 850],
            labels=["poor", "fair", "good", "very_good", "exceptional"],
        ).astype(str)