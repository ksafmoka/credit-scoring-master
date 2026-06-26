# src/features/target_encoding.py

import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy.engine import Engine


class RegularizedTargetEncoder:
    """
    Target encoding с регуляризацией для предотвращения overfitting.
    Используется только на train-периоде, применяется на всём датасете.

    target_enc = (category_mean * n + global_mean * smoothing)
                 / (n + smoothing)
    """

    def __init__(
        self,
        cols: list[str],
        target_col: str = "is_default",
        smoothing: int = 20,
        min_samples: int = 10,
        noise_level: float = 0.01,
    ):
        self.cols = cols
        self.target_col = target_col
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.noise_level = noise_level
        self.encoding_map: dict = {}
        self.global_mean: float = 0.0

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "RegularizedTargetEncoder":
        """Fit только на train данных."""
        self.global_mean = df[self.target_col].mean()

        for col in self.cols:
            stats = df.groupby(col)[self.target_col].agg(
                ["mean", "count"]
            )
            stats["encoded"] = (
                stats["mean"] * stats["count"]
                + self.global_mean * self.smoothing
            ) / (stats["count"] + self.smoothing)

            # категории с малым числом примеров → global_mean
            stats.loc[
                stats["count"] < self.min_samples, "encoded"
            ] = self.global_mean

            self.encoding_map[col] = stats["encoded"].to_dict()
            logger.info(
                f"Target encoding fitted for {col}: "
                f"{len(self.encoding_map[col])} categories"
            )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform на любом датасете."""
        result = df.copy()

        for col in self.cols:
            encoded_col = f"{col}_target_enc"
            result[encoded_col] = (
                result[col]
                .map(self.encoding_map[col])
                .fillna(self.global_mean)
            )

            # добавляем шум только на train (для regularization)
            if self.noise_level > 0:
                noise = np.random.normal(
                    0, self.noise_level, size=len(result)
                )
                result[encoded_col] += noise

        return result

    def fit_transform(
        self,
        engine: Engine,
        train_cutoff: str,
        execution_date: str,
    ) -> pd.DataFrame:
        """
        Fit на train, transform на всём датасете.
        Запись результатов в features схему.
        """
        train_df = pd.read_sql(
            f"""
            SELECT a.{', a.'.join(self.cols)},
                   a.is_default,
                   a.application_id
            FROM raw.applications a
            WHERE a.application_date <= '{train_cutoff}'
              AND a.is_default IS NOT NULL
            """,
            engine,
        )

        self.fit(train_df)

        all_df = pd.read_sql(
            f"""
            SELECT application_id,
                   {', '.join(self.cols)}
            FROM raw.applications
            WHERE application_date <= '{execution_date}'
            """,
            engine,
        )

        return self.transform(all_df)