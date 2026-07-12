"""Regularized target encoding with train-only noise and serializable maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)
from sqlalchemy import text
from sqlalchemy.engine import Engine


class RegularizedTargetEncoder:
    """
    target_enc = (category_mean * n + global_mean * smoothing) / (n + smoothing)

    Noise is applied only when apply_noise=True (training fit path).
    Encoding maps are serializable for online serving parity.
    """

    def __init__(
        self,
        cols: list[str],
        target_col: str = "is_default",
        smoothing: int = 20,
        min_samples: int = 10,
        noise_level: float = 0.01,
        random_seed: int = 42,
    ):
        self.cols = cols
        self.target_col = target_col
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.noise_level = noise_level
        self.random_seed = random_seed
        self.encoding_map: dict[str, dict] = {}
        self.global_mean: float = 0.0

    def fit(self, df: pd.DataFrame) -> "RegularizedTargetEncoder":
        self.global_mean = float(df[self.target_col].astype(float).mean())

        for col in self.cols:
            stats = (
                df.groupby(col)[self.target_col]
                .agg(["mean", "count"])
                .astype({"mean": float, "count": float})
            )
            stats["encoded"] = (
                stats["mean"] * stats["count"]
                + self.global_mean * self.smoothing
            ) / (stats["count"] + self.smoothing)
            stats.loc[stats["count"] < self.min_samples, "encoded"] = (
                self.global_mean
            )
            self.encoding_map[col] = {
                str(k): float(v) for k, v in stats["encoded"].to_dict().items()
            }
            logger.info(
                f"Target encoding fitted for {col}: "
                f"{len(self.encoding_map[col])} categories"
            )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        apply_noise: bool = False,
    ) -> pd.DataFrame:
        result = df.copy()
        rng = np.random.default_rng(self.random_seed)

        for col in self.cols:
            encoded_col = f"{col}_target_enc"
            mapping = self.encoding_map.get(col, {})
            result[encoded_col] = (
                result[col].astype(str).map(mapping).fillna(self.global_mean)
            )
            if apply_noise and self.noise_level > 0:
                noise = rng.normal(0.0, self.noise_level, size=len(result))
                result[encoded_col] = result[encoded_col] + noise

        return result

    def fit_transform(
        self,
        engine: Engine,
        train_cutoff: str,
        execution_date: str | None = None,
    ) -> pd.DataFrame:
        cols_sql = ", ".join(f"a.{c}" for c in self.cols)
        with engine.connect() as conn:
            train_df = pd.read_sql(
                text(
                    f"""
                    SELECT {cols_sql},
                           a.is_default,
                           a.application_id
                    FROM raw.applications a
                    WHERE a.application_date <= :train_cutoff
                      AND a.is_default IS NOT NULL
                    """
                ),
                conn,
                params={"train_cutoff": train_cutoff},
            )

        if train_df.empty:
            # LC dates may all be before/after configured cutoff — fit on all labeled rows
            logger.warning(
                f"Empty TE train for cutoff={train_cutoff}; fitting on ALL labeled apps"
            )
            with engine.connect() as conn:
                train_df = pd.read_sql(
                    text(
                        f"""
                        SELECT {cols_sql},
                               a.is_default,
                               a.application_id
                        FROM raw.applications a
                        WHERE a.is_default IS NOT NULL
                        """
                    ),
                    conn,
                )

        if train_df.empty:
            logger.warning("Empty train set for target encoding; using prior 0.15")
            self.global_mean = 0.15
            self.encoding_map = {c: {} for c in self.cols}
        else:
            train_df[self.target_col] = train_df[self.target_col].astype(float)
            self.fit(train_df)

        # Transform full population (no Airflow-ds filter)
        with engine.connect() as conn:
            if execution_date:
                all_df = pd.read_sql(
                    text(
                        f"""
                        SELECT application_id, {', '.join(self.cols)}
                        FROM raw.applications
                        WHERE application_date <= :execution_date
                        """
                    ),
                    conn,
                    params={"execution_date": execution_date},
                )
            else:
                all_df = pd.read_sql(
                    text(
                        f"""
                        SELECT application_id, {', '.join(self.cols)}
                        FROM raw.applications
                        """
                    ),
                    conn,
                )

        return self.transform(all_df, apply_noise=False)

    def to_dict(self) -> dict:
        return {
            "cols": self.cols,
            "target_col": self.target_col,
            "smoothing": self.smoothing,
            "min_samples": self.min_samples,
            "noise_level": self.noise_level,
            "global_mean": self.global_mean,
            "encoding_map": self.encoding_map,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RegularizedTargetEncoder":
        enc = cls(
            cols=payload["cols"],
            target_col=payload.get("target_col", "is_default"),
            smoothing=int(payload.get("smoothing", 20)),
            min_samples=int(payload.get("min_samples", 10)),
            noise_level=float(payload.get("noise_level", 0.0)),
        )
        enc.global_mean = float(payload.get("global_mean", 0.15))
        enc.encoding_map = payload.get("encoding_map", {})
        return enc

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RegularizedTargetEncoder":
        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)
