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

    def fit_from_engine(self, engine: Engine, train_cutoff: str) -> None:
        """Fit encoder from DB using train cutoff, with fallback to all labeled."""
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

    def fit_transform(
        self,
        engine: Engine,
        train_cutoff: str,
        execution_date: str | None = None,
        batch_size: int | None = None,
        max_apps: int | None = None,
        sample_strategy: str | None = None,
    ) -> pd.DataFrame:
        """Batched version: fits then transforms in chunks to avoid 2.2M RAM spike."""
        from src.config import FeatureEngineeringConfig, IngestionConfig

        batch_size = batch_size or FeatureEngineeringConfig.TARGET_ENCODING_BATCH_SIZE
        if max_apps is None:
            max_apps = FeatureEngineeringConfig.MAX_APPS
        sample_strategy = (sample_strategy or IngestionConfig.SAMPLE_STRATEGY).lower()

        self.fit_from_engine(engine, train_cutoff)

        # Count total
        count_q = "SELECT COUNT(*) FROM raw.applications"
        count_params: dict = {}
        if execution_date:
            count_q += " WHERE application_date <= :execution_date"
            count_params["execution_date"] = execution_date
        with engine.connect() as conn:
            total = int(conn.execute(text(count_q), count_params or None).scalar() or 0)
        if max_apps is not None:
            total = min(total, int(max_apps))
        if total == 0:
            return pd.DataFrame()

        logger.info(
            f"Target encoding transform batched: total={total}, batch_size={batch_size}, strategy={sample_strategy}"
        )

        # Random pre-selection for variant A (uniform coverage)
        random_ids = None
        if sample_strategy == "random" and max_apps is not None:
            with engine.connect() as conn:
                id_df = pd.read_sql(
                    text(
                        f"SELECT application_id FROM raw.applications "
                        f"{'WHERE ' + count_q.split('WHERE',1)[1] if 'WHERE' in count_q else ''} "
                        f"ORDER BY random() LIMIT :limit"
                    ),
                    conn,
                    params={**count_params, "limit": int(max_apps)},
                )
                random_ids = id_df["application_id"].tolist()
                total = len(random_ids)

        results = []
        for offset in range(0, total, batch_size):
            limit = min(batch_size, total - offset)

            if random_ids is not None:
                batch_ids = random_ids[offset : offset + limit]
                if not batch_ids:
                    continue
                q = f"""
                    SELECT application_id, {', '.join(self.cols)}
                    FROM raw.applications
                    WHERE application_id = ANY(:ids)
                """
                params = {"ids": batch_ids}
            else:
                order = "application_date DESC, application_id DESC" if max_apps else "application_id"
                q = f"""
                    SELECT application_id, {', '.join(self.cols)}
                    FROM raw.applications
                    {f'WHERE application_date <= :execution_date' if execution_date else ''}
                    ORDER BY {order}
                    LIMIT :limit OFFSET :offset
                """
                params = dict(count_params)
                params["limit"] = limit
                params["offset"] = offset

            with engine.connect() as conn:
                batch_df = pd.read_sql(text(q), conn, params=params)

            if batch_df.empty:
                continue
            transformed = self.transform(batch_df, apply_noise=False)
            results.append(transformed)
            if (offset // batch_size) % 5 == 0:
                logger.info(
                    f"Target encoding batch {offset // batch_size + 1}: {len(transformed)} rows"
                )

        if not results:
            return pd.DataFrame()
        final = pd.concat(results, ignore_index=True)
        return final

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
