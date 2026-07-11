"""Model packaging for MLflow / local serving (feature parity)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, FeatureConfig
from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ScoringArtifact:
    model: Any
    feature_names: list[str]
    feature_medians: dict[str, float]
    target_encoding: dict = field(default_factory=dict)
    model_type: str = "unknown"
    metrics: dict = field(default_factory=dict)
    global_default_rate: float = 0.15

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame = self.prepare_features(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(frame)
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        return np.asarray(self.model.predict_proba(frame), dtype=float)

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for col in self.feature_names:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[self.feature_names].apply(pd.to_numeric, errors="coerce")
        if self.feature_medians:
            frame = frame.fillna(self.feature_medians)
        return frame.fillna(0.0)

    def save(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory or ARTIFACTS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.pkl"
        meta_path = directory / "artifact_meta.json"
        te_path = directory / "target_encoding.json"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        meta = {
            "feature_names": self.feature_names,
            "feature_medians": self.feature_medians,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "global_default_rate": self.global_default_rate,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        te_path.write_text(json.dumps(self.target_encoding, indent=2))
        logger.info(f"Saved scoring artifact to {directory}")
        return directory

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "ScoringArtifact":
        directory = Path(directory or ARTIFACTS_DIR)
        model_path = directory / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Run notebook 02 or: make local-pipeline"
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        meta_path = directory / "artifact_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        te_path = directory / "target_encoding.json"
        te = json.loads(te_path.read_text()) if te_path.exists() else {}
        return cls(
            model=model,
            feature_names=meta.get(
                "feature_names", list(FeatureConfig.ALL_FEATURES)
            ),
            feature_medians=meta.get("feature_medians", {}),
            target_encoding=te,
            model_type=meta.get("model_type", "unknown"),
            metrics=meta.get("metrics", {}),
            global_default_rate=float(meta.get("global_default_rate", 0.15)),
        )


def default_fill_values() -> dict[str, float]:
    return {feat: 0.0 for feat in FeatureConfig.ALL_FEATURES}
