"""Picklable wrapper around StackingEnsemble for serving / MLflow."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.scoring.ensemble import StackingEnsemble


class EnsembleWrapper:
    """
    Sklearn-like interface so pickle / MLflow / FastAPI can load the model.

    Must live at module top-level (not inside a function) for pickle.
    """

    def __init__(self, ensemble: StackingEnsemble):
        self.ensemble = ensemble

    def predict_proba(self, X: Any) -> np.ndarray:
        if isinstance(X, np.ndarray):
            cols = self.ensemble.feature_names
            X = pd.DataFrame(X, columns=cols)
        p1 = self.ensemble.predict_proba(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
