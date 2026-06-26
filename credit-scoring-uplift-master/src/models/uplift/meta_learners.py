# src/models/uplift/meta_learners.py

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from loguru import logger


class TLearner:
    """
    T-Learner: отдельные модели для treatment и control групп.
    CATE = E[Y|X, T=1] - E[Y|X, T=0]
    """

    def __init__(self, base_model_params: dict | None = None):
        params = base_model_params or {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": 0,
            "random_seed": 42,
        }
        self.model_treatment = CatBoostClassifier(**params)
        self.model_control = CatBoostClassifier(**params)
        self.feature_names = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
    ) -> "TLearner":
        self.feature_names = X.columns.tolist()

        mask_t = treatment == 1
        mask_c = treatment == 0

        logger.info(
            f"T-Learner: treatment={mask_t.sum()}, "
            f"control={mask_c.sum()}"
        )

        self.model_treatment.fit(X[mask_t], y[mask_t])
        self.model_control.fit(X[mask_c], y[mask_c])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        p_treatment = self.model_treatment.predict_proba(X)[:, 1]
        p_control = self.model_control.predict_proba(X)[:, 1]
        return p_treatment - p_control

    def segment(self, cate: np.ndarray) -> np.ndarray:
        return np.where(
            cate > 0.1, "persuadables",
            np.where(
                cate < -0.05, "sleeping_dogs",
                np.where(cate >= 0.0, "sure_things", "lost_causes"),
            ),
        )


class XLearner:
    """
    X-Learner: лучше T-Learner при несбалансированном
    treatment/control.
    """

    def __init__(self, base_model_params: dict | None = None):
        params = base_model_params or {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": 0,
        }
        self.mu0 = CatBoostClassifier(**params)
        self.mu1 = CatBoostClassifier(**params)
        self.tau0 = CatBoostRegressor(
            iterations=200, verbose=0
        )
        self.tau1 = CatBoostRegressor(
            iterations=200, verbose=0
        )
        self.propensity = LogisticRegression(max_iter=500)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
    ) -> "XLearner":
        mask_t = treatment == 1
        mask_c = treatment == 0

        X_t = X[mask_t].reset_index(drop=True)
        y_t = y[mask_t].reset_index(drop=True)
        X_c = X[mask_c].reset_index(drop=True)
        y_c = y[mask_c].reset_index(drop=True)

        # Stage 1: outcome models
        self.mu0.fit(X_c, y_c)
        self.mu1.fit(X_t, y_t)

        # Stage 2: imputed effects
        tau_c = self.mu1.predict_proba(X_c)[:, 1] - y_c.values
        tau_t = y_t.values - self.mu0.predict_proba(X_t)[:, 1]

        # Stage 3: CATE models
        self.tau0.fit(X_c, tau_c)
        self.tau1.fit(X_t, tau_t)

        # Propensity score
        self.propensity.fit(X, treatment)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        g = self.propensity.predict_proba(X)[:, 1]
        tau0 = self.tau0.predict(X)
        tau1 = self.tau1.predict(X)
        return g * tau0 + (1 - g) * tau1


class SLearner:
    """
    S-Learner: одна модель, treatment как фича.
    Самый простой из meta-learners.
    CATE = M(X, T=1) - M(X, T=0)
    """

    def __init__(self, base_model_params: dict | None = None):
        params = base_model_params or {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": 0,
            "random_seed": 42,
        }
        self.model = CatBoostClassifier(**params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
    ) -> "SLearner":
        X_with_t = X.copy()
        X_with_t["treatment"] = treatment.values
        self.model.fit(X_with_t, y)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        X_t1 = X.copy()
        X_t1["treatment"] = 1

        X_t0 = X.copy()
        X_t0["treatment"] = 0

        p1 = self.model.predict_proba(X_t1)[:, 1]
        p0 = self.model.predict_proba(X_t0)[:, 1]
        return p1 - p0