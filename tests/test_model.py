"""Model training / metrics unit tests."""

import numpy as np
import pandas as pd
import pytest

from src.models.scoring.evaluate import compute_metrics, get_risk_bucket
from src.models.scoring.train import get_model


@pytest.fixture
def toy_dataset():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame(
        {
            "loan_to_income": np.random.uniform(0.1, 2.0, n),
            "credit_utilization": np.random.uniform(0.0, 1.0, n),
            "income_log": np.random.uniform(10, 13, n),
            "loan_amount_log": np.random.uniform(8, 11, n),
            "dti_ratio_clipped": np.random.uniform(5, 40, n),
            "employment_years": np.random.uniform(0, 20, n),
            "credit_score_norm": np.random.uniform(0.3, 0.9, n),
            "num_open_accounts": np.random.randint(1, 15, n),
            "num_delinquencies": np.random.randint(0, 3, n),
            "interest_rate": np.random.uniform(5, 20, n),
            "purpose_target_enc": np.random.uniform(0.1, 0.3, n),
            "home_ownership_target_enc": np.random.uniform(0.1, 0.3, n),
            "avg_days_overdue_30d": np.random.uniform(0, 10, n),
            "avg_days_overdue_90d": np.random.uniform(0, 10, n),
            "avg_days_overdue_180d": np.random.uniform(0, 10, n),
            "max_days_overdue_90d": np.random.uniform(0, 30, n),
            "pct_late_payments_90d": np.random.uniform(0, 0.5, n),
            "total_paid_90d": np.random.uniform(0, 5000, n),
            "payment_consistency_90d": np.random.uniform(0.5, 1.0, n),
            "bureau_balance_to_income": np.random.uniform(0, 1, n),
            "inquiries_per_account": np.random.uniform(0, 2, n),
            "loan_amount_x_dti": np.random.uniform(0, 500000, n),
            "income_x_credit_score": np.random.uniform(0, 1e8, n),
        }
    )
    y = pd.Series(np.random.choice([0, 1], n, p=[0.8, 0.2]))
    return X, y


@pytest.mark.parametrize("model_type", ["catboost", "lightgbm", "xgboost"])
def test_model_trains_and_predicts(toy_dataset, model_type):
    X, y = toy_dataset
    params = (
        {"iterations": 20}
        if model_type == "catboost"
        else {"n_estimators": 20}
    )
    model = get_model(model_type, params)
    if model_type == "catboost":
        model.fit(X, y, verbose=False)
    else:
        model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert not np.isnan(proba).any()


def test_compute_metrics(toy_dataset):
    _, y = toy_dataset
    fake_preds = np.random.uniform(0, 1, len(y))
    metrics = compute_metrics(y, fake_preds, prefix="test")
    assert "test_auc_roc" in metrics
    assert "test_gini" in metrics
    assert "test_ks_statistic" in metrics
    assert 0 <= metrics["test_auc_roc"] <= 1


def test_risk_bucket_boundaries():
    assert get_risk_bucket(0.01) == "LOW"
    assert get_risk_bucket(0.10) == "MEDIUM"
    assert get_risk_bucket(0.20) == "HIGH"
    assert get_risk_bucket(0.50) == "VERY_HIGH"


def test_scoring_artifact_roundtrip(tmp_path, toy_dataset):
    from src.models.scoring.artifacts import ScoringArtifact

    X, y = toy_dataset
    model = get_model("lightgbm", {"n_estimators": 30})
    model.fit(X, y)
    art = ScoringArtifact(
        model=model,
        feature_names=list(X.columns),
        feature_medians=X.median().to_dict(),
        target_encoding={"global_mean": 0.2, "encoding_map": {}},
        model_type="lightgbm",
        metrics={"val_auc_roc": 0.8},
    )
    art.save(tmp_path)
    loaded = ScoringArtifact.load(tmp_path)
    p1 = art.predict_proba(X.head(5))
    p2 = loaded.predict_proba(X.head(5))
    np.testing.assert_allclose(p1, p2, rtol=1e-5)
