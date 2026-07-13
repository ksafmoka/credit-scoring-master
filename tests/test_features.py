"""Feature engineering unit tests."""

import numpy as np
import pandas as pd
import pytest

from src.features.aggregations import AggregationFeatureComputer
from src.features.numerical import NumericalFeatureComputer
from src.features.target_encoding import RegularizedTargetEncoder


@pytest.fixture
def sample_applications():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "application_id": range(1, 101),
            "application_date": pd.date_range("2022-01-01", periods=100),
            "loan_amount": rng.uniform(5000, 50000, 100),
            "income": rng.uniform(30000, 200000, 100),
            "dti_ratio": rng.uniform(5, 45, 100),
            "credit_score": rng.integers(550, 800, 100),
            "total_credit_limit": rng.uniform(10000, 100000, 100),
            "employment_years": rng.uniform(0, 20, 100),
            "num_open_accounts": rng.integers(1, 15, 100),
            "num_delinquencies": rng.integers(0, 3, 100),
            "interest_rate": rng.uniform(5, 20, 100),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], 100),
            "purpose": rng.choice(
                ["debt_consolidation", "home_improvement", "other"], 100
            ),
            "is_default": rng.choice([0, 1], 100, p=[0.8, 0.2]),
        }
    )


def test_numerical_features_shape(sample_applications):
    computer = NumericalFeatureComputer()
    result = computer.compute(sample_applications)
    assert len(result) == len(sample_applications)
    assert "loan_to_income" in result.columns
    assert "credit_utilization" in result.columns
    assert "credit_score_norm" in result.columns


def test_loan_to_income_positive(sample_applications):
    result = NumericalFeatureComputer().compute(sample_applications)
    assert (result["loan_to_income"].dropna() >= 0).all()


def test_no_infinity_values(sample_applications):
    result = NumericalFeatureComputer().compute(sample_applications)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    assert not np.isinf(result[numeric_cols]).any().any()


def test_target_encoding_range(sample_applications):
    encoder = RegularizedTargetEncoder(
        cols=["home_ownership", "purpose"],
        smoothing=5,
        noise_level=0.0,
    )
    encoder.fit(sample_applications)
    result = encoder.transform(sample_applications, apply_noise=False)
    assert result["home_ownership_target_enc"].between(-0.01, 1.01).all()


def test_target_encoding_unseen_category(sample_applications):
    encoder = RegularizedTargetEncoder(cols=["home_ownership"], smoothing=5)
    encoder.fit(sample_applications)
    test_data = sample_applications.copy()
    test_data["home_ownership"] = "UNKNOWN_CATEGORY"
    result = encoder.transform(test_data, apply_noise=False)
    assert not result["home_ownership_target_enc"].isna().any()
    assert np.allclose(
        result["home_ownership_target_enc"], encoder.global_mean
    )


def test_target_encoding_no_noise_on_transform_by_default(sample_applications):
    encoder = RegularizedTargetEncoder(
        cols=["purpose"], smoothing=5, noise_level=0.05, random_seed=1
    )
    encoder.fit(sample_applications)
    a = encoder.transform(sample_applications, apply_noise=False)
    b = encoder.transform(sample_applications, apply_noise=False)
    pd.testing.assert_series_equal(
        a["purpose_target_enc"], b["purpose_target_enc"]
    )


def test_aggregation_excludes_future_payments():
    applications = pd.DataFrame(
        {
            "application_id": [1],
            "application_date": [pd.Timestamp("2023-01-15")],
        }
    )
    payments = pd.DataFrame(
        {
            "application_id": [1, 1],
            "payment_date": [
                pd.Timestamp("2023-01-10"),
                pd.Timestamp("2023-01-20"),
            ],
            "amount_due": [1000, 1000],
            "amount_paid": [1000, 0],
            "days_overdue": [0, 30],
        }
    )
    computer = AggregationFeatureComputer(windows=[90])
    result = computer.compute_from_frames(payments, applications)
    assert len(result) == 1
    # only past payment contributes → max overdue 0
    assert result.iloc[0]["max_days_overdue_90d"] == 0


def test_aggregation_canonical_columns(sample_applications):
    apps = sample_applications[["application_id", "application_date"]].copy()
    payments = pd.DataFrame(
        {
            "application_id": np.repeat(apps["application_id"].values, 3),
            "payment_date": np.tile(
                pd.date_range("2021-01-01", periods=3), len(apps)
            ),
            "amount_due": 100.0,
            "amount_paid": 90.0,
            "days_overdue": 1,
        }
    )
    # ensure payments before app dates
    payments["payment_date"] = apps["application_date"].repeat(3).values - pd.to_timedelta(
        np.tile([10, 20, 40], len(apps)), unit="D"
    )
    result = AggregationFeatureComputer(windows=[30, 90, 180]).compute_from_frames(
        payments, apps
    )
    for col in [
        "avg_days_overdue_30d",
        "avg_days_overdue_90d",
        "payment_consistency_90d",
        "pct_late_payments_90d",
    ]:
        assert col in result.columns
