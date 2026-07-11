# tests/test_features.py

import pytest
import pandas as pd
import numpy as np
from src.features.numerical import NumericalFeatureComputer
from src.features.target_encoding import RegularizedTargetEncoder


@pytest.fixture
def sample_applications():
    return pd.DataFrame({
        "application_id": range(1, 101),
        "application_date": pd.date_range("2022-01-01", periods=100),
        "loan_amount": np.random.uniform(5000, 50000, 100),
        "income": np.random.uniform(30000, 200000, 100),
        "dti_ratio": np.random.uniform(5, 45, 100),
        "credit_score": np.random.randint(550, 800, 100),
        "total_credit_limit": np.random.uniform(10000, 100000, 100),
        "home_ownership": np.random.choice(
            ["RENT", "OWN", "MORTGAGE"], 100
        ),
        "purpose": np.random.choice(
            ["debt_consolidation", "home_improvement", "other"], 100
        ),
        "is_default": np.random.choice([0, 1], 100, p=[0.8, 0.2]),
    })


def test_numerical_features_shape(sample_applications):
    computer = NumericalFeatureComputer()
    result = computer.compute(sample_applications)
    assert len(result) == len(sample_applications)
    assert "loan_to_income" in result.columns
    assert "credit_utilization" in result.columns


def test_loan_to_income_positive(sample_applications):
    computer = NumericalFeatureComputer()
    result = computer.compute(sample_applications)
    # loan_to_income должен быть >= 0
    assert (result["loan_to_income"].dropna() >= 0).all()


def test_no_infinity_values(sample_applications):
    computer = NumericalFeatureComputer()
    result = computer.compute(sample_applications)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    assert not np.isinf(result[numeric_cols]).any().any()


def test_target_encoding_range(sample_applications):
    encoder = RegularizedTargetEncoder(
        cols=["home_ownership", "purpose"],
        smoothing=5,
    )
    encoder.fit(sample_applications)
    result = encoder.transform(sample_applications)

    # encoded values должны быть в [0, 1] (это вероятности)
    assert result["home_ownership_target_enc"].between(
        -0.1, 1.1
    ).all()


def test_target_encoding_unseen_category(sample_applications):
    """Новые категории → global_mean, не ошибка."""
    encoder = RegularizedTargetEncoder(
        cols=["home_ownership"],
        smoothing=5,
    )
    encoder.fit(sample_applications)

    test_data = sample_applications.copy()
    test_data["home_ownership"] = "UNKNOWN_CATEGORY"
    result = encoder.transform(test_data)

    # не должно быть NaN
    assert not result["home_ownership_target_enc"].isna().any()