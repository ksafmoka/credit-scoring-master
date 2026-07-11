"""Validation unit tests."""

import numpy as np
import pandas as pd

from src.data.validation import validate_feature_table, validate_raw_applications


def test_valid_data_passes():
    df = pd.DataFrame(
        {
            "client_id": range(100),
            "application_date": pd.date_range("2022-01-01", periods=100),
            "loan_amount": np.random.uniform(5000, 50000, 100),
            "income": np.random.uniform(30000, 200000, 100),
            "is_default": np.random.choice(
                [True, False], 100, p=[0.2, 0.8]
            ),
        }
    )
    report = validate_raw_applications(df)
    assert report.success
    assert len(report.errors) == 0


def test_missing_columns_fails():
    df = pd.DataFrame({"some_col": [1, 2, 3]})
    report = validate_raw_applications(df)
    assert not report.success
    assert any("Missing" in e for e in report.errors)


def test_negative_loan_amount_fails():
    df = pd.DataFrame(
        {
            "client_id": range(10),
            "application_date": pd.date_range("2022-01-01", periods=10),
            "loan_amount": [1000] * 9 + [-5],
            "income": [50000] * 10,
            "is_default": [False] * 10,
        }
    )
    report = validate_raw_applications(df)
    assert not report.success


def test_high_null_income_fails():
    df = pd.DataFrame(
        {
            "client_id": range(100),
            "application_date": pd.date_range("2022-01-01", periods=100),
            "loan_amount": np.random.uniform(5000, 50000, 100),
            "income": [np.nan] * 51 + list(np.random.uniform(30000, 100000, 49)),
            "is_default": [False] * 100,
        }
    )
    report = validate_raw_applications(df)
    assert not report.success


def test_feature_validation_no_nulls():
    df = pd.DataFrame({"loan_to_income": [0.5, np.nan, 0.3]})
    report = validate_feature_table(
        df, checks={"no_nulls_in_critical": ["loan_to_income"]}
    )
    assert not report.success


def test_feature_validation_row_count():
    df = pd.DataFrame({"x": range(5)})
    report = validate_feature_table(df, checks={"row_count_min": 100})
    assert not report.success
