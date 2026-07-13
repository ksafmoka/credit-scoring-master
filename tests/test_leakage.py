"""Leakage-oriented tests."""

import numpy as np
import pandas as pd

from src.features.aggregations import AggregationFeatureComputer


def test_no_future_payments_in_aggregation():
    applications = pd.DataFrame(
        {
            "application_id": [1],
            "application_date": ["2023-01-15"],
        }
    )
    payments = pd.DataFrame(
        {
            "application_id": [1, 1],
            "payment_date": ["2023-01-10", "2023-01-20"],
            "amount_due": [1000, 1000],
            "amount_paid": [1000, 0],
            "days_overdue": [0, 30],
        }
    )
    applications["application_date"] = pd.to_datetime(
        applications["application_date"]
    )
    payments["payment_date"] = pd.to_datetime(payments["payment_date"])

    result = AggregationFeatureComputer(windows=[90]).compute_from_frames(
        payments, applications
    )
    assert len(result) == 1
    assert result.iloc[0]["max_days_overdue_90d"] == 0
    assert result.iloc[0]["avg_days_overdue_90d"] == 0


def test_train_test_no_temporal_overlap():
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame(
        {
            "application_date": dates,
            "target": np.random.randint(0, 2, len(dates)),
        }
    )
    train_end = "2022-12-31"
    val_end = "2023-06-30"
    train = df[df["application_date"] <= train_end]
    val = df[
        (df["application_date"] > train_end)
        & (df["application_date"] <= val_end)
    ]
    test = df[df["application_date"] > val_end]

    assert len(set(train.index) & set(val.index)) == 0
    assert len(set(val.index) & set(test.index)) == 0
    assert len(set(train.index) & set(test.index)) == 0
    assert train["application_date"].max() < val["application_date"].min()
    assert val["application_date"].max() < test["application_date"].min()


def test_preprocess_default_mapping_from_loan_status():
    from src.data.ingestion import _preprocess_raw_chunk

    raw = pd.DataFrame(
        {
            "loan_amnt": [10000, 20000],
            "annual_inc": [50000, 80000],
            "term": ["36 months", "60 months"],
            "int_rate": ["12.5%", "9%"],
            "emp_length": ["5 years", "10+ years"],
            "home_ownership": ["rent", "OWN"],
            "purpose": ["Debt_Consolidation", "other"],
            "dti": [15.0, 22.0],
            "fico_range_low": [690, 720],
            "fico_range_high": [694, 724],
            "open_acc": [5, 8],
            "delinq_2yrs": [0, 1],
            "revol_bal": [12000, 15000],
            "issue_d": ["Jan-2022", "Jun-2023"],
            "loan_status": ["Charged Off", "Fully Paid"],
            "member_id": [1, 2],
        }
    )
    out = _preprocess_raw_chunk(raw)
    assert list(out["is_default"]) == [True, False]
    assert out.iloc[0]["loan_term"] == 36
    assert out.iloc[0]["home_ownership"] == "RENT"
    assert out.iloc[0]["purpose"] == "debt_consolidation"
