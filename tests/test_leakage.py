# tests/test_leakage.py

import pytest
import pandas as pd
import numpy as np
from src.features.aggregations import AggregationFeatureComputer


def test_no_future_payments_in_aggregation():
    """
    Агрегаты должны включать ТОЛЬКО платежи
    до даты заявки, не после.
    """
    # заявка сделана 2023-01-15
    # платёж от 2023-01-20 (ПОСЛЕ заявки) не должен войти
    applications = pd.DataFrame({
        "application_id": [1],
        "application_date": ["2023-01-15"],
    })

    # есть платёж ДО и ПОСЛЕ заявки
    payments = pd.DataFrame({
        "application_id": [1, 1],
        "payment_date": ["2023-01-10", "2023-01-20"],
        "amount_due": [1000, 1000],
        "amount_paid": [1000, 0],
        "days_overdue": [0, 30],
    })

    applications["application_date"] = pd.to_datetime(
        applications["application_date"]
    )
    payments["payment_date"] = pd.to_datetime(
        payments["payment_date"]
    )

    df = payments.merge(applications, on="application_id")
    computer = AggregationFeatureComputer(windows=[90])

    df_window = df[
        (df["payment_date"] >= df["application_date"]
         - pd.Timedelta(days=90))
        & (df["payment_date"] < df["application_date"])
    ]

    # должен войти только платёж от 2023-01-10
    assert len(df_window) == 1
    assert df_window.iloc[0]["days_overdue"] == 0


def test_train_test_no_temporal_overlap():
    """Train должен быть строго ДО test."""
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({
        "application_date": dates,
        "target": np.random.randint(0, 2, len(dates)),
    })

    train_end = "2022-12-31"
    val_end = "2023-06-30"

    train = df[df["application_date"] <= train_end]
    val = df[
        (df["application_date"] > train_end)
        & (df["application_date"] <= val_end)
    ]
    test = df[df["application_date"] > val_end]

    # нет пересечений
    assert len(set(train.index) & set(val.index)) == 0
    assert len(set(val.index) & set(test.index)) == 0
    assert len(set(train.index) & set(test.index)) == 0

    # правильный порядок дат
    assert train["application_date"].max() < val["application_date"].min()
    assert val["application_date"].max() < test["application_date"].min()