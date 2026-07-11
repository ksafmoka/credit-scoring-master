"""Database read helpers."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def get_raw_applications(
    engine: Engine,
    date: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM raw.applications"
    params: dict = {}
    conditions: list[str] = []

    if date:
        conditions.append("application_date <= :date")
        params["date"] = date

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY application_date, application_id"

    if limit:
        query += " LIMIT :limit"
        params["limit"] = int(limit)

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def get_payment_history(
    engine: Engine,
    application_ids: list[int] | None = None,
) -> pd.DataFrame:
    if application_ids:
        query = text(
            "SELECT * FROM raw.payment_history "
            "WHERE application_id = ANY(:ids)"
        )
        with engine.connect() as conn:
            return pd.read_sql(
                query, conn, params={"ids": list(application_ids)}
            )

    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM raw.payment_history"), conn)


def get_feature_dataset(
    engine: Engine,
    train_end: str,
    val_end: str,
    feature_version: str = "v1.0",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = text(
        """
        SELECT
            f.*,
            a.is_default,
            a.application_date AS app_date
        FROM features.application_features f
        JOIN raw.applications a
            ON f.application_id = a.application_id
        WHERE f.feature_version = :feature_version
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"feature_version": feature_version})

    if df.empty:
        return df.copy(), df.copy(), df.copy()

    # Prefer application date from raw for split correctness
    if "app_date" in df.columns:
        df["application_date"] = pd.to_datetime(df["app_date"])
        df = df.drop(columns=["app_date"])
    else:
        df["application_date"] = pd.to_datetime(df["feature_date"])

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = df[df["application_date"] <= train_end_ts].copy()
    val = df[
        (df["application_date"] > train_end_ts)
        & (df["application_date"] <= val_end_ts)
    ].copy()
    test = df[df["application_date"] > val_end_ts].copy()

    return train, val, test
