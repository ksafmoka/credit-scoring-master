# src/data/queries.py

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text


def get_raw_applications(
    engine: Engine,
    date: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM raw.applications"
    conditions = []

    if date:
        conditions.append(f"application_date <= '{date}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    if limit:
        query += f" LIMIT {limit}"

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def get_payment_history(
    engine: Engine,
    application_ids: list[int] | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM raw.payment_history"
    if application_ids:
        ids_str = ", ".join(map(str, application_ids))
        query += f" WHERE application_id IN ({ids_str})"

    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def get_feature_dataset(
    engine: Engine,
    train_end: str,
    val_end: str,
    feature_version: str = "v1.0",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = f"""
        SELECT
            f.*,
            a.is_default,
            a.treatment_flag,
            a.application_date
        FROM features.application_features f
        JOIN raw.applications a
            ON f.application_id = a.application_id
        WHERE f.feature_version = '{feature_version}'
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    df["application_date"] = pd.to_datetime(df["application_date"])

    train = df[df["application_date"] <= train_end]
    val = df[
        (df["application_date"] > train_end)
        & (df["application_date"] <= val_end)
    ]
    test = df[df["application_date"] > val_end]

    return train, val, test