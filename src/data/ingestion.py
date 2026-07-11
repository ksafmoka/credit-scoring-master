# src/data/ingestion.py

import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from sqlalchemy.engine import Engine
from sqlalchemy import text


def _preprocess_raw_chunk(df: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {
        "id": "client_id",
        "issue_d": "application_date",
        "loan_amnt": "loan_amount",
        "term": "loan_term",
        "int_rate": "interest_rate",
        "annual_inc": "income",
        "emp_length": "employment_years",
        "home_ownership": "home_ownership",
        "purpose": "purpose",
        "dti": "dti_ratio",
        "fico_range_low": "credit_score",
        "open_acc": "num_open_accounts",
        "delinq_2yrs": "num_delinquencies",
        "total_rev_hi_lim": "total_credit_limit",
        "loan_status": "is_default",
    }

    existing = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df[list(existing.keys())].rename(columns=existing)

    if "is_default" in df.columns:
        df["is_default"] = df["is_default"].isin(
            ["Charged Off", "Default", "Late (31-120 days)"]
        )

    if "employment_years" in df.columns:
        df["employment_years"] = (
            df["employment_years"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    if "interest_rate" in df.columns:
        df["interest_rate"] = (
            df["interest_rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .replace("", float("nan"))
            .astype(float)
        )

    if "loan_term" in df.columns:
        df["loan_term"] = (
            df["loan_term"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(
            df["application_date"], format="%b-%Y", errors="coerce"
        ).dt.date

    df["treatment_flag"] = np.random.choice([True, False], size=len(df))
    df = df.dropna(subset=["application_date", "loan_amount"])

    # Клиппинг всех числовых полей до безопасных пределов
    numeric_limits = {
        "loan_amount": (0, 50_000_000),
        "annual_inc": (0, 100_000_000),
        "interest_rate": (0, 100),
        "dti_ratio": (0, 200),
        "total_credit_limit": (0, 100_000_000),
        "installment": (0, 10_000_000),
    }
    
    for col, (low, high) in numeric_limits.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(lower=low, upper=high)
    
    # Удалить бесконечности
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = np.where(np.isfinite(df[col]), df[col], None)
    
    return df


def _insert_dataframe(df: pd.DataFrame, table: str, schema: str, engine: Engine) -> None:
    if df.empty:
        return

    # Принудительно приводим все NaN к None
    df = df.where(pd.notna(df), None)

    columns = ", ".join(df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))
    query = f"INSERT INTO {schema}.{table} ({columns}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    with engine.begin() as conn:
        raw_conn = conn.connection
        cursor = raw_conn.cursor()

        # Преобразуем каждую строку в кортеж, заменяя numpy-типы на нативные
        def convert(row):
            return tuple(
                None if v is None else (
                    int(v) if isinstance(v, np.integer) else
                    float(v) if isinstance(v, np.floating) else
                    bool(v) if isinstance(v, np.bool_) else
                    v
                )
                for v in row
            )

        rows = [convert(row) for row in df.itertuples(index=False, name=None)]
        cursor.executemany(query, rows)
        cursor.close()


def load_lending_club_data(
    filepath: str | Path,
    engine: Engine,
    chunk_size: int = 10_000,
) -> None:
    logger.info(f"Loading data from {filepath}")
    total_loaded = 0

    for chunk in pd.read_csv(
        filepath,
        chunksize=chunk_size,
        low_memory=False,
    ):
        processed = _preprocess_raw_chunk(chunk)

        if processed.empty:
            continue

        _insert_dataframe(processed, "applications", "raw", engine)
        total_loaded += len(processed)
        logger.info(f"Loaded {total_loaded} rows so far...")

    logger.info(f"Done. Total rows loaded: {total_loaded}")


def generate_synthetic_payment_history(
    engine: Engine,
    n_payments_per_loan: int = 12,
    seed: int = 42,
) -> None:
    np.random.seed(seed)

    with engine.connect() as conn:
        apps = pd.read_sql(
            text(
                "SELECT application_id, loan_amount, loan_term, "
                "is_default, application_date FROM raw.applications"
            ),
            conn,
        )

    logger.info(f"Generating payment history for {len(apps)} applications")

    payments = []
    for _, row in apps.iterrows():
        monthly_payment = row["loan_amount"] / max(row["loan_term"] or 36, 1)

        for month in range(min(n_payments_per_loan, int(row["loan_term"] or 36))):
            payment_date = pd.to_datetime(
                row["application_date"]
            ) + pd.DateOffset(months=month + 1)

            if row["is_default"] and month > 6:
                days_overdue = int(np.random.choice(
                    [0, 0, 15, 30, 60, 90],
                    p=[0.3, 0.1, 0.2, 0.2, 0.1, 0.1],
                ))
            else:
                days_overdue = int(np.random.choice(
                    [0, 0, 0, 5, 15],
                    p=[0.7, 0.1, 0.1, 0.05, 0.05],
                ))

            paid_ratio = max(0, 1 - days_overdue / 100 * np.random.random())

            payments.append({
                "application_id": int(row["application_id"]),
                "payment_date": payment_date.date(),
                "amount_due": round(float(monthly_payment), 2),
                "amount_paid": round(float(monthly_payment * paid_ratio), 2),
                "days_overdue": days_overdue,
            })

    payments_df = pd.DataFrame(payments)
    _insert_dataframe(payments_df, "payment_history", "raw", engine)
    logger.info(f"Generated {len(payments_df)} payment records")