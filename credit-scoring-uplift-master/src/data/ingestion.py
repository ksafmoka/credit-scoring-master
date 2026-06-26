# src/data/ingestion.py

import pandas as pd
from loguru import logger
from pathlib import Path
from sqlalchemy.engine import Engine


def load_lending_club_data(
    filepath: str | Path,
    engine: Engine,
    chunk_size: int = 10_000,
) -> None:
    """
    Загрузка сырых данных Lending Club в raw.applications.
    Работает чанками — не падает на больших файлах.
    """
    logger.info(f"Loading data from {filepath}")

    total_loaded = 0

    for chunk in pd.read_csv(
        filepath,
        chunksize=chunk_size,
        low_memory=False,
    ):
        processed = _preprocess_raw_chunk(chunk)

        processed.to_sql(
            "applications",
            engine,
            schema="raw",
            if_exists="append",
            index=False,
            method="multi",
        )

        total_loaded += len(processed)
        logger.info(f"Loaded {total_loaded} rows so far...")

    logger.info(f"Done. Total rows loaded: {total_loaded}")


def _preprocess_raw_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Минимальная предобработка перед загрузкой в raw."""

    # маппинг колонок Lending Club → наша схема
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

    existing_cols = [
        c for c in column_mapping if c in df.columns
    ]
    df = df[existing_cols].rename(
        columns={k: column_mapping[k] for k in existing_cols}
    )

    # преобразование target
    if "is_default" in df.columns:
        df["is_default"] = df["is_default"].isin(
            ["Charged Off", "Default", "Late (31-120 days)"]
        )

    # employment_years: "10+ years" → 10.0
    if "employment_years" in df.columns:
        df["employment_years"] = (
            df["employment_years"]
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # interest_rate: "13.5%" → 13.5
    if "interest_rate" in df.columns:
        df["interest_rate"] = (
            df["interest_rate"]
            .astype(str)
            .str.replace("%", "")
            .astype(float)
        )

    # loan_term: " 36 months" → 36
    if "loan_term" in df.columns:
        df["loan_term"] = (
            df["loan_term"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

    # application_date
    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(
            df["application_date"], format="%b-%Y", errors="coerce"
        ).dt.date

    # treatment_flag (симуляция: рандомно 50/50)
    import numpy as np
    df["treatment_flag"] = np.random.choice(
        [True, False], size=len(df)
    )

    df = df.dropna(subset=["application_date", "loan_amount"])

    return df


def generate_synthetic_payment_history(
    engine: Engine,
    n_payments_per_loan: int = 12,
    seed: int = 42,
) -> None:
    """
    Генерация синтетической истории платежей
    (если нет реальных данных).
    """
    import numpy as np

    np.random.seed(seed)

    apps = pd.read_sql(
        "SELECT application_id, loan_amount, loan_term, "
        "is_default, application_date FROM raw.applications",
        engine,
    )

    logger.info(
        f"Generating payment history for {len(apps)} applications"
    )

    payments = []
    for _, row in apps.iterrows():
        monthly_payment = row["loan_amount"] / max(
            row["loan_term"] or 36, 1
        )

        for month in range(min(n_payments_per_loan, row["loan_term"] or 36)):
            payment_date = pd.to_datetime(
                row["application_date"]
            ) + pd.DateOffset(months=month + 1)

            # дефолтники чаще просрочивают
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

            paid_ratio = max(
                0, 1 - days_overdue / 100 * np.random.random()
            )

            payments.append({
                "application_id": row["application_id"],
                "payment_date": payment_date.date(),
                "amount_due": round(monthly_payment, 2),
                "amount_paid": round(
                    monthly_payment * paid_ratio, 2
                ),
                "days_overdue": days_overdue,
            })

    payments_df = pd.DataFrame(payments)
    payments_df.to_sql(
        "payment_history",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=10_000,
    )

    logger.info(
        f"Generated {len(payments_df)} payment records"
    )