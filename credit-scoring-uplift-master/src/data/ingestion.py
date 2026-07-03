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

    # SQLAlchemy 2.x: используем connection, не engine напрямую
    with engine.begin() as conn:
        for chunk in pd.read_csv(
            filepath,
            chunksize=chunk_size,
            low_memory=False,
        ):
            processed = _preprocess_raw_chunk(chunk)

            processed.to_sql(
                "applications",
                conn,              # <-- connection, не engine
                schema="raw",
                if_exists="append",
                index=False,
                method="multi",
            )

            total_loaded += len(processed)
            logger.info(f"Loaded {total_loaded} rows so far...")

    logger.info(f"Done. Total rows loaded: {total_loaded}")


def generate_synthetic_payment_history(
    engine: Engine,
    n_payments_per_loan: int = 12,
    seed: int = 42,
) -> None:
    import numpy as np

    np.random.seed(seed)

    # SQLAlchemy 2.x: используем connection для чтения
    with engine.connect() as conn:
        apps = pd.read_sql(
            "SELECT application_id, loan_amount, loan_term, "
            "is_default, application_date FROM raw.applications",
            conn,
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
                "amount_paid": round(monthly_payment * paid_ratio, 2),
                "days_overdue": days_overdue,
            })

    payments_df = pd.DataFrame(payments)

    # SQLAlchemy 2.x: используем connection для записи
    with engine.begin() as conn:
        payments_df.to_sql(
            "payment_history",
            conn,              # <-- connection, не engine
            schema="raw",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=10_000,
        )

    logger.info(f"Generated {len(payments_df)} payment records")