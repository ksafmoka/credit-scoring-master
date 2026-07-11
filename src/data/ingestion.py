"""Raw data ingestion: Lending Club CSV → PostgreSQL + synthetic histories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Lending Club / common aliases → internal schema columns
COLUMN_MAP: dict[str, str] = {
    "id": "external_id",
    "member_id": "client_id",
    "loan_amnt": "loan_amount",
    "loan_amount": "loan_amount",
    "funded_amnt": "loan_amount",
    "term": "loan_term",
    "loan_term": "loan_term",
    "int_rate": "interest_rate",
    "interest_rate": "interest_rate",
    "annual_inc": "income",
    "income": "income",
    "emp_length": "employment_years",
    "employment_years": "employment_years",
    "home_ownership": "home_ownership",
    "purpose": "purpose",
    "dti": "dti_ratio",
    "dti_ratio": "dti_ratio",
    "fico_range_low": "credit_score",
    "fico_range_high": "credit_score_high",
    "credit_score": "credit_score",
    "open_acc": "num_open_accounts",
    "num_open_accounts": "num_open_accounts",
    "delinq_2yrs": "num_delinquencies",
    "num_delinquencies": "num_delinquencies",
    "revol_bal": "total_credit_limit",
    "total_acc": "total_accounts",
    "total_credit_limit": "total_credit_limit",
    "issue_d": "application_date",
    "application_date": "application_date",
    "loan_status": "loan_status",
    "is_default": "is_default",
    "client_id": "client_id",
    "inq_last_6mths": "num_inquiries_6m",
    "num_inquiries_6m": "num_inquiries_6m",
}

DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
}

RAW_APPLICATION_COLUMNS = [
    "client_id",
    "application_date",
    "loan_amount",
    "loan_term",
    "interest_rate",
    "income",
    "employment_years",
    "home_ownership",
    "purpose",
    "dti_ratio",
    "credit_score",
    "num_open_accounts",
    "num_delinquencies",
    "total_credit_limit",
    "is_default",
    "data_source",
]


def _parse_term(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    text_val = str(value).lower().replace("months", "").strip()
    try:
        return int(float(text_val))
    except ValueError:
        return None


def _parse_emp_length(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().lower()
    if s in {"n/a", "na", "", "null"}:
        return None
    if s.startswith("<"):
        return 0.5
    if "10" in s and "+" in s:
        return 10.0
    digits = "".join(ch for ch in s if ch.isdigit() or ch == ".")
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def _parse_interest_rate(value) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(series: pd.Series) -> pd.Series:
    # Lending Club often uses mon-YYYY (e.g. Jan-2022)
    parsed = pd.to_datetime(series, errors="coerce", format="%b-%Y")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def _default_flag(loan_status: pd.Series | None, is_default: pd.Series | None) -> pd.Series:
    if is_default is not None and is_default.notna().any():
        return is_default.fillna(False).astype(bool)
    if loan_status is None:
        return pd.Series(False, index=range(0))
    return loan_status.astype(str).isin(DEFAULT_STATUSES)


def _preprocess_raw_chunk(chunk: pd.DataFrame, data_source: str = "lending_club") -> pd.DataFrame:
    """
    Map external / Lending Club columns to raw.applications schema.
    Safe for both real Lending Club dumps and synthetic sample CSVs.
    """
    df = chunk.copy()
    df.columns = [str(c).strip() for c in df.columns]

    renamed: dict[str, str] = {}
    for col in df.columns:
        key = col.lower()
        if key in COLUMN_MAP:
            renamed[col] = COLUMN_MAP[key]
        elif col in COLUMN_MAP.values():
            renamed[col] = col
    df = df.rename(columns=renamed)

    # Prefer mid-FICO when high bound present
    if "credit_score_high" in df.columns and "credit_score" in df.columns:
        df["credit_score"] = (
            pd.to_numeric(df["credit_score"], errors="coerce")
            + pd.to_numeric(df["credit_score_high"], errors="coerce")
        ) / 2.0

    loan_status = df["loan_status"] if "loan_status" in df.columns else None
    is_default_col = df["is_default"] if "is_default" in df.columns else None
    default_series = _default_flag(loan_status, is_default_col)

    n = len(df)
    out = pd.DataFrame(index=df.index)

    if "client_id" in df.columns:
        out["client_id"] = pd.to_numeric(df["client_id"], errors="coerce")
    elif "external_id" in df.columns:
        out["client_id"] = pd.to_numeric(df["external_id"], errors="coerce")
    else:
        out["client_id"] = np.arange(1, n + 1)

    # Fill missing client ids deterministically within chunk
    if out["client_id"].isna().any():
        missing = out["client_id"].isna()
        out.loc[missing, "client_id"] = (
            np.arange(1, missing.sum() + 1) + n
        )
    out["client_id"] = out["client_id"].astype(np.int64)

    if "application_date" in df.columns:
        out["application_date"] = _parse_date(df["application_date"])
    else:
        out["application_date"] = pd.NaT
    out["application_date"] = out["application_date"].fillna(
        pd.Timestamp("2022-06-01")
    )

    out["loan_amount"] = pd.to_numeric(
        df["loan_amount"] if "loan_amount" in df.columns else np.nan,
        errors="coerce",
    )
    out["loan_term"] = (
        df["loan_term"].map(_parse_term)
        if "loan_term" in df.columns
        else 36
    )
    out["loan_term"] = pd.to_numeric(out["loan_term"], errors="coerce").fillna(36).astype(int)

    out["interest_rate"] = (
        df["interest_rate"].map(_parse_interest_rate)
        if "interest_rate" in df.columns
        else np.nan
    )
    out["interest_rate"] = pd.to_numeric(out["interest_rate"], errors="coerce")

    out["income"] = pd.to_numeric(
        df["income"] if "income" in df.columns else np.nan, errors="coerce"
    )
    out["employment_years"] = (
        df["employment_years"].map(_parse_emp_length)
        if "employment_years" in df.columns
        else np.nan
    )
    out["employment_years"] = pd.to_numeric(
        out["employment_years"], errors="coerce"
    ).fillna(0.0)

    out["home_ownership"] = (
        df["home_ownership"].astype(str).str.upper().str.strip()
        if "home_ownership" in df.columns
        else "OTHER"
    )
    out["home_ownership"] = out["home_ownership"].replace(
        {"NAN": "OTHER", "NONE": "OTHER", "ANY": "OTHER"}
    )

    out["purpose"] = (
        df["purpose"].astype(str).str.lower().str.strip()
        if "purpose" in df.columns
        else "other"
    )
    out["purpose"] = out["purpose"].replace({"nan": "other", "none": "other"})

    out["dti_ratio"] = pd.to_numeric(
        df["dti_ratio"] if "dti_ratio" in df.columns else np.nan, errors="coerce"
    )
    out["credit_score"] = pd.to_numeric(
        df["credit_score"] if "credit_score" in df.columns else np.nan,
        errors="coerce",
    )
    out["num_open_accounts"] = pd.to_numeric(
        df["num_open_accounts"] if "num_open_accounts" in df.columns else 0,
        errors="coerce",
    ).fillna(0).astype(int)
    out["num_delinquencies"] = pd.to_numeric(
        df["num_delinquencies"] if "num_delinquencies" in df.columns else 0,
        errors="coerce",
    ).fillna(0).astype(int)

    if "total_credit_limit" in df.columns:
        out["total_credit_limit"] = pd.to_numeric(
            df["total_credit_limit"], errors="coerce"
        )
    else:
        out["total_credit_limit"] = out["loan_amount"] * 2.0

    # Align default series length if built empty
    if len(default_series) != n:
        default_series = pd.Series(False, index=df.index)
    out["is_default"] = default_series.astype(bool).values
    out["data_source"] = data_source

    # Drop unusable rows
    out = out.dropna(subset=["loan_amount", "income", "application_date"])
    out = out[out["loan_amount"] > 0]
    out = out[out["income"] > 0]

    out["application_date"] = pd.to_datetime(out["application_date"]).dt.date
    return out[RAW_APPLICATION_COLUMNS]


def load_lending_club_data(
    filepath: str | Path,
    engine: Engine,
    chunk_size: int = 10_000,
    replace: bool = True,
) -> int:
    """
    Load raw applications into raw.applications.
    By default truncates the table first (idempotent daily load).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}. "
            "Place lending_club.csv or sample_applications.csv under data/."
        )

    logger.info(f"Loading data from {filepath}")

    if replace:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE raw.payment_history RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE raw.credit_bureau RESTART IDENTITY CASCADE"))
            conn.execute(text("TRUNCATE TABLE raw.applications RESTART IDENTITY CASCADE"))
        logger.info("Truncated raw tables for idempotent reload")

    total_loaded = 0
    with engine.begin() as conn:
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
            processed = _preprocess_raw_chunk(chunk)
            if processed.empty:
                continue
            processed.to_sql(
                "applications",
                conn,
                schema="raw",
                if_exists="append",
                index=False,
                method="multi",
            )
            total_loaded += len(processed)
            logger.info(f"Loaded {total_loaded} rows so far...")

    logger.info(f"Done. Total rows loaded: {total_loaded}")
    return total_loaded


def generate_synthetic_payment_history(
    engine: Engine,
    n_payments_per_loan: int = 12,
    seed: int = 42,
) -> int:
    """
    Generate PRE-application payment history (no leakage).
    Simulates prior loan behaviour ending before application_date.
    """
    rng = np.random.default_rng(seed)

    with engine.connect() as conn:
        apps = pd.read_sql(
            """
            SELECT application_id, client_id, loan_amount, loan_term,
                   is_default, application_date
            FROM raw.applications
            """,
            conn,
        )

    if apps.empty:
        logger.warning("No applications found; skip payment history")
        return 0

    logger.info(f"Generating pre-application payment history for {len(apps)} apps")
    apps["application_date"] = pd.to_datetime(apps["application_date"])

    records: list[dict] = []
    for row in apps.itertuples(index=False):
        n_pay = int(min(n_payments_per_loan, row.loan_term or 36))
        monthly = float(row.loan_amount) / max(n_pay, 1)
        # History ends 1–30 days before application
        end_offset = int(rng.integers(1, 31))
        for month in range(n_pay):
            payment_date = row.application_date - pd.DateOffset(
                months=n_pay - month, days=end_offset
            )
            # Mild stochastic delinquency; NOT conditioned on is_default
            if rng.random() < 0.12:
                days_overdue = int(
                    rng.choice([5, 15, 30, 60], p=[0.4, 0.3, 0.2, 0.1])
                )
            else:
                days_overdue = 0
            paid_ratio = max(
                0.0, 1.0 - days_overdue / 100.0 * float(rng.random())
            )
            records.append(
                {
                    "application_id": int(row.application_id),
                    "payment_date": payment_date.date(),
                    "amount_due": round(monthly, 2),
                    "amount_paid": round(monthly * paid_ratio, 2),
                    "days_overdue": days_overdue,
                }
            )

    payments_df = pd.DataFrame.from_records(records)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE raw.payment_history RESTART IDENTITY"))
        payments_df.to_sql(
            "payment_history",
            conn,
            schema="raw",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=10_000,
        )

    logger.info(f"Generated {len(payments_df)} payment records")
    return len(payments_df)


def generate_synthetic_bureau(
    engine: Engine,
    seed: int = 42,
) -> int:
    """Generate one credit-bureau snapshot per client dated before application."""
    rng = np.random.default_rng(seed + 7)

    with engine.connect() as conn:
        apps = pd.read_sql(
            """
            SELECT application_id, client_id, income, loan_amount,
                   application_date, num_open_accounts
            FROM raw.applications
            """,
            conn,
        )

    if apps.empty:
        return 0

    apps["application_date"] = pd.to_datetime(apps["application_date"])
    rows: list[dict] = []
    for row in apps.itertuples(index=False):
        report_date = (
            row.application_date - pd.Timedelta(days=int(rng.integers(7, 60)))
        ).date()
        income = float(row.income) if row.income else 50000.0
        leverage = float(row.loan_amount) / max(income, 1.0) if hasattr(row, "loan_amount") else 0.3
        # Do not condition on is_default (would leak label into bureau features)
        balance = max(0.0, income * float(rng.uniform(0.1, 1.0)) * (0.7 + leverage))
        inquiries = int(rng.integers(0, 5))
        active = max(1, int(row.num_open_accounts or 1))
        rows.append(
            {
                "client_id": int(row.client_id),
                "report_date": report_date,
                "num_inquiries_6m": inquiries,
                "num_active_loans": active,
                "total_balance": round(balance, 2),
                "num_defaults_hist": int(rng.integers(0, 2)),
                "oldest_account_months": int(rng.integers(12, 240)),
            }
        )

    bureau_df = pd.DataFrame.from_records(rows)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE raw.credit_bureau RESTART IDENTITY"))
        bureau_df.to_sql(
            "credit_bureau",
            conn,
            schema="raw",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=10_000,
        )
    logger.info(f"Generated {len(bureau_df)} bureau records")
    return len(bureau_df)
