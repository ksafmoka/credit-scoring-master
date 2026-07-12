"""Raw data ingestion: Lending Club CSV → PostgreSQL + synthetic histories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.logging_utils import get_logger

logger = get_logger(__name__)

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
    "last_fico_range_low": "credit_score",
    "credit_score": "credit_score",
    "open_acc": "num_open_accounts",
    "num_open_accounts": "num_open_accounts",
    "delinq_2yrs": "num_delinquencies",
    "num_delinquencies": "num_delinquencies",
    "revol_bal": "total_credit_limit",
    "total_rev_hi_lim": "total_credit_limit",
    "total_bc_limit": "total_credit_limit",
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

# Soft caps for insane LC outliers (still DOUBLE PRECISION in DB)
CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "loan_amount": (0.0, 1e7),
    "income": (0.0, 1e8),
    "interest_rate": (0.0, 100.0),
    "employment_years": (0.0, 80.0),
    "dti_ratio": (-100.0, 999.0),
    "credit_score": (0.0, 1000.0),
    "num_open_accounts": (0.0, 500.0),
    "num_delinquencies": (0.0, 200.0),
    "total_credit_limit": (0.0, 1e9),
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


def _series_or_nan(df: pd.DataFrame, col: str, n: int) -> pd.Series:
    """Always return a length-n Series (never a bare scalar)."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index if len(df.index) == n else range(n))


def _to_numeric_series(values, n: int) -> pd.Series:
    if isinstance(values, pd.Series):
        s = values
    else:
        s = pd.Series(values)
    if len(s) != n:
        s = s.reindex(range(n))
    # Strip %, commas, whitespace often found in LC exports
    if s.dtype == object:
        s = (
            s.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
        )
    return pd.to_numeric(s, errors="coerce")


def _clip_series(s: pd.Series, col: str) -> pd.Series:
    if col not in CLIP_BOUNDS:
        return s
    lo, hi = CLIP_BOUNDS[col]
    return s.clip(lower=lo, upper=hi)


def _parse_term_series(s: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        text_val = str(v).lower().replace("months", "").strip()
        try:
            return float(text_val)
        except ValueError:
            return np.nan

    return s.map(one)


def _parse_emp_length_series(s: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        t = str(v).strip().lower()
        if t in {"n/a", "na", "", "null", "none"}:
            return np.nan
        if t.startswith("<"):
            return 0.5
        if "10" in t and "+" in t:
            return 10.0
        digits = "".join(ch for ch in t if ch.isdigit() or ch == ".")
        try:
            return float(digits) if digits else np.nan
        except ValueError:
            return np.nan

    return s.map(one)


def _parse_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="%b-%Y")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def _default_flag(df: pd.DataFrame, n: int) -> pd.Series:
    if "is_default" in df.columns and df["is_default"].notna().any():
        s = df["is_default"]
        if s.dtype == object:
            s = s.astype(str).str.lower().isin({"1", "true", "yes", "y", "default"})
        return s.fillna(False).astype(bool)
    if "loan_status" in df.columns:
        return df["loan_status"].astype(str).isin(DEFAULT_STATUSES)
    return pd.Series(False, index=range(n))


def _preprocess_raw_chunk(
    chunk: pd.DataFrame, data_source: str = "lending_club"
) -> pd.DataFrame:
    """Map external / Lending Club columns to raw.applications schema."""
    if chunk is None or len(chunk) == 0:
        return pd.DataFrame(columns=RAW_APPLICATION_COLUMNS)

    df = chunk.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Drop LC footer / garbage rows (all-null id or loan_status Total)
    if "id" in df.columns:
        df = df[~df["id"].astype(str).str.contains("Total|Loans", case=False, na=False)]

    renamed: dict[str, str] = {}
    for col in df.columns:
        key = col.lower()
        if key in COLUMN_MAP:
            renamed[col] = COLUMN_MAP[key]
        elif col in COLUMN_MAP.values():
            renamed[col] = col
    df = df.rename(columns=renamed)
    # If rename created duplicate columns, keep first
    df = df.loc[:, ~df.columns.duplicated()]

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=RAW_APPLICATION_COLUMNS)

    if "credit_score" in df.columns and "credit_score_high" in df.columns:
        low = _to_numeric_series(df["credit_score"], n)
        high = _to_numeric_series(df["credit_score_high"], n)
        df["credit_score"] = (low + high) / 2.0

    out = pd.DataFrame(index=df.index)

    if "client_id" in df.columns:
        out["client_id"] = _to_numeric_series(df["client_id"], n)
    elif "external_id" in df.columns:
        out["client_id"] = _to_numeric_series(df["external_id"], n)
    else:
        out["client_id"] = np.arange(1, n + 1, dtype=np.float64)

    if out["client_id"].isna().any():
        missing = out["client_id"].isna()
        # stable unique ids within chunk
        out.loc[missing, "client_id"] = (
            np.arange(1, int(missing.sum()) + 1) + n + hash(str(df.columns)) % 10_000
        )
    out["client_id"] = (
        out["client_id"].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.int64)
    )

    if "application_date" in df.columns:
        out["application_date"] = _parse_date(df["application_date"])
    else:
        out["application_date"] = pd.Series(pd.NaT, index=df.index)
    out["application_date"] = out["application_date"].fillna(pd.Timestamp("2016-06-01"))

    out["loan_amount"] = _clip_series(
        _to_numeric_series(_series_or_nan(df, "loan_amount", n), n), "loan_amount"
    )
    if "loan_term" in df.columns:
        out["loan_term"] = _parse_term_series(df["loan_term"])
    else:
        out["loan_term"] = 36.0
    out["loan_term"] = (
        pd.to_numeric(out["loan_term"], errors="coerce")
        .fillna(36)
        .clip(1, 120)
        .astype(int)
    )

    if "interest_rate" in df.columns:
        out["interest_rate"] = _to_numeric_series(df["interest_rate"], n)
    else:
        out["interest_rate"] = np.nan
    out["interest_rate"] = _clip_series(out["interest_rate"], "interest_rate")

    out["income"] = _clip_series(
        _to_numeric_series(_series_or_nan(df, "income", n), n), "income"
    )

    if "employment_years" in df.columns:
        out["employment_years"] = _parse_emp_length_series(df["employment_years"])
    else:
        out["employment_years"] = 0.0
    out["employment_years"] = _clip_series(
        pd.to_numeric(out["employment_years"], errors="coerce").fillna(0.0),
        "employment_years",
    )

    if "home_ownership" in df.columns:
        out["home_ownership"] = (
            df["home_ownership"].astype(str).str.upper().str.strip()
        )
    else:
        out["home_ownership"] = "OTHER"
    out["home_ownership"] = out["home_ownership"].replace(
        {"NAN": "OTHER", "NONE": "OTHER", "ANY": "OTHER", "NULL": "OTHER"}
    )
    out["home_ownership"] = out["home_ownership"].str.slice(0, 20)

    if "purpose" in df.columns:
        out["purpose"] = df["purpose"].astype(str).str.lower().str.strip()
    else:
        out["purpose"] = "other"
    out["purpose"] = out["purpose"].replace(
        {"nan": "other", "none": "other", "null": "other"}
    )
    out["purpose"] = out["purpose"].str.slice(0, 50)

    out["dti_ratio"] = _clip_series(
        _to_numeric_series(_series_or_nan(df, "dti_ratio", n), n), "dti_ratio"
    )
    out["credit_score"] = _clip_series(
        _to_numeric_series(_series_or_nan(df, "credit_score", n), n), "credit_score"
    )
    out["num_open_accounts"] = (
        _clip_series(
            _to_numeric_series(_series_or_nan(df, "num_open_accounts", n), n),
            "num_open_accounts",
        )
        .fillna(0)
        .astype(int)
    )
    out["num_delinquencies"] = (
        _clip_series(
            _to_numeric_series(_series_or_nan(df, "num_delinquencies", n), n),
            "num_delinquencies",
        )
        .fillna(0)
        .astype(int)
    )

    if "total_credit_limit" in df.columns:
        out["total_credit_limit"] = _to_numeric_series(df["total_credit_limit"], n)
    else:
        out["total_credit_limit"] = out["loan_amount"] * 2.0
    out["total_credit_limit"] = _clip_series(
        out["total_credit_limit"], "total_credit_limit"
    )

    out["is_default"] = _default_flag(df, n).astype(bool).values
    out["data_source"] = data_source

    # Drop unusable
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["loan_amount", "income", "application_date"])
    out = out[(out["loan_amount"] > 0) & (out["income"] > 0)]
    out["application_date"] = pd.to_datetime(out["application_date"]).dt.date

    # Final float cleanup for SQL DOUBLE PRECISION
    float_cols = [
        "loan_amount",
        "interest_rate",
        "income",
        "employment_years",
        "dti_ratio",
        "credit_score",
        "total_credit_limit",
    ]
    for c in float_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out[RAW_APPLICATION_COLUMNS]


def _ensure_raw_applications_wide_types(engine: Engine) -> None:
    """
    Widen raw.applications numeric columns so LC outliers never overflow
    old NUMERIC(12,2) / NUMERIC(5,2) schemas on existing volumes.
    """
    alters = [
        "ALTER TABLE raw.applications ALTER COLUMN loan_amount TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.applications ALTER COLUMN income TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.applications ALTER COLUMN interest_rate TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.applications ALTER COLUMN employment_years TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.applications ALTER COLUMN dti_ratio TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.applications ALTER COLUMN total_credit_limit TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.applications ALTER COLUMN credit_score TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.payment_history ALTER COLUMN amount_due TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.payment_history ALTER COLUMN amount_paid TYPE DOUBLE PRECISION",
        "ALTER TABLE raw.credit_bureau ALTER COLUMN total_balance TYPE DOUBLE PRECISION",
    ]
    with engine.begin() as conn:
        for sql in alters:
            try:
                conn.execute(text(sql))
            except Exception as exc:
                # column already double / table missing mid-init
                logger.debug(f"skip alter: {sql} ({exc})")


def load_lending_club_data(
    filepath: str | Path,
    engine: Engine,
    chunk_size: int = 20_000,
    replace: bool = True,
) -> int:
    """Load raw applications into raw.applications (idempotent when replace=True)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}. "
            "Place lending_club.csv under data/."
        )

    logger.info(f"Loading data from {filepath}")
    _ensure_raw_applications_wide_types(engine)

    if replace:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "TRUNCATE TABLE raw.payment_history, raw.credit_bureau, "
                    "raw.applications RESTART IDENTITY CASCADE"
                )
            )
        logger.info("Truncated raw tables for idempotent reload")

    total_loaded = 0
    skipped_chunks = 0
    # on_bad_lines skip for messy LC exports (pandas>=1.3)
    read_kw = dict(chunksize=chunk_size, low_memory=False)
    try:
        reader = pd.read_csv(filepath, on_bad_lines="skip", **read_kw)
    except TypeError:
        reader = pd.read_csv(filepath, **read_kw)

    with engine.begin() as conn:
        for i, chunk in enumerate(reader):
            try:
                processed = _preprocess_raw_chunk(chunk)
            except Exception as exc:
                skipped_chunks += 1
                logger.warning(f"Chunk {i} preprocess failed: {exc}")
                continue
            if processed.empty:
                continue
            processed.to_sql(
                "applications",
                conn,
                schema="raw",
                if_exists="append",
                index=False,
                method="multi",
                chunksize=5_000,
            )
            total_loaded += len(processed)
            if (i + 1) % 5 == 0 or total_loaded <= chunk_size:
                logger.info(f"Loaded {total_loaded} rows so far...")

    logger.info(
        f"Done. Total rows loaded: {total_loaded} "
        f"(skipped_chunks={skipped_chunks})"
    )
    if total_loaded == 0:
        raise ValueError(
            f"No usable rows loaded from {filepath}. "
            "Check columns (loan_amnt, annual_inc, issue_d, loan_status)."
        )
    return total_loaded


def generate_synthetic_payment_history(
    engine: Engine,
    n_payments_per_loan: int = 12,
    seed: int = 42,
) -> int:
    """Generate PRE-application payment history (no leakage)."""
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

    logger.info(
        f"Generating pre-application payment history for {len(apps)} apps"
    )
    apps["application_date"] = pd.to_datetime(apps["application_date"])
    apps["loan_amount"] = pd.to_numeric(apps["loan_amount"], errors="coerce").fillna(0)
    apps["loan_term"] = (
        pd.to_numeric(apps["loan_term"], errors="coerce").fillna(36).astype(int)
    )

    records: list[dict] = []
    for row in apps.itertuples(index=False):
        n_pay = int(min(n_payments_per_loan, max(int(row.loan_term or 36), 1)))
        monthly = float(row.loan_amount) / max(n_pay, 1)
        # keep payment amounts finite
        monthly = float(np.clip(monthly, 0, 1e9))
        end_offset = int(rng.integers(1, 31))
        for month in range(n_pay):
            payment_date = row.application_date - pd.DateOffset(
                months=n_pay - month, days=end_offset
            )
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
        conn.execute(
            text("TRUNCATE TABLE raw.payment_history RESTART IDENTITY")
        )
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


def generate_synthetic_bureau(engine: Engine, seed: int = 42) -> int:
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
    apps["income"] = pd.to_numeric(apps["income"], errors="coerce").fillna(50_000)
    apps["loan_amount"] = pd.to_numeric(apps["loan_amount"], errors="coerce").fillna(0)

    rows: list[dict] = []
    for row in apps.itertuples(index=False):
        report_date = (
            row.application_date - pd.Timedelta(days=int(rng.integers(7, 60)))
        ).date()
        income = float(np.clip(float(row.income), 1.0, 1e8))
        leverage = float(row.loan_amount) / max(income, 1.0)
        balance = float(
            np.clip(
                income * float(rng.uniform(0.1, 1.0)) * (0.7 + leverage),
                0,
                1e10,
            )
        )
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
