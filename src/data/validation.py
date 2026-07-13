"""Data quality checks for raw and feature tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from src.logging_utils import get_logger
logger = get_logger(__name__)


@dataclass
class ValidationReport:
    success: bool
    errors: list[str]
    warnings: list[str]
    stats: dict


def validate_raw_applications(df: pd.DataFrame) -> ValidationReport:
    """Validate raw applications before / after load."""
    errors: list[str] = []
    warnings: list[str] = []

    required_cols = [
        "client_id",
        "application_date",
        "loan_amount",
        "income",
        "is_default",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if "loan_amount" in df.columns:
        invalid_loan = int((df["loan_amount"] <= 0).sum())
        if invalid_loan > 0:
            errors.append(f"Found {invalid_loan} rows with loan_amount <= 0")

    if "income" in df.columns:
        null_income = int(df["income"].isna().sum())
        null_pct = null_income / max(len(df), 1)
        if null_pct > 0.3:
            errors.append(f"Too many nulls in income: {null_pct:.1%}")
        elif null_pct > 0.1:
            warnings.append(f"High null rate in income: {null_pct:.1%}")

    if "is_default" in df.columns and len(df) > 0:
        target_rate = float(df["is_default"].astype(float).mean())
        if target_rate < 0.01:
            warnings.append(f"Very low default rate: {target_rate:.2%}")
        if target_rate > 0.5:
            warnings.append(f"Very high default rate: {target_rate:.2%}")

    if "client_id" in df.columns and "application_date" in df.columns:
        dupes = int(
            df.duplicated(subset=["client_id", "application_date"]).sum()
        )
        if dupes > 0:
            warnings.append(f"Found {dupes} duplicate records")

    stats = {
        "row_count": len(df),
        "target_rate": (
            float(df["is_default"].astype(float).mean())
            if "is_default" in df.columns and len(df)
            else None
        ),
        "date_range": (
            (
                str(df["application_date"].min()),
                str(df["application_date"].max()),
            )
            if "application_date" in df.columns and len(df)
            else None
        ),
    }

    success = len(errors) == 0
    for err in errors:
        logger.error(f"Validation error: {err}")
    for warn in warnings:
        logger.warning(f"Validation warning: {warn}")

    return ValidationReport(
        success=success, errors=errors, warnings=warnings, stats=stats
    )


def validate_feature_table(df: pd.DataFrame, checks: dict) -> ValidationReport:
    """Validate the merged feature table."""
    errors: list[str] = []
    warnings: list[str] = []

    if "no_nulls_in_critical" in checks:
        for col in checks["no_nulls_in_critical"]:
            if col in df.columns:
                null_count = int(df[col].isna().sum())
                if null_count > 0:
                    errors.append(
                        f"Nulls in critical feature {col}: {null_count}"
                    )

    if "value_ranges" in checks:
        for col, (min_val, max_val) in checks["value_ranges"].items():
            if col in df.columns:
                out_of_range = int(
                    ((df[col] < min_val) | (df[col] > max_val)).sum()
                )
                if out_of_range > 0:
                    warnings.append(
                        f"{col}: {out_of_range} values out of "
                        f"range [{min_val}, {max_val}]"
                    )

    if "row_count_min" in checks and len(df) < checks["row_count_min"]:
        errors.append(
            f"Too few rows: {len(df)} < {checks['row_count_min']}"
        )

    return ValidationReport(
        success=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats={"row_count": len(df)},
    )
