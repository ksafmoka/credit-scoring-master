# src/data/validation.py

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from loguru import logger
from dataclasses import dataclass


@dataclass
class ValidationReport:
    success: bool
    errors: list[str]
    warnings: list[str]
    stats: dict


def validate_raw_applications(df: pd.DataFrame) -> ValidationReport:
    """Валидация сырых данных перед загрузкой."""
    errors = []
    warnings = []

    # проверка обязательных колонок
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

    # диапазоны значений
    if "loan_amount" in df.columns:
        invalid_loan = (df["loan_amount"] <= 0).sum()
        if invalid_loan > 0:
            errors.append(
                f"Found {invalid_loan} rows with loan_amount <= 0"
            )

    if "income" in df.columns:
        null_income = df["income"].isna().sum()
        null_pct = null_income / len(df)
        if null_pct > 0.3:
            errors.append(
                f"Too many nulls in income: {null_pct:.1%}"
            )
        elif null_pct > 0.1:
            warnings.append(
                f"High null rate in income: {null_pct:.1%}"
            )

    # target rate
    if "is_default" in df.columns:
        target_rate = df["is_default"].mean()
        if target_rate < 0.01:
            warnings.append(
                f"Very low default rate: {target_rate:.2%}"
            )
        if target_rate > 0.5:
            warnings.append(
                f"Very high default rate: {target_rate:.2%}"
            )

    # дубли
    if "client_id" in df.columns and "application_date" in df.columns:
        dupes = df.duplicated(
            subset=["client_id", "application_date"]
        ).sum()
        if dupes > 0:
            warnings.append(f"Found {dupes} duplicate records")

    stats = {
        "row_count": len(df),
        "target_rate": float(df["is_default"].mean())
        if "is_default" in df.columns
        else None,
        "date_range": (
            str(df["application_date"].min()),
            str(df["application_date"].max()),
        )
        if "application_date" in df.columns
        else None,
    }

    success = len(errors) == 0

    if not success:
        for err in errors:
            logger.error(f"Validation error: {err}")
    for warn in warnings:
        logger.warning(f"Validation warning: {warn}")

    return ValidationReport(
        success=success,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def validate_feature_table(
    df: pd.DataFrame,
    checks: dict,
) -> ValidationReport:
    """Валидация финальной таблицы фичей."""
    errors = []
    warnings = []

    # no nulls in critical features
    if "no_nulls_in_critical" in checks:
        for col in checks["no_nulls_in_critical"]:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    errors.append(
                        f"Nulls in critical feature {col}: {null_count}"
                    )

    # value ranges
    if "value_ranges" in checks:
        for col, (min_val, max_val) in checks["value_ranges"].items():
            if col in df.columns:
                out_of_range = (
                    (df[col] < min_val) | (df[col] > max_val)
                ).sum()
                if out_of_range > 0:
                    warnings.append(
                        f"{col}: {out_of_range} values out of "
                        f"range [{min_val}, {max_val}]"
                    )

    # minimum row count
    if "row_count_min" in checks:
        if len(df) < checks["row_count_min"]:
            errors.append(
                f"Too few rows: {len(df)} < {checks['row_count_min']}"
            )

    return ValidationReport(
        success=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats={"row_count": len(df)},
    )