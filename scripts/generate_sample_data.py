#!/usr/bin/env python3
"""Generate a synthetic Lending-Club-like applications CSV for local demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2023-12-31")
    days = (end - start).days

    dates = start + pd.to_timedelta(rng.integers(0, days + 1, n), unit="D")
    income = rng.lognormal(mean=10.8, sigma=0.45, size=n).clip(15_000, 400_000)
    # Learnable signal with realistic ~15–25% default rate
    loan_amount = rng.uniform(2_000, 40_000, n)
    dti = rng.uniform(5, 45, n)
    credit = rng.integers(580, 820, n)
    int_rate = rng.uniform(6, 24, n)
    emp_noise = rng.uniform(0, 10, n)
    risk = (
        1.6 * (loan_amount / income)
        + 0.035 * dti
        + 0.008 * (700 - credit)
        + 0.05 * int_rate
        - 0.06 * emp_noise
        + rng.normal(0, 0.65, n)
    )
    # Shift so mean default probability is roughly 0.15–0.25
    p_default = 1 / (1 + np.exp(-(risk - 3.8)))
    is_default = rng.random(n) < p_default

    purposes = [
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "major_purchase",
        "medical",
        "other",
    ]
    home = ["RENT", "OWN", "MORTGAGE", "OTHER"]

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "member_id": rng.integers(100000, 999999, n),
            "loan_amnt": np.round(loan_amount, 2),
            "term": rng.choice([36, 60], n, p=[0.7, 0.3]),
            "int_rate": np.round(int_rate, 2),
            "annual_inc": np.round(income, 2),
            "emp_length": np.where(
                emp_noise < 1,
                "< 1 year",
                np.where(
                    emp_noise < 3,
                    "2 years",
                    np.where(emp_noise < 7, "5 years", "10+ years"),
                ),
            ),
            "home_ownership": rng.choice(home, n, p=[0.4, 0.15, 0.4, 0.05]),
            "purpose": rng.choice(purposes, n),
            "dti": np.round(dti, 2),
            "fico_range_low": credit,
            "fico_range_high": credit + 4,
            "open_acc": rng.integers(2, 20, n),
            "delinq_2yrs": rng.integers(0, 4, n),
            "revol_bal": np.round(loan_amount * rng.uniform(0.5, 3.0, n), 2),
            "inq_last_6mths": rng.integers(0, 6, n),
            "issue_d": dates.strftime("%b-%Y"),
            "loan_status": np.where(
                is_default,
                rng.choice(
                    ["Charged Off", "Default", "Late (31-120 days)"], n
                ),
                rng.choice(
                    ["Fully Paid", "Current", "In Grace Period"], n
                ),
            ),
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "sample_applications.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
