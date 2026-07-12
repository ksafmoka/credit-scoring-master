#!/usr/bin/env python3
"""
Prepare a Lending Club CSV for this project.

- Reads a full LC dump (or accepted loans file)
- Optionally subsamples for faster training
- Writes data/lending_club.csv
- Prints recommended TRAIN_END_DATE / VAL_END_DATE
- Writes data/date_splits.env for docker compose

Usage:
  python scripts/prepare_lending_club.py --input path/to/accepted.csv --max-rows 200000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lending_club.csv"
SPLITS = ROOT / "data" / "date_splits.env"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}")
        return 1

    print(f"Reading {args.input} ...")
    # Read in chunks to handle large files
    chunks = []
    total = 0
    for chunk in pd.read_csv(args.input, chunksize=50_000, low_memory=False):
        chunks.append(chunk)
        total += len(chunk)
        print(f"  loaded {total} rows...")
        if total >= max(args.max_rows * 3, args.max_rows):
            # enough to subsample
            break
    df = pd.concat(chunks, ignore_index=True)
    print(f"Total read: {len(df)}")

    # Parse issue_d for split recommendation
    if "issue_d" in df.columns:
        dates = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        if dates.isna().mean() > 0.5:
            dates = pd.to_datetime(df["issue_d"], errors="coerce")
        df = df.assign(_issue_ts=dates).dropna(subset=["_issue_ts"])
        df = df.sort_values("_issue_ts")
        print(
            f"Date range: {df['_issue_ts'].min().date()} → {df['_issue_ts'].max().date()}"
        )
    else:
        print("WARNING: no issue_d column — cannot recommend date splits")
        df["_issue_ts"] = pd.NaT

    if len(df) > args.max_rows:
        # time-stratified: keep last max_rows by issue date for recency
        df = df.tail(args.max_rows).copy()
        print(f"Subsampled to last {len(df)} rows by issue_d")

    # Drop helper before save
    issue_ts = df["_issue_ts"]
    df = df.drop(columns=["_issue_ts"], errors="ignore")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(df)} rows)")

    if issue_ts.notna().any():
        q70 = issue_ts.quantile(0.70)
        q85 = issue_ts.quantile(0.85)
        train_end = pd.Timestamp(q70).strftime("%Y-%m-%d")
        val_end = pd.Timestamp(q85).strftime("%Y-%m-%d")
        print("\nRecommended TrainingConfig / env:")
        print(f"  TRAIN_END_DATE={train_end}")
        print(f"  VAL_END_DATE={val_end}")
        SPLITS.write_text(
            f"TRAIN_END_DATE={train_end}\n"
            f"VAL_END_DATE={val_end}\n"
            f"LENDING_CLUB_CSV_PATH=/opt/airflow/data/lending_club.csv\n"
        )
        print(f"Wrote {SPLITS}")
        print(
            "\nAppend these to .env, recreate airflow, then re-run:\n"
            "  data_ingestion → feature_engineering → model_training → batch_prediction"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
