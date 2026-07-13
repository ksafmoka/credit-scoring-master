#!/usr/bin/env python3
"""
Local end-to-end smoke pipeline without Airflow/Docker.

Generates sample data, builds features in pandas, trains a compact model,
and writes serving artifacts under artifacts/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.generate_sample_data import generate  # noqa: E402
from src.config import ARTIFACTS_DIR, DATA_DIR, FeatureConfig, TrainingConfig  # noqa: E402
from src.data.ingestion import _preprocess_raw_chunk  # noqa: E402
from src.features.aggregations import AggregationFeatureComputer  # noqa: E402
from src.features.numerical import NumericalFeatureComputer  # noqa: E402
from src.features.target_encoding import RegularizedTargetEncoder  # noqa: E402
from src.logging_utils import get_logger  # noqa: E402
from src.models.scoring.artifacts import ScoringArtifact  # noqa: E402
from src.models.scoring.evaluate import compute_metrics  # noqa: E402
from src.models.scoring.train import _fit_boosting, get_model  # noqa: E402

logger = get_logger(__name__)


def run_pandas_pipeline(n: int = 4000):
    logger.info("Running pure-pandas local pipeline")
    raw = generate(n=n)
    apps = _preprocess_raw_chunk(raw)
    apps = apps.reset_index(drop=True)
    apps["application_id"] = np.arange(1, len(apps) + 1)
    apps["application_date"] = pd.to_datetime(apps["application_date"])

    rng = np.random.default_rng(42)
    # Pre-application history correlated with observables only (no label leakage)
    pay_rows = []
    for row in apps.itertuples(index=False):
        n_pay = 8
        monthly = float(row.loan_amount) / n_pay
        riskish = float(row.loan_amount) / max(float(row.income), 1.0)
        late_p = float(np.clip(0.05 + 0.15 * riskish, 0.05, 0.35))
        for month in range(n_pay):
            pdate = row.application_date - pd.DateOffset(
                months=n_pay - month, days=5
            )
            if rng.random() < late_p:
                overdue = int(rng.choice([5, 15, 30, 60], p=[0.4, 0.3, 0.2, 0.1]))
            else:
                overdue = 0
            pay_rows.append(
                {
                    "application_id": row.application_id,
                    "payment_date": pdate,
                    "amount_due": monthly,
                    "amount_paid": monthly * max(0.0, 1 - overdue / 100),
                    "days_overdue": overdue,
                }
            )
    payments = pd.DataFrame(pay_rows)

    util = (apps["loan_amount"] / apps["income"]).astype(float)
    bureau = pd.DataFrame(
        {
            "application_id": apps["application_id"],
            "bureau_balance_to_income": (
                util * rng.uniform(0.8, 1.5, len(apps))
                + rng.normal(0, 0.05, len(apps))
            ).clip(0, None),
            "inquiries_per_account": (
                rng.uniform(0, 1.5, len(apps)) + 0.3 * util
            ).clip(0, None),
        }
    )

    num = NumericalFeatureComputer().compute(apps)
    agg = AggregationFeatureComputer(windows=[30, 90, 180]).compute_from_frames(
        payments, apps
    )
    te = RegularizedTargetEncoder(
        cols=["purpose", "home_ownership"], noise_level=0.0
    )
    train_mask = apps["application_date"] <= pd.Timestamp(
        TrainingConfig.TRAIN_END_DATE
    )
    te_train = apps.loc[
        train_mask, ["purpose", "home_ownership", "is_default"]
    ].copy()
    te_train["is_default"] = te_train["is_default"].astype(float)
    te.fit(te_train)
    te_df = te.transform(
        apps[["application_id", "purpose", "home_ownership"]],
        apply_noise=False,
    )

    feat = num.merge(agg, on="application_id", how="left")
    feat = feat.merge(te_df, on="application_id", how="left")
    feat = feat.merge(bureau, on="application_id", how="left")
    feat["is_default"] = apps["is_default"].astype(int).values
    feat["application_date"] = apps["application_date"].values

    feature_cols = [c for c in FeatureConfig.ALL_FEATURES if c in feat.columns]
    train = feat[feat["application_date"] <= TrainingConfig.TRAIN_END_DATE]
    val = feat[
        (feat["application_date"] > TrainingConfig.TRAIN_END_DATE)
        & (feat["application_date"] <= TrainingConfig.VAL_END_DATE)
    ]
    test = feat[feat["application_date"] > TrainingConfig.VAL_END_DATE]

    X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_val = val[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)
    y_train = train["is_default"].astype(int)
    y_val = val["is_default"].astype(int)
    y_test = test["is_default"].astype(int)

    model = get_model("lightgbm", {"n_estimators": 200, "learning_rate": 0.05})
    model = _fit_boosting(model, "lightgbm", X_train, y_train, X_val, y_val)

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    metrics = {}
    metrics.update(compute_metrics(y_val, val_pred, "val"))
    if y_test.nunique() > 1 and len(y_test) > 20:
        metrics.update(compute_metrics(y_test, test_pred, "test"))

    artifact = ScoringArtifact(
        model=model,
        feature_names=feature_cols,
        feature_medians=medians.to_dict(),
        target_encoding=te.to_dict(),
        model_type="lightgbm",
        metrics=metrics,
        global_default_rate=float(te.global_mean),
    )
    artifact.save(ARTIFACTS_DIR)
    te.save(ARTIFACTS_DIR / "target_encoding.json")

    sample_path = DATA_DIR / "sample_applications.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(sample_path, index=False)

    logger.info(f"Metrics: {metrics}")
    logger.info(f"Artifacts written to {ARTIFACTS_DIR}")
    logger.info(f"Sample CSV: {sample_path}")
    return metrics


if __name__ == "__main__":
    run_pandas_pipeline(n=int(os.getenv("SAMPLE_N", "4000")))
