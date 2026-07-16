"""
Learning curve for PD scoring — tradeoff data quantity vs quality.

Hybrid approach:
- Feature engineering already done once for 400k (300k with history + 100k cold-start)
- This script samples final features.application_features (400k) at
  [20k, 50k, 100k, 200k, 300k, 400k] and trains stacking ensemble
  (CatBoost/LGBM/XGB + LogReg) with reduced estimators for speed on weak PC.

Why this shows professionalism:
- Data lake 2.2M raw always kept, sampling only at FE/training for cost
- Cold-start 75% coverage: model must handle thin-file clients
- Learning curve proves data-driven optimal size, not arbitrary 200k
- Metrics: ROC-AUC (ranking), PR-AUC (threshold-free precision-recall, better for imbalanced),
  KS/Gini (bank standard), Brier/LogLoss (calibration), Precision/Recall@thr with reject inference disclaimer

Outputs:
- artifacts/learning_curve.csv (includes val_pr_auc, val_roc_auc, etc)
- artifacts/learning_curve.png (ROC-AUC vs PR-AUC vs rows + etc)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, FeatureConfig, TrainingConfig, get_db_engine
from src.logging_utils import get_logger

logger = get_logger(__name__)

SAMPLE_SIZES = [20_000, 50_000, 100_000, 200_000, 300_000, 400_000]
# For quick dev on very weak PC set FE_MAX_APPS=200k and sizes up to 200k


def _load_full_features(engine):
    from sqlalchemy import text

    query = text(
        """
        SELECT
            f.*,
            a.is_default,
            a.application_date AS app_date
        FROM features.application_features f
        JOIN raw.applications a ON f.application_id = a.application_id
        WHERE f.feature_version = :version
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"version": FeatureConfig.VERSION})

    if df.empty:
        raise ValueError("Empty features.application_features — run feature_engineering DAG first")

    df["application_date"] = pd.to_datetime(df["app_date"])
    df = df.drop(columns=["app_date"])
    return df.sort_values("application_date")


def _time_split(df, train_end, val_end):
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = df[df["application_date"] <= train_end_ts].copy()
    val = df[(df["application_date"] > train_end_ts) & (df["application_date"] <= val_end_ts)].copy()
    test = df[df["application_date"] > val_end_ts].copy()
    return train, val, test


def evaluate_all_metrics(y_true, y_proba):
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        auc,
        brier_score_loss,
        log_loss,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    from scipy import stats

    from src.models.scoring.evaluate import compute_metrics

    base = compute_metrics(y_true, y_proba, prefix="val")

    # PR-AUC (threshold-independent, better for imbalanced default ~10-15%)
    try:
        pr_auc = average_precision_score(y_true, y_proba)
        # More precise PR-AUC via precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_curve = auc(recall, precision)
    except Exception:
        pr_auc = 0.0
        pr_auc_curve = 0.0

    base["val_pr_auc"] = float(pr_auc)  # average precision
    base["val_pr_auc_curve"] = float(pr_auc_curve)

    # Additional business thresholds (with reject inference disclaimer)
    thresholds = [0.05, 0.15, 0.30]
    extra = {}
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        try:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            prec = rec = f1 = 0.0
        extra[f"val_precision@{thr}"] = float(prec)
        extra[f"val_recall@{thr}"] = float(rec)
        extra[f"val_f1@{thr}"] = float(f1)

    base.update(extra)
    return base


def train_and_eval_for_size(full_df, n_rows, train_end, val_end):
    """
    Sample n_rows from full_df (random with seed 42 for reproducibility),
    then time-based split and train stacking ensemble with reduced estimators for speed.
    """
    from src.data.queries import get_feature_dataset
    from src.models.scoring.ensemble import StackingEnsemble
    from src.models.scoring.train import get_feature_matrix

    # Sample before split to simulate having only n_rows raw data
    # Use random_state 42 for reproducible curve
    if len(full_df) > n_rows:
        sampled = full_df.sample(n=n_rows, random_state=42)
        sampled = sampled.sort_values("application_date")
    else:
        sampled = full_df.copy()

    train, val, test = _time_split(sampled, train_end, val_end)

    if len(train) < 1000 or len(val) < 500:
        logger.warning(f"Sample {n_rows}: too small after time split train={len(train)} val={len(val)}, skipping")
        return None

    # For weak PC: use reduced folds and estimators; ensemble will override to 200/300 but we monkey-patch via env
    # We'll train single LightGBM for fastest curve, then full ensemble only for 200k+ to save time
    # Decision: <150k single LGBM, >=150k full stacking
    start = time.time()

    engine = get_db_engine()  # not used for sampled data path, but needed for interface
    # Use StackingEnsemble with small estimators
    # To speed, we temporarily patch get_model params via TrainingConfig? We directly train here for control

    # Prepare matrices
    from src.models.scoring.train import get_feature_matrix

    X_train_df, feature_cols = get_feature_matrix(train)
    X_val_df, _ = get_feature_matrix(val, feature_cols)
    medians = X_train_df.median(numeric_only=True)
    X_train_df = X_train_df.fillna(medians)
    X_val_df = X_val_df.fillna(medians)

    y_train = train[TrainingConfig.TARGET_COL].astype(int).values
    y_val = val[TrainingConfig.TARGET_COL].astype(int).values

    # Train base models with small estimators for curve speed + imbalance handling
    from src.models.scoring.train import get_model
    from sklearn.linear_model import LogisticRegression

    pos_rate = float(y_train.mean()) if len(y_train) else 0.15
    scale_w = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    base_models = ["lightgbm", "catboost", "xgboost"] if n_rows >= 100_000 else ["lightgbm"]
    oof = np.zeros((len(X_train_df), len(base_models)))
    val_preds = np.zeros((len(X_val_df), len(base_models)))
    fitted_base = []

    # TimeSeriesSplit for OOF
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=min(3, max(2, len(X_train_df) // 500)))

    for i, mtype in enumerate(base_models):
        oof_col = np.zeros(len(X_train_df))
        for tr_idx, va_idx in tscv.split(X_train_df):
            params = {"scale_pos_weight": scale_w} if mtype in ("lightgbm", "xgboost") else {}
            params.update(
                {"n_estimators": 150, "learning_rate": 0.1}
                if mtype != "catboost"
                else {"iterations": 150, "learning_rate": 0.1},
            )
            model = get_model(mtype, params)
            model.fit(X_train_df.values[tr_idx], y_train[tr_idx])
            oof_col[va_idx] = model.predict_proba(X_train_df.values[va_idx])[:, 1]
        # Fill missing leading zeros
        missing = oof_col == 0
        if missing.any() and (~missing).any():
            oof_col[missing] = oof_col[~missing].mean()
        oof[:, i] = oof_col

        full_params = {"scale_pos_weight": scale_w} if mtype in ("lightgbm", "xgboost") else {}
        full_params.update(
            {"n_estimators": 200, "learning_rate": 0.08}
            if mtype != "catboost"
            else {"iterations": 200, "learning_rate": 0.08},
        )
        full_model = get_model(mtype, full_params)
        full_model.fit(X_train_df.values, y_train)
        val_preds[:, i] = full_model.predict_proba(X_val_df.values)[:, 1]
        fitted_base.append(full_model)

    meta = LogisticRegression(C=1.0, max_iter=500, random_state=TrainingConfig.RANDOM_SEED)
    if len(base_models) > 1:
        meta.fit(oof, y_train)
        final_val = meta.predict_proba(val_preds)[:, 1]
    else:
        final_val = val_preds[:, 0]

    elapsed = time.time() - start

    metrics = evaluate_all_metrics(y_val, final_val)
    metrics["n_rows"] = n_rows
    metrics["train_rows"] = len(train)
    metrics["val_rows"] = len(val)
    metrics["test_rows"] = len(test)
    metrics["train_time_sec"] = elapsed
    # Cold-start coverage in this sample
    if "avg_days_overdue_90d" in sampled.columns:
        metrics["pct_with_history"] = float((sampled["avg_days_overdue_90d"].fillna(0) != 0).mean())
    else:
        metrics["pct_with_history"] = 0.0

    logger.info(
        f"Sample {n_rows}: AUC={metrics.get('val_auc_roc',0):.4f} "
        f"KS={metrics.get('val_ks_statistic',0):.4f} Gini={metrics.get('val_gini',0):.4f} "
        f"time={elapsed:.1f}s coverage={metrics['pct_with_history']:.1%}"
    )
    return metrics


def main():
    engine = get_db_engine()
    # Try to load full features; if FE_MAX_APPS=400k, this is 400k
    try:
        full = _load_full_features(engine)
    except Exception as exc:
        logger.error(f"Failed to load features: {exc}")
        logger.info("Falling back to get_feature_dataset split")
        from src.data.queries import get_feature_dataset

        train, val, test = get_feature_dataset(
            engine, TrainingConfig.TRAIN_END_DATE, TrainingConfig.VAL_END_DATE
        )
        full = pd.concat([train, val, test], ignore_index=True)
        full["application_date"] = pd.to_datetime(full["application_date"])
        full = full.sort_values("application_date")

    print(f"Full feature table: {full.shape}, date range {full['application_date'].min()} .. {full['application_date'].max()}")
    print(f"Default rate overall: {full[TrainingConfig.TARGET_COL].astype(float).mean():.3%}")

    # Filter sample sizes to not exceed full
    max_available = len(full)
    sizes = [s for s in SAMPLE_SIZES if s <= max_available]
    if max_available not in sizes:
        sizes.append(max_available)
    sizes = sorted(set(sizes))

    print(f"Learning curve sizes: {sizes}")

    # Auto date split handling: if train/val empty with configured dates, use quantile split
    from src.models.scoring.train import prepare_time_split

    split_info = prepare_time_split(engine)
    train_end = split_info["train_end"]
    val_end = split_info["val_end"]
    print(f"Using time split: train_end={train_end}, val_end={val_end}, info={split_info}")

    results = []
    for n in sizes:
        res = train_and_eval_for_size(full, n, train_end, val_end)
        if res:
            results.append(res)

    if not results:
        print("No results")
        return

    df_res = pd.DataFrame(results).sort_values("n_rows")
    out_csv = ARTIFACTS_DIR / "learning_curve.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # AUC, KS, Gini vs rows
        ax = axes[0, 0]
        ax.plot(df_res["n_rows"], df_res["val_auc_roc"], marker="o", label="ROC AUC")
        ax.plot(df_res["n_rows"], df_res["val_pr_auc"], marker="D", label="PR AUC (avg prec)")
        ax.plot(df_res["n_rows"], df_res["val_ks_statistic"], marker="s", label="KS")
        ax.plot(df_res["n_rows"], df_res["val_gini"], marker="^", label="Gini")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Metric")
        ax.set_title("Quality vs Data (ROC-AUC vs PR-AUC)")
        ax.legend()
        ax.grid(True)

        # Brier & LogLoss
        ax = axes[0, 1]
        ax.plot(df_res["n_rows"], df_res["val_brier_score"], marker="o", label="Brier (lower better)")
        ax.plot(df_res["n_rows"], df_res["val_log_loss"], marker="s", label="LogLoss")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Calibration")
        ax.legend()
        ax.grid(True)
        ax.set_title("Calibration vs Data")

        # Precision/Recall at 0.15 threshold + PR-AUC reference
        # Note: precision/recall here are accepted-only biased (reject inference problem)
        ax = axes[1, 0]
        if "val_precision@0.15" in df_res.columns:
            ax.plot(df_res["n_rows"], df_res["val_pr_auc"], marker="D", label="PR-AUC (threshold-free)", linestyle="--")
            ax.plot(df_res["n_rows"], df_res["val_precision@0.15"], marker="o", label="Precision@0.15 (biased)")
            ax.plot(df_res["n_rows"], df_res["val_recall@0.15"], marker="s", label="Recall@0.15 (biased)")
            ax.plot(df_res["n_rows"], df_res["val_f1@0.15"], marker="^", label="F1@0.15 (biased)")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Business metric")
        ax.legend(fontsize=8)
        ax.grid(True)
        ax.set_title("PR-AUC + Precision/Recall @ thr=0.15\n(accepted-only, reject inference bias)")

        # Train time
        ax = axes[1, 1]
        ax.plot(df_res["n_rows"], df_res["train_time_sec"], marker="o", color="red")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Time sec")
        ax.grid(True)
        ax.set_title("Train time vs Data")

        plt.tight_layout()
        out_png = ARTIFACTS_DIR / "learning_curve.png"
        plt.savefig(out_png, dpi=150)
        print(f"Saved {out_png}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    # Print tradeoff analysis
    print("\n=== Tradeoff Analysis ===")
    df_res = df_res.sort_values("n_rows")
    for i in range(1, len(df_res)):
        prev = df_res.iloc[i - 1]
        cur = df_res.iloc[i]
        delta_n = cur["n_rows"] - prev["n_rows"]
        delta_auc = cur["val_auc_roc"] - prev["val_auc_roc"]
        gain_per_100k = delta_auc / (delta_n / 100_000) if delta_n else 0
        print(
            f"{int(prev['n_rows'])}->{int(cur['n_rows'])} (+{int(delta_n)}): "
            f"AUC {prev['val_auc_roc']:.4f}->{cur['val_auc_roc']:.4f} "
            f"Δ={delta_auc:+.4f} gain/100k={gain_per_100k:+.5f}, "
            f"time {prev['train_time_sec']:.0f}s->{cur['train_time_sec']:.0f}s"
        )

    # Recommend optimal
    # Find elbow where gain/100k < 0.005 (0.5% AUC)
    optimal = df_res.iloc[-1]["n_rows"]
    for i in range(1, len(df_res)):
        prev = df_res.iloc[i - 1]
        cur = df_res.iloc[i]
        delta_n = cur["n_rows"] - prev["n_rows"]
        delta_auc = cur["val_auc_roc"] - prev["val_auc_roc"]
        gain_per_100k = delta_auc / (delta_n / 100_000)
        if gain_per_100k < 0.005:
            optimal = prev["n_rows"]
            break

    print(f"\nRecommended optimal rows for weak PC: ~{int(optimal)} (elbow where gain/100k <0.005 AUC)")


if __name__ == "__main__":
    main()
