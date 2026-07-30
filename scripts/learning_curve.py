"""
Learning curve for PD scoring — dual-model approach.

Shows AUC vs data size for two segments:
- with_history: clients WITH payment history, all features
- cold_start: clients WITHOUT payment history, non-aggregation features only

This demonstrates the value of dual-model architecture:
separate models for thick-file vs thin-file clients.

Outputs:
- artifacts/learning_curve.csv (both segments)
- artifacts/learning_curve.png (dual curves)
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

# Sample sizes per segment (max = segment size)
SAMPLE_SIZES = [20_000, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000]


def _load_full_features(engine):
    """Load features + payment_history membership."""
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

    # Get payment history IDs
    with engine.connect() as conn:
        hist_ids = set(
            pd.read_sql(text("SELECT DISTINCT application_id FROM raw.payment_history"), conn)[
                "application_id"
            ].tolist()
        )

    df["has_history"] = df["application_id"].isin(hist_ids)
    print(f"Total features: {len(df)}, with_history: {df['has_history'].sum()}, "
          f"cold_start: {(~df['has_history']).sum()}")

    return df.sort_values("application_date"), hist_ids


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

    try:
        pr_auc = average_precision_score(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_curve = auc(recall, precision)
    except Exception:
        pr_auc = 0.0
        pr_auc_curve = 0.0

    base["val_pr_auc"] = float(pr_auc)
    base["val_pr_auc_curve"] = float(pr_auc_curve)

    thresholds = [0.05, 0.15, 0.30]
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        try:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            prec = rec = f1 = 0.0
        base[f"val_precision@{thr}"] = float(prec)
        base[f"val_recall@{thr}"] = float(rec)
        base[f"val_f1@{thr}"] = float(f1)

    return base


def train_and_eval_for_size(
    full_df: pd.DataFrame,
    n_rows: int,
    train_end: str,
    val_end: str,
    segment: str,
    feature_cols: list[str],
):
    """Train model on sampled data for a specific segment."""
    from src.models.scoring.train import get_feature_matrix, get_model

    if len(full_df) > n_rows:
        sampled = full_df.sample(n=n_rows, random_state=42)
        sampled = sampled.sort_values("application_date")
    else:
        sampled = full_df.copy()

    train, val, test = _time_split(sampled, train_end, val_end)

    if len(train) < 500 or len(val) < 200:
        logger.warning(f"[{segment}] Sample {n_rows}: too small train={len(train)} val={len(val)}, skipping")
        return None

    start = time.time()

    # Prepare matrices with segment-specific features
    cols = [c for c in feature_cols if c in train.columns]
    X_train_df = train[cols].apply(pd.to_numeric, errors="coerce")
    X_val_df = val[cols].apply(pd.to_numeric, errors="coerce")
    medians = X_train_df.median(numeric_only=True)
    X_train_df = X_train_df.fillna(medians)
    X_val_df = X_val_df.fillna(medians)

    y_train = train[TrainingConfig.TARGET_COL].astype(int).values
    y_val = val[TrainingConfig.TARGET_COL].astype(int).values

    pos_rate = float(y_train.mean()) if len(y_train) else 0.15
    scale_w = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    # Train best of 3 models (quick, no Optuna for speed)
    best_auc = 0
    best_proba = None

    for model_type in ["lightgbm", "catboost", "xgboost"]:
        params = {}
        if model_type in ("lightgbm", "xgboost"):
            params["scale_pos_weight"] = scale_w

        if model_type == "catboost":
            params["iterations"] = 200
            params["learning_rate"] = 0.1
        else:
            params["n_estimators"] = 200
            params["learning_rate"] = 0.1

        model = get_model(model_type, params)
        model.fit(X_train_df.values, y_train)
        proba = model.predict_proba(X_val_df.values)[:, 1]

        from sklearn.metrics import roc_auc_score
        try:
            auc_val = roc_auc_score(y_val, proba)
        except Exception:
            auc_val = 0.0

        if auc_val > best_auc:
            best_auc = auc_val
            best_proba = proba
            best_model_type = model_type

    elapsed = time.time() - start

    if best_proba is None:
        return None

    metrics = evaluate_all_metrics(y_val, best_proba)
    metrics["n_rows"] = n_rows
    metrics["train_rows"] = len(train)
    metrics["val_rows"] = len(val)
    metrics["test_rows"] = len(test)
    metrics["train_time_sec"] = elapsed
    metrics["segment"] = segment
    metrics["best_model"] = best_model_type
    metrics["default_rate_train"] = float(y_train.mean())
    metrics["default_rate_val"] = float(y_val.mean())

    logger.info(
        f"[{segment}] Sample {n_rows}: AUC={metrics.get('val_auc_roc', 0):.4f} "
        f"(best: {best_model_type}) KS={metrics.get('val_ks_statistic', 0):.4f} "
        f"time={elapsed:.1f}s"
    )
    return metrics


def main():
    engine = get_db_engine()

    try:
        full, hist_ids = _load_full_features(engine)
    except Exception as exc:
        logger.error(f"Failed to load features: {exc}")
        return

    print(f"Date range: {full['application_date'].min()} .. {full['application_date'].max()}")
    print(f"Overall default rate: {full[TrainingConfig.TARGET_COL].astype(float).mean():.3%}")

    # Auto date split
    from src.models.scoring.train import prepare_time_split
    split_info = prepare_time_split(engine)
    train_end = split_info["train_end"]
    val_end = split_info["val_end"]
    print(f"Time split: train_end={train_end}, val_end={val_end}")

    # Split into segments
    df_history = full[full["has_history"]].copy()
    df_cold = full[~full["has_history"]].copy()

    print(f"\nSegments:")
    print(f"  with_history: {len(df_history)} rows, "
          f"default_rate={df_history[TrainingConfig.TARGET_COL].astype(float).mean():.3%}")
    print(f"  cold_start:   {len(df_cold)} rows, "
          f"default_rate={df_cold[TrainingConfig.TARGET_COL].astype(float).mean():.3%}")

    # Feature sets per segment
    history_features = FeatureConfig.ALL_FEATURES
    cold_features = FeatureConfig.COLD_START_FEATURES

    all_results = []

    # Run learning curve for each segment
    for segment, seg_df, feat_cols in [
        ("with_history", df_history, history_features),
        ("cold_start", df_cold, cold_features),
    ]:
        max_available = len(seg_df)
        sizes = [s for s in SAMPLE_SIZES if s <= max_available]
        if max_available not in sizes and max_available > 5000:
            sizes.append(max_available)
        sizes = sorted(set(sizes))

        print(f"\n[{segment}] Learning curve sizes: {sizes}")

        for n in sizes:
            res = train_and_eval_for_size(seg_df, n, train_end, val_end, segment, feat_cols)
            if res:
                all_results.append(res)

    if not all_results:
        print("No results")
        return

    df_res = pd.DataFrame(all_results).sort_values(["segment", "n_rows"])
    out_csv = ARTIFACTS_DIR / "learning_curve.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Plot dual learning curves
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        df_hist = df_res[df_res["segment"] == "with_history"].sort_values("n_rows")
        df_cold = df_res[df_res["segment"] == "cold_start"].sort_values("n_rows")

        # AUC comparison
        ax = axes[0, 0]
        if not df_hist.empty:
            ax.plot(df_hist["n_rows"], df_hist["val_auc_roc"], "o-", color="steelblue",
                    label="With history (all features)", linewidth=2)
        if not df_cold.empty:
            ax.plot(df_cold["n_rows"], df_cold["val_auc_roc"], "s-", color="darkorange",
                    label="Cold-start (no aggregation)", linewidth=2)
        ax.set_xlabel("Training rows")
        ax.set_ylabel("ROC AUC")
        ax.set_title("Dual-Model Learning Curve: AUC vs Data Size")
        ax.legend()
        ax.grid(True)

        # KS + Gini
        ax = axes[0, 1]
        if not df_hist.empty:
            ax.plot(df_hist["n_rows"], df_hist["val_ks_statistic"], "o-", color="steelblue",
                    label="KS (history)")
            ax.plot(df_hist["n_rows"], df_hist["val_gini"], "^-", color="steelblue",
                    alpha=0.5, label="Gini (history)")
        if not df_cold.empty:
            ax.plot(df_cold["n_rows"], df_cold["val_ks_statistic"], "s-", color="darkorange",
                    label="KS (cold-start)")
            ax.plot(df_cold["n_rows"], df_cold["val_gini"], "D-", color="darkorange",
                    alpha=0.5, label="Gini (cold-start)")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Metric")
        ax.set_title("KS & Gini vs Data Size")
        ax.legend(fontsize=8)
        ax.grid(True)

        # Calibration: Brier & LogLoss
        ax = axes[1, 0]
        if not df_hist.empty:
            ax.plot(df_hist["n_rows"], df_hist["val_brier_score"], "o-", color="steelblue",
                    label="Brier (history)")
        if not df_cold.empty:
            ax.plot(df_cold["n_rows"], df_cold["val_brier_score"], "s-", color="darkorange",
                    label="Brier (cold-start)")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Brier score (lower better)")
        ax.set_title("Calibration vs Data")
        ax.legend()
        ax.grid(True)

        # Train time
        ax = axes[1, 1]
        if not df_hist.empty:
            ax.plot(df_hist["n_rows"], df_hist["train_time_sec"], "o-", color="steelblue",
                    label="With history")
        if not df_cold.empty:
            ax.plot(df_cold["n_rows"], df_cold["train_time_sec"], "s-", color="darkorange",
                    label="Cold-start")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Time (sec)")
        ax.set_title("Training time vs Data")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        out_png = ARTIFACTS_DIR / "learning_curve.png"
        plt.savefig(out_png, dpi=150)
        print(f"Saved {out_png}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for segment in ["with_history", "cold_start"]:
        seg_df = df_res[df_res["segment"] == segment].sort_values("n_rows")
        if seg_df.empty:
            continue
        best = seg_df.loc[seg_df["val_auc_roc"].idxmax()]
        print(f"\n{segment}:")
        print(f"  Best AUC: {best['val_auc_roc']:.4f} at {int(best['n_rows'])} rows "
              f"(model: {best.get('best_model', 'N/A')})")
        print(f"  KS: {best['val_ks_statistic']:.4f}, Gini: {best['val_gini']:.4f}")

    print("\nDual-model approach allows optimal scoring for both segments:")
    print("  - Thick-file clients: full feature set → higher AUC")
    print("  - Thin-file clients: application-level features only → realistic coverage")


if __name__ == "__main__":
    main()
