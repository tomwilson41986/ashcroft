#!/usr/bin/env python3
"""
Train Improved BFSP Model with Research-Backed Features.

Trains the enhanced model with Benter/Woods/Ziemba/syndicate features
and compares against the baseline. Shows live progress with 2-minute
status updates.

Usage:
    python train_improved.py
    python train_improved.py --db horse_racing.db
    python train_improved.py --start-date 2020-01-01
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.custom_metrics import CustomMetricsEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

# Import feature lists from train_bfsp
from train_bfsp import ALL_FEATURE_COLS, build_context_features


class ProgressTracker:
    """Live progress tracker with 2-minute status updates."""

    def __init__(self, total_folds: int, update_interval: int = 120):
        self.total_folds = total_folds
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update = self.start_time
        self.fold_metrics = []
        self.current_fold = 0
        self.phase = "Initializing"

    def update_phase(self, phase: str):
        self.phase = phase
        self._print_status(force=True)

    def fold_complete(self, fold_num: int, metrics: dict):
        self.current_fold = fold_num
        self.fold_metrics.append(metrics)
        now = time.time()
        if now - self.last_update >= self.update_interval or fold_num == self.total_folds:
            self._print_status(force=True)
        else:
            # Brief inline update
            elapsed = now - self.start_time
            pct = fold_num / self.total_folds * 100
            bar = self._make_bar(pct)
            print(
                f"\r  [{bar}] {pct:5.1f}% | Fold {fold_num}/{self.total_folds} | "
                f"MAE={metrics['log_mae']:.4f} R2={metrics['log_r2']:.4f} | "
                f"{elapsed:.0f}s elapsed",
                end="", flush=True,
            )

    def _print_status(self, force=False):
        now = time.time()
        if not force and now - self.last_update < self.update_interval:
            return
        self.last_update = now
        elapsed = now - self.start_time

        print("\n")
        print("=" * 72)
        print(f"  TRAINING STATUS UPDATE  |  {time.strftime('%H:%M:%S')}")
        print("=" * 72)
        print(f"  Phase:      {self.phase}")

        if self.total_folds > 0:
            pct = self.current_fold / self.total_folds * 100
            bar = self._make_bar(pct)
            print(f"  Progress:   [{bar}] {pct:.1f}%")
            print(f"  Folds:      {self.current_fold}/{self.total_folds}")

        print(f"  Elapsed:    {self._format_time(elapsed)}")

        if self.current_fold > 0 and self.current_fold < self.total_folds:
            avg_per_fold = elapsed / self.current_fold
            remaining = avg_per_fold * (self.total_folds - self.current_fold)
            print(f"  ETA:        {self._format_time(remaining)}")

        if self.fold_metrics:
            recent = self.fold_metrics[-1]
            avg_mae = np.mean([m["log_mae"] for m in self.fold_metrics])
            avg_r2 = np.mean([m["log_r2"] for m in self.fold_metrics])
            avg_mdape = np.mean([m["median_ape_pct"] for m in self.fold_metrics])
            print(f"\n  Running Averages:")
            print(f"    Log MAE:     {avg_mae:.4f}")
            print(f"    Log R2:      {avg_r2:.4f}")
            print(f"    Median APE:  {avg_mdape:.2f}%")
            print(f"  Last Fold:")
            print(f"    Log MAE:     {recent['log_mae']:.4f}")
            print(f"    Log R2:      {recent['log_r2']:.4f}")
            print(f"    Median APE:  {recent['median_ape_pct']:.2f}%")

        print("=" * 72)
        print("", flush=True)

    @staticmethod
    def _make_bar(pct, width=30):
        filled = int(width * pct / 100)
        return "#" * filled + "-" * (width - filled)

    @staticmethod
    def _format_time(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics on log and BFSP scales."""
    log_mae = mean_absolute_error(y_true, y_pred)
    log_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    log_r2 = r2_score(y_true, y_pred)

    bfsp_true = np.exp(y_true)
    bfsp_pred = np.exp(y_pred)
    bfsp_mae = mean_absolute_error(bfsp_true, bfsp_pred)

    pct_errors = np.abs(bfsp_true - bfsp_pred) / np.clip(bfsp_true, 1e-7, None)
    median_ape = np.median(pct_errors) * 100
    mean_ape = np.mean(pct_errors) * 100
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    return {
        "log_mae": round(log_mae, 4),
        "log_rmse": round(log_rmse, 4),
        "log_r2": round(log_r2, 4),
        "bfsp_mae": round(bfsp_mae, 2),
        "median_ape_pct": round(median_ape, 2),
        "mean_ape_pct": round(mean_ape, 2),
        "correlation": round(correlation, 4),
    }


def load_data(db_path, start_date=None):
    conn = sqlite3.connect(db_path)
    if start_date:
        df = pd.read_sql_query(
            "SELECT * FROM race_results WHERE race_date >= ? ORDER BY race_date, race_time",
            conn, params=(start_date,),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM race_results ORDER BY race_date, race_time", conn
        )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def train_model(df, feature_cols, params, tracker, min_train_days=365,
                val_window_days=30, step_days=30):
    """Walk-forward training with live progress tracking."""

    # Create folds
    min_date = df["race_date"].min()
    max_date = df["race_date"].max()
    folds = []
    train_end = min_date + timedelta(days=min_train_days)
    while train_end + timedelta(days=val_window_days) <= max_date:
        folds.append((train_end, train_end + timedelta(days=val_window_days)))
        train_end += timedelta(days=step_days)

    tracker.total_folds = len(folds)
    tracker.update_phase(f"Walk-Forward Validation ({len(folds)} folds)")

    fold_metrics = []
    all_val_preds = []

    for i, (val_start, val_end) in enumerate(folds):
        train_df = df[df["race_date"] < val_start]
        val_df = df[(df["race_date"] >= val_start) & (df["race_date"] < val_end)]

        if len(train_df) < 100 or len(val_df) < 10:
            continue

        X_train = train_df[feature_cols].astype(float)
        y_train = train_df["log_bfsp"].astype(float)
        X_val = val_df[feature_cols].astype(float)
        y_val = val_df["log_bfsp"].astype(float)

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        model = lgb.train(
            params, train_set, num_boost_round=2000,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)],
        )

        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val.values, y_pred)
        metrics["best_iteration"] = model.best_iteration
        fold_metrics.append(metrics)

        val_copy = val_df.copy()
        val_copy["predicted_log_bfsp"] = y_pred
        all_val_preds.append(val_copy)

        tracker.fold_complete(i + 1, metrics)

    print()  # newline after progress bar

    # Aggregate
    avg_metrics = {
        "wf_log_mae": round(np.mean([m["log_mae"] for m in fold_metrics]), 4),
        "wf_log_rmse": round(np.mean([m["log_rmse"] for m in fold_metrics]), 4),
        "wf_log_r2": round(np.mean([m["log_r2"] for m in fold_metrics]), 4),
        "wf_bfsp_mae": round(np.mean([m["bfsp_mae"] for m in fold_metrics]), 2),
        "wf_median_ape_pct": round(np.mean([m["median_ape_pct"] for m in fold_metrics]), 2),
        "n_folds": len(fold_metrics),
    }

    # Overall walk-forward
    overall_wf = {}
    if all_val_preds:
        combined = pd.concat(all_val_preds, ignore_index=True)
        overall_wf = compute_metrics(
            combined["log_bfsp"].values,
            combined["predicted_log_bfsp"].values,
        )
        overall_wf = {f"overall_wf_{k}": v for k, v in overall_wf.items()}

    # Final model training
    tracker.update_phase("Training Final Model (90/10 split)")
    dates = df["race_date"].sort_values().unique()
    cutoff = dates[int(len(dates) * 0.9)]

    final_train = df[df["race_date"] < cutoff]
    final_val = df[df["race_date"] >= cutoff]

    X_train = final_train[feature_cols].astype(float)
    y_train = final_train["log_bfsp"].astype(float)
    X_val = final_val[feature_cols].astype(float)
    y_val = final_val["log_bfsp"].astype(float)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    final_model = lgb.train(
        params, train_set, num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)],
    )

    y_pred = final_model.predict(X_val)
    final_metrics = compute_metrics(y_val.values, y_pred)
    final_metrics["best_iteration"] = final_model.best_iteration
    final_metrics["train_size"] = len(final_train)
    final_metrics["val_size"] = len(final_val)

    # Feature importance
    importance = pd.DataFrame({
        "feature": final_model.feature_name(),
        "importance": final_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return final_model, avg_metrics, overall_wf, final_metrics, fold_metrics, importance


def print_comparison(baseline, improved):
    """Print side-by-side comparison of baseline vs improved model."""
    print("\n")
    print("=" * 80)
    print("  MODEL COMPARISON: BASELINE vs IMPROVED (Research-Enhanced)")
    print("=" * 80)

    metrics_to_compare = [
        ("Walk-Forward Log MAE", "wf_log_mae", True),
        ("Walk-Forward Log RMSE", "wf_log_rmse", True),
        ("Walk-Forward Log R2", "wf_log_r2", False),
        ("Walk-Forward BFSP MAE", "wf_bfsp_mae", True),
        ("Walk-Forward Median APE", "wf_median_ape_pct", True),
    ]

    print(f"\n  {'Metric':<35s} {'Baseline':>12s} {'Improved':>12s} {'Delta':>12s} {'Change':>10s}")
    print("  " + "-" * 81)

    for label, key, lower_is_better in metrics_to_compare:
        b_val = baseline.get(key, 0)
        i_val = improved.get(key, 0)
        delta = i_val - b_val

        if lower_is_better:
            pct_change = -(delta / b_val * 100) if b_val != 0 else 0
            arrow = "+" if delta < 0 else "-" if delta > 0 else "="
        else:
            pct_change = (delta / b_val * 100) if b_val != 0 else 0
            arrow = "+" if delta > 0 else "-" if delta < 0 else "="

        print(
            f"  {label:<35s} {b_val:>12.4f} {i_val:>12.4f} "
            f"{delta:>+12.4f} {arrow}{abs(pct_change):>7.2f}%"
        )

    # Final model comparison
    fm_metrics = [
        ("Final Log MAE", "log_mae", True),
        ("Final Log RMSE", "log_rmse", True),
        ("Final Log R2", "log_r2", False),
        ("Final BFSP MAE", "bfsp_mae", True),
        ("Final Median APE %", "median_ape_pct", True),
        ("Final Correlation", "correlation", False),
    ]

    print(f"\n  {'Final Model Metric':<35s} {'Baseline':>12s} {'Improved':>12s} {'Delta':>12s} {'Change':>10s}")
    print("  " + "-" * 81)

    for label, key, lower_is_better in fm_metrics:
        b_val = baseline.get(f"final_{key}", 0)
        i_val = improved.get(f"final_{key}", 0)
        delta = i_val - b_val

        if lower_is_better:
            pct_change = -(delta / b_val * 100) if b_val != 0 else 0
            arrow = "+" if delta < 0 else "-" if delta > 0 else "="
        else:
            pct_change = (delta / b_val * 100) if b_val != 0 else 0
            arrow = "+" if delta > 0 else "-" if delta < 0 else "="

        print(
            f"  {label:<35s} {b_val:>12.4f} {i_val:>12.4f} "
            f"{delta:>+12.4f} {arrow}{abs(pct_change):>7.2f}%"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train improved BFSP model with research-backed features"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--output-dir", default=MODEL_DIR)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--val-window", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        print("Run: bash scripts/setup_session.sh --db")
        sys.exit(1)

    # Load data
    log.info("Loading data from S3 database...")
    df = load_data(args.db, start_date=args.start_date)
    log.info(f"  {len(df):,} rows loaded ({df['race_date'].min().date()} to {df['race_date'].max().date()})")

    # Calculate all metrics (including new research-backed ones)
    log.info("Calculating all custom metrics (including new research-backed features)...")
    engine = CustomMetricsEngine()
    df = engine.calculate_all(df)
    log.info(f"  {len(df.columns)} columns after metric calculation")

    # Build context features
    df = build_context_features(df)

    # Prepare target
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["log_bfsp"] = np.log(df["bfsp"])
    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
    log.info(f"  {len(df):,} rows with valid BFSP")

    # Determine available features
    available = [c for c in ALL_FEATURE_COLS if c in df.columns]
    seen = set()
    feature_cols = []
    for c in available:
        if c not in seen:
            seen.add(c)
            feature_cols.append(c)

    # Identify new vs baseline features
    from train_bfsp import (
        HORSE_CAREER_FEATURES, ROLLING_FEATURES, EPF_FEATURES,
        STABILITY_FEATURES, PFD_FEATURES, PRIZE_FEATURES, OFS_FEATURES,
        DSLR_FEATURES, LRP_FEATURES, PACE_FEATURES, JOCKEY_FEATURES,
        TRAINER_FEATURES, TJ_FEATURES, RACE_STRENGTH_FEATURES,
        RECENCY_FEATURES, RANK_FEATURES, CONTEXT_FEATURES,
    )
    baseline_features_list = (
        HORSE_CAREER_FEATURES + ROLLING_FEATURES + EPF_FEATURES
        + STABILITY_FEATURES + PFD_FEATURES + PRIZE_FEATURES
        + OFS_FEATURES + DSLR_FEATURES + LRP_FEATURES + PACE_FEATURES
        + JOCKEY_FEATURES + TRAINER_FEATURES + TJ_FEATURES
        + RACE_STRENGTH_FEATURES + RECENCY_FEATURES + RANK_FEATURES
        + CONTEXT_FEATURES
    )
    baseline_feature_cols = [c for c in baseline_features_list if c in df.columns]
    bseen = set()
    baseline_feature_cols = [c for c in baseline_feature_cols if not (c in bseen or bseen.add(c))]

    new_only = [c for c in feature_cols if c not in set(baseline_feature_cols)]

    log.info(f"  Baseline features: {len(baseline_feature_cols)}")
    log.info(f"  Total features (with research enhancements): {len(feature_cols)}")
    log.info(f"  New features added: {len(new_only)}")

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }

    # =====================================================================
    # PHASE 1: Train BASELINE model (original features only)
    # =====================================================================
    print("\n" + "=" * 72)
    print("  PHASE 1: TRAINING BASELINE MODEL (187 original features)")
    print("=" * 72 + "\n")

    baseline_tracker = ProgressTracker(0, update_interval=120)
    (baseline_model, baseline_wf, baseline_owf, baseline_final,
     baseline_folds, baseline_imp) = train_model(
        df, baseline_feature_cols, params, baseline_tracker,
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
    )

    baseline_tracker.update_phase("Baseline Training Complete")

    # =====================================================================
    # PHASE 2: Train IMPROVED model (all features including research-backed)
    # =====================================================================
    print("\n" + "=" * 72)
    print(f"  PHASE 2: TRAINING IMPROVED MODEL ({len(feature_cols)} features)")
    print("=" * 72 + "\n")

    improved_tracker = ProgressTracker(0, update_interval=120)
    (improved_model, improved_wf, improved_owf, improved_final,
     improved_folds, improved_imp) = train_model(
        df, feature_cols, params, improved_tracker,
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
    )

    improved_tracker.update_phase("Improved Training Complete")

    # =====================================================================
    # COMPARISON
    # =====================================================================
    baseline_all = {**baseline_wf, **{f"final_{k}": v for k, v in baseline_final.items()}}
    improved_all = {**improved_wf, **{f"final_{k}": v for k, v in improved_final.items()}}

    print_comparison(baseline_all, improved_all)

    # Show new features in top 30 importance
    print("\n  Top 30 Feature Importance (Improved Model):")
    print("  " + "-" * 60)
    for _, row in improved_imp.head(30).iterrows():
        is_new = " [NEW]" if row["feature"] in set(new_only) else ""
        bar_len = int(25 * row["importance"] / improved_imp["importance"].max())
        bar = "#" * bar_len
        print(f"  {row['feature']:<40s} {bar} {row['importance']:.0f}{is_new}")

    # Save improved model
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, "bfsp_model_improved.lgb")
    improved_model.save_model(model_path)

    meta = {
        "model_type": "bfsp_regression_improved",
        "target": "log_bfsp",
        "feature_cols": feature_cols,
        "params": params,
        "final_metrics": improved_final,
        "n_new_features": len(new_only),
        "new_features": new_only,
    }
    meta_path = os.path.join(args.output_dir, "bfsp_model_improved_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Save comparison
    comparison = {
        "baseline": {
            "n_features": len(baseline_feature_cols),
            "walk_forward": baseline_wf,
            "overall_walk_forward": baseline_owf,
            "final_model": baseline_final,
        },
        "improved": {
            "n_features": len(feature_cols),
            "n_new_features": len(new_only),
            "new_features": new_only,
            "walk_forward": improved_wf,
            "overall_walk_forward": improved_owf,
            "final_model": improved_final,
        },
    }
    comp_path = os.path.join(args.output_dir, "research_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    imp_path = os.path.join(args.output_dir, "improved_feature_importance.csv")
    improved_imp.to_csv(imp_path, index=False)

    # If improved is better, also save as the main model
    if improved_final["log_mae"] <= baseline_final["log_mae"]:
        main_path = os.path.join(args.output_dir, "bfsp_model.lgb")
        improved_model.save_model(main_path)

        main_meta = {
            "model_type": "bfsp_regression",
            "target": "log_bfsp",
            "feature_cols": feature_cols,
            "params": params,
            "final_metrics": improved_final,
        }
        with open(os.path.join(args.output_dir, "bfsp_model_meta.json"), "w") as f:
            json.dump(main_meta, f, indent=2, default=str)

        print("\n  Improved model is BETTER - saved as main model.")
    else:
        print("\n  Baseline model remains better - improved saved separately.")

    print(f"\n  All artifacts saved to: {args.output_dir}")
    print(f"  Comparison saved to: {comp_path}")


if __name__ == "__main__":
    main()
