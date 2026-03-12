#!/usr/bin/env python3
"""
Retrain BFSP Model with Proper Train / Validation / Test Splits.

Temporal splits:
    - Test set:       most recent 60 days (held out, never seen during training)
    - Validation set: 60 days before the test set (used for early stopping)
    - Training set:   everything before the validation set

Early stopping uses the validation set.
Final evaluation is reported on both validation and test sets.
"""

import json
import logging
import os
import sqlite3
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.custom_metrics import CustomMetricsEngine
from train_bfsp import (
    ALL_FEATURE_COLS,
    build_context_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

# Holdout sizes in days
VAL_DAYS = 60
TEST_DAYS = 60


START_DATE = "2022-01-01"  # Use recent data to fit in memory


def load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results WHERE race_date >= ? ORDER BY race_date, race_time",
        conn,
        params=(START_DATE,),
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    engine = CustomMetricsEngine()
    log.info("Calculating all custom metrics...")
    df = engine.calculate_all(df)
    log.info(f"  Custom metrics done: {len(df.columns)} columns")

    log.info("Building context features...")
    df = build_context_features(df)

    # Target
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["log_bfsp"] = np.log(df["bfsp"])

    # Determine available features (deduplicated, ordered)
    seen = set()
    feature_cols = []
    for c in ALL_FEATURE_COLS:
        if c in df.columns and c not in seen:
            seen.add(c)
            feature_cols.append(c)

    log.info(f"  {len(feature_cols)} features available")
    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
    return df, feature_cols


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def print_metrics(label: str, metrics: dict):
    print(f"\n  {label}")
    print("  " + "-" * 55)
    print(f"  {'MAE (log-BFSP):':<25s} {metrics['log_mae']:.4f}")
    print(f"  {'RMSE (log-BFSP):':<25s} {metrics['log_rmse']:.4f}")
    print(f"  {'R² (log-BFSP):':<25s} {metrics['log_r2']:.4f}")
    print(f"  {'MAE (BFSP £):':<25s} {metrics['bfsp_mae']:.2f}")
    print(f"  {'Median APE:':<25s} {metrics['median_ape_pct']:.1f}%")
    print(f"  {'Mean APE:':<25s} {metrics['mean_ape_pct']:.1f}%")
    print(f"  {'Correlation:':<25s} {metrics['correlation']:.4f}")


def main():
    db_path = DEFAULT_DB
    if not os.path.exists(db_path):
        log.error(f"Database not found: {db_path}")
        sys.exit(1)

    # ---- Load & prepare ----
    log.info("Loading data...")
    df = load_data(db_path)
    log.info(f"  {len(df):,} rows loaded")

    df, feature_cols = prepare_data(df)
    log.info(f"  {len(df):,} rows with valid BFSP")

    # ---- Temporal splits ----
    max_date = df["race_date"].max()
    test_cutoff = max_date - pd.Timedelta(days=TEST_DAYS)
    val_cutoff = test_cutoff - pd.Timedelta(days=VAL_DAYS)

    train_df = df[df["race_date"] < val_cutoff].copy()
    val_df = df[(df["race_date"] >= val_cutoff) & (df["race_date"] < test_cutoff)].copy()
    test_df = df[df["race_date"] >= test_cutoff].copy()

    print("\n" + "=" * 70)
    print("DATA SPLITS")
    print("=" * 70)
    print(f"  Training set:    {len(train_df):>8,} rows  "
          f"({train_df['race_date'].min().date()} to {train_df['race_date'].max().date()})")
    print(f"  Validation set:  {len(val_df):>8,} rows  "
          f"({val_df['race_date'].min().date()} to {val_df['race_date'].max().date()})")
    print(f"  Test set:        {len(test_df):>8,} rows  "
          f"({test_df['race_date'].min().date()} to {test_df['race_date'].max().date()})")
    print("=" * 70)

    # ---- Build LightGBM datasets ----
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["log_bfsp"].astype(float)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["log_bfsp"].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["log_bfsp"].astype(float)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

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

    # ---- Train with early stopping on validation set ----
    log.info("Training with early stopping (patience=50) on validation set...")
    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=5000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    log.info(f"Best iteration: {model.best_iteration}")

    # ---- Evaluate on validation set (raw predictions) ----
    y_val_pred = model.predict(X_val)
    val_metrics_raw = compute_metrics(y_val.values, y_val_pred)

    # ---- Evaluate on test set (raw predictions) ----
    y_test_pred = model.predict(X_test)
    test_metrics_raw = compute_metrics(y_test.values, y_test_pred)

    # ---- Benter-style per-race probability normalization ----
    # Convert predicted log(BFSP) -> implied prob (1/BFSP), normalize per race
    # so probabilities sum to 1.0, then back to fair-book BFSP.
    def normalize_race_probs(subset_df, log_bfsp_pred):
        """Normalize predicted BFSP so implied win probs sum to 1 per race."""
        out = subset_df[["raceid"]].copy()
        out["log_bfsp_pred_raw"] = log_bfsp_pred
        out["bfsp_pred_raw"] = np.exp(log_bfsp_pred)
        out["implied_prob"] = 1.0 / out["bfsp_pred_raw"]

        # Normalize within each race so probs sum to 1.0
        race_prob_sum = out.groupby("raceid")["implied_prob"].transform("sum")
        out["win_prob"] = out["implied_prob"] / race_prob_sum
        out["fair_bfsp"] = 1.0 / out["win_prob"]
        out["log_bfsp_norm"] = np.log(out["fair_bfsp"])
        return out

    val_norm = normalize_race_probs(val_df, y_val_pred)
    test_norm = normalize_race_probs(test_df, y_test_pred)

    val_metrics_norm = compute_metrics(y_val.values, val_norm["log_bfsp_norm"].values)
    test_metrics_norm = compute_metrics(y_test.values, test_norm["log_bfsp_norm"].values)

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("MODEL RESULTS")
    print("=" * 70)
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Features used:  {len(feature_cols)}")

    print_metrics("VALIDATION SET — Raw Predictions", val_metrics_raw)
    print_metrics("VALIDATION SET — Race-Normalized (Benter)", val_metrics_norm)
    print_metrics("TEST SET — Raw Predictions", test_metrics_raw)
    print_metrics("TEST SET — Race-Normalized (Benter)", test_metrics_norm)

    # Show per-race probability check
    print("\n  Per-Race Win Probability Check (Test Set):")
    print("  " + "-" * 55)
    race_prob_sums = test_norm.groupby("raceid")["win_prob"].sum()
    print(f"  Mean sum per race:  {race_prob_sums.mean():.6f}")
    print(f"  Min sum per race:   {race_prob_sums.min():.6f}")
    print(f"  Max sum per race:   {race_prob_sums.max():.6f}")
    print(f"  All sum to 1.0:     {np.allclose(race_prob_sums, 1.0)}")

    # ---- Feature importance (top 30) ----
    importance = model.feature_importance(importance_type="gain")
    imp_df = (
        pd.DataFrame({"feature": model.feature_name(), "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n  Top 30 Features by Gain:")
    print("  " + "-" * 55)
    for _, row in imp_df.head(30).iterrows():
        bar_len = int(30 * row["importance"] / imp_df["importance"].max())
        bar = "=" * bar_len
        print(f"  {row['feature']:<40s} {bar} {row['importance']:.0f}")

    # ---- Save artifacts ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "bfsp_model.lgb")
    model.save_model(model_path)
    log.info(f"Model saved to {model_path}")

    # Use normalized metrics as the primary ones saved
    val_metrics = val_metrics_norm
    test_metrics = test_metrics_norm

    meta = {
        "model_type": "bfsp_regression",
        "target": "log_bfsp",
        "feature_cols": feature_cols,
        "params": params,
        "best_iteration": model.best_iteration,
        "val_metrics_raw": val_metrics_raw,
        "val_metrics_normalized": val_metrics_norm,
        "test_metrics_raw": test_metrics_raw,
        "test_metrics_normalized": test_metrics_norm,
        "splits": {
            "train_rows": len(train_df),
            "train_date_range": f"{train_df['race_date'].min().date()} to {train_df['race_date'].max().date()}",
            "val_rows": len(val_df),
            "val_date_range": f"{val_df['race_date'].min().date()} to {val_df['race_date'].max().date()}",
            "test_rows": len(test_df),
            "test_date_range": f"{test_df['race_date'].min().date()} to {test_df['race_date'].max().date()}",
        },
    }
    meta_path = os.path.join(MODEL_DIR, "bfsp_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    imp_path = os.path.join(MODEL_DIR, "bfsp_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)

    summary_path = os.path.join(MODEL_DIR, "bfsp_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Model & artifacts saved to {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
