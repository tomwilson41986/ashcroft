#!/usr/bin/env python3
"""
Fast BFSP model training — skips walk-forward validation.

Reuses existing hyperparameters from bfsp_model_meta.json and trains
only the final model (90/10 temporal split) to produce bfsp_model.lgb.
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
    BFSPTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")
META_PATH = os.path.join(MODEL_DIR, "bfsp_model_meta.json")


def main():
    # Load existing params from meta
    with open(META_PATH) as f:
        meta = json.load(f)

    params = meta["params"]
    log.info(f"Using params from previous training: {params}")

    # Load data (same date range as previous training)
    log.info("Loading data from 2020-01-01...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM race_results WHERE race_date >= '2020-01-01' "
        "ORDER BY race_date, race_time",
        conn,
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    log.info(f"  {len(df):,} rows loaded")

    # Calculate custom metrics
    log.info("Calculating custom metrics...")
    engine = CustomMetricsEngine()
    df = engine.calculate_all(df)
    log.info(f"  Done: {len(df.columns)} columns")

    # Build context features
    log.info("Building context features...")
    df = build_context_features(df)

    # Target
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)]
    df["log_bfsp"] = np.log(df["bfsp"])

    # Feature columns
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    seen = set()
    unique_features = []
    for c in feature_cols:
        if c not in seen:
            seen.add(c)
            unique_features.append(c)
    feature_cols = unique_features
    log.info(f"  {len(feature_cols)} features available")

    # Keep only the columns we need to reduce memory
    keep_cols = feature_cols + ["race_date", "race_time", "log_bfsp", "bfsp"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    log.info(f"Dropping unused columns to save memory ({len(df.columns)} -> {len(keep_cols)})...")
    import gc
    df = df[keep_cols].copy()
    gc.collect()

    # Sort by date
    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
    log.info(f"Prepared {len(df):,} rows with valid BFSP")

    # 90/10 temporal split for final model
    dates = df["race_date"].sort_values().unique()
    cutoff_idx = int(len(dates) * 0.9)
    cutoff = dates[cutoff_idx]

    train_df = df[df["race_date"] < cutoff]
    val_df = df[df["race_date"] >= cutoff]

    log.info(f"  Train: {len(train_df):,} rows (< {cutoff})")
    log.info(f"  Val: {len(val_df):,} rows (>= {cutoff})")

    # Train
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["log_bfsp"].astype(float)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["log_bfsp"].astype(float)

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    log.info("Training final model...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    log_mae = mean_absolute_error(y_val, y_pred)
    log_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    log_r2 = r2_score(y_val, y_pred)

    bfsp_true = np.exp(y_val)
    bfsp_pred = np.exp(y_pred)
    bfsp_mae = mean_absolute_error(bfsp_true, bfsp_pred)
    pct_errors = np.abs(bfsp_true - bfsp_pred) / np.clip(bfsp_true, 1e-7, None)
    median_ape = np.median(pct_errors) * 100
    mean_ape = np.mean(pct_errors) * 100
    correlation = np.corrcoef(y_val, y_pred)[0, 1]

    final_metrics = {
        "log_mae": round(log_mae, 4),
        "log_rmse": round(log_rmse, 4),
        "log_r2": round(log_r2, 4),
        "bfsp_mae": round(bfsp_mae, 2),
        "median_ape_pct": round(median_ape, 2),
        "mean_ape_pct": round(mean_ape, 2),
        "correlation": round(correlation, 4),
        "best_iteration": model.best_iteration,
        "train_size": len(train_df),
        "val_size": len(val_df),
    }

    log.info(f"\nFinal model metrics:")
    log.info(f"  MAE (log):    {final_metrics['log_mae']}")
    log.info(f"  RMSE (log):   {final_metrics['log_rmse']}")
    log.info(f"  R² (log):     {final_metrics['log_r2']}")
    log.info(f"  MAE (BFSP):   {final_metrics['bfsp_mae']}")
    log.info(f"  MdAPE:        {final_metrics['median_ape_pct']}%")
    log.info(f"  Correlation:  {final_metrics['correlation']}")
    log.info(f"  Best iter:    {final_metrics['best_iteration']}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "bfsp_model.lgb")
    model.save_model(model_path)
    log.info(f"\nModel saved to {model_path}")

    # Update meta
    meta["final_metrics"] = final_metrics
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info(f"Meta updated at {META_PATH}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(MODEL_DIR, "bfsp_feature_importance.csv"), index=False)

    log.info("\nTop 20 Features by Gain:")
    for _, row in imp_df.head(20).iterrows():
        bar_len = int(30 * row["importance"] / imp_df["importance"].max())
        bar = "=" * bar_len
        log.info(f"  {row['feature']:<40s} {bar} {row['importance']:.0f}")

    print(f"\nDone! Model saved to {model_path}")
    print(f"Size: {os.path.getsize(model_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
