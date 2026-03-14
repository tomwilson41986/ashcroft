#!/usr/bin/env python3
"""
Build Train / Validation / Test Datasets and Train BFSP Model.

Loads historical data from horse_racing.db, computes all features (custom
metrics, pace, draw, context), creates deterministic temporal splits, saves
them as Parquet files to S3 for reuse, and trains a LightGBM model.

Temporal split strategy:
    Train:  everything before 2025-07-01
    Val:    2025-07-01 to 2025-12-31
    Test:   2026-01-01+

Usage:
    python build_datasets.py                          # Build, save to S3, train
    python build_datasets.py --no-upload              # Build locally only + train
    python build_datasets.py --from-s3                # Load existing splits from S3 + train
    python build_datasets.py --build-only             # Build + upload, no training
"""

import argparse
import gc
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.custom_metrics import CustomMetricsEngine
from train_bfsp import (
    ALL_FEATURE_COLS,
    BFSPTrainer,
    build_context_features,
    load_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "data", "datasets")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")
S3_BUCKET = "horseracingresults"
S3_PREFIX = "datasets"


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(db_path: str, start_date: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load data and compute all features."""
    log.info(f"Loading data from {db_path}...")
    df = load_data(db_path, start_date=start_date)
    log.info(f"  {len(df):,} rows loaded")

    engine = CustomMetricsEngine()
    log.info("Computing all custom metrics...")
    df = engine.calculate_all(df)

    # Defragment — critical for memory after 300+ column insertions
    log.info("  Defragmenting DataFrame...")
    df = df.copy()
    gc.collect()
    log.info(f"  Custom metrics done: {len(df.columns)} columns")

    log.info("Building context features...")
    df = build_context_features(df)

    # Target
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["log_bfsp"] = np.log(df["bfsp"])

    # Available features
    seen = set()
    feature_cols = []
    for c in ALL_FEATURE_COLS:
        if c in df.columns and c not in seen:
            seen.add(c)
            feature_cols.append(c)

    log.info(f"  {len(feature_cols)} features available")
    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
    return df, feature_cols


# ---------------------------------------------------------------------------
# Dataset splitting & persistence
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    train_cutoff: str = "2025-07-01",
    val_end: str = "2026-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split into train / val / test."""
    tc = pd.Timestamp(train_cutoff)
    ve = pd.Timestamp(val_end)

    train_df = df[df["race_date"] < tc].copy()
    val_df = df[(df["race_date"] >= tc) & (df["race_date"] < ve)].copy()
    test_df = df[df["race_date"] >= ve].copy()

    log.info(f"  Train: {len(train_df):,} rows  (... to {train_cutoff})")
    log.info(f"  Val:   {len(val_df):,} rows  ({train_cutoff} to {val_end})")
    log.info(f"  Test:  {len(test_df):,} rows  ({val_end} to ...)")
    return train_df, val_df, test_df


def save_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str = DATASET_DIR,
) -> dict[str, str]:
    """Save datasets as Parquet + feature list as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(output_dir, f"{name}.parquet")
        split_df.to_parquet(path, index=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        log.info(f"  Saved {name}: {path} ({size_mb:.1f} MB, {len(split_df):,} rows)")
        paths[name] = path

    meta = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_date_range": [
            str(train_df["race_date"].min().date()),
            str(train_df["race_date"].max().date()),
        ],
        "val_date_range": [
            str(val_df["race_date"].min().date()),
            str(val_df["race_date"].max().date()),
        ],
        "test_date_range": [
            str(test_df["race_date"].min().date()),
            str(test_df["race_date"].max().date()),
        ],
        "created_at": datetime.now().isoformat(),
    }
    meta_path = os.path.join(output_dir, "feature_cols.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    paths["meta"] = meta_path
    return paths


def _get_s3_client():
    import boto3
    env = {}
    env_path = os.path.join(SCRIPT_DIR, ".env.local")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return boto3.client(
        "s3",
        aws_access_key_id=env.get("AWS_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID")),
        aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY")),
        region_name=env.get("AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
    )


def upload_to_s3(paths: dict[str, str]):
    """Upload dataset files to S3."""
    s3 = _get_s3_client()
    for name, local_path in paths.items():
        filename = os.path.basename(local_path)
        s3_key = f"{S3_PREFIX}/{filename}"
        log.info(f"  Uploading {filename} -> s3://{S3_BUCKET}/{s3_key}")
        s3.upload_file(local_path, S3_BUCKET, s3_key)
    log.info("  All datasets uploaded to S3")


def download_from_s3(output_dir: str = DATASET_DIR):
    """Download existing datasets from S3."""
    s3 = _get_s3_client()
    os.makedirs(output_dir, exist_ok=True)
    for filename in ["train.parquet", "val.parquet", "test.parquet", "feature_cols.json"]:
        s3_key = f"{S3_PREFIX}/{filename}"
        local_path = os.path.join(output_dir, filename)
        log.info(f"  Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
        s3.download_file(S3_BUCKET, s3_key, local_path)
    log.info("  All datasets downloaded from S3")


def load_datasets(dataset_dir: str = DATASET_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load saved datasets from local Parquet files."""
    train_df = pd.read_parquet(os.path.join(dataset_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(dataset_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(dataset_dir, "test.parquet"))

    with open(os.path.join(dataset_dir, "feature_cols.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    # Ensure race_date is datetime
    for df in [train_df, val_df, test_df]:
        df["race_date"] = pd.to_datetime(df["race_date"])

    log.info(f"  Loaded train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    log.info(f"  {len(feature_cols)} features")
    return train_df, val_df, test_df, feature_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str = MODEL_DIR,
) -> dict:
    """Train LightGBM on train, validate on val, evaluate on test."""
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

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["log_bfsp"].astype(float)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["log_bfsp"].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["log_bfsp"].astype(float)

    log.info(f"Training LightGBM: {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test")
    log.info(f"  {len(feature_cols)} features")

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Evaluate on val and test
    def compute_metrics(y_true, y_pred, label):
        bfsp_true = np.exp(y_true)
        bfsp_pred = np.exp(y_pred)
        log_mae = mean_absolute_error(y_true, y_pred)
        log_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        log_r2 = r2_score(y_true, y_pred)
        bfsp_mae = mean_absolute_error(bfsp_true, bfsp_pred)
        pct_errors = np.abs(bfsp_true - bfsp_pred) / np.clip(bfsp_true, 1e-7, None)
        median_ape = np.median(pct_errors) * 100
        mean_ape = np.mean(pct_errors) * 100
        corr = np.corrcoef(y_true, y_pred)[0, 1]

        metrics = {
            f"{label}_log_mae": round(log_mae, 4),
            f"{label}_log_rmse": round(log_rmse, 4),
            f"{label}_log_r2": round(log_r2, 4),
            f"{label}_bfsp_mae": round(bfsp_mae, 2),
            f"{label}_median_ape_pct": round(median_ape, 2),
            f"{label}_mean_ape_pct": round(mean_ape, 2),
            f"{label}_correlation": round(corr, 4),
        }
        return metrics

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = compute_metrics(y_val.values, val_pred, "val")
    test_metrics = compute_metrics(y_test.values, test_pred, "test")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "bfsp_model.lgb")
    model.save_model(model_path)

    meta = {
        "model_type": "bfsp_regression",
        "target": "log_bfsp",
        "feature_cols": feature_cols,
        "params": params,
        "best_iteration": model.best_iteration,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    meta_path = os.path.join(output_dir, "bfsp_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    imp_path = os.path.join(output_dir, "bfsp_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)

    summary_path = os.path.join(output_dir, "bfsp_training_summary.json")
    summary = {**meta, "top_20_features": imp_df.head(20).to_dict("records")}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print report
    print("\n" + "=" * 70)
    print("BFSP MODEL — TRAINING REPORT")
    print("=" * 70)
    print(f"  Features:       {len(feature_cols)}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Train rows:     {len(train_df):,}")
    print(f"  Val rows:       {len(val_df):,}")
    print(f"  Test rows:      {len(test_df):,}")

    print(f"\n  VALIDATION SET ({val_df['race_date'].min().date()} to {val_df['race_date'].max().date()})")
    print("  " + "-" * 50)
    for k, v in val_metrics.items():
        print(f"  {k:<30s} {v}")

    print(f"\n  TEST SET ({test_df['race_date'].min().date()} to {test_df['race_date'].max().date()})")
    print("  " + "-" * 50)
    for k, v in test_metrics.items():
        print(f"  {k:<30s} {v}")

    print(f"\n  TOP 20 FEATURES (by gain)")
    print("  " + "-" * 50)
    for _, row in imp_df.head(20).iterrows():
        bar_len = int(30 * row["importance"] / imp_df["importance"].max())
        bar = "=" * bar_len
        print(f"  {row['feature']:<40s} {bar} {row['importance']:.0f}")

    print("=" * 70)
    print(f"  Model saved to: {output_dir}")

    return {**val_metrics, **test_metrics, "best_iteration": model.best_iteration}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build datasets + train BFSP model")
    parser.add_argument("--db", default=os.path.join(SCRIPT_DIR, "horse_racing.db"))
    parser.add_argument("--output-dir", default=DATASET_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--train-cutoff", default="2025-07-01")
    parser.add_argument("--val-end", default="2026-01-01")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--from-s3", action="store_true",
                        help="Download existing datasets from S3 and train")
    parser.add_argument("--build-only", action="store_true",
                        help="Build and save datasets without training")
    parser.add_argument("--start-date", default="2022-01-01",
                        help="Only load data from this date (default: 2022-01-01)")
    args = parser.parse_args()

    if args.from_s3:
        log.info("Downloading datasets from S3...")
        download_from_s3(args.output_dir)
        train_df, val_df, test_df, feature_cols = load_datasets(args.output_dir)
        log.info("Training model from saved datasets...")
        train_model(train_df, val_df, test_df, feature_cols, args.model_dir)
        return

    # Build features
    df, feature_cols = build_features(args.db, start_date=args.start_date)

    # Split
    log.info("Splitting data...")
    train_df, val_df, test_df = split_data(df, args.train_cutoff, args.val_end)

    # Free the full dataframe
    del df
    gc.collect()

    # Save datasets
    log.info("Saving datasets...")
    paths = save_datasets(train_df, val_df, test_df, feature_cols, args.output_dir)

    # Upload to S3
    if not args.no_upload:
        log.info("Uploading to S3...")
        upload_to_s3(paths)

    if args.build_only:
        log.info("Done (build only).")
        return

    # Train
    log.info("Training model...")
    train_model(train_df, val_df, test_df, feature_cols, args.model_dir)
    log.info("Done!")


if __name__ == "__main__":
    main()
