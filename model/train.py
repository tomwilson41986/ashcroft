"""
Train and evaluate BFSP prediction model.

Uses a temporal train/test split to prevent data leakage:
  - Train on races before a cutoff date
  - Test on races after the cutoff date

Model: LightGBM regression on log(BFSP).

Usage:
    python -m model.train                           # Train with defaults
    python -m model.train --db horse_racing.db      # Specify database
    python -m model.train --cutoff 2024-01-01       # Custom train/test split date
    python -m model.train --importance              # Show feature importance
"""

import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.features import FEATURE_COLS, TARGET_COL, build_features, load_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
MODEL_PATH = os.path.join(PROJECT_DIR, "bfsp_model.lgb")
METRICS_PATH = os.path.join(PROJECT_DIR, "model_metrics.json")


def temporal_split(
    df: pd.DataFrame, cutoff_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by date. Train on everything before cutoff, test on after."""
    cutoff = pd.Timestamp(cutoff_date)
    train = df[df["race_date"] < cutoff].copy()
    test = df[df["race_date"] >= cutoff].copy()
    return train, test


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> lgb.Booster:
    """Train LightGBM model."""
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_valid = valid_df[FEATURE_COLS]
    y_valid = valid_df[TARGET_COL]

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    return model


def evaluate(
    model: lgb.Booster, test_df: pd.DataFrame
) -> dict:
    """Evaluate model on test set. Returns metrics dict."""
    X_test = test_df[FEATURE_COLS]
    y_true = test_df[TARGET_COL].values
    y_pred = model.predict(X_test)

    # Metrics on log scale
    mae_log = mean_absolute_error(y_true, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_log = r2_score(y_true, y_pred)

    # Convert back to actual BFSP for interpretable metrics
    bfsp_true = np.exp(y_true)
    bfsp_pred = np.exp(y_pred)
    mae_bfsp = mean_absolute_error(bfsp_true, bfsp_pred)
    # Median absolute percentage error (more robust than MAPE for odds)
    pct_errors = np.abs(bfsp_true - bfsp_pred) / bfsp_true
    median_ape = np.median(pct_errors) * 100

    metrics = {
        "test_samples": len(y_true),
        "log_scale": {
            "mae": round(mae_log, 4),
            "rmse": round(rmse_log, 4),
            "r2": round(r2_log, 4),
        },
        "bfsp_scale": {
            "mae": round(mae_bfsp, 2),
            "median_ape_pct": round(median_ape, 2),
        },
    }
    return metrics


def show_feature_importance(model: lgb.Booster, top_n: int = 20):
    """Print top feature importances."""
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    pairs = sorted(zip(feature_names, importance), key=lambda x: -x[1])

    print(f"\n  Top {top_n} Features by Gain")
    print("  " + "=" * 50)
    max_imp = pairs[0][1] if pairs else 1
    for name, imp in pairs[:top_n]:
        bar_len = int(30 * imp / max_imp)
        bar = "█" * bar_len
        print(f"  {name:<30s} {bar} {imp:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Train BFSP prediction model")
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB, help="Path to SQLite database"
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Train/test cutoff date (YYYY-MM-DD). Default: last 20%% by date",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Show feature importance after training",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save trained model to disk (default: True)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        print("Run scraper.py first to collect data.")
        sys.exit(1)

    # Load and build features
    print("Loading data...")
    df = load_data(args.db)
    print(f"  {len(df):,} raw rows loaded")

    print("Building features...")
    df = build_features(df)
    print(f"  {len(df):,} rows with valid BFSP")

    # Determine cutoff date
    if args.cutoff:
        cutoff = args.cutoff
    else:
        # Use last 20% of dates as test set
        dates = df["race_date"].sort_values().unique()
        cutoff_idx = int(len(dates) * 0.8)
        cutoff = str(dates[cutoff_idx])[:10]

    print(f"  Train/test cutoff: {cutoff}")

    # Split
    train_df, test_df = temporal_split(df, cutoff)

    # Drop rows with missing target
    train_df = train_df.dropna(subset=[TARGET_COL])
    test_df = test_df.dropna(subset=[TARGET_COL])

    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    if len(train_df) < 100:
        print("Not enough training data. Need at least 100 rows.")
        sys.exit(1)

    if len(test_df) < 10:
        print("Not enough test data. Adjust --cutoff to an earlier date.")
        sys.exit(1)

    # Train
    print("\nTraining LightGBM model...")
    model = train_model(train_df, test_df)

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate(model, test_df)

    print(f"\n  Results ({metrics['test_samples']:,} test samples)")
    print("  " + "=" * 40)
    print(f"  Log-scale MAE:      {metrics['log_scale']['mae']:.4f}")
    print(f"  Log-scale RMSE:     {metrics['log_scale']['rmse']:.4f}")
    print(f"  Log-scale R²:       {metrics['log_scale']['r2']:.4f}")
    print(f"  BFSP MAE:           {metrics['bfsp_scale']['mae']:.2f}")
    print(f"  Median APE:         {metrics['bfsp_scale']['median_ape_pct']:.1f}%")

    if args.importance:
        show_feature_importance(model)

    # Save model and metrics
    if args.save:
        model.save_model(MODEL_PATH)
        print(f"\n  Model saved to: {MODEL_PATH}")

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to: {METRICS_PATH}")

    return model, metrics


if __name__ == "__main__":
    main()
