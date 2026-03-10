#!/usr/bin/env python3
"""
Retrain Model on Latest Data with Accuracy Comparison.

Loads the latest data from the database, trains a new model using the
full 3-stage pipeline, and compares accuracy metrics against the
previous model (if one exists).

Usage:
    python retrain.py                              # Train with defaults
    python retrain.py --db horse_racing.db         # Specify database
    python retrain.py --generate-sample            # Generate sample data first
    python retrain.py --min-train-days 180         # Shorter training window
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from model.benter_blend import BenterBlender
from model.custom_metrics import CustomMetricsEngine
from model.evaluator import ModelEvaluator
from model.probability_model import FundamentalModel
from model.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")
BACKUP_DIR = os.path.join(SCRIPT_DIR, "data", "models_backup")


def load_previous_summary(model_dir: str) -> dict | None:
    """Load training summary from a previous model run."""
    summary_path = os.path.join(model_dir, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return None


def backup_previous_model(model_dir: str, backup_dir: str):
    """Back up the current model before retraining."""
    if not os.path.exists(model_dir):
        return

    model_file = os.path.join(model_dir, "probability_model.lgb")
    if not os.path.exists(model_file):
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = f"{backup_dir}_{timestamp}"
    shutil.copytree(model_dir, dest)
    log.info(f"Previous model backed up to {dest}")


def load_data(db_path: str) -> pd.DataFrame:
    """Load race results from SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def format_metric(value, fmt=".4f"):
    """Format a metric value for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:{fmt}}"


def print_comparison(old_summary: dict | None, new_summary: dict):
    """Print a side-by-side comparison of old vs new model metrics."""
    sep = "=" * 72
    thin_sep = "-" * 72

    print(f"\n{sep}")
    print("  MODEL ACCURACY COMPARISON: Old vs New")
    print(sep)

    # Data range
    new_range = new_summary.get("data_range", {})
    print(f"\n  New model data range: {new_range.get('min_date', '?')} "
          f"to {new_range.get('max_date', '?')}")
    print(f"  Total rows: {new_range.get('total_rows', '?'):,}")

    if old_summary and "data_range" in old_summary:
        old_range = old_summary["data_range"]
        print(f"  Old model data range: {old_range.get('min_date', '?')} "
              f"to {old_range.get('max_date', '?')}")
        print(f"  Old total rows: {old_range.get('total_rows', '?'):,}")

    # Core model metrics comparison
    print(f"\n{thin_sep}")
    print(f"  {'METRIC':<40} {'OLD':>12} {'NEW':>12} {'CHANGE':>10}")
    print(thin_sep)

    old_fm = old_summary.get("final_model_metrics", {}) if old_summary else {}
    new_fm = new_summary.get("final_model_metrics", {})

    core_metrics = [
        ("Log-Loss (normalised)", "val_logloss_normalised", ".6f", True),
        ("Log-Loss (raw)", "val_logloss_raw", ".6f", True),
        ("Best Iteration", "best_iteration", "d", False),
        ("Num Features", "n_features", "d", False),
        ("Train Size", "train_size", ",d", False),
        ("Val Size", "val_size", ",d", False),
    ]

    for label, key, fmt, lower_is_better in core_metrics:
        old_val = old_fm.get(key)
        new_val = new_fm.get(key)
        old_str = format_metric(old_val, fmt) if old_val is not None else "N/A"
        new_str = format_metric(new_val, fmt) if new_val is not None else "N/A"

        change_str = ""
        if old_val is not None and new_val is not None:
            diff = new_val - old_val
            if isinstance(diff, float):
                arrow = "v" if (diff < 0 and lower_is_better) or (diff > 0 and not lower_is_better) else "^" if diff != 0 else "="
                if lower_is_better:
                    indicator = " BETTER" if diff < 0 else " WORSE" if diff > 0 else ""
                else:
                    indicator = " BETTER" if diff > 0 else " WORSE" if diff < 0 else ""
                change_str = f"{diff:+.4f}{indicator}"
            else:
                change_str = f"{diff:+d}"

        print(f"  {label:<40} {old_str:>12} {new_str:>12} {change_str:>10}")

    # Walk-forward evaluation metrics
    old_eval = old_summary.get("evaluation", {}) if old_summary else {}
    new_eval = new_summary.get("evaluation", {})

    if new_eval:
        print(f"\n{thin_sep}")
        print("  WALK-FORWARD EVALUATION")
        print(thin_sep)

        eval_metrics = [
            ("WF Log-Loss", "log_loss", ".6f", True),
            ("WF Brier Score", "brier_score", ".6f", True),
            ("AUC-ROC", "auc_roc", ".4f", False),
            ("Top-1 Accuracy", "top_1_accuracy", ".4f", False),
            ("Top-2 Accuracy", "top_2_accuracy", ".4f", False),
            ("Top-3 Accuracy", "top_3_accuracy", ".4f", False),
            ("Favourite Accuracy", "favourite_accuracy", ".4f", False),
            ("N Predictions", "n_predictions", ",d", False),
            ("N Winners", "n_winners", ",d", False),
            ("Base Rate", "base_rate", ".4f", False),
        ]

        for label, key, fmt, lower_is_better in eval_metrics:
            old_val = old_eval.get(key)
            new_val = new_eval.get(key)
            old_str = format_metric(old_val, fmt) if old_val is not None else "N/A"
            new_str = format_metric(new_val, fmt) if new_val is not None else "N/A"

            change_str = ""
            if old_val is not None and new_val is not None:
                diff = new_val - old_val
                if isinstance(diff, float):
                    if lower_is_better:
                        indicator = " BETTER" if diff < 0 else " WORSE" if diff > 0 else ""
                    else:
                        indicator = " BETTER" if diff > 0 else " WORSE" if diff < 0 else ""
                    change_str = f"{diff:+.4f}{indicator}"

            print(f"  {label:<40} {old_str:>12} {new_str:>12} {change_str:>10}")

    # Market comparison
    if new_eval and "market_logloss" in new_eval:
        print(f"\n{thin_sep}")
        print("  MARKET BASELINE COMPARISON")
        print(thin_sep)

        market_metrics = [
            ("Market Log-Loss", "market_logloss", ".6f"),
            ("Model vs Market Improvement", "model_vs_market_improvement", ".6f"),
            ("Random Log-Loss", "random_logloss", ".6f"),
        ]

        for label, key, fmt in market_metrics:
            old_val = old_eval.get(key)
            new_val = new_eval.get(key)
            old_str = format_metric(old_val, fmt) if old_val is not None else "N/A"
            new_str = format_metric(new_val, fmt) if new_val is not None else "N/A"
            print(f"  {label:<40} {old_str:>12} {new_str:>12}")

    # Betting simulation
    if new_eval and "betting_n_bets" in new_eval:
        print(f"\n{thin_sep}")
        print("  BETTING SIMULATION")
        print(thin_sep)

        betting_metrics = [
            ("Total Bets", "betting_n_bets", ",d"),
            ("Winners", "betting_n_winners", ",d"),
            ("Strike Rate", "betting_strike_rate", ".4f"),
            ("Total Staked", "betting_total_staked", ".2f"),
            ("Total P&L", "betting_total_pl", ".2f"),
            ("ROI %", "betting_roi_pct", ".2f"),
            ("Max Drawdown", "max_drawdown", ".2f"),
            ("Sharpe Ratio", "sharpe_ratio", ".4f"),
        ]

        for label, key, fmt in betting_metrics:
            old_val = old_eval.get(key)
            new_val = new_eval.get(key)
            old_str = format_metric(old_val, fmt) if old_val is not None else "N/A"
            new_str = format_metric(new_val, fmt) if new_val is not None else "N/A"
            print(f"  {label:<40} {old_str:>12} {new_str:>12}")

    # Lambda comparison
    old_lambda = old_summary.get("optimal_lambda") if old_summary else None
    new_lambda = new_summary.get("optimal_lambda")
    print(f"\n{thin_sep}")
    print("  BLENDING CONFIGURATION")
    print(thin_sep)
    old_str = format_metric(old_lambda, ".2f") if old_lambda is not None else "N/A"
    new_str = format_metric(new_lambda, ".2f") if new_lambda is not None else "N/A"
    print(f"  {'Optimal Lambda':<40} {old_str:>12} {new_str:>12}")

    # Fold summary
    old_folds = old_summary.get("n_folds", 0) if old_summary else 0
    new_folds = new_summary.get("n_folds", 0)
    print(f"  {'Walk-Forward Folds':<40} {old_folds:>12} {new_folds:>12}")

    # Calibration table
    if new_eval and "calibration_table" in new_eval:
        print(f"\n{thin_sep}")
        print("  CALIBRATION TABLE (New Model)")
        print(thin_sep)
        print(f"  {'Bin':<30} {'Count':>8} {'Predicted':>12} {'Actual':>12}")
        print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*12}")
        for row in new_eval["calibration_table"]:
            print(f"  {row['bin']:<30} {row['count']:>8} "
                  f"{row['avg_predicted']:>12.4f} {row['actual_win_rate']:>12.4f}")

    print(f"\n{sep}")

    # Overall verdict
    if old_fm and new_fm:
        old_ll = old_fm.get("val_logloss_normalised")
        new_ll = new_fm.get("val_logloss_normalised")
        if old_ll and new_ll:
            diff = new_ll - old_ll
            pct = (diff / old_ll) * 100 if old_ll != 0 else 0
            if diff < 0:
                print(f"  VERDICT: New model IMPROVED log-loss by {abs(diff):.6f} "
                      f"({abs(pct):.2f}%)")
            elif diff > 0:
                print(f"  VERDICT: New model REGRESSED log-loss by {diff:.6f} "
                      f"({pct:.2f}%)")
            else:
                print("  VERDICT: No change in model log-loss")
    else:
        print("  VERDICT: First training run — no previous model to compare")

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain model on latest data with accuracy comparison"
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir", type=str, default=MODEL_DIR,
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--min-train-days", type=int, default=365,
        help="Minimum training window in days",
    )
    parser.add_argument(
        "--val-window", type=int, default=30,
        help="Validation window in days",
    )
    parser.add_argument(
        "--step-days", type=int, default=30,
        help="Step between folds in days",
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate sample data for testing (if no DB exists)",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip backing up the previous model",
    )
    args = parser.parse_args()

    # Generate sample data if requested or no DB exists
    if args.generate_sample or not os.path.exists(args.db):
        if not os.path.exists(args.db):
            log.info("No database found. Generating sample data...")
        from generate_sample_data import generate_sample_database
        generate_sample_database(args.db)

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        print("Run scraper.py first to collect data, or use --generate-sample.")
        sys.exit(1)

    # Load previous model summary for comparison
    log.info("Checking for previous model...")
    old_summary = load_previous_summary(args.output_dir)
    if old_summary:
        log.info("  Found previous model — will compare after retraining")
        if not args.no_backup:
            backup_previous_model(args.output_dir, BACKUP_DIR)
    else:
        log.info("  No previous model found — this is a fresh training run")

    # Load data
    log.info("Loading data from database...")
    df = load_data(args.db)
    log.info(f"  {len(df):,} rows loaded")
    log.info(f"  Date range: {df['race_date'].min().date()} to {df['race_date'].max().date()}")

    # Train new model
    log.info("Starting retraining pipeline...")
    trainer = ModelTrainer(
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
    )

    new_summary = trainer.train(df, output_dir=args.output_dir)

    # Print comparison
    print_comparison(old_summary, new_summary)

    # Save comparison report
    comparison = {
        "retrain_date": datetime.now().isoformat(),
        "old_summary": old_summary,
        "new_summary": new_summary,
    }
    comparison_path = os.path.join(args.output_dir, "retrain_comparison.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    log.info(f"Comparison report saved to {comparison_path}")


if __name__ == "__main__":
    main()
