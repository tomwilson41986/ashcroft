"""
Model Training with Walk-Forward Validation.

Trains the full prediction pipeline using strict temporal ordering.
No random splits, no shuffling — simulates real-world deployment
where we can only use past data to predict future races.

Usage:
    python -m model.trainer                           # Train with defaults
    python -m model.trainer --db horse_racing.db      # Specify database
    python -m model.trainer --min-train-days 365      # Minimum training window
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from model.benter_blend import BenterBlender
from model.custom_metrics import CustomMetricsEngine
from model.evaluator import ModelEvaluator
from model.prerace_builder import PreRaceBuilder
from model.probability_model import FundamentalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")


class ModelTrainer:
    """Train and validate the prediction model using strict temporal ordering.

    Walk-forward approach:
    1. Sort all data by date
    2. Define an expanding training window
    3. For each validation fold, train on all data BEFORE the fold date
    4. Predict on the fold period
    5. Evaluate predictions against actual results

    This simulates real-world deployment where we can only use past data.
    """

    def __init__(
        self,
        min_train_days: int = 365,
        val_window_days: int = 30,
        step_days: int = 30,
        metrics_engine: CustomMetricsEngine | None = None,
    ):
        self.min_train_days = min_train_days
        self.val_window_days = val_window_days
        self.step_days = step_days
        self.metrics_engine = metrics_engine or CustomMetricsEngine()
        self.blender = BenterBlender()
        self.evaluator = ModelEvaluator()

    def create_folds(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Create temporal train/validation folds.

        Returns list of (train_end_date, val_start_date, val_end_date) tuples.
        """
        df = df.sort_values("race_date")
        min_date = df["race_date"].min()
        max_date = df["race_date"].max()

        folds = []
        train_end = min_date + timedelta(days=self.min_train_days)

        while train_end + timedelta(days=self.val_window_days) <= max_date:
            val_start = train_end
            val_end = val_start + timedelta(days=self.val_window_days)
            folds.append((train_end, val_start, val_end))
            train_end += timedelta(days=self.step_days)

        return folds

    def train(
        self,
        full_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        output_dir: str = MODEL_DIR,
    ) -> dict:
        """Full training pipeline with walk-forward validation.

        Steps:
        1. Calculate custom metrics on the full dataset
        2. Create walk-forward folds
        3. For each fold: train, predict, blend, evaluate
        4. Retrain final model on all data
        5. Save model artifacts

        Args:
            full_df: Complete historical race data (raw, from DB).
            feature_cols: Feature columns to use (auto-detected if None).
            output_dir: Where to save model artifacts.

        Returns:
            Dict of aggregated metrics across all folds.
        """
        log.info("Step 1: Calculating custom metrics...")
        df = self.metrics_engine.calculate_all(full_df)
        log.info(f"  {len(df):,} rows with metrics calculated")

        # Ensure we have targets
        df["won"] = (df["placing_numerical"] == 1).astype(float)

        # Determine feature columns
        if feature_cols is None:
            builder = PreRaceBuilder(df)
            feature_cols = builder.get_feature_columns()

        # Filter to available columns and deduplicate
        available_features = list(dict.fromkeys(c for c in feature_cols if c in df.columns))
        log.info(f"  Using {len(available_features)} features")

        # Create folds
        folds = self.create_folds(df)
        log.info(f"Step 2: Created {len(folds)} walk-forward folds")

        if not folds:
            log.warning("Not enough data for walk-forward validation.")
            log.info("Training on 80/20 temporal split instead.")
            return self._train_simple_split(df, available_features, output_dir)

        # Walk-forward validation
        all_predictions = []
        fold_metrics = []

        for i, (train_end, val_start, val_end) in enumerate(folds):
            log.info(
                f"  Fold {i + 1}/{len(folds)}: "
                f"train < {train_end.date()}, "
                f"val {val_start.date()} to {val_end.date()}"
            )

            train_df = df[df["race_date"] < train_end].copy()
            val_df = df[
                (df["race_date"] >= val_start)
                & (df["race_date"] < val_end)
            ].copy()

            if len(train_df) < 100 or len(val_df) < 10:
                log.warning(f"  Skipping fold {i + 1}: insufficient data")
                continue

            # Train fundamental model
            model = FundamentalModel()
            metrics = model.train(
                train_df, val_df, available_features, target_col="won"
            )

            # Predict on validation set
            val_with_preds = model.predict(val_df)

            # Blend with market prices
            if "bfsp" in val_with_preds.columns:
                val_with_preds = self.blender.blend(val_with_preds)

            all_predictions.append(val_with_preds)
            fold_metrics.append(metrics)

            log.info(
                f"    Log-loss: {metrics['val_logloss_normalised']:.4f}"
            )

        # Aggregate results
        if all_predictions:
            combined_preds = pd.concat(all_predictions, ignore_index=True)
        else:
            combined_preds = pd.DataFrame()

        # Step 3: Optimise blending lambda on last fold
        if len(all_predictions) > 0 and "bfsp" in combined_preds.columns:
            log.info("Step 3: Optimising blending lambda...")
            optimal_lambda = self.blender.optimise_lambda(combined_preds)
            log.info(f"  Optimal lambda: {optimal_lambda}")
        else:
            optimal_lambda = 0.80

        # Step 4: Retrain final model on ALL data
        log.info("Step 4: Training final model on all data...")
        dates = df["race_date"].sort_values().unique()
        cutoff_idx = int(len(dates) * 0.9)
        cutoff = dates[cutoff_idx]

        final_train = df[df["race_date"] < cutoff].copy()
        final_val = df[df["race_date"] >= cutoff].copy()

        final_model = FundamentalModel()
        final_metrics = final_model.train(
            final_train, final_val, available_features, target_col="won"
        )

        # Step 5: Evaluate on walk-forward predictions
        eval_results = {}
        if len(combined_preds) > 0:
            log.info("Step 5: Evaluating walk-forward predictions...")
            eval_results = self.evaluator.evaluate(combined_preds)

        # Save artifacts
        log.info(f"Step 6: Saving model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        final_model.save(output_dir)

        # Save blender config
        blend_config = {
            "optimal_lambda": optimal_lambda,
        }
        with open(os.path.join(output_dir, "blend_config.json"), "w") as f:
            json.dump(blend_config, f, indent=2)

        # Save training summary
        summary = {
            "n_folds": len(fold_metrics),
            "feature_cols": available_features,
            "final_model_metrics": final_metrics,
            "walk_forward_metrics": fold_metrics,
            "optimal_lambda": optimal_lambda,
            "evaluation": eval_results,
            "data_range": {
                "min_date": str(df["race_date"].min().date()),
                "max_date": str(df["race_date"].max().date()),
                "total_rows": len(df),
            },
        }
        with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        log.info("Training complete!")
        log.info(f"  Final log-loss: {final_metrics['val_logloss_normalised']:.4f}")
        if eval_results:
            log.info(f"  Walk-forward log-loss: {eval_results.get('log_loss', 'N/A')}")

        return summary

    def _train_simple_split(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        output_dir: str,
    ) -> dict:
        """Fallback: simple 80/20 temporal split when insufficient data for walk-forward."""
        dates = df["race_date"].sort_values().unique()
        cutoff_idx = int(len(dates) * 0.8)
        cutoff = dates[cutoff_idx]

        train_df = df[df["race_date"] < cutoff].copy()
        val_df = df[df["race_date"] >= cutoff].copy()

        log.info(f"  Train: {len(train_df):,} rows (< {cutoff.date()})")
        log.info(f"  Val: {len(val_df):,} rows (>= {cutoff.date()})")

        model = FundamentalModel()
        metrics = model.train(train_df, val_df, feature_cols, target_col="won")

        os.makedirs(output_dir, exist_ok=True)
        model.save(output_dir)

        summary = {
            "n_folds": 0,
            "split_type": "simple_temporal_80_20",
            "final_model_metrics": metrics,
            "feature_cols": feature_cols,
        }
        with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def tune_hyperparameters(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict:
        """Tune model hyperparameters on validation fold.

        Grid search over key LightGBM parameters, optimising log-loss.

        Returns:
            Best hyperparameter configuration.
        """
        from sklearn.metrics import log_loss as sk_log_loss

        param_grid = {
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_child_samples": [20, 50, 100],
            "feature_fraction": [0.7, 0.8, 0.9],
            "bagging_fraction": [0.7, 0.8, 0.9],
        }

        best_ll = float("inf")
        best_params = {}

        # Simple grid search (subset of combinations for efficiency)
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            params.update(
                {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "boosting_type": "gbdt",
                    "bagging_freq": 5,
                    "lambda_l1": 0.1,
                    "lambda_l2": 0.1,
                    "verbose": -1,
                    "is_unbalance": True,
                }
            )

            model = FundamentalModel(params=params)
            try:
                model.train(
                    train_df,
                    val_df,
                    feature_cols,
                    num_boost_round=500,
                    early_stopping=30,
                )
                preds = model.predict(val_df)
                ll = sk_log_loss(
                    val_df["won"],
                    preds["p_model"].clip(1e-7, 1 - 1e-7),
                )
                if ll < best_ll:
                    best_ll = ll
                    best_params = params
            except Exception:
                continue

        log.info(f"  Best hyperparameters (log-loss={best_ll:.4f}):")
        for k, v in best_params.items():
            if k not in ["objective", "metric", "verbose", "boosting_type"]:
                log.info(f"    {k}: {v}")

        return best_params


def load_data(db_path: str) -> pd.DataFrame:
    """Load race results from SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Train predictive model with walk-forward validation"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=MODEL_DIR,
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=365,
        help="Minimum training window in days",
    )
    parser.add_argument(
        "--val-window",
        type=int,
        default=30,
        help="Validation window in days",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=30,
        help="Step between folds in days",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (slow)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        print("Run scraper.py first to collect data.")
        sys.exit(1)

    log.info("Loading data...")
    df = load_data(args.db)
    log.info(f"  {len(df):,} rows loaded")

    trainer = ModelTrainer(
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
    )

    summary = trainer.train(df, output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Folds: {summary.get('n_folds', 0)}")
    if "final_model_metrics" in summary:
        fm = summary["final_model_metrics"]
        print(f"  Final log-loss: {fm.get('val_logloss_normalised', 'N/A')}")
    if "optimal_lambda" in summary:
        print(f"  Optimal lambda: {summary['optimal_lambda']}")
    print(f"  Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
