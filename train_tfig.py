#!/usr/bin/env python3
"""
Train Time Figure (TFig) Prediction Model.

Trains a LightGBM regression model to predict per-horse Time Figures
using the full suite of custom racing metrics. TFig measures how fast
a horse ran relative to the standard time for its track+distance+going
combination, adjusted by lengths beaten.

TFig derivation:
    1. horse_time = race_comptime + lengths_beaten * 0.2 (secs/length)
    2. standard_time = median(horse_time) per track+distance+going
    3. TFig = (standard_time - horse_time) / standard_time * 100
    Positive TFig = faster than standard.

Pipeline:
    1. Load data from SQLite database
    2. Calculate all custom metrics (same as BFSP model)
    3. Derive TFig target from comptime + lengths beaten
    4. Train LightGBM regression on TFig with walk-forward validation
    5. Evaluate and save model artifacts

Usage:
    python train_tfig.py                          # Train with defaults
    python train_tfig.py --db horse_racing.db     # Specify database
    python train_tfig.py --min-train-days 730     # 2-year minimum window
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.custom_metrics import CustomMetricsEngine
from model.draw_metrics import ALL_DRAW_FEATURES
from model.pace_metrics import ALL_PACE_FEATURES
from train_bfsp import (
    ALL_FEATURE_COLS,
    build_context_features,
    load_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

SECS_PER_LENGTH = 0.2  # Standard: 1 length ≈ 0.2 seconds (flat racing)


# ---------------------------------------------------------------------------
# TFig derivation
# ---------------------------------------------------------------------------

def parse_lengths(val) -> float:
    """Parse beaten-length strings to numeric."""
    if pd.isna(val) or val == "" or val == "0":
        return 0.0
    s = str(val).strip().lower()
    if s in ("dht", "dh"):
        return 0.0
    if s == "nse":
        return 0.05
    if s == "shd":
        return 0.1
    if s in ("hd", "sht-hd"):
        return 0.15
    if s in ("nk", "snk"):
        return 0.2
    if s == "dist":
        return 30.0
    m = re.match(r"(\d+\.?\d*)\s*(nk|shd|hd|nse)?", s)
    if m:
        base = float(m.group(1))
        frac = m.group(2)
        if frac == "nk":
            base += 0.2
        elif frac == "shd":
            base += 0.1
        elif frac == "hd":
            base += 0.15
        elif frac == "nse":
            base += 0.05
        return base
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def derive_tfig(df: pd.DataFrame, min_group_size: int = 30) -> pd.DataFrame:
    """Derive per-horse Time Figure from race comptime + lengths beaten.

    Args:
        df: DataFrame with comptime_numeric, total_dst_bt, track,
            dist_furlongs, going_description columns.
        min_group_size: Minimum number of observations per
            track+distance+going group for reliable standard times.

    Returns:
        DataFrame with 'TFig' column added.
    """
    # Parse lengths beaten
    df["_lb_tfig"] = df["total_dst_bt"].apply(parse_lengths)

    # Individual horse time estimate
    comptime = pd.to_numeric(df["comptime_numeric"], errors="coerce")
    df["_horse_time"] = comptime + df["_lb_tfig"] * SECS_PER_LENGTH

    # Filter invalid
    invalid = (
        comptime.isna()
        | (comptime <= 10)
        | df["_lb_tfig"].isna()
        | (df["_lb_tfig"] >= 30)  # "dist" finishers
        | df["dist_furlongs"].isna()
        | (df["dist_furlongs"] <= 0)
    )

    # Standard time per track+distance+going (median)
    df["_going_lower"] = df["going_description"].fillna("unknown").str.lower().str.strip()
    df["_dist_round"] = df["dist_furlongs"].round(0)
    df["_track_lower"] = df["track"].fillna("unknown").str.lower().str.strip()

    grp_key = ["_track_lower", "_dist_round", "_going_lower"]
    std_times = df.groupby(grp_key)["_horse_time"].transform("median")
    std_counts = df.groupby(grp_key)["_horse_time"].transform("count")

    # TFig: positive = faster than standard
    df["TFig"] = (std_times - df["_horse_time"]) / std_times.replace(0, np.nan) * 100

    # Null out unreliable
    df.loc[invalid, "TFig"] = np.nan
    df.loc[std_counts < min_group_size, "TFig"] = np.nan

    # Cleanup
    df.drop(
        columns=["_lb_tfig", "_horse_time", "_going_lower", "_dist_round", "_track_lower"],
        errors="ignore",
        inplace=True,
    )

    return df


# ---------------------------------------------------------------------------
# TFig Trainer
# ---------------------------------------------------------------------------

class TFigTrainer:
    """Train a Time Figure prediction model using all custom metrics.

    Uses LightGBM regression on TFig with walk-forward temporal validation.
    Same features as the BFSP model but different target.
    """

    DEFAULT_PARAMS = {
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

    def __init__(
        self,
        min_train_days: int = 365,
        val_window_days: int = 30,
        step_days: int = 30,
        params: dict | None = None,
    ):
        self.min_train_days = min_train_days
        self.val_window_days = val_window_days
        self.step_days = step_days
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.metrics_engine = CustomMetricsEngine()
        self.model: lgb.Booster | None = None
        self.feature_cols: list[str] = []

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full data preparation: custom metrics + context + TFig target."""
        log.info("Calculating all custom metrics...")
        df = self.metrics_engine.calculate_all(df)
        log.info(f"  Custom metrics done: {len(df.columns)} columns")

        log.info("Building context features...")
        df = build_context_features(df)

        log.info("Deriving Time Figure (TFig)...")
        df = derive_tfig(df)

        # Filter to rows with valid TFig
        n_before = len(df)
        df = df[df["TFig"].notna()].copy()
        log.info(f"  TFig derived: {len(df):,} valid rows ({len(df)/n_before*100:.1f}% of total)")

        # TFig distribution
        log.info(
            f"  TFig stats: mean={df['TFig'].mean():.3f}, "
            f"std={df['TFig'].std():.3f}, "
            f"min={df['TFig'].min():.1f}, max={df['TFig'].max():.1f}"
        )

        # Determine available features
        available = [c for c in ALL_FEATURE_COLS if c in df.columns]
        seen = set()
        unique_features = []
        for c in available:
            if c not in seen:
                seen.add(c)
                unique_features.append(c)
        self.feature_cols = unique_features
        log.info(f"  {len(self.feature_cols)} features available for training")

        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
        return df

    def create_folds(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Create temporal walk-forward folds."""
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

    def train_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_boost_round: int = 2000,
        early_stopping: int = 50,
    ) -> tuple[lgb.Booster, dict]:
        """Train a single fold and return model + metrics."""
        X_train = train_df[self.feature_cols].astype(float)
        y_train = train_df["TFig"].astype(float)
        X_val = val_df[self.feature_cols].astype(float)
        y_val = val_df["TFig"].astype(float)

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        callbacks = [
            lgb.log_evaluation(period=0),
            lgb.early_stopping(stopping_rounds=early_stopping),
        ]

        model = lgb.train(
            self.params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        y_pred = model.predict(X_val)
        metrics = self._compute_metrics(y_val.values, y_pred)
        metrics["best_iteration"] = model.best_iteration
        metrics["train_size"] = len(train_df)
        metrics["val_size"] = len(val_df)

        return model, metrics

    def train(
        self,
        df: pd.DataFrame,
        output_dir: str = MODEL_DIR,
    ) -> dict:
        """Full training pipeline with walk-forward validation."""
        # Step 1: Prepare data
        df = self.prepare_data(df)
        log.info(f"Prepared {len(df):,} rows with valid TFig")

        if len(df) < 200:
            log.error("Insufficient data for training (need >= 200 rows)")
            return {"error": "insufficient_data"}

        # Step 2: Walk-forward validation
        folds = self.create_folds(df)
        log.info(f"Created {len(folds)} walk-forward folds")

        if not folds:
            log.warning("Not enough data for walk-forward. Using 80/20 split.")
            return self._train_simple_split(df, output_dir)

        fold_metrics = []
        all_val_preds = []

        wf_patience = 10
        best_running_mae = float("inf")
        stale_count = 0

        for i, (train_end, val_start, val_end) in enumerate(folds):
            log.info(
                f"  Fold {i + 1}/{len(folds)}: "
                f"train < {train_end.date()}, "
                f"val {val_start.date()} to {val_end.date()}"
            )

            train_df = df[df["race_date"] < train_end].copy()
            val_df = df[
                (df["race_date"] >= val_start) & (df["race_date"] < val_end)
            ].copy()

            if len(train_df) < 100 or len(val_df) < 10:
                log.warning(f"  Skipping fold {i + 1}: insufficient data")
                continue

            fold_model, metrics = self.train_fold(train_df, val_df)
            fold_metrics.append(metrics)

            # Collect validation predictions
            X_val = val_df[self.feature_cols].astype(float)
            val_df = val_df.copy()
            val_df["predicted_TFig"] = fold_model.predict(X_val)
            all_val_preds.append(val_df)

            log.info(
                f"    MAE: {metrics['mae']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}, "
                f"R²: {metrics['r2']:.4f}, "
                f"Corr: {metrics['correlation']:.4f}"
            )

            # Walk-forward early stopping
            if len(fold_metrics) >= 5:
                recent_mae = np.mean([m["mae"] for m in fold_metrics[-5:]])
                if recent_mae < best_running_mae - 0.001:
                    best_running_mae = recent_mae
                    stale_count = 0
                else:
                    stale_count += 1
                if stale_count >= wf_patience:
                    log.info(
                        f"  Walk-forward early stopping at fold {i + 1}: "
                        f"no improvement for {wf_patience} folds"
                    )
                    break

        # Aggregate walk-forward metrics
        if fold_metrics:
            avg_metrics = {
                "wf_mae": round(np.mean([m["mae"] for m in fold_metrics]), 4),
                "wf_rmse": round(np.mean([m["rmse"] for m in fold_metrics]), 4),
                "wf_r2": round(np.mean([m["r2"] for m in fold_metrics]), 4),
                "wf_correlation": round(
                    np.mean([m["correlation"] for m in fold_metrics]), 4
                ),
                "n_folds": len(fold_metrics),
            }
        else:
            avg_metrics = {"n_folds": 0}

        # Overall walk-forward evaluation
        overall_wf_metrics = {}
        if all_val_preds:
            combined = pd.concat(all_val_preds, ignore_index=True)
            overall_wf_metrics = self._compute_metrics(
                combined["TFig"].values,
                combined["predicted_TFig"].values,
            )
            overall_wf_metrics = {
                f"overall_wf_{k}": v for k, v in overall_wf_metrics.items()
            }
            log.info(
                f"\nOverall walk-forward: "
                f"MAE={overall_wf_metrics['overall_wf_mae']:.4f}, "
                f"R²={overall_wf_metrics['overall_wf_r2']:.4f}, "
                f"Corr={overall_wf_metrics['overall_wf_correlation']:.4f}"
            )

        # Step 3: Train final model
        log.info("\nTraining final model on all data...")
        dates = df["race_date"].sort_values().unique()
        cutoff_idx = int(len(dates) * 0.9)
        cutoff = dates[cutoff_idx]

        final_train = df[df["race_date"] < cutoff].copy()
        final_val = df[df["race_date"] >= cutoff].copy()

        log.info(f"  Train: {len(final_train):,} rows (< {cutoff})")
        log.info(f"  Val: {len(final_val):,} rows (>= {cutoff})")

        self.model, final_metrics = self.train_fold(final_train, final_val)

        log.info(
            f"  Final model: MAE={final_metrics['mae']:.4f}, "
            f"R²={final_metrics['r2']:.4f}, "
            f"Corr={final_metrics['correlation']:.4f}"
        )

        # Step 4: Feature importance
        importance_df = self._feature_importance()
        log.info("\nTop 30 Features by Gain:")
        for _, row in importance_df.head(30).iterrows():
            bar_len = int(30 * row["importance"] / importance_df["importance"].max())
            bar = "=" * bar_len
            log.info(f"  {row['feature']:<40s} {bar} {row['importance']:.0f}")

        # Step 5: Save artifacts
        log.info(f"\nSaving model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        self._save(output_dir, avg_metrics, overall_wf_metrics,
                   final_metrics, fold_metrics, importance_df, df)

        summary = {
            "model_type": "tfig_regression",
            "target": "TFig",
            "n_features": len(self.feature_cols),
            "feature_cols": self.feature_cols,
            "walk_forward": avg_metrics,
            "overall_walk_forward": overall_wf_metrics,
            "final_model": final_metrics,
            "fold_details": fold_metrics,
            "data_range": {
                "min_date": str(df["race_date"].min().date()),
                "max_date": str(df["race_date"].max().date()),
                "total_rows": len(df),
            },
        }

        self._print_summary(summary)
        return summary

    def _train_simple_split(self, df: pd.DataFrame, output_dir: str) -> dict:
        """Fallback: 80/20 temporal split."""
        dates = df["race_date"].sort_values().unique()
        cutoff_idx = int(len(dates) * 0.8)
        cutoff = dates[cutoff_idx]

        train_df = df[df["race_date"] < cutoff].copy()
        val_df = df[df["race_date"] >= cutoff].copy()

        log.info(f"  Train: {len(train_df):,} rows (< {cutoff})")
        log.info(f"  Val: {len(val_df):,} rows (>= {cutoff})")

        self.model, metrics = self.train_fold(train_df, val_df)
        os.makedirs(output_dir, exist_ok=True)
        self._save(output_dir, {}, {}, metrics, [], self._feature_importance(), df)

        summary = {
            "model_type": "tfig_regression",
            "target": "TFig",
            "split_type": "simple_temporal_80_20",
            "final_model": metrics,
            "n_features": len(self.feature_cols),
            "feature_cols": self.feature_cols,
        }
        self._print_summary(summary)
        return summary

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict TFig for prepared data."""
        if self.model is None:
            raise ValueError("Model not trained.")
        X = df[self.feature_cols].astype(float)
        result = df.copy()
        result["predicted_TFig"] = self.model.predict(X)
        if "TFig" in result.columns:
            result["TFig_error"] = result["predicted_TFig"] - result["TFig"]
        return result

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute evaluation metrics for TFig."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

        # Within 0.5 TFig points accuracy
        errors = np.abs(y_true - y_pred)
        within_05 = (errors <= 0.5).mean() * 100
        within_10 = (errors <= 1.0).mean() * 100

        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "correlation": round(correlation, 4),
            "within_0.5": round(within_05, 1),
            "within_1.0": round(within_10, 1),
        }

    def _feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self.model.feature_name(),
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def _save(self, output_dir, avg_metrics, overall_wf_metrics,
              final_metrics, fold_metrics, importance_df, df):
        """Save model and training artifacts."""
        model_path = os.path.join(output_dir, "tfig_model.lgb")
        self.model.save_model(model_path)
        log.info(f"  Model saved to {model_path}")

        meta = {
            "model_type": "tfig_regression",
            "target": "TFig",
            "feature_cols": self.feature_cols,
            "params": self.params,
            "final_metrics": final_metrics,
        }
        meta_path = os.path.join(output_dir, "tfig_model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        summary = {
            "model_type": "tfig_regression",
            "target": "TFig",
            "n_features": len(self.feature_cols),
            "walk_forward_avg": avg_metrics,
            "overall_walk_forward": overall_wf_metrics,
            "final_model": final_metrics,
            "fold_details": fold_metrics,
            "params": self.params,
            "data_range": {
                "min_date": str(df["race_date"].min().date()),
                "max_date": str(df["race_date"].max().date()),
                "total_rows": len(df),
            },
        }
        summary_path = os.path.join(output_dir, "tfig_training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if len(importance_df) > 0:
            imp_path = os.path.join(output_dir, "tfig_feature_importance.csv")
            importance_df.to_csv(imp_path, index=False)

        log.info(f"  All artifacts saved to {output_dir}")

    @staticmethod
    def _print_summary(summary: dict):
        print("\n" + "=" * 70)
        print("TIME FIGURE (TFig) PREDICTION MODEL — TRAINING SUMMARY")
        print("=" * 70)

        print(f"  Model type:   {summary.get('model_type', 'tfig_regression')}")
        print(f"  Target:       {summary.get('target', 'TFig')}")
        print(f"  Features:     {summary.get('n_features', 0)}")

        if "data_range" in summary:
            dr = summary["data_range"]
            print(f"  Data range:   {dr['min_date']} to {dr['max_date']}")
            print(f"  Total rows:   {dr['total_rows']:,}")

        wf = summary.get("walk_forward", {})
        if wf.get("n_folds", 0) > 0:
            print(f"\n  Walk-Forward Validation ({wf['n_folds']} folds)")
            print("  " + "-" * 50)
            print(f"  Avg MAE:          {wf.get('wf_mae', 'N/A')}")
            print(f"  Avg RMSE:         {wf.get('wf_rmse', 'N/A')}")
            print(f"  Avg R²:           {wf.get('wf_r2', 'N/A')}")
            print(f"  Avg Correlation:  {wf.get('wf_correlation', 'N/A')}")

        owf = summary.get("overall_walk_forward", {})
        if owf:
            print(f"\n  Overall Walk-Forward (combined)")
            print("  " + "-" * 50)
            print(f"  MAE:          {owf.get('overall_wf_mae', 'N/A')}")
            print(f"  RMSE:         {owf.get('overall_wf_rmse', 'N/A')}")
            print(f"  R²:           {owf.get('overall_wf_r2', 'N/A')}")
            print(f"  Correlation:  {owf.get('overall_wf_correlation', 'N/A')}")
            print(f"  Within 0.5:   {owf.get('overall_wf_within_0.5', 'N/A')}%")
            print(f"  Within 1.0:   {owf.get('overall_wf_within_1.0', 'N/A')}%")

        fm = summary.get("final_model", {})
        if fm:
            print(f"\n  Final Model")
            print("  " + "-" * 50)
            print(f"  MAE:          {fm.get('mae', 'N/A')}")
            print(f"  RMSE:         {fm.get('rmse', 'N/A')}")
            print(f"  R²:           {fm.get('r2', 'N/A')}")
            print(f"  Correlation:  {fm.get('correlation', 'N/A')}")
            print(f"  Within 0.5:   {fm.get('within_0.5', 'N/A')}%")
            print(f"  Within 1.0:   {fm.get('within_1.0', 'N/A')}%")

        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train TFig (Time Figure) prediction model"
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
        help="Minimum training window in days (default: 365)",
    )
    parser.add_argument(
        "--val-window", type=int, default=30,
        help="Validation window in days (default: 30)",
    )
    parser.add_argument(
        "--step-days", type=int, default=30,
        help="Step between folds in days (default: 30)",
    )
    parser.add_argument(
        "--start-date", type=str, default="2015-01-01",
        help="Earliest date to load (default: 2015-01-01)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    log.info(f"Loading data from {args.db} (from {args.start_date})...")
    df = load_data(args.db, start_date=args.start_date)
    log.info(f"Loaded {len(df):,} rows")

    trainer = TFigTrainer(
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
    )

    summary = trainer.train(df, output_dir=args.output_dir)

    if "error" in summary:
        log.error(f"Training failed: {summary['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
