#!/usr/bin/env python3
"""
Train BFSP Prediction Model Using All Custom Metrics.

Trains a LightGBM regression model to predict Betfair Starting Price (BFSP)
using the full suite of 19 custom racing metrics from CustomMetricsEngine,
plus engineered features from the historic S3-backed SQLite database.

Pipeline:
    1. Load data from S3 database (or local SQLite)
    2. Calculate all custom metrics (NFP, RB, WIV, WAX, WOA, CWO, ORR2,
       EPF, FSS, FCS, PFD, WPMRF, PMW, OFS, DSLR, LRP, race strength,
       pace, trainer-jockey combos, within-race ranks)
    3. Build additional engineered features (cross-features, context)
    4. Train LightGBM regression on log(BFSP) with walk-forward validation
    5. Evaluate and save model artifacts

Usage:
    python train_bfsp.py                            # Train with defaults
    python train_bfsp.py --db horse_racing.db       # Specify database
    python train_bfsp.py --generate-data            # Generate sample data first
    python train_bfsp.py --fetch-s3                 # Fetch DB from S3 first
    python train_bfsp.py --min-train-days 365       # Minimum training window
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from dataclasses import replace

from model.bfsp_model import (
    DEFAULT_PARAMS,
    OBJECTIVES,
    TARGETS,
    assert_meta_is_servable,
    model_meta,
    TrainConfig,
    build_target,
    fit_bfsp,
    predict_prices,
    profit_weighted_metric,
    profit_weighted_objective,
)
from model.custom_metrics import CustomMetricsEngine
from model.draw_metrics import ALL_DRAW_FEATURES, GP_DRAW_FEATURES
from model.financial_features import FINANCIAL_FEATURES, FINANCIAL_RANK_FEATURES
from model.pace_metrics import (
    ALL_PACE_FEATURES,
    ENTITY_STYLE_FEATURES,
    HORSE_STYLE_FEATURES,
    PACE_FIT_FEATURES,
    PACE_RANK_FEATURES,
    PACE_SCENARIO_FEATURES,
    SUSTAINABILITY_FEATURES,
    TACTICAL_FEATURES,
    TRACK_PACE_BIAS_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

# ---------------------------------------------------------------------------
# Feature columns and pre-race context
#
# These live in model/bfsp_features.py so the feature cache hashes the feature
# code and not the trainer: an edit to the objective or a CLI flag used to
# invalidate hours of cached feature building. Re-exported here because a dozen
# scripts and tests import them from this module.
# ---------------------------------------------------------------------------

from model.bfsp_features import (  # noqa: E402,F401
    ALL_FEATURE_COLS,
    CATEGORICAL_COLS,
    EXTRA_FEATURE_COLS,
    assert_no_post_race_features,
    build_context_features,
    categorical_vocab,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path: str, start_date: str | None = None) -> pd.DataFrame:
    """Load race results from SQLite, optionally filtering by start date."""
    conn = sqlite3.connect(db_path)
    if start_date:
        df = pd.read_sql_query(
            "SELECT * FROM race_results WHERE race_date >= ? ORDER BY race_date, race_time",
            conn,
            params=(start_date,),
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM race_results ORDER BY race_date, race_time", conn
        )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def fetch_from_s3(db_path: str) -> bool:
    """Fetch horse_racing.db from S3."""
    try:
        from scripts.fetch_data import fetch_db_from_s3
        return fetch_db_from_s3(dest=db_path)
    except Exception as e:
        log.error(f"Failed to fetch from S3: {e}")
        return False


def generate_data(db_path: str) -> bool:
    """Generate sample data for testing."""
    try:
        from generate_sample_data import generate_sample_database
        generate_sample_database(db_path, n_days=730)
        return True
    except Exception as e:
        log.error(f"Failed to generate sample data: {e}")
        return False




# ---------------------------------------------------------------------------
# Custom Loss Functions (Profit-Optimized)
# ---------------------------------------------------------------------------


def profit_weighted_objective(preds, train_data):
    """Custom LightGBM objective: asymmetric, price-weighted loss.

    Penalizes prediction errors more for short-priced horses (where
    Rank 1/2 bets are placed) and penalizes under-prediction (predicting
    shorter than reality) 1.5x more because it creates false overlays.
    """
    labels = train_data.get_label()

    residuals = preds - labels  # in log space

    # Price weighting: errors on short-priced horses matter more
    # labels are log(bfsp), so lower labels = shorter prices
    implied_prob = np.exp(-labels)  # ~1/bfsp
    price_weight = np.sqrt(implied_prob)

    # Asymmetric: under-prediction (preds < labels) creates false overlays
    # that lose money, so penalize 1.5x more
    asymmetry = np.where(residuals < 0, 1.5, 1.0)

    grad = residuals * price_weight * asymmetry
    hess = np.ones_like(grad) * price_weight * asymmetry

    return grad, hess


def profit_weighted_metric(preds, train_data):
    """Custom evaluation metric: price-weighted MAE."""
    labels = train_data.get_label()
    implied_prob = np.exp(-labels)
    price_weight = np.sqrt(implied_prob)
    weighted_mae = np.mean(np.abs(preds - labels) * price_weight)
    return "profit_wmae", weighted_mae, False  # False = lower is better


# ---------------------------------------------------------------------------
# BFSP Training Pipeline
# ---------------------------------------------------------------------------

class BFSPTrainer:
    """Train a BFSP prediction model using all custom metrics.

    Uses LightGBM regression on log(BFSP) with walk-forward temporal
    validation. All 19 custom metrics from CustomMetricsEngine are
    used as features alongside race context features.
    """

    #: Re-exported from model/bfsp_model.py, where the recipe now lives.
    DEFAULT_PARAMS = DEFAULT_PARAMS

    def __init__(
        self,
        min_train_days: int = 365,
        val_window_days: int = 30,
        step_days: int = 30,
        params: dict | None = None,
        decay_rate: float | None = None,
        use_custom_objective: bool | None = None,
        gp_draw_surface: bool = False,
        cfg: TrainConfig | None = None,
        wf_folds: int = 0,
    ):
        self.wf_folds = int(wf_folds)
        self.min_train_days = min_train_days
        self.val_window_days = val_window_days
        self.step_days = step_days

        # The recipe. `decay_rate` and `use_custom_objective` stay as arguments
        # because callers pass them, but they now feed one config that is also
        # written into the model metadata -- the previous defaults (decay 1.0,
        # profit-weighted on) were a different model from the one the
        # evaluation measured, and nothing recorded which had been trained.
        if cfg is None:
            cfg = TrainConfig(
                objective="profit_weighted" if use_custom_objective else "l2",
                decay_rate=0.0 if decay_rate is None else float(decay_rate),
                params=params or dict(DEFAULT_PARAMS),
            )
        self.cfg = cfg
        self.params = cfg.params
        self.decay_rate = cfg.decay_rate
        self.use_custom_objective = cfg.objective == "profit_weighted"
        self.metrics_engine = CustomMetricsEngine(gp_draw_surface=gp_draw_surface)
        self.model: lgb.Booster | None = None
        self.feature_cols: list[str] = []

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full data preparation: custom metrics + context features + target."""
        log.info("Calculating all custom metrics...")
        df = self.metrics_engine.calculate_all(df)
        log.info(f"  Custom metrics done: {len(df.columns)} columns")

        log.info("Building context features...")
        df = build_context_features(df)

        # Target: log(BFSP)
        df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
        df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
        df["log_bfsp"] = np.log(df["bfsp"])

        # Determine available feature columns (+ any opt-in extra blocks)
        available = [c for c in list(ALL_FEATURE_COLS) + EXTRA_FEATURE_COLS if c in df.columns]

        # Deduplicate while preserving order
        seen = set()
        unique_features = []
        for c in available:
            if c not in seen:
                seen.add(c)
                unique_features.append(c)

        self.feature_cols = unique_features
        log.info(f"  {len(self.feature_cols)} features available for training")

        # Sort by date for temporal operations
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

    def _compute_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Exponential decay weights: recent races weighted higher."""
        max_date = df["race_date"].max()
        days_ago = (max_date - df["race_date"]).dt.days.values.astype(float)
        weights = np.exp(-self.decay_rate * days_ago / 365.0)
        return weights

    def train_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_boost_round: int = 2000,
        early_stopping: int = 50,
    ) -> tuple[lgb.Booster, dict]:
        """Train a single fold and return model + metrics."""
        # One recipe, shared with evaluate_oos.py. This used to early-stop on
        # `val_df` -- the very rows it then scored -- and to default to the
        # profit-weighted objective plus recency decay while the evaluation ran
        # plain L2 with neither. Two models, one set of published numbers.
        fit = fit_bfsp(train_df, self.feature_cols, self.cfg)
        model = fit.booster

        y_val = build_target(val_df, self.cfg.target)
        y_pred = model.predict(
            val_df[self.feature_cols].astype(float), num_iteration=fit.best_iteration
        )
        metrics = self._compute_metrics(np.asarray(y_val, dtype=float), y_pred)
        metrics["best_iteration"] = fit.best_iteration
        metrics["train_size"] = len(train_df)
        metrics["val_size"] = len(val_df)
        metrics["holdout_mae"] = fit.holdout_metrics["mae"]

        return model, metrics

    def train(
        self,
        df: pd.DataFrame,
        output_dir: str = MODEL_DIR,
    ) -> dict:
        """Full training pipeline with walk-forward validation.

        Steps:
        1. Prepare data (custom metrics + features + target)
        2. Walk-forward validation across temporal folds
        3. Train final model on all data
        4. Save model artifacts and evaluation report
        """
        # Step 1: Prepare data
        df = self.prepare_data(df)
        log.info(f"Prepared {len(df):,} rows with valid BFSP")

        if len(df) < 200:
            log.error("Insufficient data for training (need >= 200 rows)")
            return {"error": "insufficient_data"}

        # Step 2: Walk-forward validation.
        #
        # Off by default. `evaluate_oos.py` is the evaluation -- it runs the
        # same recipe over the same folds and writes the reports -- so doing it
        # again here bought a second set of numbers nobody read, and the cost
        # changed when early stopping moved off the scored fold: a fold is now
        # about thirteen minutes, and the full history is roughly fifty of
        # them, against a 330-minute job ceiling. `--wf-folds N` keeps the
        # first N folds for anyone who wants a quick in-trainer sanity check.
        folds = self.create_folds(df)[: self.wf_folds] if self.wf_folds else []
        if self.wf_folds:
            log.info(f"Walk-forward: {len(folds)} folds (--wf-folds {self.wf_folds})")
        else:
            log.info("Skipping the in-trainer walk-forward; evaluate_oos.py is the "
                     "evaluation. Use --wf-folds N for a quick check.")

        fold_metrics: list = []
        all_val_preds: list = []

        # Walk-forward early stopping: stop if no MAE improvement over
        # the last `patience` folds (comparing running average)
        wf_patience = 5
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

            _, metrics = self.train_fold(train_df, val_df)
            fold_metrics.append(metrics)

            # Collect validation predictions for aggregate evaluation
            X_val = val_df[self.feature_cols].astype(float)
            val_df = val_df.copy()
            val_df["predicted_log_bfsp"] = _.predict(X_val)
            val_df["predicted_bfsp_raw"] = np.exp(val_df["predicted_log_bfsp"])

            # Normalise implied probabilities per race
            if "raceid" not in val_df.columns:
                val_df["raceid"] = (
                    val_df["race_date"].dt.strftime("%Y-%m-%d")
                    + "_" + val_df["track"].astype(str)
                    + "_" + val_df["race_time"].astype(str)
                )
            val_df["_ip"] = 1.0 / val_df["predicted_bfsp_raw"]
            _rsum = val_df.groupby("raceid")["_ip"].transform("sum")
            val_df["predicted_win_prob_norm"] = val_df["_ip"] / _rsum
            val_df["predicted_bfsp"] = 1.0 / val_df["predicted_win_prob_norm"]
            val_df.drop(columns=["_ip"], inplace=True)
            all_val_preds.append(val_df)

            log.info(
                f"    MAE(log): {metrics['log_mae']:.4f}, "
                f"R²(log): {metrics['log_r2']:.4f}, "
                f"MAE(BFSP): {metrics['bfsp_mae']:.2f}, "
                f"MdAPE: {metrics['median_ape_pct']:.1f}%"
            )

            # Walk-forward early stopping check
            if len(fold_metrics) >= 5:
                recent_mae = np.mean(
                    [m["log_mae"] for m in fold_metrics[-5:]]
                )
                if recent_mae < best_running_mae - 0.001:
                    best_running_mae = recent_mae
                    stale_count = 0
                else:
                    stale_count += 1

                if stale_count >= wf_patience:
                    log.info(
                        f"  Walk-forward early stopping at fold {i + 1}: "
                        f"no MAE improvement for {wf_patience} folds "
                        f"(best running avg: {best_running_mae:.4f})"
                    )
                    break

        # Aggregate walk-forward metrics
        if fold_metrics:
            avg_metrics = {
                "wf_log_mae": round(np.mean([m["log_mae"] for m in fold_metrics]), 4),
                "wf_log_rmse": round(np.mean([m["log_rmse"] for m in fold_metrics]), 4),
                "wf_log_r2": round(np.mean([m["log_r2"] for m in fold_metrics]), 4),
                "wf_bfsp_mae": round(np.mean([m["bfsp_mae"] for m in fold_metrics]), 2),
                "wf_median_ape_pct": round(
                    np.mean([m["median_ape_pct"] for m in fold_metrics]), 2
                ),
                "n_folds": len(fold_metrics),
            }
        else:
            avg_metrics = {"n_folds": 0}

        # Overall walk-forward evaluation on combined predictions
        overall_wf_metrics = {}
        if all_val_preds:
            combined = pd.concat(all_val_preds, ignore_index=True)
            overall_wf_metrics = self._compute_metrics(
                combined["log_bfsp"].values,
                combined["predicted_log_bfsp"].values,
            )
            overall_wf_metrics = {
                f"overall_wf_{k}": v for k, v in overall_wf_metrics.items()
            }
            log.info(
                f"\nOverall walk-forward: "
                f"MAE(log)={overall_wf_metrics['overall_wf_log_mae']:.4f}, "
                f"R²(log)={overall_wf_metrics['overall_wf_log_r2']:.4f}, "
                f"MdAPE={overall_wf_metrics['overall_wf_median_ape_pct']:.1f}%"
            )

        # Step 2b: Fit calibrators on walk-forward OOS predictions.
        #
        # Two different jobs, two different targets, and conflating them is how
        # the price forecast went wrong before: the isotonic map is fitted on
        # win/lose outcomes and answers "how often does this horse win". BFSP
        # is not a win probability -- it is a price, carrying the market's book
        # and its favourite-longshot slope -- so the price forecast is
        # calibrated against realised BFSP instead. See model/price_calibration.py.
        self.calibrator = None
        self.price_calibrator = None
        if all_val_preds:
            from model.calibration import IsotonicCalibrator
            from model.price_calibration import BSPPriceCalibrator
            combined_for_cal = pd.concat(all_val_preds, ignore_index=True)
            raw_probs = 1.0 / combined_for_cal["predicted_bfsp"].clip(lower=1.01)
            actual_wins = (combined_for_cal["placing_numerical"] == 1).astype(float).values
            self.calibrator = IsotonicCalibrator()
            self.calibrator.fit(raw_probs.values, actual_wins)
            log.info("  Fitted isotonic probability calibrator on walk-forward OOS predictions")

            price_rows = combined_for_cal[pd.to_numeric(combined_for_cal.get("bfsp"), errors="coerce") > 1.0] \
                if "bfsp" in combined_for_cal.columns else combined_for_cal.iloc[0:0]
            if len(price_rows) >= 1000:
                # Fit on the same quantity predict() will feed it: the model's raw
                # exp(log prediction), before the per-race probability normalisation.
                self.price_calibrator = BSPPriceCalibrator().fit(
                    price_rows, price_col="predicted_bfsp_raw", target_col="bfsp"
                )
                log.info(
                    f"  Fitted BFSP price calibrator on {self.price_calibrator.meta_['rows']:,} "
                    f"walk-forward rows / {self.price_calibrator.meta_['races']:,} races"
                )
            else:
                log.warning("  Not enough rows with a realised BFSP to fit the price calibrator")

        # Step 3: Train the final model on ALL the data.
        #
        # This used to fit the first 90% of dates and early-stop on the last
        # 10%, so the published model never trained on its most recent months --
        # the ones most like tomorrow. `fit_bfsp` carves the holdout itself,
        # picks the iteration count on it, then refits on everything at that
        # count (cfg.refit_on_full), which is the same protocol without the
        # permanently withheld tail.
        log.info("\nTraining final model on all data...")
        dates = df["race_date"].sort_values().unique()
        cutoff = dates[int(len(dates) * 0.9)]
        final_train = df[df["race_date"] < cutoff].copy()
        final_val = df[df["race_date"] >= cutoff].copy()

        final_cfg = replace(self.cfg, refit_on_full=True)
        fit = fit_bfsp(df, self.feature_cols, final_cfg)
        self.model = fit.booster
        self.final_fit = fit
        self.categorical_vocab = categorical_vocab(df)
        log.info("  Fitted on %d rows, early stopping on the %d days from %s "
                 "(%d rows), best_iteration %d, then refitted on all %d rows",
                 fit.n_train, final_cfg.holdout_days,
                 pd.Timestamp(fit.holdout_start).date(), fit.n_holdout,
                 fit.best_iteration, len(df))

        y_final = build_target(final_val, final_cfg.target)
        final_metrics = self._compute_metrics(
            np.asarray(y_final, dtype=float),
            self.model.predict(final_val[self.feature_cols].astype(float)),
        )
        final_metrics["best_iteration"] = fit.best_iteration
        final_metrics["holdout_mae"] = fit.holdout_metrics["mae"]

        log.info(
            f"  Final model: MAE(log)={final_metrics['log_mae']:.4f}, "
            f"R²(log)={final_metrics['log_r2']:.4f}, "
            f"MAE(BFSP)={final_metrics['bfsp_mae']:.2f}, "
            f"MdAPE={final_metrics['median_ape_pct']:.1f}%"
        )

        # Step 4: Feature importance
        importance_df = self._feature_importance()
        log.info("\nTop 20 Features by Gain:")
        for _, row in importance_df.head(20).iterrows():
            bar_len = int(30 * row["importance"] / importance_df["importance"].max())
            bar = "=" * bar_len
            log.info(f"  {row['feature']:<40s} {bar} {row['importance']:.0f}")

        # Step 5: Save artifacts
        log.info(f"\nSaving model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        self._save(output_dir, avg_metrics, overall_wf_metrics,
                   final_metrics, fold_metrics, importance_df, df)

        # Save calibrators if fitted
        if self.calibrator is not None:
            cal_path = os.path.join(output_dir, "bfsp_calibrator.json")
            self.calibrator.save(cal_path)
            log.info(f"  Saved probability calibrator to {cal_path}")
        if getattr(self, "price_calibrator", None) is not None:
            price_path = os.path.join(output_dir, "bsp_price_calibrator.json")
            self.price_calibrator.save(price_path)
            log.info(f"  Saved BFSP price calibrator to {price_path}")

        # Step 6: Train win probability model (Benter Stage 2)
        self.prob_model = None
        self.blender = None
        if "won" in df.columns:
            from model.benter_blend import BenterBlender
            from model.probability_model import FundamentalModel

            log.info("\nTraining win probability model (Benter Stage 2)...")
            self.prob_model = FundamentalModel()
            prob_metrics = self.prob_model.train(
                final_train, final_val, self.feature_cols,
                target_col="won", raceid_col="raceid",
            )
            log.info(
                f"  Prob model: logloss={prob_metrics['val_logloss_normalised']:.6f}, "
                f"iters={prob_metrics['best_iteration']}"
            )
            self.prob_model.save(output_dir)

            # Optimize blending lambda on validation set
            log.info("Optimizing Benter blending lambda...")
            val_with_probs = self.prob_model.predict(
                final_val, raceid_col="raceid", normalise=True
            )
            self.blender = BenterBlender()
            optimal_lambda = self.blender.optimise_lambda(
                val_with_probs,
                p_model_col="p_model",
                bfsp_col="bfsp",
                target_col="won",
            )
            log.info(f"  Optimal lambda: {optimal_lambda:.2f}")

            # Save blender config
            blender_path = os.path.join(output_dir, "benter_lambda.json")
            with open(blender_path, "w") as f:
                json.dump({"optimal_lambda": optimal_lambda}, f)
            log.info(f"  Saved blender config to {blender_path}")

        # Build full summary
        summary = {
            "model_type": "bfsp_regression",
            "target": self.cfg.target,
            "train_config": self.cfg.describe(),
            "n_features": len(self.feature_cols),
            "feature_cols": self.feature_cols,
            "walk_forward": avg_metrics,
            "overall_walk_forward": overall_wf_metrics,
            "final_model": final_metrics,
            "fold_details": fold_metrics,
            "decay_rate": self.decay_rate,
            "benter_lambda": (
                self.blender.optimal_lambda
                if self.blender is not None
                else None
            ),
            "has_calibrator": self.calibrator is not None,
            "has_price_calibrator": getattr(self, "price_calibrator", None) is not None,
            "has_prob_model": self.prob_model is not None,
            "data_range": {
                "min_date": str(df["race_date"].min().date()),
                "max_date": str(df["race_date"].max().date()),
                "total_rows": len(df),
            },
        }

        self._print_summary(summary)

        return summary

    def _train_simple_split(
        self, df: pd.DataFrame, output_dir: str
    ) -> dict:
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
            "model_type": "bfsp_regression",
            "split_type": "simple_temporal_80_20",
            "final_model": metrics,
            "n_features": len(self.feature_cols),
            "feature_cols": self.feature_cols,
        }

        self._print_summary(summary)
        return summary

    def predict(self, df: pd.DataFrame, apply_calibration: bool = True) -> pd.DataFrame:
        """Predict BFSP for prepared data.

        Args:
            df: Prepared DataFrame with feature columns.
            apply_calibration: If True and calibrator is fitted, apply
                isotonic calibration to predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        # One price, one probability, race book = 1. See model/bfsp_model.py:
        # the quantile calibrator used to overwrite the price here, fitted on
        # the un-normalised column, which broke the book it had just been
        # normalised to. It is a research diagnostic now, not a serving step.
        result = predict_prices(
            self.model, df, self.feature_cols, race_col="raceid",
            target=self.cfg.target,
            offset=getattr(getattr(self, "final_fit", None), "init_offset", 0.0) or 0.0,
        )
        result["predicted_bfsp_norm"] = result["predicted_bfsp"]

        # The isotonic map is a probability, not a price, so it gets its own
        # column and never touches predicted_bfsp.
        if apply_calibration and self.calibrator is not None:
            result["predicted_win_prob_cal"] = self.calibrator.calibrate(
                result["predicted_win_prob_norm"].values
            )

        # Benter blending: combine model probs with market odds
        if (
            self.prob_model is not None
            and self.blender is not None
            and "bfsp" in result.columns
            and "raceid" in result.columns
        ):
            prob_result = self.prob_model.predict(
                result, raceid_col="raceid", normalise=True
            )
            result["p_model"] = prob_result["p_model"]
            blended = self.blender.blend_with_optimal(
                result, p_model_col="p_model", bfsp_col="bfsp"
            )
            result["p_combined"] = blended["p_combined"]
            result["model_fair_bfsp"] = blended["model_fair_bfsp"]
            result["edge_pct"] = blended["edge_pct"]

        if "bfsp" in result.columns:
            result["bfsp_diff"] = result["predicted_bfsp"] - result["bfsp"]
            result["bfsp_diff_pct"] = (
                (result["predicted_bfsp"] - result["bfsp"])
                / result["bfsp"]
                * 100
            )

        return result

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute evaluation metrics on log and BFSP scales."""
        # Log scale
        log_mae = mean_absolute_error(y_true, y_pred)
        log_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        log_r2 = r2_score(y_true, y_pred)

        # BFSP scale
        bfsp_true = np.exp(y_true)
        bfsp_pred = np.exp(y_pred)
        bfsp_mae = mean_absolute_error(bfsp_true, bfsp_pred)

        pct_errors = np.abs(bfsp_true - bfsp_pred) / np.clip(bfsp_true, 1e-7, None)
        median_ape = np.median(pct_errors) * 100
        mean_ape = np.mean(pct_errors) * 100

        # Directional accuracy: how often do we correctly rank horses?
        # (within a race, does predicted BFSP ordering match actual?)
        # Correlation between predicted and actual log_bfsp
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

    def _feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self.model.feature_name(),
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def _save(
        self,
        output_dir: str,
        avg_metrics: dict,
        overall_wf_metrics: dict,
        final_metrics: dict,
        fold_metrics: list,
        importance_df: pd.DataFrame,
        df: pd.DataFrame,
    ):
        """Save model and training artifacts."""
        # Save LightGBM model
        model_path = os.path.join(output_dir, "bfsp_model.lgb")
        self.model.save_model(model_path)
        log.info(f"  Model saved to {model_path}")

        # Save model metadata. Records the recipe, not just the params: the
        # artefact this replaces said nothing about objective, weighting or
        # provenance, so nobody could tell it was a different model from the
        # one the published numbers described.
        meta = model_meta(
            self.cfg,
            self.feature_cols,
            fit=getattr(self, "final_fit", None),
            vocab=getattr(self, "categorical_vocab", None),
            final_metrics=final_metrics,
            trained_through=str(df["race_date"].max().date()),
            train_rows_total=len(df),
            top_gain=[
                [r.feature, float(r.importance)]
                for r in importance_df.head(20).itertuples()
            ] if len(importance_df) else [],
        )
        assert_meta_is_servable(meta)
        meta_path = os.path.join(output_dir, "bfsp_model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Save training summary
        summary = {
            "model_type": "bfsp_regression",
            "target": "log_bfsp",
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
        summary_path = os.path.join(output_dir, "bfsp_training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save feature importance
        if len(importance_df) > 0:
            imp_path = os.path.join(output_dir, "bfsp_feature_importance.csv")
            importance_df.to_csv(imp_path, index=False)

        log.info(f"  All artifacts saved to {output_dir}")

    @staticmethod
    def _print_summary(summary: dict):
        """Print a formatted training summary."""
        print("\n" + "=" * 70)
        print("BFSP PREDICTION MODEL — TRAINING SUMMARY")
        print("=" * 70)

        print(f"  Model type:   {summary.get('model_type', 'bfsp_regression')}")
        print(f"  Target:       {summary.get('target', 'log_bfsp')}")
        print(f"  Features:     {summary.get('n_features', 0)}")

        if "data_range" in summary:
            dr = summary["data_range"]
            print(f"  Data range:   {dr['min_date']} to {dr['max_date']}")
            print(f"  Total rows:   {dr['total_rows']:,}")

        # Walk-forward results
        wf = summary.get("walk_forward", {})
        if wf.get("n_folds", 0) > 0:
            print(f"\n  Walk-Forward Validation ({wf['n_folds']} folds)")
            print("  " + "-" * 50)
            print(f"  Avg MAE (log):    {wf.get('wf_log_mae', 'N/A')}")
            print(f"  Avg RMSE (log):   {wf.get('wf_log_rmse', 'N/A')}")
            print(f"  Avg R² (log):     {wf.get('wf_log_r2', 'N/A')}")
            print(f"  Avg MAE (BFSP):   {wf.get('wf_bfsp_mae', 'N/A')}")
            print(f"  Avg MdAPE:        {wf.get('wf_median_ape_pct', 'N/A')}%")

        owf = summary.get("overall_walk_forward", {})
        if owf:
            print(f"\n  Overall Walk-Forward (combined)")
            print("  " + "-" * 50)
            print(f"  MAE (log):    {owf.get('overall_wf_log_mae', 'N/A')}")
            print(f"  R² (log):     {owf.get('overall_wf_log_r2', 'N/A')}")
            print(f"  MdAPE:        {owf.get('overall_wf_median_ape_pct', 'N/A')}%")
            print(f"  Correlation:  {owf.get('overall_wf_correlation', 'N/A')}")

        # Final model
        fm = summary.get("final_model", {})
        if fm:
            print(f"\n  Final Model")
            print("  " + "-" * 50)
            print(f"  MAE (log):    {fm.get('log_mae', 'N/A')}")
            print(f"  RMSE (log):   {fm.get('log_rmse', 'N/A')}")
            print(f"  R² (log):     {fm.get('log_r2', 'N/A')}")
            print(f"  MAE (BFSP):   {fm.get('bfsp_mae', 'N/A')}")
            print(f"  MdAPE:        {fm.get('median_ape_pct', 'N/A')}%")
            print(f"  Correlation:  {fm.get('correlation', 'N/A')}")

        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train BFSP prediction model using all custom metrics"
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
        "--fetch-s3", action="store_true",
        help="Fetch database from S3 before training",
    )
    parser.add_argument(
        "--generate-data", action="store_true",
        help="Generate sample data if database doesn't exist",
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
        "--learning-rate", type=float, default=0.03,
        help="Learning rate (default: 0.03)",
    )
    parser.add_argument(
        "--num-leaves", type=int, default=127,
        help="Number of leaves (default: 127)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Only load data from this date onward (YYYY-MM-DD). "
             "Reduces memory usage for large databases.",
    )
    parser.add_argument(
        "--decay-rate", type=float, default=0.0,
        help="Exponential recency weight exp(-rate*days/365). Default 0 (off). "
             "This defaulted to 1.0, which hands a three-year-old race 5%% of "
             "today's weight -- a silent decision to discard most of the history "
             "that the evaluation never made",
    )
    parser.add_argument(
        "--objective", default="l2", choices=list(OBJECTIVES),
        help="l2 (default): squared error on log(BFSP), every runner weighted "
             "equally -- the model forecasts the price of the whole field, and "
             "this is what the walk-forward evaluation measures. "
             "profit_weighted weights by 1/sqrt(BFSP) and penalises "
             "under-prediction 1.5x",
    )
    parser.add_argument(
        "--custom-objective", action="store_true",
        help="Shorthand for --objective profit_weighted",
    )
    parser.add_argument(
        "--no-custom-objective", action="store_true",
        help="Kept for callers that pass it; the profit-weighted loss is now off "
             "by default, so this is a no-op",
    )
    parser.add_argument(
        "--target", default=TrainConfig.target, choices=list(TARGETS),
        help=f"What to regress on (default {TrainConfig.target}, adopted after the "
             f"six-variant head-to-head -- see reports/h2h_summary.md)",
    )
    parser.add_argument("--holdout-days", type=int, default=60,
                        help="Days at the end of the training window used for "
                             "early stopping, never the rows being scored")
    parser.add_argument("--purge-days", type=int, default=30)
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wf-folds", type=int, default=0,
        help="Run N walk-forward folds inside the trainer as a sanity check "
             "(default 0 = skip). evaluate_oos.py is the evaluation; a fold "
             "costs about thirteen minutes, so the full history does not fit "
             "in the job ceiling",
    )
    parser.add_argument(
        "--optuna-params", type=str, default=None,
        help="Path to Optuna best params JSON to use instead of defaults",
    )
    parser.add_argument(
        "--abm-features", type=str, default=None,
        help="Path to precomputed ABM feature file (parquet/csv from "
             "`research_lab.py abm-features`); merged and added as features",
    )
    parser.add_argument(
        "--market-features", action="store_true",
        help="Add lag-safe Betfair market-movement features from the "
             "betfair_prices table (load it with betfair_prices.py first)",
    )
    parser.add_argument(
        "--perf-features", action="store_true",
        help="Add lag-safe performance-figure (lbs) features",
    )
    parser.add_argument(
        "--pedigree-features", action="store_true",
        help="Add the Part 5.3-5.4 sire / damsire / dam / sibling / nick block "
             "(aptitudes as residuals against the entity's own level)",
    )
    parser.add_argument(
        "--connection-features", action="store_true",
        help="Add the Part 5.1-5.2 trainer and jockey block (context splits, "
             "schedule-adjusted strike rates, season form)",
    )
    parser.add_argument(
        "--odds-features", action="store_true",
        help="Add the Part 4 odds-derived block (effective field size, rating "
             "ranks, Shin). STAGE C: market-derived, so never for a market-free model",
    )
    parser.add_argument(
        "--gp-draw", action="store_true",
        help="Add the Gaussian-process per-stall draw surface. Off by default: "
             "it costs one fit per course per year, and the shrunk cells already "
             "resolve single stalls",
    )
    parser.add_argument(
        "--blandford-features", action="store_true",
        help="Add Timeform-feed features from the blandford_results table "
             "(load it with blandford_sync.py first)",
    )
    args = parser.parse_args()

    # Fetch data if needed
    if args.fetch_s3:
        log.info("Fetching database from S3...")
        if not fetch_from_s3(args.db):
            log.error("Failed to fetch from S3.")
            sys.exit(1)

    if not os.path.exists(args.db):
        if args.generate_data:
            log.info("Database not found. Generating sample data...")
            if not generate_data(args.db):
                sys.exit(1)
        else:
            print(f"Database not found: {args.db}")
            print("Options:")
            print("  --fetch-s3        Fetch from S3")
            print("  --generate-data   Generate sample data for testing")
            print("  --db <path>       Specify database path")
            sys.exit(1)

    # Load data
    log.info("Loading data...")
    df = load_data(args.db, start_date=args.start_date)

    # Opt-in research feature blocks (RESEARCH_FRAMEWORK.md)
    if getattr(args, "gp_draw", False):
        log.info("Adding the Gaussian-process per-stall draw surface...")
        EXTRA_FEATURE_COLS.extend(GP_DRAW_FEATURES)
    if getattr(args, "pedigree_features", False) or getattr(args, "connection_features", False):
        # Both blocks aggregate a horse's normalised finishing position over its
        # sire's, dam's or yard's *earlier* runners, so the primitive has to
        # exist before they can lag it.
        from model.primitives import add_run_primitives
        log.info("Adding run primitives (needed by the pedigree and connection blocks)...")
        df = add_run_primitives(df)
    if getattr(args, "pedigree_features", False):
        from model.pedigree import add_pedigree_features
        log.info("Adding pedigree features...")
        df, feats = add_pedigree_features(df)
        EXTRA_FEATURE_COLS.extend(feats)
        log.info(f"  {len(feats)} pedigree features")
    if getattr(args, "connection_features", False):
        from model.connections import add_connection_features
        log.info("Adding connection features...")
        # price_col/sp_col left unset: the A/E and ROI block is market-derived
        # and belongs to Stage C, not to a model whose target is already the market.
        df, feats = add_connection_features(df)
        EXTRA_FEATURE_COLS.extend(feats)
        log.info(f"  {len(feats)} connection features")
    if getattr(args, "odds_features", False):
        from model.odds_metrics import ODDS_FEATURES, add_odds_metrics
        log.info("Adding odds-derived (Stage C) features...")
        df = add_odds_metrics(df)
        EXTRA_FEATURE_COLS.extend([c for c in ODDS_FEATURES if c in df.columns])
    if args.perf_features:
        from model.perf_figures import PERF_FIGURE_FEATURES, add_perf_figure_features
        log.info("Adding performance-figure features...")
        df = add_perf_figure_features(df)
        EXTRA_FEATURE_COLS.extend(PERF_FIGURE_FEATURES)
    if args.market_features:
        from model.market_features import MARKET_FEATURES, add_market_features
        log.info("Adding Betfair market-movement features...")
        df = add_market_features(df, db_path=args.db)
        EXTRA_FEATURE_COLS.extend(MARKET_FEATURES)
    if args.blandford_features:
        from model.blandford_features import BLANDFORD_FEATURES, add_blandford_features
        log.info("Adding Blandford/Timeform feed features...")
        df = add_blandford_features(df, db_path=args.db)
        EXTRA_FEATURE_COLS.extend(BLANDFORD_FEATURES)
    if args.abm_features:
        from model.abm.features import ABM_FEATURES, load_abm_features, merge_abm_features
        log.info(f"Merging ABM features from {args.abm_features}...")
        df = merge_abm_features(df, load_abm_features(args.abm_features))
        EXTRA_FEATURE_COLS.extend(ABM_FEATURES)
    log.info(f"  {len(df):,} rows loaded")

    # Load Optuna-tuned params if specified
    decay_rate = args.decay_rate
    objective = "profit_weighted" if args.custom_objective else args.objective
    use_custom_obj = objective == "profit_weighted"
    params = dict(DEFAULT_PARAMS)

    if args.optuna_params:
        log.info(f"Loading Optuna-tuned params from {args.optuna_params}...")
        with open(args.optuna_params) as f:
            optuna_data = json.load(f)
        best = optuna_data["best_params"]
        params["num_leaves"] = best.get("num_leaves", params["num_leaves"])
        params["learning_rate"] = best.get("learning_rate", params["learning_rate"])
        params["min_child_samples"] = best.get("min_child_samples", params["min_child_samples"])
        params["feature_fraction"] = best.get("feature_fraction", params["feature_fraction"])
        params["bagging_fraction"] = best.get("bagging_fraction", params["bagging_fraction"])
        params["bagging_freq"] = best.get("bagging_freq", params["bagging_freq"])
        params["lambda_l1"] = best.get("lambda_l1", params["lambda_l1"])
        params["lambda_l2"] = best.get("lambda_l2", params["lambda_l2"])
        if "max_depth" in best:
            params["max_depth"] = best["max_depth"]
        decay_rate = best.get("decay_rate", decay_rate)
        if "use_custom_obj" in best:
            use_custom_obj = best["use_custom_obj"]
            objective = "profit_weighted" if use_custom_obj else "l2"
        log.info(f"  Loaded params: {params}")
        log.info(f"  Decay rate: {decay_rate}, Custom objective: {use_custom_obj}")
    else:
        # Override params from CLI
        params["learning_rate"] = args.learning_rate
        params["num_leaves"] = args.num_leaves

    # Train
    cfg = TrainConfig(
        objective=objective,
        decay_rate=decay_rate,
        target=args.target,
        holdout_days=args.holdout_days,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        refit_on_full=True,   # the published model sees the most recent weeks
        seed=args.seed,
        params=params,
    )
    log.info("Training recipe: objective=%s target=%s decay=%.2f holdout=%dd "
             "purge=%dd seed=%d", cfg.objective, cfg.target, cfg.decay_rate,
             cfg.holdout_days, cfg.purge_days, cfg.seed)

    trainer = BFSPTrainer(
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
        gp_draw_surface=getattr(args, "gp_draw", False),
        cfg=cfg,
        wf_folds=args.wf_folds,
    )

    summary = trainer.train(df, output_dir=args.output_dir)

    if "error" not in summary:
        print(f"\n  Model saved to: {args.output_dir}")
    else:
        print(f"\n  Training failed: {summary['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
