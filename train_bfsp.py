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

from model.custom_metrics import CustomMetricsEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

# ---------------------------------------------------------------------------
# Feature columns: all custom metrics + engineered features
# ---------------------------------------------------------------------------

# Horse career metrics (from CustomMetricsEngine)
HORSE_CAREER_FEATURES = [
    "preracehorsecareerNFP",
    "preracehorsecareerRB",
    "preracehorsecareerFSARB",
    "preracehorsecareerFSARB2",
    "preracehorsecareerWins",
    "preracehorsecareerRuns",
    "preracehorsecareerPlaces",
    "preracehorsecareerWIV",
    "preracehorsecareerWAX",
    "preracehorsecareerWOA",
    "preracehorsecareerCWO",
    "preracehorsecareerORR2",
    "Horse_Career_EPF",
    "horsepaceindex",
]

# Last-run and rolling-window metrics
ROLLING_FEATURES = [
    "LRNFP",
    "LR3NFPtotal",
    "LR5NFPtotal",
    "LR10NFPtotal",
    "LR_ORR2",
    "LR3_ORR2",
    "LR5_ORR2",
    "LR10_ORR2",
    "LR3_RWO",
    "LR5_RWO",
    "LR10_RWO",
]

# EPF (Early Position Figure) features
EPF_FEATURES = [
    "LR_EPF",
    "LR2_EPF",
    "LR3_EPF",
    "LR4_EPF",
    "LR5_EPF",
    "LR_EPF2",
    "LR2_EPF2",
    "LR3_EPF2",
    "LR4_EPF2",
    "LR5_EPF2",
    "LR_EPF3",
    "LR2_EPF3",
    "LR3_EPF3",
    "LR4_EPF3",
    "LR5_EPF3",
    "RPS",
    "pace_pressure",
    "prom_runner",
]

# Stability and class metrics
STABILITY_FEATURES = [
    "FSS",
    "FCS",
]

# Probability-Field Difference
PFD_FEATURES = [
    "PFD3",
    "PFD5",
    "PFD10",
]

# Prize money metrics
PRIZE_FEATURES = [
    "WPMRF3",
    "WPMRF5",
    "WPMRF10",
    "PMW3",
    "PMW5",
    "PMW10",
    "RACE_WPMRF",
]

# Odds x Field Size
OFS_FEATURES = [
    "OFS1",
    "OFS3",
    "OFS5",
    "OFS10",
]

# Days Since Last Run (enhanced)
DSLR_FEATURES = [
    "DSLR1",
    "DSLR2",
    "DSLR3",
    "DSLR4",
    "DSLR12diff",
    "DSLR23diff",
    "DSLR34diff",
    "WgtDSLR",
    "FinalDSLR",
]

# Jockey momentum
LRP_FEATURES = [
    "LRPTotalScore",
    "totaljockeyLRPscore",
    "totaljockeyrides",
    "totalLRPjockeyindex",
]

# Pace metrics
PACE_FEATURES = [
    "racepacescore",
    "racepaceindex",
    "trainerpaceindex",
    "jockeypaceindex",
]

# Jockey career metrics
JOCKEY_FEATURES = [
    "preracejockeycareerWins",
    "preracejockeycareerRuns",
    "preracejockeycareerPlaces",
    "preracejockeycareerWIV",
    "preracejockeycareerWAX",
    "preracejockeycareerWOA",
    "preracejockeycareerCWO",
    "Jockey_Career_EPF",
]

# Trainer career metrics
TRAINER_FEATURES = [
    "preracetrainercareerWins",
    "preracetrainercareerRuns",
    "preracetrainercareerPlaces",
    "preracetrainercareerWIV",
    "preracetrainercareerWAX",
    "preracetrainercareerWOA",
    "preracetrainercareerCWO",
    "trainer_Career_EPF",
]

# Trainer-jockey combination metrics
TJ_FEATURES = [
    "trainerjockeycareerWIV",
    "trainerjockeycareerNFP",
    "trainerjockeyWAX",
    "trainerjockeyWOA",
    "trainerjockeyCWO",
]

# Race strength (per-race averages)
RACE_STRENGTH_FEATURES = [
    "RACE_RB",
    "RACE_WIV",
    "RACE_NFP",
    "RACE_Wins",
    "RACE_WOA",
    "LR_RACE_RB",
    "LR_RACE_WIV",
    "LR_RACE_NFP",
    "LR_RACE_Wins",
    "LR_RACE_WOA",
]

# Recency and confidence intervals
RECENCY_FEATURES = [
    "LR3COUNT",
    "LR5COUNT",
    "LR10COUNT",
    "LR3wsum",
    "LR5wsum",
    "LR10wsum",
    "CIL3",
    "CIL5",
    "CIL10",
]

# --- NEW RESEARCH-BACKED FEATURES (Benter/Woods/Ziemba/Syndicate) ---

# Exponential decay form (Benter/Woods: superior to harmonic weights)
EXPONENTIAL_DECAY_FEATURES = [
    "EXP_NFP3", "EXP_NFP5", "EXP_NFP10",
    "EXP_RB3", "EXP_RB5", "EXP_RB10",
    "EXP_ORR23", "EXP_ORR25", "EXP_ORR210",
]

# Expectation residuals (Woods/Ziemba: market-expected vs actual)
RESIDUAL_FEATURES = [
    "NFP_residual",
    "career_residual",
    "residual_exp3",
    "residual_exp5",
    "win_surprise",
    "career_win_surprise",
]

# Unexposure (Syndicate: novel conditions detection)
UNEXPOSURE_FEATURES = [
    "is_debut",
    "dist_experience",
    "first_at_distance",
    "going_experience",
    "first_at_going",
    "course_experience",
    "first_at_course",
    "cd_experience",
    "first_at_cd",
    "unexposure_score",
    "dist_avg_nfp",
    "going_avg_nfp",
    "course_avg_nfp",
]

# Class movement (Ziemba: class drops are strong signals)
CLASS_MOVEMENT_FEATURES = [
    "class_change",
    "avg_class_3",
    "class_vs_avg",
    "is_class_drop",
    "is_class_rise",
]

# Distance aptitude (Benter fundamental variable)
DISTANCE_APTITUDE_FEATURES = [
    "preferred_distance",
    "dist_from_preferred",
    "dist_change_signed",
    "dist_change_lr",
]

# Going preference (Benter fundamental variable)
GOING_PREFERENCE_FEATURES = [
    "preferred_going",
    "going_from_preferred",
    "going_change_lr",
]

# Draw bias (Benter/Woods: post-position effect)
DRAW_BIAS_FEATURES = [
    "draw_relative",
    "draw_quartile",
]

# Form trajectory (Syndicate: improvement/decline detection)
FORM_TRAJECTORY_FEATURES = [
    "form_slope_3",
    "form_slope_5",
    "form_var_3",
    "form_var_5",
    "is_improving",
    "is_declining",
]

# Consistency (Ziemba: reliability measure)
CONSISTENCY_FEATURES = [
    "career_nfp_std",
    "recent_nfp_std",
    "career_place_rate",
    "career_win_rate",
    "recent_win_rate",
    "recent_place_rate",
]

# Weight differential (Benter fundamental variable)
WEIGHT_FEATURES = [
    "weight_vs_avg",
    "weight_vs_min",
    "weight_range",
    "weight_change_lr",
]

# Pedigree features (sire / damsire)
PEDIGREE_FEATURES = [
    # Sire career stats (Bayesian-shrunk)
    "sire_win_rate",
    "sire_place_rate",
    "sire_avg_nfp",
    "sire_wiv",
    "sire_runners",
    # Sire going aptitude
    "sire_going_nfp",
    "sire_going_win_rate",
    # Sire distance aptitude
    "sire_dist_nfp",
    "sire_dist_win_rate",
    # Damsire stats (Bayesian-shrunk)
    "damsire_avg_nfp",
    "damsire_win_rate",
    "damsire_going_nfp",
    "damsire_dist_nfp",
    "damsire_runners",
    # Debut interactions
    "debut_x_sire_nfp",
    "debut_x_sire_wiv",
    "debut_x_trainer_wiv",
]

# Within-race rankings
RANK_FEATURES = [
    "rNFP",
    "rNFPLR3",
    "rNFPLR5",
    "rNFPLR10",
    "horseRBrank",
    "horseFSARBrank",
    "horseFSARB2rank",
    "horseNFPrank",
    "horseWIVrank",
    "horseWAXrank",
    "horseWOArank",
    "horseCWOrank",
    "horseRunsrank",
    "horseWinsrank",
    "horsePlacesrank",
    "rORR2LR",
    "rRWOLR3",
    "rRWOLR5",
    "rRWOLR10",
    "rEPF_LR",
    "rEPF2_LR",
    "rEPF3_LR",
    "rJockeyEPF",
    "rTrainerEPF",
    "rHorseCareerEPF",
    "rDSLR",
    "rFSS",
    "rFCS",
    "rPFD3",
    "rPFD5",
    "rPFD10",
    "rWPMRF3",
    "rWPMRF5",
    "rWPMRF10",
    "rPMW3",
    "rPMW5",
    "rPMW10",
    "rOFS3",
    "rOFS5",
    "rOFS10",
    "rTJWIV",
    "rTJNFP",
    "trainerWIVrank",
    "trainerWAXrank",
    "trainerWOArank",
    "trainerCWOrank",
    "jockeyWIVrank",
    "jockeyWAXrank",
    "jockeyWOArank",
    "jockeyCWOrank",
    "jockeyLRIrank",
    # New research-backed rankings
    "rEXP_NFP5",
    "rEXP_RB5",
    "rResidual",
    "rFormSlope3",
    "rConsistency",
    "rDistApt",
    "rGoingPref",
    "rWeightVsAvg",
    "rUnexposure",
    # Pedigree rankings
    "rSireNFP",
    "rSireWIV",
    "rSireGoingNFP",
    "rSireDistNFP",
    "rDamsireNFP",
]

# Race context features (known pre-race)
CONTEXT_FEATURES = [
    "number_of_runners",
    "dist_furlongs",
    "race_class_num",
    "going_numeric",
    "surface_type_cat",
    "race_type_cat",
    "track_cat",
    "horse_sex_cat",
    "headgear_cat",
    "horse_age_num",
    "pounds_num",
    "stall_num",
    "days_since_lr_num",
    "career_runs_num",
    "or_num",
    "max_or_race",
    "median_or_num",
    "jockeys_claim_num",
    "or_vs_max",
    "or_vs_median",
]

# All feature columns combined
ALL_FEATURE_COLS = (
    HORSE_CAREER_FEATURES
    + ROLLING_FEATURES
    + EPF_FEATURES
    + STABILITY_FEATURES
    + PFD_FEATURES
    + PRIZE_FEATURES
    + OFS_FEATURES
    + DSLR_FEATURES
    + LRP_FEATURES
    + PACE_FEATURES
    + JOCKEY_FEATURES
    + TRAINER_FEATURES
    + TJ_FEATURES
    + RACE_STRENGTH_FEATURES
    + RECENCY_FEATURES
    + RANK_FEATURES
    + CONTEXT_FEATURES
    + EXPONENTIAL_DECAY_FEATURES
    + RESIDUAL_FEATURES
    + UNEXPOSURE_FEATURES
    + CLASS_MOVEMENT_FEATURES
    + DISTANCE_APTITUDE_FEATURES
    + GOING_PREFERENCE_FEATURES
    + DRAW_BIAS_FEATURES
    + FORM_TRAJECTORY_FEATURES
    + CONSISTENCY_FEATURES
    + WEIGHT_FEATURES
    + PEDIGREE_FEATURES
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
# Feature engineering
# ---------------------------------------------------------------------------

def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add race context features that are known pre-race."""

    # Numeric conversions
    df["race_class_num"] = (
        df["race_class"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df["horse_age_num"] = pd.to_numeric(df["horse_age"], errors="coerce")
    df["pounds_num"] = pd.to_numeric(df["pounds"], errors="coerce")
    df["stall_num"] = pd.to_numeric(df["stall"], errors="coerce")
    df["days_since_lr_num"] = pd.to_numeric(
        df.get("days_since_lr", pd.Series(dtype=float)), errors="coerce"
    )
    df["career_runs_num"] = pd.to_numeric(
        df.get("career_runs", pd.Series(dtype=float)), errors="coerce"
    )
    df["or_num"] = pd.to_numeric(df["official_rating"], errors="coerce")
    df["max_or_race"] = pd.to_numeric(
        df.get("max_or_in_race", pd.Series(dtype=float)), errors="coerce"
    )
    df["median_or_num"] = pd.to_numeric(
        df.get("median_or", pd.Series(dtype=float)), errors="coerce"
    )
    df["jockeys_claim_num"] = pd.to_numeric(
        df.get("jockeys_claim", pd.Series(dtype=float)), errors="coerce"
    )

    # OR relative to field
    df["or_vs_max"] = df["or_num"] - df["max_or_race"]
    df["or_vs_median"] = df["or_num"] - df["median_or_num"]

    # Encode going description to numeric scale
    going_map = {
        "heavy": 1.0, "soft": 2.0, "yielding": 2.5,
        "good to soft": 3.0, "good": 4.0, "good to firm": 5.0,
        "firm": 6.0, "hard": 7.0, "standard": 4.0,
        "standard to slow": 3.0, "slow": 2.0,
    }

    def encode_going(g):
        if not g or not isinstance(g, str):
            return 4.0
        gl = g.lower().strip()
        for key, val in going_map.items():
            if key in gl:
                return val
        return 4.0

    df["going_numeric"] = df["going_description"].apply(encode_going)

    # Encode categoricals
    for col in ["surface_type", "race_type", "track", "horse_sex", "headgear"]:
        if col in df.columns:
            df[f"{col}_cat"] = df[col].astype("category").cat.codes
        else:
            df[f"{col}_cat"] = 0

    return df


# ---------------------------------------------------------------------------
# BFSP Training Pipeline
# ---------------------------------------------------------------------------

class BFSPTrainer:
    """Train a BFSP prediction model using all custom metrics.

    Uses LightGBM regression on log(BFSP) with walk-forward temporal
    validation. All 19 custom metrics from CustomMetricsEngine are
    used as features alongside race context features.
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

        # Determine available feature columns
        available = [c for c in ALL_FEATURE_COLS if c in df.columns]

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

    def train_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_boost_round: int = 2000,
        early_stopping: int = 50,
    ) -> tuple[lgb.Booster, dict]:
        """Train a single fold and return model + metrics."""
        X_train = train_df[self.feature_cols].astype(float)
        y_train = train_df["log_bfsp"].astype(float)
        X_val = val_df[self.feature_cols].astype(float)
        y_val = val_df["log_bfsp"].astype(float)

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

        # Evaluate
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

        # Step 2: Walk-forward validation
        folds = self.create_folds(df)
        log.info(f"Created {len(folds)} walk-forward folds")

        if not folds:
            log.warning("Not enough data for walk-forward. Using 80/20 split.")
            return self._train_simple_split(df, output_dir)

        fold_metrics = []
        all_val_preds = []

        # Walk-forward early stopping: stop if no MAE improvement over
        # the last `patience` folds (comparing running average)
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

            _, metrics = self.train_fold(train_df, val_df)
            fold_metrics.append(metrics)

            # Collect validation predictions for aggregate evaluation
            X_val = val_df[self.feature_cols].astype(float)
            val_df = val_df.copy()
            val_df["predicted_log_bfsp"] = _.predict(X_val)
            val_df["predicted_bfsp"] = np.exp(val_df["predicted_log_bfsp"])
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

        # Step 3: Train final model on all data (90/10 split for early stopping)
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

        # Build full summary
        summary = {
            "model_type": "bfsp_regression",
            "target": "log_bfsp",
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

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict BFSP for prepared data."""
        if self.model is None:
            raise ValueError("Model not trained.")

        X = df[self.feature_cols].astype(float)
        log_pred = self.model.predict(X)

        result = df.copy()
        result["predicted_log_bfsp"] = log_pred
        result["predicted_bfsp"] = np.exp(log_pred)

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

        # Save model metadata
        meta = {
            "model_type": "bfsp_regression",
            "target": "log_bfsp",
            "feature_cols": self.feature_cols,
            "params": self.params,
            "final_metrics": final_metrics,
        }
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
    log.info(f"  {len(df):,} rows loaded")

    # Override params if specified
    params = BFSPTrainer.DEFAULT_PARAMS.copy()
    params["learning_rate"] = args.learning_rate
    params["num_leaves"] = args.num_leaves

    # Train
    trainer = BFSPTrainer(
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
        params=params,
    )

    summary = trainer.train(df, output_dir=args.output_dir)

    if "error" not in summary:
        print(f"\n  Model saved to: {args.output_dir}")
    else:
        print(f"\n  Training failed: {summary['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
