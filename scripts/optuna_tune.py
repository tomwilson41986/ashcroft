#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization for BFSP Model.

Uses Optuna to find optimal LightGBM hyperparameters by maximizing
Rank 1 ROI on walk-forward out-of-sample predictions. This is a
profit-focused objective, not MAE — directly optimizing what matters.

Usage:
    python scripts/optuna_tune.py                    # 50 trials
    python scripts/optuna_tune.py --n-trials 100     # More trials
    python scripts/optuna_tune.py --db horse_racing.db
"""

import argparse
import json
import logging
import os
import sqlite3
import sys

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_bfsp import (
    ALL_FEATURE_COLS,
    BFSPTrainer,
    build_context_features,
    profit_weighted_metric,
    profit_weighted_objective,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
COMMISSION = 0.05


def load_data(db_path: str, min_date: str = "2022-01-01") -> pd.DataFrame:
    """Load race data from SQLite database."""
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT * FROM race_results
        WHERE race_date >= '{min_date}'
        ORDER BY race_date, race_time
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def compute_rank1_roi(
    val_df: pd.DataFrame,
    model: lgb.Booster,
    feature_cols: list[str],
) -> float:
    """Compute Rank 1 level-stakes ROI on validation data."""
    X_val = val_df[feature_cols].astype(float)
    preds = model.predict(X_val)
    predicted_bfsp = np.exp(preds)

    result = val_df[["raceid", "bfsp", "placing_numerical"]].copy()
    result["predicted_bfsp"] = predicted_bfsp
    result["won"] = result["placing_numerical"] == 1

    # Rank within race
    result["rank"] = result.groupby("raceid")["predicted_bfsp"].rank(
        method="first"
    )

    # Rank 1 bets only
    r1 = result[result["rank"] == 1].copy()
    if len(r1) == 0:
        return -100.0

    n_bets = len(r1)
    wins = r1[r1["won"]]
    gross_profit = (wins["bfsp"] - 1).sum()
    commission = gross_profit * COMMISSION
    total_returned = len(wins) + gross_profit - commission
    pl = total_returned - n_bets
    roi = pl / n_bets * 100

    return roi


def objective(trial, prepared_df, feature_cols, folds):
    """Optuna objective: maximize Rank 1 ROI on walk-forward folds."""
    params = {
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "verbose": -1,
    }

    decay_rate = trial.suggest_float("decay_rate", 0.3, 3.0)
    use_custom_obj = trial.suggest_categorical("use_custom_obj", [True, False])

    fold_rois = []

    for fold_idx, (train_end, val_start, val_end) in enumerate(folds):
        train_df = prepared_df[prepared_df["race_date"] < train_end].copy()
        val_df = prepared_df[
            (prepared_df["race_date"] >= val_start)
            & (prepared_df["race_date"] < val_end)
        ].copy()

        if len(train_df) < 100 or len(val_df) < 10:
            continue

        X_train = train_df[feature_cols].astype(float)
        y_train = train_df["log_bfsp"].astype(float)
        X_val = val_df[feature_cols].astype(float)
        y_val = val_df["log_bfsp"].astype(float)

        # Exponential decay weighting
        max_date = train_df["race_date"].max()
        days_ago = (max_date - train_df["race_date"]).dt.days.values.astype(float)
        weights = np.exp(-decay_rate * days_ago / 365.0)

        train_set = lgb.Dataset(X_train, label=y_train, weight=weights)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        callbacks = [
            lgb.log_evaluation(period=0),
            lgb.early_stopping(stopping_rounds=30),
        ]

        train_kwargs = {
            "params": params,
            "train_set": train_set,
            "num_boost_round": 1500,
            "valid_sets": [val_set],
            "valid_names": ["valid"],
            "callbacks": callbacks,
        }

        if use_custom_obj:
            p = params.copy()
            p["objective"] = profit_weighted_objective
            p.pop("metric", None)
            train_kwargs["params"] = p
            train_kwargs["feval"] = profit_weighted_metric
        else:
            params["objective"] = "regression"
            params["metric"] = "mae"

        try:
            model = lgb.train(**train_kwargs)
        except Exception:
            continue

        roi = compute_rank1_roi(val_df, model, feature_cols)
        fold_rois.append(roi)

        # Prune unpromising trials early
        trial.report(np.mean(fold_rois), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if not fold_rois:
        return -100.0

    return np.mean(fold_rois)


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite database")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--min-date", default="2022-01-01", help="Minimum training date")
    parser.add_argument("--n-folds", type=int, default=6, help="Number of WF folds to use")
    args = parser.parse_args()

    log.info("Loading data...")
    df = load_data(args.db, args.min_date)
    log.info(f"  Loaded {len(df):,} rows")

    log.info("Preparing features...")
    trainer = BFSPTrainer()
    prepared = trainer.prepare_data(df)
    feature_cols = trainer.feature_cols
    log.info(f"  {len(feature_cols)} features, {len(prepared):,} rows")

    # Create walk-forward folds (use last N folds for speed)
    all_folds = trainer.create_folds(prepared)
    folds = all_folds[-args.n_folds:]
    log.info(f"  Using {len(folds)} walk-forward folds (of {len(all_folds)} total)")

    # Run optimization
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name="bfsp_rank1_roi",
    )

    log.info(f"\nStarting {args.n_trials} Optuna trials...")
    study.optimize(
        lambda trial: objective(trial, prepared, feature_cols, folds),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Report results
    best = study.best_trial
    log.info(f"\n{'=' * 60}")
    log.info(f"BEST TRIAL: #{best.number}")
    log.info(f"  Rank 1 ROI: {best.value:+.2f}%")
    log.info(f"  Params:")
    for k, v in best.params.items():
        if isinstance(v, float):
            log.info(f"    {k}: {v:.6f}")
        else:
            log.info(f"    {k}: {v}")
    log.info(f"{'=' * 60}")

    # Save best params
    output_path = os.path.join(SCRIPT_DIR, "data", "models", "optuna_best_params.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "best_roi": best.value,
                "best_params": best.params,
                "n_trials": args.n_trials,
                "n_folds": len(folds),
            },
            f,
            indent=2,
        )
    log.info(f"\nSaved best params to {output_path}")

    # Show top 10 trials
    log.info("\nTop 10 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df["state"] == "COMPLETE"]
    trials_df = trials_df.sort_values("value", ascending=False).head(10)
    for _, row in trials_df.iterrows():
        log.info(f"  Trial {row['number']:>3}: ROI {row['value']:>+6.2f}%")


if __name__ == "__main__":
    main()
