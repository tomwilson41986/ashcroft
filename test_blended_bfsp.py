#!/usr/bin/env python3
"""
Benter Blended BFSP Model — Test Harness.

Tests the Benter approach: blend our fundamental model's predicted BFSP
with actual market BFSP (as a proxy for pre-race exchange odds).

Benter's key insight: the market is ~90% efficient. A fundamental model
adds value by correcting the remaining ~10%. The optimal strategy is NOT
to replace the market entirely, but to blend:

    log(P_blend) = (1 - α) · log(P_market) + α · log(P_model)

Equivalently in BFSP space:
    log(BFSP_blend) = (1 - α) · log(BFSP_actual) + α · log(BFSP_predicted)

Where:
    α = 0  → pure market (just use BFSP)
    α = 1  → pure model (ignore market entirely)
    α*     → optimal blend weight (what we're searching for)

Evaluation is against actual OUTCOMES (win/lose), not against BFSP itself,
which avoids circularity. Metrics:
    - Log-loss of implied win probabilities vs actual outcomes
    - Brier score
    - F2 statistic (goodness-of-fit vs uniform baseline)
    - Betting simulation ROI (value bets where blend disagrees with market)

Usage:
    python test_blended_bfsp.py                          # With real DB
    python test_blended_bfsp.py --generate-data           # Generate sample data first
    python test_blended_bfsp.py --db horse_racing.db      # Specify DB path
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
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

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
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "models")


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def bfsp_to_probs(bfsp_series: pd.Series, race_ids: pd.Series) -> pd.Series:
    """Convert BFSP to normalised win probabilities per race."""
    raw_prob = 1.0 / bfsp_series.clip(lower=1.01)
    # Normalise within each race
    race_sums = raw_prob.groupby(race_ids).transform("sum")
    return raw_prob / race_sums


def blend_log_bfsp(
    log_bfsp_pred: np.ndarray,
    log_bfsp_actual: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Benter blend in log-BFSP space.

    alpha=0 → pure market, alpha=1 → pure model.
    """
    return (1 - alpha) * log_bfsp_actual + alpha * log_bfsp_pred


def f2_statistic(y_true: np.ndarray, probs: np.ndarray, n_per_race: np.ndarray) -> float:
    """F2 goodness-of-fit: improvement over uniform (1/N) baseline.

    F2 = 1 - sum(-y * ln(p)) / sum(-y * ln(1/N))

    F2 > 0 means better than random; F2 = 1 means perfect.
    """
    eps = 1e-10
    winners = y_true == 1
    if winners.sum() == 0:
        return 0.0

    model_term = -np.sum(np.log(probs[winners] + eps))
    uniform_probs = 1.0 / n_per_race[winners]
    baseline_term = -np.sum(np.log(uniform_probs + eps))

    if baseline_term == 0:
        return 0.0
    return 1.0 - model_term / baseline_term


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_blend(
    val_df: pd.DataFrame,
    alpha: float,
) -> dict:
    """Evaluate a single blend weight on validation data.

    val_df must have columns:
        - log_bfsp (actual)
        - predicted_log_bfsp (model prediction)
        - race_id (unique race identifier)
        - won (1 if winner, 0 otherwise)
        - number_of_runners
    """
    # Blend
    blended_log = blend_log_bfsp(
        val_df["predicted_log_bfsp"].values,
        val_df["log_bfsp"].values,
        alpha,
    )
    blended_bfsp = np.exp(blended_log)

    # Convert to normalised probabilities per race
    raw_prob = 1.0 / np.clip(blended_bfsp, 1.01, None)
    race_sums = val_df.groupby("race_id")["won"].transform("count")  # just for grouping
    prob_df = val_df[["race_id"]].copy()
    prob_df["raw_prob"] = raw_prob
    prob_df["norm_prob"] = prob_df.groupby("race_id")["raw_prob"].transform(
        lambda x: x / x.sum()
    )
    probs = prob_df["norm_prob"].values

    # Clip for numerical stability
    probs = np.clip(probs, 1e-10, 1 - 1e-10)

    y_true = val_df["won"].values.astype(int)
    n_runners = val_df["number_of_runners"].values.astype(float)

    # Metrics
    ll = log_loss(y_true, probs)
    brier = brier_score_loss(y_true, probs)

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.5

    f2 = f2_statistic(y_true, probs, n_runners)

    # Market log-loss (alpha=0 baseline)
    market_prob = 1.0 / np.clip(np.exp(val_df["log_bfsp"].values), 1.01, None)
    market_prob_df = val_df[["race_id"]].copy()
    market_prob_df["raw"] = market_prob
    market_prob_df["norm"] = market_prob_df.groupby("race_id")["raw"].transform(
        lambda x: x / x.sum()
    )
    market_probs = np.clip(market_prob_df["norm"].values, 1e-10, 1 - 1e-10)
    market_ll = log_loss(y_true, market_probs)

    # Improvement over market
    ll_improvement = (market_ll - ll) / market_ll * 100  # % improvement

    # BFSP prediction accuracy (how close is blend to actual BFSP)
    bfsp_actual = np.exp(val_df["log_bfsp"].values)
    bfsp_blend = blended_bfsp
    bfsp_mae = np.mean(np.abs(bfsp_actual - bfsp_blend))
    median_ape = np.median(np.abs(bfsp_actual - bfsp_blend) / bfsp_actual) * 100

    # Betting simulation: bet when blend probability > market probability
    # (i.e., model thinks horse is underpriced)
    model_prob_raw = 1.0 / np.clip(np.exp(val_df["predicted_log_bfsp"].values), 1.01, None)
    model_prob_df = val_df[["race_id"]].copy()
    model_prob_df["raw"] = model_prob_raw
    model_prob_df["norm"] = model_prob_df.groupby("race_id")["raw"].transform(
        lambda x: x / x.sum()
    )

    edge = probs - market_probs  # blend says higher prob than market
    value_mask = edge > 0.02  # minimum 2% edge to bet

    n_bets = value_mask.sum()
    if n_bets > 0:
        bet_winners = y_true[value_mask]
        bet_bfsp = bfsp_actual[value_mask]
        # Level-stake PnL: win pays (bfsp - 1), lose pays -1
        pnl = np.where(bet_winners == 1, bet_bfsp - 1, -1)
        total_pnl = pnl.sum()
        roi = total_pnl / n_bets * 100
        strike_rate = bet_winners.mean() * 100
    else:
        total_pnl = 0
        roi = 0
        strike_rate = 0
        n_bets = 0

    return {
        "alpha": alpha,
        "log_loss": round(ll, 6),
        "brier_score": round(brier, 6),
        "auc_roc": round(auc, 4),
        "f2_statistic": round(f2, 4),
        "market_log_loss": round(market_ll, 6),
        "ll_improvement_pct": round(ll_improvement, 4),
        "bfsp_mae": round(bfsp_mae, 2),
        "median_ape_pct": round(median_ape, 2),
        "n_value_bets": int(n_bets),
        "betting_roi_pct": round(roi, 2),
        "strike_rate_pct": round(strike_rate, 2),
        "total_pnl": round(total_pnl, 2),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    db_path: str,
    start_date: str | None = None,
    min_train_days: int = 365,
    val_window_days: int = 60,
) -> dict:
    """Run the full Benter blended BFSP experiment."""

    # Load data
    log.info("Loading data...")
    df = load_data(db_path, start_date=start_date)
    log.info(f"  {len(df):,} rows loaded ({df['race_date'].min()} to {df['race_date'].max()})")

    # Prepare features
    trainer = BFSPTrainer(min_train_days=min_train_days)
    df = trainer.prepare_data(df)
    log.info(f"  {len(df):,} rows with valid BFSP, {len(trainer.feature_cols)} features")

    # Create race ID for grouping
    df["race_id"] = (
        df["race_date"].dt.strftime("%Y-%m-%d")
        + "_"
        + df["race_time"].astype(str)
        + "_"
        + df["track"].astype(str)
    )

    # Win indicator
    df["won"] = (df["placing_numerical"] == 1).astype(int)

    # Temporal split: train / val / test
    dates = sorted(df["race_date"].unique())
    n_dates = len(dates)

    # 70% train, 15% val (for alpha tuning), 15% test (for final eval)
    train_cutoff = dates[int(n_dates * 0.70)]
    val_cutoff = dates[int(n_dates * 0.85)]

    train_df = df[df["race_date"] < train_cutoff].copy()
    val_df = df[(df["race_date"] >= train_cutoff) & (df["race_date"] < val_cutoff)].copy()
    test_df = df[df["race_date"] >= val_cutoff].copy()

    log.info(f"\n  Train: {len(train_df):,} rows ({train_df['race_date'].min().date()} to {train_df['race_date'].max().date()})")
    log.info(f"  Val:   {len(val_df):,} rows ({val_df['race_date'].min().date()} to {val_df['race_date'].max().date()})")
    log.info(f"  Test:  {len(test_df):,} rows ({test_df['race_date'].min().date()} to {test_df['race_date'].max().date()})")

    # Ensure each split has winners
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_races = split["race_id"].nunique()
        n_winners = split["won"].sum()
        log.info(f"    {name}: {n_races} races, {n_winners} winners")

    # Train baseline BFSP model
    log.info("\nTraining baseline BFSP model...")
    feature_cols = trainer.feature_cols

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["log_bfsp"].astype(float)
    X_val = val_df[feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)

    train_set = lgb.Dataset(X_train, label=y_train)

    # Use a small holdout from train for early stopping
    es_cutoff = dates[int(n_dates * 0.65)]
    es_train = train_df[train_df["race_date"] < es_cutoff]
    es_val = train_df[train_df["race_date"] >= es_cutoff]

    es_train_set = lgb.Dataset(
        es_train[feature_cols].astype(float),
        label=es_train["log_bfsp"].astype(float),
    )
    es_val_set = lgb.Dataset(
        es_val[feature_cols].astype(float),
        label=es_val["log_bfsp"].astype(float),
        reference=es_train_set,
    )

    params = BFSPTrainer.DEFAULT_PARAMS.copy()
    callbacks = [
        lgb.log_evaluation(period=0),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        es_train_set,
        num_boost_round=2000,
        valid_sets=[es_train_set, es_val_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    log.info(f"  Model trained: {model.best_iteration} iterations")

    # Generate predictions on val and test
    val_df = val_df.copy()
    val_df["predicted_log_bfsp"] = model.predict(X_val)
    val_df["predicted_bfsp"] = np.exp(val_df["predicted_log_bfsp"])

    test_df = test_df.copy()
    test_df["predicted_log_bfsp"] = model.predict(X_test)
    test_df["predicted_bfsp"] = np.exp(test_df["predicted_log_bfsp"])

    # Baseline model metrics
    from train_bfsp import BFSPTrainer as BT
    baseline_val = BT._compute_metrics(
        val_df["log_bfsp"].values, val_df["predicted_log_bfsp"].values
    )
    baseline_test = BT._compute_metrics(
        test_df["log_bfsp"].values, test_df["predicted_log_bfsp"].values
    )
    log.info(f"\n  Baseline (val):  R²={baseline_val['log_r2']:.4f}, MAE={baseline_val['log_mae']:.4f}, MdAPE={baseline_val['median_ape_pct']:.1f}%")
    log.info(f"  Baseline (test): R²={baseline_test['log_r2']:.4f}, MAE={baseline_test['log_mae']:.4f}, MdAPE={baseline_test['median_ape_pct']:.1f}%")

    # -----------------------------------------------------------------------
    # Benter blend: sweep alpha values on VALIDATION set
    # -----------------------------------------------------------------------
    alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]

    log.info("\n" + "=" * 80)
    log.info("BENTER BLEND SWEEP — VALIDATION SET")
    log.info("=" * 80)
    log.info(f"{'Alpha':>7s} | {'Log-Loss':>10s} | {'Brier':>8s} | {'AUC':>6s} | {'F2':>7s} | {'LL vs Mkt':>10s} | {'Bets':>6s} | {'ROI%':>7s}")
    log.info("-" * 80)

    val_results = []
    for alpha in alphas:
        result = evaluate_blend(val_df, alpha)
        val_results.append(result)
        log.info(
            f"{alpha:7.2f} | {result['log_loss']:10.6f} | {result['brier_score']:8.6f} | "
            f"{result['auc_roc']:6.4f} | {result['f2_statistic']:7.4f} | "
            f"{result['ll_improvement_pct']:+9.4f}% | {result['n_value_bets']:6d} | "
            f"{result['betting_roi_pct']:+6.2f}%"
        )

    # Find optimal alpha (minimum log-loss on validation)
    best_val = min(val_results, key=lambda x: x["log_loss"])
    optimal_alpha = best_val["alpha"]

    log.info(f"\n  Optimal alpha (min log-loss on val): {optimal_alpha:.2f}")
    log.info(f"  Best log-loss: {best_val['log_loss']:.6f}")
    log.info(f"  LL improvement over market: {best_val['ll_improvement_pct']:+.4f}%")

    # Fine-grained search around optimal
    if 0 < optimal_alpha < 1:
        fine_alphas = np.arange(
            max(0, optimal_alpha - 0.10),
            min(1, optimal_alpha + 0.11),
            0.01,
        )
        log.info(f"\n  Fine-grained sweep around α={optimal_alpha:.2f}...")
        fine_results = []
        for alpha in fine_alphas:
            result = evaluate_blend(val_df, round(alpha, 2))
            fine_results.append(result)

        best_fine = min(fine_results, key=lambda x: x["log_loss"])
        if best_fine["log_loss"] < best_val["log_loss"]:
            optimal_alpha = best_fine["alpha"]
            best_val = best_fine
            log.info(f"  Refined optimal alpha: {optimal_alpha:.2f}")

    # -----------------------------------------------------------------------
    # Final evaluation on TEST set with optimal alpha
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 80)
    log.info("FINAL EVALUATION — TEST SET")
    log.info("=" * 80)

    # Evaluate key alphas on test
    test_alphas = [0.0, optimal_alpha, 0.50, 1.0]
    if optimal_alpha not in [0.0, 0.50, 1.0]:
        test_alphas = sorted(set(test_alphas))

    log.info(f"\n{'Alpha':>7s} | {'Log-Loss':>10s} | {'Brier':>8s} | {'AUC':>6s} | {'F2':>7s} | {'LL vs Mkt':>10s} | {'Bets':>6s} | {'ROI%':>7s}")
    log.info("-" * 80)

    test_results = []
    for alpha in test_alphas:
        result = evaluate_blend(test_df, alpha)
        test_results.append(result)
        label = " <-- optimal" if alpha == optimal_alpha else ""
        label = " <-- market" if alpha == 0.0 else label
        label = " <-- model only" if alpha == 1.0 else label
        log.info(
            f"{alpha:7.2f} | {result['log_loss']:10.6f} | {result['brier_score']:8.6f} | "
            f"{result['auc_roc']:6.4f} | {result['f2_statistic']:7.4f} | "
            f"{result['ll_improvement_pct']:+9.4f}% | {result['n_value_bets']:6d} | "
            f"{result['betting_roi_pct']:+6.2f}%{label}"
        )

    # Best on test
    test_optimal = next(r for r in test_results if r["alpha"] == optimal_alpha)
    test_market = next(r for r in test_results if r["alpha"] == 0.0)
    test_model = next(r for r in test_results if r["alpha"] == 1.0)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BENTER BLENDED BFSP — EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\n  Data: {len(df):,} rows, {df['race_date'].min().date()} to {df['race_date'].max().date()}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train/Val/Test split: 70/15/15%")

    print(f"\n  BASELINE MODEL (α=1.0, pure fundamental model)")
    print(f"    Val  — R²: {baseline_val['log_r2']:.4f}, MAE(log): {baseline_val['log_mae']:.4f}, Corr: {baseline_val['correlation']:.4f}")
    print(f"    Test — R²: {baseline_test['log_r2']:.4f}, MAE(log): {baseline_test['log_mae']:.4f}, Corr: {baseline_test['correlation']:.4f}")

    print(f"\n  OPTIMAL BENTER BLEND (α={optimal_alpha:.2f})")
    print(f"    Interpretation: {(1-optimal_alpha)*100:.0f}% market + {optimal_alpha*100:.0f}% model")
    print(f"    Val  log-loss:  {best_val['log_loss']:.6f}")
    print(f"    Test log-loss:  {test_optimal['log_loss']:.6f}")

    print(f"\n  TEST SET COMPARISON")
    print(f"  {'Approach':<25s} | {'Log-Loss':>10s} | {'Brier':>8s} | {'AUC':>6s} | {'F2':>7s} | {'ROI%':>7s}")
    print(f"  {'-'*72}")
    print(f"  {'Market only (α=0.0)':<25s} | {test_market['log_loss']:10.6f} | {test_market['brier_score']:8.6f} | {test_market['auc_roc']:6.4f} | {test_market['f2_statistic']:7.4f} | {test_market['betting_roi_pct']:+6.2f}%")
    print(f"  {'Benter blend (α='+str(optimal_alpha)+')':<25s} | {test_optimal['log_loss']:10.6f} | {test_optimal['brier_score']:8.6f} | {test_optimal['auc_roc']:6.4f} | {test_optimal['f2_statistic']:7.4f} | {test_optimal['betting_roi_pct']:+6.2f}%")
    print(f"  {'Model only (α=1.0)':<25s} | {test_model['log_loss']:10.6f} | {test_model['brier_score']:8.6f} | {test_model['auc_roc']:6.4f} | {test_model['f2_statistic']:7.4f} | {test_model['betting_roi_pct']:+6.2f}%")

    ll_gain_vs_market = (test_market['log_loss'] - test_optimal['log_loss']) / test_market['log_loss'] * 100
    ll_gain_vs_model = (test_model['log_loss'] - test_optimal['log_loss']) / test_model['log_loss'] * 100
    print(f"\n  Blend improvement over market:      {ll_gain_vs_market:+.4f}% log-loss")
    print(f"  Blend improvement over model-only:  {ll_gain_vs_model:+.4f}% log-loss")

    if optimal_alpha == 0.0:
        print(f"\n  VERDICT: Model adds NO value beyond the market.")
        print(f"  The market (BFSP) is the best predictor of outcomes.")
    elif optimal_alpha < 0.15:
        print(f"\n  VERDICT: Model adds MARGINAL value. Market dominates.")
        print(f"  A light blend ({optimal_alpha*100:.0f}% model) slightly improves on pure market.")
    elif optimal_alpha < 0.40:
        print(f"\n  VERDICT: Model adds MODERATE value. Benter blend is optimal.")
        print(f"  The {optimal_alpha*100:.0f}/{(1-optimal_alpha)*100:.0f} model/market blend beats both pure approaches.")
    elif optimal_alpha < 0.70:
        print(f"\n  VERDICT: Model adds SIGNIFICANT value. Near-equal blend.")
        print(f"  Our fundamental model captures substantial edge beyond the market.")
    else:
        print(f"\n  VERDICT: Model adds DOMINANT value. Model outweighs market.")
        print(f"  The fundamental model is more predictive than market prices alone.")

    print("=" * 80)

    # Save results
    results = {
        "experiment": "benter_blended_bfsp",
        "optimal_alpha": optimal_alpha,
        "interpretation": f"{(1-optimal_alpha)*100:.0f}% market + {optimal_alpha*100:.0f}% model",
        "baseline_model": {
            "val": baseline_val,
            "test": baseline_test,
        },
        "validation_sweep": val_results,
        "test_results": {
            "market_only": test_market,
            "optimal_blend": test_optimal,
            "model_only": test_model,
        },
        "splits": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_dates": f"{train_df['race_date'].min().date()} to {train_df['race_date'].max().date()}",
            "val_dates": f"{val_df['race_date'].min().date()} to {val_df['race_date'].max().date()}",
            "test_dates": f"{test_df['race_date'].min().date()} to {test_df['race_date'].max().date()}",
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "benter_blend_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test Benter blended BFSP approach"
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--generate-data", action="store_true",
        help="Generate sample data if database doesn't exist",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Only load data from this date onward (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--min-train-days", type=int, default=365,
        help="Minimum training window in days (default: 365)",
    )
    parser.add_argument(
        "--val-window", type=int, default=60,
        help="Validation window in days (default: 60)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        if args.generate_data:
            log.info("Database not found. Generating sample data...")
            from generate_sample_data import generate_sample_database
            generate_sample_database(args.db, n_days=730)
        else:
            print(f"Database not found: {args.db}")
            print("Options:")
            print("  --generate-data   Generate sample data for testing")
            print("  --db <path>       Specify database path")
            sys.exit(1)

    results = run_experiment(
        db_path=args.db,
        start_date=args.start_date,
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
    )


if __name__ == "__main__":
    main()
