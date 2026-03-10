"""
BFSP Prediction Model — Predict Betfair Starting Price from Pre-Race Features.

Uses LightGBM regression on log(BFSP) with the full custom metrics engine.
At prediction time, we only have pre-race data (no market prices).
After BFSP becomes available, we compare predicted vs actual to find overlays.

Pipeline:
1. Calculate 19 custom metrics from historical data (lag-safe)
2. Predict log(BFSP) using LightGBM regression
3. Convert to implied probability: P_model = 1 / exp(predicted_log_bfsp)
4. Compare to actual BFSP when available
5. Bet when actual BFSP > predicted fair price (overlay)

Walk-forward validation with strict temporal ordering.

Usage:
    python -m model.train_bfsp                         # Train with defaults
    python -m model.train_bfsp --db horse_racing.db    # Specify database
    python -m model.train_bfsp --importance            # Show feature importance
"""

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")

# All features available BEFORE race (no BFSP, no placing, no post-race data).
# These are what we'd have at declaration time + historical metrics.
BFSP_FEATURES = [
    # Horse career performance (lagged — from prior races only)
    "preracehorsecareerNFP",
    "preracehorsecareerRB",
    "preracehorsecareerFSARB",
    "preracehorsecareerFSARB2",
    "preracehorsecareerWIV",
    "preracehorsecareerWAX",
    "preracehorsecareerWOA",
    "preracehorsecareerCWO",
    "preracehorsecareerORR2",
    "preracehorsecareerWins",
    "preracehorsecareerRuns",
    "preracehorsecareerPlaces",
    # Horse recent form (lagged windows)
    "LRNFP",
    "LR3NFPtotal",
    "LR5NFPtotal",
    "LR10NFPtotal",
    "LR3_RWO",
    "LR5_RWO",
    "LR10_RWO",
    "LR_ORR2",
    # EPF (early position / running style)
    "Horse_Career_EPF",
    "LR_EPF",
    "LR2_EPF",
    "LR3_EPF",
    "LR_EPF2",
    "LR2_EPF2",
    "LR3_EPF2",
    # Pace
    "horsepaceindex",
    # Stability metrics
    "FSS",
    "FCS",
    # Market-derived historical (from PRIOR races, not today's race)
    "PFD3",
    "PFD5",
    "PFD10",
    "OFS3",
    "OFS5",
    "OFS10",
    # Prize money
    "WPMRF3",
    "WPMRF5",
    "WPMRF10",
    "PMW3",
    "PMW5",
    "PMW10",
    # Timing
    "FinalDSLR",
    "DSLR1",
    # Confidence / data availability
    "CIL3",
    "CIL5",
    "CIL10",
    "LR3COUNT",
    "LR5COUNT",
    "LR10COUNT",
    # Jockey career (lagged)
    "preracejockeycareerWins",
    "preracejockeycareerRuns",
    "preracejockeycareerWIV",
    "preracejockeycareerWAX",
    "preracejockeycareerWOA",
    "Jockey_Career_EPF",
    "jockeypaceindex",
    "totalLRPjockeyindex",
    # Trainer career (lagged)
    "preracetrainercareerWins",
    "preracetrainercareerRuns",
    "preracetrainercareerWIV",
    "preracetrainercareerWAX",
    "preracetrainercareerWOA",
    "trainer_Career_EPF",
    "trainerpaceindex",
    # Trainer-Jockey combo
    "trainerjockeycareerWIV",
    "trainerjockeycareerNFP",
    # Race context (known from declarations)
    "number_of_runners",
    "dist_furlongs",
    # Race-level strength (avg of pre-race horse metrics across field)
    "RACE_RB",
    "RACE_WIV",
    "RACE_NFP",
    "RACE_Wins",
    "RACE_WOA",
    # Within-race ranks (relative position vs field)
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
    # Known pre-race attributes
    "horse_age",
    "official_rating",
    "pounds",
    "stall",
    "days_since_lr",
    "career_runs",
    "jockeys_claim",
    "max_or_in_race",
    "median_or",
]


def load_data(db_path: str) -> pd.DataFrame:
    """Load race results from SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create log(BFSP) target, filtering invalid rows."""
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["log_bfsp"] = np.log(df["bfsp"])
    return df


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:
    """Train LightGBM regression model for log(BFSP)."""
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    callbacks = [
        lgb.log_evaluation(period=200),
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

    return model


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate regression predictions on log-scale and BFSP-scale."""
    mae_log = mean_absolute_error(y_true, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_log = r2_score(y_true, y_pred)

    bfsp_true = np.exp(y_true)
    bfsp_pred = np.exp(y_pred)
    mae_bfsp = mean_absolute_error(bfsp_true, bfsp_pred)
    pct_errors = np.abs(bfsp_true - bfsp_pred) / bfsp_true
    median_ape = np.median(pct_errors) * 100

    return {
        "mae_log": round(mae_log, 6),
        "rmse_log": round(rmse_log, 6),
        "r2_log": round(r2_log, 6),
        "mae_bfsp": round(mae_bfsp, 4),
        "median_ape_pct": round(median_ape, 2),
    }


def evaluate_overlays(
    predictions_df: pd.DataFrame,
    min_edge: float = 0.05,
    commission: float = 0.05,
) -> dict:
    """Evaluate overlay detection and simulated betting P&L.

    An overlay is when actual BFSP > predicted fair price (market thinks
    the horse is less likely than our model does).

    Overlay edge = (actual_bfsp / predicted_bfsp) - 1
    Positive edge = the market is offering better odds than we think fair.
    """
    df = predictions_df.copy()

    # Predicted fair BFSP vs actual BFSP
    df["predicted_bfsp"] = np.exp(df["pred_log_bfsp"])
    df["actual_bfsp"] = df["bfsp"]

    # Edge: positive when actual > predicted (market undervalues horse)
    df["overlay_edge"] = (df["actual_bfsp"] / df["predicted_bfsp"]) - 1

    # Identify overlay bets (positive edge above threshold)
    overlays = df[df["overlay_edge"] >= min_edge].copy()

    if len(overlays) == 0:
        return {
            "n_overlays": 0,
            "n_total": len(df),
            "overlay_pct": 0,
        }

    # Simulated flat-stake betting (£10 per bet)
    stake = 10.0
    overlays["returns"] = np.where(
        overlays["placing_numerical"] == 1,
        stake * overlays["actual_bfsp"] * (1 - commission),
        0,
    )
    overlays["pnl"] = overlays["returns"] - stake

    total_staked = len(overlays) * stake
    total_returns = overlays["returns"].sum()
    total_pnl = overlays["pnl"].sum()
    n_winners = (overlays["placing_numerical"] == 1).sum()

    # By edge bucket
    edge_buckets = []
    for lo, hi, label in [
        (0.05, 0.10, "5-10%"),
        (0.10, 0.20, "10-20%"),
        (0.20, 0.50, "20-50%"),
        (0.50, 1.00, "50-100%"),
        (1.00, 100.0, "100%+"),
    ]:
        bucket = overlays[
            (overlays["overlay_edge"] >= lo) & (overlays["overlay_edge"] < hi)
        ]
        if len(bucket) > 0:
            b_winners = (bucket["placing_numerical"] == 1).sum()
            b_pnl = bucket["pnl"].sum()
            edge_buckets.append({
                "edge_range": label,
                "n_bets": len(bucket),
                "n_winners": int(b_winners),
                "strike_rate": round(b_winners / len(bucket) * 100, 1),
                "pnl": round(b_pnl, 2),
                "roi_pct": round(b_pnl / (len(bucket) * stake) * 100, 1),
            })

    return {
        "n_overlays": len(overlays),
        "n_total": len(df),
        "overlay_pct": round(len(overlays) / len(df) * 100, 1),
        "n_winners": int(n_winners),
        "strike_rate_pct": round(n_winners / len(overlays) * 100, 2),
        "total_staked": round(total_staked, 2),
        "total_returns": round(total_returns, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / total_staked * 100, 2),
        "avg_edge": round(overlays["overlay_edge"].mean() * 100, 1),
        "avg_winner_bfsp": round(
            overlays[overlays["placing_numerical"] == 1]["actual_bfsp"].mean(), 2
        ) if n_winners > 0 else 0,
        "edge_buckets": edge_buckets,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train BFSP prediction model with walk-forward validation"
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB, help="Path to SQLite database"
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
        "--val-window", type=int, default=60,
        help="Validation window in days",
    )
    parser.add_argument(
        "--step-days", type=int, default=60,
        help="Step between folds in days",
    )
    parser.add_argument(
        "--importance", action="store_true",
        help="Show feature importance after training",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)

    # Step 1: Load data
    log.info("Loading data...")
    raw_df = load_data(args.db)
    log.info(f"  {len(raw_df):,} rows loaded")

    # Step 2: Calculate custom metrics
    log.info("Calculating custom metrics...")
    engine = CustomMetricsEngine()
    df = engine.calculate_all(raw_df)
    df = create_target(df)
    log.info(f"  {len(df):,} rows with valid BFSP target")

    # Ensure numeric types for pre-race attributes
    for col in ["horse_age", "official_rating", "pounds", "stall",
                "days_since_lr", "career_runs", "jockeys_claim",
                "max_or_in_race", "median_or"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to available features
    feature_cols = [c for c in BFSP_FEATURES if c in df.columns]
    log.info(f"  Using {len(feature_cols)} / {len(BFSP_FEATURES)} features")

    # Step 3: Walk-forward validation
    from datetime import timedelta

    df = df.sort_values("race_date").reset_index(drop=True)
    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    folds = []
    train_end = min_date + timedelta(days=args.min_train_days)
    while train_end + timedelta(days=args.val_window) <= max_date:
        val_start = train_end
        val_end = val_start + timedelta(days=args.val_window)
        folds.append((train_end, val_start, val_end))
        train_end += timedelta(days=args.step_days)

    log.info(f"Created {len(folds)} walk-forward folds")

    all_predictions = []
    fold_metrics = []

    for i, (train_end, val_start, val_end) in enumerate(folds):
        log.info(
            f"Fold {i + 1}/{len(folds)}: "
            f"train < {train_end.date()}, "
            f"val {val_start.date()} to {val_end.date()}"
        )

        train_df = df[df["race_date"] < train_end].copy()
        val_df = df[
            (df["race_date"] >= val_start) & (df["race_date"] < val_end)
        ].copy()

        # Drop rows with missing target
        train_df = train_df.dropna(subset=["log_bfsp"])
        val_df = val_df.dropna(subset=["log_bfsp"])

        if len(train_df) < 500 or len(val_df) < 100:
            log.warning(f"  Skipping fold {i + 1}: insufficient data")
            continue

        X_train = train_df[feature_cols].astype(float)
        y_train = train_df["log_bfsp"]
        X_val = val_df[feature_cols].astype(float)
        y_val = val_df["log_bfsp"]

        model = train_lightgbm(X_train, y_train, X_val, y_val)

        # Predict
        y_pred = model.predict(X_val)
        reg_metrics = evaluate_regression(y_val.values, y_pred)

        # Store predictions for overlay analysis
        val_preds = val_df[
            ["race_date", "raceid", "horse_name", "bfsp", "placing_numerical",
             "log_bfsp", "number_of_runners"]
        ].copy()
        val_preds["pred_log_bfsp"] = y_pred
        all_predictions.append(val_preds)
        fold_metrics.append(reg_metrics)

        log.info(
            f"  MAE(log): {reg_metrics['mae_log']:.4f}, "
            f"R²: {reg_metrics['r2_log']:.4f}, "
            f"Median APE: {reg_metrics['median_ape_pct']:.1f}%, "
            f"trees: {model.best_iteration}"
        )

    if not all_predictions:
        log.error("No successful folds. Exiting.")
        sys.exit(1)

    # Step 4: Aggregate and evaluate
    combined = pd.concat(all_predictions, ignore_index=True)

    overall_reg = evaluate_regression(
        combined["log_bfsp"].values, combined["pred_log_bfsp"].values
    )

    log.info("=" * 60)
    log.info("WALK-FORWARD RESULTS — BFSP REGRESSION")
    log.info("=" * 60)
    log.info(f"  Folds:          {len(fold_metrics)}")
    log.info(f"  Predictions:    {len(combined):,}")
    log.info(f"  MAE (log):      {overall_reg['mae_log']:.4f}")
    log.info(f"  RMSE (log):     {overall_reg['rmse_log']:.4f}")
    log.info(f"  R² (log):       {overall_reg['r2_log']:.4f}")
    log.info(f"  MAE (BFSP):     {overall_reg['mae_bfsp']:.2f}")
    log.info(f"  Median APE:     {overall_reg['median_ape_pct']:.1f}%")

    # Step 5: Overlay detection and betting simulation
    log.info("\nOverlay Analysis:")
    for min_edge in [0.05, 0.10, 0.20]:
        overlay_results = evaluate_overlays(combined, min_edge=min_edge)
        log.info(
            f"  Edge >= {min_edge*100:.0f}%: "
            f"{overlay_results['n_overlays']:,} bets "
            f"({overlay_results['overlay_pct']:.1f}% of runners), "
            f"strike rate {overlay_results['strike_rate_pct']:.1f}%, "
            f"ROI {overlay_results['roi_pct']:+.1f}%"
        )

    # Detailed overlay results at 5% threshold
    overlay_5pct = evaluate_overlays(combined, min_edge=0.05)
    log.info("\nOverlay Breakdown (5% min edge):")
    for bucket in overlay_5pct.get("edge_buckets", []):
        log.info(
            f"  {bucket['edge_range']:>8s}: "
            f"{bucket['n_bets']:>6,} bets, "
            f"{bucket['strike_rate']:.1f}% SR, "
            f"ROI {bucket['roi_pct']:+.1f}%"
        )

    # Step 6: Train final model on all data
    log.info("\nTraining final model on all available data...")
    dates = df["race_date"].sort_values().unique()
    cutoff = dates[int(len(dates) * 0.9)]
    final_train = df[df["race_date"] < cutoff].dropna(subset=["log_bfsp"])
    final_val = df[df["race_date"] >= cutoff].dropna(subset=["log_bfsp"])

    X_ft = final_train[feature_cols].astype(float)
    y_ft = final_train["log_bfsp"]
    X_fv = final_val[feature_cols].astype(float)
    y_fv = final_val["log_bfsp"]

    final_model = train_lightgbm(X_ft, y_ft, X_fv, y_fv)
    final_metrics = evaluate_regression(
        y_fv.values, final_model.predict(X_fv)
    )

    # Feature importance
    if args.importance:
        importance = final_model.feature_importance(importance_type="gain")
        feat_names = final_model.feature_name()
        pairs = sorted(zip(feat_names, importance), key=lambda x: -x[1])
        log.info("\nTop 25 Features by Gain:")
        max_imp = pairs[0][1] if pairs else 1
        for name, imp in pairs[:25]:
            bar_len = int(30 * imp / max_imp)
            log.info(f"  {name:<35s} {'█' * bar_len} {imp:.0f}")

    # Step 7: Save artifacts
    log.info(f"\nSaving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, "bfsp_regression.lgb")
    final_model.save_model(model_path)

    # Save feature list
    meta = {
        "feature_cols": feature_cols,
        "target": "log_bfsp",
        "model_type": "lightgbm_regression",
        "final_metrics": final_metrics,
    }
    with open(os.path.join(args.output_dir, "bfsp_regression_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save full training summary
    summary = {
        "model_type": "bfsp_regression",
        "n_folds": len(fold_metrics),
        "feature_cols": feature_cols,
        "walk_forward_regression": overall_reg,
        "fold_metrics": fold_metrics,
        "overlay_analysis_5pct": overlay_5pct,
        "overlay_analysis_10pct": evaluate_overlays(combined, min_edge=0.10),
        "overlay_analysis_20pct": evaluate_overlays(combined, min_edge=0.20),
        "final_model_metrics": final_metrics,
        "data_range": {
            "min_date": str(df["race_date"].min().date()),
            "max_date": str(df["race_date"].max().date()),
            "total_rows": len(df),
        },
    }
    with open(
        os.path.join(args.output_dir, "bfsp_training_summary.json"), "w"
    ) as f:
        json.dump(summary, f, indent=2, default=str)

    # Print final summary
    print("\n" + "=" * 60)
    print("BFSP REGRESSION MODEL — TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Features:         {len(feature_cols)}")
    print(f"  Folds:            {len(fold_metrics)}")
    print(f"  R² (log-BFSP):    {overall_reg['r2_log']:.4f}")
    print(f"  MAE (log):        {overall_reg['mae_log']:.4f}")
    print(f"  Median APE:       {overall_reg['median_ape_pct']:.1f}%")
    print(f"\n  Overlay Results (5% min edge):")
    print(f"    Bets:           {overlay_5pct['n_overlays']:,}")
    print(f"    Strike rate:    {overlay_5pct['strike_rate_pct']:.1f}%")
    print(f"    ROI:            {overlay_5pct['roi_pct']:+.1f}%")
    print(f"    P&L:            £{overlay_5pct['total_pnl']:+,.2f}")
    print(f"\n  Model saved to:   {args.output_dir}")


if __name__ == "__main__":
    main()
