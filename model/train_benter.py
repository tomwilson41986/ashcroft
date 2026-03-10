"""
Train Benter-inspired conditional logit model with walk-forward validation.

Benter principles adapted to our data (WITHOUT BFSP as input):
1. Conditional logit (multinomial) — models within-race competition directly
2. Probabilities sum to 1 per race (softmax normalisation)
3. Race-relative features (within-race ranks capture field composition)
4. Temporal walk-forward validation (no lookahead)
5. Fair BFSP derived from model: BFSP_fair = 1 / P(win)
6. Overlays: when actual BFSP > BFSP_fair, the market underestimates the horse

Two model types trained in parallel:
- Conditional logit (Benter-style softmax, linear in features)
- LightGBM regression on log(BFSP) (nonlinear, captures interactions)

Usage:
    python -m model.train_benter                       # Train with defaults
    python -m model.train_benter --db horse_racing.db  # Specify database
    python -m model.train_benter --tune                # Tune regularisation
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
from sklearn.metrics import log_loss

from model.conditional_logit import BenterConditionalLogit
from model.custom_metrics import CustomMetricsEngine
from model.evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")

# Features for conditional logit (pre-race only, NO BFSP).
# Includes Benter-style within-race ranks and race strength metrics.
CLOGIT_FEATURES = [
    # Horse career metrics (lagged)
    "preracehorsecareerNFP",
    "preracehorsecareerRB",
    "preracehorsecareerFSARB",
    "preracehorsecareerWIV",
    "preracehorsecareerWAX",
    "preracehorsecareerWOA",
    "preracehorsecareerCWO",
    "preracehorsecareerORR2",
    "preracehorsecareerWins",
    "preracehorsecareerRuns",
    "preracehorsecareerPlaces",
    # Horse recent form
    "LRNFP",
    "LR3NFPtotal",
    "LR5NFPtotal",
    "LR10NFPtotal",
    "LR3_RWO",
    "LR5_RWO",
    "LR10_RWO",
    "LR_ORR2",
    # Pace and style
    "Horse_Career_EPF",
    "horsepaceindex",
    # Stability
    "FSS",
    "FCS",
    # Historical market-derived (from PRIOR races, not today's)
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
    # Confidence
    "CIL3",
    "CIL5",
    "CIL10",
    # Jockey
    "preracejockeycareerWIV",
    "preracejockeycareerWAX",
    "preracejockeycareerWOA",
    "Jockey_Career_EPF",
    "jockeypaceindex",
    "totalLRPjockeyindex",
    # Trainer
    "preracetrainercareerWIV",
    "preracetrainercareerWAX",
    "preracetrainercareerWOA",
    "trainer_Career_EPF",
    "trainerpaceindex",
    # Trainer-Jockey combo
    "trainerjockeycareerWIV",
    "trainerjockeycareerNFP",
    # Within-race ranks (Benter's key insight — relative position matters)
    "rNFP",
    "rNFPLR5",
    "horseWIVrank",
    "horseRBrank",
    "rRWOLR5",
    "rRWOLR10",
    "rPFD5",
    "rFSS",
    "rFCS",
    # Race context
    "number_of_runners",
    "dist_furlongs",
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


def create_folds(
    df: pd.DataFrame,
    min_train_days: int = 365,
    val_window_days: int = 60,
    step_days: int = 60,
) -> list[tuple]:
    """Create temporal walk-forward folds."""
    df = df.sort_values("race_date")
    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    folds = []
    train_end = min_date + timedelta(days=min_train_days)

    while train_end + timedelta(days=val_window_days) <= max_date:
        val_start = train_end
        val_end = val_start + timedelta(days=val_window_days)
        folds.append((train_end, val_start, val_end))
        train_end += timedelta(days=step_days)

    return folds


def tune_alpha(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, dict]:
    """Find best regularisation alpha via grid search."""
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    best_ll = float("inf")
    best_alpha = 1.0
    results = {}

    for alpha in alphas:
        model = BenterConditionalLogit(alpha=alpha, max_iter=300)
        metrics = model.train(train_df, val_df, feature_cols)
        if "error" in metrics:
            continue

        ll = metrics["val_logloss"]
        imp = metrics["improvement_vs_random"]
        results[alpha] = {"logloss": ll, "improvement": imp}
        log.info(
            f"  alpha={alpha:>5.1f}: log-loss={ll:.6f}, "
            f"improvement vs random={imp:+.6f}"
        )

        if ll < best_ll:
            best_ll = ll
            best_alpha = alpha

    return best_alpha, results


def evaluate_overlays(
    predictions_df: pd.DataFrame,
    min_edge: float = 0.05,
    commission: float = 0.05,
) -> dict:
    """Evaluate overlay detection: where actual BFSP > model fair BFSP.

    The model predicts P(win), giving a fair BFSP = 1/P(win).
    If actual BFSP > fair BFSP, the market thinks the horse is less
    likely to win than our model does — this is an overlay (value bet).
    """
    df = predictions_df.copy()

    if "bfsp" not in df.columns or "predicted_bfsp" not in df.columns:
        return {"error": "missing columns"}

    # Overlay edge
    df["overlay_edge"] = (df["bfsp"] / df["predicted_bfsp"]) - 1

    overlays = df[df["overlay_edge"] >= min_edge].copy()

    if len(overlays) == 0:
        return {"n_overlays": 0, "n_total": len(df), "roi_pct": 0}

    # Flat-stake simulation (£10 per bet)
    stake = 10.0
    overlays["returns"] = np.where(
        overlays["won"] == 1,
        stake * overlays["bfsp"] * (1 - commission),
        0,
    )
    overlays["pnl"] = overlays["returns"] - stake
    total_staked = len(overlays) * stake
    total_pnl = overlays["pnl"].sum()
    n_winners = (overlays["won"] == 1).sum()

    # By edge bucket
    buckets = []
    for lo, hi, label in [
        (0.05, 0.10, "5-10%"),
        (0.10, 0.20, "10-20%"),
        (0.20, 0.50, "20-50%"),
        (0.50, 1.00, "50-100%"),
        (1.00, 100.0, "100%+"),
    ]:
        b = overlays[
            (overlays["overlay_edge"] >= lo) & (overlays["overlay_edge"] < hi)
        ]
        if len(b) > 0:
            b_w = (b["won"] == 1).sum()
            b_pnl = b["pnl"].sum()
            buckets.append({
                "edge_range": label,
                "n_bets": len(b),
                "n_winners": int(b_w),
                "strike_rate": round(b_w / len(b) * 100, 1),
                "pnl": round(b_pnl, 2),
                "roi_pct": round(b_pnl / (len(b) * stake) * 100, 1),
            })

    return {
        "n_overlays": len(overlays),
        "n_total": len(df),
        "overlay_pct": round(len(overlays) / len(df) * 100, 1),
        "n_winners": int(n_winners),
        "strike_rate_pct": round(n_winners / len(overlays) * 100, 2),
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / total_staked * 100, 2),
        "avg_edge": round(overlays["overlay_edge"].mean() * 100, 1),
        "edge_buckets": buckets,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Benter conditional logit model"
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
        "--tune", action="store_true",
        help="Tune alpha regularisation",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)

    # Step 1: Load data
    log.info("Loading data...")
    df = load_data(args.db)
    log.info(f"  {len(df):,} rows loaded")

    # Step 2: Calculate custom metrics
    log.info("Calculating custom metrics...")
    engine = CustomMetricsEngine()
    df = engine.calculate_all(df)
    df["won"] = (df["placing_numerical"] == 1).astype(float)
    log.info(f"  {len(df):,} rows with metrics")

    # Filter features
    feature_cols = [c for c in CLOGIT_FEATURES if c in df.columns]
    log.info(f"  Using {len(feature_cols)} features")

    # Step 3: Walk-forward
    folds = create_folds(
        df, args.min_train_days, args.val_window, args.step_days,
    )
    log.info(f"Created {len(folds)} walk-forward folds")

    all_predictions = []
    fold_metrics = []

    # Alpha tuning on first fold
    best_alpha = 1.0
    if args.tune and len(folds) >= 2:
        log.info("Tuning alpha on first fold...")
        te, vs, ve = folds[0]
        best_alpha, _ = tune_alpha(
            df[df["race_date"] < te].copy(),
            df[(df["race_date"] >= vs) & (df["race_date"] < ve)].copy(),
            feature_cols,
        )
        log.info(f"  Best alpha: {best_alpha}")
        folds = folds[1:]

    # Walk-forward
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

        if len(train_df) < 500 or len(val_df) < 100:
            log.warning(f"  Skipping: insufficient data")
            continue

        model = BenterConditionalLogit(alpha=best_alpha, max_iter=500)
        metrics = model.train(train_df, val_df, feature_cols)

        if "error" in metrics:
            continue

        # Predict on validation
        val_preds = model.predict(val_df)
        all_predictions.append(val_preds)
        fold_metrics.append(metrics)

        log.info(
            f"  Log-loss: {metrics['val_logloss']:.6f} "
            f"(random: {metrics['random_logloss']:.6f}, "
            f"improvement: {metrics['improvement_vs_random']:+.6f})"
        )

    if not all_predictions:
        log.error("No successful folds.")
        sys.exit(1)

    combined = pd.concat(all_predictions, ignore_index=True)

    # Overall metrics
    y_true = combined["won"].values
    y_pred = combined["p_model"].clip(1e-7, 1 - 1e-7).values
    overall_ll = log_loss(y_true, y_pred)

    log.info("=" * 60)
    log.info("WALK-FORWARD RESULTS (Benter Conditional Logit)")
    log.info("=" * 60)
    log.info(f"  Folds:           {len(fold_metrics)}")
    log.info(f"  Predictions:     {len(combined):,}")
    log.info(f"  Model log-loss:  {overall_ll:.6f}")

    avg_random_ll = np.mean([m["random_logloss"] for m in fold_metrics])
    log.info(f"  Random log-loss: {avg_random_ll:.6f}")
    log.info(f"  Improvement:     {avg_random_ll - overall_ll:+.6f}")

    # Overlay analysis (compare model predicted BFSP vs actual BFSP)
    if "bfsp" in combined.columns:
        log.info("\nOverlay Analysis:")
        for min_edge in [0.05, 0.10, 0.20]:
            ov = evaluate_overlays(combined, min_edge=min_edge)
            log.info(
                f"  Edge >= {min_edge*100:.0f}%: "
                f"{ov['n_overlays']:,} bets "
                f"({ov.get('overlay_pct', 0):.1f}%), "
                f"SR {ov.get('strike_rate_pct', 0):.1f}%, "
                f"ROI {ov.get('roi_pct', 0):+.1f}%"
            )

        overlay_5 = evaluate_overlays(combined, min_edge=0.05)
        log.info("\nOverlay Breakdown (5% min edge):")
        for b in overlay_5.get("edge_buckets", []):
            log.info(
                f"  {b['edge_range']:>8s}: {b['n_bets']:>6,} bets, "
                f"{b['strike_rate']:.1f}% SR, ROI {b['roi_pct']:+.1f}%"
            )

    # Full evaluation
    log.info("\nRunning full evaluation...")
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate(combined)

    # Train final model
    log.info("Training final model...")
    dates = df["race_date"].sort_values().unique()
    cutoff = dates[int(len(dates) * 0.9)]
    final_model = BenterConditionalLogit(alpha=best_alpha, max_iter=500)
    final_metrics = final_model.train(
        df[df["race_date"] < cutoff].copy(),
        df[df["race_date"] >= cutoff].copy(),
        feature_cols,
    )

    # Save
    log.info(f"Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    final_model.save(args.output_dir)

    summary = {
        "model_type": "benter_conditional_logit",
        "n_folds": len(fold_metrics),
        "best_alpha": best_alpha,
        "feature_cols": feature_cols,
        "walk_forward_metrics": fold_metrics,
        "final_model_metrics": final_metrics,
        "overall": {
            "model_logloss": round(overall_ll, 6),
            "n_predictions": len(combined),
        },
        "overlay_5pct": evaluate_overlays(combined, 0.05) if "bfsp" in combined.columns else {},
        "overlay_10pct": evaluate_overlays(combined, 0.10) if "bfsp" in combined.columns else {},
        "evaluation": eval_results,
        "data_range": {
            "min_date": str(df["race_date"].min().date()),
            "max_date": str(df["race_date"].max().date()),
            "total_rows": len(df),
        },
    }
    with open(
        os.path.join(args.output_dir, "benter_training_summary.json"), "w"
    ) as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("BENTER CONDITIONAL LOGIT — SUMMARY")
    print("=" * 60)
    print(f"  Folds:            {len(fold_metrics)}")
    print(f"  Model log-loss:   {overall_ll:.6f}")
    print(f"  Alpha:            {best_alpha}")
    if "bfsp" in combined.columns:
        ov = evaluate_overlays(combined, 0.05)
        print(f"\n  Overlays (5% edge):")
        print(f"    Bets: {ov['n_overlays']:,}, SR: {ov.get('strike_rate_pct',0):.1f}%")
        print(f"    ROI:  {ov.get('roi_pct',0):+.1f}%, P&L: £{ov.get('total_pnl',0):+,.2f}")
    print(f"\n  Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
