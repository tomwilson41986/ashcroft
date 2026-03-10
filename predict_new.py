#!/usr/bin/env python3
"""
Generate predictions for upcoming races using the trained model.

Loads the saved model from data/models/, builds features from the
database, and runs the full 3-stage pipeline:
  1. FundamentalModel → win probabilities
  2. BenterBlender → blended probabilities with market
  3. OverlayDetector → value bets with Kelly staking

Usage:
    python predict_new.py                           # Latest races in DB
    python predict_new.py --date 2025-12-30         # Specific date
    python predict_new.py --track Cheltenham        # Filter by track
    python predict_new.py --date 2025-12-30 --top 5 # Top 5 picks only
    python predict_new.py --bet-card                # Show only qualifying bets
"""

import argparse
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

from model.benter_blend import BenterBlender
from model.custom_metrics import CustomMetricsEngine
from model.overlay_detector import OverlayDetector
from model.probability_model import FundamentalModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")


def load_model(model_dir: str) -> tuple[FundamentalModel, dict]:
    """Load trained model and its configuration."""
    model = FundamentalModel.load(model_dir)

    blend_path = os.path.join(model_dir, "blend_config.json")
    if os.path.exists(blend_path):
        with open(blend_path) as f:
            blend_config = json.load(f)
    else:
        blend_config = {"optimal_lambda": 0.80}

    return model, blend_config


def load_data(db_path: str) -> pd.DataFrame:
    """Load race results from SQLite."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def print_predictions(df: pd.DataFrame, top_n: int | None = None):
    """Pretty-print prediction results grouped by race."""
    if len(df) == 0:
        print("No predictions to show.")
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    races = df.groupby(["race_date", "track", "race_time"])
    for (date, track, time), race_df in races:
        race_df = race_df.sort_values("p_model", ascending=False)
        if top_n:
            race_df = race_df.head(top_n)

        print(f"\n{'=' * 80}")
        print(f"  {date.strftime('%Y-%m-%d')}  {time}  {track}")
        print(f"  {race_df.iloc[0].get('race_name', '')}  "
              f"({int(race_df.iloc[0]['number_of_runners'])} runners)")
        print(f"{'=' * 80}")
        print(f"  {'#':<3} {'Horse':<22} {'P(win)':>8} {'Fair BSP':>10} "
              f"{'Mkt BSP':>10} {'Edge%':>8} {'Verdict':>10}")
        print(f"  {'-' * 74}")

        for rank, (_, row) in enumerate(race_df.iterrows(), 1):
            p_model = row.get("p_combined", row.get("p_model", 0))
            fair_bfsp = row.get("model_fair_bfsp", 1 / p_model if p_model > 0 else 0)
            mkt_bfsp = row.get("bfsp", 0)
            edge_pct = row.get("edge_pct", 0)

            if edge_pct > 10:
                verdict = "STRONG"
            elif edge_pct > 5:
                verdict = "VALUE"
            elif edge_pct > 0:
                verdict = "slight"
            else:
                verdict = ""

            print(f"  {rank:<3} {row['horse_name']:<22} {p_model:>8.1%} "
                  f"{fair_bfsp:>10.2f} {mkt_bfsp:>10.2f} {edge_pct:>7.1f}% "
                  f"{verdict:>10}")

    print()


def print_bet_card(bet_card: pd.DataFrame):
    """Pretty-print the qualifying bets."""
    if len(bet_card) == 0:
        print("\nNo qualifying bets found (no overlays exceeding minimum edge).")
        return

    print(f"\n{'=' * 90}")
    print(f"  BET CARD — {len(bet_card)} qualifying bets")
    print(f"{'=' * 90}")
    print(f"  {'Track':<14} {'Time':<6} {'Horse':<22} {'P(win)':>7} "
          f"{'BSP':>7} {'Fair':>7} {'Edge%':>7} {'Stake':>8} {'EV':>8}")
    print(f"  {'-' * 86}")

    total_staked = 0
    total_ev = 0

    for _, row in bet_card.iterrows():
        print(f"  {row.get('track', '?'):<14} "
              f"{row.get('race_time', '?'):<6} "
              f"{row['horse_name']:<22} "
              f"{row['p_combined']:>6.1%} "
              f"{row['bfsp']:>7.2f} "
              f"{row['model_fair_bfsp']:>7.2f} "
              f"{row['edge_pct']:>6.1f}% "
              f"£{row['stake_gbp']:>7.2f} "
              f"£{row['expected_value']:>7.2f}")
        total_staked += row["stake_gbp"]
        total_ev += row["expected_value"]

    print(f"  {'-' * 86}")
    print(f"  {'TOTAL':<50} {' ' * 22} "
          f"£{total_staked:>7.2f} £{total_ev:>7.2f}")
    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions using the trained model"
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB, help="Path to SQLite database"
    )
    parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR, help="Model artifacts directory"
    )
    parser.add_argument(
        "--date", type=str, help="Race date to predict (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--track", type=str, help="Filter by track name"
    )
    parser.add_argument(
        "--horse", type=str, help="Filter by horse name"
    )
    parser.add_argument(
        "--top", type=int, help="Show only top N picks per race"
    )
    parser.add_argument(
        "--bet-card", action="store_true",
        help="Show only qualifying value bets with Kelly stakes"
    )
    parser.add_argument(
        "--min-edge", type=float, default=5.0,
        help="Minimum edge %% for bet card (default 5)"
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Bankroll for Kelly staking (default £1000)"
    )
    args = parser.parse_args()

    # Check model exists
    model_file = os.path.join(args.model_dir, "probability_model.lgb")
    if not os.path.exists(model_file):
        print(f"No trained model found at {args.model_dir}")
        print("Run 'python retrain.py' first to train a model.")
        sys.exit(1)

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)

    # Load model
    print("Loading model...")
    model, blend_config = load_model(args.model_dir)
    optimal_lambda = blend_config.get("optimal_lambda", 0.80)
    print(f"  Features: {len(model.feature_cols)}")
    print(f"  Blend lambda: {optimal_lambda}")

    # Load data and compute metrics
    print("Loading data and computing features...")
    df = load_data(args.db)

    engine = CustomMetricsEngine()
    df = engine.calculate_all(df)

    # Create raceid and won columns
    if "raceid" not in df.columns:
        df["raceid"] = (
            df["race_date"].astype(str) + "_" + df["track"] + "_" + df["race_time"]
        )
    if "won" not in df.columns:
        df["won"] = (df["placing_numerical"] == 1).astype(int)

    # Apply filters
    if args.date:
        df = df[df["race_date"] == pd.Timestamp(args.date)]
    elif not args.horse:
        # Default: latest race date in the database
        latest = df["race_date"].max()
        df = df[df["race_date"] == latest]
        print(f"  Using latest date: {latest.strftime('%Y-%m-%d')}")

    if args.track:
        df = df[df["track"].str.contains(args.track, case=False, na=False)]
    if args.horse:
        # Find races containing this horse, show full race
        horse_races = df[
            df["horse_name"].str.contains(args.horse, case=False, na=False)
        ]["raceid"].unique()
        df = df[df["raceid"].isin(horse_races)]

    if len(df) == 0:
        print("No matching races found.")
        sys.exit(0)

    print(f"  {len(df)} runners across {df['raceid'].nunique()} races")

    # Stage 1: Predict win probabilities
    print("Running predictions...")
    available_features = [f for f in model.feature_cols if f in df.columns]
    missing_features = [f for f in model.feature_cols if f not in df.columns]
    if missing_features:
        print(f"  Warning: {len(missing_features)} features missing, filling with NaN")
        for f in missing_features:
            df[f] = np.nan

    predictions = model.predict(df, raceid_col="raceid")

    # Stage 2: Blend with market
    blender = BenterBlender()
    blender.optimal_lambda = optimal_lambda
    predictions = blender.blend_with_optimal(predictions)

    # Stage 3: Detect overlays
    if args.bet_card:
        detector = OverlayDetector(
            min_edge=args.min_edge / 100.0,
            bankroll=args.bankroll,
        )
        bet_card = detector.generate_bet_card(predictions)
        print_bet_card(bet_card)
    else:
        print_predictions(predictions, top_n=args.top)

    # Summary stats
    n_races = predictions["raceid"].nunique()
    n_overlays = (predictions["edge_pct"] > args.min_edge).sum()
    print(f"  Summary: {n_races} races, {len(predictions)} runners, "
          f"{n_overlays} overlays (>{args.min_edge}% edge)")


if __name__ == "__main__":
    main()
