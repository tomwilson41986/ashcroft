"""
Predict BFSP for upcoming races using a trained model.

Usage:
    python -m model.predict --date 2025-03-15         # Predict for a specific date
    python -m model.predict --horse "Frankel"          # Predict for a specific horse
    python -m model.predict --date 2025-03-15 --track "Cheltenham"
"""

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.features import FEATURE_COLS, build_features, load_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
MODEL_PATH = os.path.join(PROJECT_DIR, "bfsp_model.lgb")


def predict_bfsp(
    model: lgb.Booster, df: pd.DataFrame
) -> pd.DataFrame:
    """Add predicted BFSP to a features DataFrame.

    Normalises implied probabilities (1/BFSP) per race so they sum to 1,
    then derives fair-book BFSP from the normalised probabilities.
    """
    X = df[FEATURE_COLS]
    log_pred = model.predict(X)
    df = df.copy()
    df["predicted_log_bfsp"] = log_pred
    df["predicted_bfsp_raw"] = np.exp(log_pred)

    # Build raceid if missing
    if "raceid" not in df.columns:
        df["raceid"] = (
            df["race_date"].dt.strftime("%Y-%m-%d")
            + "_" + df["track"].astype(str)
            + "_" + df["race_time"].astype(str)
        )

    # Normalise implied probabilities per race
    df["implied_prob"] = 1.0 / df["predicted_bfsp_raw"]
    race_prob_sum = df.groupby("raceid")["implied_prob"].transform("sum")
    df["predicted_win_prob_norm"] = df["implied_prob"] / race_prob_sum
    df["predicted_bfsp"] = 1.0 / df["predicted_win_prob_norm"]
    df.drop(columns=["implied_prob"], inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Predict BFSP for races")
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB, help="Path to SQLite database"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to trained model"
    )
    parser.add_argument("--date", type=str, help="Race date to predict (YYYY-MM-DD)")
    parser.add_argument("--track", type=str, help="Filter by track name")
    parser.add_argument("--horse", type=str, help="Filter by horse name")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Run 'python -m model.train' first.")
        sys.exit(1)

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)

    # Load model
    model = lgb.Booster(model_file=args.model)

    # Load and build features for all data
    print("Loading data and building features...")
    df = load_data(args.db)
    df = build_features(df)

    # Apply filters
    if args.date:
        df = df[df["race_date"] == pd.Timestamp(args.date)]
    if args.track:
        df = df[df["track"].str.contains(args.track, case=False, na=False)]
    if args.horse:
        df = df[df["horse_name"].str.contains(args.horse, case=False, na=False)]

    if len(df) == 0:
        print("No matching races found.")
        sys.exit(0)

    # Predict
    df = predict_bfsp(model, df)

    # Display results
    display_cols = [
        "race_date", "race_time", "track", "horse_name",
        "official_rating", "jockey_name", "trainer",
        "bfsp", "predicted_bfsp",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    result = df[available_cols].copy()

    if "bfsp" in result.columns and "predicted_bfsp" in result.columns:
        result["diff"] = result["predicted_bfsp"] - result["bfsp"]
        result["diff_pct"] = (
            (result["predicted_bfsp"] - result["bfsp"]) / result["bfsp"] * 100
        ).round(1)

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    print(f"\n  Predictions ({len(result)} entries)")
    print("  " + "=" * 80)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
