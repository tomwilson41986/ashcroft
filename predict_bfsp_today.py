#!/usr/bin/env python3
"""
Generate BFSP Predictions for Today's Races.

Fetches today's race cards from horseracebase.com, calculates all custom
metrics on the historical database, builds feature vectors for declared
runners, and predicts BFSP using the trained regression model.

Usage:
    python predict_bfsp_today.py                         # Predict today
    python predict_bfsp_today.py --date 2026-03-10       # Specific date
    python predict_bfsp_today.py --dry-run               # Skip email
    python predict_bfsp_today.py --from-db               # Use DB data (no HRB login)

Environment variables (in .env):
    HRB_USERNAME        - horseracebase.com login
    HRB_PASSWORD        - horseracebase.com password
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import date, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from model.custom_metrics import CustomMetricsEngine
from train_bfsp import (
    ALL_FEATURE_COLS,
    BFSPTrainer,
    build_context_features,
)

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_historical(db_path: str, start_date: str | None = None) -> pd.DataFrame:
    """Load historical race results, optionally filtered by start date."""
    conn = sqlite3.connect(db_path)
    if start_date:
        df = pd.read_sql_query(
            "SELECT * FROM race_results WHERE race_date >= ? ORDER BY race_date, race_time",
            conn,
            params=[start_date],
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM race_results ORDER BY race_date, race_time", conn
        )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def load_bfsp_model(model_dir: str) -> tuple[lgb.Booster, list[str]]:
    """Load the trained BFSP regression model and its feature columns."""
    model_path = os.path.join(model_dir, "bfsp_model.lgb")
    meta_path = os.path.join(model_dir, "bfsp_model_meta.json")

    if not os.path.exists(model_path):
        log.error(f"BFSP model not found at {model_path}")
        log.error("Run 'python train_bfsp.py' first to train the model.")
        sys.exit(1)

    model = lgb.Booster(model_file=model_path)

    feature_cols = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta.get("feature_cols", [])

    if not feature_cols:
        feature_cols = [c for c in ALL_FEATURE_COLS]

    log.info(f"Loaded BFSP model ({len(feature_cols)} features)")
    return model, feature_cols


# ---------------------------------------------------------------------------
# Race Card Fetching
# ---------------------------------------------------------------------------

def fetch_racecard_from_hrb(target_date: date) -> pd.DataFrame:
    """Fetch today's race card from horseracebase.com."""
    try:
        from daily_predictions import (
            create_session,
            get_todays_runners,
            login,
        )
    except ImportError:
        log.error("Could not import daily_predictions module")
        return pd.DataFrame()

    session = create_session()
    user_id = login(session)
    if not user_id:
        log.error("Failed to login to horseracebase.com")
        log.error("Set HRB_USERNAME and HRB_PASSWORD in .env file")
        return pd.DataFrame()

    runners = get_todays_runners(session, user_id, target_date)
    return runners


def get_runners_from_db(db_path: str, target_date: str) -> pd.DataFrame:
    """Get runners for a specific date from the local database.

    Used when HRB credentials aren't available — simulates predictions
    on historical race cards stored in the database.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results WHERE race_date = ? ORDER BY race_time",
        conn,
        params=[target_date],
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


# ---------------------------------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------------------------------

def prepare_and_predict(
    historical_df: pd.DataFrame,
    target_runners: pd.DataFrame,
    model: lgb.Booster,
    feature_cols: list[str],
    target_date: date,
) -> pd.DataFrame:
    """Calculate metrics on history, build features for runners, predict BFSP.

    This is the core prediction function that:
    1. Concatenates history with target runners (if not already in history)
    2. Runs CustomMetricsEngine on the full dataset
    3. Extracts feature vectors for target date runners
    4. Predicts log(BFSP) and converts to BFSP
    """
    engine = CustomMetricsEngine()

    # Check if target runners are already in the historical data
    target_date_str = str(target_date)
    runners_in_db = (
        historical_df["race_date"].dt.strftime("%Y-%m-%d") == target_date_str
    ).any()

    if runners_in_db:
        # Runners are in the database — use the full dataset
        log.info("Target date found in database, using full historical data")
        full_df = historical_df.copy()
    else:
        # Append target runners to history for metric calculation
        log.info("Appending target runners to historical data")
        target_runners = target_runners.copy()
        target_runners["race_date"] = pd.to_datetime(target_date)

        # Ensure compatible schemas
        for col in historical_df.columns:
            if col not in target_runners.columns:
                target_runners[col] = np.nan
        for col in target_runners.columns:
            if col not in historical_df.columns:
                historical_df[col] = np.nan

        full_df = pd.concat(
            [historical_df, target_runners], ignore_index=True
        )

    # Calculate all custom metrics on the full dataset
    log.info("Calculating all custom metrics...")
    full_df = engine.calculate_all(full_df)

    # Build context features
    log.info("Building context features...")
    full_df = build_context_features(full_df)

    # Extract rows for the target date
    full_df["race_date"] = pd.to_datetime(full_df["race_date"])
    target_mask = full_df["race_date"].dt.strftime("%Y-%m-%d") == target_date_str
    target_df = full_df[target_mask].copy()

    if len(target_df) == 0:
        log.error(f"No runners found for {target_date}")
        return pd.DataFrame()

    log.info(f"Predicting BFSP for {len(target_df)} runners on {target_date}")

    # Ensure all features exist
    for col in feature_cols:
        if col not in target_df.columns:
            target_df[col] = np.nan

    # Predict
    X = target_df[feature_cols].astype(float)
    log_bfsp_pred = model.predict(X)

    target_df["predicted_log_bfsp"] = log_bfsp_pred
    target_df["predicted_bfsp"] = np.exp(log_bfsp_pred)

    # Calculate implied win probability from predicted BFSP
    target_df["predicted_win_prob"] = 1.0 / target_df["predicted_bfsp"]

    # Normalise probabilities per race
    if "raceid" in target_df.columns:
        target_df["predicted_win_prob_norm"] = target_df.groupby("raceid")[
            "predicted_win_prob"
        ].transform(lambda x: x / x.sum())
    else:
        # Create raceid from track + time
        target_df["raceid"] = (
            target_df["race_date"].dt.strftime("%Y-%m-%d")
            + "_" + target_df["track"].astype(str)
            + "_" + target_df["race_time"].astype(str)
        )
        target_df["predicted_win_prob_norm"] = target_df.groupby("raceid")[
            "predicted_win_prob"
        ].transform(lambda x: x / x.sum())

    # If actual BFSP is available, compute edge
    if "bfsp" in target_df.columns:
        actual_bfsp = pd.to_numeric(target_df["bfsp"], errors="coerce")
        target_df["actual_bfsp"] = actual_bfsp
        target_df["bfsp_diff"] = target_df["predicted_bfsp"] - actual_bfsp
        target_df["bfsp_diff_pct"] = (
            (target_df["predicted_bfsp"] - actual_bfsp) / actual_bfsp * 100
        )
        # Value detection: model thinks horse is shorter (more likely to win)
        # than the market suggests
        target_df["value_edge"] = (
            actual_bfsp / target_df["predicted_bfsp"] - 1
        ) * 100

    return target_df


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def format_predictions(predictions: pd.DataFrame, target_date: date) -> str:
    """Format predictions into a readable report."""
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(
        f"  BFSP PREDICTIONS — {target_date.strftime('%A %d %B %Y')}"
    )
    lines.append("=" * 78)

    if len(predictions) == 0:
        lines.append("  No runners found.")
        return "\n".join(lines)

    n_races = predictions["raceid"].nunique() if "raceid" in predictions.columns else 0
    lines.append(
        f"  {len(predictions)} runners across {n_races} races"
    )
    lines.append("")

    group_col = "raceid" if "raceid" in predictions.columns else None
    has_actual = "actual_bfsp" in predictions.columns and predictions["actual_bfsp"].notna().any()

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            race_df = race_df.sort_values("predicted_bfsp")

            track = race_df["track"].iloc[0] if "track" in race_df.columns else "?"
            rtime = race_df["race_time"].iloc[0] if "race_time" in race_df.columns else "?"
            n_run = len(race_df)

            lines.append(f"  {track} — {rtime} ({n_run} runners)")
            lines.append("  " + "-" * 74)

            if has_actual:
                lines.append(
                    f"  {'#':<3} {'Horse':<22} {'Pred BFSP':>10} "
                    f"{'Actual':>8} {'Diff%':>7} {'P(Win)':>7} "
                    f"{'Value':>7} {'Result':>7}"
                )
            else:
                lines.append(
                    f"  {'#':<3} {'Horse':<22} {'Pred BFSP':>10} "
                    f"{'P(Win)':>7} {'Rank':>5}"
                )
            lines.append("  " + "-" * 74)

            for rank, (_, row) in enumerate(race_df.iterrows(), 1):
                horse = str(row.get("horse_name", "?"))[:21]
                pred_bfsp = row.get("predicted_bfsp", 0)
                p_win = row.get("predicted_win_prob_norm", 0)

                if has_actual:
                    actual = row.get("actual_bfsp", np.nan)
                    diff_pct = row.get("bfsp_diff_pct", np.nan)
                    value = row.get("value_edge", np.nan)
                    placing = row.get("placing_numerical", np.nan)

                    actual_str = f"{actual:.1f}" if pd.notna(actual) else "-"
                    diff_str = f"{diff_pct:+.1f}%" if pd.notna(diff_pct) else "-"
                    value_str = f"{value:+.1f}%" if pd.notna(value) else "-"

                    if pd.notna(placing) and placing == 1:
                        result_str = "  WON"
                    elif pd.notna(placing) and placing <= 3:
                        result_str = f"  {int(placing)}nd" if placing == 2 else f"  {int(placing)}rd"
                    elif pd.notna(placing):
                        result_str = f"  {int(placing)}th"
                    else:
                        result_str = "  -"

                    lines.append(
                        f"  {rank:<3} {horse:<22} {pred_bfsp:>10.2f} "
                        f"{actual_str:>8} {diff_str:>7} {p_win:>6.1%} "
                        f"{value_str:>7} {result_str:>7}"
                    )
                else:
                    lines.append(
                        f"  {rank:<3} {horse:<22} {pred_bfsp:>10.2f} "
                        f"{p_win:>6.1%} {rank:>5}"
                    )

            lines.append("")

    # Top picks
    lines.append("  TOP SELECTIONS (lowest predicted BFSP per race)")
    lines.append("  " + "=" * 74)

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            top = race_df.sort_values("predicted_bfsp").iloc[0]
            track = top.get("track", "?")
            rtime = top.get("race_time", "?")
            horse = top.get("horse_name", "?")
            pred_bfsp = top.get("predicted_bfsp", 0)
            p_win = top.get("predicted_win_prob_norm", 0)

            extra = ""
            if has_actual:
                actual = top.get("actual_bfsp", np.nan)
                placing = top.get("placing_numerical", np.nan)
                if pd.notna(actual):
                    extra += f" (Actual: {actual:.1f}"
                    if pd.notna(placing):
                        result = "WON" if placing == 1 else f"Finished {int(placing)}"
                        extra += f", {result}"
                    extra += ")"

            lines.append(
                f"  {track} {rtime}: {horse} "
                f"— Pred BFSP {pred_bfsp:.2f}, P(Win) {p_win:.1%}{extra}"
            )

    # Value bets (horses whose predicted BFSP is shorter than actual)
    if has_actual:
        value_df = predictions[
            predictions["value_edge"].notna() & (predictions["value_edge"] > 5)
        ].sort_values("value_edge", ascending=False)

        if len(value_df) > 0:
            lines.append("")
            lines.append("  VALUE BETS (Model shorter than market by >5%)")
            lines.append("  " + "=" * 74)
            for _, row in value_df.head(20).iterrows():
                horse = row.get("horse_name", "?")
                track = row.get("track", "?")
                rtime = row.get("race_time", "?")
                pred = row.get("predicted_bfsp", 0)
                actual = row.get("actual_bfsp", 0)
                edge = row.get("value_edge", 0)
                placing = row.get("placing_numerical", np.nan)

                result = ""
                if pd.notna(placing):
                    result = " -> WON" if placing == 1 else f" -> {int(placing)}th"

                lines.append(
                    f"  {track} {rtime}: {horse} "
                    f"(Pred {pred:.2f} vs Actual {actual:.2f}, "
                    f"Edge {edge:+.1f}%){result}"
                )

    # Accuracy summary if actual BFSP available
    if has_actual:
        actual_vals = predictions["actual_bfsp"].dropna()
        pred_vals = predictions.loc[actual_vals.index, "predicted_bfsp"]

        if len(actual_vals) > 0:
            mae = np.mean(np.abs(pred_vals - actual_vals))
            median_ape = np.median(np.abs(pred_vals - actual_vals) / actual_vals) * 100
            corr = np.corrcoef(np.log(pred_vals), np.log(actual_vals))[0, 1]

            # How often does our top pick (lowest predicted BFSP) win?
            top_pick_wins = 0
            total_races = 0
            for _, race_df in predictions.groupby(group_col):
                if race_df["placing_numerical"].notna().any():
                    total_races += 1
                    top = race_df.sort_values("predicted_bfsp").iloc[0]
                    if top.get("placing_numerical") == 1:
                        top_pick_wins += 1

            lines.append("")
            lines.append("  ACCURACY SUMMARY")
            lines.append("  " + "=" * 74)
            lines.append(f"  BFSP MAE:           {mae:.2f}")
            lines.append(f"  Median APE:         {median_ape:.1f}%")
            lines.append(f"  Log correlation:    {corr:.4f}")
            if total_races > 0:
                lines.append(
                    f"  Top pick wins:      {top_pick_wins}/{total_races} "
                    f"({top_pick_wins/total_races:.1%})"
                )

    lines.append("")
    lines.append("  " + "-" * 74)
    lines.append("  Generated by BFSP Prediction Model (all custom metrics)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate BFSP predictions for today's races"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR,
        help="Path to model artifacts",
    )
    parser.add_argument(
        "--from-db", action="store_true",
        help="Use runners from the database (no HRB login needed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print predictions to stdout (no email)",
    )
    parser.add_argument(
        "--last-n-days", type=int, default=None,
        help="Run predictions on the last N days in the database",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Save predictions to CSV file",
    )
    parser.add_argument(
        "--start-date", type=str, default="2020-01-01",
        help="Earliest historical date to load (default: 2020-01-01). Reduces memory.",
    )
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )

    if not os.path.exists(args.db):
        log.error(f"Database not found: {args.db}")
        log.error("Run 'python train_bfsp.py --generate-data' first.")
        sys.exit(1)

    # Load model
    model, feature_cols = load_bfsp_model(args.model_dir)

    # Load historical data
    log.info(f"Loading historical data (from {args.start_date})...")
    historical = load_historical(args.db, start_date=args.start_date)
    log.info(f"  {len(historical):,} historical rows loaded")

    if args.last_n_days:
        # Run backtesting over last N days
        log.info(f"Running predictions for last {args.last_n_days} days...")
        all_dates = sorted(historical["race_date"].dt.date.unique())
        test_dates = all_dates[-args.last_n_days:]

        all_predictions = []
        for td in test_dates:
            # Use all data before this date as history
            history_before = historical[
                historical["race_date"].dt.date < td
            ].copy()
            runners_on_date = historical[
                historical["race_date"].dt.date == td
            ].copy()

            if len(history_before) < 100 or len(runners_on_date) == 0:
                continue

            preds = prepare_and_predict(
                history_before, runners_on_date, model, feature_cols, td
            )
            if len(preds) > 0:
                all_predictions.append(preds)

        if all_predictions:
            combined = pd.concat(all_predictions, ignore_index=True)
            report = format_predictions(combined, target_date)
            print(report)

            if args.output_csv:
                output_cols = [
                    "race_date", "race_time", "track", "horse_name",
                    "predicted_bfsp", "actual_bfsp", "bfsp_diff_pct",
                    "predicted_win_prob_norm", "value_edge",
                    "placing_numerical",
                ]
                avail = [c for c in output_cols if c in combined.columns]
                combined[avail].to_csv(args.output_csv, index=False)
                log.info(f"Saved predictions to {args.output_csv}")
        else:
            log.warning("No predictions generated")
        return

    # Single date prediction
    if args.from_db:
        log.info(f"Getting runners from database for {target_date}...")
        target_runners = get_runners_from_db(args.db, str(target_date))
    else:
        # Try HRB login
        username = os.getenv("HRB_USERNAME")
        if username:
            log.info(f"Fetching race card from horseracebase.com for {target_date}...")
            target_runners = fetch_racecard_from_hrb(target_date)
        else:
            log.warning(
                "No HRB credentials found. Using database data. "
                "Set HRB_USERNAME and HRB_PASSWORD in .env for live race cards."
            )
            target_runners = get_runners_from_db(args.db, str(target_date))

    if len(target_runners) == 0:
        # Fall back to latest date in DB
        latest_date = historical["race_date"].max().date()
        log.warning(
            f"No runners found for {target_date}. "
            f"Falling back to latest DB date: {latest_date}"
        )
        target_date = latest_date
        target_runners = get_runners_from_db(args.db, str(target_date))

    if len(target_runners) == 0:
        log.error("No runners found in database.")
        sys.exit(1)

    log.info(f"Found {len(target_runners)} runners for {target_date}")

    # Separate history (before target date) from target runners
    history_before = historical[
        historical["race_date"].dt.date < target_date
    ].copy()

    if len(history_before) < 100:
        log.warning("Limited history — using full dataset for metrics")
        history_before = historical.copy()

    # Generate predictions
    predictions = prepare_and_predict(
        history_before, target_runners, model, feature_cols, target_date
    )

    if len(predictions) == 0:
        log.error("No predictions generated")
        sys.exit(1)

    # Format and display
    report = format_predictions(predictions, target_date)
    print(report)

    # Save to CSV if requested
    if args.output_csv:
        output_cols = [
            "race_date", "race_time", "track", "horse_name",
            "predicted_bfsp", "actual_bfsp", "bfsp_diff_pct",
            "predicted_win_prob_norm", "value_edge",
            "placing_numerical",
        ]
        avail = [c for c in output_cols if c in predictions.columns]
        predictions[avail].to_csv(args.output_csv, index=False)
        log.info(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
