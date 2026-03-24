#!/usr/bin/env python3
"""
Review Predictions vs Actual Results for given dates.

Generates predictions from the database (no HRB login needed) and compares
them against actual BFSP/BSP and finishing positions.

Usage:
    python review_predictions.py --date 2026-03-22
    python review_predictions.py --date 2026-03-22 --date 2026-03-23
    python review_predictions.py --from 2026-03-22 --to 2026-03-23
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")


def get_race_results(db_path: str, target_date: str) -> pd.DataFrame:
    """Load actual race results for a date from the database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """SELECT horse_name, track, racetime, race_date, placing_numerical,
                  BFSP, odds, race_name, race_class, number_of_runners,
                  going_description, distance
           FROM race_results
           WHERE race_date = ?
           ORDER BY racetime, track, BFSP""",
        conn,
        params=[target_date],
    )
    conn.close()
    return df


def generate_predictions_for_date(db_path: str, target_date: date) -> pd.DataFrame:
    """Generate BFSP predictions for a date using the trained model."""
    try:
        import lightgbm as lgb
        from model.custom_metrics import CustomMetricsEngine
        from predict_bfsp_today import (
            get_runners_from_db,
            load_bfsp_model,
            load_historical,
            prepare_and_predict,
        )
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return pd.DataFrame()

    model, feature_cols = load_bfsp_model(MODEL_DIR)
    historical = load_historical(db_path, start_date="2020-01-01")

    target_runners = get_runners_from_db(db_path, str(target_date))
    if len(target_runners) == 0:
        log.warning(f"No runners found in DB for {target_date}")
        return pd.DataFrame()

    history_before = historical[
        historical["race_date"].dt.date < target_date
    ].copy()

    if len(history_before) < 100:
        log.warning("Limited history — using full dataset")
        history_before = historical.copy()

    preds = prepare_and_predict(
        history_before, target_runners, model, feature_cols, target_date
    )
    return preds


def print_date_summary(target_date: str, results: pd.DataFrame, predictions: pd.DataFrame):
    """Print a formatted summary for one date."""
    print(f"\n{'='*80}")
    print(f"  REVIEW: {target_date}")
    print(f"{'='*80}")

    if len(results) == 0:
        print(f"  No results found in database for {target_date}")
        print(f"  (Has the results scraper run? Try: python betfair_sync.py --bsp --from {target_date} --to {target_date})")
        return

    # Race summary
    races = results.groupby(["racetime", "track"]).agg(
        runners=("horse_name", "count"),
        winner=("placing_numerical", lambda x: results.loc[x.index[results.loc[x.index, "placing_numerical"] == 1], "horse_name"].values[0] if 1 in x.values else "N/A"),
        winner_bfsp=("BFSP", lambda x: results.loc[x.index[results.loc[x.index, "placing_numerical"] == 1], "BFSP"].values[0] if 1 in results.loc[x.index, "placing_numerical"].values else None),
    ).reset_index()

    num_races = len(races)
    num_runners = len(results)
    print(f"\n  Races: {num_races}  |  Runners: {num_runners}")

    # Show race-by-race results
    print(f"\n  {'Time':<8} {'Track':<18} {'Runners':>7}  {'Winner':<25} {'BFSP':>8}")
    print(f"  {'-'*72}")

    for _, race in races.iterrows():
        bfsp_str = f"{race['winner_bfsp']:.2f}" if pd.notna(race['winner_bfsp']) else "N/A"
        print(f"  {race['racetime']:<8} {race['track']:<18} {race['runners']:>7}  {race['winner']:<25} {bfsp_str:>8}")

    # Prediction accuracy (if predictions available)
    if len(predictions) > 0:
        print(f"\n  --- Prediction Analysis ---")
        print(f"  Predictions generated: {len(predictions)}")

        if "predicted_bfsp" in predictions.columns and "actual_bfsp" in predictions.columns:
            valid = predictions.dropna(subset=["predicted_bfsp", "actual_bfsp"])
            valid = valid[valid["actual_bfsp"] > 0]

            if len(valid) > 0:
                # Log-space correlation (more meaningful for prices)
                log_pred = np.log(valid["predicted_bfsp"])
                log_actual = np.log(valid["actual_bfsp"])
                corr = log_pred.corr(log_actual)

                # MAE on log scale
                log_mae = np.mean(np.abs(log_pred - log_actual))

                # Percentage error
                pct_errors = np.abs(valid["predicted_bfsp"] - valid["actual_bfsp"]) / valid["actual_bfsp"] * 100
                median_pct_error = np.median(pct_errors)
                mean_pct_error = np.mean(pct_errors)

                print(f"  Valid pred vs actual pairs: {len(valid)}")
                print(f"  Log-space correlation:      {corr:.4f}")
                print(f"  Log-space MAE:              {log_mae:.4f}")
                print(f"  Median % error:             {median_pct_error:.1f}%")
                print(f"  Mean % error:               {mean_pct_error:.1f}%")

        # Value betting analysis
        if "value_edge" in predictions.columns or "bfsp_diff_pct" in predictions.columns:
            edge_col = "value_edge" if "value_edge" in predictions.columns else "bfsp_diff_pct"
            has_placing = "placing_numerical" in predictions.columns

            overlay = predictions[predictions[edge_col] > 5].copy()
            if len(overlay) > 0 and has_placing:
                winners = overlay[overlay["placing_numerical"] == 1]
                strike_rate = len(winners) / len(overlay) * 100 if len(overlay) > 0 else 0

                print(f"\n  --- Value Bets (edge > 5%) ---")
                print(f"  Selections: {len(overlay)}")
                print(f"  Winners:    {len(winners)}")
                print(f"  Strike rate: {strike_rate:.1f}%")

                if len(overlay) > 0:
                    print(f"\n  {'Time':<8} {'Track':<16} {'Horse':<22} {'Pred':>7} {'Actual':>7} {'Edge':>6} {'Pos':>4}")
                    print(f"  {'-'*74}")
                    for _, row in overlay.sort_values(edge_col, ascending=False).head(20).iterrows():
                        pred = f"{row['predicted_bfsp']:.1f}" if pd.notna(row.get('predicted_bfsp')) else "?"
                        actual = f"{row.get('actual_bfsp', row.get('BFSP', 0)):.1f}" if pd.notna(row.get('actual_bfsp', row.get('BFSP'))) else "?"
                        edge = f"{row[edge_col]:.1f}%"
                        pos = str(int(row['placing_numerical'])) if pd.notna(row.get('placing_numerical')) else "?"
                        time_str = str(row.get('race_time', ''))[:7]
                        print(f"  {time_str:<8} {str(row.get('track', ''))[:15]:<16} {str(row['horse_name'])[:21]:<22} {pred:>7} {actual:>7} {edge:>6} {pos:>4}")

        # Top ranked predictions (shortest predicted BFSP per race = predicted favourite)
        if has_placing := "placing_numerical" in predictions.columns:
            # Rank within each race
            pred_copy = predictions.copy()
            pred_copy["rank"] = pred_copy.groupby(
                ["race_time", "track"]
            )["predicted_bfsp"].rank(method="first")

            top_picks = pred_copy[pred_copy["rank"] == 1].copy()
            if len(top_picks) > 0:
                tp_winners = top_picks[top_picks["placing_numerical"] == 1]
                tp_placed = top_picks[top_picks["placing_numerical"].isin([1, 2, 3])]

                print(f"\n  --- Top Ranked (Predicted Favourite) ---")
                print(f"  Selections: {len(top_picks)}")
                print(f"  Winners:    {len(tp_winners)} ({len(tp_winners)/len(top_picks)*100:.0f}%)")
                print(f"  Placed 1-3: {len(tp_placed)} ({len(tp_placed)/len(top_picks)*100:.0f}%)")

                print(f"\n  {'Time':<8} {'Track':<16} {'Horse':<22} {'Pred':>7} {'Actual':>7} {'Pos':>4} {'Result'}")
                print(f"  {'-'*74}")
                for _, row in top_picks.sort_values("race_time").iterrows():
                    pred = f"{row['predicted_bfsp']:.1f}" if pd.notna(row.get('predicted_bfsp')) else "?"
                    actual = f"{row.get('actual_bfsp', row.get('BFSP', 0)):.1f}" if pd.notna(row.get('actual_bfsp', row.get('BFSP'))) else "?"
                    pos = int(row['placing_numerical']) if pd.notna(row.get('placing_numerical')) else "?"
                    result = "WIN" if pos == 1 else f"#{pos}" if isinstance(pos, int) else "?"
                    time_str = str(row.get('race_time', ''))[:7]
                    print(f"  {time_str:<8} {str(row.get('track', ''))[:15]:<16} {str(row['horse_name'])[:21]:<22} {pred:>7} {actual:>7} {str(pos):>4} {result}")

    else:
        print(f"\n  No predictions generated (model files may be missing)")

    # P&L from bets table (if it exists)
    try:
        conn = sqlite3.connect(DB_PATH)
        bets = pd.read_sql_query(
            "SELECT * FROM bets WHERE race_date = ? ORDER BY race_time",
            conn,
            params=[target_date],
        )
        conn.close()

        if len(bets) > 0:
            settled = bets[bets["status"].isin(["WON", "LOST", "VOID"])]
            pending = bets[bets["status"] == "PENDING"]
            total_pnl = settled["net_pnl"].sum() if "net_pnl" in settled.columns else 0
            total_staked = settled["stake"].sum() if "stake" in settled.columns else 0
            roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
            winners = settled[settled["status"] == "WON"]

            print(f"\n  --- P&L (from bets table) ---")
            print(f"  Bets: {len(bets)}  |  Settled: {len(settled)}  |  Pending: {len(pending)}")
            print(f"  Record: {len(winners)}W / {len(settled) - len(winners)}L")
            print(f"  Staked: {total_staked:.2f}  |  Net P&L: {total_pnl:+.2f}  |  ROI: {roi:+.1f}%")
    except Exception:
        pass  # bets table may not exist


def print_combined_summary(all_predictions: list[pd.DataFrame]):
    """Print a combined summary across all dates."""
    if not all_predictions or all(len(p) == 0 for p in all_predictions):
        return

    combined = pd.concat([p for p in all_predictions if len(p) > 0], ignore_index=True)

    print(f"\n{'='*80}")
    print(f"  COMBINED SUMMARY")
    print(f"{'='*80}")

    if "predicted_bfsp" in combined.columns and "actual_bfsp" in combined.columns:
        valid = combined.dropna(subset=["predicted_bfsp", "actual_bfsp"])
        valid = valid[valid["actual_bfsp"] > 0]

        if len(valid) > 0:
            log_pred = np.log(valid["predicted_bfsp"])
            log_actual = np.log(valid["actual_bfsp"])
            corr = log_pred.corr(log_actual)
            log_mae = np.mean(np.abs(log_pred - log_actual))
            pct_errors = np.abs(valid["predicted_bfsp"] - valid["actual_bfsp"]) / valid["actual_bfsp"] * 100

            print(f"  Total predictions:     {len(valid)}")
            print(f"  Log-space correlation: {corr:.4f}")
            print(f"  Log-space MAE:         {log_mae:.4f}")
            print(f"  Median % error:        {np.median(pct_errors):.1f}%")

    if "placing_numerical" in combined.columns:
        combined_copy = combined.copy()
        combined_copy["rank"] = combined_copy.groupby(
            ["race_date", "race_time", "track"]
        )["predicted_bfsp"].rank(method="first")

        top_picks = combined_copy[combined_copy["rank"] == 1]
        if len(top_picks) > 0:
            tp_winners = top_picks[top_picks["placing_numerical"] == 1]
            tp_placed = top_picks[top_picks["placing_numerical"].isin([1, 2, 3])]
            print(f"\n  Top picks: {len(tp_winners)}/{len(top_picks)} winners ({len(tp_winners)/len(top_picks)*100:.0f}%)")
            print(f"  Top picks placed: {len(tp_placed)}/{len(top_picks)} ({len(tp_placed)/len(top_picks)*100:.0f}%)")

        if "value_edge" in combined.columns or "bfsp_diff_pct" in combined.columns:
            edge_col = "value_edge" if "value_edge" in combined.columns else "bfsp_diff_pct"
            overlay = combined[combined[edge_col] > 5]
            if len(overlay) > 0:
                ov_winners = overlay[overlay["placing_numerical"] == 1]
                print(f"\n  Value bets (edge>5%): {len(ov_winners)}/{len(overlay)} winners ({len(ov_winners)/len(overlay)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Review predictions vs results")
    parser.add_argument(
        "--date", type=str, action="append", default=None,
        help="Date(s) to review (YYYY-MM-DD). Can specify multiple.",
    )
    parser.add_argument("--from", type=str, dest="from_date", help="Start date")
    parser.add_argument("--to", type=str, dest="to_date", help="End date")
    parser.add_argument("--db", type=str, default=DB_PATH)
    parser.add_argument(
        "--results-only", action="store_true",
        help="Only show results (skip prediction generation)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        log.error(f"Database not found: {args.db}")
        log.error("Run: bash scripts/setup_session.sh --db")
        sys.exit(1)

    # Build list of dates
    dates = []
    if args.date:
        dates = [date.fromisoformat(d) for d in args.date]
    elif args.from_date and args.to_date:
        d = date.fromisoformat(args.from_date)
        end = date.fromisoformat(args.to_date)
        while d <= end:
            dates.append(d)
            d += timedelta(days=1)
    else:
        parser.error("Specify --date or --from/--to")

    all_predictions = []

    for target_date in dates:
        target_str = target_date.isoformat()

        results = get_race_results(args.db, target_str)

        if args.results_only:
            predictions = pd.DataFrame()
        else:
            log.info(f"Generating predictions for {target_str}...")
            predictions = generate_predictions_for_date(args.db, target_date)
            all_predictions.append(predictions)

        print_date_summary(target_str, results, predictions)

    if len(dates) > 1:
        print_combined_summary(all_predictions)

    print()


if __name__ == "__main__":
    main()
