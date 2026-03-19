#!/usr/bin/env python3
"""
Populate dashboard with real prediction data.

- Runs the trained BFSP model on recent dates from the database
- Settles predictions against actual results (BFSP + placing)
- Reads live pipeline predictions from S3 for unsettled dates
- Writes everything to the dashboard S3 keys

Usage:
    python populate_dashboard.py                     # Last 3 settled days + pending
    python populate_dashboard.py --days 7            # Last 7 settled days + pending
    python populate_dashboard.py --min-edge 10       # Minimum edge for value bets
"""

import argparse
import io
import json
import logging
import os
import sqlite3
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local")

from edge_scanner import BETFAIR_COMMISSION
from web.s3_store import save_bets, save_daily_pnl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def get_settled_dates(db_path: str, n_days: int) -> list[str]:
    """Get the last N dates that have both BFSP and placing data in the DB."""
    conn = sqlite3.connect(db_path)
    dates_df = pd.read_sql_query(
        """
        SELECT DISTINCT race_date FROM race_results
        WHERE bfsp IS NOT NULL AND bfsp > 0
          AND placing_numerical IS NOT NULL
        ORDER BY race_date DESC
        LIMIT ?
        """,
        conn,
        params=(n_days,),
    )
    conn.close()
    return sorted(dates_df["race_date"].tolist())


def run_model_predictions(target_date_str: str, db_path: str) -> pd.DataFrame:
    """Run the real BFSP model on a date from the database.

    Returns DataFrame with predicted_bfsp, predicted_win_prob_norm,
    value_edge, actual_bfsp, placing_numerical etc.
    """
    from predict_bfsp_today import load_historical, load_bfsp_model, prepare_and_predict

    target_date = date.fromisoformat(target_date_str)
    model, feature_cols = load_bfsp_model(MODEL_DIR)
    historical = load_historical(db_path, start_date="2020-01-01")

    history_before = historical[historical["race_date"].dt.date < target_date].copy()
    runners = historical[historical["race_date"].dt.date == target_date].copy()

    if len(runners) == 0 or len(history_before) < 100:
        return pd.DataFrame()

    preds = prepare_and_predict(history_before, runners, model, feature_cols, target_date)
    return preds


def read_s3_predictions(dt_str: str) -> pd.DataFrame:
    """Read pipeline predictions from S3 for a given date."""
    import boto3

    bucket = os.getenv("ULTRA_BETTING_S3_BUCKET", "ashcroft")
    key = f"predictions/{dt_str}.csv"

    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.StringIO(obj["Body"].read().decode()))
    except Exception as e:
        log.debug(f"No S3 predictions for {dt_str}: {e}")
        return pd.DataFrame()


def run(
    n_days: int = 3,
    min_edge: float = 10.0,
    stake: float = 10.0,
    min_price: float = 1.5,
    max_price: float = 30.0,
    db_path: str = DB_PATH,
):
    """Generate dashboard data from real model predictions."""
    all_bets = []
    all_daily_pnl = []
    cumulative_pnl = 0.0
    bank = 1000.0

    # --- Part 1: Settled dates (real model predictions vs actual results) ---
    settled_dates = get_settled_dates(db_path, n_days)
    log.info(f"Found {len(settled_dates)} settled dates: {settled_dates}")

    for race_date in settled_dates:
        log.info(f"Running real model predictions for {race_date}...")
        preds = run_model_predictions(race_date, db_path)

        if preds.empty:
            log.warning(f"  No predictions for {race_date}")
            continue

        day_bets = 0
        day_winners = 0
        day_pnl = 0.0
        day_staked = 0.0

        # Compute rank per race (by predicted_bfsp ascending — rank 1 = shortest odds)
        if "race_time" in preds.columns and "track" in preds.columns:
            preds["race_key"] = preds["race_date"].astype(str) + "|" + preds["race_time"].astype(str) + "|" + preds["track"].astype(str)
        else:
            preds["race_key"] = preds.index.astype(str)
        preds["rank"] = preds.groupby("race_key")["predicted_bfsp"].rank(method="first").astype(int)

        for _, row in preds.iterrows():
            pred_bfsp = row.get("predicted_bfsp", 0)
            if pd.isna(pred_bfsp) or pred_bfsp <= 0:
                continue
            actual_bfsp = row.get("actual_bfsp", 0)
            if pd.isna(actual_bfsp) or actual_bfsp <= 0:
                continue
            if actual_bfsp < min_price or actual_bfsp > max_price:
                continue

            edge = row.get("value_edge", 0)
            if pd.isna(edge):
                edge = 0
            win_prob = row.get("predicted_win_prob_norm", 0)
            if pd.isna(win_prob):
                win_prob = 0

            placing = row.get("placing_numerical")
            won = pd.notna(placing) and placing == 1

            if won:
                gross = stake * (actual_bfsp - 1)
                commission = gross * BETFAIR_COMMISSION
                net = gross - commission
                status = "WON"
                day_winners += 1
            else:
                gross = -stake
                commission = 0.0
                net = -stake
                status = "LOST"

            all_bets.append({
                "race_date": race_date,
                "race_time": str(row.get("race_time", "")),
                "track": str(row.get("track", "")),
                "horse_name": str(row.get("horse_name", "")),
                "predicted_bfsp": round(float(pred_bfsp), 2),
                "predicted_win_prob": round(float(win_prob), 4),
                "betfair_back": round(float(actual_bfsp), 2),
                "edge_pct": round(float(edge), 1),
                "bet_type": "BACK",
                "stake": stake,
                "price": round(float(actual_bfsp), 2),
                "status": status,
                "actual_bsp": round(float(actual_bfsp), 2),
                "placing": int(placing) if pd.notna(placing) else None,
                "gross_pnl": round(gross, 2),
                "commission": round(commission, 2),
                "net_pnl": round(net, 2),
                "rank": int(row.get("rank", 0)),
            })

            day_bets += 1
            day_pnl += net
            day_staked += stake

        if day_bets > 0:
            cumulative_pnl += day_pnl
            bank_start = bank
            bank += day_pnl
            roi = (day_pnl / day_staked * 100) if day_staked > 0 else 0

            all_daily_pnl.append({
                "date": race_date,
                "num_bets": day_bets,
                "winners": day_winners,
                "losers": day_bets - day_winners,
                "voids": 0,
                "total_staked": round(day_staked, 2),
                "gross_pnl": round(day_pnl, 2),
                "commission": 0,
                "net_pnl": round(day_pnl, 2),
                "roi_pct": round(roi, 2),
                "cumulative_pnl": round(cumulative_pnl, 2),
                "bank_start": round(bank_start, 2),
                "bank_end": round(bank, 2),
            })
            log.info(f"  {race_date}: {day_bets} bets, {day_winners}W, P&L {day_pnl:+.2f}")

    # --- Part 2: Pending predictions from S3 (not yet settled) ---
    today = date.today()
    for day_offset in range(2, -1, -1):
        dt = today - timedelta(days=day_offset)
        dt_str = str(dt)

        # Skip dates we already have settled data for
        if dt_str in [d for d in [b.get("race_date") for b in all_bets]]:
            continue

        s3_preds = read_s3_predictions(dt_str)
        if s3_preds.empty:
            continue

        log.info(f"Adding {len(s3_preds)} pending predictions from S3 for {dt_str}")

        # Compute rank for S3 predictions
        if not s3_preds.empty:
            bfsp_col = "predicted_bfsp" if "predicted_bfsp" in s3_preds.columns else None
            time_col = "race_time" if "race_time" in s3_preds.columns else None
            track_col = next((c for c in ["venue", "track"] if c in s3_preds.columns), None)
            if bfsp_col and time_col and track_col:
                s3_preds["_race_key"] = dt_str + "|" + s3_preds[time_col].astype(str) + "|" + s3_preds[track_col].astype(str)
                s3_preds["_rank"] = s3_preds.groupby("_race_key")[bfsp_col].rank(method="first").astype(int)
            else:
                s3_preds["_rank"] = 0

        for _, row in s3_preds.iterrows():
            pred_bfsp = row.get("predicted_bfsp")
            win_prob = row.get("predicted_win_prob")

            all_bets.append({
                "race_date": dt_str,
                "race_time": str(row.get("race_time", "")),
                "track": str(row.get("venue", row.get("track", ""))),
                "horse_name": str(row.get("runner_name", row.get("horse_name", ""))),
                "predicted_bfsp": round(float(pred_bfsp), 2) if pd.notna(pred_bfsp) else None,
                "predicted_win_prob": round(float(win_prob), 4) if pd.notna(win_prob) else None,
                "betfair_back": None,
                "edge_pct": None,
                "bet_type": "BACK",
                "stake": stake,
                "price": None,
                "status": "PENDING",
                "actual_bsp": None,
                "placing": None,
                "gross_pnl": 0,
                "commission": 0,
                "net_pnl": 0,
                "rank": int(row.get("_rank", 0)),
            })

    # Sort and write
    all_bets.sort(key=lambda b: (b.get("race_date", ""), b.get("race_time", "")))
    all_daily_pnl.sort(key=lambda p: p.get("date", ""))

    log.info(f"Writing {len(all_bets)} bets to S3...")
    save_bets(all_bets)

    log.info(f"Writing {len(all_daily_pnl)} daily P&L records to S3...")
    save_daily_pnl(all_daily_pnl)

    settled = [b for b in all_bets if b["status"] in ("WON", "LOST")]
    pending = [b for b in all_bets if b["status"] == "PENDING"]
    total_winners = sum(1 for b in settled if b["status"] == "WON")

    print(f"\n{'='*60}")
    print(f"  DASHBOARD DATA — REAL PREDICTIONS")
    print(f"{'='*60}")
    if settled_dates:
        print(f"  Settled:     {settled_dates[0]} to {settled_dates[-1]} ({len(settled_dates)} days)")
    print(f"  Settled bets: {len(settled)}")
    if settled:
        print(f"  Winners:     {total_winners} ({total_winners/len(settled)*100:.1f}%)")
        print(f"  Net P&L:     £{cumulative_pnl:+.2f}")
        total_staked = sum(b.get("stake", 0) for b in settled)
        print(f"  ROI:         {cumulative_pnl/total_staked*100:+.1f}%")
    print(f"  Pending:     {len(pending)} predictions")
    print(f"  Bank:        £{bank:.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Populate dashboard with real prediction data")
    parser.add_argument("--days", type=int, default=3, help="Number of settled days to include")
    parser.add_argument("--min-edge", type=float, default=10.0, help="Minimum edge %%")
    parser.add_argument("--stake", type=float, default=10.0, help="Stake per bet")
    args = parser.parse_args()

    run(n_days=args.days, min_edge=args.min_edge, stake=args.stake)


if __name__ == "__main__":
    main()
