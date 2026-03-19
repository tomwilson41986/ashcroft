#!/usr/bin/env python3
"""
Backtest Bets — Populate S3 with historical bets and P&L for the dashboard.

Runs the prediction model over historical dates, uses actual BFSP as the
"Betfair price", finds value bets, settles them, and writes results to S3.

Usage:
    python backtest_bets.py --days 7
    python backtest_bets.py --days 30 --min-edge 10 --stake 10
"""

import argparse
import logging
import os
from datetime import date

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from db_schema import DB_PATH
from edge_scanner import BETFAIR_COMMISSION
from predict_bfsp_today import (
    load_bfsp_model,
    load_historical,
    prepare_and_predict,
)
from web.s3_store import save_bets, save_daily_pnl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_backtest(
    n_days: int = 7,
    min_edge: float = 10.0,
    stake: float = 10.0,
    min_price: float = 1.5,
    max_price: float = 30.0,
    start_date: str = "2020-01-01",
    db_path: str = DB_PATH,
):
    """Run backtest and write bets + daily_pnl to S3."""
    model, feature_cols = load_bfsp_model(MODEL_DIR)

    log.info(f"Loading historical data (from {start_date})...")
    historical = load_historical(db_path, start_date=start_date)
    log.info(f"  {len(historical):,} rows loaded")

    all_dates = sorted(historical["race_date"].dt.date.unique())
    test_dates = all_dates[-n_days:]
    log.info(f"Backtesting {len(test_dates)} days: {test_dates[0]} to {test_dates[-1]}")

    all_bets = []
    all_daily_pnl = []
    cumulative_pnl = 0.0
    bank = 1000.0

    for td in test_dates:
        history_before = historical[historical["race_date"].dt.date < td].copy()
        runners_on_date = historical[historical["race_date"].dt.date == td].copy()

        if len(history_before) < 100 or len(runners_on_date) == 0:
            continue

        try:
            preds = prepare_and_predict(history_before, runners_on_date, model, feature_cols, td)
        except Exception as e:
            log.warning(f"  Prediction failed for {td}: {e}")
            continue

        if preds.empty:
            continue

        # Use actual BFSP as the "Betfair back price"
        preds["betfair_back"] = pd.to_numeric(preds.get("bfsp", preds.get("actual_bfsp")), errors="coerce")
        preds["edge_pct"] = ((preds["betfair_back"] / preds["predicted_bfsp"]) - 1) * 100

        # Filter value bets
        has_odds = preds["betfair_back"].notna()
        edge_ok = preds["edge_pct"] >= min_edge
        price_ok = (preds["betfair_back"] >= min_price) & (preds["betfair_back"] <= max_price)
        prob_ok = preds["predicted_win_prob_norm"] >= 0.03

        value = preds[has_odds & edge_ok & price_ok & prob_ok].copy()
        if value.empty:
            continue

        day_bets = 0
        day_winners = 0
        day_pnl = 0.0
        day_staked = 0.0

        for _, row in value.iterrows():
            placing = row.get("placing_numerical")
            actual_bsp = row.get("betfair_back")
            won = pd.notna(placing) and placing == 1

            if won:
                gross = stake * (actual_bsp - 1)
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
                "race_date": td.isoformat(),
                "race_time": str(row.get("race_time", "")),
                "track": str(row.get("track", "")),
                "horse_name": str(row.get("horse_name", "")),
                "predicted_bfsp": round(float(row.get("predicted_bfsp", 0)), 2),
                "predicted_win_prob": round(float(row.get("predicted_win_prob_norm", 0)), 4),
                "betfair_back": round(float(actual_bsp), 2) if pd.notna(actual_bsp) else None,
                "edge_pct": round(float(row.get("edge_pct", 0)), 1),
                "bet_type": "BACK",
                "stake": stake,
                "price": round(float(actual_bsp), 2) if pd.notna(actual_bsp) else None,
                "status": status,
                "actual_bsp": round(float(actual_bsp), 2) if pd.notna(actual_bsp) else None,
                "placing": int(placing) if pd.notna(placing) else None,
                "gross_pnl": round(gross, 2),
                "commission": round(commission, 2),
                "net_pnl": round(net, 2),
            })

            day_bets += 1
            day_pnl += net
            day_staked += stake

        cumulative_pnl += day_pnl
        bank_start = bank
        bank += day_pnl
        roi = (day_pnl / day_staked * 100) if day_staked > 0 else 0

        if day_bets > 0:
            all_daily_pnl.append({
                "date": td.isoformat(),
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

        total_bets = len(all_bets)
        total_winners = sum(1 for b in all_bets if b["status"] == "WON")

        log.info(f"  {td}: {day_bets} bets, {day_winners}W, P&L {day_pnl:+.2f}, Bank {bank:.2f}")

    # Write to S3
    log.info(f"Writing {len(all_bets)} bets to S3...")
    save_bets(all_bets)

    log.info(f"Writing {len(all_daily_pnl)} daily P&L records to S3...")
    save_daily_pnl(all_daily_pnl)

    total_bets = len(all_bets)
    total_winners = sum(1 for b in all_bets if b["status"] == "WON")

    print(f"\n{'='*60}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Period:      {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    print(f"  Min edge:    {min_edge}%")
    print(f"  Stake:       £{stake:.2f}")
    print(f"  Total bets:  {total_bets}")
    if total_bets:
        print(f"  Winners:     {total_winners} ({total_winners/total_bets*100:.1f}%)")
        print(f"  Net P&L:     £{cumulative_pnl:+.2f}")
        print(f"  ROI:         {cumulative_pnl/(total_bets*stake)*100:+.1f}%")
    print(f"  Bank:        £{bank:.2f}")
    print(f"  Data written to S3 (dashboard/bets.json, dashboard/daily_pnl.json)")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest to populate S3 dashboard data")
    parser.add_argument("--days", type=int, default=7, help="Number of days to backtest")
    parser.add_argument("--min-edge", type=float, default=10.0, help="Minimum edge %%")
    parser.add_argument("--stake", type=float, default=10.0, help="Fixed stake per bet")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Earliest history date")
    args = parser.parse_args()

    run_backtest(
        n_days=args.days,
        min_edge=args.min_edge,
        stake=args.stake,
        start_date=args.start_date,
    )


if __name__ == "__main__":
    main()
