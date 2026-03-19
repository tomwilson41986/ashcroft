#!/usr/bin/env python3
"""
Backtest Bets — Populate bets and P&L tables from historical data.

Runs the prediction model over historical dates, uses actual BFSP as the
"Betfair price", finds value bets, and settles them to build a realistic
P&L history for the web dashboard.

Usage:
    python backtest_bets.py --days 7           # Last 7 days in DB
    python backtest_bets.py --days 30 --min-edge 10 --stake 10
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import date

import numpy as np
import pandas as pd

from db_schema import ensure_tables, DB_PATH
from edge_scanner import BETFAIR_COMMISSION
from predict_bfsp_today import (
    load_bfsp_model,
    load_historical,
    prepare_and_predict,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def normalise_name(name: str) -> str:
    import re
    return re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", name.strip().lower()).strip()


def run_backtest(
    n_days: int = 7,
    min_edge: float = 10.0,
    stake: float = 10.0,
    min_price: float = 1.5,
    max_price: float = 30.0,
    start_date: str = "2020-01-01",
    db_path: str = DB_PATH,
):
    """Run backtest over the last N days and populate bets + daily_pnl tables."""
    ensure_tables(db_path)

    # Clear existing backtest data
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM bets")
    conn.execute("DELETE FROM daily_pnl")
    conn.commit()
    conn.close()
    log.info("Cleared existing bets and daily_pnl tables")

    # Load model
    model, feature_cols = load_bfsp_model(MODEL_DIR)

    # Load historical data
    log.info(f"Loading historical data (from {start_date})...")
    historical = load_historical(db_path, start_date=start_date)
    log.info(f"  {len(historical):,} rows loaded")

    # Get test dates
    all_dates = sorted(historical["race_date"].dt.date.unique())
    test_dates = all_dates[-n_days:]
    log.info(f"Backtesting {len(test_dates)} days: {test_dates[0]} to {test_dates[-1]}")

    total_bets = 0
    total_winners = 0
    cumulative_pnl = 0.0
    bank = 1000.0

    for td in test_dates:
        # History = everything before this date
        history_before = historical[historical["race_date"].dt.date < td].copy()
        runners_on_date = historical[historical["race_date"].dt.date == td].copy()

        if len(history_before) < 100 or len(runners_on_date) == 0:
            continue

        # Generate predictions
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

        # Store bets and settle immediately
        conn = sqlite3.connect(db_path)
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

            date_str = td.isoformat()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO bets
                       (race_date, race_time, track, horse_name,
                        predicted_bfsp, predicted_win_prob, betfair_back,
                        edge_pct, bet_type, stake, price, status,
                        actual_bsp, placing, gross_pnl, commission, net_pnl,
                        settled_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
                    (
                        date_str,
                        str(row.get("race_time", "")),
                        str(row.get("track", "")),
                        str(row.get("horse_name", "")),
                        float(row.get("predicted_bfsp", 0)),
                        float(row.get("predicted_win_prob_norm", 0)),
                        float(actual_bsp) if pd.notna(actual_bsp) else None,
                        float(row.get("edge_pct", 0)),
                        "BACK",
                        stake,
                        float(actual_bsp) if pd.notna(actual_bsp) else None,
                        status,
                        float(actual_bsp) if pd.notna(actual_bsp) else None,
                        int(placing) if pd.notna(placing) else None,
                        round(gross, 2),
                        round(commission, 2),
                        round(net, 2),
                    ),
                )
                day_bets += 1
                day_pnl += net
                day_staked += stake
            except Exception as e:
                log.warning(f"  Failed to store bet: {e}")

        # Update daily P&L
        cumulative_pnl += day_pnl
        bank_start = bank
        bank += day_pnl
        roi = (day_pnl / day_staked * 100) if day_staked > 0 else 0

        if day_bets > 0:
            conn.execute(
                """INSERT OR REPLACE INTO daily_pnl
                   (date, num_bets, winners, losers, voids, total_staked,
                    gross_pnl, commission, net_pnl, roi_pct,
                    cumulative_pnl, bank_start, bank_end)
                   VALUES (?,?,?,?,0,?,?,?,?,?,?,?,?)""",
                (
                    td.isoformat(), day_bets, day_winners, day_bets - day_winners,
                    round(day_staked, 2), round(day_pnl, 2), 0,
                    round(day_pnl, 2), round(roi, 2),
                    round(cumulative_pnl, 2), round(bank_start, 2), round(bank, 2),
                ),
            )

        conn.commit()
        conn.close()

        total_bets += day_bets
        total_winners += day_winners

        log.info(
            f"  {td}: {day_bets} bets, {day_winners}W, "
            f"P&L {day_pnl:+.2f}, Bank {bank:.2f}"
        )

    # Summary
    print(f"\n{'='*60}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Period:      {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    print(f"  Min edge:    {min_edge}%")
    print(f"  Stake:       £{stake:.2f}")
    print(f"  Total bets:  {total_bets}")
    print(f"  Winners:     {total_winners} ({total_winners/total_bets*100:.1f}%)" if total_bets else "  Winners:     0")
    print(f"  Net P&L:     £{cumulative_pnl:+.2f}")
    print(f"  ROI:         {cumulative_pnl/(total_bets*stake)*100:+.1f}%" if total_bets else "  ROI:         0%")
    print(f"  Bank:        £{bank:.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Backtest to populate bets/P&L tables")
    parser.add_argument("--days", type=int, default=7, help="Number of days to backtest")
    parser.add_argument("--min-edge", type=float, default=10.0, help="Minimum edge %% to bet")
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
