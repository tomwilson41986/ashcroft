#!/usr/bin/env python3
"""
Paper Trader — Records simulated bets against live Betfair exchange prices.

Runs the BFSP model, fetches live Betfair odds, identifies value bets,
and records them as paper trades in S3. Tracks what we *would* have bet
at live exchange prices, then settles against actual BSP after the race.

Usage:
    python paper_trader.py                           # Scan today, record paper trades
    python paper_trader.py --date 2026-03-19         # Specific date
    python paper_trader.py --settle                  # Settle paper trades for the date
    python paper_trader.py --settle --from 2026-03-01 --to 2026-03-19
    python paper_trader.py --from-db                 # Use DB runners (no HRB)
    python paper_trader.py --min-edge 5              # Lower edge threshold for paper trades
    python paper_trader.py --sync-dashboard          # Push paper trades to S3 dashboard
"""

import argparse
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local")

from betfair_client import BetfairClient, BetfairAPIError
from db_schema import ensure_tables, DB_PATH
from edge_scanner import (
    BETFAIR_COMMISSION,
    fetch_live_odds,
    match_predictions_to_odds,
    calculate_edges,
    normalise_name,
    store_odds_snapshot,
)
from predict_bfsp_today import (
    load_bfsp_model,
    load_historical,
    prepare_and_predict,
    get_runners_from_db,
    fetch_racecard_from_hrb,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def ensure_paper_trades_table(db_path: str = DB_PATH) -> None:
    """Create paper_trades table if it doesn't exist."""
    ensure_tables(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT NOT NULL,
            race_time TEXT,
            track TEXT NOT NULL,
            horse_name TEXT NOT NULL,
            market_id TEXT,
            selection_id INTEGER,
            rank INTEGER,
            predicted_bfsp REAL,
            predicted_win_prob REAL,
            betfair_back REAL,
            betfair_lay REAL,
            edge_pct REAL,
            bet_type TEXT DEFAULT 'BACK',
            stake REAL DEFAULT 10,
            entry_price REAL,
            actual_bsp REAL,
            placing INTEGER,
            status TEXT DEFAULT 'PENDING',
            gross_pnl REAL,
            commission REAL,
            net_pnl REAL,
            created_at TEXT DEFAULT (datetime('now')),
            settled_at TEXT,
            UNIQUE(race_date, race_time, track, horse_name)
        );

        CREATE INDEX IF NOT EXISTS idx_pt_date ON paper_trades(race_date);
        CREATE INDEX IF NOT EXISTS idx_pt_status ON paper_trades(status);
    """)
    conn.commit()
    conn.close()


def record_paper_trades(
    target_date: date,
    from_db: bool = False,
    min_edge: float = 5.0,
    stake: float = 10.0,
    min_price: float = 1.5,
    max_price: float = 30.0,
    db_path: str = DB_PATH,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Run full pipeline and record paper trades for value bets."""
    log.info(f"=== Paper Trader: {target_date} ===")

    ensure_paper_trades_table(db_path)

    # 1. Load model
    model, feature_cols = load_bfsp_model(MODEL_DIR)

    # 2. Load history
    log.info(f"Loading historical data (from {start_date})...")
    historical = load_historical(db_path, start_date=start_date)
    log.info(f"  {len(historical):,} rows loaded")

    # 3. Get today's runners
    if from_db:
        runners = get_runners_from_db(db_path, str(target_date))
    else:
        runners = fetch_racecard_from_hrb(target_date)
        if runners.empty:
            log.warning("HRB fetch failed, falling back to DB")
            runners = get_runners_from_db(db_path, str(target_date))

    if runners.empty:
        log.error("No runners found")
        return pd.DataFrame()

    log.info(f"Found {len(runners)} runners")

    # 4. Generate predictions
    history_before = historical[historical["race_date"].dt.date < target_date].copy()
    predictions = prepare_and_predict(
        history_before, runners, model, feature_cols, target_date
    )

    if predictions.empty:
        log.error("No predictions generated")
        return pd.DataFrame()

    # 5. Compute rank per race
    if "race_time" in predictions.columns and "track" in predictions.columns:
        predictions["race_key"] = (
            predictions["race_date"].astype(str) + "|"
            + predictions["race_time"].astype(str) + "|"
            + predictions["track"].astype(str)
        )
    else:
        predictions["race_key"] = predictions.index.astype(str)
    predictions["rank"] = (
        predictions.groupby("race_key")["predicted_bfsp"]
        .rank(method="first")
        .astype(int)
    )

    # 6. Fetch live Betfair odds
    log.info("Fetching live Betfair odds...")
    odds_df = fetch_live_odds(target_date)

    if odds_df.empty:
        log.warning("No Betfair odds available — cannot paper trade")
        return predictions

    # Store odds snapshot
    store_odds_snapshot(odds_df, target_date, db_path)

    # Match predictions to odds
    predictions = match_predictions_to_odds(predictions, odds_df)
    predictions = calculate_edges(predictions)

    # 7. Filter for paper trade candidates (all runners with odds + edge)
    has_odds = predictions["betfair_back"].notna()
    has_pred = predictions["predicted_bfsp"].notna() & (predictions["predicted_bfsp"] > 0)
    price_ok = (predictions["betfair_back"] >= min_price) & (predictions["betfair_back"] <= max_price)
    edge_ok = predictions["edge_pct"] >= min_edge

    paper_bets = predictions[has_odds & has_pred & price_ok & edge_ok].copy()
    paper_bets = paper_bets.sort_values("edge_pct", ascending=False)

    log.info(f"Found {len(paper_bets)} paper trade candidates (edge >= {min_edge}%)")

    # 8. Store paper trades in SQLite
    conn = sqlite3.connect(db_path)
    count = 0

    for _, row in paper_bets.iterrows():
        race_date_str = (
            row["race_date"].strftime("%Y-%m-%d")
            if hasattr(row.get("race_date"), "strftime")
            else str(row.get("race_date", ""))
        )
        try:
            conn.execute(
                """INSERT OR REPLACE INTO paper_trades
                   (race_date, race_time, track, horse_name, market_id,
                    selection_id, rank, predicted_bfsp, predicted_win_prob,
                    betfair_back, betfair_lay, edge_pct,
                    bet_type, stake, entry_price, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    race_date_str,
                    str(row.get("race_time", "")),
                    str(row.get("track", "")),
                    str(row.get("horse_name", "")),
                    str(row.get("market_id", "")),
                    int(row["selection_id"]) if pd.notna(row.get("selection_id")) else None,
                    int(row.get("rank", 0)),
                    float(row.get("predicted_bfsp", 0)),
                    float(row.get("predicted_win_prob_norm", 0)),
                    float(row["betfair_back"]) if pd.notna(row.get("betfair_back")) else None,
                    float(row["betfair_lay"]) if pd.notna(row.get("betfair_lay")) else None,
                    float(row.get("edge_pct", 0)),
                    "BACK",
                    stake,
                    float(row["betfair_back"]) if pd.notna(row.get("betfair_back")) else None,
                    "PENDING",
                ),
            )
            count += 1
        except Exception as e:
            log.warning(f"Failed to store paper trade for {row.get('horse_name')}: {e}")

    conn.commit()
    conn.close()
    log.info(f"Recorded {count} paper trades")

    # Display summary
    print(f"\n{'='*80}")
    print(f"  PAPER TRADES — {target_date}")
    print(f"{'='*80}")
    for _, row in paper_bets.iterrows():
        rank = row.get("rank", "?")
        horse = str(row.get("horse_name", "?"))[:22]
        track = str(row.get("track", "?"))[:12]
        rtime = str(row.get("race_time", "?"))
        pred = row.get("predicted_bfsp", 0)
        back = row.get("betfair_back", 0)
        edge = row.get("edge_pct", 0)
        print(
            f"  [{rank:>2}] {track:<12} {rtime} {horse:<22} "
            f"Pred {pred:>6.2f}  Back {back:>6.2f}  Edge {edge:>+5.1f}%"
        )
    print(f"\n  Total: {count} paper trades @ £{stake:.2f} each")
    print(f"{'='*80}\n")

    return paper_bets


def settle_paper_trades(target_date: date, db_path: str = DB_PATH) -> dict:
    """Settle paper trades using actual BSP and results from the database."""
    ensure_paper_trades_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    trades = conn.execute(
        "SELECT * FROM paper_trades WHERE race_date = ? AND status = 'PENDING'",
        (str(target_date),),
    ).fetchall()

    if not trades:
        conn.close()
        log.info(f"No pending paper trades for {target_date}")
        return {"settled": 0, "winners": 0, "losers": 0}

    # Get actual results from race_results
    results = conn.execute(
        "SELECT horse_name, track, race_time, bfsp, placing_numerical "
        "FROM race_results WHERE race_date = ?",
        (str(target_date),),
    ).fetchall()

    result_lookup = {}
    for r in results:
        key = (
            normalise_name(r["horse_name"]),
            (r["track"] or "").strip().lower(),
        )
        result_lookup[key] = {
            "actual_bsp": r["bfsp"],
            "placing": r["placing_numerical"],
        }

    settled = winners = losers = 0

    for trade in trades:
        key = (
            normalise_name(trade["horse_name"]),
            (trade["track"] or "").strip().lower(),
        )
        result = result_lookup.get(key)
        if not result:
            continue

        actual_bsp = result["actual_bsp"]
        placing = result["placing"]

        if placing is None:
            continue

        won = placing == 1
        stake = trade["stake"] or 10
        price = actual_bsp or trade["entry_price"] or 0

        if won:
            gross = stake * (price - 1)
            commission = gross * BETFAIR_COMMISSION
            net = gross - commission
            status = "WON"
            winners += 1
        else:
            gross = -stake
            commission = 0
            net = -stake
            status = "LOST"
            losers += 1

        conn.execute(
            """UPDATE paper_trades SET status = ?, actual_bsp = ?, placing = ?,
               gross_pnl = ?, commission = ?, net_pnl = ?,
               settled_at = datetime('now')
               WHERE id = ?""",
            (status, actual_bsp, placing, round(gross, 2),
             round(commission, 2), round(net, 2), trade["id"]),
        )
        settled += 1

    conn.commit()
    conn.close()

    log.info(f"Settled {settled} paper trades: {winners}W / {losers}L")
    return {"settled": settled, "winners": winners, "losers": losers}


def sync_paper_trades_to_dashboard(db_path: str = DB_PATH) -> int:
    """Push all paper trades from SQLite to S3 dashboard JSON."""
    from web.s3_store import save_paper_trades

    ensure_paper_trades_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT * FROM paper_trades ORDER BY race_date, race_time"
    ).fetchall()
    conn.close()

    trades = []
    for r in rows:
        trades.append({
            "race_date": r["race_date"],
            "race_time": r["race_time"],
            "track": r["track"],
            "horse_name": r["horse_name"],
            "market_id": r["market_id"],
            "selection_id": r["selection_id"],
            "rank": r["rank"],
            "predicted_bfsp": r["predicted_bfsp"],
            "predicted_win_prob": r["predicted_win_prob"],
            "betfair_back": r["betfair_back"],
            "betfair_lay": r["betfair_lay"],
            "edge_pct": r["edge_pct"],
            "bet_type": r["bet_type"],
            "stake": r["stake"],
            "entry_price": r["entry_price"],
            "actual_bsp": r["actual_bsp"],
            "placing": r["placing"],
            "status": r["status"],
            "gross_pnl": r["gross_pnl"],
            "commission": r["commission"],
            "net_pnl": r["net_pnl"],
        })

    save_paper_trades(trades)
    log.info(f"Synced {len(trades)} paper trades to S3 dashboard")
    return len(trades)


def main():
    parser = argparse.ArgumentParser(description="Paper trade against live Betfair odds")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--from-db", action="store_true", help="Use DB runners (no HRB)")
    parser.add_argument("--min-edge", type=float, default=5.0, help="Min edge %% for paper trades")
    parser.add_argument("--stake", type=float, default=10.0, help="Paper trade stake")
    parser.add_argument("--settle", action="store_true", help="Settle paper trades")
    parser.add_argument("--from", dest="from_date", type=str, help="Start date for settlement range")
    parser.add_argument("--to", dest="to_date", type=str, help="End date for settlement range")
    parser.add_argument("--sync-dashboard", action="store_true", help="Sync paper trades to S3")
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.sync_dashboard:
        sync_paper_trades_to_dashboard()
        return

    if args.settle:
        if args.from_date and args.to_date:
            d = date.fromisoformat(args.from_date)
            end = date.fromisoformat(args.to_date)
            total = {"settled": 0, "winners": 0, "losers": 0}
            while d <= end:
                result = settle_paper_trades(d)
                for k in total:
                    total[k] += result[k]
                d += timedelta(days=1)
            print(f"\nSettlement totals: {total}")
        else:
            result = settle_paper_trades(target)
            print(f"Settlement: {result}")

        # Auto-sync to dashboard after settlement
        sync_paper_trades_to_dashboard()
        return

    record_paper_trades(
        target,
        from_db=args.from_db,
        min_edge=args.min_edge,
        stake=args.stake,
        start_date=args.start_date,
    )

    # Auto-sync to dashboard
    sync_paper_trades_to_dashboard()


if __name__ == "__main__":
    main()
