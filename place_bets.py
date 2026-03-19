#!/usr/bin/env python3
"""
Bet Placement CLI — Execute bets on Betfair Exchange.

Reads pending paper trades or edge scanner output and places bets via the
Betfair API. Supports dry-run mode (default) for safety.

Usage:
    # Dry run — log what would be placed (default)
    python place_bets.py

    # Place real bets from today's paper trades
    python place_bets.py --live

    # Place from edge scanner instead of paper trades
    python place_bets.py --live --source edge-scanner

    # Place a single bet manually
    python place_bets.py --live --market-id 1.234567890 --selection-id 12345 \\
        --side BACK --stake 10 --price 5.0

    # Specific date
    python place_bets.py --date 2026-03-19

    # Only place bets meeting minimum edge
    python place_bets.py --live --min-edge 15

    # Only top-ranked selections
    python place_bets.py --live --strategy top1_edge

    # Check status of placed bets
    python place_bets.py --status
"""

import argparse
import logging
import os
import sqlite3
from datetime import date, datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local")

from betfair_client import BetfairClient, BetfairAPIError
from db_schema import ensure_tables, DB_PATH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BETFAIR_COMMISSION = 0.05

# Strategy filtering (matches frontend STRATEGIES)
STRATEGY_FILTERS = {
    "all":        lambda t: True,
    "all_edge":   lambda t: (t.get("edge_pct") or 0) > 0,
    "top1":       lambda t: t.get("rank") == 1,
    "top1_edge":  lambda t: t.get("rank") == 1 and (t.get("edge_pct") or 0) > 0,
    "top3":       lambda t: 1 <= (t.get("rank") or 0) <= 3,
    "top3_edge":  lambda t: 1 <= (t.get("rank") or 0) <= 3 and (t.get("edge_pct") or 0) > 0,
}


def ensure_executions_table(db_path: str = DB_PATH) -> None:
    """Create executions table for tracking placed bets."""
    ensure_tables(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT NOT NULL,
            race_time TEXT,
            track TEXT,
            horse_name TEXT,
            market_id TEXT NOT NULL,
            selection_id INTEGER NOT NULL,
            side TEXT DEFAULT 'BACK',
            stake REAL,
            price_requested REAL,
            price_matched REAL,
            size_matched REAL,
            bet_id TEXT,
            status TEXT DEFAULT 'PENDING',
            dry_run INTEGER DEFAULT 1,
            edge_pct REAL,
            rank INTEGER,
            predicted_bfsp REAL,
            source TEXT DEFAULT 'paper_trades',
            actual_bsp REAL,
            placing INTEGER,
            gross_pnl REAL,
            commission REAL,
            net_pnl REAL,
            created_at TEXT DEFAULT (datetime('now')),
            settled_at TEXT,
            UNIQUE(race_date, market_id, selection_id, side)
        );

        CREATE INDEX IF NOT EXISTS idx_exec_date ON executions(race_date);
        CREATE INDEX IF NOT EXISTS idx_exec_status ON executions(status);
        CREATE INDEX IF NOT EXISTS idx_exec_bet_id ON executions(bet_id);
    """)
    conn.commit()
    conn.close()


def get_pending_paper_trades(
    target_date: date,
    min_edge: float = 0.0,
    strategy: str = "all",
    db_path: str = DB_PATH,
) -> list[dict]:
    """Get pending paper trades for a date, filtered by strategy."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM paper_trades WHERE race_date = ? AND status = 'PENDING' ORDER BY edge_pct DESC",
        (str(target_date),),
    ).fetchall()
    conn.close()

    trades = [dict(r) for r in rows]

    # Apply strategy filter
    strat_fn = STRATEGY_FILTERS.get(strategy, STRATEGY_FILTERS["all"])
    trades = [t for t in trades if strat_fn(t)]

    # Apply minimum edge filter
    if min_edge > 0:
        trades = [t for t in trades if (t.get("edge_pct") or 0) >= min_edge]

    return trades


def place_bet_on_betfair(
    client: BetfairClient,
    market_id: str,
    selection_id: int,
    side: str,
    stake: float,
    price: float,
    dry_run: bool = True,
) -> dict:
    """Place a single bet on Betfair. Returns execution result."""

    if dry_run:
        log.info(
            f"[DRY RUN] {side} £{stake:.2f} @ {price:.2f} "
            f"(market={market_id}, selection={selection_id})"
        )
        return {
            "status": "DRY_RUN",
            "bet_id": "",
            "price_matched": price,
            "size_matched": stake,
        }

    params = {
        "marketId": market_id,
        "instructions": [{
            "selectionId": selection_id,
            "side": side,
            "orderType": "LIMIT",
            "limitOrder": {
                "size": round(stake, 2),
                "price": price,
                "persistenceType": "LAPSE",
            },
        }],
    }

    try:
        result = client._api_call("placeOrders", params)

        if result.get("status") == "SUCCESS":
            report = result["instructionReports"][0]
            return {
                "status": report.get("status", "MATCHED"),
                "bet_id": str(report.get("betId", "")),
                "price_matched": report.get("averagePriceMatched", price),
                "size_matched": report.get("sizeMatched", 0),
            }
        else:
            error_code = result.get("errorCode", "UNKNOWN")
            log.error(f"Bet placement failed: {error_code}")
            return {"status": "FAILED", "bet_id": "", "price_matched": 0, "size_matched": 0}

    except BetfairAPIError as e:
        log.error(f"Betfair API error: {e}")
        return {"status": "FAILED", "bet_id": "", "price_matched": 0, "size_matched": 0}


def place_bets(
    target_date: date,
    dry_run: bool = True,
    source: str = "paper_trades",
    min_edge: float = 0.0,
    strategy: str = "all",
    db_path: str = DB_PATH,
) -> list[dict]:
    """Place bets from paper trades or edge scanner output."""
    ensure_executions_table(db_path)

    mode = "DRY RUN" if dry_run else "LIVE"
    log.info(f"=== Bet Placement ({mode}): {target_date} ===")

    # Get candidates
    if source == "paper_trades":
        candidates = get_pending_paper_trades(target_date, min_edge, strategy, db_path)
    else:
        # Read from bets table (edge scanner output)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM bets WHERE race_date = ? AND status = 'PENDING' ORDER BY edge_pct DESC",
            (str(target_date),),
        ).fetchall()
        conn.close()
        candidates = [dict(r) for r in rows]
        strat_fn = STRATEGY_FILTERS.get(strategy, STRATEGY_FILTERS["all"])
        candidates = [c for c in candidates if strat_fn(c)]
        if min_edge > 0:
            candidates = [c for c in candidates if (c.get("edge_pct") or 0) >= min_edge]

    if not candidates:
        log.info("No candidates to place bets on")
        return []

    log.info(f"Found {len(candidates)} candidates (strategy={strategy}, min_edge={min_edge}%)")

    # Login to Betfair (only for live mode)
    client = None
    if not dry_run:
        client = BetfairClient()
        try:
            client.login()
            log.info("Betfair login successful")
        except BetfairAPIError as e:
            log.error(f"Betfair login failed: {e}")
            return []

    executions = []
    conn = sqlite3.connect(db_path)

    try:
        for trade in candidates:
            market_id = trade.get("market_id", "")
            selection_id = trade.get("selection_id")
            stake = trade.get("stake", 10)
            price = trade.get("entry_price") or trade.get("betfair_back") or trade.get("price")
            side = trade.get("bet_type", "BACK")

            if not market_id or not selection_id or not price:
                log.warning(f"Skipping {trade.get('horse_name')}: missing market data")
                continue

            result = place_bet_on_betfair(
                client, market_id, int(selection_id), side, stake, price, dry_run
            )

            # Record execution
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO executions
                       (race_date, race_time, track, horse_name, market_id,
                        selection_id, side, stake, price_requested, price_matched,
                        size_matched, bet_id, status, dry_run, edge_pct, rank,
                        predicted_bfsp, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(target_date),
                        trade.get("race_time", ""),
                        trade.get("track", ""),
                        trade.get("horse_name", ""),
                        market_id,
                        int(selection_id),
                        side,
                        stake,
                        price,
                        result["price_matched"],
                        result["size_matched"],
                        result["bet_id"],
                        result["status"],
                        1 if dry_run else 0,
                        trade.get("edge_pct", 0),
                        trade.get("rank"),
                        trade.get("predicted_bfsp"),
                        source,
                    ),
                )
            except Exception as e:
                log.warning(f"Failed to record execution: {e}")

            executions.append({
                "horse_name": trade.get("horse_name"),
                "track": trade.get("track"),
                "race_time": trade.get("race_time"),
                "side": side,
                "stake": stake,
                "price": price,
                "edge_pct": trade.get("edge_pct"),
                "rank": trade.get("rank"),
                **result,
            })

        conn.commit()
    finally:
        conn.close()
        if client and not dry_run:
            try:
                client.logout()
            except Exception:
                pass

    # Display summary
    print(f"\n{'='*80}")
    print(f"  BET PLACEMENT — {mode} — {target_date}")
    print(f"{'='*80}")
    for ex in executions:
        status_icon = {
            "DRY_RUN": "[PAPER]",
            "MATCHED": "[LIVE]",
            "PENDING": "[QUEUE]",
            "FAILED": "[FAIL]",
        }.get(ex["status"], f"[{ex['status']}]")

        print(
            f"  {status_icon:>8} {ex['side']} £{ex['stake']:.2f} @ {ex['price']:.2f} "
            f"{ex.get('horse_name','?'):<22} {ex.get('track','?'):<12} "
            f"Edge {ex.get('edge_pct',0):>+5.1f}%"
        )
        if ex.get("bet_id"):
            print(f"           Bet ID: {ex['bet_id']}")

    placed = [e for e in executions if e["status"] not in ("FAILED",)]
    failed = [e for e in executions if e["status"] == "FAILED"]
    total_stake = sum(e["stake"] for e in placed)

    print(f"\n  Placed: {len(placed)}  Failed: {len(failed)}  Total stake: £{total_stake:.2f}")
    print(f"{'='*80}\n")

    return executions


def show_status(target_date: date, db_path: str = DB_PATH) -> None:
    """Show status of executions for a date."""
    ensure_executions_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM executions WHERE race_date = ? ORDER BY race_time, track",
        (str(target_date),),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No executions found for {target_date}")
        return

    print(f"\n{'='*90}")
    print(f"  EXECUTIONS — {target_date}")
    print(f"{'='*90}")
    print(f"  {'Status':<10} {'Mode':<7} {'Side':<5} {'Stake':>7} {'Price':>7} "
          f"{'Matched':>7} {'Horse':<22} {'Track':<12} {'Edge':>6}")
    print(f"  {'-'*90}")

    for r in rows:
        mode = "Paper" if r["dry_run"] else "Live"
        print(
            f"  {r['status']:<10} {mode:<7} {r['side']:<5} "
            f"£{r['stake']:>5.2f} {r['price_requested']:>7.2f} "
            f"{r['price_matched']:>7.2f} {r['horse_name']:<22} "
            f"{r['track']:<12} {r['edge_pct']:>+5.1f}%"
        )
        if r["bet_id"]:
            print(f"  {'':>10} Bet ID: {r['bet_id']}")

    total = len(rows)
    live = sum(1 for r in rows if not r["dry_run"])
    paper = total - live
    print(f"\n  Total: {total} ({live} live, {paper} paper)")
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Place bets on Betfair Exchange")
    parser.add_argument("--date", type=str, default=None, help="Target date (default: today)")
    parser.add_argument("--live", action="store_true", help="Place REAL bets (default: dry run)")
    parser.add_argument("--source", choices=["paper_trades", "edge-scanner"], default="paper_trades")
    parser.add_argument("--min-edge", type=float, default=0.0, help="Minimum edge %% filter")
    parser.add_argument("--strategy", choices=list(STRATEGY_FILTERS.keys()), default="all")
    parser.add_argument("--stake", type=float, default=None, help="Override stake amount")
    parser.add_argument("--status", action="store_true", help="Show execution status")

    # Manual single bet
    parser.add_argument("--market-id", type=str, help="Place single bet: market ID")
    parser.add_argument("--selection-id", type=int, help="Place single bet: selection ID")
    parser.add_argument("--side", choices=["BACK", "LAY"], default="BACK")
    parser.add_argument("--price", type=float, help="Place single bet: price")

    args = parser.parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today()
    dry_run = not args.live

    if args.status:
        show_status(target)
        return

    # Single bet mode
    if args.market_id and args.selection_id and args.price:
        stake = args.stake or 10.0
        client = None
        if not dry_run:
            client = BetfairClient()
            client.login()

        result = place_bet_on_betfair(
            client, args.market_id, args.selection_id,
            args.side, stake, args.price, dry_run
        )

        if client:
            client.logout()

        mode = "DRY RUN" if dry_run else "LIVE"
        print(f"\n[{mode}] {result['status']}: {args.side} £{stake:.2f} @ {args.price:.2f}")
        if result.get("bet_id"):
            print(f"  Bet ID: {result['bet_id']}")
        return

    # Batch mode
    if args.live:
        confirm = input(
            f"\n  WARNING: This will place REAL bets on Betfair for {target}.\n"
            f"  Strategy: {args.strategy}, Min edge: {args.min_edge}%\n"
            f"  Type 'YES' to confirm: "
        )
        if confirm != "YES":
            print("Aborted.")
            return

    place_bets(
        target,
        dry_run=dry_run,
        source=args.source,
        min_edge=args.min_edge,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
