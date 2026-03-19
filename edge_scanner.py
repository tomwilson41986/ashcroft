#!/usr/bin/env python3
"""
Edge Scanner — Compares model predictions against live Betfair odds to find value bets.

Usage:
    python edge_scanner.py                           # Scan today
    python edge_scanner.py --date 2026-03-19         # Specific date
    python edge_scanner.py --from-db                 # Use DB runners (no HRB)
    python edge_scanner.py --min-edge 10             # Minimum edge threshold
    python edge_scanner.py --stake 10                # Fixed stake per bet
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from betfair_client import BetfairClient, BetfairAPIError
from db_schema import ensure_tables, DB_PATH
from predict_bfsp_today import (
    load_bfsp_model,
    load_historical,
    prepare_and_predict,
    get_runners_from_db,
    fetch_racecard_from_hrb,
)

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BETFAIR_COMMISSION = 0.05


def fetch_live_odds(target_date: date) -> pd.DataFrame:
    """Fetch live Betfair exchange odds and return as DataFrame."""
    client = BetfairClient()
    try:
        client.login()
    except BetfairAPIError as e:
        log.error(f"Betfair login failed: {e}")
        return pd.DataFrame()

    try:
        raw = client.get_live_odds_for_date(target_date)
    finally:
        client.logout()

    if not raw:
        log.warning("No live odds returned from Betfair")
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    return df


def store_odds_snapshot(odds_df: pd.DataFrame, target_date: date, db_path: str = DB_PATH) -> int:
    """Store a snapshot of live odds into betfair_odds table."""
    if odds_df.empty:
        return 0

    ensure_tables(db_path)
    conn = sqlite3.connect(db_path)
    now = datetime.utcnow().isoformat()
    count = 0

    for _, row in odds_df.iterrows():
        conn.execute(
            """INSERT INTO betfair_odds
               (market_id, selection_id, race_date, race_time, venue,
                runner_name, best_back, best_back_size, best_lay, best_lay_size,
                sp_near, sp_far, last_traded, total_matched,
                runner_status, market_status, snapshot_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row.get("market_id", ""),
                row.get("selection_id"),
                str(target_date),
                row.get("start_time", row.get("market_start_time", "")),
                row.get("venue", row.get("market_name", "")),
                row.get("runner_name", ""),
                row.get("best_back", row.get("best_back_price")),
                row.get("best_back_size"),
                row.get("best_lay", row.get("best_lay_price")),
                row.get("best_lay_size"),
                row.get("sp_near", row.get("sp_near_price")),
                row.get("sp_far", row.get("sp_far_price")),
                row.get("last_traded", row.get("last_traded_price")),
                row.get("total_matched"),
                row.get("runner_status", ""),
                row.get("market_status", ""),
                now,
            ),
        )
        count += 1

    conn.commit()
    conn.close()
    log.info(f"Stored {count} odds snapshots")
    return count


def normalise_name(name: str) -> str:
    """Normalise horse name for matching."""
    import re
    name = name.strip().lower()
    name = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", name)
    return name.strip()


def match_predictions_to_odds(
    predictions: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join predictions with live Betfair odds by venue + horse name."""
    if predictions.empty or odds_df.empty:
        return predictions

    # Normalise names for matching
    predictions = predictions.copy()
    predictions["_match_horse"] = predictions["horse_name"].apply(normalise_name)
    predictions["_match_track"] = predictions["track"].str.strip().str.lower()

    odds_df = odds_df.copy()
    odds_df["_match_horse"] = odds_df["runner_name"].apply(normalise_name)

    venue_col = "venue" if "venue" in odds_df.columns else "market_name"
    odds_df["_match_venue"] = odds_df[venue_col].str.strip().str.lower()

    back_col = "best_back" if "best_back" in odds_df.columns else "best_back_price"
    lay_col = "best_lay" if "best_lay" in odds_df.columns else "best_lay_price"
    last_col = "last_traded" if "last_traded" in odds_df.columns else "last_traded_price"
    sp_col = "sp_near" if "sp_near" in odds_df.columns else "sp_near_price"

    matched = 0
    predictions["betfair_back"] = np.nan
    predictions["betfair_lay"] = np.nan
    predictions["betfair_last"] = np.nan
    predictions["betfair_sp_near"] = np.nan
    predictions["market_id"] = ""
    predictions["selection_id"] = np.nan

    for idx, pred_row in predictions.iterrows():
        track = pred_row["_match_track"]
        horse = pred_row["_match_horse"]

        # Find matching odds row
        mask = odds_df["_match_horse"] == horse
        venue_matches = odds_df[mask]

        if len(venue_matches) == 0:
            continue

        # Try to match venue too
        venue_filtered = venue_matches[
            venue_matches["_match_venue"].str.contains(track, na=False)
            | pd.Series([track in v for v in venue_matches["_match_venue"]], index=venue_matches.index)
        ]

        if len(venue_filtered) > 0:
            odds_row = venue_filtered.iloc[0]
        elif len(venue_matches) == 1:
            odds_row = venue_matches.iloc[0]
        else:
            continue

        predictions.at[idx, "betfair_back"] = odds_row.get(back_col)
        predictions.at[idx, "betfair_lay"] = odds_row.get(lay_col)
        predictions.at[idx, "betfair_last"] = odds_row.get(last_col)
        predictions.at[idx, "betfair_sp_near"] = odds_row.get(sp_col)
        predictions.at[idx, "market_id"] = odds_row.get("market_id", "")
        predictions.at[idx, "selection_id"] = odds_row.get("selection_id")
        matched += 1

    # Drop temp columns
    predictions.drop(columns=["_match_horse", "_match_track"], inplace=True)

    log.info(f"Matched {matched}/{len(predictions)} predictions to live odds")
    return predictions


def calculate_edges(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate edge between predicted BFSP and Betfair back price."""
    predictions = predictions.copy()

    # Edge = how much higher the Betfair price is vs our predicted fair price
    # Positive edge means the market is offering better odds than our model says
    bf_price = predictions["betfair_back"]
    pred_price = predictions["predicted_bfsp"]

    predictions["edge_pct"] = ((bf_price / pred_price) - 1) * 100

    # Also compute overlay on lay side (for potential lay bets on overrated horses)
    bf_lay = predictions["betfair_lay"]
    predictions["lay_edge_pct"] = ((pred_price / bf_lay) - 1) * 100

    return predictions


def find_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = 10.0,
    min_price: float = 1.5,
    max_price: float = 30.0,
    min_prob: float = 0.03,
) -> pd.DataFrame:
    """Filter for value bets meeting edge and price criteria."""
    df = predictions.copy()

    # Must have betfair odds
    has_odds = df["betfair_back"].notna()

    # Back bets: our model says horse is shorter than market
    back_edge = df["edge_pct"] >= min_edge
    price_ok = (df["betfair_back"] >= min_price) & (df["betfair_back"] <= max_price)
    prob_ok = df["predicted_win_prob_norm"] >= min_prob

    value = df[has_odds & back_edge & price_ok & prob_ok].copy()
    value["bet_type"] = "BACK"
    value["bet_price"] = value["betfair_back"]

    value = value.sort_values("edge_pct", ascending=False)

    log.info(f"Found {len(value)} value bets (edge >= {min_edge}%)")
    return value


def store_bets(
    value_bets: pd.DataFrame,
    stake: float = 10.0,
    db_path: str = DB_PATH,
) -> int:
    """Insert value bets into the bets table."""
    if value_bets.empty:
        return 0

    ensure_tables(db_path)
    conn = sqlite3.connect(db_path)
    count = 0

    for _, row in value_bets.iterrows():
        try:
            conn.execute(
                """INSERT OR REPLACE INTO bets
                   (race_date, race_time, track, horse_name, market_id,
                    selection_id, predicted_bfsp, predicted_win_prob,
                    betfair_back, betfair_lay, edge_pct,
                    bet_type, stake, price, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row.get("race_date", pd.Timestamp("today")).strftime("%Y-%m-%d")
                    if hasattr(row.get("race_date"), "strftime")
                    else str(row.get("race_date", "")),
                    str(row.get("race_time", "")),
                    str(row.get("track", "")),
                    str(row.get("horse_name", "")),
                    str(row.get("market_id", "")),
                    int(row["selection_id"]) if pd.notna(row.get("selection_id")) else None,
                    float(row.get("predicted_bfsp", 0)),
                    float(row.get("predicted_win_prob_norm", 0)),
                    float(row["betfair_back"]) if pd.notna(row.get("betfair_back")) else None,
                    float(row["betfair_lay"]) if pd.notna(row.get("betfair_lay")) else None,
                    float(row.get("edge_pct", 0)),
                    row.get("bet_type", "BACK"),
                    stake,
                    float(row.get("bet_price", row.get("betfair_back", 0))),
                    "PENDING",
                ),
            )
            count += 1
        except Exception as e:
            log.warning(f"Failed to store bet for {row.get('horse_name')}: {e}")

    conn.commit()
    conn.close()
    log.info(f"Stored {count} bets")
    return count


def settle_bets(target_date: date, db_path: str = DB_PATH) -> dict:
    """Settle bets for a given date using actual results from the database.

    Pulls actual BSP and placing from race_results and updates bets table.
    """
    ensure_tables(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get unsettled bets for this date
    bets = conn.execute(
        "SELECT * FROM bets WHERE race_date = ? AND status = 'PENDING'",
        (str(target_date),),
    ).fetchall()

    if not bets:
        conn.close()
        return {"settled": 0, "winners": 0, "losers": 0}

    # Get actual results
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

    for bet in bets:
        key = (
            normalise_name(bet["horse_name"]),
            (bet["track"] or "").strip().lower(),
        )
        result = result_lookup.get(key)
        if not result:
            continue

        actual_bsp = result["actual_bsp"]
        placing = result["placing"]

        if placing is None:
            continue

        won = placing == 1
        stake = bet["stake"] or 0
        price = actual_bsp or bet["price"] or 0

        if bet["bet_type"] == "BACK":
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
        else:
            # LAY
            if won:
                gross = -(stake * (price - 1))
                commission = 0
                net = gross
                status = "LOST"
                losers += 1
            else:
                gross = stake
                commission = gross * BETFAIR_COMMISSION
                net = gross - commission
                status = "WON"
                winners += 1

        conn.execute(
            """UPDATE bets SET status = ?, actual_bsp = ?, placing = ?,
               gross_pnl = ?, commission = ?, net_pnl = ?,
               settled_at = datetime('now')
               WHERE id = ?""",
            (status, actual_bsp, placing, round(gross, 2),
             round(commission, 2), round(net, 2), bet["id"]),
        )
        settled += 1

    conn.commit()

    # Update daily_pnl
    _update_daily_pnl(conn, target_date)

    conn.close()
    log.info(f"Settled {settled} bets: {winners}W / {losers}L")
    return {"settled": settled, "winners": winners, "losers": losers}


def _update_daily_pnl(conn: sqlite3.Connection, target_date: date) -> None:
    """Calculate and upsert daily P&L from settled bets."""
    date_str = str(target_date)

    row = conn.execute(
        """SELECT
            COUNT(*) as num_bets,
            SUM(CASE WHEN status='WON' THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN status='LOST' THEN 1 ELSE 0 END) as losers,
            SUM(CASE WHEN status='VOID' THEN 1 ELSE 0 END) as voids,
            SUM(stake) as total_staked,
            SUM(gross_pnl) as gross_pnl,
            SUM(commission) as commission,
            SUM(net_pnl) as net_pnl
           FROM bets WHERE race_date = ? AND status IN ('WON','LOST','VOID')""",
        (date_str,),
    ).fetchone()

    if not row or row[0] == 0:
        return

    total_staked = row[5] or 0
    net_pnl = row[7] or 0
    roi = (net_pnl / total_staked * 100) if total_staked > 0 else 0

    # Get previous day's cumulative P&L
    prev = conn.execute(
        "SELECT cumulative_pnl, bank_end FROM daily_pnl WHERE date < ? ORDER BY date DESC LIMIT 1",
        (date_str,),
    ).fetchone()

    cum_pnl = (prev[0] if prev else 0) + net_pnl
    bank_start = prev[1] if prev else 1000
    bank_end = bank_start + net_pnl

    conn.execute(
        """INSERT OR REPLACE INTO daily_pnl
           (date, num_bets, winners, losers, voids, total_staked,
            gross_pnl, commission, net_pnl, roi_pct,
            cumulative_pnl, bank_start, bank_end)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (date_str, row[0], row[1], row[2], row[3],
         round(total_staked, 2), round(row[6] or 0, 2),
         round(row[6] or 0, 2), round(net_pnl, 2), round(roi, 2),
         round(cum_pnl, 2), round(bank_start, 2), round(bank_end, 2)),
    )
    conn.commit()


def scan_edges(
    target_date: date,
    from_db: bool = False,
    min_edge: float = 10.0,
    stake: float = 10.0,
    db_path: str = DB_PATH,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Full edge scanning pipeline: predict -> fetch odds -> find value -> store."""
    log.info(f"=== Edge Scanner: {target_date} ===")

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

    # 5. Fetch live Betfair odds
    log.info("Fetching live Betfair odds...")
    odds_df = fetch_live_odds(target_date)

    if not odds_df.empty:
        # Store odds snapshot
        store_odds_snapshot(odds_df, target_date, db_path)

        # Match predictions to odds
        predictions = match_predictions_to_odds(predictions, odds_df)

        # Calculate edges
        predictions = calculate_edges(predictions)

        # Find value bets
        value_bets = find_value_bets(predictions, min_edge=min_edge)

        # Store bets
        if not value_bets.empty:
            store_bets(value_bets, stake=stake, db_path=db_path)
    else:
        log.warning("No Betfair odds available — predictions only")
        value_bets = pd.DataFrame()

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Scan for value bets")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--from-db", action="store_true")
    parser.add_argument("--min-edge", type=float, default=10.0)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--settle", action="store_true", help="Settle bets for the date")
    parser.add_argument("--start-date", type=str, default="2020-01-01")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.settle:
        result = settle_bets(target)
        print(f"Settlement: {result}")
        return

    predictions = scan_edges(
        target,
        from_db=args.from_db,
        min_edge=args.min_edge,
        stake=args.stake,
        start_date=args.start_date,
    )

    if predictions.empty:
        print("No predictions generated")
        return

    # Display results
    has_odds = "betfair_back" in predictions.columns and predictions["betfair_back"].notna().any()

    print(f"\n{'='*90}")
    print(f"  EDGE SCANNER — {target}")
    print(f"{'='*90}")

    for race_id, race_df in predictions.groupby("raceid"):
        race_df = race_df.sort_values("predicted_bfsp")
        track = race_df["track"].iloc[0]
        rtime = race_df["race_time"].iloc[0]

        print(f"\n  {track} — {rtime} ({len(race_df)} runners)")
        print(f"  {'-'*86}")

        if has_odds:
            print(f"  {'Horse':<24} {'Pred':>7} {'Back':>7} {'Lay':>7} {'Edge%':>7} {'P(Win)':>7} {'Signal':>8}")
            print(f"  {'-'*24} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
        else:
            print(f"  {'Horse':<24} {'Pred BFSP':>10} {'P(Win)':>7}")
            print(f"  {'-'*24} {'-'*10} {'-'*7}")

        for _, row in race_df.iterrows():
            name = str(row.get("horse_name", "?"))[:23]
            pred = row.get("predicted_bfsp", 0)
            pwin = row.get("predicted_win_prob_norm", 0)

            if has_odds:
                back = row.get("betfair_back")
                lay = row.get("betfair_lay")
                edge = row.get("edge_pct")

                back_s = f"{back:.2f}" if pd.notna(back) else "-"
                lay_s = f"{lay:.2f}" if pd.notna(lay) else "-"
                edge_s = f"{edge:+.1f}%" if pd.notna(edge) else "-"

                signal = ""
                if pd.notna(edge) and edge >= 10:
                    signal = "** BET **"
                elif pd.notna(edge) and edge >= 5:
                    signal = "watch"

                print(f"  {name:<24} {pred:>7.2f} {back_s:>7} {lay_s:>7} {edge_s:>7} {pwin:>6.1%} {signal:>8}")
            else:
                print(f"  {name:<24} {pred:>10.2f} {pwin:>6.1%}")

    # Summary of value bets
    if has_odds:
        value = predictions[predictions.get("edge_pct", pd.Series(dtype=float)).fillna(0) >= 10]
        if len(value) > 0:
            print(f"\n  {'='*86}")
            print(f"  VALUE BETS ({len(value)} selections, edge >= 10%)")
            print(f"  {'='*86}")
            for _, row in value.sort_values("edge_pct", ascending=False).iterrows():
                print(
                    f"  {row['track']} {row['race_time']}: {row['horse_name']} "
                    f"(Pred {row['predicted_bfsp']:.2f} vs Back {row['betfair_back']:.2f}, "
                    f"Edge {row['edge_pct']:+.1f}%)"
                )

    print()


if __name__ == "__main__":
    main()
