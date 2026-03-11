"""
Betfair BSP & Live Market Sync.

Pulls actual BSP data from Betfair for completed races and stores it in the
database. Also fetches live exchange odds for upcoming/non-completed races.

Usage:
    # Pull actual BSPs for yesterday's races into the database
    python betfair_sync.py --bsp --date 2026-03-10

    # Pull BSPs for a date range
    python betfair_sync.py --bsp --from 2026-03-01 --to 2026-03-10

    # Show live markets and odds for today
    python betfair_sync.py --live

    # Show live markets for a specific date
    python betfair_sync.py --live --date 2026-03-12

    # Show only non-completed/upcoming markets
    python betfair_sync.py --upcoming

    # Both: pull BSPs for completed + show live for upcoming
    python betfair_sync.py --bsp --live

    # Export live odds to CSV
    python betfair_sync.py --live --csv live_odds.csv

Environment variables (in .env):
    BETFAIR_USERNAME   - Betfair account username
    BETFAIR_PASSWORD   - Betfair account password
    BETFAIR_APP_KEY    - Betfair API application key
    BETFAIR_CERT_FILE  - (Optional) Client certificate .crt
    BETFAIR_KEY_FILE   - (Optional) Client certificate .key
"""

import argparse
import logging
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

from betfair_client import BetfairClient, BetfairAPIError, match_runner_name

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(SCRIPT_DIR, "betfair_sync.log")
        ),
    ],
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# BSP Sync: Pull actual BSPs and update the database
# ------------------------------------------------------------------

def sync_bsps_for_date(
    client: BetfairClient,
    target_date: date,
    db_path: str = DB_PATH,
) -> dict:
    """Pull actual BSPs from Betfair and update the database.

    Matches Betfair runners to database rows by venue + horse name,
    then updates the BFSP column with the actual Betfair Starting Price.

    Args:
        client: Authenticated BetfairClient instance.
        target_date: Date to sync BSPs for.
        db_path: Path to SQLite database.

    Returns:
        Summary dict with counts of matched, updated, and unmatched runners.
    """
    log.info(f"Syncing BSPs for {target_date}...")

    # Get BSPs from Betfair
    bsp_data = client.get_bsps_for_date(target_date)
    if not bsp_data:
        log.warning(f"No BSP data returned for {target_date}")
        return {"date": str(target_date), "matched": 0, "updated": 0, "unmatched": 0}

    # Filter to only runners with actual BSP values
    with_bsp = [r for r in bsp_data if r.get("actual_bsp") is not None]
    log.info(f"  {len(with_bsp)} runners with actual BSP out of {len(bsp_data)} total")

    if not with_bsp:
        return {"date": str(target_date), "matched": 0, "updated": 0, "unmatched": 0}

    # Load database rows for this date
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    date_str = target_date.isoformat()
    cursor.execute(
        "SELECT rowid, horse_name, track, race_time, BFSP "
        "FROM race_results WHERE race_date = ?",
        (date_str,),
    )
    db_rows = cursor.fetchall()

    if not db_rows:
        log.warning(f"  No database rows found for {target_date}")
        conn.close()
        return {"date": str(target_date), "matched": 0, "updated": 0, "unmatched": len(with_bsp)}

    log.info(f"  {len(db_rows)} database rows for {target_date}")

    # Build lookup: (normalised_track, normalised_horse) -> rowid, existing_bfsp
    db_lookup = {}
    for row in db_rows:
        track = (row["track"] or "").strip().lower()
        horse = (row["horse_name"] or "").strip().lower()
        key = (track, horse)
        db_lookup[key] = {
            "rowid": row["rowid"],
            "existing_bfsp": row["BFSP"],
            "race_time": row["race_time"],
        }

    matched = 0
    updated = 0
    unmatched_runners = []

    for runner in with_bsp:
        venue = (runner["venue"] or "").strip().lower()
        bf_name = runner["runner_name"]
        actual_bsp = runner["actual_bsp"]

        # Try direct match
        bf_name_clean = bf_name.strip().lower()
        # Strip country suffix for matching
        import re
        bf_name_norm = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", bf_name_clean).strip()

        found = False
        for (track, horse), db_info in db_lookup.items():
            if venue in track or track in venue:
                horse_norm = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", horse).strip()
                if bf_name_norm == horse_norm:
                    matched += 1
                    found = True

                    # Update BFSP if it's different or missing
                    existing = db_info["existing_bfsp"]
                    if existing is None or abs(float(existing or 0) - actual_bsp) > 0.01:
                        cursor.execute(
                            "UPDATE race_results SET BFSP = ? WHERE rowid = ?",
                            (actual_bsp, db_info["rowid"]),
                        )
                        updated += 1
                        log.debug(
                            f"  Updated {bf_name} at {venue}: "
                            f"BFSP {existing} -> {actual_bsp}"
                        )
                    break

        if not found:
            unmatched_runners.append(
                f"{bf_name} @ {runner['venue']} (BSP={actual_bsp:.2f})"
            )

    conn.commit()
    conn.close()

    log.info(
        f"  BSP sync complete: {matched} matched, {updated} updated, "
        f"{len(unmatched_runners)} unmatched"
    )

    if unmatched_runners and len(unmatched_runners) <= 20:
        for u in unmatched_runners:
            log.debug(f"    Unmatched: {u}")

    return {
        "date": str(target_date),
        "matched": matched,
        "updated": updated,
        "unmatched": len(unmatched_runners),
    }


def sync_bsp_range(
    client: BetfairClient,
    from_date: date,
    to_date: date,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Sync BSPs for a range of dates.

    Args:
        client: Authenticated BetfairClient instance.
        from_date: Start date (inclusive).
        to_date: End date (inclusive).
        db_path: Path to SQLite database.

    Returns:
        List of per-date summary dicts.
    """
    results = []
    current = from_date
    while current <= to_date:
        try:
            result = sync_bsps_for_date(client, current, db_path)
            results.append(result)
        except Exception as e:
            log.error(f"Failed to sync BSPs for {current}: {e}")
            results.append({
                "date": str(current), "matched": 0, "updated": 0,
                "unmatched": 0, "error": str(e),
            })
        current += timedelta(days=1)

    total_updated = sum(r["updated"] for r in results)
    total_matched = sum(r["matched"] for r in results)
    log.info(
        f"BSP range sync complete: {total_matched} matched, "
        f"{total_updated} updated across {len(results)} days"
    )
    return results


# ------------------------------------------------------------------
# Live Markets: Show upcoming/in-play odds
# ------------------------------------------------------------------

def show_live_markets(
    client: BetfairClient,
    target_date: date,
    upcoming_only: bool = False,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """Display live Betfair exchange odds for horse racing markets.

    Args:
        client: Authenticated BetfairClient instance.
        target_date: Date to query.
        upcoming_only: If True, only show non-completed markets.
        csv_path: Optional path to save results as CSV.

    Returns:
        DataFrame with live odds data.
    """
    if upcoming_only:
        log.info(f"Fetching non-completed markets for {target_date}...")
        markets = client.get_non_completed_markets(target_date)
        market_ids = [m["marketId"] for m in markets]
        # Get full odds only for non-completed
        all_runners = []
        for market in markets:
            mid = market["marketId"]
            event = market.get("event", {})
            venue = event.get("venue", event.get("name", ""))
            start_time = market.get("marketStartTime", "")
            runner_names = {
                r["selectionId"]: r["runnerName"]
                for r in market.get("runners", [])
            }
            try:
                book = client.get_market_odds(mid)
            except Exception as e:
                log.warning(f"Failed to get odds for {mid}: {e}")
                continue

            for runner in book.get("runners", []):
                sel_id = runner["selectionId"]
                back = runner.get("ex", {}).get("availableToBack", [])
                lay = runner.get("ex", {}).get("availableToLay", [])
                sp = runner.get("sp", {})
                best_back = back[0] if back else {}
                best_lay = lay[0] if lay else {}

                all_runners.append({
                    "venue": client._session and venue,
                    "start_time": start_time,
                    "market_status": market.get("_status", ""),
                    "runner_name": runner_names.get(sel_id, ""),
                    "runner_status": runner.get("status", ""),
                    "best_back": best_back.get("price"),
                    "best_back_size": best_back.get("size"),
                    "best_lay": best_lay.get("price"),
                    "best_lay_size": best_lay.get("size"),
                    "sp_near": sp.get("nearPrice"),
                    "sp_far": sp.get("farPrice"),
                    "last_traded": runner.get("lastPriceTraded"),
                    "market_id": mid,
                    "selection_id": sel_id,
                })
    else:
        log.info(f"Fetching all live odds for {target_date}...")
        raw = client.get_live_odds_for_date(target_date)
        all_runners = raw

    if not all_runners:
        log.info("No live market data found")
        return pd.DataFrame()

    df = pd.DataFrame(all_runners)

    # Display summary
    if "venue" in df.columns:
        venues = df["venue"].nunique() if "venue" in df.columns else 0
    else:
        venues = 0

    print(f"\n{'='*80}")
    print(f"LIVE BETFAIR MARKETS - {target_date}")
    print(f"{'='*80}")

    if upcoming_only:
        print(f"Showing non-completed markets only")

    # Group by venue/market
    group_col = "venue" if "venue" in df.columns else "market_name"
    time_col = "start_time" if "start_time" in df.columns else "market_start_time"

    if group_col in df.columns and time_col in df.columns:
        for venue, venue_df in df.groupby(group_col):
            print(f"\n  {venue}")
            print(f"  {'-'*70}")

            # Sub-group by market/race time
            for start_time, race_df in venue_df.groupby(time_col):
                # Parse start time for display
                try:
                    st = datetime.fromisoformat(
                        start_time.replace("Z", "+00:00")
                    )
                    time_display = st.strftime("%H:%M")
                except (ValueError, AttributeError):
                    time_display = str(start_time)

                status_col = "market_status" if "market_status" in race_df.columns else ""
                status = race_df[status_col].iloc[0] if status_col else ""

                print(f"\n    {time_display} [{status}]")

                # Sort by best back price
                back_col = "best_back" if "best_back" in race_df.columns else "best_back_price"
                name_col = "runner_name"

                if back_col in race_df.columns:
                    race_df = race_df.sort_values(back_col, na_position="last")

                print(f"    {'Horse':<30} {'Back':>8} {'Lay':>8} {'SP Near':>8} {'Last':>8}")
                print(f"    {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

                lay_col = "best_lay" if "best_lay" in race_df.columns else "best_lay_price"
                sp_col = "sp_near" if "sp_near" in race_df.columns else "sp_near_price"
                last_col = "last_traded" if "last_traded" in race_df.columns else "last_traded_price"

                for _, row in race_df.iterrows():
                    name = str(row.get(name_col, "?"))[:29]
                    back = row.get(back_col)
                    lay = row.get(lay_col)
                    sp_n = row.get(sp_col)
                    last = row.get(last_col)

                    back_s = f"{back:.2f}" if pd.notna(back) else "-"
                    lay_s = f"{lay:.2f}" if pd.notna(lay) else "-"
                    sp_s = f"{sp_n:.2f}" if pd.notna(sp_n) else "-"
                    last_s = f"{last:.2f}" if pd.notna(last) else "-"

                    r_status = row.get("runner_status", "")
                    flag = " [NR]" if r_status == "REMOVED" else ""

                    print(
                        f"    {name:<30} {back_s:>8} {lay_s:>8} "
                        f"{sp_s:>8} {last_s:>8}{flag}"
                    )

    print(f"\n{'='*80}")
    print(f"Total: {len(df)} runners across {venues} venues")
    print(f"{'='*80}\n")

    if csv_path:
        df.to_csv(csv_path, index=False)
        log.info(f"Live odds exported to {csv_path}")

    return df


# ------------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Betfair BSP sync and live market data"
    )

    # Mode flags
    parser.add_argument(
        "--bsp", action="store_true",
        help="Pull actual BSPs for completed races and update database",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Show live exchange odds for all markets",
    )
    parser.add_argument(
        "--upcoming", action="store_true",
        help="Show only non-completed/upcoming markets with live odds",
    )

    # Date options
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date (YYYY-MM-DD). Default: today for --live, yesterday for --bsp",
    )
    parser.add_argument(
        "--from", dest="from_date", type=str, default=None,
        help="Start date for BSP range sync (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to", dest="to_date", type=str, default=None,
        help="End date for BSP range sync (YYYY-MM-DD)",
    )

    # Options
    parser.add_argument(
        "--countries", type=str, default="GB,IE",
        help="Comma-separated country codes (default: GB,IE)",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export live odds to CSV file",
    )

    args = parser.parse_args()

    if not args.bsp and not args.live and not args.upcoming:
        parser.print_help()
        print("\nSpecify at least one of: --bsp, --live, --upcoming")
        sys.exit(1)

    country_codes = [c.strip() for c in args.countries.split(",")]

    # Login to Betfair
    client = BetfairClient()
    try:
        client.login()
    except BetfairAPIError as e:
        log.error(f"Betfair login failed: {e}")
        sys.exit(1)

    try:
        # BSP sync
        if args.bsp:
            if args.from_date and args.to_date:
                from_d = date.fromisoformat(args.from_date)
                to_d = date.fromisoformat(args.to_date)
                results = sync_bsp_range(client, from_d, to_d, args.db)
                print(f"\nBSP Sync Summary ({from_d} to {to_d}):")
                print(f"{'Date':<12} {'Matched':>8} {'Updated':>8} {'Unmatched':>10}")
                print("-" * 42)
                for r in results:
                    print(
                        f"{r['date']:<12} {r['matched']:>8} "
                        f"{r['updated']:>8} {r['unmatched']:>10}"
                    )
            else:
                target = (
                    date.fromisoformat(args.date)
                    if args.date
                    else date.today() - timedelta(days=1)
                )
                result = sync_bsps_for_date(client, target, args.db)
                print(
                    f"\nBSP Sync for {target}: "
                    f"{result['matched']} matched, {result['updated']} updated, "
                    f"{result['unmatched']} unmatched"
                )

        # Live markets
        if args.live or args.upcoming:
            target = (
                date.fromisoformat(args.date)
                if args.date
                else date.today()
            )
            show_live_markets(
                client, target,
                upcoming_only=args.upcoming,
                csv_path=args.csv,
            )

    finally:
        client.logout()


if __name__ == "__main__":
    main()
