"""
Merge Betfair SP Price Data with Horse Racing Results

Joins the processed Betfair price data (from data/betfair/betfair_prices.csv)
with the race results from the SQLite database (horse_racing.db) or CSV files.

Merge strategy:
    Primary: race date + normalized horse name + normalized track name
    Fallback: race date + normalized horse name (ignoring track differences)

Output:
    data/merged/results_with_prices.csv  - Full merged dataset ready for modeling

Usage:
    python -m betfair_prices.merge                         # Merge from DB
    python -m betfair_prices.merge --results results.csv   # Merge from CSV
    python -m betfair_prices.merge --export-db             # Export DB to CSV first
"""

import argparse
import csv
import logging
import os
import re
import sqlite3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_DIR, "horse_racing.db")
BETFAIR_PRICES = os.path.join(PROJECT_DIR, "data", "betfair", "betfair_prices.csv")
MERGED_DIR = os.path.join(PROJECT_DIR, "data", "merged")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Normalize a horse name for fuzzy matching."""
    name = name.strip().lower()
    # Remove content in parentheses like "(IRE)", "(FR)"
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = name.replace("'", "").replace("\u2019", "").replace("-", " ")
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalize_track(track: str) -> str:
    """Normalize track name for matching."""
    track = normalize_name(track)
    replacements = {
        "kempton park": "kempton",
        "haydock park": "haydock",
    }
    for old, new in replacements.items():
        track = track.replace(old, new)
    return track.strip()


def load_betfair_prices(filepath: str = BETFAIR_PRICES) -> dict:
    """Load Betfair prices indexed by (date, normalized_horse, normalized_track)."""
    prices = {}

    if not os.path.exists(filepath):
        log.error(f"Betfair prices file not found: {filepath}")
        return prices

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            horse = normalize_name(row.get("selection_name", ""))
            track = normalize_track(row.get("track", ""))
            date_str = row.get("event_date", "")

            if not horse or not date_str:
                continue

            key = (date_str, horse, track)
            prices[key] = row

    log.info(f"Loaded {len(prices)} Betfair price records")
    return prices


def load_results_from_db(db_path: str = DB_PATH) -> list[dict]:
    """Load race results from SQLite database."""
    if not os.path.exists(db_path):
        log.error(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM race_results ORDER BY race_date, race_time")
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    log.info(f"Loaded {len(results)} results from database")
    return results


def load_results_from_csv(filepath: str) -> list[dict]:
    """Load race results from CSV file."""
    with open(filepath, "r", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    log.info(f"Loaded {len(results)} results from {filepath}")
    return results


def export_db_to_csv(db_path: str = DB_PATH, output_dir: str = MERGED_DIR) -> str:
    """Export database to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results_export.csv")

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT * FROM race_results ORDER BY race_date, race_time")
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    conn.close()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    log.info(f"Exported {len(rows)} rows to {output_path}")
    return output_path


# Betfair columns to add to merged output
BF_PRICE_COLS = [
    ("bf_win_bsp", "win_bsp"),
    ("bf_place_bsp", "place_bsp"),
    ("bf_win_result", "win_result"),
    ("bf_place_result", "place_result"),
    ("bf_ppwap", "ppwap"),
    ("bf_morningwap", "morningwap"),
    ("bf_ppmax", "ppmax"),
    ("bf_ppmin", "ppmin"),
    ("bf_ipmax", "ipmax"),
    ("bf_ipmin", "ipmin"),
    ("bf_ppwap_place", "ppwap_place"),
    ("bf_morningwap_place", "morningwap_place"),
    ("bf_ppmax_place", "ppmax_place"),
    ("bf_ppmin_place", "ppmin_place"),
    ("bf_ipmax_place", "ipmax_place"),
    ("bf_ipmin_place", "ipmin_place"),
    ("bf_event_id", "event_id"),
    ("bf_selection_id", "selection_id"),
    ("bf_source", "source"),
]


def merge_data(results: list[dict], prices: dict,
               output_dir: str = MERGED_DIR) -> str:
    """Merge results with Betfair prices and write output CSV."""
    os.makedirs(output_dir, exist_ok=True)

    if not results:
        log.error("No results to merge")
        return ""

    result_columns = list(results[0].keys())
    price_col_names = [c[0] for c in BF_PRICE_COLS]
    output_columns = result_columns + price_col_names + ["bf_match_type"]
    output_path = os.path.join(output_dir, "results_with_prices.csv")

    matched = 0
    matched_fuzzy = 0
    unmatched = 0

    merged_rows = []

    for row in results:
        race_date = row.get("race_date", "")
        horse = normalize_name(row.get("horse_name", ""))
        track = normalize_track(row.get("track", ""))

        price_row = None
        match_type = ""

        # Exact match on (date, horse, track)
        key = (race_date, horse, track)
        if key in prices:
            price_row = prices[key]
            match_type = "exact"
            matched += 1
        else:
            # Fallback: match on (date, horse) ignoring track
            for pkey, pval in prices.items():
                if pkey[0] == race_date and pkey[1] == horse:
                    price_row = pval
                    match_type = "horse_only"
                    matched_fuzzy += 1
                    break

        if not price_row:
            unmatched += 1

        merged = dict(row)
        if price_row:
            for out_col, src_col in BF_PRICE_COLS:
                merged[out_col] = price_row.get(src_col, "")
            merged["bf_match_type"] = match_type
        else:
            for out_col, _ in BF_PRICE_COLS:
                merged[out_col] = ""
            merged["bf_match_type"] = ""

        merged_rows.append(merged)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    total = len(results)
    total_matched = matched + matched_fuzzy
    match_rate = (total_matched / total * 100) if total > 0 else 0

    log.info(f"Merge complete: {total} results")
    log.info(f"  Exact matches:     {matched}")
    log.info(f"  Fuzzy matches:     {matched_fuzzy}")
    log.info(f"  Unmatched:         {unmatched}")
    log.info(f"  Match rate:        {match_rate:.1f}%")
    log.info(f"  Output: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge Betfair prices with race results"
    )
    parser.add_argument("--results", type=str,
                        help="Path to results CSV (default: load from DB)")
    parser.add_argument("--prices", type=str, default=BETFAIR_PRICES,
                        help=f"Path to Betfair prices CSV (default: {BETFAIR_PRICES})")
    parser.add_argument("--output-dir", type=str, default=MERGED_DIR,
                        help=f"Output directory (default: {MERGED_DIR})")
    parser.add_argument("--export-db", action="store_true",
                        help="Export database to CSV before merging")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help=f"Database path (default: {DB_PATH})")
    args = parser.parse_args()

    if args.export_db:
        export_db_to_csv(args.db, args.output_dir)

    if args.results:
        results = load_results_from_csv(args.results)
    else:
        results = load_results_from_db(args.db)

    if not results:
        log.error("No results loaded. Use --results <csv> or ensure horse_racing.db exists.")
        return

    prices = load_betfair_prices(args.prices)
    if not prices:
        log.error("No Betfair prices loaded. Run download + process first.")
        return

    merge_data(results, prices, args.output_dir)


if __name__ == "__main__":
    main()
