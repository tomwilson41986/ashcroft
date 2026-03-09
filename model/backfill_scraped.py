"""
Backfill horse_racing.db with scraped data from horseracebase.com CSVs.

The current DB was populated from Betfair price CSVs which only contain
market data (BSP, WAP, etc). The scraper CSVs from horseracebase.com
contain rich metadata: jockey, trainer, going, distance, official rating,
race class, comment, prize money, headgear, age, weight, etc.

This script matches scraped records to existing Betfair records by
(race_date, track, horse_name) and backfills the missing columns.

Usage:
    python -m model.backfill_scraped --csv-dir csv/
    python -m model.backfill_scraped --csv-dir csv/ --db horse_racing.db
"""

import argparse
import csv
import glob
import io
import logging
import os
import re
import sqlite3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
DEFAULT_CSV_DIR = os.path.join(PROJECT_DIR, "csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Columns to backfill from scraped data
BACKFILL_COLUMNS = {
    "race_class": ("race_class", str),
    "going_description": ("going_description", str),
    "dist_furlongs": ("Dist_Furlongs", float),
    "jockey_name": ("jockey_name", str),
    "trainer": ("trainer", str),
    "horse_age": ("horse_age", int),
    "stall": ("stall", int),
    "pounds": ("pounds", int),
    "official_rating": ("official_rating", int),
    "median_or": ("MedianOR", int),
    "max_or_in_race": ("MaxORinRace", int),
    "odds": ("odds", float),
    "prize_money": ("prize_money", str),
    "race_type": ("RaceType", str),
    "surface_type": ("surfacetype", str),
    "horse_sex": ("HorseSex", str),
    "headgear": ("Headgear", str),
    "comment": ("Comment", str),
    "days_since_lr": ("dayssincelr", int),
    "career_runs": ("careerruns", int),
}


def _safe_val(val, dtype):
    """Convert a scraped value to the target type."""
    if val is None or (isinstance(val, str) and not val.strip()):
        return None
    try:
        if dtype == int:
            return int(float(val))
        elif dtype == float:
            return float(val)
        return str(val).strip()
    except (ValueError, TypeError):
        return None


def backfill(db_path: str, csv_dir: str):
    """Backfill scraped metadata into existing DB records."""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "results_*.csv")))
    if not csv_files:
        log.error(f"No scraped CSV files found in {csv_dir}")
        return

    log.info(f"Found {len(csv_files)} scraped CSV files")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Build SET clause
    set_parts = [f"{db_col} = ?" for db_col in BACKFILL_COLUMNS.keys()]
    set_clause = ", ".join(set_parts)

    update_sql = f"""
        UPDATE race_results
        SET {set_clause}
        WHERE race_date = ? AND track = ? AND horse_name = ?
        AND (jockey_name IS NULL OR jockey_name = '')
    """

    total_updated = 0
    total_rows = 0

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                batch = []

                for row in reader:
                    total_rows += 1
                    race_date = row.get("racedate", "").strip()
                    track = row.get("track", "").strip()
                    # Strip country suffix: "Horse Name (IRE)" -> "Horse Name"
                    horse_name = re.sub(
                        r"\s*\([A-Z]{2,3}\)\s*$", "",
                        row.get("horse_name", "").strip()
                    )

                    if not race_date or not track or not horse_name:
                        continue

                    # Build values list
                    values = []
                    for db_col, (csv_col, dtype) in BACKFILL_COLUMNS.items():
                        values.append(_safe_val(row.get(csv_col), dtype))

                    # WHERE clause values
                    values.extend([race_date, track, horse_name])
                    batch.append(tuple(values))

                    if len(batch) >= 1000:
                        conn.executemany(update_sql, batch)
                        total_updated += conn.total_changes
                        batch = []

                if batch:
                    before = conn.total_changes
                    conn.executemany(update_sql, batch)
                    total_updated += conn.total_changes - before

                conn.commit()

        except Exception as e:
            log.warning(f"Error processing {filename}: {e}")

        if total_rows % 50000 == 0 and total_rows > 0:
            log.info(f"  Processed {total_rows:,} rows...")

    conn.close()
    log.info(f"Done. Processed {total_rows:,} scraped rows, "
             f"updated {total_updated:,} DB records")


def check_coverage(db_path: str):
    """Report how many records have backfilled data."""
    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]

    log.info(f"\nData coverage report ({total:,} total records):")
    for col in BACKFILL_COLUMNS.keys():
        n = conn.execute(
            f"SELECT COUNT(*) FROM race_results WHERE [{col}] IS NOT NULL AND [{col}] != ''"
        ).fetchone()[0]
        pct = n / total * 100 if total > 0 else 0
        log.info(f"  {col:<25} {n:>8,} / {total:,}  ({pct:5.1f}%)")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill scraped metadata into horse_racing.db"
    )
    parser.add_argument(
        "--csv-dir", type=str, default=DEFAULT_CSV_DIR,
        help=f"Directory with scraped CSV files (default: {DEFAULT_CSV_DIR})",
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB,
        help=f"Database path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only check current coverage, don't backfill",
    )
    args = parser.parse_args()

    if args.check:
        check_coverage(args.db)
    else:
        backfill(args.db, args.csv_dir)
        check_coverage(args.db)


if __name__ == "__main__":
    main()
