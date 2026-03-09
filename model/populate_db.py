"""
Populate horse_racing.db from Betfair price CSV files.

Reads the processed betfair CSVs (data/betfair/betfair_prices_*.csv)
and loads them into a SQLite database with a schema suitable for
the BFSP prediction model.

Usage:
    python -m model.populate_db
    python -m model.populate_db --db horse_racing.db
"""

import argparse
import logging
import os
import sqlite3

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
BETFAIR_DIR = os.path.join(PROJECT_DIR, "data", "betfair")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def init_db(db_path: str) -> sqlite3.Connection:
    """Create database and race_results table."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS race_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT NOT NULL,
            race_time TEXT,
            track TEXT NOT NULL,
            country TEXT,
            event_id TEXT,
            selection_id TEXT,
            horse_name TEXT NOT NULL,
            event_name TEXT,
            race_number TEXT,
            bfsp REAL,
            bfsp_place REAL,
            win_result INTEGER,
            place_result INTEGER,
            placing_numerical INTEGER,
            number_of_runners INTEGER,
            ppwap REAL,
            morningwap REAL,
            ppmax REAL,
            ppmin REAL,
            ipmax REAL,
            ipmin REAL,
            ppwap_place REAL,
            morningwap_place REAL,
            ppmax_place REAL,
            ppmin_place REAL,
            ipmax_place REAL,
            ipmin_place REAL,
            source TEXT,
            -- Columns expected by existing model code (filled where possible)
            race_class TEXT,
            race_name TEXT,
            going_description TEXT,
            dist_furlongs REAL,
            jockey_name TEXT,
            trainer TEXT,
            horse_age INTEGER,
            stall INTEGER,
            pounds INTEGER,
            official_rating INTEGER,
            median_or INTEGER,
            max_or_in_race INTEGER,
            odds REAL,
            prize_money REAL,
            race_type TEXT,
            surface_type TEXT,
            horse_sex TEXT,
            headgear TEXT,
            comment TEXT,
            days_since_lr INTEGER,
            career_runs INTEGER,
            UNIQUE(race_date, event_id, selection_id)
        );

        CREATE INDEX IF NOT EXISTS idx_rr_date ON race_results(race_date);
        CREATE INDEX IF NOT EXISTS idx_rr_track ON race_results(track);
        CREATE INDEX IF NOT EXISTS idx_rr_horse ON race_results(horse_name);
        CREATE INDEX IF NOT EXISTS idx_rr_event ON race_results(event_id);
    """)
    conn.commit()
    return conn


def load_betfair_csvs(betfair_dir: str) -> pd.DataFrame:
    """Load and combine all betfair CSV files."""
    import glob

    csv_files = sorted(glob.glob(os.path.join(betfair_dir, "betfair_prices_*.csv")))
    if not csv_files:
        log.error(f"No betfair CSV files found in {betfair_dir}")
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        log.info(f"Reading {os.path.basename(f)}...")
        df = pd.read_csv(f, dtype=str)
        frames.append(df)
        log.info(f"  {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total: {len(combined):,} rows from {len(csv_files)} files")
    return combined


def _safe_float(val):
    """Convert to float or None."""
    if pd.isna(val) or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    """Convert to int or None."""
    f = _safe_float(val)
    if f is None:
        return None
    return int(f)


def populate(db_path: str, betfair_dir: str):
    """Main population routine."""
    df = load_betfair_csvs(betfair_dir)
    if df.empty:
        return

    conn = init_db(db_path)

    # Compute number_of_runners per race (event_id)
    runner_counts = df.groupby("event_id").size().to_dict()

    # Derive placing_numerical from win_result:
    # win_result=1 means the horse won (placing=1)
    # We don't have exact placing for non-winners, but we can use bfsp rank
    # within each race as a proxy for finishing position

    # Compute rank within race by BSP (lower BSP = higher market rank)
    df["_bfsp_float"] = pd.to_numeric(df["win_bsp"], errors="coerce")
    df["_win_result_int"] = pd.to_numeric(df["win_result"], errors="coerce").fillna(0).astype(int)

    # For placing: winners get 1, others get rank by BSP within race
    # (This is an approximation - real placing data would come from scraper)
    log.info("Computing derived fields...")
    placing = []
    for event_id, group in df.groupby("event_id"):
        n = len(group)
        for idx in group.index:
            if group.loc[idx, "_win_result_int"] == 1:
                placing.append((idx, 1))
            else:
                # Use BSP rank as proxy (lower BSP = better expected finish)
                bsp = group.loc[idx, "_bfsp_float"]
                if pd.notna(bsp):
                    rank = (group["_bfsp_float"] < bsp).sum() + 1
                    # Clamp: winner already assigned 1, so non-winners start at 2
                    placing.append((idx, max(2, int(rank))))
                else:
                    placing.append((idx, n))

    placing_dict = dict(placing)
    df["_placing"] = df.index.map(placing_dict)

    log.info("Inserting into database...")
    rows_inserted = 0

    for _, row in df.iterrows():
        event_id = row.get("event_id", "")
        n_runners = runner_counts.get(event_id, None)

        try:
            conn.execute("""
                INSERT OR IGNORE INTO race_results (
                    race_date, race_time, track, country, event_id,
                    selection_id, horse_name, event_name, race_number,
                    bfsp, bfsp_place, win_result, place_result,
                    placing_numerical, number_of_runners,
                    ppwap, morningwap, ppmax, ppmin, ipmax, ipmin,
                    ppwap_place, morningwap_place, ppmax_place, ppmin_place,
                    ipmax_place, ipmin_place, source, race_name
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                str(row.get("event_date", "")).strip(),
                str(row.get("event_time", "")).strip(),
                str(row.get("track", "")).strip(),
                str(row.get("country", "")).strip(),
                str(event_id).strip(),
                str(row.get("selection_id", "")).strip(),
                str(row.get("selection_name", "")).strip(),
                str(row.get("event_name", "")).strip(),
                str(row.get("race_number", "")).strip(),
                _safe_float(row.get("win_bsp")),
                _safe_float(row.get("place_bsp")),
                _safe_int(row.get("win_result")),
                _safe_int(row.get("place_result")),
                row.get("_placing"),
                n_runners,
                _safe_float(row.get("ppwap")),
                _safe_float(row.get("morningwap")),
                _safe_float(row.get("ppmax")),
                _safe_float(row.get("ppmin")),
                _safe_float(row.get("ipmax")),
                _safe_float(row.get("ipmin")),
                _safe_float(row.get("ppwap_place")),
                _safe_float(row.get("morningwap_place")),
                _safe_float(row.get("ppmax_place")),
                _safe_float(row.get("ppmin_place")),
                _safe_float(row.get("ipmax_place")),
                _safe_float(row.get("ipmin_place")),
                str(row.get("source", "")).strip(),
                str(row.get("event_name", "")).strip(),
            ))
            rows_inserted += 1
        except sqlite3.Error as e:
            log.warning(f"DB error: {e}")

        if rows_inserted % 10000 == 0 and rows_inserted > 0:
            conn.commit()
            log.info(f"  {rows_inserted:,} rows inserted...")

    conn.commit()
    conn.close()
    log.info(f"Done. {rows_inserted:,} rows inserted into {db_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Populate horse_racing.db from Betfair price CSVs"
    )
    parser.add_argument(
        "--db", type=str, default=DEFAULT_DB,
        help=f"Database path (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--betfair-dir", type=str, default=BETFAIR_DIR,
        help=f"Betfair CSV directory (default: {BETFAIR_DIR})",
    )
    args = parser.parse_args()

    populate(args.db, args.betfair_dir)


if __name__ == "__main__":
    main()
