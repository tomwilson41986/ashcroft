"""
Betfair SP Price Data Processor

Combines raw Betfair CSV files into clean datasets stored in data/betfair/.
Handles two input formats:

1. Automation Hub files (UK_IE_Thoroughbred_Racing_Model_*.csv)
   - Columns: LOCAL_RACE_DATE, RACETIME, TRACK, EVENT_ID, SELECTION_ID,
     RUNNER_NAME, WIN_BSP, PLACE_BSP, WIN_RESULT, PLACE_RESULT, etc.

2. Betfair Promo files (dwbfprices*.csv)
   - Columns: EVENT_ID, MENU_HINT, EVENT_NAME, EVENT_DT, SELECTION_ID,
     SELECTION_NAME, WIN_LOSE, BSP, PPWAP, MORNINGWAP, etc.

Output files in data/betfair/:
    betfair_prices.csv  - Combined master file with all price data per runner

Usage:
    python -m betfair_prices.process                    # Process all raw files
    python -m betfair_prices.process --year 2023        # Process single year
"""

import argparse
import csv
import glob
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_DIR, "data", "betfair_raw")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "betfair")

# Unified output columns
OUTPUT_COLUMNS = [
    "event_date", "event_time", "track", "country",
    "event_id", "selection_id", "selection_name",
    "event_name", "race_number",
    "win_bsp", "place_bsp", "win_result", "place_result",
    # Extended price columns (promo files only)
    "ppwap", "morningwap", "ppmax", "ppmin", "ipmax", "ipmin",
    "ppwap_place", "morningwap_place", "ppmax_place", "ppmin_place",
    "ipmax_place", "ipmin_place",
    "source",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Regex for promo filenames: dwbfprices{country}{market}{DDMMYYYY}.csv
PROMO_RE = re.compile(r"dwbfprices(uk|ire)(win|place[d]?)(\d{8})\.csv")

# Regex for hub filenames: UK_IE_Thoroughbred_Racing_Model_YYYY[-MM].csv
HUB_RE = re.compile(r"UK_IE_Thoroughbred_Racing_Model_(\d{4})(?:-(\d{2}))?\.csv")

# Country ID mapping for Automation Hub data
COUNTRY_MAP = {
    "1": "uk",
    "1.0": "uk",
    "35": "ire",
    "35.0": "ire",
}


def safe_float(val: str) -> str:
    """Return cleaned float string or empty."""
    if not val:
        return ""
    val = val.strip()
    if not val:
        return ""
    try:
        return str(float(val))
    except ValueError:
        return ""


def parse_hub_date(local_date: str, racetime: str) -> tuple[str, str]:
    """Parse Automation Hub date fields into (YYYY-MM-DD, HH:MM).

    LOCAL_RACE_DATE: "6/01/2024"
    RACETIME: "7/01/2024 0:25" (UTC)
    """
    event_date = ""
    event_time = ""

    # Parse LOCAL_RACE_DATE
    for fmt in ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]:
        try:
            dt = datetime.strptime(local_date.strip(), fmt)
            event_date = dt.strftime("%Y-%m-%d")
            break
        except ValueError:
            continue

    # Parse RACETIME for time component
    racetime = racetime.strip()
    if racetime:
        for fmt in ["%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S",
                    "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S"]:
            try:
                dt = datetime.strptime(racetime, fmt)
                event_time = dt.strftime("%H:%M")
                break
            except ValueError:
                continue

    return (event_date, event_time)


def parse_promo_event_dt(event_dt: str) -> tuple[str, str]:
    """Parse promo EVENT_DT into (YYYY-MM-DD, HH:MM)."""
    event_dt = event_dt.strip()
    if not event_dt:
        return ("", "")

    for fmt in ["%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S",
                "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
        try:
            dt = datetime.strptime(event_dt, fmt)
            return (dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M"))
        except ValueError:
            continue

    parts = event_dt.split()
    if len(parts) >= 2:
        return (parts[0], parts[1])
    return (event_dt, "")


def process_hub_file(filepath: str) -> list[dict]:
    """Process an Automation Hub CSV file into standardized rows."""
    rows = []
    basename = os.path.basename(filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty/blank rows
                if not row.get("LOCAL_RACE_DATE", "").strip():
                    continue

                event_date, event_time = parse_hub_date(
                    row.get("LOCAL_RACE_DATE", ""),
                    row.get("RACETIME", "")
                )

                country_id = row.get("COUNTRY_ID", "").strip()
                country = COUNTRY_MAP.get(country_id, country_id)

                rows.append({
                    "event_date": event_date,
                    "event_time": event_time,
                    "track": row.get("TRACK", "").strip(),
                    "country": country,
                    "event_id": row.get("EVENT_ID", "").strip(),
                    "selection_id": row.get("SELECTION_ID", "").strip(),
                    "selection_name": row.get("RUNNER_NAME", "").strip(),
                    "event_name": row.get("RACE_NAME", "").strip(),
                    "race_number": row.get("RACE_NUMBER", "").strip(),
                    "win_bsp": safe_float(row.get("WIN_BSP", "")),
                    "place_bsp": safe_float(row.get("PLACE_BSP", "")),
                    "win_result": row.get("WIN_RESULT", "").strip(),
                    "place_result": row.get("PLACE_RESULT", "").strip(),
                    "ppwap": "", "morningwap": "", "ppmax": "", "ppmin": "",
                    "ipmax": "", "ipmin": "",
                    "ppwap_place": "", "morningwap_place": "",
                    "ppmax_place": "", "ppmin_place": "",
                    "ipmax_place": "", "ipmin_place": "",
                    "source": "hub",
                })
    except Exception as e:
        log.warning(f"Error reading {filepath}: {e}")

    return rows


def process_promo_files(win_path: str, place_path: str | None,
                        country: str, date_str: str) -> list[dict]:
    """Process a pair of promo win/place files into standardized rows."""
    win_rows = {}
    basename = os.path.basename(win_path)

    try:
        with open(win_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_date, event_time = parse_promo_event_dt(
                    row.get("EVENT_DT", "")
                )
                if not event_date:
                    event_date = date_str

                sel_id = row.get("SELECTION_ID", "").strip()
                evt_id = row.get("EVENT_ID", "").strip()
                key = (evt_id, sel_id)

                win_rows[key] = {
                    "event_date": event_date,
                    "event_time": event_time,
                    "track": row.get("MENU_HINT", "").strip(),
                    "country": country,
                    "event_id": evt_id,
                    "selection_id": sel_id,
                    "selection_name": row.get("SELECTION_NAME", "").strip(),
                    "event_name": row.get("EVENT_NAME", "").strip(),
                    "race_number": "",
                    "win_bsp": safe_float(row.get("BSP", "")),
                    "place_bsp": "",
                    "win_result": row.get("WIN_LOSE", "").strip(),
                    "place_result": "",
                    "ppwap": safe_float(row.get("PPWAP", "")),
                    "morningwap": safe_float(row.get("MORNINGWAP", "")),
                    "ppmax": safe_float(row.get("PPMAX", "")),
                    "ppmin": safe_float(row.get("PPMIN", "")),
                    "ipmax": safe_float(row.get("IPMAX", "")),
                    "ipmin": safe_float(row.get("IPMIN", "")),
                    "ppwap_place": "", "morningwap_place": "",
                    "ppmax_place": "", "ppmin_place": "",
                    "ipmax_place": "", "ipmin_place": "",
                    "source": "promo",
                }
    except Exception as e:
        log.warning(f"Error reading {win_path}: {e}")

    # Merge place data if available
    if place_path and os.path.exists(place_path):
        try:
            with open(place_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sel_id = row.get("SELECTION_ID", "").strip()
                    evt_id = row.get("EVENT_ID", "").strip()
                    key = (evt_id, sel_id)

                    if key in win_rows:
                        win_rows[key]["place_bsp"] = safe_float(row.get("BSP", ""))
                        win_rows[key]["place_result"] = row.get("WIN_LOSE", "").strip()
                        win_rows[key]["ppwap_place"] = safe_float(row.get("PPWAP", ""))
                        win_rows[key]["morningwap_place"] = safe_float(row.get("MORNINGWAP", ""))
                        win_rows[key]["ppmax_place"] = safe_float(row.get("PPMAX", ""))
                        win_rows[key]["ppmin_place"] = safe_float(row.get("PPMIN", ""))
                        win_rows[key]["ipmax_place"] = safe_float(row.get("IPMAX", ""))
                        win_rows[key]["ipmin_place"] = safe_float(row.get("IPMIN", ""))
        except Exception as e:
            log.warning(f"Error reading {place_path}: {e}")

    return list(win_rows.values())


def find_promo_pairs(raw_dir: str) -> list[tuple[str, str | None, str, str]]:
    """Find pairs of promo win+place files.

    Returns list of (win_path, place_path_or_none, country, date_str).
    """
    files = {}
    for filepath in glob.glob(os.path.join(raw_dir, "dwbfprices*.csv")):
        m = PROMO_RE.match(os.path.basename(filepath))
        if not m:
            continue
        country = m.group(1)
        market = "place" if m.group(2).startswith("place") else "win"
        ddmmyyyy = m.group(3)
        try:
            dt = datetime.strptime(ddmmyyyy, "%d%m%Y")
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

        key = (country, date_str)
        if key not in files:
            files[key] = {"win": None, "place": None}
        files[key][market] = filepath

    pairs = []
    for (country, date_str), paths in sorted(files.items()):
        if paths["win"]:
            pairs.append((paths["win"], paths["place"], country, date_str))

    return pairs


def process_all(raw_dir: str = RAW_DIR, output_dir: str = OUTPUT_DIR,
                year_filter: int | None = None):
    """Process all raw files into a single combined output."""
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    # Process Automation Hub files
    hub_files = sorted(glob.glob(os.path.join(raw_dir, "UK_IE_Thoroughbred_Racing_Model_*.csv")))
    for filepath in hub_files:
        m = HUB_RE.match(os.path.basename(filepath))
        if not m:
            continue

        file_year = int(m.group(1))
        if year_filter and file_year != year_filter:
            continue

        rows = process_hub_file(filepath)
        log.info(f"Hub: {os.path.basename(filepath)} -> {len(rows)} rows")
        all_rows.extend(rows)

    # Process Promo files
    promo_pairs = find_promo_pairs(raw_dir)
    for win_path, place_path, country, date_str in promo_pairs:
        file_year = int(date_str[:4])
        if year_filter and file_year != year_filter:
            continue

        rows = process_promo_files(win_path, place_path, country, date_str)
        all_rows.extend(rows)

    if promo_pairs:
        log.info(f"Promo: {len(promo_pairs)} date/country pairs processed")

    # Filter out empty rows
    all_rows = [r for r in all_rows if r["event_date"] and r["selection_name"]]

    # Deduplicate: prefer promo rows (more price data) over hub rows
    seen = {}
    for row in all_rows:
        key = (row["event_date"], row["event_id"], row["selection_id"])
        if key in seen:
            existing = seen[key]
            # Prefer promo source (has extended price data)
            if existing["source"] == "promo":
                continue
        seen[key] = row

    deduped = list(seen.values())
    deduped.sort(key=lambda r: (r["event_date"], r["event_time"], r["track"],
                                r["selection_name"]))

    # Write combined output
    output_path = os.path.join(output_dir, "betfair_prices.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)

    log.info(f"Wrote {len(deduped)} rows to {output_path}")

    # Also write per-year files
    by_year = defaultdict(list)
    for row in deduped:
        if row["event_date"]:
            by_year[row["event_date"][:4]].append(row)

    for year, rows in sorted(by_year.items()):
        year_path = os.path.join(output_dir, f"betfair_prices_{year}.csv")
        with open(year_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Wrote {len(rows)} rows to betfair_prices_{year}.csv")

    return len(deduped)


def main():
    parser = argparse.ArgumentParser(
        description="Process raw Betfair price CSVs into clean datasets"
    )
    parser.add_argument("--year", type=int,
                        help="Process only a specific year")
    parser.add_argument("--raw-dir", type=str, default=RAW_DIR,
                        help=f"Raw CSV directory (default: {RAW_DIR})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    process_all(args.raw_dir, args.output_dir, args.year)


if __name__ == "__main__":
    main()
