"""
Horse Racing Results Scraper for horseracebase.com

Downloads CSV results per date and stores them in a SQLite database.
Also saves raw CSV files to a csv/ directory.
Credentials are read from environment variables (HRB_USERNAME, HRB_PASSWORD)
or a .env file.

Usage:
    python scraper.py                  # Scrape from last saved date to today
    python scraper.py --from 2020-01-01 --to 2020-12-31  # Scrape specific range
    python scraper.py --full           # Full scrape from 2010-01-01 to today
"""

import argparse
import csv
import io
import logging
import os
import sqlite3
import sys
import time
from datetime import date, timedelta

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
CSV_DIR = os.path.join(SCRIPT_DIR, "csv")
BASE_URL = "https://www.horseracebase.com"
RESULTS_URL = f"{BASE_URL}/horse-racing-results.php"
LOGIN_URL = f"{BASE_URL}/horsebase1.php"
CSV_URL = f"{BASE_URL}/excelresults.php"

# Rate limiting: seconds between requests
REQUEST_DELAY = 1.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SCRIPT_DIR, "scraper.log")),
    ],
)
log = logging.getLogger(__name__)

# CSV columns from horseracebase.com
CSV_COLUMNS = [
    "racedate", "racetime", "track", "race_name", "race_restrictions_age",
    "race_class", "major", "race_distance", "prize_money", "going_description",
    "number_of_runners", "place", "distbt", "horse_name", "stall", "trainer",
    "horse_age", "jockey_name", "jockeys_claim", "pounds", "odds", "fav",
    "official_rating", "comptime", "comptime_numeric", "TotalDstBt", "MedianOR",
    "Dist_Furlongs", "placing_numerical", "RCode", "BFSP", "BFSP_Place",
    "PlcsPaid", "BFPlcsPaid", "Yards", "RailMove", "RaceType", "Comment",
    "CardNo", "StallPositioning", "TrackDirection", "Headgear", "dayssincelr",
    "careerruns", "stallion", "surfacetype", "horse_prizewin", "HorseSex",
    "dam", "damstallion", "MaxORinRace",
]


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create database and tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS race_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT NOT NULL,
            race_time TEXT,
            track TEXT NOT NULL,
            race_name TEXT,
            race_restrictions_age TEXT,
            race_class TEXT,
            major TEXT,
            race_distance TEXT,
            prize_money TEXT,
            going_description TEXT,
            number_of_runners INTEGER,
            place TEXT,
            distbt TEXT,
            horse_name TEXT NOT NULL,
            stall INTEGER,
            trainer TEXT,
            horse_age INTEGER,
            jockey_name TEXT,
            jockeys_claim INTEGER,
            pounds INTEGER,
            odds REAL,
            fav TEXT,
            official_rating INTEGER,
            comptime TEXT,
            comptime_numeric REAL,
            total_dst_bt TEXT,
            median_or INTEGER,
            dist_furlongs REAL,
            placing_numerical INTEGER,
            race_code TEXT,
            bfsp REAL,
            bfsp_place REAL,
            plcs_paid INTEGER,
            bf_plcs_paid INTEGER,
            yards INTEGER,
            rail_move TEXT,
            race_type TEXT,
            comment TEXT,
            card_no INTEGER,
            stall_positioning TEXT,
            track_direction TEXT,
            headgear TEXT,
            days_since_lr INTEGER,
            career_runs INTEGER,
            stallion TEXT,
            surface_type TEXT,
            horse_prizewin TEXT,
            horse_sex TEXT,
            dam TEXT,
            dam_stallion TEXT,
            max_or_in_race INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(race_date, race_time, track, horse_name)
        );

        CREATE INDEX IF NOT EXISTS idx_race_date ON race_results(race_date);
        CREATE INDEX IF NOT EXISTS idx_track ON race_results(track);
        CREATE INDEX IF NOT EXISTS idx_horse_name ON race_results(horse_name);
        CREATE INDEX IF NOT EXISTS idx_trainer ON race_results(trainer);
        CREATE INDEX IF NOT EXISTS idx_jockey ON race_results(jockey_name);
        CREATE INDEX IF NOT EXISTS idx_stallion ON race_results(stallion);
        CREATE INDEX IF NOT EXISTS idx_place ON race_results(place);
        CREATE INDEX IF NOT EXISTS idx_race_type ON race_results(race_type);

        CREATE TABLE IF NOT EXISTS scrape_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scrape_date TEXT NOT NULL UNIQUE,
            rows_found INTEGER DEFAULT 0,
            status TEXT DEFAULT 'ok',
            scraped_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    return conn


def login(session: requests.Session) -> str | None:
    """Log in to horseracebase.com. Returns user ID on success, None on failure."""
    username = os.getenv("HRB_USERNAME")
    password = os.getenv("HRB_PASSWORD")

    if not username or not password:
        log.error("HRB_USERNAME and HRB_PASSWORD environment variables must be set")
        return None

    # Get CSRF token
    resp = session.get(RESULTS_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    csrf_input = soup.find("input", {"name": "CSRFtoken"})
    if not csrf_input:
        log.error("Could not find CSRF token on page")
        return None

    # Post login
    resp2 = session.post(LOGIN_URL, data={
        "login": username,
        "password": password,
        "CSRFtoken": csrf_input.get("value"),
    })
    resp2.raise_for_status()

    if "not recognised" in resp2.text.lower():
        log.error("Login failed: credentials not recognised")
        return None

    # Get user ID from results page
    resp3 = session.get(RESULTS_URL)
    soup3 = BeautifulSoup(resp3.text, "lxml")
    user_input = soup3.find("input", {"name": "user"})
    if not user_input:
        log.error("Login failed: could not find user ID")
        return None

    user_id = user_input.get("value")
    log.info(f"Logged in successfully (user_id={user_id})")
    return user_id


def safe_int(val: str) -> int | None:
    """Convert string to int, returning None for empty/invalid values."""
    if not val or not val.strip():
        return None
    try:
        return int(val.strip())
    except ValueError:
        return None


def safe_float(val: str) -> float | None:
    """Convert string to float, returning None for empty/invalid values."""
    if not val or not val.strip():
        return None
    try:
        return float(val.strip())
    except ValueError:
        return None


def download_csv(session: requests.Session, user_id: str,
                 target_date: date) -> str | None:
    """Download CSV for a given date. Returns CSV text or None."""
    racedate = f"{target_date.year}-{target_date.month}-{target_date.day}"

    resp = session.post(CSV_URL, data={
        "csv": "1",
        "user": user_id,
        "racedate": racedate,
    })
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "csv" not in content_type and "text" not in content_type:
        # Might be redirected to login page
        log.warning(f"Unexpected content type: {content_type}")
        return None

    if not resp.text.strip() or "racedate" not in resp.text[:100]:
        return None

    return resp.text


def save_csv_file(csv_text: str, target_date: date):
    """Save raw CSV to csv/ directory."""
    os.makedirs(CSV_DIR, exist_ok=True)
    filepath = os.path.join(CSV_DIR, f"results_{target_date.isoformat()}.csv")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(csv_text)


def parse_and_save(conn: sqlite3.Connection, csv_text: str,
                   target_date: date) -> int:
    """Parse CSV text and insert rows into database. Returns row count."""
    reader = csv.DictReader(io.StringIO(csv_text))
    rows_inserted = 0

    for row in reader:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO race_results (
                    race_date, race_time, track, race_name, race_restrictions_age,
                    race_class, major, race_distance, prize_money, going_description,
                    number_of_runners, place, distbt, horse_name, stall, trainer,
                    horse_age, jockey_name, jockeys_claim, pounds, odds, fav,
                    official_rating, comptime, comptime_numeric, total_dst_bt,
                    median_or, dist_furlongs, placing_numerical, race_code,
                    bfsp, bfsp_place, plcs_paid, bf_plcs_paid, yards, rail_move,
                    race_type, comment, card_no, stall_positioning, track_direction,
                    headgear, days_since_lr, career_runs, stallion, surface_type,
                    horse_prizewin, horse_sex, dam, dam_stallion, max_or_in_race
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                row.get("racedate", "").strip(),
                row.get("racetime", "").strip(),
                row.get("track", "").strip(),
                row.get("race_name", "").strip(),
                row.get("race_restrictions_age", "").strip(),
                row.get("race_class", "").strip(),
                row.get("major", "").strip(),
                row.get("race_distance", "").strip(),
                row.get("prize_money", "").strip(),
                row.get("going_description", "").strip(),
                safe_int(row.get("number_of_runners", "")),
                row.get("place", "").strip(),
                row.get("distbt", "").strip(),
                row.get("horse_name", "").strip(),
                safe_int(row.get("stall", "")),
                row.get("trainer", "").strip(),
                safe_int(row.get("horse_age", "")),
                row.get("jockey_name", "").strip(),
                safe_int(row.get("jockeys_claim", "")),
                safe_int(row.get("pounds", "")),
                safe_float(row.get("odds", "")),
                row.get("fav", "").strip(),
                safe_int(row.get("official_rating", "")),
                row.get("comptime", "").strip(),
                safe_float(row.get("comptime_numeric", "")),
                row.get("TotalDstBt", "").strip(),
                safe_int(row.get("MedianOR", "")),
                safe_float(row.get("Dist_Furlongs", "")),
                safe_int(row.get("placing_numerical", "")),
                row.get("RCode", "").strip(),
                safe_float(row.get("BFSP", "")),
                safe_float(row.get("BFSP_Place", "")),
                safe_int(row.get("PlcsPaid", "")),
                safe_int(row.get("BFPlcsPaid", "")),
                safe_int(row.get("Yards", "")),
                row.get("RailMove", "").strip(),
                row.get("RaceType", "").strip(),
                row.get("Comment", "").strip(),
                safe_int(row.get("CardNo", "")),
                row.get("StallPositioning", "").strip(),
                row.get("TrackDirection", "").strip(),
                row.get("Headgear", "").strip(),
                safe_int(row.get("dayssincelr", "")),
                safe_int(row.get("careerruns", "")),
                row.get("stallion", "").strip(),
                row.get("surfacetype", "").strip(),
                row.get("horse_prizewin", "").strip(),
                row.get("HorseSex", "").strip(),
                row.get("dam", "").strip(),
                row.get("damstallion", "").strip(),
                safe_int(row.get("MaxORinRace", "")),
            ))
            rows_inserted += 1
        except sqlite3.Error as e:
            log.warning(f"DB error for row: {e}")

    conn.execute("""
        INSERT OR REPLACE INTO scrape_log (scrape_date, rows_found, status)
        VALUES (?, ?, 'ok')
    """, (target_date.isoformat(), rows_inserted))
    conn.commit()

    return rows_inserted


def get_last_scraped_date(conn: sqlite3.Connection) -> date | None:
    """Get the most recent date that was successfully scraped."""
    row = conn.execute(
        "SELECT MAX(scrape_date) FROM scrape_log WHERE status='ok'"
    ).fetchone()
    if row and row[0]:
        return date.fromisoformat(row[0])
    return None


def create_session() -> requests.Session:
    """Create an HTTP session with appropriate headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session


def scrape_date_range(start_date: date, end_date: date, db_path: str = DB_PATH):
    """Scrape all dates in the given range (inclusive), going backwards from end_date."""
    conn = init_db(db_path)
    session = create_session()

    user_id = login(session)
    if not user_id:
        log.error("Failed to login. Exiting.")
        sys.exit(1)

    total_days = (end_date - start_date).days + 1
    current = end_date
    scraped_count = 0
    total_rows = 0

    log.info(f"Scraping {total_days} days: {start_date} to {end_date}")

    while current >= start_date:
        # Check if already scraped
        existing = conn.execute(
            "SELECT rows_found FROM scrape_log WHERE scrape_date = ? AND status = 'ok'",
            (current.isoformat(),)
        ).fetchone()

        if existing:
            log.debug(f"Skipping {current} (already scraped, {existing[0]} rows)")
            current -= timedelta(days=1)
            continue

        try:
            csv_text = download_csv(session, user_id, current)

            if csv_text:
                save_csv_file(csv_text, current)
                rows = parse_and_save(conn, csv_text, current)
                total_rows += rows
                log.info(f"{current}: {rows} rows saved")
            else:
                # No results for this date (no racing)
                conn.execute("""
                    INSERT OR REPLACE INTO scrape_log (scrape_date, rows_found, status)
                    VALUES (?, 0, 'ok')
                """, (current.isoformat(),))
                conn.commit()
                log.info(f"{current}: no racing")

            scraped_count += 1

            # Progress update every 50 days
            days_done = (end_date - current).days + 1
            if days_done % 50 == 0:
                log.info(f"Progress: {days_done}/{total_days} days, "
                         f"{total_rows} total rows")

        except requests.RequestException as e:
            log.error(f"Network error on {current}: {e}")
            conn.execute("""
                INSERT OR REPLACE INTO scrape_log (scrape_date, rows_found, status)
                VALUES (?, 0, 'error')
            """, (current.isoformat(),))
            conn.commit()
            # Re-login on network errors
            time.sleep(5)
            user_id_new = login(session)
            if user_id_new:
                user_id = user_id_new

        except Exception as e:
            log.error(f"Error scraping {current}: {e}")
            conn.execute("""
                INSERT OR REPLACE INTO scrape_log (scrape_date, rows_found, status)
                VALUES (?, 0, 'error')
            """, (current.isoformat(),))
            conn.commit()

        current -= timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    conn.close()
    log.info(f"Done. Scraped {scraped_count} new days, {total_rows} total rows.")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape horse racing results from horseracebase.com"
    )
    parser.add_argument("--from", dest="from_date", type=str,
                        help="Start date (YYYY-MM-DD). Default: last scraped or 2010-01-01")
    parser.add_argument("--to", dest="to_date", type=str,
                        help="End date (YYYY-MM-DD). Default: yesterday")
    parser.add_argument("--full", action="store_true",
                        help="Full scrape from 2010-01-01 to today")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help=f"Database path (default: {DB_PATH})")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip S3 backup after scraping")
    args = parser.parse_args()

    yesterday = date.today() - timedelta(days=1)

    if args.full:
        start = date(2010, 1, 1)
        end = yesterday
    elif args.from_date:
        start = date.fromisoformat(args.from_date)
        end = date.fromisoformat(args.to_date) if args.to_date else yesterday
    else:
        conn = init_db(args.db)
        last = get_last_scraped_date(conn)
        conn.close()
        if last:
            start = last + timedelta(days=1)
            log.info(f"Resuming from {start} (last scraped: {last})")
        else:
            start = date(2010, 1, 1)
        end = yesterday

    if start > end:
        log.info("Nothing to scrape - already up to date.")
        return

    scrape_date_range(start, end, args.db)

    if not args.no_backup:
        log.info("Backing up database to S3...")
        try:
            from backup_to_s3 import upload_db
            upload_db(versioned=False)
        except Exception as e:
            log.warning(f"S3 backup failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
