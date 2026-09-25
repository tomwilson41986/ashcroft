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
import re
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
LOGIN_URL = f"{BASE_URL}/horseracebase_login.php"
CSV_URL = f"{BASE_URL}/excelresults.php"

# Rate limiting: seconds between requests
REQUEST_DELAY = 1.5

#: An empty download for a date this recent means horseracebase has not
#: published it yet, not that there was no racing. The nightly job asks for the
#: day itself at 21:30 UTC, before its results are up; recording that empty
#: answer as done moved the resume point past the day for good, and the per-day
#: skip then refused to ask again even when told to. Every day from 18 to 22
#: September 2026 was lost that way while the job reported success. Past this
#: age the nightly look-back stops asking: a results file with no runners in it
#: records no racing (Christmas Day, say); no file at all is logged 'error', for
#: a backfill to ask again.
RETRY_EMPTY_DAYS = 7

#: After this many days in a row without a results file, stop asking. The site
#: refuses in bulk when an account has downloaded too much -- on 17 Sep 2026 a
#: 179-day run got nothing for every day of a range that had returned 62,973
#: rows the night before -- and asking on only spends more of the allowance.
MAX_UNAVAILABLE_IN_A_ROW = 5


def is_recent(d: date, today: date | None = None) -> bool:
    """Too recent for an empty download to be believed."""
    return ((today or date.today()) - d).days < RETRY_EMPTY_DAYS

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

    # Get CSRF token from the login page
    resp = session.get(LOGIN_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    csrf_input = (soup.find("input", {"name": "csrf_token"})
                  or soup.find("input", {"name": "CSRFtoken"}))
    if not csrf_input:
        log.error("Could not find CSRF token on page")
        return None

    # Post login back to the login page
    resp2 = session.post(LOGIN_URL, data={
        "login": username,
        "password": password,
        csrf_input.get("name"): csrf_input.get("value"),
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
        # Not a results file: unpublished, refused (horseracebase limits
        # downloads per account), or logged out. None of those is "no racing",
        # so the caller must not record it as done. What came back is logged
        # because it is the only evidence of which it was.
        # the page's words, not its markup: the first 160 characters of the HTML
        # were only the styling of the box the message sits in
        words = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", resp.text, flags=re.S | re.I)
        snippet = " ".join(re.sub(r"<[^>]+>", " ", words).split())[:400]
        log.info(f"{target_date}: no results file ({content_type or 'no content type'}): {snippet!r}")
        return None

    return resp.text


def save_csv_file(csv_text: str, target_date: date):
    """Save raw CSV to csv/ directory."""
    os.makedirs(CSV_DIR, exist_ok=True)
    filepath = os.path.join(CSV_DIR, f"results_{target_date.isoformat()}.csv")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(csv_text)


#: The columns parse_and_save writes, in the order it writes them.
RESULT_COLS = (
    "race_date",
    "race_time",
    "track",
    "race_name",
    "race_restrictions_age",
    "race_class",
    "major",
    "race_distance",
    "prize_money",
    "going_description",
    "number_of_runners",
    "place",
    "distbt",
    "horse_name",
    "stall",
    "trainer",
    "horse_age",
    "jockey_name",
    "jockeys_claim",
    "pounds",
    "odds",
    "fav",
    "official_rating",
    "comptime",
    "comptime_numeric",
    "total_dst_bt",
    "median_or",
    "dist_furlongs",
    "placing_numerical",
    "race_code",
    "bfsp",
    "bfsp_place",
    "plcs_paid",
    "bf_plcs_paid",
    "yards",
    "rail_move",
    "race_type",
    "comment",
    "card_no",
    "stall_positioning",
    "track_direction",
    "headgear",
    "days_since_lr",
    "career_runs",
    "stallion",
    "surface_type",
    "horse_prizewin",
    "horse_sex",
    "dam",
    "dam_stallion",
    "max_or_in_race",
)
_RESULT_KEY = ("race_date", "race_time", "track", "horse_name")

#: Asking for a day again fills in what the first download lacked -- a BSP not
#: out yet, a result amended after an enquiry -- and never blanks what is
#: stored: an empty field in the new download keeps the old value. (It was
#: INSERT OR IGNORE, which made a second download of a day a no-op, so a day
#: saved before its BSPs were published stayed without them.)
RESULT_UPSERT = (
    f"INSERT INTO race_results ({', '.join(RESULT_COLS)}) "
    f"VALUES ({', '.join('?' for _ in RESULT_COLS)}) "
    f"ON CONFLICT({', '.join(_RESULT_KEY)}) DO UPDATE SET "
    + ", ".join(f"{c} = COALESCE(NULLIF(excluded.{c}, ''), race_results.{c})"
                for c in RESULT_COLS if c not in _RESULT_KEY)
)


def parse_and_save(conn: sqlite3.Connection, csv_text: str,
                   target_date: date) -> int:
    """Parse CSV text and insert rows into database. Returns row count."""
    reader = csv.DictReader(io.StringIO(csv_text))
    rows_inserted = 0

    for row in reader:
        try:
            conn.execute(RESULT_UPSERT, (
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
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return session


def scrape_date_range(start_date: date, end_date: date, db_path: str = DB_PATH,
                      recheck: bool = False):
    """Scrape all dates in the given range (inclusive), going backwards from end_date.

    ``recheck`` asks again for every day in the range, including days already
    logged done: the way to repair an old day logged empty, or a day saved
    before its BSPs were out (the upsert fills them in).
    """
    conn = init_db(db_path)
    session = create_session()

    user_id = login(session)
    if not user_id:
        log.error("Failed to login. Exiting.")
        sys.exit(1)

    total_days = (end_date - start_date).days + 1
    current = end_date
    scraped_count = 0
    pending_count = 0
    unavailable_in_a_row = 0
    total_rows = 0

    log.info(f"Scraping {total_days} days: {start_date} to {end_date}")

    while current >= start_date:
        # Check if already scraped
        existing = conn.execute(
            "SELECT rows_found FROM scrape_log WHERE scrape_date = ? AND status = 'ok'",
            (current.isoformat(),)
        ).fetchone()

        # A recent day logged as done with nothing in it was, in all
        # likelihood, asked for before it was published -- so ask again.
        if existing and not recheck and not (existing[0] == 0 and is_recent(current)):
            log.debug(f"Skipping {current} (already scraped, {existing[0]} rows)")
            current -= timedelta(days=1)
            continue

        try:
            csv_text = download_csv(session, user_id, current)

            unavailable_in_a_row = 0 if csv_text else unavailable_in_a_row + 1
            if csv_text:
                # A real results file. With no runners in it, that is the
                # site saying there was no racing -- the only way a day is
                # recorded as done and empty.
                save_csv_file(csv_text, current)
                rows = parse_and_save(conn, csv_text, current)
                total_rows += rows
                log.info(f"{current}: {rows} rows saved" if rows else f"{current}: no racing")
            elif (held := conn.execute(
                    # A range, not substr(), so the race_date index is used.
                    "SELECT COUNT(*) FROM race_results WHERE race_date >= ? AND race_date < ?",
                    (current.isoformat(), (current + timedelta(days=1)).isoformat())
                    ).fetchone()[0]):
                # Only reachable by asking again for a day already in: an
                # empty answer now says nothing about the rows we hold.
                log.warning(f"{current}: empty download, but {held} rows are "
                            f"already held -- keeping them and the log as they are")
            elif is_recent(current):
                # Not published yet. 'pending' is not 'ok', so it neither
                # advances the resume point nor satisfies the skip above, and
                # the next run asks for it again.
                conn.execute("""
                    INSERT OR REPLACE INTO scrape_log (scrape_date, rows_found, status)
                    VALUES (?, 0, 'pending')
                """, (current.isoformat(),))
                conn.commit()
                pending_count += 1
                log.info(f"{current}: no results yet -- will retry on the next run")
            else:
                # No file for an old day is not "no racing" either: it was
                # refused. 'error' keeps it out of the resume point and makes
                # the next backfill over it ask again.
                conn.execute("""
                    INSERT OR REPLACE INTO scrape_log (scrape_date, rows_found, status)
                    VALUES (?, 0, 'error')
                """, (current.isoformat(),))
                conn.commit()
                log.warning(f"{current}: no results file -- logged 'error' to retry, not 'no racing'")

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
        if unavailable_in_a_row >= MAX_UNAVAILABLE_IN_A_ROW:
            left = (current - start_date).days + 1
            log.error(f"No results file for {unavailable_in_a_row} days in a row -- "
                      f"horseracebase is refusing (download limit?) or down. Stopping with "
                      f"{left} days not asked; nothing was recorded as done for them.")
            print(f"::warning::horseracebase refused {unavailable_in_a_row} days in a row; "
                  f"stopped with {left} days not asked")
            break
        time.sleep(REQUEST_DELAY)

    conn.close()
    log.info(f"Done. Scraped {scraped_count} new days, {total_rows} total rows"
             + (f"; {pending_count} not published yet, to retry." if pending_count else "."))


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
    parser.add_argument("--recheck", action="store_true",
                        help="Ask again for every day in the range, even days logged "
                             "done; fills in missing values without blanking any")
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
            # Never resume later than the retry window. A day asked for before
            # it was published is 'pending', and a later day coming in does
            # not mean it has; the loop skips days already in without a
            # request, so this costs one request per day still missing. The
            # window's oldest day is no longer recent, so a pending day gets
            # its final answer there -- an empty one is then no racing.
            start = min(last + timedelta(days=1),
                        date.today() - timedelta(days=RETRY_EMPTY_DAYS))
            log.info(f"Resuming from {start} (last scraped: {last}; the last "
                     f"{RETRY_EMPTY_DAYS} days are rechecked for late results)")
        else:
            start = date(2010, 1, 1)
        end = date.fromisoformat(args.to_date) if args.to_date else yesterday

    if start > end:
        log.info("Nothing to scrape - already up to date.")
        return

    scrape_date_range(start, end, args.db, recheck=args.recheck)

    if not args.no_backup:
        log.info("Backing up database to S3...")
        try:
            from backup_to_s3 import upload_db
            upload_db(versioned=False)
        except Exception as e:
            log.warning(f"S3 backup failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
