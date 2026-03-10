"""
Live progress monitor for horse_racing.db scraper.
Shows a progress bar per year, updates every 60 seconds.
"""

import os
import sqlite3
import sys
import time
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")

START_YEAR = 2010
END_YEAR = date.today().year

# Racing days per year (approximate: ~365 minus non-racing days, but scrape_log
# tracks every calendar day including non-racing days)
DAYS_PER_YEAR = {y: (date(y, 12, 31) - date(y, 1, 1)).days + 1 for y in range(START_YEAR, END_YEAR + 1)}
# Cap current year at today
DAYS_PER_YEAR[END_YEAR] = (date.today() - date(END_YEAR, 1, 1)).days


def get_year_stats(conn):
    """Get scrape progress and row counts per year."""
    rows = conn.execute("""
        SELECT
            CAST(SUBSTR(scrape_date, 1, 4) AS INTEGER) AS year,
            COUNT(*) AS days_scraped,
            SUM(rows_found) AS total_rows,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors
        FROM scrape_log
        GROUP BY year
        ORDER BY year DESC
    """).fetchall()
    return rows


def get_total_results(conn):
    """Get total rows in race_results."""
    return conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]


def bar(pct, width=30):
    """Render a progress bar."""
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def display(conn):
    clear_screen()
    year_stats = get_year_stats(conn)
    total_results = get_total_results(conn)

    stats_by_year = {r[0]: (r[1], r[2], r[3]) for r in year_stats}

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  HORSE RACING SCRAPER - LIVE PROGRESS   ({now})")
    print(f"  Total race result rows: {total_results:,}")
    print("─" * 68)
    print(f"  {'Year':<6} {'Progress':<36} {'Days':>6} {'Rows':>9} {'Err':>4}")
    print("─" * 68)

    total_days_scraped = 0
    total_days_target = 0
    total_rows_all = 0

    for year in range(END_YEAR, START_YEAR - 1, -1):
        target = DAYS_PER_YEAR.get(year, 365)
        if year in stats_by_year:
            days_done, rows, errors = stats_by_year[year]
            pct = min(days_done / target, 1.0) if target > 0 else 0
            err_str = str(errors) if errors else ""
            print(f"  {year:<6} {bar(pct)} {pct:>5.1%}  {days_done:>4}/{target:<4} {rows:>7,}  {err_str:>3}")
            total_days_scraped += days_done
            total_rows_all += rows
        else:
            print(f"  {year:<6} {bar(0)}  0.0%   0/{target:<4}       0     ")
        total_days_target += target

    print("─" * 68)
    overall_pct = total_days_scraped / total_days_target if total_days_target else 0
    print(f"  {'TOTAL':<6} {bar(overall_pct)} {overall_pct:>5.1%}  {total_days_scraped:>4}/{total_days_target:<4} {total_rows_all:>7,}")
    print()

    # Check if scraper process is still running
    pid_alive = False
    try:
        import subprocess
        result = subprocess.run(["pgrep", "-f", "scraper.py"], capture_output=True, text=True)
        pid_alive = result.returncode == 0
    except Exception:
        pass

    status = "RUNNING" if pid_alive else "STOPPED"
    print(f"  Scraper status: {status}")
    print(f"  Next refresh in 60s... (Ctrl+C to exit)")


def main():
    while True:
        if not os.path.exists(DB_PATH):
            print("Waiting for horse_racing.db to be created...")
            time.sleep(5)
            continue

        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            display(conn)
            conn.close()
        except sqlite3.OperationalError as e:
            print(f"DB locked, retrying... ({e})")

        time.sleep(60)


if __name__ == "__main__":
    main()
