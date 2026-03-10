"""
QA checker for horse_racing.db - runs every 10 minutes.
Validates data integrity, checks for anomalies, and logs results.
"""

import os
import sqlite3
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
QA_LOG = os.path.join(SCRIPT_DIR, "qa_check.log")


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [{level}] {msg}"
    print(line)
    with open(QA_LOG, "a") as f:
        f.write(line + "\n")


def run_checks(conn):
    log("=" * 60)
    log("QA CHECK STARTED")
    issues = 0

    # 1. Total row count
    total = conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]
    log(f"Total race_results rows: {total:,}")

    if total == 0:
        log("No data yet - skipping checks")
        return

    # 2. Check for NULL horse_name or track (NOT NULL columns)
    nulls = conn.execute(
        "SELECT COUNT(*) FROM race_results WHERE horse_name IS NULL OR horse_name = '' OR track IS NULL OR track = ''"
    ).fetchone()[0]
    if nulls > 0:
        log(f"ISSUE: {nulls} rows with empty horse_name or track", "WARN")
        issues += 1
    else:
        log("OK: No NULL/empty horse_name or track")

    # 3. Check for invalid race_date format
    bad_dates = conn.execute(
        "SELECT COUNT(*) FROM race_results WHERE race_date NOT LIKE '____-__-__'"
    ).fetchone()[0]
    if bad_dates > 0:
        log(f"ISSUE: {bad_dates} rows with malformed race_date", "WARN")
        issues += 1
    else:
        log("OK: All race_date values well-formed")

    # 4. Check for negative odds
    neg_odds = conn.execute(
        "SELECT COUNT(*) FROM race_results WHERE odds < 0"
    ).fetchone()[0]
    if neg_odds > 0:
        log(f"ISSUE: {neg_odds} rows with negative odds", "WARN")
        issues += 1
    else:
        log("OK: No negative odds")

    # 5. Check placing_numerical range
    bad_placing = conn.execute(
        "SELECT COUNT(*) FROM race_results WHERE placing_numerical IS NOT NULL AND (placing_numerical < 0 OR placing_numerical > 40)"
    ).fetchone()[0]
    if bad_placing > 0:
        log(f"ISSUE: {bad_placing} rows with placing_numerical out of range", "WARN")
        issues += 1
    else:
        log("OK: placing_numerical values in range")

    # 6. Check number_of_runners sanity
    bad_runners = conn.execute(
        "SELECT COUNT(*) FROM race_results WHERE number_of_runners IS NOT NULL AND (number_of_runners < 1 OR number_of_runners > 40)"
    ).fetchone()[0]
    if bad_runners > 0:
        log(f"ISSUE: {bad_runners} rows with number_of_runners out of range", "WARN")
        issues += 1
    else:
        log("OK: number_of_runners values in range")

    # 7. Date coverage - check for years with suspiciously few rows
    year_counts = conn.execute("""
        SELECT SUBSTR(race_date, 1, 4) AS yr, COUNT(*) AS cnt
        FROM race_results
        GROUP BY yr ORDER BY yr
    """).fetchall()
    for yr, cnt in year_counts:
        if cnt < 100:
            log(f"NOTE: Year {yr} has only {cnt} rows (may still be scraping)", "NOTE")

    # 8. Duplicate check (should be 0 due to UNIQUE constraint)
    dupes = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT race_date, race_time, track, horse_name, COUNT(*) as c
            FROM race_results
            GROUP BY race_date, race_time, track, horse_name
            HAVING c > 1
        )
    """).fetchone()[0]
    if dupes > 0:
        log(f"ISSUE: {dupes} duplicate entries found", "WARN")
        issues += 1
    else:
        log("OK: No duplicate entries")

    # 9. Scrape log error rate
    scrape_total = conn.execute("SELECT COUNT(*) FROM scrape_log").fetchone()[0]
    scrape_errors = conn.execute("SELECT COUNT(*) FROM scrape_log WHERE status = 'error'").fetchone()[0]
    if scrape_total > 0:
        error_rate = scrape_errors / scrape_total * 100
        log(f"Scrape log: {scrape_total} days attempted, {scrape_errors} errors ({error_rate:.1f}%)")
        if error_rate > 10:
            log(f"ISSUE: High scrape error rate ({error_rate:.1f}%)", "WARN")
            issues += 1

    # 10. Latest data check
    latest = conn.execute("SELECT MAX(race_date) FROM race_results").fetchone()[0]
    earliest = conn.execute("SELECT MIN(race_date) FROM race_results").fetchone()[0]
    log(f"Date range in DB: {earliest} to {latest}")

    # Summary
    if issues == 0:
        log("QA PASSED - No issues found")
    else:
        log(f"QA COMPLETE - {issues} issue(s) found", "WARN")
    log("=" * 60)


def main():
    while True:
        if not os.path.exists(DB_PATH):
            log("Waiting for horse_racing.db...")
            time.sleep(30)
            continue

        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            run_checks(conn)
            conn.close()
        except sqlite3.OperationalError as e:
            log(f"DB locked: {e}", "WARN")

        log("Next QA check in 10 minutes...")
        time.sleep(600)


if __name__ == "__main__":
    main()
