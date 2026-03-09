"""Show scraping progress by year with progress bars."""
import sqlite3
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "horse_racing.db")

def show_progress():
    if not os.path.exists(DB_PATH):
        print("Database not found. Scraper hasn't started yet.")
        return

    conn = sqlite3.connect(DB_PATH)

    # Get scrape log stats by year
    years = list(range(2010, 2027))
    start_date = date(2010, 1, 1)
    end_date = date(2026, 3, 8)

    total_scraped = 0
    total_expected = 0
    total_rows = 0

    print()
    print("  Horse Racing Results Scraper - Progress by Year")
    print("  " + "=" * 60)
    print()

    for year in years:
        year_start = max(date(year, 1, 1), start_date)
        year_end = min(date(year, 12, 31), end_date)

        if year_start > end_date:
            continue

        expected_days = (year_end - year_start).days + 1

        # Count scraped days for this year
        row = conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(rows_found), 0)
            FROM scrape_log
            WHERE scrape_date >= ? AND scrape_date <= ? AND status = 'ok'
        """, (year_start.isoformat(), year_end.isoformat())).fetchone()

        scraped_days = row[0]
        rows_found = row[1]

        total_scraped += scraped_days
        total_expected += expected_days
        total_rows += rows_found

        pct = (scraped_days / expected_days * 100) if expected_days > 0 else 0
        bar_width = 35
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        status = "✓" if pct >= 100 else " "

        print(f"  {status} {year}  {bar}  {pct:5.1f}%  ({scraped_days:3d}/{expected_days:3d} days, {rows_found:,} rows)")

    print()
    print("  " + "-" * 60)

    overall_pct = (total_scraped / total_expected * 100) if total_expected > 0 else 0
    bar_width = 35
    filled = int(bar_width * overall_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"  TOTAL  {bar}  {overall_pct:5.1f}%  ({total_scraped:,}/{total_expected:,} days, {total_rows:,} rows)")

    # Show current activity
    last_row = conn.execute(
        "SELECT scrape_date FROM scrape_log ORDER BY scraped_at DESC LIMIT 1"
    ).fetchone()
    if last_row:
        print(f"\n  Currently at: {last_row[0]}")

    # Errors
    errors = conn.execute(
        "SELECT COUNT(*) FROM scrape_log WHERE status = 'error'"
    ).fetchone()[0]
    if errors:
        print(f"  Errors: {errors} dates failed (will retry on next run)")

    print()
    conn.close()


if __name__ == "__main__":
    show_progress()
