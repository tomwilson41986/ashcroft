#!/usr/bin/env python3
"""Print database info (row count, date range)."""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "horse_racing.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM race_results")
rows = cur.fetchone()[0]
cur.execute("SELECT MIN(race_date), MAX(race_date) FROM race_results")
d = cur.fetchone()
print(f"Rows: {rows:,}, Date range: {d[0]} to {d[1]}")
conn.close()
