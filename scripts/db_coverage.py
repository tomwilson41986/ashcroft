#!/usr/bin/env python3
"""Month-by-month coverage of every table in horse_racing.db, and every hole in it.

    python scripts/db_coverage.py horse_racing.db --out coverage

The database went six months without a night being saved and nobody was told,
then lost five more days to a scrape that asked before the results were up. A
row count and a date range say nothing about either: the range was right both
times. This says, for each month, how much is there against how much the same
month holds in other years, and lists every day with nothing in it next to what
the scrape log believes about that day -- which is what decides whether a
backfill will actually ask for it again.

Writes summary.md, months.csv (race_results by month), gaps.csv (runs of empty
days), low_days.csv (days far below their month) and tables.csv.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import date, timedelta

import pandas as pd

#: No racing in Britain or Ireland on these, in any year.
NO_RACING = {(12, 24), (12, 25)}

#: A month below this share of its usual size is flagged.
LOW_MONTH = 0.6

#: A day below this share of its month's median day is listed as partial.
LOW_DAY = 0.25

#: The date column to count a table by, in order of preference.
DATE_COLS = ("race_date", "event_dt", "event_date", "date", "scrape_date",
             "snapshot_at", "created_at", "loaded_at")


def _q(conn, sql, params=()):
    return pd.read_sql_query(sql, conn, params=params)


def tables(conn) -> pd.DataFrame:
    """Every table: rows, the date column it is counted by, and its range."""
    rows = []
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for t in names:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
        n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        dcol = next((c for c in DATE_COLS if c in cols), None)
        lo = hi = None
        if dcol and n:
            lo, hi = conn.execute(f'SELECT MIN("{dcol}"), MAX("{dcol}") FROM "{t}"').fetchone()
        rows.append({"table": t, "rows": n, "date_col": dcol, "min": lo, "max": hi,
                     "has_race_results_id": "race_results_id" in cols})
    return pd.DataFrame(rows)


def results_by_month(conn) -> pd.DataFrame:
    """race_results per month, with coverage of the columns training depends on."""
    m = _q(conn, """
        SELECT substr(race_date, 1, 7)                           AS month,
               COUNT(*)                                          AS runners,
               COUNT(DISTINCT race_date || '|' || track || '|' || COALESCE(race_time, '')) AS races,
               COUNT(DISTINCT substr(race_date, 1, 10))          AS racing_days,
               COUNT(DISTINCT track)                             AS tracks,
               AVG(CASE WHEN bfsp > 1 THEN 1.0 ELSE 0 END)       AS bsp_cov,
               AVG(CASE WHEN odds > 1 THEN 1.0 ELSE 0 END)       AS sp_cov,
               AVG(CASE WHEN placing_numerical IS NOT NULL THEN 1.0 ELSE 0 END) AS result_cov,
               AVG(CASE WHEN comptime_numeric > 0 THEN 1.0 ELSE 0 END)          AS time_cov,
               AVG(CASE WHEN official_rating > 0 THEN 1.0 ELSE 0 END)           AS or_cov
        FROM race_results
        GROUP BY month ORDER BY month
    """)
    m["year"] = m["month"].str.slice(0, 4)
    m["cal_month"] = m["month"].str.slice(5, 7)
    # What this calendar month usually holds, from the other years. The current,
    # unfinished month and the first, possibly partial, one are left out of it.
    full = m.iloc[1:-1] if len(m) > 2 else m.iloc[0:0]
    usual = {}
    for cm, g in full.groupby("cal_month"):
        for mo in m.loc[m["cal_month"] == cm, "month"]:
            others = g.loc[g["month"] != mo, "runners"]
            usual[mo] = float(others.median()) if len(others) else float("nan")
    m["usual_runners"] = m["month"].map(usual)
    m["share_of_usual"] = (m["runners"] / m["usual_runners"]).round(2)
    m["flag"] = ""
    m.loc[m["share_of_usual"] < LOW_MONTH, "flag"] = "LOW"
    for c in ("bsp_cov", "sp_cov", "result_cov", "time_cov", "or_cov"):
        m[c] = (100 * m[c]).round(1)
    return m.drop(columns=["year", "cal_month"])


def _days_between(lo: date, hi: date):
    d = lo
    while d <= hi:
        yield d
        d += timedelta(days=1)


def gaps(conn, today: date | None = None) -> pd.DataFrame:
    """Runs of consecutive days with no race_results rows, and the scrape log's view."""
    have = {r[0] for r in conn.execute(
        "SELECT DISTINCT substr(race_date, 1, 10) FROM race_results")}
    have = {d for d in have if d and len(d) == 10}
    if not have:
        return pd.DataFrame()
    lo = date.fromisoformat(min(have))
    hi = today or date.today()
    log = {}
    try:
        log = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT scrape_date, rows_found, status FROM scrape_log")}
    except sqlite3.Error:
        pass
    runs, cur = [], []
    for d in _days_between(lo, hi):
        iso = d.isoformat()
        if iso in have or (d.month, d.day) in NO_RACING:
            if cur:
                runs.append(cur)
                cur = []
            continue
        cur.append(iso)
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        states = pd.Series([_log_state(log.get(d)) for d in run]).value_counts()
        out.append({"from": run[0], "to": run[-1], "days": len(run),
                    "scrape_log": ", ".join(f"{k} {v}" for k, v in states.items())})
    return pd.DataFrame(out)


def _log_state(entry) -> str:
    if entry is None:
        return "never scraped"
    rows, status = entry
    if status == "ok" and not rows:
        return "logged ok, 0 rows"
    return f"logged {status}"


def low_days(conn) -> pd.DataFrame:
    """Days with rows, but far fewer than their month's median day: a partial download."""
    d = _q(conn, """
        SELECT substr(race_date, 1, 10) AS day, COUNT(*) AS runners,
               COUNT(DISTINCT track || '|' || COALESCE(race_time, '')) AS races
        FROM race_results GROUP BY day ORDER BY day
    """)
    if d.empty:
        return d
    d["month"] = d["day"].str.slice(0, 7)
    d["month_median_day"] = d.groupby("month")["runners"].transform("median")
    d["share"] = (d["runners"] / d["month_median_day"]).round(2)
    return d[d["share"] < LOW_DAY].drop(columns=["month"])


def log_vs_results(conn) -> pd.DataFrame:
    """Days where the scrape log's count and the table disagree by more than 10%."""
    try:
        d = _q(conn, """
            SELECT s.scrape_date AS day, s.rows_found AS logged, s.status,
                   COALESCE(r.n, 0) AS in_table
            FROM scrape_log s
            LEFT JOIN (SELECT substr(race_date, 1, 10) AS day, COUNT(*) AS n
                       FROM race_results GROUP BY day) r ON r.day = s.scrape_date
        """)
    except Exception:
        return pd.DataFrame()
    bad = d[(d["logged"] > 0) & ((d["in_table"] - d["logged"]).abs() > 0.1 * d["logged"])]
    return bad


def side_table_by_month(conn, table: str, info: pd.Series) -> pd.DataFrame | None:
    """Rows per month for a side table, and how many of them link to a result."""
    # A missing column comes out of the DataFrame as NaN, not None -- and SQLite
    # reads an unknown double-quoted name as a string literal, so "nan" would
    # count every row into a month called 'nan' without complaint.
    dcol = info["date_col"] if isinstance(info["date_col"], str) else None
    link = bool(info["has_race_results_id"])
    if dcol is None and link:
        sql = f"""SELECT substr(r.race_date, 1, 7) AS month, COUNT(*) AS rows,
                         SUM(CASE WHEN t.race_results_id IS NOT NULL THEN 1 ELSE 0 END) AS linked
                  FROM "{table}" t JOIN race_results r ON r.id = t.race_results_id
                  GROUP BY month ORDER BY month"""
    elif dcol is not None:
        linked = ("SUM(CASE WHEN race_results_id IS NOT NULL THEN 1 ELSE 0 END)" if link else "NULL")
        sql = f"""SELECT substr("{dcol}", 1, 7) AS month, COUNT(*) AS rows, {linked} AS linked
                  FROM "{table}" GROUP BY month ORDER BY month"""
    else:
        return None
    try:
        return _q(conn, sql)
    except Exception as e:  # a table this script has never seen: say so, carry on
        return pd.DataFrame({"month": [f"error: {e}"]})


def _md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_none_"
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db")
    ap.add_argument("--out", default="coverage")
    ap.add_argument("--today", help="YYYY-MM-DD; default today")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    today = date.fromisoformat(a.today) if a.today else date.today()
    conn = sqlite3.connect(a.db)

    t = tables(conn)
    t.to_csv(os.path.join(a.out, "tables.csv"), index=False)
    md = ["# Database coverage", "", f"Audited {today.isoformat()}.", "", "## Tables", "", _md(t), ""]

    if "race_results" in set(t["table"]):
        bad_fmt = conn.execute(
            "SELECT COUNT(*) FROM race_results WHERE race_date NOT GLOB "
            "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*'").fetchone()[0]
        m = results_by_month(conn)
        g = gaps(conn, today)
        lo = low_days(conn)
        mism = log_vs_results(conn)
        m.to_csv(os.path.join(a.out, "months.csv"), index=False)
        g.to_csv(os.path.join(a.out, "gaps.csv"), index=False)
        lo.to_csv(os.path.join(a.out, "low_days.csv"), index=False)
        mism.to_csv(os.path.join(a.out, "log_mismatch.csv"), index=False)
        long_gaps = g[g["days"] >= 2] if not g.empty else g
        md += [
            "## race_results",
            "",
            f"Rows whose race_date is not ISO-formatted: {bad_fmt}.",
            "",
            f"### Empty stretches ({len(g)} runs, {int(g['days'].sum()) if not g.empty else 0} days; "
            "24-25 Dec excluded)",
            "",
            "A backfill asks again for a day that was never scraped or logged `error`/`pending`. "
            "A day logged `ok` with 0 rows is skipped unless it is recent -- it needs a forced recheck.",
            "",
            "Runs of two days or more:",
            "",
            _md(long_gaps),
            "",
            "Single days: " + (", ".join(g.loc[g["days"] == 1, "from"]) if not g.empty else "none"),
            "",
            f"### Months below {int(100 * LOW_MONTH)}% of the same month in other years",
            "",
            _md(m[m["flag"] != ""][["month", "runners", "usual_runners", "share_of_usual", "races",
                                   "racing_days", "bsp_cov"]]),
            "",
            f"### Partial days (under {int(100 * LOW_DAY)}% of their month's median day): {len(lo)}",
            "",
            _md(lo.head(60)),
            "",
            f"### Days the scrape log counts differently from the table: {len(mism)}",
            "",
            _md(mism.head(40)),
            "",
            "### Every month",
            "",
            "Coverage columns are the % of runners with a BSP > 1, an SP > 1, a result, a "
            "finishing time and an official rating.",
            "",
            _md(m),
            "",
        ]
        try:
            st = _q(conn, "SELECT substr(scrape_date, 1, 4) AS year, status, COUNT(*) AS days, "
                          "SUM(rows_found = 0) AS zero_row_days FROM scrape_log "
                          "GROUP BY year, status ORDER BY year, status")
            md += ["## scrape_log by year and status", "", _md(st), ""]
        except Exception:
            pass

    for _, info in t.iterrows():
        if info["table"] in ("race_results", "scrape_log", "sqlite_sequence"):
            continue
        s = side_table_by_month(conn, info["table"], info)
        if s is None:
            continue
        s.to_csv(os.path.join(a.out, f"{info['table']}_months.csv"), index=False)
        md += [f"## {info['table']} by month", "", _md(s), ""]

    conn.close()
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
