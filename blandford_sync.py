"""
Blandford Bloodstock backend (Timeform-derived results feed) -> SQLite.

The Blandford app's backend exposes ``/api/APIData_Table2`` — one row per
runner per race with Timeform-style content that horseracebase does not
carry:

    preRaceMasterRating, preRaceAdjustedRating   ability ratings known BEFORE the race
    performanceRating, timefigure                 the run's performance rating / timefigure
    finishingTime, leaderSectional, winnerSectional, distanceSectional
                                                  race time and last-N-furlong sectionals
    betfairWinSP, betfairPlaceSP, ipMin, ipMax, bSPAdvantage
    horseCode (stable id), sire / dam / damsire, foalingDate, headGear, ...

Coverage: essentially every UK/IRE (plus French) race from at least
mid-2022; pre-race ratings on ~80-90 % of runners, sectionals on ~55-70 %.

Usage:
    python blandford_sync.py --fetch --from 2024-01-01 --to 2024-12-31 --load --match
    python blandford_sync.py --fetch --days 3 --load --match          # nightly
    python blandford_sync.py --report

The endpoint is unauthenticated today; if that changes, set
BLANDFORD_USERNAME / BLANDFORD_PASSWORD in .env.local and the script logs in
via /api/login and sends the returned token as a Bearer header.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from betfair_prices import db_time_to_24h, normalise_horse, normalise_track

log = logging.getLogger(__name__)

BASE_URL = "https://horseracesbackend-production.up.railway.app"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = str(SCRIPT_DIR / "horse_racing.db")
RAW_DIR = SCRIPT_DIR / "data" / "blandford_raw"
UK_IRE = {"GBR", "IRE", "GB", "IRL", "UK"}

# API field -> table column
FIELD_MAP = {
    "meetingDate": "meeting_date", "courseName": "course_bf", "raceNumber": "race_number", "raceTitle": "race_title",
    "distance": "distance_f", "raceClass": "race_class", "raceType": "race_type", "raceSurfaceName": "surface",
    "going": "going", "numberOfRunners": "n_runners", "horseName": "horse_name", "horseCode": "horse_code",
    "countryCode": "country", "positionOfficial": "position", "distanceBeaten": "distance_beaten",
    "distanceCumulative": "distance_cumulative", "finishingTime": "finishing_time",
    "sectionalFinishingTime": "sectional_finishing_time", "leaderSectional": "leader_sectional",
    "winnerSectional": "winner_sectional", "distanceSectional": "distance_sectional", "timefigure": "timefigure",
    "performanceRating": "performance_rating", "preRaceAdjustedRating": "pre_race_adjusted_rating",
    "preRaceMasterRating": "pre_race_master_rating", "ispDecimal": "isp_decimal", "betfairWinSP": "betfair_win_sp",
    "betfairPlaceSP": "betfair_place_sp", "ipMin": "ip_min", "ipMax": "ip_max", "bSPAdvantage": "bsp_advantage",
    "xWR": "x_wr", "xWM": "x_wm", "Percent_RB2": "percent_rb2", "draw": "draw", "horseAge": "age",
    "horseGender": "gender", "jockeyFullName": "jockey", "trainerFullName": "trainer", "ownerFullName": "owner",
    "sireName": "sire", "damName": "dam", "damsireName": "damsire", "foalingDate": "foaling_date",
    "officialRating": "official_rating", "racingPostOR": "rp_or", "racingPostSpeedFig": "rp_speed_fig",
    "racingPostRPR": "rp_rpr", "headGear": "headgear", "weightKg": "weight_kg", "prizeMoneyWon": "prize_won",
    "hotRace": "hot_race", "Win": "win_flag",
}
NUMERIC = {"distance_f", "race_class", "n_runners", "position", "distance_beaten", "distance_cumulative", "finishing_time",
           "sectional_finishing_time", "leader_sectional", "winner_sectional", "distance_sectional", "timefigure",
           "performance_rating", "pre_race_adjusted_rating", "pre_race_master_rating", "isp_decimal", "betfair_win_sp",
           "betfair_place_sp", "ip_min", "ip_max", "bsp_advantage", "x_wr", "x_wm", "percent_rb2", "draw", "age",
           "official_rating", "rp_or", "rp_speed_fig", "rp_rpr", "weight_kg", "prize_won", "hot_race", "win_flag", "horse_code"}
COLUMNS = list(FIELD_MAP.values()) + ["race_time", "horse_norm", "source_file"]


def _session(token: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ashcroft/blandford_sync",
                      "Origin": "https://www.blandfordbloodstock.tech", "Referer": "https://www.blandfordbloodstock.tech/"})
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def login(session: requests.Session) -> str | None:
    """Optional login (only needed if the API stops being open). Token stays in memory."""
    user, pw = os.getenv("BLANDFORD_USERNAME"), os.getenv("BLANDFORD_PASSWORD")
    if not user or not pw:
        return None
    r = session.post(f"{BASE_URL}/api/login", json={"email": user, "password": pw}, timeout=60)
    if r.status_code != 200:
        log.warning("Blandford login failed: HTTP %s", r.status_code)
        return None
    j = r.json()
    tok = j.get("token") or j.get("accessToken") or (j.get("user") or {}).get("token")
    if tok:
        session.headers["Authorization"] = f"Bearer {tok}"
    return tok


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_range(d_from: date, d_to: date, session: requests.Session | None = None, dest: Path = RAW_DIR,
                chunk_days: int = 7, skip_existing: bool = True, sleep: float = 0.5) -> list[Path]:
    """Download the results table in weekly chunks to data/blandford_raw/*.json."""
    s = session or _session()
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    d = d_from
    while d <= d_to:
        e = min(d + timedelta(days=chunk_days - 1), d_to)
        path = dest / f"apidata_{d.isoformat()}_{e.isoformat()}.json"
        if skip_existing and path.exists() and path.stat().st_size > 50 and e < date.today() - timedelta(days=1):
            paths.append(path); d = e + timedelta(days=1); continue
        url = f"{BASE_URL}/api/APIData_Table2?startDate={d.isoformat()}&endDate={e.isoformat()}"
        for attempt in range(3):
            try:
                r = s.get(url, timeout=300)
                if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                    rows = r.json().get("data", [])
                    path.write_text(json.dumps(rows))
                    log.info("fetched %s..%s: %d rows", d, e, len(rows))
                    paths.append(path)
                    break
                log.warning("%s: HTTP %s (attempt %d)", url, r.status_code, attempt + 1)
            except requests.RequestException as exc:
                log.warning("%s: %s (attempt %d)", url, exc, attempt + 1)
            time.sleep(2 * (attempt + 1))
        time.sleep(sleep)
        d = e + timedelta(days=1)
    return paths


# ---------------------------------------------------------------------------
# Parse + load
# ---------------------------------------------------------------------------

def parse_rows(rows: list[dict], source_file: str = "") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.DataFrame({col: df[api] if api in df.columns else None for api, col in FIELD_MAP.items()})
    out["meeting_date"] = pd.to_datetime(out["meeting_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    t = pd.to_datetime(df["scheduledTimeOfRaceLocal"], errors="coerce", utc=True) if "scheduledTimeOfRaceLocal" in df.columns else pd.Series(pd.NaT, index=df.index)
    out["race_time"] = t.dt.strftime("%H:%M")  # the feed's 'Z' is nominal: values are local race times
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["horse_norm"] = out["horse_name"].map(normalise_horse)
    out["source_file"] = source_file
    return out[COLUMNS]


def ensure_table(conn: sqlite3.Connection) -> None:
    cols = ", ".join(f"{c} REAL" if c in NUMERIC and c != "horse_code" else f"{c} TEXT" for c in COLUMNS if c != "horse_code")
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS blandford_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            horse_code INTEGER,
            {cols},
            race_results_id INTEGER,
            loaded_at TEXT DEFAULT (datetime('now')),
            UNIQUE(meeting_date, course_bf, race_number, horse_code)
        );
        CREATE INDEX IF NOT EXISTS idx_blf_date ON blandford_results(meeting_date);
        CREATE INDEX IF NOT EXISTS idx_blf_rr ON blandford_results(race_results_id);
        CREATE INDEX IF NOT EXISTS idx_blf_horse ON blandford_results(horse_norm);
    """)
    ensure_point_in_time_tables(conn)
    conn.commit()


# ---------------------------------------------------------------------------
# Point in time: what the feed said when we first saw it, and every change after
# ---------------------------------------------------------------------------

#: The rating fields a model reads. blandford_results keeps only their latest value (each
#: re-fetch replaces the row), so a backtest sees every rating as it stands today, after any
#: revision -- a look-ahead no live run would have. These two tables keep what live runs see.
TRACKED_FIELDS = ("performance_rating", "timefigure", "pre_race_master_rating", "pre_race_adjusted_rating",
                  "position", "distance_beaten")
_KEY = ("meeting_date", "course_bf", "race_number", "horse_code")


def ensure_point_in_time_tables(conn: sqlite3.Connection) -> None:
    fields = ", ".join(f"{f} REAL" for f in TRACKED_FIELDS)
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS blandford_first_seen (
            meeting_date TEXT, course_bf TEXT, race_number TEXT, horse_code INTEGER,
            {fields},
            first_seen_at TEXT DEFAULT (datetime('now')),
            UNIQUE(meeting_date, course_bf, race_number, horse_code)
        );
        CREATE TABLE IF NOT EXISTS blandford_revisions (
            meeting_date TEXT, course_bf TEXT, race_number TEXT, horse_code INTEGER,
            field TEXT, old_value REAL, new_value REAL,
            seen_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_blf_first_date ON blandford_first_seen(meeting_date);
        CREATE INDEX IF NOT EXISTS idx_blf_rev_date ON blandford_revisions(meeting_date);
    """)


def record_point_in_time(conn: sqlite3.Connection, df: pd.DataFrame) -> dict:
    """Before `df` replaces rows in blandford_results: log every tracked field whose value it
    changes (a blank filled in, a figure revised, or one withdrawn), and keep the first value
    ever seen of each runner. Never overwrites blandford_first_seen."""
    cols = list(_KEY) + list(TRACKED_FIELDS)
    inc = df[cols].astype(object).where(df[cols].notna(), None)
    conn.execute("DROP TABLE IF EXISTS _blf_incoming")
    # typed like blandford_results, so a race number arriving as 8 still joins the stored '8'
    types = {"meeting_date": "TEXT", "course_bf": "TEXT", "race_number": "TEXT", "horse_code": "INTEGER"}
    decl = ", ".join(c + " " + types.get(c, "REAL") for c in cols)
    conn.execute(f"CREATE TEMP TABLE _blf_incoming ({decl})")
    conn.executemany(f"INSERT INTO _blf_incoming VALUES ({', '.join('?' * len(cols))})", inc.values.tolist())
    on = " AND ".join(f"r.{k} = i.{k}" for k in _KEY)
    n_rev = 0
    for f in TRACKED_FIELDS:
        cur = conn.execute(f"""
            INSERT INTO blandford_revisions (meeting_date, course_bf, race_number, horse_code, field, old_value, new_value)
            SELECT i.meeting_date, i.course_bf, i.race_number, i.horse_code, '{f}', r.{f}, i.{f}
            FROM _blf_incoming i JOIN blandford_results r ON {on}
            WHERE r.{f} IS NOT i.{f}""")
        n_rev += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    cur = conn.execute(f"""INSERT OR IGNORE INTO blandford_first_seen ({', '.join(cols)})
                           SELECT {', '.join(cols)} FROM _blf_incoming""")
    n_new = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    conn.execute("DROP TABLE IF EXISTS _blf_incoming")
    return {"revised_fields": n_rev, "first_seen": n_new}


def load_files(paths, db_path: str = DEFAULT_DB, uk_ire_only: bool = True) -> dict:
    conn = sqlite3.connect(db_path)
    ensure_table(conn)
    n_rows = n_files = n_first = n_rev = 0
    cols = ["horse_code"] + [c for c in COLUMNS if c != "horse_code"]
    for p in paths:
        try:
            df = parse_rows(json.loads(Path(p).read_text()), Path(p).name)
        except Exception as exc:
            log.warning("skip %s: %s", p, exc)
            continue
        if uk_ire_only:
            df = df[df["country"].isin(UK_IRE) | df["country"].isna()]
        df = df.dropna(subset=["meeting_date", "horse_code"])
        df = df.drop_duplicates(list(_KEY), keep="last")
        pit = record_point_in_time(conn, df)
        n_rev += pit["revised_fields"]; n_first += pit["first_seen"]
        vals = df[cols].astype(object).where(df[cols].notna(), None).values.tolist()
        conn.executemany(f"INSERT OR REPLACE INTO blandford_results ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
        n_rows += len(vals); n_files += 1
    conn.commit(); conn.close()
    log.info("loaded %d rows from %d files (%d first seen, %d rating changes logged)", n_rows, n_files, n_first, n_rev)
    return {"files": n_files, "rows": n_rows, "first_seen": n_first, "revised_fields": n_rev}


# ---------------------------------------------------------------------------
# Match to race_results
# ---------------------------------------------------------------------------

_STRIP = re.compile(r"\b(park|city|racecourse|aw|the)\b")


def course_key(name) -> str:
    s = normalise_track(re.sub(r"[-]", " ", str(name or "").lower()))
    return _STRIP.sub("", str(name or "").lower()).replace("-", " ").replace(" ", "")[:6] if s else ""


def match_to_results(db_path: str = DEFAULT_DB, date_from: str | None = None, date_to: str | None = None) -> dict:
    """Link blandford_results to race_results by (date, course, time, horse) with fallbacks."""
    conn = sqlite3.connect(db_path)
    ensure_table(conn)
    where, params = [], []
    if date_from:
        where.append("{d} >= ?"); params.append(date_from)
    if date_to:
        where.append("{d} <= ?"); params.append(date_to)
    w_rr = (" WHERE " + " AND ".join(x.format(d="race_date") for x in where)) if where else ""
    w_bf = (" WHERE " + " AND ".join(x.format(d="meeting_date") for x in where)) if where else ""
    rr = pd.read_sql_query(f"SELECT id, race_date, race_time, track, horse_name FROM race_results{w_rr}", conn, params=params)
    bf = pd.read_sql_query(f"SELECT id, meeting_date, race_time, course_bf, horse_norm FROM blandford_results{w_bf}", conn, params=params)
    if rr.empty or bf.empty:
        conn.close()
        return {"total": int(len(bf)), "matched": 0, "rate": 0.0, "unmatched_courses": {}}
    rr["ck"] = rr["track"].map(course_key); rr["t24"] = rr["race_time"].map(db_time_to_24h); rr["hn"] = rr["horse_name"].map(normalise_horse)
    bf["ck"] = bf["course_bf"].map(course_key)
    full = rr.drop_duplicates(["race_date", "ck", "t24", "hn"]).set_index(["race_date", "ck", "t24", "hn"])["id"]
    k3 = rr.groupby(["race_date", "ck", "hn"])["id"].agg(["first", "size"]); k3 = k3[k3["size"] == 1]["first"]
    k2 = rr.groupby(["race_date", "hn"])["id"].agg(["first", "size"]); k2 = k2[k2["size"] == 1]["first"]
    m1 = full.reindex(pd.MultiIndex.from_frame(bf[["meeting_date", "ck", "race_time", "horse_norm"]])).values
    m2 = k3.reindex(pd.MultiIndex.from_frame(bf[["meeting_date", "ck", "horse_norm"]])).values
    m3 = k2.reindex(pd.MultiIndex.from_frame(bf[["meeting_date", "horse_norm"]])).values
    matched = pd.Series(m1).fillna(pd.Series(m2)).fillna(pd.Series(m3))
    how = pd.Series(np.where(~pd.isna(m1), "full", np.where(~pd.isna(m2), "course", np.where(~pd.isna(m3), "day", "none"))))
    bf["rid"] = matched.values
    upd = bf.dropna(subset=["rid"])
    conn.executemany("UPDATE blandford_results SET race_results_id = ? WHERE id = ?", [(int(r), int(i)) for r, i in zip(upd["rid"], upd["id"])])
    conn.commit(); conn.close()
    unmatched = bf.loc[matched.isna().values, "course_bf"].value_counts().head(20).to_dict()
    stats = {"total": int(len(bf)), "matched": int(len(upd)), "rate": float(len(upd) / len(bf)), "by_method": how.value_counts().to_dict(),
             "unmatched_courses": unmatched}
    log.info("matched %d/%d (%.1f%%) %s", stats["matched"], stats["total"], 100 * stats["rate"], stats["by_method"])
    return stats


def coverage_report(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    conn = sqlite3.connect(db_path); ensure_table(conn)
    q = """SELECT substr(meeting_date,1,7) AS month, COUNT(*) AS rows_loaded,
                  ROUND(100.0*SUM(race_results_id IS NOT NULL)/COUNT(*),1) AS match_pct,
                  ROUND(100.0*SUM(pre_race_master_rating IS NOT NULL)/COUNT(*),1) AS master_rating_pct,
                  ROUND(100.0*SUM(winner_sectional IS NOT NULL)/COUNT(*),1) AS sectional_pct,
                  ROUND(100.0*SUM(ip_min IS NOT NULL)/COUNT(*),1) AS ip_pct
           FROM blandford_results GROUP BY 1 ORDER BY 1"""
    df = pd.read_sql_query(q, conn); conn.close()
    return df


import numpy as np  # noqa: E402  (used in match_to_results)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Blandford / Timeform results feed: fetch / load / match")
    ap.add_argument("--db", default=DEFAULT_DB); ap.add_argument("--dir", default=str(RAW_DIR))
    ap.add_argument("--fetch", action="store_true"); ap.add_argument("--load", action="store_true")
    ap.add_argument("--match", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--from", dest="date_from", default=None); ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--days", type=int, default=None); ap.add_argument("--all-countries", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.days:
        d_to = date.today(); d_from = d_to - timedelta(days=args.days - 1)
    else:
        d_from = datetime.strptime(args.date_from, "%Y-%m-%d").date() if args.date_from else None
        d_to = datetime.strptime(args.date_to, "%Y-%m-%d").date() if args.date_to else date.today()
    raw = Path(args.dir)
    if args.fetch:
        if d_from is None:
            ap.error("--fetch needs --from or --days")
        s = _session(); login(s)
        fetch_range(d_from, d_to, s, raw)
    if args.load:
        paths = sorted(raw.glob("apidata_*.json"))
        if d_from:
            paths = [p for p in paths if p.name.split("_")[2].rstrip(".json") >= d_from.isoformat() and p.name.split("_")[1] <= d_to.isoformat()]
        load_files(paths, args.db, uk_ire_only=not args.all_countries)
    if args.match:
        st = match_to_results(args.db, d_from.isoformat() if d_from else None, d_to.isoformat() if d_to else None)
        print(f"matched {st['matched']}/{st['total']} ({100*st['rate']:.1f}%) by {st.get('by_method')}")
        if st["unmatched_courses"]:
            print("unmatched courses:", st["unmatched_courses"])
    if args.report:
        print(coverage_report(args.db).to_string(index=False))
    if not any([args.fetch, args.load, args.match, args.report]):
        ap.print_help(); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
