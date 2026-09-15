"""
HorseRaceBase Ratings Machine downloads -> hrb_ratings table.

The Ratings Machine (ratingsmachinev2.php, "Download ratings for a
previous date") returns one CSV per rating set per day via
excelratingsbyprevdate.php. Each of the user's saved sets is a separate
pre-race rating (Speed, Ability, Recency, jockey, sire, ...), legitimate
as a same-day feature.

HRB rate-limits data downloads per account ("Data downloads temporarily
unavailable ... will become available again at <time>") and the nightly
results scrape uses the same allowance, so this downloader is throttled
(``--spacing`` seconds between requests), capped (``--max-requests``),
resumable (skips files already on disk) and stops cleanly on the notice.

Usage:
    python hrb_ratings.py --fetch --sets 269,132216 --from 2026-01-20 --to 2026-02-18 --spacing 5 --max-requests 60
    python hrb_ratings.py --load --match --report
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from betfair_prices import db_time_to_24h, normalise_horse, normalise_track

log = logging.getLogger(__name__)

BASE_URL = "https://www.horseracebase.com"
DOWNLOAD_URL = f"{BASE_URL}/excelratingsbyprevdate.php"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = str(SCRIPT_DIR / "horse_racing.db")
RAW_DIR = SCRIPT_DIR / "data" / "hrb_ratings_raw"

# set id -> name, as listed on the Ratings Machine download form
RATING_SETS = {
    269: "HRB Standard", 194: "Like Old HRB Std", 132216: "Speed HRB", 132274: "Flat Class Ratings",
    132275: "NH Class Ratings", 133985: "jockey2", 134569: "Recency 2", 134658: "Ability", 134659: "Conex New",
    135439: "SpeedRatingsLR", 139796: "Sire", 139797: "sire2", 139798: "sire3", 142359: "trainer runs", 142799: "speed",
}
DEFAULT_SETS = [269, 132216, 133985, 134569, 134658, 134659, 135439]


class RateLimited(RuntimeError):
    pass


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _rate_limit_notice(text: str) -> str | None:
    if "temporarily unavailable" in text and "download" in text.lower():
        t = re.sub(r"<[^>]+>", " ", text); t = re.sub(r"\s+", " ", t)
        m = re.search(r"available again:?\s*([^.]{5,80})", t)
        return m.group(1).strip() if m else "rate limited"
    return None


def download_day(session, set_id: int, d: date, user_id: str) -> str:
    """One set, one day, as CSV text. Raises RateLimited on the HRB notice."""
    r = session.post(DOWNLOAD_URL, data={"user": user_id, "usecustomsettings": str(set_id), "day": str(d.day),
                                          "month": str(d.month), "year": str(d.year), "csv": "on"}, timeout=300)
    r.raise_for_status()
    notice = _rate_limit_notice(r.text)
    if notice:
        raise RateLimited(notice)
    if "<html" in r.text[:300].lower() and "," not in r.text[:300]:
        raise ValueError("unexpected HTML response (login expired?)")
    return r.text


def fetch_range(sets, d_from: date, d_to: date, dest: Path = RAW_DIR, spacing: float = 4.0,
                max_requests: int | None = None, user_id: str | None = None) -> dict:
    """Throttled, resumable download of sets x days. Stops on rate limit."""
    import scraper  # HRB login lives there (HRB_USERNAME / HRB_PASSWORD env)
    dest.mkdir(parents=True, exist_ok=True)
    session = scraper.create_session()
    uid = scraper.login(session)
    if not uid:
        raise RuntimeError("HRB login failed (set HRB_USERNAME / HRB_PASSWORD)")
    user_id = user_id or os.getenv("HRB_USER_ID") or str(uid)
    n_req = n_skip = 0; stopped = None
    d = d_from
    while d <= d_to and stopped is None:
        for set_id in sets:
            path = dest / f"ratings_{set_id}_{d.isoformat()}.csv"
            if path.exists() and path.stat().st_size > 20:
                n_skip += 1; continue
            if max_requests is not None and n_req >= max_requests:
                stopped = f"max_requests={max_requests} reached"; break
            try:
                text = download_day(session, set_id, d, user_id)
            except RateLimited as exc:
                stopped = f"rate limited: {exc}"; log.error("HRB %s", stopped); break
            except Exception as exc:
                log.warning("%s %s: %s", set_id, d, exc); time.sleep(spacing); n_req += 1; continue
            path.write_text(text); n_req += 1
            log.info("downloaded set %s %s (%d bytes)", set_id, d, len(text))
            time.sleep(spacing)
        d += timedelta(days=1)
    return {"requests": n_req, "skipped": n_skip, "stopped": stopped, "last_date": (d - timedelta(days=1)).isoformat()}


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

_COLS = {
    "horse": re.compile(r"^(horse|horse ?name|name|runner)$", re.I),
    "track": re.compile(r"^(track|course|venue|meeting)$", re.I),
    "time": re.compile(r"^(time|race ?time|off|off ?time)$", re.I),
    "date": re.compile(r"^(date|race ?date|racedate)$", re.I),
    "rating": re.compile(r"^(rating|rtg|total|score|total ?rating|hrb ?rating|final ?rating)$", re.I),
    "rank": re.compile(r"^(rank|rating ?rank|pos|position)$", re.I),
}


def parse_csv(text: str, set_id: int, d: date | None = None) -> pd.DataFrame:
    """Standardise a Ratings Machine CSV: race_date, track, race_time, horse_name, rating, rating_rank, extras."""
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    found = {}
    for key, rx in _COLS.items():
        for c in df.columns:
            if rx.match(c):
                found[key] = c; break
    if "horse" not in found:
        raise ValueError(f"no horse column in {list(df.columns)[:12]}")
    if "rating" not in found:
        numeric = [c for c in df.columns if c not in found.values() and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
        if not numeric:
            raise ValueError(f"no rating column in {list(df.columns)[:12]}")
        found["rating"] = numeric[-1]
    out = pd.DataFrame({
        "set_id": set_id, "set_name": RATING_SETS.get(set_id, str(set_id)),
        "race_date": pd.to_datetime(df[found["date"]], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d") if "date" in found else (d.isoformat() if d else None),
        "track": df[found["track"]].str.strip() if "track" in found else "",
        "race_time": df[found["time"]].str.strip() if "time" in found else "",
        "horse_name": df[found["horse"]].str.strip(),
        "rating": pd.to_numeric(df[found["rating"]], errors="coerce"),
        "rating_rank": pd.to_numeric(df[found["rank"]], errors="coerce") if "rank" in found else np.nan,
    })
    extras = [c for c in df.columns if c not in found.values()]
    out["extras"] = [json.dumps({c: r[c] for c in extras}) for _, r in df.iterrows()] if extras else "{}"
    out["horse_norm"] = out["horse_name"].map(normalise_horse)
    out["race_time_24"] = out["race_time"].map(lambda t: db_time_to_24h(t) if re.match(r"^\d{1,2}[.:]\d{2}", str(t)) else str(t))
    return out.dropna(subset=["rating"])


# ---------------------------------------------------------------------------
# Load + match
# ---------------------------------------------------------------------------

def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS hrb_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            set_id INTEGER NOT NULL, set_name TEXT, race_date TEXT, track TEXT, race_time TEXT, race_time_24 TEXT,
            horse_name TEXT, horse_norm TEXT, rating REAL, rating_rank REAL, extras TEXT,
            race_results_id INTEGER, loaded_at TEXT DEFAULT (datetime('now')),
            UNIQUE(set_id, race_date, track, race_time, horse_norm)
        );
        CREATE INDEX IF NOT EXISTS idx_hrbr_date ON hrb_ratings(race_date);
        CREATE INDEX IF NOT EXISTS idx_hrbr_rr ON hrb_ratings(race_results_id);
    """)
    conn.commit()


def load_files(paths, db_path: str = DEFAULT_DB) -> dict:
    conn = sqlite3.connect(db_path); ensure_table(conn)
    cols = ["set_id", "set_name", "race_date", "track", "race_time", "race_time_24", "horse_name", "horse_norm", "rating", "rating_rank", "extras"]
    n_rows = n_files = 0
    for p in paths:
        m = re.match(r"ratings_(\d+)_(\d{4}-\d{2}-\d{2})\.csv$", Path(p).name)
        if not m:
            continue
        try:
            df = parse_csv(Path(p).read_text(), int(m.group(1)), date.fromisoformat(m.group(2)))
        except Exception as exc:
            log.warning("skip %s: %s", p, exc); continue
        vals = df[cols].astype(object).where(df[cols].notna(), None).values.tolist()
        conn.executemany(f"INSERT OR REPLACE INTO hrb_ratings ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals)
        n_rows += len(vals); n_files += 1
    conn.commit(); conn.close()
    return {"files": n_files, "rows": n_rows}


def match_to_results(db_path: str = DEFAULT_DB) -> dict:
    conn = sqlite3.connect(db_path); ensure_table(conn)
    rr = pd.read_sql_query("SELECT id, race_date, race_time, track, horse_name FROM race_results", conn)
    hr = pd.read_sql_query("SELECT id, race_date, track, race_time_24, horse_norm FROM hrb_ratings", conn)
    if rr.empty or hr.empty:
        conn.close(); return {"total": int(len(hr)), "matched": 0, "rate": 0.0}
    rr["tk"] = rr["track"].map(normalise_track); rr["t24"] = rr["race_time"].map(db_time_to_24h); rr["hn"] = rr["horse_name"].map(normalise_horse)
    hr["tk"] = hr["track"].map(normalise_track)
    full = rr.drop_duplicates(["race_date", "tk", "t24", "hn"]).set_index(["race_date", "tk", "t24", "hn"])["id"]
    k3 = rr.groupby(["race_date", "tk", "hn"])["id"].agg(["first", "size"]); k3 = k3[k3["size"] == 1]["first"]
    k2 = rr.groupby(["race_date", "hn"])["id"].agg(["first", "size"]); k2 = k2[k2["size"] == 1]["first"]
    m1 = full.reindex(pd.MultiIndex.from_frame(hr[["race_date", "tk", "race_time_24", "horse_norm"]])).values
    m2 = k3.reindex(pd.MultiIndex.from_frame(hr[["race_date", "tk", "horse_norm"]])).values
    m3 = k2.reindex(pd.MultiIndex.from_frame(hr[["race_date", "horse_norm"]])).values
    matched = pd.Series(m1).fillna(pd.Series(m2)).fillna(pd.Series(m3))
    upd = hr.assign(rid=matched.values).dropna(subset=["rid"])
    conn.executemany("UPDATE hrb_ratings SET race_results_id = ? WHERE id = ?", [(int(r), int(i)) for r, i in zip(upd["rid"], upd["id"])])
    conn.commit(); conn.close()
    return {"total": int(len(hr)), "matched": int(len(upd)), "rate": float(len(upd) / len(hr))}


def coverage_report(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    conn = sqlite3.connect(db_path); ensure_table(conn)
    df = pd.read_sql_query("""SELECT set_id, set_name, MIN(race_date) first_date, MAX(race_date) last_date, COUNT(DISTINCT race_date) days,
                              COUNT(*) rows_loaded, ROUND(100.0*SUM(race_results_id IS NOT NULL)/COUNT(*),1) match_pct
                              FROM hrb_ratings GROUP BY 1,2 ORDER BY 1""", conn)
    conn.close(); return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="HRB Ratings Machine: fetch / load / match")
    ap.add_argument("--db", default=DEFAULT_DB); ap.add_argument("--dir", default=str(RAW_DIR))
    ap.add_argument("--fetch", action="store_true"); ap.add_argument("--load", action="store_true")
    ap.add_argument("--match", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--sets", default=",".join(map(str, DEFAULT_SETS)))
    ap.add_argument("--from", dest="date_from", default=None); ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--spacing", type=float, default=4.0); ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--user-id", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = Path(args.dir)
    if args.fetch:
        if not args.date_from:
            ap.error("--fetch needs --from")
        d_from = date.fromisoformat(args.date_from); d_to = date.fromisoformat(args.date_to) if args.date_to else d_from
        st = fetch_range([int(x) for x in args.sets.split(",")], d_from, d_to, raw, args.spacing, args.max_requests, args.user_id)
        print("fetch:", st)
    if args.load:
        print("load:", load_files(sorted(raw.glob("ratings_*.csv")), args.db))
    if args.match:
        print("match:", match_to_results(args.db))
    if args.report:
        print(coverage_report(args.db).to_string(index=False))
    if not any([args.fetch, args.load, args.match, args.report]):
        ap.print_help(); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
