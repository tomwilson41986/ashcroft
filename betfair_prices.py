"""
Betfair historic price files: download, load, and match to race_results.

Source: https://promo.betfair.com/betfairsp/prices — one CSV per country,
market and day, named ``dwbfprices{uk|ire}{win|place}{DDMMYYYY}.csv``.
Each row is one runner in one market with the full pre-off and in-play
price history compressed into a handful of numbers:

    BSP              Betfair Starting Price
    PPWAP            pre-play (pre-off) volume-weighted average price
    MORNINGWAP       morning volume-weighted average price
    PPMAX / PPMIN    highest / lowest price traded pre-off
    IPMAX / IPMIN    highest / lowest price traded in-play
    MORNINGTRADEDVOL, PPTRADEDVOL, IPTRADEDVOL   volumes (GBP)

These give the market's *movement* (morning -> pre-off -> BSP), its
*confidence* (volumes) and — via IPMIN — how close a beaten horse came to
winning, which is hidden form that finishing position does not show.

The site sits behind Cloudflare and refuses some hosts (HTTP 403). The
downloader reports that clearly; ``--dir`` ingests files you fetched by
other means (browser, another machine, scripts/find_proxies.py).

Usage:
    python betfair_prices.py --fetch --from 2024-01-01 --to 2024-01-31   # download to data/betfair_raw/
    python betfair_prices.py --load                                        # parse + upsert every file in the raw dir
    python betfair_prices.py --match --from 2024-01-01                     # link rows to race_results
    python betfair_prices.py --fetch --days 3 --load --match               # nightly: last 3 days
    python betfair_prices.py --report                                      # coverage + unmatched course hints
"""

from __future__ import annotations

import argparse
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

log = logging.getLogger(__name__)

BASE_URL = "https://promo.betfair.com/betfairsp/prices"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DB = str(SCRIPT_DIR / "horse_racing.db")
RAW_DIR = SCRIPT_DIR / "data" / "betfair_raw"
COUNTRIES = ("uk", "ire")
MARKETS = ("win", "place")

RAW_COLUMNS = ["EVENT_ID", "MENU_HINT", "EVENT_NAME", "EVENT_DT", "SELECTION_ID", "SELECTION_NAME",
               "WIN_LOSE", "BSP", "PPWAP", "MORNINGWAP", "PPMAX", "PPMIN", "IPMAX", "IPMIN",
               "MORNINGTRADEDVOL", "PPTRADEDVOL", "IPTRADEDVOL"]

# Betfair MENU_HINT course abbreviations -> race_results.track (lower-case).
# Best effort; anything not listed falls back to prefix matching against the
# tracks present in the database and is reported by --report.
BF_COURSE_MAP = {
    "kemp": "kempton", "wolv": "wolverhampton", "ling": "lingfield", "newc": "newcastle",
    "sthl": "southwell", "chelm": "chelmsford", "chelmsford": "chelmsford", "ascot": "ascot",
    "newm": "newmarket", "york": "york", "donc": "doncaster", "newb": "newbury", "hayd": "haydock",
    "good": "goodwood", "ches": "chester", "epsm": "epsom", "epsom": "epsom", "sand": "sandown",
    "wind": "windsor", "leic": "leicester", "thsk": "thirsk", "thirsk": "thirsk", "pont": "pontefract",
    "bev": "beverley", "ripon": "ripon", "redc": "redcar", "catt": "catterick", "muss": "musselburgh",
    "haml": "hamilton", "ayr": "ayr", "brig": "brighton", "bath": "bath", "yarm": "yarmouth",
    "salis": "salisbury", "nott": "nottingham", "carl": "carlisle", "ffos": "ffos las", "ffosl": "ffos las",
    "chep": "chepstow", "warw": "warwick", "curr": "curragh", "leop": "leopardstown", "dund": "dundalk",
    "naas": "naas", "gowr": "gowran park", "cork": "cork", "galw": "galway", "tipp": "tipperary",
    "fair": "fairyhouse", "navan": "navan", "kill": "killarney", "bell": "bellewstown", "lim": "limerick",
    "sligo": "sligo", "ball": "ballinrobe", "rosc": "roscommon", "list": "listowel", "downp": "downpatrick",
    "droyal": "down royal", "wex": "wexford", "clon": "clonmel", "thur": "thurles", "punch": "punchestown",
    "layt": "laytown", "worc": "worcester", "sedg": "sedgefield", "uttox": "uttoxeter", "font": "fontwell",
    "plump": "plumpton", "hunt": "huntingdon", "kels": "kelso", "hex": "hexham", "perth": "perth",
    "cart": "cartmel", "bang": "bangor-on-dee", "ludl": "ludlow", "strat": "stratford", "tntn": "taunton",
    "taun": "taunton", "wcnt": "wincanton", "winc": "wincanton", "exet": "exeter", "chel": "cheltenham",
    "aint": "aintree", "mras": "market rasen", "faken": "fakenham", "towc": "towcester", "weth": "wetherby",
    "newt": "newton abbot", "ntab": "newton abbot", "here": "hereford",
}


# ---------------------------------------------------------------------------
# Naming / normalisation
# ---------------------------------------------------------------------------

def file_name(country: str, market: str, d: date) -> str:
    return f"dwbfprices{country}{market}{d.strftime('%d%m%Y')}.csv"


def parse_file_name(name: str) -> tuple[str, str, date] | None:
    m = re.match(r"dwbfprices(uk|ire)(win|place)(\d{2})(\d{2})(\d{4})\.csv$", Path(name).name)
    if not m:
        return None
    return m.group(1), m.group(2), date(int(m.group(5)), int(m.group(4)), int(m.group(3)))


def normalise_horse(name) -> str:
    """'1. Casts Tasha (IRE)' -> 'caststasha'."""
    s = str(name or "").lower()
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def normalise_track(name) -> str:
    s = str(name or "").lower()
    s = re.sub(r"\(.*?\)", "", s)  # "newmarket (july)" -> "newmarket"
    return re.sub(r"[^a-z]", "", s)


def course_from_hint(menu_hint: str, db_tracks_norm: dict[str, str] | None = None) -> str | None:
    """'GB / Kemp 12th Mar' -> 'kempton' (or the matching DB track name)."""
    if not menu_hint:
        return None
    body = menu_hint.split("/", 1)[-1].strip()
    abbrev = re.split(r"\s+\d", body, 1)[0].strip().lower()
    key = re.sub(r"[^a-z]", "", abbrev)
    canon = BF_COURSE_MAP.get(key)
    if canon is None and db_tracks_norm:
        for tn, original in db_tracks_norm.items():
            if len(key) >= 3 and (tn.startswith(key) or key.startswith(tn)):
                return original
        return None
    if canon is None:
        return None
    if db_tracks_norm:
        cn = normalise_track(canon)
        for tn, original in db_tracks_norm.items():
            if tn == cn or tn.startswith(cn) or cn.startswith(tn):
                return original
    return canon


def db_time_to_24h(t) -> str | None:
    """race_results.race_time like '1.00.' / '2.30' / '12.15' -> 'HH:MM' (UK racing: <10 means pm)."""
    m = re.match(r"^\s*(\d{1,2})[.:](\d{2})", str(t or ""))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h < 10:
        h += 12
    return f"{h:02d}:{mi:02d}"


def bf_event_time(event_dt) -> tuple[str | None, str | None]:
    """'12-03-2026 14:30' -> ('2026-03-12', '14:30')."""
    m = re.match(r"^\s*(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2})", str(event_dt or ""))
    if not m:
        return None, None
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}", f"{m.group(4)}:{m.group(5)}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_day(d: date, dest: Path = RAW_DIR, countries=COUNTRIES, markets=MARKETS,
                 session: requests.Session | None = None, retries: int = 3, sleep: float = 1.0,
                 skip_existing: bool = True) -> dict:
    """Fetch the four files for one day. Returns {file: status}."""
    dest.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    status = {}
    for c in countries:
        for m in markets:
            fn = file_name(c, m, d)
            path = dest / fn
            if skip_existing and path.exists() and path.stat().st_size > 200:
                status[fn] = "exists"
                continue
            url = f"{BASE_URL}/{fn}"
            for attempt in range(retries):
                try:
                    r = s.get(url, timeout=30)
                except requests.RequestException as exc:
                    log.warning("%s: %s (attempt %d)", fn, exc, attempt + 1)
                    time.sleep(sleep * (attempt + 1))
                    continue
                if r.status_code == 200 and r.text.lstrip().upper().startswith("EVENT_ID"):
                    path.write_text(r.text, encoding="utf-8")
                    status[fn] = "downloaded"
                    break
                if r.status_code == 403:
                    status[fn] = "blocked_403"
                    log.error("%s: HTTP 403 — promo.betfair.com (Cloudflare) refuses this host. "
                              "Download from a browser / another machine and use --dir.", fn)
                    break
                if r.status_code == 404:
                    status[fn] = "missing_404"
                    break
                status[fn] = f"http_{r.status_code}"
                time.sleep(sleep * (attempt + 1))
            else:
                status.setdefault(fn, "failed")
            time.sleep(sleep)
    return status


# ---------------------------------------------------------------------------
# Parse + load
# ---------------------------------------------------------------------------

def parse_file(path: str | Path) -> pd.DataFrame:
    """Parse one Betfair price CSV into a typed frame with country/market/date."""
    meta = parse_file_name(Path(path).name)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    out = pd.DataFrame({
        "event_id": pd.to_numeric(df["EVENT_ID"], errors="coerce").astype("Int64"),
        "menu_hint": df["MENU_HINT"].str.strip(),
        "event_name": df["EVENT_NAME"].str.strip(),
        "event_dt": df["EVENT_DT"].str.strip(),
        "selection_id": pd.to_numeric(df["SELECTION_ID"], errors="coerce").astype("Int64"),
        "selection_name": df["SELECTION_NAME"].str.strip(),
        "win_lose": pd.to_numeric(df["WIN_LOSE"], errors="coerce"),
    })
    for src, dst in [("BSP", "bsp"), ("PPWAP", "ppwap"), ("MORNINGWAP", "morningwap"), ("PPMAX", "ppmax"),
                     ("PPMIN", "ppmin"), ("IPMAX", "ipmax"), ("IPMIN", "ipmin"), ("MORNINGTRADEDVOL", "morning_vol"),
                     ("PPTRADEDVOL", "pp_vol"), ("IPTRADEDVOL", "ip_vol")]:
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    dates_times = out["event_dt"].map(bf_event_time)
    out["race_date"] = [d for d, _ in dates_times]
    out["race_time"] = [t for _, t in dates_times]
    out["horse_norm"] = out["selection_name"].map(normalise_horse)
    if meta:
        out["country"], out["market_type"], file_date = meta
        out["race_date"] = out["race_date"].fillna(file_date.isoformat())
    else:
        out["country"], out["market_type"] = "unknown", "win"
    out["source_file"] = Path(path).name
    return out


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS betfair_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT, market_type TEXT NOT NULL, race_date TEXT, race_time TEXT,
            event_id INTEGER, menu_hint TEXT, event_name TEXT, event_dt TEXT,
            track_bf TEXT, selection_id INTEGER, selection_name TEXT, horse_norm TEXT,
            win_lose REAL, bsp REAL, ppwap REAL, morningwap REAL, ppmax REAL, ppmin REAL,
            ipmax REAL, ipmin REAL, morning_vol REAL, pp_vol REAL, ip_vol REAL,
            race_results_id INTEGER, source_file TEXT,
            loaded_at TEXT DEFAULT (datetime('now')),
            UNIQUE(event_id, selection_id, market_type)
        );
        CREATE INDEX IF NOT EXISTS idx_bfp_date ON betfair_prices(race_date);
        CREATE INDEX IF NOT EXISTS idx_bfp_rr ON betfair_prices(race_results_id);
        CREATE INDEX IF NOT EXISTS idx_bfp_horse ON betfair_prices(horse_norm);
    """)
    conn.commit()


def load_files(paths, db_path: str = DEFAULT_DB) -> dict:
    """Parse and upsert files into betfair_prices. Returns counts."""
    conn = sqlite3.connect(db_path)
    ensure_table(conn)
    n_rows = n_files = 0
    cols = ["country", "market_type", "race_date", "race_time", "event_id", "menu_hint", "event_name", "event_dt",
            "track_bf", "selection_id", "selection_name", "horse_norm", "win_lose", "bsp", "ppwap", "morningwap",
            "ppmax", "ppmin", "ipmax", "ipmin", "morning_vol", "pp_vol", "ip_vol", "source_file"]
    for p in paths:
        try:
            df = parse_file(p)
        except Exception as exc:
            log.warning("skip %s: %s", p, exc)
            continue
        df["track_bf"] = df["menu_hint"].map(lambda h: course_from_hint(h) or "")
        rows = df[cols].astype(object).where(df[cols].notna(), None).values.tolist()
        conn.executemany(
            f"INSERT OR REPLACE INTO betfair_prices ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows)
        n_rows += len(rows); n_files += 1
    conn.commit(); conn.close()
    log.info("loaded %d rows from %d files", n_rows, n_files)
    return {"files": n_files, "rows": n_rows}


# ---------------------------------------------------------------------------
# Match to race_results
# ---------------------------------------------------------------------------

def match_to_results(db_path: str = DEFAULT_DB, date_from: str | None = None, date_to: str | None = None) -> dict:
    """Link betfair_prices rows to race_results by (date, track, time, horse).

    Fallbacks: (date, track, horse) then (date, horse) when unique. Sets
    race_results_id; returns match statistics and unmatched course hints.
    """
    conn = sqlite3.connect(db_path)
    ensure_table(conn)
    where, params = [], []
    if date_from:
        where.append("race_date >= ?"); params.append(date_from)
    if date_to:
        where.append("race_date <= ?"); params.append(date_to)
    w = (" WHERE " + " AND ".join(where)) if where else ""
    rr = pd.read_sql_query(f"SELECT id, race_date, race_time, track, horse_name FROM race_results{w}", conn, params=params)
    bf = pd.read_sql_query(f"SELECT id, race_date, race_time, menu_hint, horse_norm, market_type FROM betfair_prices{w}", conn, params=params)
    if rr.empty or bf.empty:
        conn.close()
        return {"matched": 0, "total": int(len(bf)), "rate": 0.0, "unmatched_hints": []}

    tracks_norm = {normalise_track(t): t for t in rr["track"].dropna().unique()}
    rr["track_n"] = rr["track"].map(normalise_track)
    rr["time24"] = rr["race_time"].map(db_time_to_24h)
    rr["horse_n"] = rr["horse_name"].map(normalise_horse)
    bf["track_db"] = bf["menu_hint"].map(lambda h: course_from_hint(h, tracks_norm))
    bf["track_n"] = bf["track_db"].map(normalise_track)
    bf["horse_n"] = bf["horse_norm"]

    key_full = rr.drop_duplicates(["race_date", "track_n", "time24", "horse_n"]).set_index(["race_date", "track_n", "time24", "horse_n"])["id"]
    k3 = rr.groupby(["race_date", "track_n", "horse_n"])["id"].agg(["first", "size"])
    key_track = k3[k3["size"] == 1]["first"]
    k2 = rr.groupby(["race_date", "horse_n"])["id"].agg(["first", "size"])
    key_day = k2[k2["size"] == 1]["first"]

    idx_full = pd.MultiIndex.from_frame(bf[["race_date", "track_n", "race_time", "horse_n"]].rename(columns={"race_time": "time24"}))
    m1 = key_full.reindex(idx_full).values
    idx_track = pd.MultiIndex.from_frame(bf[["race_date", "track_n", "horse_n"]])
    m2 = key_track.reindex(idx_track).values
    idx_day = pd.MultiIndex.from_frame(bf[["race_date", "horse_n"]])
    m3 = key_day.reindex(idx_day).values
    matched = pd.Series(m1).fillna(pd.Series(m2)).fillna(pd.Series(m3))
    bf["race_results_id"] = matched.values
    how = pd.Series(["full"] * len(bf))
    how[pd.isna(m1) & ~pd.isna(m2)] = "track"; how[pd.isna(m1) & pd.isna(m2) & ~pd.isna(m3)] = "day"; how[matched.isna()] = "none"

    upd = bf.dropna(subset=["race_results_id"])
    conn.executemany("UPDATE betfair_prices SET race_results_id = ? WHERE id = ?",
                     [(int(r), int(i)) for r, i in zip(upd["race_results_id"], upd["id"])])
    conn.commit(); conn.close()
    unmatched_hints = (bf.loc[bf["track_db"].isna(), "menu_hint"].str.replace(r"\s+\d.*$", "", regex=True)
                       .value_counts().head(30).to_dict())
    stats = {"total": int(len(bf)), "matched": int(len(upd)), "rate": float(len(upd) / len(bf)),
             "by_method": how.value_counts().to_dict(), "unmatched_hints": unmatched_hints}
    log.info("matched %d/%d (%.1f%%) %s", stats["matched"], stats["total"], 100 * stats["rate"], stats["by_method"])
    return stats


def coverage_report(db_path: str = DEFAULT_DB) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    ensure_table(conn)
    q = """
        SELECT substr(race_date, 1, 7) AS month, market_type,
               COUNT(*) AS rows_loaded, SUM(race_results_id IS NOT NULL) AS matched,
               ROUND(100.0 * SUM(race_results_id IS NOT NULL) / COUNT(*), 1) AS match_pct
        FROM betfair_prices GROUP BY 1, 2 ORDER BY 1, 2
    """
    df = pd.read_sql_query(q, conn); conn.close()
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Betfair historic price files: fetch / load / match")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--dir", default=str(RAW_DIR), help="raw CSV directory (default data/betfair_raw)")
    ap.add_argument("--fetch", action="store_true", help="download files for the date range")
    ap.add_argument("--load", action="store_true", help="parse + upsert files in --dir into betfair_prices")
    ap.add_argument("--match", action="store_true", help="link betfair_prices rows to race_results")
    ap.add_argument("--report", action="store_true", help="print monthly coverage")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--days", type=int, default=None, help="last N days (alternative to --from/--to)")
    ap.add_argument("--countries", default="uk,ire")
    ap.add_argument("--markets", default="win,place")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.days:
        d_to = date.today() - timedelta(days=1)
        d_from = d_to - timedelta(days=args.days - 1)
    else:
        d_from = datetime.strptime(args.date_from, "%Y-%m-%d").date() if args.date_from else None
        d_to = datetime.strptime(args.date_to, "%Y-%m-%d").date() if args.date_to else (date.today() - timedelta(days=1))
    raw = Path(args.dir)

    if args.fetch:
        if d_from is None:
            ap.error("--fetch needs --from or --days")
        sess = requests.Session()
        summary: dict[str, int] = {}
        for d in _daterange(d_from, d_to):
            st = download_day(d, raw, tuple(args.countries.split(",")), tuple(args.markets.split(",")), session=sess)
            for v in st.values():
                summary[v] = summary.get(v, 0) + 1
            if "blocked_403" in st.values():
                log.error("Stopping fetch: host is blocked (403). Use --dir with files fetched elsewhere.")
                break
        log.info("fetch summary: %s", summary)

    if args.load:
        paths = sorted(p for p in raw.glob("dwbfprices*.csv"))
        if d_from:
            paths = [p for p in paths if (m := parse_file_name(p.name)) and d_from <= m[2] <= d_to]
        load_files(paths, args.db)

    if args.match:
        stats = match_to_results(args.db, d_from.isoformat() if d_from else None, d_to.isoformat() if d_to else None)
        print(f"matched {stats['matched']}/{stats['total']} ({100 * stats['rate']:.1f}%) by {stats.get('by_method')}")
        if stats["unmatched_hints"]:
            print("unmatched course hints (add to BF_COURSE_MAP):", stats["unmatched_hints"])

    if args.report:
        print(coverage_report(args.db).to_string(index=False))

    if not any([args.fetch, args.load, args.match, args.report]):
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
