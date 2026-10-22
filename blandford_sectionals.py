#!/usr/bin/env python3
"""The Blandford backend's sectional tables (the owner, 22 Oct: "build these"): At The Races and Racing TV runs,
archived raw to S3, loaded into horse_racing.db and read to one table of per-run sectional readings for the
sectionals block (model/blocks/sectionals.py), which so far reads only the site's files (30 Jun 2024 - 29 Jun 2025 and
from 18 Apr 2026).

The backend (the same one blandford_sync.py reads) serves its MySQL tables without a login:

- ``/api/attheraces?limit&page``: At The Races' results with sectionals, worldwide, about 2.5m rows from 2018 (UK
  and Irish tracks carry no country suffix in Racename); per run the finishing speed (``Horse Finish %``), early,
  mid, final and top speeds, the segment times (``Start-5f`` .. ``1f-Finish``, a time and a position word) and the
  stride graphs. The route is LIMIT/OFFSET with no order clause and about 20 s a page far in, so the archive is
  pulled in pages of 5,000, resumed from a cursor, over a few nights.
- ``/api/racingtv/{horse}``: Racing TV's tracking metrics for every run of a horse (finishing speed %, 0-20 mph,
  top speed, stride length and frequency, each with its rank in the race), keyed by the race's URL (date, course
  slug, off time). Pulled a horse at a time for the horses our results hold in the window asked.

    python blandford_sectionals.py --fetch-atr --max-minutes 140          # resume the archive pull
    python blandford_sectionals.py --load-atr                             # S3 pages -> blandford_atr (UK/IE)
    python blandford_sectionals.py --fetch-rtv --from 2025-07-01 --to 2025-12-31 --max-minutes 60
    python blandford_sectionals.py --build                                # -> sectional_runs
    python blandford_sectionals.py --report

Raw pages go to s3://$CAPTURE_BUCKET/blandford_raw/{attheraces,racingtv}/ (CAPTURE_BUCKET, default ``ashcroft``);
a local copy under data/blandford_raw/. Read-only against the backend; nothing here places a bet.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from betfair_prices import normalise_horse  # noqa: E402

log = logging.getLogger("blandford_sectionals")
BASE_URL = "https://horseracesbackend-production.up.railway.app"
ROOT = Path(__file__).resolve().parent
DEFAULT_DB = str(ROOT / "horse_racing.db")
RAW_DIR = ROOT / "data" / "blandford_raw"
BUCKET = os.environ.get("CAPTURE_BUCKET") or "ashcroft"
REGION = os.environ.get("PRED_REGION") or "eu-west-2"
PREFIX = "blandford_raw"
PAGE_SIZE = 5000
FOREIGN = re.compile(r"\(([A-Za-z .'-]+)\)\s*$")
IRISH = {"ire", "ireland", "irl"}
ATR_COLS = ["day", "track", "race_time", "horse_norm", "horse", "pos", "dist_btn", "runners", "distance", "ground",
            "race_name", "finish_pct", "early_mph", "mid_mph", "final_mph", "top_mph", "seg_2f_1f", "seg_1f_fin",
            "finish_time", "segments"]
RTV_COLS = ["race_url", "day", "track_slug", "race_time", "horse_norm", "horse", "pos", "fsp_pct", "fsp_rank",
            "zero20_s", "zero20_rank", "top_mph", "top_rank", "stride_m", "stride_rank", "freq_avg", "notes"]


def _s3():
    import boto3
    return boto3.client("s3", region_name=REGION)


def _put(key: str, body: bytes) -> None:
    _s3().put_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}", Body=body)


def _get(key: str) -> bytes | None:
    try:
        return _s3().get_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}")["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def _list(prefix: str) -> list[str]:
    s3, out, token = _s3(), [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": f"{PREFIX}/{prefix}"}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += [o["Key"][len(PREFIX) + 1:] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        token = r.get("NextContinuationToken")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "ashcroft-blandford-sectionals/1.0"
    return s


def _num(v):
    if v is None:
        return np.nan
    s = str(v).strip().strip("'").replace(",", "")
    m = re.match(r"^-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


# ---------------------------------------------------------------------------
# At The Races: the archive, a page at a time
# ---------------------------------------------------------------------------

def fetch_atr(max_minutes: float, page_size: int = PAGE_SIZE, sleep: float = 0.5) -> dict:
    """Pull pages from the cursor on; each page to S3 and the local dir; stop at the budget or an empty page."""
    t0 = time.time()
    cur = json.loads(_get("attheraces/_cursor.json") or b'{"next_page": 1, "done": false}')
    if cur.get("done"):
        log.info("At The Races archive already complete (%d pages)", cur["next_page"] - 1)
        return cur
    s = _session()
    (RAW_DIR / "atr").mkdir(parents=True, exist_ok=True)
    page = int(cur["next_page"])
    pulled = 0
    while time.time() - t0 < max_minutes * 60:
        rows = None
        for attempt in range(3):
            try:
                r = s.get(f"{BASE_URL}/api/attheraces", params={"limit": page_size, "page": page}, timeout=300)
                if r.status_code == 200:
                    rows = r.json().get("data", [])
                    break
                log.warning("page %d: HTTP %s (attempt %d)", page, r.status_code, attempt + 1)
            except (requests.RequestException, ValueError) as exc:
                log.warning("page %d: %s (attempt %d)", page, exc, attempt + 1)
            time.sleep(5 * (attempt + 1))
        if rows is None:
            break
        if not rows:
            cur["done"] = True
            log.info("the archive ends before page %d", page)
            break
        body = gzip.compress(json.dumps(rows).encode())
        _put(f"attheraces/page_{page:06d}.json.gz", body)
        (RAW_DIR / "atr" / f"page_{page:06d}.json.gz").write_bytes(body)
        pulled += len(rows)
        page += 1
        cur["next_page"] = page
        cur["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _put("attheraces/_cursor.json", json.dumps(cur).encode())
        if len(rows) < page_size:
            cur["done"] = True
            _put("attheraces/_cursor.json", json.dumps(cur).encode())
            break
        time.sleep(sleep)
    log.info("At The Races: %d rows pulled, next page %d, done %s", pulled, cur["next_page"], cur.get("done"))
    return cur


def _uk_ie(racename: str) -> bool:
    m = FOREIGN.search(racename or "")
    return m is None or m.group(1).strip().lower() in IRISH


def parse_atr(rows: list[dict]) -> pd.DataFrame:
    keep = [r for r in rows if _uk_ie(r.get("Racename"))]
    if not keep:
        return pd.DataFrame(columns=ATR_COLS)
    seg_cols = sorted({k for r in keep for k in r if re.match(r"^(Start-\d+f|\d+f-\d+f|\d+f-Finish)$", k)})
    out = []
    for r in keep:
        day = pd.to_datetime(r.get("Date"), format="%d-%m-%Y", errors="coerce")
        if pd.isna(day):
            continue
        name = FOREIGN.sub("", str(r.get("Racename") or "")).strip()
        segs = {k: r[k] for k in seg_cols if r.get(k) not in (None, "", "''")}
        out.append({"day": day.strftime("%Y-%m-%d"), "track": name, "race_time": r.get("Time"),
                    "horse_norm": normalise_horse(r.get("Horse")), "horse": r.get("Horse"), "pos": _num(r.get("Position")),
                    "dist_btn": _num(r.get("Dist Btn")), "runners": _num(r.get("Runners")), "distance": r.get("Distance"),
                    "ground": r.get("Ground"), "race_name": r.get("Type"), "finish_pct": _num(r.get("Horse Finish %")),
                    "early_mph": _num(r.get("Early Speed (MPH)")), "mid_mph": _num(r.get("Mid Speed (MPH)")),
                    "final_mph": _num(r.get("Final Speed (MPH)")), "top_mph": _num(r.get("Top Speed (MPH)")),
                    "seg_2f_1f": _num(r.get("2f-1f")), "seg_1f_fin": _num(r.get("1f-Finish")),
                    "finish_time": r.get("Finish"), "segments": json.dumps(segs) if segs else None})
    d = pd.DataFrame(out, columns=ATR_COLS)
    return d[d.horse_norm.str.len() > 0]


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blandford_atr (
            day TEXT, track TEXT, race_time TEXT, horse_norm TEXT, horse TEXT, pos REAL, dist_btn REAL, runners REAL,
            distance TEXT, ground TEXT, race_name TEXT, finish_pct REAL, early_mph REAL, mid_mph REAL, final_mph REAL,
            top_mph REAL, seg_2f_1f REAL, seg_1f_fin REAL, finish_time TEXT, segments TEXT,
            loaded_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (day, track, race_time, horse_norm));
        CREATE INDEX IF NOT EXISTS idx_batr_horse ON blandford_atr(horse_norm);
        CREATE TABLE IF NOT EXISTS blandford_atr_files (page TEXT PRIMARY KEY, rows INTEGER, kept INTEGER,
            loaded_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS blandford_rtv (
            race_url TEXT, day TEXT, track_slug TEXT, race_time TEXT, horse_norm TEXT, horse TEXT, pos REAL,
            fsp_pct REAL, fsp_rank REAL, zero20_s REAL, zero20_rank REAL, top_mph REAL, top_rank REAL, stride_m REAL,
            stride_rank REAL, freq_avg REAL, notes TEXT, loaded_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (race_url, horse_norm));
        CREATE INDEX IF NOT EXISTS idx_brtv_horse ON blandford_rtv(horse_norm);
        CREATE INDEX IF NOT EXISTS idx_brtv_day ON blandford_rtv(day);
        CREATE TABLE IF NOT EXISTS blandford_rtv_fetched (horse_norm TEXT PRIMARY KEY, rows INTEGER,
            fetched_at TEXT DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS sectional_runs (
            day TEXT, name_norm TEXT, src TEXT, n_field REAL, pos REAL, fsp REAL, fsp_rel REAL, fsp_vs_pos REAL,
            early_pct REAL, late_rel REAL, sl_rel REAL, built_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (day, name_norm, src));
        CREATE INDEX IF NOT EXISTS idx_secruns_day ON sectional_runs(day);
    """)


def load_atr(db_path: str, from_s3: bool = True) -> dict:
    """Every page not yet loaded, UK and Irish rows only, into blandford_atr."""
    conn = sqlite3.connect(db_path)
    ensure_tables(conn)
    done = {r[0] for r in conn.execute("SELECT page FROM blandford_atr_files")}
    keys = sorted(k for k in (_list("attheraces/page_") if from_s3 else
                              [f"attheraces/{p.name}" for p in (RAW_DIR / "atr").glob("page_*.json.gz")])
                  if k.endswith(".json.gz"))
    n_rows = n_kept = 0
    for k in keys:
        page = k.rsplit("/", 1)[-1]
        if page in done:
            continue
        raw = _get(k) if from_s3 else (RAW_DIR / "atr" / page).read_bytes()
        if raw is None:
            continue
        rows = json.loads(gzip.decompress(raw))
        d = parse_atr(rows)
        if len(d):
            d = d.drop_duplicates(["day", "track", "race_time", "horse_norm"], keep="last")
            cols = ATR_COLS
            conn.executemany(f"INSERT OR REPLACE INTO blandford_atr ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                             d[cols].astype(object).where(d[cols].notna(), None).values.tolist())
        conn.execute("INSERT OR REPLACE INTO blandford_atr_files (page, rows, kept) VALUES (?, ?, ?)", (page, len(rows), len(d)))
        conn.commit()
        n_rows += len(rows)
        n_kept += len(d)
        log.info("%s: %d rows, %d UK/IE", page, len(rows), len(d))
    conn.close()
    return {"pages": len(keys) - len(done), "rows": n_rows, "kept": n_kept}


# ---------------------------------------------------------------------------
# Racing TV: a horse at a time
# ---------------------------------------------------------------------------

_RTV_URL = re.compile(r"/results/(\d{4}-\d{2}-\d{2})/([^/]+)/(\d{3,4})")


def parse_rtv(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        m = _RTV_URL.search(str(r.get("RaceURL") or ""))
        if not m:
            continue
        hhmm = m.group(3).zfill(4)
        stride = r.get("Stride Length Value")
        sl = _num(stride)
        if stride and "ft" in str(stride).lower():
            sl *= 0.3048
        out.append({"race_url": r.get("RaceURL"), "day": m.group(1), "track_slug": m.group(2),
                    "race_time": f"{hhmm[:2]}:{hhmm[2:]}", "horse_norm": normalise_horse(r.get("horseName")),
                    "horse": r.get("horseName"), "pos": _num(r.get("Position")),
                    "fsp_pct": _num(r.get("Finishing Speed % Value")), "fsp_rank": _num(r.get("Finishing Speed % Pos")),
                    "zero20_s": _num(r.get("0-20MPH Value")), "zero20_rank": _num(r.get("0-20MPH Pos")),
                    "top_mph": _num(r.get("Top Speed Value")), "top_rank": _num(r.get("Top Speed Pos")),
                    "stride_m": sl, "stride_rank": _num(r.get("Stride Length Pos")),
                    "freq_avg": _num(r.get("Avg Frequency Value")), "notes": r.get("Notes")})
    d = pd.DataFrame(out, columns=RTV_COLS)
    return d[d.horse_norm.str.len() > 0].drop_duplicates(["race_url", "horse_norm"], keep="last")


def fetch_rtv(db_path: str, d_from: str, d_to: str, max_minutes: float, sleep: float = 0.3) -> dict:
    """The horses our results hold in the window (Flat and all-weather, UK and Ireland), not yet fetched: each one's
    Racing TV rows to S3 and blandford_rtv."""
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    ensure_tables(conn)
    horses = pd.read_sql_query("""SELECT DISTINCT horse_name FROM race_results WHERE race_date >= ? AND race_date <= ?
                                  AND (race_type IS NULL OR LOWER(race_type) NOT LIKE '%hurdle%')
                                  AND (race_type IS NULL OR LOWER(race_type) NOT LIKE '%chase%')
                                  AND (race_type IS NULL OR LOWER(race_type) NOT LIKE '%nh flat%')""",
                               conn, params=(d_from, d_to))
    horses["horse_norm"] = horses.horse_name.map(normalise_horse)
    done = {r[0] for r in conn.execute("SELECT horse_norm FROM blandford_rtv_fetched")}
    todo = horses[~horses.horse_norm.isin(done) & (horses.horse_norm.str.len() > 0)].drop_duplicates("horse_norm")
    log.info("Racing TV: %d horses in %s..%s, %d to fetch", len(horses.horse_norm.unique()), d_from, d_to, len(todo))
    s = _session()
    n = n_rows = 0
    for r in todo.itertuples():
        if time.time() - t0 > max_minutes * 60:
            break
        name = re.sub(r"\s*\([A-Za-z]{2,3}\)\s*$", "", str(r.horse_name)).strip()
        rows = None
        for attempt in range(2):
            try:
                resp = s.get(f"{BASE_URL}/api/racingtv/{urllib.parse.quote(name)}", timeout=120)
                if resp.status_code == 200:
                    j = resp.json()
                    rows = j.get("data", j) if isinstance(j, dict) else j
                    break
            except (requests.RequestException, ValueError) as exc:
                log.warning("%s: %s", name, exc)
            time.sleep(3)
        if rows is None:
            continue
        if rows:
            _put(f"racingtv/{r.horse_norm}.json.gz", gzip.compress(json.dumps(rows).encode()))
            d = parse_rtv(rows)
            if len(d):
                conn.executemany(f"INSERT OR REPLACE INTO blandford_rtv ({', '.join(RTV_COLS)}) VALUES ({', '.join('?' * len(RTV_COLS))})",
                                 d[RTV_COLS].astype(object).where(d[RTV_COLS].notna(), None).values.tolist())
                n_rows += len(d)
        conn.execute("INSERT OR REPLACE INTO blandford_rtv_fetched (horse_norm, rows) VALUES (?, ?)", (r.horse_norm, len(rows or [])))
        conn.commit()
        n += 1
        time.sleep(sleep)
    conn.close()
    log.info("Racing TV: %d horses fetched, %d runs loaded, %d horses left", n, n_rows, len(todo) - n)
    return {"horses": n, "runs": n_rows, "left": len(todo) - n}


# ---------------------------------------------------------------------------
# The readings the block reads: one row a run, each against the field it met
# ---------------------------------------------------------------------------

def _within(h: pd.DataFrame, race: pd.Series, fsp: str, early: str, early_ascending: bool, late: str | None,
            sl: str | None) -> pd.DataFrame:
    g = h.groupby(race)
    out = pd.DataFrame(index=h.index)
    out["n_field"] = g[fsp].transform("size")
    out["pos"] = h["pos"]
    out["fsp"] = h[fsp]
    out["fsp_rel"] = h[fsp] - g[fsp].transform("mean")
    fsp_rank = g[fsp].rank(ascending=False, pct=True)
    pos_rank = g["pos"].rank(ascending=True, pct=True)
    out["fsp_vs_pos"] = pos_rank - fsp_rank
    er = g[early].rank(ascending=early_ascending, method="average")
    n_e = g[early].transform("count")
    out["early_pct"] = (er - 1) / (n_e - 1).where(n_e > 1)
    out["late_rel"] = (h[late] - g[late].transform("mean")) if late else np.nan
    out["sl_rel"] = (h[sl] - g[sl].transform("mean")) if sl else np.nan
    thin = out["n_field"] < 3
    out.loc[thin, ["fsp_rel", "fsp_vs_pos", "early_pct", "late_rel", "sl_rel"]] = np.nan
    return out


def build(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    ensure_tables(conn)
    frames = []
    a = pd.read_sql_query("SELECT day, track, race_time, horse_norm, pos, finish_pct, early_mph, seg_2f_1f, seg_1f_fin "
                          "FROM blandford_atr WHERE finish_pct > 0", conn)
    if len(a):
        a["late2"] = a.seg_2f_1f + a.seg_1f_fin
        w = _within(a, a.day + "|" + a.track + "|" + a.race_time.astype(str), "finish_pct", "early_mph", False, "late2", None)
        w["day"], w["name_norm"], w["src"] = a.day, a.horse_norm, "atr"
        frames.append(w)
    r = pd.read_sql_query("SELECT race_url, day, horse_norm, pos, fsp_pct, zero20_s, stride_m FROM blandford_rtv "
                          "WHERE fsp_pct > 0", conn)
    if len(r):
        w = _within(r, r.race_url, "fsp_pct", "zero20_s", True, None, "stride_m")
        w["day"], w["name_norm"], w["src"] = r.day, r.horse_norm, "rtv"
        frames.append(w)
    cols = ["day", "name_norm", "src", "n_field", "pos", "fsp", "fsp_rel", "fsp_vs_pos", "early_pct", "late_rel", "sl_rel"]
    if not frames:
        conn.close()
        return {"rows": 0}
    t = pd.concat(frames, ignore_index=True)[cols]
    t = t.dropna(subset=["day", "name_norm"]).drop_duplicates(["day", "name_norm", "src"], keep="last")
    conn.execute("DELETE FROM sectional_runs")
    conn.executemany(f"INSERT OR REPLACE INTO sectional_runs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     t.astype(object).where(t.notna(), None).values.tolist())
    conn.commit()
    by = t.groupby("src").agg(runs=("day", "size"), first=("day", "min"), last=("day", "max"))
    conn.close()
    log.info("sectional_runs: %d rows\n%s", len(t), by.to_string())
    return {"rows": int(len(t)), "by_source": by.to_dict()}


def report(db_path: str) -> None:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    for sql in ("SELECT COUNT(*), MIN(day), MAX(day), SUM(finish_pct > 0) FROM blandford_atr",
                "SELECT COUNT(*) FROM blandford_atr_files",
                "SELECT COUNT(*), MIN(day), MAX(day), COUNT(DISTINCT horse_norm) FROM blandford_rtv",
                "SELECT COUNT(*) FROM blandford_rtv_fetched",
                "SELECT src, COUNT(*), MIN(day), MAX(day) FROM sectional_runs GROUP BY src"):
        try:
            print(sql, "->", conn.execute(sql).fetchall())
        except sqlite3.Error as exc:
            print(sql, "->", exc)
    conn.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--fetch-atr", action="store_true")
    ap.add_argument("--load-atr", action="store_true")
    ap.add_argument("--fetch-rtv", action="store_true")
    ap.add_argument("--from", dest="d_from", default="2025-07-01")
    ap.add_argument("--to", dest="d_to", default=date.today().isoformat())
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--page-size", type=int, default=PAGE_SIZE)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if a.fetch_atr:
        fetch_atr(a.max_minutes, a.page_size)
    if a.load_atr:
        print(load_atr(a.db))
    if a.fetch_rtv:
        print(fetch_rtv(a.db, a.d_from, a.d_to, a.max_minutes))
    if a.build:
        print(build(a.db))
    if a.report:
        report(a.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
