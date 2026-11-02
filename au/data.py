"""The Australian BSP targets: every Australian thoroughbred runner's Betfair SP and result, from Sep 2012.

The model's target and benchmark, as `betfair_prices` is for the UK/IE model. Read from our own price-file archive
(s3://$CAPTURE_BUCKET/betfair_prices_raw/dwbfpricesauswin<DDMMYYYY>.csv, archived from the UK server by
betfair_prices.py --archive), never loaded into horse_racing.db. One row a runner a market:

    market_id, selection_id, horse, horse_norm, cloth, track, meeting_date, race_no, distance_m, race_type,
    race_family, off_uk, bsp, won, pp_vol, ip_vol, source_file

Kept: the BSP, the result, the race and the money. Left out: the files' MORNINGWAP and PPWAP, which for Australian
races (run in the UK night) do not measure a pre-off price (reports/au_nz_market_scan.md, point 4). Harness markets
(Pace/Trot) and anything not '(AUS)' are dropped, and counted.

``--anz-check`` reconciles the table with Betfair's own ANZ thoroughbred files (Betfair Data Scientists,
https://betfair-datascientists.github.io/data/dataListing/, 2020 on): markets and runners found both ways, BSP and
result agreement, race number, distance and track agreement, and the clock of ``off_uk`` against the ANZ files'.

    python -m au.data --build --anz-check                 # S3 in, S3 out (sources/au/)
    python -m au.data --build --root /tmp/au --limit 30   # a local store, the 30 newest files (a dry run)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from au import config

log = logging.getLogger(__name__)

KEEP = ["EVENT_ID", "MENU_HINT", "EVENT_NAME", "EVENT_DT", "SELECTION_ID", "SELECTION_NAME", "WIN_LOSE", "BSP",
        "PPTRADEDVOL", "IPTRADEDVOL"]
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                                      "DEC"], start=1)}
HINT_RE = re.compile(r"^\s*(?P<track>.*?)\s*\((?P<country>[A-Z]{2,4})\)\s*(?P<day>\d{1,2})(?:ST|ND|RD|TH)?\s+"
                     r"(?P<mon>[A-Z]{3})", re.I)
PREFIX_RE = re.compile(r"^[A-Z]{2,4}\s*/\s*", re.I)
NAME_RE = re.compile(r"^\s*R(?P<race>\d+)\s+(?P<dist>\d+)\s*m\s*(?P<type>.*)$", re.I)
ANZ_LISTING = "https://betfair-datascientists.github.io/data/dataListing/"
ANZ_ASSETS = "https://betfair-datascientists.github.io/data/assets/"


# ---------------------------------------------------------------------------------------------------------- parsing

def parse_hint(hint) -> dict | None:
    """'WAGGA (AUS) 28TH FEB' -> {'track': 'Wagga', 'country': 'AUS', 'day': 28, 'month': 2}; None when unreadable."""
    m = HINT_RE.match(str(hint or ""))
    if not m or m.group("mon").upper() not in MONTHS:
        return None
    track = m.group("track").strip()
    pre = PREFIX_RE.match(track)                      # some files write 'AUS / Flem (AUS) 2nd Nov': a short name
    if pre:
        track = track[pre.end():]
    return {"track": track.strip().title(), "country": m.group("country").upper(),
            "day": int(m.group("day")), "month": MONTHS[m.group("mon").upper()], "abbrev": bool(pre)}


def parse_event_name(name) -> dict:
    """'R3 1200m Hcap' -> {'race_no': 3, 'distance_m': 1200, 'race_type': 'Hcap', 'harness': False}."""
    s = str(name or "")
    m = NAME_RE.match(s)
    out = {"race_no": np.nan, "distance_m": np.nan, "race_type": None, "harness": bool(config.HARNESS_RE.search(s))}
    if m:
        out.update(race_no=int(m.group("race")), distance_m=int(m.group("dist")),
                   race_type=m.group("type").strip() or None)
    return out


def event_time(event_dt) -> datetime | None:
    """The price file's EVENT_DT, 'DD-MM-YYYY HH:MM' (betfair_prices.bf_event_time's format), as a naive datetime."""
    try:
        return datetime.strptime(str(event_dt or "").strip()[:16], "%d-%m-%Y %H:%M")
    except ValueError:
        return None


def meeting_date(day: int, month: int, when: datetime | None) -> date | None:
    """The meeting's own date: the day and month in its name, the year from the event time. An Australian meeting
    on 1 Jan runs on 31 Dec in the UK, and the reverse cannot happen, but both turns of the year are handled."""
    if when is None:
        return None
    year = when.year
    if month == 1 and when.month == 12:
        year += 1
    elif month == 12 and when.month == 1:
        year -= 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def cloth_and_name(selection_name) -> tuple[float, str]:
    """'3. Mahmoa' -> (3, 'Mahmoa'); a name without a number keeps NaN."""
    s = str(selection_name or "").strip()
    m = re.match(r"^(\d+)\.\s*(.*)$", s)
    return (float(m.group(1)), m.group(2).strip()) if m else (np.nan, s)


# ------------------------------------------------------------------------------------------------------------ build

def build_targets(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The raw price-file rows (KEEP columns, as strings, plus FILE) -> the targets table and its data report."""
    import betfair_prices as bp                              # import only: the UK/IE module is not changed

    rep: dict = {"rows_in": int(len(raw))}
    d = raw.copy()
    hints = d["MENU_HINT"].map(parse_hint)
    names = d["EVENT_NAME"].map(parse_event_name)
    d["track"] = [h["track"] if h else None for h in hints]
    d["country"] = [h["country"] if h else None for h in hints]
    d["track_abbrev"] = [bool(h and h["abbrev"]) for h in hints]
    rep["unreadable_hint_examples"] = (d.loc[hints.isna(), "MENU_HINT"].astype(str).value_counts().head(10)
                                       .to_dict())
    d["harness"] = [n["harness"] for n in names]
    when = d["EVENT_DT"].map(event_time)
    d["meeting_date"] = [meeting_date(h["day"], h["month"], w) if h else None for h, w in zip(hints, when)]
    d["off_uk"] = [w.strftime("%Y-%m-%d %H:%M") if w else None for w in when]
    d["race_no"] = [n["race_no"] for n in names]
    d["distance_m"] = [n["distance_m"] for n in names]
    d["race_type"] = [n["race_type"] for n in names]
    cn = d["SELECTION_NAME"].map(cloth_and_name)
    d["cloth"] = [c for c, _ in cn]
    d["horse"] = [n for _, n in cn]
    d["horse_norm"] = d["horse"].map(bp.normalise_horse)
    d["market_id"] = pd.to_numeric(d["EVENT_ID"], errors="coerce").astype("Int64")
    d["selection_id"] = pd.to_numeric(d["SELECTION_ID"], errors="coerce").astype("Int64")
    d["bsp"] = pd.to_numeric(d["BSP"], errors="coerce")
    d["won"] = pd.to_numeric(d["WIN_LOSE"], errors="coerce")
    d["pp_vol"] = pd.to_numeric(d["PPTRADEDVOL"], errors="coerce")
    d["ip_vol"] = pd.to_numeric(d["IPTRADEDVOL"], errors="coerce")
    d["source_file"] = d["FILE"]

    dropped = {}

    def drop(mask, why):
        dropped[why] = int(mask.sum())
        return d[~mask]

    d = drop(d["market_id"].isna() | d["selection_id"].isna(), "no market or selection id")
    d = drop(d["country"].isna(), "meeting name unreadable (no '(AUS)')")
    d = drop(d["country"] != config.COUNTRY, "not Australian (country in the meeting name)")
    d = drop(d["harness"], "harness (Pace/Trot)")
    d = drop(d["meeting_date"].isna(), "no meeting date")
    dup = d.duplicated(["market_id", "selection_id"], keep="first")
    d = drop(dup, "duplicate (market, selection): first kept")
    rep["dropped"] = dropped
    cols = ["market_id", "selection_id", "horse", "horse_norm", "cloth", "track", "track_abbrev", "meeting_date",
            "race_no",
            "distance_m", "race_type", "off_uk", "bsp", "won", "pp_vol", "ip_vol", "source_file"]
    t = d[cols].copy()
    t["race_family"] = t["race_type"].map(config.race_family)
    t["meeting_date"] = pd.to_datetime(t["meeting_date"]).dt.date.astype(str)
    t = t.sort_values(["meeting_date", "off_uk", "market_id", "selection_id"]).reset_index(drop=True)
    rep["rows_out"] = int(len(t))
    rep["markets"] = int(t["market_id"].nunique())
    rep["date_range"] = [t["meeting_date"].min(), t["meeting_date"].max()] if len(t) else None
    rep["no_bsp"] = int((~(t["bsp"] > 1)).sum())
    rep["no_result"] = int((~t["won"].isin([0, 1])).sum())
    rep["nulls"] = {c: int(t[c].isna().sum()) for c in ("horse_norm", "track", "race_no", "distance_m", "off_uk")}
    w = t[t["won"] == 1].groupby("market_id").size()
    rep["markets_winners"] = {"one": int((w == 1).sum()), "dead_heat": int((w > 1).sum()),
                              "none": int(t["market_id"].nunique() - len(w))}
    by_year = t.assign(y=t["meeting_date"].str[:4]).groupby("y").agg(
        races=("market_id", "nunique"), runners=("selection_id", "size"), days=("meeting_date", "nunique"))
    rep["by_year"] = {y: {k: int(v) for k, v in r.items()} for y, r in by_year.iterrows()}
    rep["rows_in_minus_out_equals_dropped"] = rep["rows_in"] - rep["rows_out"] == sum(dropped.values())
    return t, rep


# --------------------------------------------------------------------------------------------------------- the reads

def archive_keys(s3, bucket: str, market: str = config.PRICE_FILE_MARKET) -> list[tuple[date, str]]:
    """Every archived file of the market, by day, oldest first."""
    import betfair_prices as bp
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{bp.S3_PREFIX}/"):
        for o in page.get("Contents", []):
            got = bp.parse_listed_name(o["Key"].rsplit("/", 1)[-1])
            if got and got[0] == market:
                out.append((got[1], o["Key"]))
    return sorted(out)


def read_csv_bytes(raw: bytes) -> pd.DataFrame | None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    if not text.strip():
        return pd.DataFrame(columns=KEEP)
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df[KEEP] if all(c in df.columns for c in KEEP) else None


def read_archive(s3, bucket: str, limit: int | None = None, workers: int = 24) -> tuple[pd.DataFrame, dict]:
    if s3 is None:                                      # our own client, its pool as large as the workers
        import os

        import boto3
        from botocore.config import Config
        s3 = boto3.client("s3", region_name=os.getenv("CAPTURE_REGION") or "eu-west-2",
                          config=Config(max_pool_connections=workers + 4, retries={"max_attempts": 5}))
    keys = archive_keys(s3, bucket)
    if limit:
        keys = keys[-limit:]

    def one(item):
        _, key = item
        try:
            df = read_csv_bytes(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        except Exception as exc:                        # a bad file is counted and named, never fatal
            return key, None, str(exc)[:120]
        if df is None:
            return key, None, "columns missing"
        return key, df.assign(FILE=key.rsplit("/", 1)[-1]), None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(one, keys))
    bad = [(k, e) for k, df, e in got if df is None]
    frames = [df for _, df, _ in got if df is not None]
    rep = {"files": len(keys), "unreadable": len(bad), "unreadable_examples": bad[:10],
           "empty": sum(1 for _, df, _ in got if df is not None and df.empty),
           "first_file_day": keys[0][0].isoformat() if keys else None,
           "last_file_day": keys[-1][0].isoformat() if keys else None}
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP + ["FILE"])
    return raw, rep


ANZ_COLS = ["LOCAL_MEETING_DATE", "SCHEDULED_RACE_TIME", "TRACK", "STATE_CODE", "RACE_NO", "WIN_MARKET_ID",
            "DISTANCE", "SELECTION_ID", "WIN_RESULT", "WIN_BSP"]


def read_anz(session=None) -> tuple[pd.DataFrame, list[str]]:
    """Betfair's ANZ thoroughbred files, read in memory (never unpacked to disk): every file the listing names."""
    import requests
    s = session or requests.Session()
    page = s.get(ANZ_LISTING, timeout=60)
    page.raise_for_status()
    names = sorted(set(re.findall(r"assets/(ANZ_Thoroughbreds_[0-9_]+\.(?:zip|csv))", page.text)))
    frames = []
    for n in names:
        r = s.get(ANZ_ASSETS + n, timeout=300)
        r.raise_for_status()
        blobs = []
        if n.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                blobs = [z.read(m) for m in z.namelist() if m.lower().endswith(".csv")]
        else:
            blobs = [r.content]
        for b in blobs:
            frames.append(pd.read_csv(io.BytesIO(b), dtype=str, keep_default_na=False, usecols=ANZ_COLS))
    return pd.concat(frames, ignore_index=True), names


def anz_check(t: pd.DataFrame, anz: pd.DataFrame) -> dict:
    """The targets against Betfair's ANZ files, over the dates both hold (AU only: the ANZ NZ rows are left out)."""
    a = anz[anz["STATE_CODE"] != "NZ"].copy()
    raw = a["LOCAL_MEETING_DATE"].str.strip()
    a["date"] = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce"))
    a["market_id"] = pd.to_numeric(a["WIN_MARKET_ID"], errors="coerce").astype("Int64")
    a["selection_id"] = pd.to_numeric(a["SELECTION_ID"], errors="coerce").astype("Int64")
    a = a.dropna(subset=["date", "market_id", "selection_id"]).drop_duplicates(["market_id", "selection_id"])
    lo, hi = max(a["date"].min(), pd.Timestamp(t["meeting_date"].min())), min(
        a["date"].max(), pd.Timestamp(t["meeting_date"].max()))
    a = a[(a["date"] >= lo) & (a["date"] <= hi)]
    tt = t[(pd.to_datetime(t["meeting_date"]) >= lo) & (pd.to_datetime(t["meeting_date"]) <= hi)]
    j = tt.merge(a, on=["market_id", "selection_id"], how="inner")
    rep = {"window": [str(lo.date()), str(hi.date())], "targets_rows": int(len(tt)), "anz_rows": int(len(a)),
           "matched_rows": int(len(j)),
           "targets_found_in_anz": round(len(j) / max(len(tt), 1), 4),
           "anz_found_in_targets": round(len(j) / max(len(a), 1), 4),
           "markets_targets": int(tt["market_id"].nunique()), "markets_anz": int(a["market_id"].nunique()),
           "markets_both": int(j["market_id"].nunique())}
    bsp_a = pd.to_numeric(j["WIN_BSP"], errors="coerce")
    both = (j["bsp"] > 1) & (bsp_a > 1)
    rep["bsp_compared"] = int(both.sum())
    rep["bsp_equal_within_0.01"] = round(float(((j["bsp"] - bsp_a).abs() <= 0.01)[both].mean()), 4) if both.any() else None
    won_a = (j["WIN_RESULT"] == "WINNER").astype(int)
    res = j["won"].isin([0, 1])
    rep["result_agrees"] = round(float((j["won"][res] == won_a[res]).mean()), 4) if res.any() else None
    rep["race_no_agrees"] = round(float((j["race_no"] == pd.to_numeric(j["RACE_NO"], errors="coerce")).mean()), 4)
    rep["distance_agrees"] = round(float((j["distance_m"] == pd.to_numeric(j["DISTANCE"], errors="coerce")).mean()), 4)
    norm = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)  # noqa: E731
    rep["track_agrees"] = round(float((norm(j["track"]) == norm(j["TRACK"])).mean()), 4)
    rep["track_disagreements_top"] = (j.loc[norm(j["track"]) != norm(j["TRACK"]), ["track", "TRACK"]]
                                      .drop_duplicates().head(15).values.tolist())
    rep["meeting_date_agrees"] = round(float((pd.to_datetime(j["meeting_date"]) == j["date"]).mean()), 4)
    # the clock: the ANZ scheduled time on its own clock, less the price file's EVENT_DT, in hours
    anz_t = pd.to_datetime(j["date"].dt.strftime("%Y-%m-%d") + " " + j["SCHEDULED_RACE_TIME"].str[:5], errors="coerce")
    off = (anz_t - pd.to_datetime(j["off_uk"], errors="coerce")).dt.total_seconds() / 3600
    rep["clock_offset_hours_top"] = {str(k): int(v) for k, v in off.round(1).value_counts().head(8).items()}
    return rep


def learn_track_names(t: pd.DataFrame, anz: pd.DataFrame) -> tuple[dict, dict]:
    """Betfair's short meeting names ('Flem') to the ANZ files' full ones ('Flemington'), learned from the runners
    both hold (matched on market and selection): each short name takes the full name most of its rows carry."""
    a = anz[["WIN_MARKET_ID", "TRACK"]].drop_duplicates("WIN_MARKET_ID").copy()
    a["market_id"] = pd.to_numeric(a["WIN_MARKET_ID"], errors="coerce").astype("Int64")
    m = (t[t["track_abbrev"]][["market_id", "track"]].drop_duplicates("market_id")
         .merge(a[["market_id", "TRACK"]], on="market_id", how="inner"))
    out, shares = {}, []
    for short, g in m.groupby("track"):
        vc = g["TRACK"].value_counts()
        out[short] = vc.index[0]
        shares.append((short, vc.index[0], int(len(g)), round(float(vc.iloc[0] / len(g)), 4)))
    shaky = [s for s in shares if s[3] < 0.99]
    rep = {"short_names_learned": len(out), "markets_used": int(len(m)),
           "short_names_under_99pct_agreement": shaky[:20]}
    return out, rep


def apply_track_names(t: pd.DataFrame, names_map: dict) -> tuple[pd.DataFrame, dict]:
    t = t.copy()
    short = t["track_abbrev"].astype(bool)
    full = t["track"].map(names_map)
    ok = short & full.notna()
    t.loc[ok, "track"] = full[ok]
    left = t.loc[short & ~ok, "track"]
    rep = {"rows_short": int(short.sum()), "rows_mapped": int(ok.sum()),
           "rows_left_short": int(len(left)), "short_names_left": left.value_counts().head(20).to_dict()}
    return t, rep


# -------------------------------------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="read the archive, build and store the targets")
    ap.add_argument("--anz-check", action="store_true", help="reconcile with Betfair's ANZ thoroughbred files")
    ap.add_argument("--root", help="a local store folder in place of S3 (sources.common.Store)")
    ap.add_argument("--limit", type=int, help="only the newest N archive files (a dry run)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    store = Store(root=args.root)
    report = {"built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if not args.build:
        ap.error("nothing to do: pass --build")
    raw, read_rep = read_archive(None, store.bucket, limit=args.limit)
    report["archive"] = read_rep
    t, build_rep = build_targets(raw)
    report["build"] = build_rep
    names_map = None
    if args.anz_check:
        anz, names = read_anz()
        report["anz_files"] = names
        names_map, report["track_names"] = learn_track_names(t, anz)
        store.put(config.TRACK_NAMES_KEY, json.dumps(names_map, indent=1, sort_keys=True).encode())
    else:
        held = store.get(config.TRACK_NAMES_KEY)
        names_map = json.loads(held) if held else None
    if names_map:
        t, report["track_names_applied"] = apply_track_names(t, names_map)
    if args.anz_check:
        report["anz_check"] = anz_check(t, anz)
    store.put_parquet(config.TARGETS_KEY, t)
    store.put(config.TARGETS_REPORT_KEY, json.dumps(report, indent=1, default=str).encode())
    report["written"] = [store.where(config.TARGETS_KEY), store.where(config.TARGETS_REPORT_KEY)]
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
