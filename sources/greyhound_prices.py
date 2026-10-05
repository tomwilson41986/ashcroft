"""Betfair's greyhound price files (dwbfgreyhoundwin/place<DDMMYYYY>.csv, UK and Australia), read into a table a year:
the BSP, the morning and pre-play weighted average prices and their traded volumes, the in-play range, the result.

The files are fetched by the price-file archive on the UK server (betfair_prices.py --archive, into
s3://$CAPTURE_BUCKET/betfair_prices_raw/); this source only builds from them, so its ``fetch`` does nothing. The
build is incremental: a year's table is rebuilt only when the archive holds a file it was not built from
(``greyhound_prices/built_<year>.json`` lists them).

Table: ``greyhound_prices/prices_<year>.parquet``, one row a runner a market; ``trap`` and ``dog`` (the name without
its trap) are read from the selection name ("1. Goldcash Warrior"), ``track`` from the menu hint.
"""

from __future__ import annotations

import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from sources.common import Store, gunzip

log = logging.getLogger(__name__)

RAW_PREFIX = "betfair_prices_raw"
NAME_RE = re.compile(r"dwbfgreyhound(win|place)(\d{8})\.csv$", re.I)
NUMERIC = {"BSP": "bsp", "PPWAP": "ppwap", "MORNINGWAP": "morningwap", "PPMAX": "ppmax", "PPMIN": "ppmin",
           "IPMAX": "ipmax", "IPMIN": "ipmin", "MORNINGTRADEDVOL": "morning_vol", "PPTRADEDVOL": "pp_vol",
           "IPTRADEDVOL": "ip_vol", "WIN_LOSE": "win_lose"}


def fetch(store: Store, **kw) -> dict:
    return {"note": "the files are archived by betfair_prices.py --archive on the UK server; this source builds only"}


def _raw_listing(store: Store) -> dict[str, str]:
    """{file name: where to read it} for every greyhound price file the archive holds."""
    out = {}
    if store.root:                                              # tests and local runs: <root>/../betfair_prices_raw
        base = Path(store.root).parent / RAW_PREFIX
        if base.exists():
            for p in base.iterdir():
                if NAME_RE.search(p.name):
                    out[p.name] = str(p)
        return out
    for page in store.s3.get_paginator("list_objects_v2").paginate(Bucket=store.bucket, Prefix=f"{RAW_PREFIX}/dwbfgreyhound"):
        for o in page.get("Contents", []):
            name = o["Key"].rsplit("/", 1)[-1]
            if NAME_RE.search(name):
                out[name] = o["Key"]
    return out


def _read_raw(store: Store, where: str) -> bytes:
    if store.root:
        return Path(where).read_bytes()
    return store.s3.get_object(Bucket=store.bucket, Key=where)["Body"].read()


def file_day(name: str):
    m = NAME_RE.search(name)
    return (m.group(1).lower(), datetime.strptime(m.group(2), "%d%m%Y").date()) if m else (None, None)


def parse(raw: bytes, name: str) -> pd.DataFrame:
    """One day file -> its runners. Tolerant of the columns a year's files carry."""
    raw = gunzip(raw)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    if text.lstrip()[:1] == "<":                                # an HTML refusal stored by mistake
        return pd.DataFrame()
    d = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    d.columns = [c.strip().upper() for c in d.columns]
    market, day = file_day(name)
    out = pd.DataFrame({
        "market": market,
        "event_id": pd.to_numeric(d.get("EVENT_ID"), errors="coerce"),
        "menu_hint": d.get("MENU_HINT", pd.Series("", index=d.index)).str.strip(),
        "event_name": d.get("EVENT_NAME", pd.Series("", index=d.index)).str.strip(),
        "event_dt": d.get("EVENT_DT", pd.Series("", index=d.index)).str.strip(),
        "selection_id": pd.to_numeric(d.get("SELECTION_ID"), errors="coerce"),
        "selection_name": d.get("SELECTION_NAME", pd.Series("", index=d.index)).str.strip(),
    })
    for src, dst in NUMERIC.items():
        out[dst] = pd.to_numeric(d[src], errors="coerce") if src in d.columns else float("nan")
    dt = pd.to_datetime(out.event_dt, format="%d-%m-%Y %H:%M", errors="coerce")
    out["race_date"] = dt.dt.strftime("%Y-%m-%d").fillna(day.isoformat() if day else None)
    out["race_time"] = dt.dt.strftime("%H:%M")
    out["trap"] = pd.to_numeric(out.selection_name.str.extract(r"^(\d+)\.")[0], errors="coerce")
    out["dog"] = out.selection_name.str.replace(r"^\d+\.\s*", "", regex=True)
    # "Romford 3rd Oct" -> Romford; "AUS / Albion Park (AUS) 3rd Oct" -> Albion Park
    hint = out.menu_hint.str.replace(r"^[A-Z]{2,3}\s*/\s*", "", regex=True)
    out["track"] = hint.str.replace(r"\s*\([A-Z]{2,3}\)", "", regex=True) \
        .str.replace(r"\s+\d{1,2}(st|nd|rd|th)\s+\w+$", "", regex=True).str.strip()
    out["country"] = out.menu_hint.str.extract(r"\(([A-Z]{2,3})\)")[0].fillna("GB")
    out["source_file"] = name
    return out


def build(store: Store, years: list[int] | None = None) -> dict:
    files = _raw_listing(store)
    by_year: dict[int, list[str]] = {}
    for name in files:
        _, day = file_day(name)
        if day and (not years or day.year in years):
            by_year.setdefault(day.year, []).append(name)
    out = {}
    for y, names in sorted(by_year.items()):
        mkey = f"greyhound_prices/built_{y}.json"
        raw = store.get(mkey)
        built = set(json.loads(raw)) if raw else set()
        if set(names) <= built:
            out[y] = {"files": len(names), "unchanged": True}
            continue
        frames, bad = [], 0
        for name in sorted(names):
            try:
                frames.append(parse(_read_raw(store, files[name]), name))
            except Exception as exc:
                bad += 1
                log.warning("%s not read (%s)", name, exc)
        df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            df = df.drop_duplicates(["market", "event_id", "selection_id"], keep="last")
        store.put_parquet(f"greyhound_prices/prices_{y}.parquet", df)
        store.put(mkey, json.dumps(sorted(names)).encode())
        out[y] = {"files": len(names), "unreadable": bad, "runners": len(df),
                  "win_markets": int(df[df.market == "win"].event_id.nunique()) if len(df) else 0,
                  "gb_share": round(float((df.country == "GB").mean()), 3) if len(df) else 0}
        log.info("greyhound prices %d: %s", y, out[y])
    return out
