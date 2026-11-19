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


#: a market that is not the plain win market of a race: forecasts, "AvB" matches, "without the favourite" books
VARIANT_RE = re.compile(r"\((?:F/?C|AvB0?|W/O|R/F|T/C|OvE|Dist)\)?|without\s+favs?|forecast|reverse|\bavb\b|"
                        r"trap\s+challenges?|antepost|televised\s+cards", re.I)
#: the abbreviated track labels of the older files (2014-2022: "BAGS cards / BVue", "SIS / Harl", "RPGTV / Crayfd"),
#: mapped to GBGB's names; learnt from the files by date and off time against the GBGB results, 2014-2022 (research/
#: greyhound/join_diag.py --aliases, run 44341763541: at least 20 markets and 80% on one track), and a few
#: unambiguous abbreviations whose vote is thinned by other tracks' races at the same minute (the join still needs the
#: date, off time and trap to agree). Irish (Shelb), Australian, NZ and non-track labels are left unmapped
TRACK_ALIASES: dict[str, str] = {
    "BVue": "Belle Vue",  # learnt: 9,670 markets, 90.8%
    "CPark": "Central Park",  # learnt: 13,459 markets, 92.2%
    "Crayfd": "Crayford",  # learnt: 27,299 markets, 82.1%
    "Donc": "Doncaster",  # learnt: 8,550 markets, 98.6%
    "HGrn": "Hall Green",  # learnt: 3,975 markets, 99.4%
    "Harl": "Harlow",  # learnt: 10,011 markets, 98.4%
    "Henl": "Henlow",  # learnt: 15,626 markets, 98.5%
    "Kinsl": "Kinsley",  # learnt: 12,615 markets, 99.1%
    "Monm": "Monmore",  # learnt: 24,237 markets, 92.8%
    "Newc": "Newcastle",  # learnt: 21,642 markets, 89.1%
    "Nott": "Nottingham",  # learnt: 14,718 markets, 93.7%
    "PBarr": "Perry Barr",  # learnt: 13,394 markets, 99.2%
    "PBoro": "Peterborough",  # learnt: 47 markets, 91.5%
    "PGran": "Star Pelaw",  # learnt: 1,462 markets, 99.9%
    "Pboro": "Peterborough",  # learnt: 5,104 markets, 92.0%
    "Pelaw Grange": "Star Pelaw",  # learnt: 847 markets, 99.9%
    "Romfd": "Romford",  # learnt: 32,549 markets, 81.0%
    "Sheff": "Sheffield",  # learnt: 18,627 markets, 93.0%
    "Sund": "Sunderland",  # learnt: 19,730 markets, 88.1%
    "Swin": "Swindon",  # learnt: 13,434 markets, 98.6%
    "Yarm": "Yarmouth",  # learnt: 7,094 markets, 97.6%
    "Cpark": "Central Park",  # by name; learnt top track the same, 10 markets, 100.0%
    "Crayf": "Crayford",  # by name; learnt top track the same, 21 markets, 57.1%
    "Notts": "Nottingham",  # by name; learnt top track the same, 6 markets, 100.0%
    "Pbarr": "Perry Barr",  # by name; learnt top track the same, 12 markets, 100.0%
    "Romf": "Romford",  # by name; learnt top track the same, 26 markets, 50.0%
    "Shaw": "Shawfield",  # by name; learnt top track the same, 3 markets, 66.7%
    "Shawf": "Shawfield",  # by name; learnt top track the same, 421 markets, 39.0%
    "Sittb": "Central Park",  # by name; learnt top track the same, 5,251 markets, 73.5%
    "Towc": "Towcester",  # by name; learnt top track the same, 9,411 markets, 64.7%
    "Wdon": "Wimbledon",  # by name; learnt top track the same, 9,977 markets, 34.4%
}


def market_label(menu_hint: str) -> tuple[str, str, str]:
    """A menu hint -> (track label, variant tag or "", country). "BAGS cards / BVue (AvB) 31st Dec" -> ("BVue", "AvB",
    "GB"); "AUS / Albion Park (AUS) 3rd Oct" -> ("Albion Park", "", "AUS"); "Cann (Aus) 31st Dec" -> ("Cann", "", "AUS")."""
    h = re.sub(r"\s+", " ", str(menu_hint or "")).strip()
    h = re.sub(r"\s*\d{1,2}\s*(st|nd|rd|th)\s*[A-Za-z]+$", "", h)             # the date suffix ("31st Dec", "31stDec")
    m = VARIANT_RE.search(h)
    variant = re.sub(r"[()]", "", m.group(0)).strip() if m else ""
    h = VARIANT_RE.sub("", h)                                                    # before the prefix: "(F/C)" has a "/"
    c = re.search(r"\((AUS|NZ|IRE|IRL|USA)\)", h, re.I)
    country = c.group(1).upper() if c else "GB"
    h = re.sub(r"\((AUS|NZ|IRE|IRL|USA)\)", "", h, flags=re.I)
    h = h.split("/")[-1]                                                         # the channel: "SIS / Harl" -> "Harl"
    return re.sub(r"\s+", " ", h).strip(" -"), variant, country


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
    # a few files (2015-17 most) carry a runner or race name with an unquoted comma: that row, not the file, is lost
    d = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, on_bad_lines="skip")
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
    # "Romford 3rd Oct" -> Romford; "AUS / Albion Park (AUS) 3rd Oct" -> Albion Park; "SIS / Harl 31st Dec" -> Harlow
    # (TRACK_ALIASES); a forecast, AvB or without-the-favourite book is not the race's win market: market "<m>_variant"
    lab = pd.DataFrame([market_label(h) for h in out.menu_hint], columns=["label", "variant", "country"], index=out.index)
    out["track"] = lab.label.map(lambda x: TRACK_ALIASES.get(x, x))
    out["variant"] = lab.variant
    out["country"] = lab.country
    out.loc[lab.variant != "", "market"] = market + "_variant" if market else "variant"
    out["source_file"] = name
    return out


def _force_years(env=None) -> set[int]:
    """GREYHOUND_PRICES_REBUILD ("2014-2022" or "2015,2018"): years rebuilt even when every file was read before (a
    change to ``parse``)."""
    import os
    v = str((os.environ if env is None else env).get("GREYHOUND_PRICES_REBUILD", "")).strip()
    if not v:
        return set()
    if "-" in v:
        lo, hi = (int(x) for x in v.split("-"))
        return set(range(lo, hi + 1))
    return {int(x) for x in v.split(",") if x.strip()}


def build(store: Store, years: list[int] | None = None, force_years: set[int] | None = None) -> dict:
    force = _force_years() if force_years is None else set(force_years)
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
        if set(names) <= built and y not in force:
            out[y] = {"files": len(names), "unchanged": True}
            continue
        frames, bad, read = [], 0, []
        for name in sorted(names):
            try:
                frames.append(parse(_read_raw(store, files[name]), name))
                read.append(name)
            except Exception as exc:
                bad += 1
                log.warning("%s not read (%s)", name, exc)
        df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            df = df.drop_duplicates(["market", "event_id", "selection_id"], keep="last")
        store.put_parquet(f"greyhound_prices/prices_{y}.parquet", df)
        store.put(mkey, json.dumps(sorted(read)).encode())        # a file not read is tried again next build
        out[y] = {"files": len(names), "unreadable": bad, "runners": len(df),
                  "win_markets": int(df[df.market == "win"].event_id.nunique()) if len(df) else 0,
                  "gb_share": round(float((df.country == "GB").mean()), 3) if len(df) else 0}
        log.info("greyhound prices %d: %s", y, out[y])
    return out
