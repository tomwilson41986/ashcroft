"""Hourly weather at the greyhound tracks: NOAA's Integrated Surface Database, the ISD-Lite files (US government data,
public domain: free for any use, commercial included; https://www.ncei.noaa.gov/products/land-based-station/
integrated-surface-database). Chosen over Open-Meteo, whose free tier is for non-commercial use only.

For each track in the reference table (research/greyhound/reference/gb_tracks.csv: coordinates with their sources)
the nearest stations within ``MAX_KM`` that report in the years wanted (from NOAA's station list, isd-history.csv);
each station-year file is fetched once (the current year again each run) and kept raw in the sources store
(``weather/isd/<usaf>-<wban>/<year>.gz``); the table ``weather/isd_hourly.parquet`` holds every hour: UTC time, air
temperature and dew point (deg C), sea-level pressure (hPa), wind direction (deg) and speed (m/s), precipitation in
the hour (mm, often unreported at UK stations). ``weather/track_stations.json`` maps each track to its stations,
nearest first, with the distance.

    python -m sources.noaa_isd --fetch --build --from 2014
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import math
from datetime import date
from pathlib import Path

import pandas as pd

from sources.common import Store, get, session

log = logging.getLogger(__name__)

HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
LITE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{y}/{s}-{y}.gz"
TRACKS = Path(__file__).resolve().parent.parent / "research" / "greyhound" / "reference" / "gb_tracks.csv"
MAX_KM = 60.0
PER_TRACK = 3
COLS = ["year", "month", "day", "hour", "temp", "dewp", "slp", "wdir", "wspd", "sky", "p1", "p6"]


def km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) *
         math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def tracks(path: Path = TRACKS) -> list[dict]:
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("latitude") and r.get("longitude")]


def stations(raw: bytes) -> pd.DataFrame:
    st = pd.read_csv(io.BytesIO(raw), dtype=str)
    st = st[st.CTRY.isin(["UK", "EI"])].copy()
    st["lat"] = pd.to_numeric(st.LAT, errors="coerce")
    st["lon"] = pd.to_numeric(st.LON, errors="coerce")
    st["begin"] = pd.to_numeric(st.BEGIN.str[:4], errors="coerce")
    st["end"] = pd.to_numeric(st.END.str[:4], errors="coerce")
    st["id"] = st.USAF + "-" + st.WBAN
    return st.dropna(subset=["lat", "lon"])


def assign(track_rows: list[dict], st: pd.DataFrame, first_year: int) -> dict:
    """Each track's stations reporting from ``first_year`` (or later) to now, within MAX_KM, nearest first."""
    out = {}
    cur = date.today().year
    for t in track_rows:
        lat, lon = float(t["latitude"]), float(t["longitude"])
        d = st.assign(km=[km(lat, lon, a, b) for a, b in zip(st.lat, st.lon)])
        d = d[(d.km <= MAX_KM) & (d.end >= min(cur - 1, max(first_year, 2014)))].sort_values("km")
        out[t["gbgb_name"]] = [{"id": r.id, "name": r["STATION NAME"], "km": round(float(r.km), 1),
                                "begin": int(r.begin), "end": int(r.end)} for _, r in d.head(PER_TRACK).iterrows()]
    return out


def parse(raw: bytes, sid: str) -> pd.DataFrame:
    text = gzip.decompress(raw).decode("ascii", "replace")
    rows = [line.split() for line in text.splitlines() if line.strip()]
    df = pd.DataFrame([r[:12] for r in rows if len(r) >= 12], columns=COLS).apply(pd.to_numeric, errors="coerce")
    df = df.replace(-9999, float("nan"))
    ts = pd.to_datetime(df[["year", "month", "day", "hour"]], utc=True, errors="coerce")
    return pd.DataFrame({"station": sid, "ts": ts, "temp_c": df.temp / 10, "dewp_c": df.dewp / 10,
                         "slp_hpa": df.slp / 10, "wdir_deg": df.wdir, "wspd_ms": df.wspd / 10,
                         "precip1h_mm": df.p1 / 10}).dropna(subset=["ts"])


def raw_key(sid: str, y: int) -> str:
    return f"weather/isd/{sid}/{y}.gz"


def fetch(store: Store, first_year: int = 2014, s=None) -> dict:
    s = s or session()
    st = stations(get(s, HISTORY).content)
    tr = tracks()
    plan = assign(tr, st, first_year)
    store.put("weather/track_stations.json", json.dumps(plan, indent=1).encode())
    have = set(store.listing("weather/isd/"))
    cur = date.today().year
    got, missing = 0, []
    for sid in sorted({x["id"] for v in plan.values() for x in v}):
        for y in range(first_year, cur + 1):
            k = raw_key(sid, y)
            if k in have and y < cur:
                continue
            try:
                r = s.get(LITE.format(y=y, s=sid), timeout=60)
                if r.status_code == 404:
                    missing.append(f"{sid}/{y}")
                    continue
                r.raise_for_status()
                store.put(k, r.content)
                got += 1
            except Exception as exc:
                log.warning("%s %d not fetched (%s)", sid, y, exc)
                missing.append(f"{sid}/{y}")
    return {"tracks": len(tr), "tracks_with_station": sum(bool(v) for v in plan.values()),
            "stations": len({x["id"] for v in plan.values() for x in v}), "files_fetched": got,
            "not_found": len(missing)}


def build(store: Store) -> dict:
    keys = [k for k in store.listing("weather/isd/") if k.endswith(".gz")]
    frames = [parse(store.get(k), k.split("/")[2]) for k in sorted(keys)]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df = df.drop_duplicates(["station", "ts"], keep="last").sort_values(["station", "ts"])
    store.put_parquet("weather/isd_hourly.parquet", df)
    return {"files": len(keys), "hours": int(len(df)), "stations": int(df.station.nunique()) if len(df) else 0,
            "from": str(df.ts.min()) if len(df) else None, "to": str(df.ts.max()) if len(df) else None}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--from", dest="first_year", type=int, default=2014)
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    rep = {}
    if a.fetch:
        rep["fetch"] = fetch(store, a.first_year)
    if a.build:
        rep["build"] = build(store)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
