"""The weather at the off and what it does to a track: the nearest NOAA station's hour (sources/noaa_isd.py), and a
learnt reading of how the weather moves each track's times.

Each race's off (UK local time, as GBGB gives it) in UTC is matched to its track's stations' readings within 90
minutes, the nearest station first. The columns: temperature, relative humidity (from the dew point), wind speed,
wind as east-west and north-south components, pressure, rain in the hour. The learnt part: for each track, a ridge
regression of the meeting's variant (the engine's: how much faster or slower than standard its winners ran, in
seconds) on the wind components, their squares and the temperature, fitted on the years before the run's year only,
gives ``wx_pred_variant``: how fast tonight's weather should make the track run, in lengths.

The readings are observed at the off. Live, a forecast stands in for them; the card-safety note says so.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "wx_"
SOURCE = "NOAA ISD-Lite hourly station data (public domain; sources/noaa_isd.py), nearest station to each track"
CARD_UNSAFE = ("wx_temp", "wx_rh", "wx_wspd", "wx_u", "wx_v", "wx_slp", "wx_rain", "wx_pred_variant")
MIN_FIT = 300


def load_context(store) -> dict:
    hourly = store.get_parquet("weather/isd_hourly.parquet")
    raw = store.get("weather/track_stations.json")
    return {"weather_hourly": hourly, "track_stations": json.loads(raw) if raw else {}}


def rh(temp: pd.Series, dewp: pd.Series) -> pd.Series:
    a, b = 17.625, 243.04
    return 100 * np.exp(a * dewp / (b + dewp)) / np.exp(a * temp / (b + temp))


def readings(runs: pd.DataFrame, hourly: pd.DataFrame, plan: dict) -> pd.DataFrame:
    """Each race's reading at the off from its track's stations, nearest first."""
    races = runs.drop_duplicates("race_id")[["race_id", "track", "t"]].copy()
    races["ts"] = (pd.to_datetime(races.t).dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
                   .dt.tz_convert("UTC"))
    races = races.dropna(subset=["ts"])
    out = []
    h = hourly.dropna(subset=["ts"]).sort_values("ts")
    for rank in range(3):
        st = {tr: v[rank]["id"] for tr, v in plan.items() if len(v) > rank}
        q = races.assign(station=races.track.map(st)).dropna(subset=["station"]).sort_values("ts")
        if not len(q):
            continue
        m = pd.merge_asof(q, h, on="ts", by="station", direction="nearest", tolerance=pd.Timedelta(minutes=90))
        out.append(m.assign(_rank=rank))
    if not out:
        return pd.DataFrame(columns=["race_id"])
    m = pd.concat(out, ignore_index=True)
    m = m[m.temp_c.notna() | m.wspd_ms.notna()].sort_values(["race_id", "_rank"]).drop_duplicates("race_id")
    return m


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    ctx = ctx or {}
    hourly, plan = ctx.get("weather_hourly"), ctx.get("track_stations") or {}
    out = pd.DataFrame({"race_id": runs.race_id, "dog_id": runs.dog_id})
    if hourly is None or not len(hourly) or not plan:
        for c in CARD_UNSAFE:
            out[c] = np.nan
        return out.reset_index(drop=True), {"note": "no data found: no weather table (run python -m sources.noaa_isd)"}
    r = readings(runs, hourly, plan)
    rad = np.deg2rad(r.wdir_deg)
    r["wx_temp"] = r.temp_c
    r["wx_rh"] = rh(r.temp_c, r.dewp_c)
    r["wx_wspd"] = r.wspd_ms
    r["wx_u"] = -r.wspd_ms * np.sin(rad)                      # the wind's direction is where it blows FROM
    r["wx_v"] = -r.wspd_ms * np.cos(rad)
    r["wx_slp"] = r.slp_hpa
    r["wx_rain"] = r.precip1h_mm
    # the learnt part: each track's meeting variant on the weather, fitted on earlier years
    df = figs if figs is not None else core.per_run_figures(runs)
    var = df.drop_duplicates("race_id")[["race_id", "track", "race_date", "variant"]]
    r = r.merge(var[["race_id", "variant", "race_date"]], on="race_id", how="left")
    r["year"] = pd.to_datetime(r.race_date).dt.year
    X = np.column_stack([np.ones(len(r)), r.wx_u.fillna(0), r.wx_v.fillna(0), r.wx_u.fillna(0) ** 2,
                         r.wx_v.fillna(0) ** 2, r.wx_temp.fillna(r.wx_temp.mean())])
    pred = np.full(len(r), np.nan)
    fits = {}
    yv = r.variant.to_numpy(float)
    for (track, year), idx in r.groupby(["track", "year"]).groups.items():
        tr = np.flatnonzero((r.track.to_numpy() == track) & (r.year.to_numpy() < year) & np.isfinite(yv)
                            & r.wx_wspd.notna().to_numpy())
        if len(tr) < MIN_FIT:
            continue
        A = X[tr]
        beta = np.linalg.solve(A.T @ A + 5.0 * np.diag([0, 1, 1, 1, 1, 1]), A.T @ np.clip(yv[tr], -1, 1))
        ii = r.index.get_indexer(idx)
        pred[ii] = -(X[ii] @ beta - beta[0]) / core.LENGTH_S     # lengths faster (+) than the track's usual
        fits[f"{track} {year}"] = int(len(tr))
    r["wx_pred_variant"] = pred
    cols = list(CARD_UNSAFE)
    out = out.merge(r[["race_id"] + cols], on="race_id", how="left")
    meta = {"races": int(runs.race_id.nunique()), "races_with_reading": int(r.race_id.nunique()),
            "track_years_fitted": len(fits)}
    return out.reset_index(drop=True), meta


COLUMNS = {
    "wx_temp": "air temperature at the off, deg C (nearest NOAA station; observed: live needs a forecast)",
    "wx_rh": "relative humidity at the off, %",
    "wx_wspd": "wind speed at the off, m/s",
    "wx_u": "wind towards the east, m/s",
    "wx_v": "wind towards the north, m/s",
    "wx_slp": "sea-level pressure, hPa",
    "wx_rain": "rain in the hour, mm (often unreported)",
    "wx_pred_variant": "how much faster (lengths) the track should run in this weather, learnt per track on earlier years",
}
