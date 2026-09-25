"""Form in today's conditions: a horse's earlier runs at today's trip, going, course, race type and headgear, against its form everywhere.

The engine reads a horse's NFP at the distance, going and course
(dist_avg_nfp, going_avg_nfp, course_avg_nfp) and whether it has run in them.
This reads the measures the price model leans on most, the rating-scale
performance figure and the market's view of past runs (model/form_windows.py),
in five conditions, and how far a horse's form in the condition sits from its
form overall. Today's value of each condition is on the 06:00 card:

    trip     the distance band (race_shape.DIST_BANDS: to 6.5f, 8.5, 12.5, 16.5, 20.5, 24.5, beyond)
    going    firm or good to firm | good | good to soft or soft | heavy | all-weather
    course   the track
    hcap     a handicap (or nursery) or not, from the race type and name
    gear     wearing headgear or not

For each condition c and measure m (perf, nfp, mkt), over the horse's earlier
days in today's value of the condition, never the day itself:

    cf_<c>_n          earlier days in the condition
    cf_<c>_<m>_car    the mean over them
    cf_<c>_<m>_m3     the mean over the last three of them
    cf_<c>_<m>_d      that career mean less the horse's career mean everywhere,
                      shrunk towards 0 by n / (n + K), K = 3

NaN where the horse has no earlier day in the condition, or the condition is
unknown today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.draw_metrics import classify_going
from model.form_windows import run_measures, window_ladder
from model.freshness_features import _Days, _col
from model.race_shape import DIST_BANDS, _codes, bands, day_index

CONDITIONS = ("trip", "going", "course", "hcap", "gear")
MEASURES = ("perf", "nfp", "mkt")
STATS = ("car", "m3", "d")
FEATURES = [f"cf_{c}_n" for c in CONDITIONS] + [f"cf_{c}_{m}_{s}" for c in CONDITIONS for m in MEASURES
                                                for s in STATS]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners", "placing_numerical",
         "LB", "total_dst_bt", "dist_furlongs", "official_rating", "RSR", "bfsp", "going_description",
         "race_type", "race_name", "headgear"]
K = 3.0
GOING = {"Firm": 0, "GtF": 0, "Good": 1, "GtS": 2, "Soft": 2, "Heavy": 3, "AW": 4}
_HCAP_RX = r"handicap|nursery|\bh'cap\b|\bhcap\b"          # model/intent_features.py's


def conditions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each row's value of each condition: an integer from 0, or -1 if unknown."""
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(dtype=float)
    trip = bands(dist, DIST_BANDS)
    going = _col(df, "going_description").map(classify_going).map(GOING)
    track = _col(df, "track").fillna("").astype(str).str.strip().str.lower()
    course = np.where(track.isin(["", "nan", "none"]).to_numpy(), -1, _codes(track))
    text = (_col(df, "race_type").fillna("").astype(str) + " "
            + _col(df, "race_name").fillna("").astype(str)).str.lower()
    hg = _col(df, "headgear").fillna("").astype(str).str.strip().str.lower()
    out = {"trip": trip, "going": going.to_numpy(dtype=float), "course": course,
           "hcap": text.str.contains(_HCAP_RX).to_numpy().astype(float),
           "gear": (~hg.isin(["", "nan", "none"])).to_numpy().astype(float)}
    return {c: np.where(np.isfinite(v.astype(float)), v, -1).astype(np.int64) for c, v in out.items()}


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    horse = np.where(name.isin(["", "nan", "none"]).to_numpy(), -1, _codes(name))
    measures = run_measures(df)

    H = _Days(horse, day)
    overall = {m: H.to_rows(window_ladder(H, H.per_block(measures[m]))["car"]) for m in MEASURES}

    cols = {}
    for c, v in conditions(df).items():
        width = int(v.max()) + 1 if (v >= 0).any() else 1
        key = np.where((horse >= 0) & (v >= 0), horse * width + v, -1)
        C = _Days(key, day)
        n = C.to_rows(np.where(C.valid, C.pos, np.nan).astype(float))
        cols[f"cf_{c}_n"] = n
        for m in MEASURES:
            ladder = window_ladder(C, C.per_block(measures[m]))
            car = C.to_rows(ladder["car"])
            cols[f"cf_{c}_{m}_car"] = car
            cols[f"cf_{c}_{m}_m3"] = C.to_rows(ladder["m3"])
            with np.errstate(invalid="ignore"):
                cols[f"cf_{c}_{m}_d"] = (car - overall[m]) * n / (n + K)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
