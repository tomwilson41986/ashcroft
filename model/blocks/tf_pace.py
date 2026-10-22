"""Timeform's pace readings of a horse's past races: how fast and how contested they were run, where the pace map put
the ideal position, and how the horse fared in each kind (the owner, 22 Oct: the Timeform database as a resource).

The readings are race-level (scripts/build_timeform_pace.py from the Timeform archives, data/external/timeform_pace.csv.gz,
UK/IE Flat and all-weather 2016 - Feb 2026): the pace's strength (0 Very Weak .. 5 Extreme) and contest (0 .. 3) as
Timeform read the race afterwards, the race's finishing speed % (the winner's closing sectional against its average
speed; a fast finish is a slow early pace) and the pre-race pace map's expected ideal early position (0 front .. 4
rear). A horse's past runs tell what it has met, and its finishing position against each tells what suits it; the
market may already know, or not.

    tp_pace_l1  tp_pace_m3  tp_pace_car      the pace strength of its last race, last 3, career
    tp_contest_m3                            how contested its last 3 races were
    tp_rfsp_l1  tp_rfsp_m3                   the finishing speed % of its last race, last 3
    tp_ideal_l1  tp_ideal_m3                 the pace map's ideal position in its last race, last 3
    tp_front_m3                              the pace map's front share in its last 3
    tp_pos_strong  tp_pos_weak               its finishing position as a share of the field (0 the winner) in
                                             strong-paced (3+) and weak-paced (0-1) races, career
    tp_pos_fastfin  tp_pos_slowfin           the same in fast-finishing (race_fsp >= 102) and slow-finishing
                                             (< 100) races, career
    tp_runs                                  earlier runs with a Timeform pace reading

Every window reads the horse's earlier racing days only. A run joins its reading by day and the normalised horse
name. A jumps run has no reading and leaves its window as it was.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from model.blocks.form_variants import _ladder, career_totals
from model.blocks.sectionals import norm_name
from model.freshness_features import _Days, _col
from model.race_shape import _codes, day_index

PREFIX = "tp_"
FEATURES = ["tp_pace_l1", "tp_pace_m3", "tp_pace_car", "tp_contest_m3", "tp_rfsp_l1", "tp_rfsp_m3", "tp_ideal_l1",
            "tp_ideal_m3", "tp_front_m3", "tp_pos_strong", "tp_pos_weak", "tp_pos_fastfin", "tp_pos_slowfin",
            "tp_runs"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "placing_numerical", "number_of_runners"]
READINGS = ["pace", "contest", "race_fsp", "ideal_exp", "ideal_front", "pos_share"]

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TABLE_PATH = os.path.join(_ROOT, "data", "external", "timeform_pace.csv.gz")
SOURCE = "table"
TEST_REACH = {"SOURCE": "synthetic"}


@lru_cache(maxsize=2)
def _table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["day", "name_norm", *READINGS])
    t = pd.read_csv(path, usecols=["day", "name_norm", *READINGS], dtype={"name_norm": str})
    return t.dropna(subset=["day", "name_norm"]).drop_duplicates(["day", "name_norm"], keep=False)


def _synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Made-up readings from the frame's results (block tests only)."""
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(dtype=float)
    keys = (df["raceid"].astype(str) + "|" + pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")).to_numpy()
    u = np.array([int(hashlib.md5(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for k in keys])   # per race
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where((pos > 0) & (n > 1), (pos - 1) / (n - 1), np.nan)
    return pd.DataFrame({"pace": np.floor(u * 6), "contest": np.floor(u * 4), "race_fsp": 96 + 8 * u,
                         "ideal_exp": 4 * u, "ideal_front": 0.4 * (1 - u), "pos_share": share}, index=df.index)


def readings(df: pd.DataFrame) -> pd.DataFrame:
    if SOURCE == "synthetic":
        return _synthetic(df)
    q = pd.DataFrame({"day": pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy(),
                      "name_norm": df["horse_name"].map(norm_name).to_numpy()})
    out = q.merge(_table(TABLE_PATH), on=["day", "name_norm"], how="left")[READINGS]
    out.index = df.index
    return out.apply(pd.to_numeric, errors="coerce")


def _cond_mean(H: _Days, value: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """The career mean of `value` over the key's earlier blocks where `cond` held."""
    w = np.where(np.isfinite(value) & cond, 1.0, 0.0)
    s, c = career_totals(H, np.where(w > 0, value, np.nan), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    horse = np.where(bad, -1, _codes(name))
    H = _Days(horse, day)
    r = readings(df)
    pb = {c: H.per_block(r[c].to_numpy(dtype=float)) for c in READINGS}

    cols: dict[str, np.ndarray] = {}
    lad = _ladder(H, pb["pace"])
    cols["tp_pace_l1"], cols["tp_pace_m3"], cols["tp_pace_car"] = (H.to_rows(lad[k]) for k in ("l1", "m3", "car"))
    cols["tp_runs"] = H.to_rows(lad["_cnt"])
    cols["tp_contest_m3"] = H.to_rows(_ladder(H, pb["contest"])["m3"])
    lad = _ladder(H, pb["race_fsp"])
    cols["tp_rfsp_l1"], cols["tp_rfsp_m3"] = H.to_rows(lad["l1"]), H.to_rows(lad["m3"])
    lad = _ladder(H, pb["ideal_exp"])
    cols["tp_ideal_l1"], cols["tp_ideal_m3"] = H.to_rows(lad["l1"]), H.to_rows(lad["m3"])
    cols["tp_front_m3"] = H.to_rows(_ladder(H, pb["ideal_front"])["m3"])
    pace, fsp, share = pb["pace"], pb["race_fsp"], pb["pos_share"]
    with np.errstate(invalid="ignore"):
        cols["tp_pos_strong"] = H.to_rows(_cond_mean(H, share, pace >= 3))
        cols["tp_pos_weak"] = H.to_rows(_cond_mean(H, share, pace <= 1))
        cols["tp_pos_fastfin"] = H.to_rows(_cond_mean(H, share, fsp >= 102))
        cols["tp_pos_slowfin"] = H.to_rows(_cond_mean(H, share, fsp < 100))

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
