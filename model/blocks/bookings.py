"""Bookings and the market's confidence in the connections: who rides today, and how the yard's runners are being backed.

The engine rates jockeys and trainers by their results (wins, NFP, WAX). It
never asks how the market has rated their runners, which is a different
reading: a jockey whose mounts go off short is the one yards book when they
mean business, and a yard whose runners have been backed lately is one the
market thinks is in form. A booking is also a choice made for this horse: a
jockey stronger than the horse's usual riders says something about today.
All from EARLIER DAYS, on the market's view of each past run (the run's BSP
normalised in its race: model/form_windows.run_measures "mkt", 0 for a runner
at the field's average price, positive for one the market backed):

    bk_jk_mkt            the jockey's rides as the market rated them, decayed
                         (half-life 180 days) and shrunk to 0 over 30 rides
    bk_jk_upgrade        bk_jk_mkt today less the mean of the same reading for
                         the jockeys of the horse's last three runs, each as it
                         stood on that run's day
    bk_jk_same           1 when today's jockey rode the horse's last run
    bk_jk_rides_on_horse how many of the horse's earlier runs today's jockey rode
    bk_tr_mkt            the trainer's runners as the market rated them, decayed
                         (half-life 365 days), shrunk to 0 over 50 runners
    bk_tr_mkt_trend      the last fortnight's reading (half-life 10 days, shrunk
                         to the long one over 10 runners) less the long one

The jockey is the row's jockey: as booked on the 06:00 card, as ridden in
history, the same reading the engine's jockey features take.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from model.form_windows import run_measures
from model.race_shape import _codes, asof_decayed_mean, day_index
from model.shrinkage import shrunk_mean

FEATURES = ["bk_jk_mkt", "bk_jk_upgrade", "bk_jk_same", "bk_jk_rides_on_horse", "bk_tr_mkt", "bk_tr_mkt_trend"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "jockey_name", "trainer", "number_of_runners",
         "placing_numerical", "bfsp"]


def _key(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s))


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    mkt = run_measures(df)["mkt"]
    jk = _key(df["jockey_name"]) if "jockey_name" in df.columns else np.full(n, -1)
    tr = _key(df["trainer"]) if "trainer" in df.columns else np.full(n, -1)
    horse = _key(df["horse_name"])

    m, c = asof_decayed_mean(jk, day, mkt, jk, day, halflife_days=180.0)
    jk_mkt = np.where(jk >= 0, shrunk_mean(np.nan_to_num(m) * c, c, 0.0, 30), np.nan)

    # the horse's earlier runs, in day order: their jockeys' readings as they stood then
    h = pd.DataFrame({"horse": horse, "day": day, "jk": jk, "q": jk_mkt, "i": np.arange(n)})
    h = h[h["horse"] >= 0].sort_values(["horse", "day", "i"], kind="mergesort")
    g = h.groupby("horse", sort=False)
    prev = []
    for k in (1, 2, 3):
        q = g["q"].shift(k)
        same_day = g["day"].shift(k).to_numpy() == h["day"].to_numpy()
        prev.append(np.where(same_day, np.nan, q.to_numpy(dtype=float)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # a horse's first run has no earlier jockeys
        prev_mean = np.nanmean(np.vstack(prev), axis=0) if len(h) else np.array([])
    last_jk = g["jk"].shift(1).to_numpy(dtype=float)
    last_day = g["day"].shift(1).to_numpy(dtype=float)
    last_jk = np.where(last_day == h["day"].to_numpy(), np.nan, last_jk)
    # rides of this horse by today's jockey on earlier rows (a horse runs once a day)
    rides = h.groupby(["horse", "jk"], sort=False).cumcount().to_numpy(dtype=float)
    rides = np.where(h["jk"].to_numpy() >= 0, rides, np.nan)

    upgrade = np.full(n, np.nan)
    same = np.full(n, np.nan)
    on_horse = np.full(n, np.nan)
    idx = h["i"].to_numpy()
    upgrade[idx] = h["q"].to_numpy() - prev_mean
    same[idx] = np.where(np.isfinite(last_jk) & (h["jk"].to_numpy() >= 0),
                         (last_jk == h["jk"].to_numpy()).astype(float), np.nan)
    on_horse[idx] = rides

    # the trainer's runners as the market rated them, long and recent
    tm, tc = asof_decayed_mean(tr, day, mkt, tr, day, halflife_days=365.0)
    tr_long = shrunk_mean(np.nan_to_num(tm) * tc, tc, 0.0, 50)
    sm, sc = asof_decayed_mean(tr, day, mkt, tr, day, halflife_days=10.0)
    tr_short = shrunk_mean(np.nan_to_num(sm) * sc, sc, tr_long, 10)
    cols = {
        "bk_jk_mkt": jk_mkt,
        "bk_jk_upgrade": upgrade,
        "bk_jk_same": same,
        "bk_jk_rides_on_horse": on_horse,
        "bk_tr_mkt": np.where(tr >= 0, tr_long, np.nan),
        "bk_tr_mkt_trend": np.where(tr >= 0, tr_short - tr_long, np.nan),
    }
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
