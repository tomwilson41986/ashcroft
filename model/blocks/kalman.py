"""A Kalman rating on the performance figure: ability as a state that drifts between runs, read with its uncertainty.

The form windows average a horse's figures over fixed windows; they cannot say how
sure they are, and a horse off for two hundred days reads as sure as one that ran
last week. A local-level filter (model/state_space.py, the framework's "highest-value
import") treats ability as a latent state that drifts, more over a long gap and faster
early in a career, and each run's performance figure (model/perf_figures.py, pounds on
the official-rating scale) as a noisy reading of it. The screen against the best
model's errors (research/queries/travel_residuals.py, 27 Sep) found the rating's
uncertainty and its gap to today's mark the readings the model lacks. Strictly from
earlier days: every run of a day is priced on the state before the day, and the day's
figures enter only afterwards.

    kr_rating    the rating before today (the prior mean)
    kr_sd        its standard deviation: wide for a lightly raced horse and after a gap
    kr_n         runs with a figure behind it
    kr_vs_or     kr_rating less today's official rating (NaN unrated)
    kr_z         kr_rating standardised within today's race
    kr_rank      kr_rating ranked in today's race (1 the highest)
    kr_vs_max    kr_rating less the race's highest

A horse's first rating is anchored on the median official rating of the race it first
runs in (on the card), with the filter's full initial variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import day_index, race_key

FEATURES = ["kr_rating", "kr_sd", "kr_n", "kr_vs_or", "kr_z", "kr_rank", "kr_vs_max"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "official_rating", "median_or", "total_dst_bt",
         "placing_numerical", "dist_furlongs"]
Q = 25.0                  # drift variance a year (lb^2)
R = 50.0                  # a run's reading variance (lb^2)
INIT_VAR = 100.0
Q_CAREER_MULT = 2.0       # faster drift over a horse's first runs
Q_CAREER_RUNS = 4
ANCHOR_FALLBACK = 70.0


def _name_codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _filter(order, h, day, perf, anchor):
    """Priors (mean, variance, readings) per row, rows visited horse by horse, day by day (`order`)."""
    n = len(h)
    pm, pv, cnt = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    cur_h, cur_day = None, None
    m = P = 0.0
    state_day = 0
    k = 0
    m0 = P0 = 0.0          # the prior of the current day
    pending: list[float] = []

    def settle():
        nonlocal m, P, k, state_day
        mm, pp = m0, P0
        for y in pending:
            s = pp + R
            g = pp / s
            mm, pp = mm + g * (y - mm), (1.0 - g) * pp
            k += 1
        m, P, state_day = mm, pp, cur_day

    for i in order:
        hh, dd = h[i], day[i]
        if hh != cur_h:
            cur_h, cur_day, pending = hh, dd, []
            m, P, state_day, k = anchor[i], INIT_VAR, dd, 0
            m0, P0 = m, P
        elif dd != cur_day:
            settle()
            cur_day, pending = dd, []
            q = Q * (Q_CAREER_MULT if k < Q_CAREER_RUNS else 1.0)
            m0, P0 = m, P + q * max((dd - state_day) / 365.0, 1.0 / 365.0)
        pm[i], pv[i], cnt[i] = m0, P0, k
        if np.isfinite(perf[i]):
            pending.append(float(perf[i]))
    return pm, pv, cnt


def build(df: pd.DataFrame) -> pd.DataFrame:
    from model.perf_figures import performance_figure_lbs
    n = len(df)
    day = day_index(df)
    h = _name_codes(df["horse_name"])
    race = pd.factorize(race_key(df), sort=True)[0]
    src = df[[c for c in READS if c in df.columns]].copy()
    src["raceid"] = race_key(df).to_numpy()
    perf = performance_figure_lbs(src).to_numpy(float)              # post-race: enters later days only
    orr = pd.to_numeric(df["official_rating"], errors="coerce").where(lambda s: s > 0)
    med = pd.to_numeric(df["median_or"], errors="coerce").where(lambda s: s > 0)
    card_med = orr.groupby(race).transform("median")
    anchor = med.fillna(card_med).fillna(ANCHOR_FALLBACK).to_numpy(float)
    order = np.lexsort((race, day, h))
    order = order[h[order] >= 0]
    pm, pv, cnt = _filter(order, h, day, perf, anchor)
    rating = pd.Series(pm, index=df.index)
    g = rating.groupby(race)
    sd_race = g.transform("std").replace(0, np.nan)
    new = pd.DataFrame({
        "kr_rating": pm,
        "kr_sd": np.sqrt(pv),
        "kr_n": np.where(h >= 0, cnt, np.nan),
        "kr_vs_or": pm - orr.to_numpy(float),
        "kr_z": ((rating - g.transform("mean")) / sd_race).to_numpy(float),
        "kr_rank": g.rank(ascending=False, method="min").to_numpy(float),
        "kr_vs_max": (rating - g.transform("max")).to_numpy(float),
    }, index=df.index)[FEATURES]
    assert len(new) == n
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
