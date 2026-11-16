"""Time figures by projection: the horse, the meeting and the race fitted together, on earlier races only.

The time figure (model/blocks/time_figure.py) takes the day's going allowance from how the day's winners ran against
the course's standard, and reads a run's distance behind with a hard cap. Two faults follow: a card of good horses
makes the ground look fast, and a slowly run race marks every runner down. The remedy the figure-makers use is
projection: set the allowance and judge the pace by what the field was entitled to run (the owner, 12 Nov 2026:
"continue with the time figures by projection"; kind C of the closed-form metrics).

A run's raw figure r is the time figure as it stands, but with the distance behind squashed (LIMIT x tanh(lengths /
LIMIT): nothing counts beyond about LIMIT lengths). On the races of the WINDOW days before each snapshot (every
SNAP_DAYS), three things are fitted together by alternating shrunk means until they settle:

    r = the horse's level + the meeting's correction x the race's scale + the race's own effect + noise

    the horse's level        weighted to its recent runs (2^(-age / HALF)), shrunk to the population's by K_H runs
    the meeting's correction to the going allowance, in seconds a furlong (the scale turns it into pounds for the
                             race's trip and speed), shrunk to zero - the allowance as it stands - by K_M runners
    the race's effect        how far the whole field ran below or above itself (a slowly run race), shrunk to zero
                             by K_R runners

and a run's projected figure is r less its meeting's correction and its race's effect. A row reads the fit of the
last snapshot before its day, so nothing of its own day or later; its figures are up to SNAP_DAYS old. One level a
horse over the window: an improver's early meetings are read a little high (a first version).

    tp_h  tp_h_n          the horse's level, and the weighted runs behind it
    tp_l1  tp_m3  tp_best the projected figure of its last run, the mean of its last three, its best in the window
    tp_z  tp_gap  tp_field   the level within today's race: standardised, against the best rival, the field's mean
    tp_vs_tf              the level less the decayed mean of its plain time figures: what projection changes
    tp_l1_race            the race effect of its last race (below zero: slowly run, the figure understated it)
    tp_l1_meet            the meeting's correction at its last race, in pounds
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import time_figure as _tf
from model.freshness_features import _col
from model.perf_figures import lbs_per_length
from model.race_shape import asof_decayed_mean, day_index, race_key

FEATURES = ["tp_h", "tp_h_n", "tp_l1", "tp_m3", "tp_best", "tp_z", "tp_gap", "tp_field", "tp_vs_tf", "tp_l1_race",
            "tp_l1_meet"]
POST_RACE: set[str] = set()
READS = list(_tf.READS)
SNAP_DAYS = 30
WINDOW = 540
HALF = 240.0
K_H, K_M, K_R = 3.0, 20.0, 6.0
ITERS = 15
LIMIT = 15.0
TEST_REACH = {"SNAP_DAYS": 5, "WINDOW": 60, "HALF": 30.0}


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def raw_figures(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(the raw figure with the squashed margin, the race's scale: pounds for a second a furlong)."""
    f = _tf.figures(df)
    race = pd.factorize(race_key(df), sort=True)[0]
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    t = pd.to_numeric(_col(df, "comptime_numeric"), errors="coerce").to_numpy(float)
    t = pd.Series(np.where(t > 0, t, np.nan)).groupby(race).transform("min").to_numpy(float)
    lpl = np.asarray(lbs_per_length(np.nan_to_num(dist, nan=8.0)), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = dist * (dist * _tf.FURLONG_METRES / t) / _tf.LENGTH_METRES * lpl
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    fin = np.isfinite(pos) & (pos > 0)
    if "LB" in df.columns:
        lb = pd.to_numeric(df["LB"], errors="coerce").to_numpy(float)
    else:
        from model.perf_figures import parse_beaten_lengths
        lb = _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(float)
    lb = np.where(pos == 1, 0.0, lb)
    wt = pd.to_numeric(_col(df, "pounds"), errors="coerce").to_numpy(float)
    wadj = np.where(np.isfinite(wt), wt - _tf.WEIGHT_BASE, 0.0)
    r = np.where(fin & np.isfinite(lb), f["race_fig"] - LIMIT * np.tanh(np.maximum(lb, 0.0) / LIMIT) * lpl + wadj, np.nan)
    return r, np.where(np.isfinite(scale), scale, np.nan)


def fit(h, m, rc, r, s, w):
    """Alternating shrunk means. h, m, rc: contiguous horse, meeting and race codes; r: raw figure; s: scale;
    w: recency weight. Returns (level by horse, weight by horse, correction by meeting, effect by race, the mean)."""
    nh, nm, nrc = int(h.max()) + 1, int(m.max()) + 1, int(rc.max()) + 1
    mu = float(np.average(r, weights=w))
    hw = np.bincount(h, weights=w, minlength=nh)
    theta = (np.bincount(h, weights=w * r, minlength=nh) + K_H * mu) / (hw + K_H)
    delta, pi = np.zeros(nm), np.zeros(nrc)
    s2 = float(np.mean(s ** 2))
    cnt = np.bincount(rc, minlength=nrc).astype(float)
    for _ in range(ITERS):
        res = r - theta[h] - pi[rc]
        delta = np.bincount(m, weights=s * res, minlength=nm) / (np.bincount(m, weights=s * s, minlength=nm) + K_M * s2)
        res = r - theta[h] - delta[m] * s
        pi = np.bincount(rc, weights=res, minlength=nrc) / (cnt + K_R)
        pf = r - delta[m] * s - pi[rc]
        mu = float(np.average(pf, weights=w))
        theta = (np.bincount(h, weights=w * pf, minlength=nh) + K_H * mu) / (hw + K_H)
    return theta, hw, delta, pi, mu


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    rs = pd.Series(race, index=df.index)
    horse = _codes(df["horse_name"])
    track = _codes(_col(df, "track"))
    r, s = raw_figures(df)
    out = {c: np.full(len(df), np.nan) for c in ("h", "h_n", "l1", "m3", "best", "l1_race", "l1_meet")}
    ok = day >= 0
    usable = ok & np.isfinite(r) & np.isfinite(s) & (horse >= 0) & (track >= 0)
    if usable.any():
        d0, d1 = int(day[ok].min()), int(day[ok].max())
        for snap in range(d0 + SNAP_DAYS, d1 + 1, SNAP_DAYS):
            tr = np.flatnonzero(usable & (day < snap) & (day >= snap - WINDOW))
            te = np.flatnonzero(ok & (day >= snap) & (day < snap + SNAP_DAYS))
            if len(tr) < 50 or not len(te):
                continue
            hu, h = np.unique(horse[tr], return_inverse=True)
            m = pd.factorize(track[tr].astype(np.int64) * 100000 + day[tr], sort=True)[0]
            rc = pd.factorize(race[tr], sort=True)[0]
            w = np.power(2.0, -(snap - day[tr]) / HALF)
            theta, hw, delta, pi, _ = fit(h, m, rc, r[tr], s[tr], w)
            pf = r[tr] - delta[m] * s[tr] - pi[rc]
            t = pd.DataFrame({"h": h, "d": day[tr], "pf": pf, "pi": pi[rc], "dm": delta[m] * s[tr]}).sort_values(["h", "d"])
            g = t.groupby("h", sort=True)
            last = g.tail(1).set_index("h")
            m3 = g["pf"].rolling(3, min_periods=1).mean().groupby(level=0).tail(1).droplevel(1)
            best = g["pf"].max()
            where = np.searchsorted(hu, horse[te])
            where = np.clip(where, 0, len(hu) - 1)
            seen = (horse[te] >= 0) & (hu[where] == horse[te])
            out["h"][te] = np.where(seen, theta[where], np.nan)
            out["h_n"][te] = np.where(seen, hw[where], np.nan)
            for name, ser in (("l1", last["pf"]), ("m3", m3), ("best", best), ("l1_race", last["pi"]), ("l1_meet", last["dm"])):
                v = ser.reindex(range(len(hu))).to_numpy(float)
                out[name][te] = np.where(seen, v[where], np.nan)

    lvl = pd.Series(out["h"], index=df.index)
    g = lvl.groupby(rs)
    top = g.transform("max")
    second = lvl.where(lvl < top).groupby(rs).transform("max")
    ties = (lvl == top).groupby(rs).transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((lvl - g.transform("mean")) / g.transform("std")).where(g.transform("count") >= 3)
    plain, _ = asof_decayed_mean(horse, day, np.asarray(_tf.figures(df)["fig"], dtype=float), horse, day, halflife_days=365.0)
    cols = {"tp_h": out["h"], "tp_h_n": out["h_n"], "tp_l1": out["l1"], "tp_m3": out["m3"], "tp_best": out["best"],
            "tp_z": z.replace([np.inf, -np.inf], np.nan).to_numpy(float),
            "tp_gap": np.where(g.transform("count") >= 2, lvl - np.where((lvl == top) & (ties < 2), second, top), np.nan),
            "tp_field": g.transform("mean").to_numpy(float), "tp_vs_tf": out["h"] - plain,
            "tp_l1_race": out["l1_race"], "tp_l1_meet": out["l1_meet"]}
    new = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
