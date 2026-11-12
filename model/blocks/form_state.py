"""Form as a state: a Kalman rating on the time figure, and a Markov chain on whether the horse is in form.

Two of the models the owner asked to be tried (12 Nov 2026: "Kalman filters ... Markov chains"):

1. The Kalman rating (model/blocks/kalman.py) filters the performance figure, which is built from the official
   ratings and distances beaten. The clock's figure (model/blocks/time_figure.py) has never been filtered: its
   windows are plain means. Here it is a local-level state with the same filter (drift a year, faster over the
   first runs, a reading's noise), started at the yard's runners' mean figure on earlier days.

       tk_rating  tk_sd  tk_n     the rating before today, its standard deviation, the readings behind it
       tk_z  tk_vs_max            the rating within today's race: standardised, and against the best

2. A two-state Markov chain: a horse is in form or out of it. A run's result against the price (s: its normalised
   finishing position less the position its Betfair SP rank implied) is a noisy reading of the state (in form:
   mean +MU, out: -MU, standard deviation SIGMA). The state persists between runs with probability
   0.5 + STAY x exp(-days / MEMORY_DAYS), so a long break forgets it. The forward filter gives, before today:

       fs_p_form      the probability the horse is in form
       fs_p_form_z    that standardised within today's race
       fs_run         consecutive earlier runs with the probability above (+) or below (-) one half

Fixed constants, set before the first screen. Earlier days only: a day's runs enter after its features are read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import kalman as _k
from model.blocks import time_figure as _tf
from model.freshness_features import _Days, _col
from model.race_shape import asof_decayed_mean, day_index, race_key

FEATURES = ["tk_rating", "tk_sd", "tk_n", "tk_z", "tk_vs_max", "fs_p_form", "fs_p_form_z", "fs_run"]
POST_RACE: set[str] = set()
READS = sorted(set(_tf.READS) | {"trainer", "number_of_runners", "bfsp"})
ANCHOR = -15.0
MU, SIGMA, STAY, MEMORY_DAYS = 0.08, 0.30, 0.35, 180.0


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    h = _k._name_codes(df["horse_name"])
    race = pd.factorize(race_key(df), sort=True)[0]
    rs = pd.Series(race, index=df.index)
    fig = np.asarray(_tf.figures(df)["fig"], dtype=float)
    tr = _k._name_codes(_col(df, "trainer"))
    yard, cnt = asof_decayed_mean(tr, day, fig, tr, day, halflife_days=730.0)
    anchor = np.where(np.isfinite(yard) & (cnt >= 5), yard, ANCHOR)
    order = np.lexsort((race, day, h))
    order = order[h[order] >= 0]
    pm, pv, n = _k._filter(order, h, day, fig, anchor)
    rating = pd.Series(np.where(n > 0, pm, np.nan), index=df.index)
    g = rating.groupby(rs)
    cols = {"tk_rating": rating.to_numpy(float), "tk_sd": np.where(n > 0, np.sqrt(pv), np.nan),
            "tk_n": np.where(h >= 0, n, np.nan),
            "tk_z": ((rating - g.transform("mean")) / g.transform("std").replace(0, np.nan)).to_numpy(float),
            "tk_vs_max": (rating - g.transform("max")).to_numpy(float)}

    # the Markov chain on form
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    nr = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    rk = pd.Series(np.where(priced, bsp, np.nan), index=df.index).groupby(rs).rank(method="average").to_numpy()
    npr = pd.Series(priced.astype(float), index=df.index).groupby(rs).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.where(np.isfinite(pos) & (pos >= 1) & (nr > 1), np.clip((nr - pos) / (nr - 1), 0.0, 1.0),
                       np.where(nr > 1, 0.0, np.nan))
        s = nfp - np.where(priced & (npr > 1), (npr - rk) / (npr - 1), np.nan)
    H = _Days(h, day)
    sb = H.per_block(s)
    nb = len(H.u)
    prior, run = np.full(nb, 0.5), np.zeros(nb)
    post, streak = np.full(nb, 0.5), np.zeros(nb)
    for p in range(int(H.pos.max()) + 1 if nb else 0):
        i = np.flatnonzero(H.valid & (H.pos == p))
        if not len(i):
            break
        if p > 0:
            gap = np.maximum(H.day[i] - H.day[i - 1], 1)
            stay = 0.5 + STAY * np.exp(-gap / MEMORY_DAYS)
            prior[i] = stay * post[i - 1] + (1.0 - stay) * (1.0 - post[i - 1])
            run[i] = streak[i - 1]
        y = sb[i]
        ok = np.isfinite(y)
        lg = np.exp(-0.5 * ((np.where(ok, y, 0.0) - MU) / SIGMA) ** 2)
        lb = np.exp(-0.5 * ((np.where(ok, y, 0.0) + MU) / SIGMA) ** 2)
        up = prior[i] * lg / (prior[i] * lg + (1.0 - prior[i]) * lb)
        post[i] = np.where(ok, up, prior[i])
        sign = np.sign(post[i] - 0.5)
        streak[i] = np.where(sign == np.sign(run[i]), run[i] + sign, sign)
    pf = pd.Series(H.to_rows(np.where(H.valid & (H.pos > 0), prior, np.nan)), index=df.index)
    gp = pf.groupby(rs)
    cols["fs_p_form"] = pf.to_numpy(float)
    cols["fs_p_form_z"] = ((pf - gp.transform("mean")) / gp.transform("std").replace(0, np.nan)).to_numpy(float)
    cols["fs_run"] = H.to_rows(np.where(H.valid & (H.pos > 0), run, np.nan))
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
