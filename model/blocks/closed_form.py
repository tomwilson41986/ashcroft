"""Closed-form metrics: any within-race metric, credited for the field it was earned against, solved for everyone at once.

The owner, 12 Nov 2026: "think about min-max on every feature / metric we have." A finishing-position metric says how
a runner did against ITS field: fifth at Royal Ascot and fifth at Hamilton read the same. The iterated ("closed
form") version is the one fixed point of

    rating = the mean, over its runs, of (how far above the field's average it ran) + (the field's average rating)

which is solved here for every entity together, on the races of the WINDOW days before each snapshot (every
SNAP_DAYS; a race weighted 2^(-age / HALF); each rating shrunk to zero by K runs), so a row reads only earlier
days. With n runners, a run d above its field's mean, and the rivals' mean rating f:  rating <- f + d x n / (n - 1).

The same engine for four metrics and four kinds of entity:

    metric   n   normalised finishing position (a non-finisher 0)
             c   the capped margin (1 the winner, falling by tanh to 0 at about fifteen lengths)
             t   the time figure, in pounds (model/blocks/time_figure.py)
             x   the log of the runner's share of the Betfair SP book: the market's own view, closed the same way
    entity   h the horse, tr the trainer, jk the jockey, sr the sire

    cf_<m>_<e>          the rating (0 for an entity with no runs in the window; NaN where it is not named)
    cf_<m>_h_z          the horse's rating standardised within today's race
    cf_<m>_field        the mean of today's field's horse ratings: the race's real class on that metric
    cf_<m>_field_l1     the same for the horse's last race, less today's: a rise or a drop in real class
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import time_figure as _tf
from model.freshness_features import _Days, _col
from model.race_shape import day_index, race_key

METRICS = ("n", "c", "t", "x")
ENTITIES = (("h", "horse_name"), ("tr", "trainer"), ("jk", "jockey_name"), ("sr", "stallion"))
FEATURES = ([f"cf_{m}_{e}" for m in METRICS for e, _ in ENTITIES] + [f"cf_{m}_h_z" for m in METRICS]
            + [f"cf_{m}_field" for m in METRICS] + [f"cf_{m}_field_l1" for m in METRICS])
POST_RACE: set[str] = set()
READS = sorted(set(_tf.READS) | {"trainer", "jockey_name", "stallion", "number_of_runners", "bfsp"})
SNAP_DAYS = 30
WINDOW = 3 * 365
HALF = 365.0
K = 5.0
ITERS = 40
DAMP = 0.5
LIMIT = 15.0
TEST_REACH = {"SNAP_DAYS": 5, "WINDOW": 60, "HALF": 30.0, "ITERS": 20}


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def solve(race, ent, y, w, size):
    """The fixed point, by damped iteration. race: contiguous codes; ent: entity codes (-1 none); y: the metric
    (NaN: no reading); w: row weights. Returns the rating of each entity code below `size`."""
    use = np.isfinite(y) & (ent >= 0)
    theta = np.zeros(size)
    if not use.any():
        return theta
    r, e, v, ww = race[use], ent[use], y[use], w[use]
    nr = int(r.max()) + 1
    cnt = np.bincount(r, minlength=nr).astype(float)
    many = cnt[r] >= 2
    r, e, v, ww = r[many], e[many], v[many], ww[many]
    if not len(r):
        return theta
    cnt = np.bincount(r, minlength=nr).astype(float)
    mean = np.bincount(r, weights=v, minlength=nr) / np.where(cnt > 0, cnt, 1.0)
    d = (v - mean[r]) * cnt[r] / (cnt[r] - 1.0)
    den = np.bincount(e, weights=ww, minlength=size) + K
    for _ in range(ITERS):
        tot = np.bincount(r, weights=theta[e], minlength=nr)
        f = (tot[r] - theta[e]) / (cnt[r] - 1.0)
        new = np.bincount(e, weights=ww * (f + d), minlength=size) / den
        theta = DAMP * theta + (1.0 - DAMP) * new
    return theta


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    rs = pd.Series(race, index=df.index)
    codes = {e: _codes(_col(df, col)) for e, col in ENTITIES}
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    lb = pd.to_numeric(_col(df, "total_dst_bt"), errors="coerce").to_numpy(float)
    fin = np.isfinite(pos) & (pos >= 1)
    lb = np.where(pos == 1, 0.0, np.where(fin & np.isfinite(lb), np.maximum(lb, 0.0), np.nan))
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=df.index)
    tot = inv.groupby(rs).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        metric = {
            "n": np.where(fin & (n > 1), np.clip((n - pos) / (n - 1), 0.0, 1.0), np.where(n > 1, 0.0, np.nan)),
            "c": np.where(fin, 1.0 - np.tanh(np.where(np.isfinite(lb), lb, LIMIT * 3) / LIMIT), np.where(n > 1, 0.0, np.nan)),
            "t": np.asarray(_tf.figures(df)["fig"], dtype=float),
            "x": np.where(priced & (tot > 0), np.log(inv.to_numpy() / np.where(tot > 0, tot, 1.0)), np.nan)}

    out = {(m, e): np.full(len(df), np.nan) for m in METRICS for e, _ in ENTITIES}
    ok = day >= 0
    if ok.any():
        d0, d1 = int(day[ok].min()), int(day[ok].max())
        sizes = {e: int(c.max()) + 1 if len(c) and c.max() >= 0 else 1 for e, c in codes.items()}
        for snap in range(d0 + SNAP_DAYS, d1 + 1, SNAP_DAYS):
            tr = np.flatnonzero(ok & (day < snap) & (day >= snap - WINDOW))
            te = ok & (day >= snap) & (day < snap + SNAP_DAYS)
            if not len(tr) or not te.any():
                continue
            r = pd.factorize(race[tr], sort=True)[0].astype(np.int64)
            w = np.power(2.0, -(snap - day[tr]) / HALF)
            for m in METRICS:
                for e, _ in ENTITIES:
                    theta = solve(r, codes[e][tr], metric[m][tr], w, sizes[e])
                    k = codes[e][te]
                    out[(m, e)][te] = np.where(k >= 0, theta[np.where(k >= 0, k, 0)], np.nan)

    cols = {f"cf_{m}_{e}": out[(m, e)] for m in METRICS for e, _ in ENTITIES}
    H = _Days(codes["h"], day)
    prev = np.clip(np.arange(len(H.u)) - 1, 0, None)
    has_prev = H.valid & (H.pos >= 1)
    for m in METRICS:
        s = pd.Series(out[(m, "h")], index=df.index)
        g = s.groupby(rs)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = ((s - g.transform("mean")) / g.transform("std")).where(g.transform("count") >= 3)
        cols[f"cf_{m}_h_z"] = z.replace([np.inf, -np.inf], np.nan).to_numpy(float)
        field = g.transform("mean").to_numpy(float)
        cols[f"cf_{m}_field"] = field
        fb = H.per_block(field)
        cols[f"cf_{m}_field_l1"] = H.to_rows(np.where(has_prev, fb[prev], np.nan)) - field
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
