"""Learned forgiveness: the market's verdict on each past run, set against what the result alone would have earned.

The closing crowd knows excuses that reach no database (the vet's and the jockey's reports, the replay, the yard's
health). Its next price shows which runs it believed (reports/discovery_1111.md 4.3). The market rating
(model/blocks/market_rating.py) leaves on each row `mr_surprise`: the last reading less the rating before it, which
is the market's update after the run before the last one. Here that update is set against the update a result of
that kind usually earns:

    s_j     the run's result surprise: its normalised finishing position (a non-finisher 0) less the mean position
            of earlier runners at its price (model/blocks/_resid.py; the first version's rank-implied position
            leaned with the price, iteration 118)
    a_j     the market's update after run j less BETA x s_j: above zero, the market forgave (or liked) the run more
            than its result said; below, it held the run against the horse more

BETA (by pool, the Flat and jumps) and the weight scale C = 1 / sd(a) are fitted on runs before FIT_UNTIL only, so
they are the same in every fold and on the live path. a_j needs the price of run j + 1, so it exists for every run
but the horse's last: nothing of today's enters.

    fg_a_l1  fg_a_m3  fg_a_m5  fg_a_car     a of the latest judged run (the one before the last), the mean of the
                                            last 3 and 5 judged runs, career
    fg_nfp_w5  fg_s_w5                      finishing position and result surprise over the last 5 judged runs,
                                            each weighted exp(C x a): the form the market believed
    fg_nfp_gap                              fg_nfp_w5 less the plain mean of the same runs
    fg_last_s                               the last run's result surprise (its verdict is not in yet)
    fg_last_pred                            the verdict forecast for it: BETA x s plus the horse's career mean a

Earlier racing days only. A horse's readings live in two pools; a verdict is read only where the judged run, the
next run and the row that carries the update are in one pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks._resid import price_residual
from model.freshness_features import _Days, _col
from model.race_shape import _codes, day_index, race_code, race_key

FEATURES = ["fg_a_l1", "fg_a_m3", "fg_a_m5", "fg_a_car", "fg_nfp_w5", "fg_s_w5", "fg_nfp_gap", "fg_last_s",
            "fg_last_pred"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "race_type", "surface_type", "placing_numerical",
         "number_of_runners", "bfsp", "mr_surprise"]
AFTER = ("market_rating",)
FIT_UNTIL = "2021-01-01"
MIN_FIT = 5000
DEFAULT_BETA, DEFAULT_C = 0.5, 2.0
WINDOW = 5


def _shift(H: _Days, v: np.ndarray, k: int) -> np.ndarray:
    """The value k of the key's blocks earlier (NaN where there is none)."""
    if k == 0:
        return np.where(H.valid, v, np.nan)
    idx = np.clip(np.arange(len(H.u)) - k, 0, None)
    return np.where(H.valid & (H.pos >= k), v[idx], np.nan)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    H = _Days(np.where(bad, -1, _codes(name)), day)

    race = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    s_row, nfp = price_residual(df, day, race)

    s = H.per_block(s_row)
    f = H.per_block(nfp)
    upd = H.per_block(pd.to_numeric(_col(df, "mr_surprise"), errors="coerce").to_numpy(float))
    pool = H.per_block(race_code(df).isin(["hurdle", "chase", "nhflat"]).to_numpy().astype(float))
    same = (pool == _shift(H, pool, 1)) & (pool == _shift(H, pool, 2))
    # at block b: the market's update after run b-2, and that run's result surprise
    u = np.where(same, upd, np.nan)
    s2, f2 = _shift(H, s, 2), _shift(H, f, 2)

    cut = int(np.datetime64(FIT_UNTIL, "D").astype(np.int64))
    beta = np.full(2, DEFAULT_BETA)
    fit = np.isfinite(u) & np.isfinite(s2) & (H.day < cut) & H.valid
    for p in (0, 1):
        m = fit & (pool == p)
        if m.sum() >= MIN_FIT and np.var(s2[m]) > 0:
            beta[p] = float(np.cov(s2[m], u[m])[0, 1] / np.var(s2[m]))
    b_blk = beta[np.where(np.isfinite(pool), pool, 0).astype(int)]
    a = u - b_blk * s2                                   # a of run b-2, known at block b
    c = 1.0 / float(np.std(a[fit])) if fit.sum() >= MIN_FIT and np.std(a[fit]) > 0 else DEFAULT_C

    def mean_last(k):
        st = np.stack([_shift(H, a, j) for j in range(k)])
        cnt = np.isfinite(st).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, np.nansum(st, axis=0) / cnt, np.nan)

    key = pd.Series(H.key)
    ok = np.isfinite(a)
    cs = pd.Series(np.where(ok, a, 0.0)).groupby(key).cumsum().to_numpy()
    cn = pd.Series(ok.astype(float)).groupby(key).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        car = np.where(H.valid & (cn > 0), cs / cn, np.nan)

    wsum = fsum = ssum = fplain = cnt = 0.0
    for j in range(WINDOW):
        aj, fj, sj = _shift(H, a, j), _shift(H, f2, j), _shift(H, s2, j)
        g = np.isfinite(aj) & np.isfinite(fj) & np.isfinite(sj)
        w = np.where(g, np.exp(np.clip(c * np.where(g, aj, 0.0), -3.0, 3.0)), 0.0)
        wsum, cnt = wsum + w, cnt + g
        fsum, ssum = fsum + w * np.where(g, fj, 0.0), ssum + w * np.where(g, sj, 0.0)
        fplain = fplain + np.where(g, fj, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp_w = np.where(wsum > 0, fsum / wsum, np.nan)
        s_w = np.where(wsum > 0, ssum / wsum, np.nan)
        plain = np.where(cnt > 0, fplain / np.where(cnt > 0, cnt, 1.0), np.nan)
    last_s = _shift(H, s, 1)
    cols = {"fg_a_l1": a, "fg_a_m3": mean_last(3), "fg_a_m5": mean_last(5), "fg_a_car": car, "fg_nfp_w5": nfp_w,
            "fg_s_w5": s_w, "fg_nfp_gap": nfp_w - plain, "fg_last_s": last_s,
            "fg_last_pred": _shift(H, b_blk, 1) * last_s + np.where(np.isfinite(car), car, 0.0)}
    new = pd.DataFrame({k: H.to_rows(np.where(H.valid, v, np.nan)) for k, v in cols.items()}, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
