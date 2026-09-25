"""How exposed a horse is, how it has been improving, and how far horses like it go on improving.

The owner's question of 25 Sep: the relationship between how exposed or
unexposed a horse is and its improvement. A horse with three runs has shown a
fraction of what it can do; one with forty has shown it. So, on the
performance-figure scale (model/form_windows.py `perf`: the rating that day less
the pounds beaten, plus a winner's margin), from EARLIER DAYS only:

    ex_runs_code     earlier runs held in this code (flat, all-weather, hurdle, chase,
                     bumper), the exposure the figures below condition on
    ex_perf_best     the best figure so far; ex_perf_best3 the mean of the best three
                     of the last ten
    ex_perf_slope5   the least-squares slope of the last five figures, per run
    ex_perf_d1       the last figure less the one before
    ex_perf_d_car    the last figure less the career mean
    ex_gain_cell     how much horses at this exposure (career runs), age and code
                     went on to improve at their next run: the mean of (figure -
                     previous figure) over earlier days, decayed over two years
    ex_gain_trainer  the same for this trainer's horses at this exposure, shrunk to
                     the cell (k = 20 runs)
    ex_gain_sire     the same for this sire's progeny, shrunk to the cell (k = 20)
    ex_nfp_gain_cell the cell's mean improvement in NFP, for horses with no figure
    ex_headroom      best figure + the cell's expected gain - today's official rating
    ex_headroom_w5   the recency-weighted last five figures + the expected gain - rating

Today's official rating and career runs are on the 06:00 card. A run's own
figure never enters its own features: a card row and the same row with its
result in get the same values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.form_windows import run_measures
from model.freshness_features import _Days, _col
from model.race_shape import _codes, asof_decayed_mean, day_index, race_code
from model.shrinkage import shrunk_mean

FEATURES = [
    "ex_runs_code", "ex_perf_best", "ex_perf_best3", "ex_perf_slope5", "ex_perf_d1", "ex_perf_d_car",
    "ex_gain_cell", "ex_gain_trainer", "ex_gain_sire", "ex_nfp_gain_cell", "ex_headroom", "ex_headroom_w5",
]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "stallion", "horse_age",
         "career_runs", "race_type", "surface_type", "number_of_runners", "placing_numerical", "LB",
         "total_dst_bt", "dist_furlongs", "official_rating"]

#: Career-run buckets: the exposure a gain is conditioned on.
RUN_EDGES = [0, 1, 2, 3, 4, 5, 7, 10, 15, 25]
HALFLIFE_DAYS = 730.0
CONN_K = 20.0


def _num(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(_col(df, name), errors="coerce").to_numpy(dtype=float)


def _key(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s))


def _earlier(H: _Days, v: np.ndarray, j: int) -> np.ndarray:
    """The key's j-th block back (1 = the last day before), per block."""
    blocks = np.arange(len(H.u))
    return np.where(H.valid & (H.pos >= j), v[np.clip(blocks - j, 0, None)], np.nan)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    horse = _key(df["horse_name"])
    code = race_code(df).to_numpy()
    code_id = pd.factorize(code)[0].astype(np.int64)
    m = run_measures(df)
    perf, nfp = m["perf"], m["nfp"]

    # per horse and code: the figures of its earlier days in the code
    hc = np.where(horse >= 0, horse * 8 + code_id, -1)
    H = _Days(hc, day)
    pb = H.per_block(perf)
    runs_code = np.where(H.valid, H.pos, np.nan).astype(float)

    # the best figure over every earlier day, and the mean of the best three of the last ten
    lags10 = np.column_stack([_earlier(H, pb, j) for j in range(1, 11)])        # newest first
    lags = lags10[:, :5]
    key = pd.Series(H.key)
    prev = np.clip(np.arange(len(pb)) - 1, 0, None)
    has = H.valid & (H.pos >= 1)
    cm = pd.Series(np.where(np.isfinite(pb), pb, -np.inf)).groupby(H.key).cummax().to_numpy()
    best = np.where(has & np.isfinite(cm[prev]), cm[prev], np.nan)
    top = -np.sort(-np.where(np.isfinite(lags10), lags10, -np.inf), axis=1)[:, :3]
    held = np.isfinite(top)
    with np.errstate(invalid="ignore", divide="ignore"):
        best3 = np.where(held.any(1), np.where(held, top, 0.0).sum(1) / held.sum(1), np.nan)

    # the slope of the last five, oldest to newest, over the figures held
    t = -np.arange(1, 6, dtype=float)                       # newest is -1
    f = np.isfinite(lags)
    n = f.sum(1)
    tt = np.where(f, t, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = tt.sum(1) / np.where(n > 0, n, np.nan)
        ym = np.where(f, lags, 0.0).sum(1) / np.where(n > 0, n, np.nan)
        cov = (np.where(f, (t - tm[:, None]) * (lags - ym[:, None]), 0.0)).sum(1)
        var = (np.where(f, (t - tm[:, None]) ** 2, 0.0)).sum(1)
        slope = np.where((n >= 3) & (var > 0), cov / var, np.nan)
        w = np.where(f, np.arange(5, 0, -1, dtype=float), 0.0)
        w5 = np.where(w.sum(1) > 0, (np.where(f, lags, 0.0) * w).sum(1) / w.sum(1), np.nan)
    cs = pd.Series(np.where(np.isfinite(pb), pb, 0.0)).groupby(key).cumsum().to_numpy()
    cn = pd.Series(np.isfinite(pb).astype(float)).groupby(key).cumsum().to_numpy()
    car = np.where(has & (cn[prev] > 0), cs[prev] / np.where(cn[prev] > 0, cn[prev], 1.0), np.nan)

    # the gain at each run: this day's figure less the one before, and the cell the
    # run was in (career runs before it, age, code); post-race, for later days only
    gain = pb - _earlier(H, pb, 1)
    nb = H.per_block(nfp)
    ngain = nb - _earlier(H, nb, 1)
    gain_r, ngain_r = H.to_rows(gain), H.to_rows(ngain)
    runs = _num(df, "career_runs")
    runs = np.where(np.isfinite(runs), runs, H.to_rows(runs_code))
    bucket = np.searchsorted(RUN_EDGES, np.nan_to_num(runs, nan=-1), side="right")
    age = np.clip(np.nan_to_num(_num(df, "horse_age"), nan=0), 0, 9).astype(np.int64)
    cell = (bucket.astype(np.int64) * 10 + age) * 8 + code_id
    cell = np.where(np.isfinite(runs), cell, -1)

    g_cell, _ = asof_decayed_mean(cell, day, gain_r, cell, day, halflife_days=HALFLIFE_DAYS)
    ng_cell, _ = asof_decayed_mean(cell, day, ngain_r, cell, day, halflife_days=HALFLIFE_DAYS)
    prior = np.nan_to_num(g_cell)

    def conn_gain(name: str) -> np.ndarray:
        who = _key(_col(df, name))
        k = np.where((who >= 0) & (cell >= 0), who * 1024 + bucket, -1)
        mm, nn = asof_decayed_mean(k, day, gain_r, k, day, halflife_days=HALFLIFE_DAYS)
        out = shrunk_mean(np.nan_to_num(mm) * nn, nn, prior, CONN_K)
        return np.where(np.isfinite(g_cell), out, np.nan)

    orr = _num(df, "official_rating")
    orr = np.where(orr > 0, orr, np.nan)
    rows = {
        "ex_runs_code": H.to_rows(runs_code),
        "ex_perf_best": H.to_rows(best),
        "ex_perf_best3": H.to_rows(best3),
        "ex_perf_slope5": H.to_rows(slope),
        "ex_perf_d1": H.to_rows(lags[:, 0] - lags[:, 1]),
        "ex_perf_d_car": H.to_rows(lags[:, 0] - car),
        "ex_gain_cell": g_cell,
        "ex_gain_trainer": conn_gain("trainer"),
        "ex_gain_sire": conn_gain("stallion"),
        "ex_nfp_gain_cell": ng_cell,
    }
    rows["ex_headroom"] = rows["ex_perf_best"] + np.nan_to_num(g_cell) - orr
    rows["ex_headroom_w5"] = H.to_rows(w5) + np.nan_to_num(g_cell) - orr
    new = pd.DataFrame(rows, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
