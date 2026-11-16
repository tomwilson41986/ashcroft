"""A conditions-aware market rating: what past Betfair SPs say of the horse at today's trip and ground.

The market rating (model/blocks/market_rating.py) is one number a horse: the BSP of a 5f run on firm ground and of
a 7f run on soft are read as the same ability, and the trees must find the horse's aptitude again from thin
results. Here the rating is a level plus two slopes (reports/discovery_1111.md 4.2):

    r_i(x) = a_i + v_i . g(x),   g(x) = (trip, ground)

with trip the log distance against its pool's usual (TRIP: centre and scale, the Flat and jumps) and ground a score
of the coarse going group (firm -1, good 0, soft 1, heavy 1.5; all-weather and unknown 0). The coarse groups are
used because the result's going and the morning's differ on about a quarter of races. Readings are the market
rating's own: y_i = log p_i + C_i, the race's level from the rivals' ratings taken at that race's conditions. The
state (a, v1, v2) is a Kalman filter by horse and pool: the level drifts as the market rating's does (its q and R,
chosen on races before 2021), the slopes do not drift and start at zero with variance V0, chosen from GRID_V by
the one-step predictive likelihood of the readings before FIT_UNTIL.

This first version takes g as given and has no sire or trainer prior on the slopes (the specification's learned
embedding and priors are left for a second version if this one shows anything).

From earlier days only:

    mr2_rating_x   the rating at today's trip and ground (NaN before the horse's first reading)
    mr2_sd_x       its standard deviation there: wide where the horse has not been priced near these conditions
    mr2_apt        v . g(x): today's conditions against the horse's usual, in rating units
    mr2_share_x    the softmax of the field's ratings at today's conditions
    mr2_lshare_x   log(field size x mr2_share_x)
    mr2_z_x        (rating - race mean) / race sd over read runners, at least 3 of them
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import market_rating as _mr
from model.blocks.cond_form import GOING, classify_going
from model.freshness_features import _col
from model.race_shape import day_index, race_key

FEATURES = ["mr2_rating_x", "mr2_sd_x", "mr2_apt", "mr2_share_x", "mr2_lshare_x", "mr2_z_x"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "bfsp", "race_type", "surface_type",
         "dist_furlongs", "going_description"]
FIT_UNTIL = _mr.FIT_UNTIL
GRID_V = (0.02, 0.06, 0.18)                 # a slope's prior variance
DEFAULT_V = 0.06
TRIP = {0: (np.log(8.0), 0.45), 1: (np.log(20.0), 0.2)}     # pool: centre and scale of log furlongs
GROUND = {0: -1.0, 1: 0.0, 2: 1.0, 3: 1.5, 4: 0.0}           # cond_form.GOING's groups


def _g(df: pd.DataFrame, pool: np.ndarray) -> np.ndarray:
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    c = np.where(pool == 1, TRIP[1][0], TRIP[0][0])
    sc = np.where(pool == 1, TRIP[1][1], TRIP[0][1])
    with np.errstate(invalid="ignore", divide="ignore"):
        trip = np.where(np.isfinite(dist) & (dist > 0), (np.log(np.where(dist > 0, dist, 1.0)) - c) / sc, 0.0)
    going = _col(df, "going_description").map(classify_going).map(GOING).map(GROUND)
    ground = pd.to_numeric(going, errors="coerce").fillna(0.0).to_numpy(float)
    return np.column_stack([np.ones(len(df)), np.clip(trip, -3.0, 3.0), ground])


def _run(day, key, race, lp, readable, G, q_year, r_var, v0, fit_until_day=None, want=True):
    n = len(day)
    nk = int(key.max()) + 1 if (key >= 0).any() else 0
    m = np.zeros((nk, 3))
    P = np.zeros((nk, 3, 3))
    seen = np.zeros(nk, dtype=bool)
    cnt, last = np.zeros(nk), np.zeros(nk)
    P0 = np.diag([_mr.INIT_VAR, v0, v0])
    out = {c: np.full(n, np.nan) for c in FEATURES} if want else None
    ll, n_ll = 0.0, 0
    q_day = q_year / 365.0
    order = np.lexsort((key, race, day))
    order = order[(day[order] >= 0) & (key[order] >= 0)]
    if fit_until_day is not None:
        order = order[day[order] < fit_until_day]
    if not len(order):
        return out if want else (ll, n_ll)
    days = day[order]
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        rows = order[s:e]
        d = days[s]
        k = key[rows]
        h = G[rows]
        known = seen[k]
        Pk = np.where(known[:, None, None], P[k], P0[None])
        mult = np.where(cnt[k] < _mr.Q_CAREER_RUNS, _mr.Q_CAREER_MULT, 1.0)
        Pk[:, 0, 0] += np.where(known, q_day * mult * np.maximum(d - last[k], 1.0), 0.0)
        Ph = np.einsum("nij,nj->ni", Pk, h)
        pv = np.einsum("ni,ni->n", h, Ph)
        rc, rloc = np.unique(race[rows], return_inverse=True)
        nr = len(rc)
        own = np.einsum("ni,ni->n", m[k], h)
        k_sum = np.bincount(rloc, weights=np.where(known, own, 0.0), minlength=nr)
        k_cnt = np.bincount(rloc, weights=known.astype(float), minlength=nr)
        start = np.where(k_cnt > 0, k_sum / np.where(k_cnt > 0, k_cnt, 1.0), 0.0)
        pm = np.where(known, own, start[rloc])
        if want:
            size = np.bincount(rloc, minlength=nr).astype(float)
            top = np.full(nr, -np.inf)
            np.maximum.at(top, rloc, pm)
            field = np.log(np.bincount(rloc, weights=np.exp(pm - top[rloc]), minlength=nr)) + top
            share = np.exp(pm - field[rloc])
            mean_k = np.where(k_cnt > 0, k_sum / np.where(k_cnt > 0, k_cnt, 1.0), np.nan)
            sq = np.bincount(rloc, weights=np.where(known, (own - mean_k[rloc]) ** 2, 0.0), minlength=nr)
            sd_k = np.where(k_cnt >= 3, np.sqrt(sq / np.maximum(k_cnt - 1, 1)), np.nan)
            out["mr2_rating_x"][rows] = np.where(known, pm, np.nan)
            out["mr2_sd_x"][rows] = np.where(known, np.sqrt(np.maximum(pv, 0.0)), np.nan)
            out["mr2_apt"][rows] = np.where(known, own - m[k, 0], np.nan)
            out["mr2_share_x"][rows] = share
            out["mr2_lshare_x"][rows] = np.log(size[rloc] * share)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["mr2_z_x"][rows] = np.where(known & (sd_k[rloc] > 0), (pm - mean_k[rloc]) / sd_k[rloc], np.nan)
        ok = readable[rows]
        if not ok.any():
            continue
        w = np.where(ok, 1.0 / (pv + r_var), 0.0)
        a = np.where(ok, w * (pm - lp[rows]), 0.0)
        ws = np.bincount(rloc, weights=w, minlength=nr)
        as_ = np.bincount(rloc, weights=a, minlength=nr)
        w_o, a_o = ws[rloc] - w, as_[rloc] - a
        lvl = np.where(w_o > 0, a_o / np.where(w_o > 0, w_o, 1.0), as_[rloc] / np.where(ws[rloc] > 0, ws[rloc], 1.0))
        y = lp[rows] + lvl
        upd = ok & np.isfinite(y)
        if fit_until_day is not None:
            f = upd & known
            sv = pv[f] + r_var
            ll += float(np.sum(-0.5 * (np.log(2 * np.pi * sv) + (y[f] - pm[f]) ** 2 / sv)))
            n_ll += int(f.sum())
        idx = np.flatnonzero(upd)
        ku = k[idx]
        gain = Ph[idx] / (pv[idx] + r_var)[:, None]
        base = np.where(known[idx, None], m[ku], np.column_stack([pm[idx], np.zeros(len(idx)), np.zeros(len(idx))]))
        m[ku] = base + gain * (y[idx] - pm[idx])[:, None]
        P[ku] = Pk[idx] - np.einsum("ni,nj->nij", gain, Ph[idx])
        seen[ku] = True
        cnt[ku] += 1
        last[ku] = d
    return out if want else (ll, n_ll)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    pool = _mr._pool(df)
    horse = _mr._sorted_codes(df["horse_name"])
    key = np.where(horse >= 0, horse * 2 + pool, -1)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    lp, readable = _mr._log_share(df, race)
    G = _g(df, pool)
    q, r = _mr._choose(day, key, race, lp, readable)
    cut = int(np.datetime64(FIT_UNTIL, "D").astype(np.int64))
    v0, best = DEFAULT_V, None
    for v in GRID_V:
        ll, n_ll = _run(day, key, race, lp, readable, G, q, r, v, fit_until_day=cut, want=False)
        if n_ll < _mr.MIN_FIT_READINGS:
            v0 = DEFAULT_V
            break
        if best is None or ll / n_ll > best:
            best, v0 = ll / n_ll, v
    out = _run(day, key, race, lp, readable, G, q, r, v0)
    new = pd.DataFrame(out, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
