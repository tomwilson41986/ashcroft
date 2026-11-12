"""A connections rating: what the trainer and the jockey add, each credited jointly, as states with their uncertainty.

The connection windows (model/blocks/connection_windows.py) average a yard's and a rider's recent runners one at a
time, so a jockey who rides mostly for one strong yard carries the yard's record. Here both are rated together (the
owner, 12 Nov 2026: "a connections model and rating ... trainer, owner, jockey - uplift, change, residuals"), on two
residuals of each past run:

    against the price    s = the run's normalised finishing position (a non-finisher 0) less the position its rank
                         in the Betfair SP book implied: what the run did beyond what the market expected
    against the horse    f = the run's time figure (model/blocks/time_figure.py, pounds) less the horse's own
                         decayed mean figure before the run: what the run did beyond the horse's own level

    against the rating   m = the run's log share of the Betfair SP book (x runners) less the share the field's market
                         ratings implied before the race (mr_lshare, model/blocks/market_rating.py): how much shorter
                         the market made the horse than its own price history said - the yard's and the rider's pull
                         on the price itself (the model review's "market plus-minus", 11 Nov)

Each residual is read as trainer effect + jockey effect + noise. Both effects are local-level Kalman states that
drift with time; a day's runs update the trainer on the residual less the jockey's rating (and the reverse), each
weighted by the other's uncertainty, so shared runs are split by how well each is known. Variances are fixed
(PRIOR_VAR, DRIFT_YEAR, NOISE), set before the first screen. The owner of a horse is not held in the data: no
owner rating.

    cr_tr_s  cr_jk_s          the trainer's and the jockey's rating against the price (before today)
    cr_tr_s_sd  cr_jk_s_sd    their standard deviations
    cr_tr_f  cr_jk_f          their ratings against the horse's own level, in pounds
    cr_tr_f_sd  cr_jk_f_sd
    cr_tr_m  cr_jk_m          their ratings against the market rating (log share), with cr_tr_m_sd, cr_jk_m_sd
    cr_sum_s  cr_sum_f  cr_sum_m        trainer + jockey
    cr_sum_s_z  cr_sum_f_z  cr_sum_m_z  the sums standardised within today's race
    cr_jk_change              1 when today's jockey did not ride the horse last time (NaN on a debut)
    cr_jk_delta_s  cr_jk_delta_f  cr_jk_delta_m   today's jockey's rating less the last run's jockey's, as both stand today
    cr_jk_new_s               the jockey's decayed mean of s on the rides where they took a horse over (uplift on a
                              change), shrunk to 0 by twenty rides
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import time_figure as _tf
from model.freshness_features import _Days, _col
from model.race_shape import asof_decayed_mean, day_index, race_key

FEATURES = ["cr_tr_s", "cr_jk_s", "cr_tr_s_sd", "cr_jk_s_sd", "cr_tr_f", "cr_jk_f", "cr_tr_f_sd", "cr_jk_f_sd",
            "cr_tr_m", "cr_jk_m", "cr_tr_m_sd", "cr_jk_m_sd", "cr_sum_s", "cr_sum_f", "cr_sum_m", "cr_sum_s_z",
            "cr_sum_f_z", "cr_sum_m_z", "cr_jk_change", "cr_jk_delta_s", "cr_jk_delta_f", "cr_jk_delta_m",
            "cr_jk_new_s"]
POST_RACE: set[str] = set()
READS = sorted(set(_tf.READS) | {"trainer", "jockey_name", "number_of_runners", "bfsp", "mr_lshare"})
AFTER = ("market_rating",)
PRIOR_VAR = {"s": 0.05 ** 2, "f": 2.0 ** 2, "m": 0.10 ** 2}
DRIFT_YEAR = {"s": 0.02 ** 2, "f": 0.8 ** 2, "m": 0.04 ** 2}
NOISE = {"s": 0.30 ** 2, "f": 12.0 ** 2, "m": 0.50 ** 2}
K_NEW = 20.0


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _filter(day, tr, jk, r, which):
    """Ratings before each row's day: (trainer mean, sd, jockey mean, sd), and each jockey's final-state lookup."""
    n = len(day)
    nt, nj = int(tr.max()) + 1 if len(tr) else 0, int(jk.max()) + 1 if len(jk) else 0
    p0, q, R = PRIOR_VAR[which], DRIFT_YEAR[which] / 365.0, NOISE[which]
    m = [np.zeros(max(nt, 1)), np.zeros(max(nj, 1))]
    P = [np.full(max(nt, 1), p0), np.full(max(nj, 1), p0)]
    last = [np.full(max(nt, 1), -1.0), np.full(max(nj, 1), -1.0)]
    out = [np.full(n, np.nan) for _ in range(4)]
    jk_at = np.full((n, 2), np.nan)                 # the jockey's rating and sd as of each row's day, any jockey code
    order = np.argsort(day, kind="stable")
    order = order[day[order] >= 0]
    days = day[order]
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]]) if len(order) else np.array([], dtype=int)
    ends = np.r_[starts[1:], len(order)]
    snaps = []
    for s, e in zip(starts, ends):
        rows = order[s:e]
        d = float(days[s])
        ks = (tr[rows], jk[rows])
        pm, pv = [], []
        for i in (0, 1):
            k = ks[i]
            has = k >= 0
            kk = np.where(has, k, 0)
            gap = np.where(last[i][kk] >= 0, d - last[i][kk], 0.0)
            v = np.minimum(P[i][kk] + q * gap, p0)
            pm.append(np.where(has, m[i][kk], 0.0))
            pv.append(np.where(has, v, 0.0))
            out[2 * i][rows] = np.where(has, m[i][kk], np.nan)
            out[2 * i + 1][rows] = np.where(has, np.sqrt(v), np.nan)
        snaps.append((rows, d))
        ok = np.isfinite(r[rows])
        if not ok.any():
            continue
        new = []
        for i in (0, 1):
            k, o = ks[i], 1 - i
            use = ok & (k >= 0)
            if not use.any():
                new.append(None)
                continue
            z = r[rows][use] - pm[o][use]
            w = 1.0 / (pv[o][use] + R)
            ku = k[use]
            u, inv = np.unique(ku, return_inverse=True)
            prec = 1.0 / np.maximum(pv[i][use][np.unique(inv, return_index=True)[1]], 1e-12) + np.bincount(inv, weights=w)
            num = pm[i][use][np.unique(inv, return_index=True)[1]] / np.maximum(
                pv[i][use][np.unique(inv, return_index=True)[1]], 1e-12) + np.bincount(inv, weights=w * z)
            new.append((u, num / prec, 1.0 / prec))
        for i in (0, 1):
            if new[i] is not None:
                u, mm, pp = new[i]
                m[i][u], P[i][u], last[i][u] = mm, pp, d
    return out, (m[1], P[1], last[1], q, p0)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    tr, jk = _codes(_col(df, "trainer")), _codes(_col(df, "jockey_name"))
    horse = _codes(df["horse_name"])
    race = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)

    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    rk = pd.Series(np.where(priced, bsp, np.nan), index=df.index).groupby(race).rank(method="average").to_numpy()
    npr = pd.Series(priced.astype(float), index=df.index).groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.where(np.isfinite(pos) & (pos >= 1) & (n > 1), np.clip((n - pos) / (n - 1), 0.0, 1.0),
                       np.where(n > 1, 0.0, np.nan))
        s = nfp - np.where(priced & (npr > 1), (npr - rk) / (npr - 1), np.nan)
    fig = np.asarray(_tf.figures(df)["fig"], dtype=float)
    own, _ = asof_decayed_mean(horse, day, fig, horse, day, halflife_days=365.0)
    f = fig - own

    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=df.index)
    rows_n = pd.Series(1.0, index=df.index).groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        lshare = np.where(priced & (npr >= 0.9 * rows_n), np.log(rows_n * inv / inv.groupby(race).transform("sum")), np.nan)
    mres = np.clip(lshare - pd.to_numeric(_col(df, "mr_lshare"), errors="coerce").to_numpy(float), -3.0, 3.0)

    cols: dict[str, np.ndarray] = {}
    for which, r in (("s", s), ("f", f), ("m", mres)):
        (tm, tsd, jm, jsd), _ = _filter(day, tr, jk, r, which)
        cols[f"cr_tr_{which}"], cols[f"cr_tr_{which}_sd"] = tm, tsd
        cols[f"cr_jk_{which}"], cols[f"cr_jk_{which}_sd"] = jm, jsd
        tot = pd.Series(np.nan_to_num(tm) + np.nan_to_num(jm), index=df.index).where(np.isfinite(tm) | np.isfinite(jm))
        cols[f"cr_sum_{which}"] = tot.to_numpy(float)
        g = tot.groupby(race)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (tot - g.transform("mean")) / g.transform("std")
        cols[f"cr_sum_{which}_z"] = z.where(g.transform("count") >= 3).replace([np.inf, -np.inf], np.nan).to_numpy(float)

    # the jockey change: today's rider against the last run's, both as rated before today
    H = _Days(horse, day)
    jb = H.per_block(jk.astype(float))
    prev = np.clip(np.arange(len(H.u)) - 1, 0, None)
    last_jk = H.to_rows(np.where(H.valid & (H.pos >= 1), jb[prev], np.nan))
    known = np.isfinite(last_jk) & (last_jk >= 0) & (jk >= 0)
    change = np.where(known, (last_jk != jk).astype(float), np.nan)
    cols["cr_jk_change"] = change
    for which in ("s", "f", "m"):
        now = cols[f"cr_jk_{which}"]
        # the last run's jockey as rated before today: their latest earlier rating, by an as-of read of their own rows
        lj = np.where(known, last_jk, -1).astype(np.int64)
        prior, cnt = asof_decayed_mean(jk, day, now, lj, day, halflife_days=1.0)
        cols[f"cr_jk_delta_{which}"] = np.where(known & (change == 1) & (cnt > 0), now - prior, np.where(change == 0, 0.0, np.nan))
    took = np.where(change == 1, s, np.nan)
    mean_new, n_new = asof_decayed_mean(jk, day, took, jk, day, halflife_days=730.0)
    cols["cr_jk_new_s"] = np.where(jk >= 0, np.nan_to_num(mean_new) * np.nan_to_num(n_new) / (np.nan_to_num(n_new) + K_NEW), np.nan)

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
