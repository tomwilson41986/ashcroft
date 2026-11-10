"""A market rating: every past Betfair SP put on one scale across classes, as a horse's ability with its uncertainty.

The model reads past prices as they stand: `fw_mkt` is log(field size x the horse's
share of the book), ORR2 is field size / SP. Both say how a horse stood against that
day's rivals, not against horses in general: a 4/1 shot in a Group race and a 4/1
shot in a seller read the same, and prize money ends up standing in for class.

Here each past race is read as a multinomial logit on latent strengths, the market's
share of horse i being exp(r_i) / sum_j exp(r_j). A runner's log share of the book is
then its strength less the race's level, and the race's level comes from the rivals'
own ratings: the reading of horse i is

    y_i = log p_i + C_i,   C_i = sum_{j != i} w_j (m_j - log p_j) / sum_{j != i} w_j

with m_j the rival's rating before the day and w_j = 1 / (P_j + R) its precision, so a
horse beaten in the betting by Group horses reads higher than one beaten by sellers.
The rating is a local-level Kalman filter on those readings (as model/blocks/kalman.py
on the performance figure): ability drifts with the days since the last run, faster
over the first runs, and each reading has variance R. The drift and R are chosen by
the one-step predictive likelihood on the races before FIT_UNTIL only, so the choice
is the same in every fold and on the live path (history from 2018). A horse's first
rating is the mean of its rated rivals' that day (else 0) with variance INIT_VAR.
Ratings live in two pools, the Flat (turf and all-weather) and jumps. A race is read
only when 90% of its runners have a usable SP.

From earlier days only (the day's prices enter after its features are read):

    mr_rating     the rating before today (NaN before the horse's first reading)
    mr_sd         its standard deviation
    mr_n          readings behind it
    mr_share      today's implied share: the softmax of the field's ratings (unread horses at their prior)
    mr_lshare     log(field size x mr_share), on fw_mkt's scale
    mr_gap        rating less the best rival's (leave the horse out; NaN unread)
    mr_z          (rating - race mean) / race sd over read runners, at least 3 of them
    mr_field      the race's mean rating over read runners (field strength)
    mr_read_share share of today's runners with a reading
    mr_surprise   the last reading less the rating before it (the last run's news)
    mr_trend      rating less the rating three readings ago
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import day_index, race_code, race_key

FEATURES = ["mr_rating", "mr_sd", "mr_n", "mr_share", "mr_lshare", "mr_gap", "mr_z", "mr_field",
            "mr_read_share", "mr_surprise", "mr_trend"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "bfsp", "race_type", "surface_type"]

INIT_VAR = 1.0             # a first rating's variance (log-strength units^2)
Q_CAREER_MULT = 2.0        # faster drift over a horse's first readings
Q_CAREER_RUNS = 4
MIN_PRICED = 0.9           # share of a race's runners with a usable SP before it is read
LOG_P_FLOOR = np.log(0.002)
TREND_LAG = 3
FIT_UNTIL = "2021-01-01"   # q and R are chosen on races before this day only
GRID_Q = (0.05, 0.15, 0.45)     # drift variance a year
GRID_R = (0.15, 0.3, 0.6)       # a reading's variance
DEFAULT_Q, DEFAULT_R = 0.15, 0.3
MIN_FIT_READINGS = 5000


def _pool(df: pd.DataFrame) -> np.ndarray:
    """0 for the Flat (turf and all-weather), 1 for jumps (hurdles, chases, bumpers), from race_type."""
    return race_code(df).isin(["hurdle", "chase", "nhflat"]).to_numpy().astype(np.int64)


def _sorted_codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _log_share(df: pd.DataFrame, race: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each runner's log share of its race's book, and whether its race is read."""
    bsp = pd.to_numeric(df["bfsp"], errors="coerce").to_numpy(float) if "bfsp" in df.columns \
        else np.full(len(df), np.nan)
    ok = np.isfinite(bsp) & (bsp > 1.0)
    inv = np.where(ok, 1.0 / np.where(ok, bsp, 1.0), 0.0)
    nr = int(race.max()) + 1 if len(race) else 0
    tot = np.bincount(race, weights=inv, minlength=nr)
    priced = np.bincount(race, weights=ok.astype(float), minlength=nr)
    rows = np.bincount(race, minlength=nr).astype(float)
    read = (priced[race] >= 2) & (priced[race] >= MIN_PRICED * rows[race])
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(ok & read, np.log(inv / np.where(tot[race] > 0, tot[race], 1.0)), np.nan)
    return np.maximum(lp, LOG_P_FLOOR), read & ok


def _run(day, key, race, lp, readable, q_year, r_var, fit_until_day=None, want=True):
    """The filter, day by day. Returns the features (want=True) or the predictive log likelihood
    of the readings before fit_until_day."""
    n = len(day)
    nk = int(key.max()) + 1 if (key >= 0).any() else 0
    m, P = np.zeros(nk), np.full(nk, np.nan)       # P NaN: no reading yet
    cnt, last = np.zeros(nk), np.zeros(nk)
    surprise = np.full(nk, np.nan)
    hist = np.full((nk, TREND_LAG), np.nan)
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
        known = np.isfinite(P[k])
        mult = np.where(cnt[k] < Q_CAREER_RUNS, Q_CAREER_MULT, 1.0)
        pv = np.where(known, P[k] + q_day * mult * np.maximum(d - last[k], 1.0), INIT_VAR)
        # an unread horse starts at the mean of its read rivals that day, else 0
        rc, rloc = np.unique(race[rows], return_inverse=True)
        nr = len(rc)
        k_sum = np.bincount(rloc, weights=np.where(known, m[k], 0.0), minlength=nr)
        k_cnt = np.bincount(rloc, weights=known.astype(float), minlength=nr)
        start = np.where(k_cnt > 0, k_sum / np.where(k_cnt > 0, k_cnt, 1.0), 0.0)
        pm = np.where(known, m[k], start[rloc])
        if want:
            size = np.bincount(rloc, minlength=nr).astype(float)
            top = np.full(nr, -np.inf)
            np.maximum.at(top, rloc, pm)
            ex = np.exp(pm - top[rloc])
            field = np.log(np.bincount(rloc, weights=ex, minlength=nr)) + top
            share = np.exp(pm - field[rloc])
            # best rival: the race max, or the second best for the leader
            kp = np.where(known, pm, -np.inf)
            first = np.full(nr, -np.inf)
            np.maximum.at(first, rloc, kp)
            is_top = known & (kp == first[rloc])
            n_top = np.bincount(rloc, weights=is_top.astype(float), minlength=nr)
            second = np.full(nr, -np.inf)
            np.maximum.at(second, rloc, np.where(is_top, -np.inf, kp))
            rival = np.where(is_top & (n_top[rloc] < 2), second[rloc], first[rloc])
            mean_k = np.where(k_cnt > 0, k_sum / np.where(k_cnt > 0, k_cnt, 1.0), np.nan)
            sq = np.bincount(rloc, weights=np.where(known, (m[k] - mean_k[rloc]) ** 2, 0.0), minlength=nr)
            sd_k = np.where(k_cnt >= 3, np.sqrt(sq / np.maximum(k_cnt - 1, 1)), np.nan)
            o = rows
            out["mr_rating"][o] = np.where(known, pm, np.nan)
            out["mr_sd"][o] = np.where(known, np.sqrt(pv), np.nan)
            out["mr_n"][o] = cnt[k]
            out["mr_share"][o] = share
            out["mr_lshare"][o] = np.log(size[rloc] * share)
            out["mr_gap"][o] = np.where(known & np.isfinite(rival), pm - rival, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["mr_z"][o] = np.where(known & (sd_k[rloc] > 0), (pm - mean_k[rloc]) / sd_k[rloc], np.nan)
            out["mr_field"][o] = mean_k[rloc]
            out["mr_read_share"][o] = k_cnt[rloc] / size[rloc]
            out["mr_surprise"][o] = surprise[k]
            out["mr_trend"][o] = np.where(known, pm - hist[k, 0], np.nan)
        # the day's readings, after its features: the race level from the rivals, leave the horse out
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
        # a horse read twice in a day (two pools share no key, so only a data error): the last reading stands
        ku, idx = k[upd], np.flatnonzero(upd)
        gain = pv[idx] / (pv[idx] + r_var)
        new_m = pm[idx] + gain * (y[idx] - pm[idx])
        hist[ku, :-1] = hist[ku, 1:]
        hist[ku, -1] = pm[idx]
        surprise[ku] = y[idx] - pm[idx]
        m[ku], P[ku] = new_m, (1.0 - gain) * pv[idx]
        cnt[ku] += 1
        last[ku] = d
    return out if want else (ll, n_ll)


def _choose(day, key, race, lp, readable) -> tuple[float, float]:
    """q and R by the predictive likelihood of the readings before FIT_UNTIL; the defaults when too few."""
    cut = int(np.datetime64(FIT_UNTIL, "D").astype(np.int64))
    best, n_best = None, 0
    for q in GRID_Q:
        for r in GRID_R:
            ll, n_ll = _run(day, key, race, lp, readable, q, r, fit_until_day=cut, want=False)
            if n_ll < MIN_FIT_READINGS:
                return DEFAULT_Q, DEFAULT_R
            if best is None or ll / n_ll > best[0]:
                best, n_best = (ll / n_ll, q, r), n_ll
    return best[1], best[2]


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    horse = _sorted_codes(df["horse_name"])
    key = np.where(horse >= 0, horse * 2 + _pool(df), -1)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    lp, readable = _log_share(df, race)
    q, r = _choose(day, key, race, lp, readable)
    out = _run(day, key, race, lp, readable, q, r)
    new = pd.DataFrame(out, index=df.index)[FEATURES]
    assert len(new) == n
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
