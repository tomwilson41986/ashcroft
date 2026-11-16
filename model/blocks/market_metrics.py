"""Market metrics: what the price expected of every run, what happened, and the horse's record on the difference.

The owner's ask of 12 Nov 2026: "a full suite of metrics purely derived from market prices ... expected time
market, expected lengths beaten market, expected normalised finishing position market, expected improvement
market, expected form cycle market, expected intent market; vs actuals; normalised; sum of squares; Bayesian".

1. What the market expects at a price. A runner's price is read as x = log(runners x its share of the Betfair SP
   book). For each of three outcomes the market's expectation at x is the mean outcome of the runners on EARLIER
   days whose x fell in the same step (model/blocks/_resid.py's steps):
       n   the normalised finishing position (1 won, 0 last; a non-finisher 0)
       l   the lengths beaten, capped at LB_CAP (a non-finisher: no reading)
       t   the time figure against the race's mean figure, in pounds (model/blocks/time_figure.py)
   and the run's residual is the outcome less that expectation, signed so that above zero is better than priced.

2. The horse's record on each residual, from its earlier racing days:
       mm_<o>_l1  mm_<o>_m3      the last run, the mean of the last three
       mm_<o>_bayes              the decayed mean (a year's half-life) shrunk to zero by K_RUNS runs
       mm_<o>_ss                 the root of the decayed mean square: how far from its price it runs, either way
       mm_n_t                    the n record as a t-like score: the decayed mean over its standard error

3. What the market expects today. Today's price is not known at 06:00; the market rating's implied share is
   (mr_lshare, model/blocks/market_rating.py), so the expectation is read at that:
       mm_exp_n  mm_exp_l  mm_exp_t     expected finishing position, lengths beaten and relative time figure

4. What the market has expected of the horse. m = a run's x less the x its market rating implied before the race:
   above zero, the market made it shorter than its own price history said (it expected improvement).
       mm_m_l1  mm_m_m3  mm_m_bayes     the market's expected improvement: last run, last three, shrunk decayed mean
       mm_deliver                       its n residual on runs the market expected more of it, less on the others
       mm_cycle_m  mm_cycle_m_tr        the market's m at this point of a campaign (first, second, third, later run
                                        back from 60 days off): the horse's own, and the yard's runners' (shrunk)
       mm_intent_m                      the yard's m with today's moves, summed over those made: a jockey change,
                                        a prize at least half as much again as the horse's usual, a prize at most
                                        two thirds of it (each shrunk to zero by twenty runners)

Earlier days only throughout; today's inputs are the card's (runners, prize, jockey) and the market rating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import _resid
from model.blocks import time_figure as _tf
from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.race_shape import asof_decayed_mean, day_index, race_key

OUT = ("n", "l", "t")
FEATURES = ([f"mm_{o}_{w}" for o in OUT for w in ("l1", "m3", "bayes", "ss")] + ["mm_n_t", "mm_exp_n", "mm_exp_l",
            "mm_exp_t", "mm_m_l1", "mm_m_m3", "mm_m_bayes", "mm_deliver", "mm_cycle_m", "mm_cycle_m_tr", "mm_intent_m"])
POST_RACE: set[str] = set()
READS = sorted(set(_tf.READS) | {"trainer", "jockey_name", "number_of_runners", "bfsp", "prize_money", "mr_lshare"})
AFTER = ("market_rating",)
LB_CAP = 30.0
K_RUNS = 3.0
K_YARD = 20.0
BREAK_DAYS = 60


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _step(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), np.digitize(np.clip(x, _resid.BINS[0], _resid.BINS[-1]), _resid.BINS), -1).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    race = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    horse, trainer, jockey = _codes(df["horse_name"]), _codes(_col(df, "trainer")), _codes(_col(df, "jockey_name"))
    H = _Days(horse, day)
    nb = len(H.u)
    blocks = np.arange(nb)
    prev = np.clip(blocks - 1, 0, None)
    has_prev = H.valid & (H.pos >= 1)

    # each run's price position, and the three outcomes
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=df.index)
    tot = inv.groupby(race).transform("sum").to_numpy()
    rows = pd.Series(1.0, index=df.index).groupby(race).transform("sum").to_numpy()
    npr = pd.Series(priced.astype(float), index=df.index).groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(priced & (tot > 0) & (npr >= 0.9 * rows), np.log(rows * inv.to_numpy() / np.where(tot > 0, tot, 1.0)), np.nan)
        nfp = np.where(np.isfinite(pos) & (pos >= 1) & (n > 1), np.clip((n - pos) / (n - 1), 0.0, 1.0),
                       np.where(n > 1, 0.0, np.nan))
    lb = pd.to_numeric(_col(df, "total_dst_bt"), errors="coerce").to_numpy(float)
    lb = np.where(pos == 1, 0.0, np.where(np.isfinite(pos) & (pos > 1) & np.isfinite(lb), np.minimum(lb, LB_CAP), np.nan))
    fig = pd.Series(np.asarray(_tf.figures(df)["fig"], dtype=float), index=df.index)
    rel = (fig - fig.groupby(race).transform("mean")).to_numpy(float)
    outcome = {"n": nfp, "l": -lb, "t": rel}                     # signed: higher is better
    b = _step(x)
    x_today = pd.to_numeric(_col(df, "mr_lshare"), errors="coerce").to_numpy(float)
    b_today = _step(x_today)

    cols: dict[str, np.ndarray] = {}
    res = {}
    for o in OUT:
        exp, cnt = asof_decayed_mean(b, day, outcome[o], b, day)
        res[o] = np.where((b >= 0) & (cnt >= 20), outcome[o] - exp, np.nan)
        e_today, c_today = asof_decayed_mean(b, day, outcome[o], b_today, day)
        cols[f"mm_exp_{o}"] = np.where((b_today >= 0) & (c_today >= 20), e_today * (-1.0 if o == "l" else 1.0), np.nan)
        lad = _ladder(H, H.per_block(res[o]))
        cols[f"mm_{o}_l1"], cols[f"mm_{o}_m3"] = H.to_rows(lad["l1"]), H.to_rows(lad["m3"])
        m, k = asof_decayed_mean(horse, day, res[o], horse, day, halflife_days=365.0)
        sq, _ = asof_decayed_mean(horse, day, res[o] ** 2, horse, day, halflife_days=365.0)
        cols[f"mm_{o}_bayes"] = np.where(k > 0, m * k / (k + K_RUNS), np.nan)
        cols[f"mm_{o}_ss"] = np.sqrt(sq)
        if o == "n":
            with np.errstate(invalid="ignore", divide="ignore"):
                cols["mm_n_t"] = np.where((k >= 2) & (sq > 0), m / np.sqrt(sq / k), np.nan)

    # the market's own expectation of the horse: its price against its market rating
    mres = np.clip(x - x_today, -3.0, 3.0)
    lad = _ladder(H, H.per_block(mres))
    cols["mm_m_l1"], cols["mm_m_m3"] = H.to_rows(lad["l1"]), H.to_rows(lad["m3"])
    m, k = asof_decayed_mean(horse, day, mres, horse, day, halflife_days=365.0)
    cols["mm_m_bayes"] = np.where(k > 0, m * k / (k + K_RUNS), np.nan)
    up, n_up = asof_decayed_mean(horse, day, np.where(mres > 0, res["n"], np.nan), horse, day)
    dn, n_dn = asof_decayed_mean(horse, day, np.where(mres <= 0, res["n"], np.nan), horse, day)
    cols["mm_deliver"] = np.where((n_up > 0) & (n_dn > 0), up - dn, np.nan)

    gap = np.where(has_prev, H.day - H.day[prev], np.inf)
    idx = np.where(((~has_prev) | (gap >= BREAK_DAYS)) & H.valid, blocks, -1)
    since = blocks - pd.Series(idx).groupby(H.key).cummax().to_numpy()
    cyc = H.to_rows(np.where(H.valid, np.minimum(since, 3), np.nan))
    cyc = np.where(np.isfinite(cyc), cyc, -1).astype(np.int64)

    def key(a, c):
        bad = (a < 0) | (c < 0)
        return np.where(bad, -1, pd.factorize(a.astype(np.int64) * 8 + np.where(c < 0, 0, c), sort=True)[0]).astype(np.int64)

    hm, hn = asof_decayed_mean(key(horse, cyc), day, mres, key(horse, cyc), day)
    cols["mm_cycle_m"] = np.where(hn > 0, hm, np.nan)
    tm, tn = asof_decayed_mean(key(trainer, cyc), day, mres, key(trainer, cyc), day, halflife_days=1095.0)
    cols["mm_cycle_m_tr"] = np.where(trainer >= 0, np.nan_to_num(tm) * np.nan_to_num(tn) / (np.nan_to_num(tn) + K_YARD), np.nan)

    jb = H.per_block(jockey.astype(float))
    last_jk = H.to_rows(np.where(has_prev, jb[prev], np.nan))
    prize = pd.to_numeric(_col(df, "prize_money"), errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.where(prize > 0, np.log(np.where(prize > 0, prize, 1.0)), np.nan)
    usual, _ = asof_decayed_mean(horse, day, lp, horse, day, halflife_days=730.0)
    moves = [np.isfinite(last_jk) & (last_jk >= 0) & (jockey >= 0) & (last_jk != jockey),
             (lp - usual) >= np.log(1.5), (lp - usual) <= np.log(2.0 / 3.0)]
    intent = np.zeros(len(df))
    for flag in moves:
        ym, yn = asof_decayed_mean(trainer, day, np.where(flag, mres, np.nan), trainer, day, halflife_days=1095.0)
        intent += np.where(flag & (trainer >= 0), np.nan_to_num(ym) * np.nan_to_num(yn) / (np.nan_to_num(yn) + K_YARD), 0.0)
    cols["mm_intent_m"] = intent

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
