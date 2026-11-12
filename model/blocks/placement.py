"""Placement: where the yard has put the horse today against where it has been, and the yard's record with each move.

The intent block reads the card's one-off choices (a new yard, first handicap, gelding, headgear, a break, a class
drop, a first run in a code). The owner's list of 12 Nov 2026 goes further: handicapping, race planning, form cycles,
peaking, placement, preferred races, preferred track and distance, prize money. What the other blocks do not read:

  prize money     pl_prize_vs_usual   log of today's prize over the horse's usual (the decayed mean of the log
                                      prize of its earlier races): up in grade for money, or dropped to win
                  pl_prize_vs_last    the same against its last race
                  pl_prize_rank_tr    today's prize ranked among the yard's runners today (1 = the yard's richest
                                      race of the day), as a share
  handicapping    pl_mark_d1          today's mark less the mark at its last run
                  pl_mark_d3          today's mark less its highest mark over the last three runs
                  pl_runs_since_win   racing days since it last won (NaN if never)
  form cycle      pl_cycle_s          the horse's own result against the price (s) at this point of a campaign
                                      (first, second, third, later run back from a break of 60 days) less its mean
                                      over all its runs
  preferred race  pl_repeat           1 when it ran at this course within 30 days of this date in an earlier year,
                                      over this trip band
                  pl_repeat_nfp       its finishing position then (the best of them)
  trip            pl_trip_change      log of today's trip over the last run's
                  pl_first_trip       1 for a first run in this trip band

and the yard's record against the price with each move, for a runner making it today (else 0), over its earlier
runners, decayed over three years, shrunk to zero by twenty runners:

    pl_tr_prize_up_s  pl_tr_prize_down_s    prize at least half as much again as usual / at most two thirds of it
    pl_tr_mark_drop_s                       a mark at least 5 lb below its highest of the last three runs
    pl_tr_cycle_s                           runners at this point of a campaign
    pl_tr_repeat_s                          runners back for a race they ran in before
    pl_tr_trip_up_s  pl_tr_trip_down_s      a trip at least a fifth longer / shorter than last time

s is the run's normalised finishing position (a non-finisher 0) less the position its Betfair SP rank implied.
Earlier days only for every record; today's prize, mark and trip are the card's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _Days, _col
from model.race_shape import DIST_BANDS, asof_decayed_mean, bands, day_index, race_key

FEATURES = ["pl_prize_vs_usual", "pl_prize_vs_last", "pl_prize_rank_tr", "pl_mark_d1", "pl_mark_d3",
            "pl_runs_since_win", "pl_cycle_s", "pl_repeat", "pl_repeat_nfp", "pl_trip_change", "pl_first_trip",
            "pl_tr_prize_up_s", "pl_tr_prize_down_s", "pl_tr_mark_drop_s", "pl_tr_cycle_s", "pl_tr_repeat_s",
            "pl_tr_trip_up_s", "pl_tr_trip_down_s"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "prize_money", "official_rating",
         "dist_furlongs", "placing_numerical", "number_of_runners", "bfsp"]
K = 20.0
HALF = 1095.0
BREAK_DAYS = 60
REPEAT_WINDOW = 30
YEAR = 365.0
MIN_AGO = 300
# the test history is 60 days long: a "year" of 20 days exercises the repeat reading (mechanics only)
TEST_REACH = {"YEAR": 20.0, "MIN_AGO": 15, "REPEAT_WINDOW": 4}


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _key(*parts) -> np.ndarray:
    bad = np.zeros(len(parts[0]), dtype=bool)
    out = np.zeros(len(parts[0]), dtype=np.int64)
    for p in parts:
        p = np.asarray(p, dtype=np.int64)
        bad |= p < 0
        out = out * (int(p.max()) + 2) + np.where(p < 0, 0, p)
    return np.where(bad, -1, pd.factorize(out, sort=True)[0]).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    horse, trainer, course = _codes(df["horse_name"]), _codes(_col(df, "trainer")), _codes(_col(df, "track"))
    H = _Days(horse, day)
    nb = len(H.u)
    blocks = np.arange(nb)
    prev = np.clip(blocks - 1, 0, None)
    has_prev = H.valid & (H.pos >= 1)

    def back(v, j=1, how="first"):
        b = H.per_block(v, how=how)
        return np.where(H.valid & (H.pos >= j), b[np.clip(blocks - j, 0, None)], np.nan)

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

    cols: dict[str, np.ndarray] = {}
    prize = pd.to_numeric(_col(df, "prize_money"), errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.where(prize > 0, np.log(np.where(prize > 0, prize, 1.0)), np.nan)
    usual, _ = asof_decayed_mean(horse, day, lp, horse, day, halflife_days=730.0)
    cols["pl_prize_vs_usual"] = lp - usual
    cols["pl_prize_vs_last"] = lp - H.to_rows(back(lp))
    td = pd.Series(lp, index=df.index).groupby([pd.Series(trainer, index=df.index), pd.Series(day, index=df.index)])
    cols["pl_prize_rank_tr"] = np.where((trainer >= 0) & np.isfinite(lp),
                                        (td.rank(ascending=False, method="min") / td.transform("count")).to_numpy(float), np.nan)

    mark = pd.to_numeric(_col(df, "official_rating"), errors="coerce").to_numpy(float)
    mark = np.where(mark > 0, mark, np.nan)
    m1, m2, m3 = (H.to_rows(back(mark, j)) for j in (1, 2, 3))
    cols["pl_mark_d1"] = mark - m1
    hi3 = np.fmax(np.fmax(m1, m2), m3)
    cols["pl_mark_d3"] = mark - hi3
    won_b = np.nan_to_num(H.per_block((pos == 1).astype(float), how="max")) > 0
    lw = H.last_before(won_b)
    cols["pl_runs_since_win"] = H.to_rows(np.where(lw >= 0, blocks - lw, np.nan))

    # the point of the campaign: racing days since a break of BREAK_DAYS (0 = first run back), capped at 3
    gap = np.where(has_prev, H.day - H.day[prev], np.inf)
    fresh = (~has_prev) | (gap >= BREAK_DAYS)
    idx = np.where(fresh & H.valid, blocks, -1)
    since = blocks - pd.Series(idx).groupby(H.key).cummax().to_numpy()
    cyc = H.to_rows(np.where(H.valid, np.minimum(since, 3), np.nan))
    cyc_i = np.where(np.isfinite(cyc), cyc, -1).astype(np.int64)
    at, at_n = asof_decayed_mean(_key(horse, cyc_i), day, s, _key(horse, cyc_i), day)
    allm, all_n = asof_decayed_mean(horse, day, s, horse, day)
    cols["pl_cycle_s"] = np.where((at_n > 0) & (all_n > 0), at - allm, np.nan)

    # back for a race it ran in before: this course and trip band, within REPEAT_WINDOW days of the date, 300+ days ago
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    tb = pd.Series(bands(dist, DIST_BANDS)).fillna(-1).to_numpy().astype(np.int64)
    rep, rep_nfp = np.zeros(len(df)), np.full(len(df), np.nan)
    kk = _key(horse, course, tb)
    o = np.lexsort((day, kk))
    ks, ds, fs = kk[o], day[o], nfp[o]
    for i in range(1, 6):                                     # up to five earlier runs in the same cell
        same = np.r_[np.zeros(i, dtype=bool), ks[i:] == ks[:-i]] & (ks >= 0)
        ago = np.r_[np.zeros(i), ds[i:] - ds[:-i]]
        off = np.abs(((ago + YEAR / 2) % YEAR) - YEAR / 2)
        hit = same & (ago >= MIN_AGO) & (off <= REPEAT_WINDOW)
        was = np.r_[np.full(i, np.nan), fs[:-i]]
        rep[o] = np.where(hit, 1.0, rep[o])
        rep_nfp[o] = np.where(hit, np.fmax(rep_nfp[o], was), rep_nfp[o])
    cols["pl_repeat"], cols["pl_repeat_nfp"] = rep, rep_nfp

    with np.errstate(invalid="ignore", divide="ignore"):
        ld = np.where(dist > 0, np.log(np.where(dist > 0, dist, 1.0)), np.nan)
    cols["pl_trip_change"] = ld - H.to_rows(back(ld))
    hb = _Days(_key(horse, tb), day)
    cols["pl_first_trip"] = np.where((horse >= 0) & (tb >= 0), (hb.to_rows(hb.pos.astype(float)) == 0).astype(float), np.nan)

    def yard(flag, cell=None):
        key = trainer if cell is None else _key(trainer, cell)
        m, cnt = asof_decayed_mean(key, day, np.where(flag, s, np.nan), key, day, halflife_days=HALF)
        return np.where(flag, np.nan_to_num(m) * np.nan_to_num(cnt) / (np.nan_to_num(cnt) + K), 0.0)

    up, down = cols["pl_prize_vs_usual"] >= np.log(1.5), cols["pl_prize_vs_usual"] <= np.log(2.0 / 3.0)
    cols["pl_tr_prize_up_s"], cols["pl_tr_prize_down_s"] = yard(up), yard(down)
    cols["pl_tr_mark_drop_s"] = yard(cols["pl_mark_d3"] <= -5)
    cols["pl_tr_cycle_s"] = yard(cyc_i >= 0, cyc_i)
    cols["pl_tr_repeat_s"] = yard(rep == 1)
    cols["pl_tr_trip_up_s"] = yard(cols["pl_trip_change"] >= np.log(1.2))
    cols["pl_tr_trip_down_s"] = yard(cols["pl_trip_change"] <= -np.log(1.2))

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
