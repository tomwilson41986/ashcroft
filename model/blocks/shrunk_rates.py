"""Records on small samples, shrunk toward what the larger sample says: trainers, jockeys, sires and horses by condition.

The owner's question of 25 Sep: can Bayesian approaches handle the different
sample sizes? The engine's cell records are raw means: a trainer 2 from 3 at a
course reads 67%, a horse's one run on soft reads as its soft form. Here each
cell's mean is pulled toward the level above it by its own sample size (the
empirical-Bayes mean of model/shrinkage.py, (sum + k * prior) / (n + k)), so a
small cell says little and a big one speaks for itself. All from EARLIER DAYS
(time-decayed; model/race_shape.asof_decayed_mean):

    sr_tr_recent_win    the trainer's win rate over the last few weeks (half-life
                        14 days), shrunk to its two-year rate (k = 10)
    sr_tr_course_win    the trainer's win rate at this course, shrunk to its overall (k = 20)
    sr_jk_course_win    the same for the jockey
    sr_tj_win           the trainer and jockey together, shrunk to the mean of their
                        two rates (k = 20)
    sr_tr_course_ae     the trainer's wins against the market's expectation at this
                        course (won - the BSP's chance), shrunk to 0 (k = 50)
    sr_jk_recent_ae     the jockey's, over the last month or so (half-life 30), shrunk to 0 (k = 30)
    sr_sire_trip_nfp    the sire's progeny's NFP at this trip band, shrunk to the
                        sire's overall, itself shrunk to 0.5 (k = 30 each)
    sr_sire_going_nfp   the same on this going (fast, soft, all-weather)
    sr_horse_course_nfp the horse's NFP at this course, shrunk to its career NFP,
                        itself shrunk to 0.5 (k = 2 and 3)
    sr_horse_trip_nfp   the same at this trip band
    sr_horse_going_nfp  the same on this going
    sr_horse_course_n, sr_horse_trip_n, sr_horse_going_n  the (decayed) runs behind them

A run's own result never enters its own features: a 06:00 card and the same
row with its result in get the same values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.draw_v2 import going_group
from model.form_windows import run_measures
from model.freshness_features import _col
from model.race_shape import DIST_BANDS, _codes, asof_decayed_mean, bands, day_index, race_key
from model.shrinkage import shrunk_mean

FEATURES = [
    "sr_tr_recent_win", "sr_tr_course_win", "sr_jk_course_win", "sr_tj_win",
    "sr_tr_course_ae", "sr_jk_recent_ae",
    "sr_sire_trip_nfp", "sr_sire_going_nfp",
    "sr_horse_course_nfp", "sr_horse_trip_nfp", "sr_horse_going_nfp",
    "sr_horse_course_n", "sr_horse_trip_n", "sr_horse_going_n",
]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "stallion",
         "going_description", "number_of_runners", "placing_numerical", "LB", "total_dst_bt", "dist_furlongs",
         "official_rating", "bfsp"]

LONG = 730.0            # days: the level a small cell is pulled toward


def _key(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s))


def _pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One code per (a, b), -1 where either is missing."""
    ok = (a >= 0) & (b >= 0)
    return np.where(ok, pd.factorize(pd.Series(a.astype(str)) + "|" + pd.Series(b.astype(str)))[0], -1)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    nfp = run_measures(df)["nfp"]
    # a win or not, a non-finisher counting as a loss, in a race with a result;
    # the market's chance from the BSP, normalised over the race's priced runners
    race = pd.factorize(race_key(df))[0]
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    has_result = pd.Series(np.isfinite(pos) & (pos > 0)).groupby(race).transform("any").to_numpy()
    won = np.where(has_result, (pos == 1).astype(float), np.nan)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(dtype=float)
    p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
    pn = (p / p.groupby(race).transform("sum")).to_numpy(dtype=float)
    ae = won - pn
    trainer, jockey = _key(_col(df, "trainer")), _key(_col(df, "jockey_name"))
    horse, sire, track = _key(df["horse_name"]), _key(_col(df, "stallion")), _key(_col(df, "track"))
    dband = bands(pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(dtype=float), DIST_BANDS)
    dband = np.where(np.isfinite(dband), dband, -1).astype(np.int64)
    going = pd.factorize(going_group(df))[0].astype(np.int64)

    def mean(key, value, half):
        return asof_decayed_mean(key, day, value, key, day, halflife_days=half)

    def cell(key, value, prior, k, half=LONG):
        mm, nn = mean(key, value, half)
        return shrunk_mean(np.nan_to_num(mm) * nn, nn, prior, k), nn

    cols: dict[str, np.ndarray] = {}
    # trainers and jockeys: win rates and wins against the market
    tr_long, tr_n = mean(trainer, won, LONG)
    tr_long = shrunk_mean(np.nan_to_num(tr_long) * tr_n, tr_n, 0.1, 20)            # a new yard: 10%
    jk_long, jk_n = mean(jockey, won, LONG)
    jk_long = shrunk_mean(np.nan_to_num(jk_long) * jk_n, jk_n, 0.1, 20)
    cols["sr_tr_recent_win"], _ = cell(trainer, won, tr_long, 10, half=14.0)
    cols["sr_tr_course_win"], _ = cell(_pair(trainer, track), won, tr_long, 20)
    cols["sr_jk_course_win"], _ = cell(_pair(jockey, track), won, jk_long, 20)
    cols["sr_tj_win"], _ = cell(_pair(trainer, jockey), won, 0.5 * (tr_long + jk_long), 20)
    cols["sr_tr_course_ae"], _ = cell(_pair(trainer, track), ae, 0.0, 50)
    cols["sr_jk_recent_ae"], _ = cell(jockey, ae, 0.0, 30, half=30.0)

    # sires: the progeny by trip and going, toward the sire's overall, toward 0.5
    s_all, s_n = mean(sire, nfp, LONG)
    s_all = shrunk_mean(np.nan_to_num(s_all) * s_n, s_n, 0.5, 30)
    cols["sr_sire_trip_nfp"], _ = cell(_pair(sire, dband), nfp, s_all, 30)
    cols["sr_sire_going_nfp"], _ = cell(_pair(sire, going), nfp, s_all, 30)

    # the horse itself, by course, trip and going, toward its career, toward 0.5
    h_all, h_n = mean(horse, nfp, np.inf)
    h_all = shrunk_mean(np.nan_to_num(h_all) * h_n, h_n, 0.5, 3)
    for name, key in (("course", _pair(horse, track)), ("trip", _pair(horse, dband)), ("going", _pair(horse, going))):
        v, n = cell(key, nfp, h_all, 2, half=np.inf)
        cols[f"sr_horse_{name}_nfp"] = np.where(horse >= 0, v, np.nan)
        cols[f"sr_horse_{name}_n"] = np.where(key >= 0, n, np.nan)

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
