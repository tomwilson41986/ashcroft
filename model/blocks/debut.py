"""Debutants: a forecast of the price of a horse with little or no form, and the field's book with it in.

The forecaster's largest errors are in the races of lightly raced horses (maidens, novices, bumpers: 0.50-0.57
against 0.34-0.37 in handicaps). There the market rating has nothing to read: an unraced horse sits at its rivals'
average. What prices it is the pedigree, the yard's record with its unknowns, the rider booked and the race (the
owner, 12 Nov 2026: "optimise handling of debutants, first time out"). A first-stage boosted model, fitted only on
the runs of horses with at most MAX_RUNS earlier runs, forecasts where the market put them: x = log(runners x the
runner's share of the Betfair SP book), from

    pedigree       the Bayesian pedigree record against the market rating: sire, sire's debutants, dam, dam's
                   debutants, dam sire, the cross, the combined prior, and the uncertainty (pedigree_bayes.py)
    the yard       how the market has priced the yard's debutants, second- and later-time-outers (debut_market.py),
                   and the trainer's and the jockey's pull on the price (connections_rating.py)
    the horse      its earlier runs (0, 1, 2), its age, the month; from its second run, what the market made of its
                   earlier runs (market_metrics.py) and the market rating's own reading
    the race       runners, prize, trip, code, the share of the field that is unexposed

refitted every fold on the four years before it (model/blocks/expected_time.forecast): no forecast has seen its
own day or a later one.

    db_pred      the forecast x for a horse with at most MAX_RUNS earlier runs (NaN for the others)
    db_vs_mr     db_pred less the market rating's reading of the same horse
    db_x         the field's book rebuilt: each unexposed horse at db_pred, every other at its market rating;
                 log(runners x the softmax share) - the price position this view gives every runner in the race
    db_x_vs_mr   db_x less the market rating's reading: how the exposed horses' prices move when the unknowns in
                 their race are priced
    db_unexp     unexposed runners in the race, as a share
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import expected_time as _et
from model.freshness_features import _col
from model.race_shape import day_index, race_code, race_key

FEATURES = ["db_pred", "db_vs_mr", "db_x", "db_x_vs_mr", "db_unexp"]
POST_RACE: set[str] = set()
FROM = ["pb_sire_m", "pb_sire_m_sd", "pb_sire_debut_m", "pb_dam_m", "pb_dam_m_sd", "pb_dam_debut_m", "pb_ds_m", "pb_ss_m",
        "pb_cross_m", "pb_ped_m", "pb_sire_n", "pb_dam_n", "dm_tr_debut_mkt", "dm_tr_debut_n", "dm_tr_second_mkt",
        "dm_tr_early_mkt", "dm_trc_debut_mkt", "dm_jk_debut_mkt", "dm_stage_mkt", "dm_race_debut_share",
        "dm_race_unexposed_share", "cr_tr_m", "cr_jk_m", "cr_tr_m_sd", "cr_jk_m_sd", "mm_m_l1", "mm_m_m3", "mm_n_l1",
        "mr_lshare", "mr_read_share", "mr_n"]
READS = sorted({"raceid", "race_date", "race_time", "track", "horse_name", "bfsp", "career_runs", "horse_age",
                "number_of_runners", "prize_money", "dist_furlongs", "race_type", "surface_type"} | set(FROM))
AFTER = ("market_rating", "pedigree_bayes", "debut_market", "connections_rating", "market_metrics")
MAX_RUNS = 2
FOLD_DAYS = 365
MIN_TRAIN = 5000
N_TREES = 250
MIN_CHILD = 100
TEST_REACH = {"FOLD_DAYS": 10, "MIN_TRAIN": 40, "N_TREES": 20, "MIN_CHILD": 5, "MAX_RUNS": 40}


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    rk = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    horse = _et._name_codes(df["horse_name"])
    runs = pd.to_numeric(_col(df, "career_runs"), errors="coerce").to_numpy(float)
    unexp = np.isfinite(runs) & (runs <= MAX_RUNS)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=df.index)
    tot = inv.groupby(rk).transform("sum").to_numpy()
    rows = pd.Series(1.0, index=df.index).groupby(rk).transform("sum").to_numpy()
    npr = pd.Series(priced.astype(float), index=df.index).groupby(rk).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(priced & (tot > 0) & (npr >= 0.9 * rows), np.log(rows * inv.to_numpy() / np.where(tot > 0, tot, 1.0)), np.nan)

    X = {c: pd.to_numeric(_col(df, c), errors="coerce").to_numpy(float) for c in FROM}
    X["runs"], X["runners"] = runs, n
    X["age"] = pd.to_numeric(_col(df, "horse_age"), errors="coerce").to_numpy(float)
    X["month"] = pd.to_datetime(df["race_date"], errors="coerce").dt.month.to_numpy(float)
    prize = pd.to_numeric(_col(df, "prize_money"), errors="coerce").to_numpy(float)
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        X["log_prize"] = np.where(prize > 0, np.log(np.where(prize > 0, prize, 1.0)), np.nan)
        X["log_trip"] = np.where(dist > 0, np.log(np.where(dist > 0, dist, 1.0)), np.nan)
    X["code"] = pd.factorize(race_code(df), sort=True)[0].astype(float)
    share_u = pd.Series(unexp.astype(float), index=df.index).groupby(rk).transform("mean").to_numpy()
    X["unexp_share"] = share_u
    # rounded: the inputs carry summation noise around zero (1e-17, either sign) that a tree can split on
    Xf = pd.DataFrame(X, index=df.index).round(6)

    y = np.where(unexp, x, np.nan)                                  # only the unexposed are learned from
    order = np.lexsort((horse, rk.to_numpy(), day))
    saved = (_et.FOLD_DAYS, _et.MIN_TRAIN, _et.N_TREES, _et.MIN_CHILD)
    try:
        _et.FOLD_DAYS, _et.MIN_TRAIN, _et.N_TREES, _et.MIN_CHILD = FOLD_DAYS, MIN_TRAIN, N_TREES, MIN_CHILD
        pred = _et.forecast(Xf, y, day, order)
    finally:
        _et.FOLD_DAYS, _et.MIN_TRAIN, _et.N_TREES, _et.MIN_CHILD = saved
    pred = np.where(unexp, pred, np.nan)
    mr = X["mr_lshare"]
    view = pd.Series(np.where(np.isfinite(pred), pred, mr), index=df.index)
    ok = view.notna().groupby(rk).transform("all").to_numpy()
    top = view.groupby(rk).transform("max")
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        e = np.exp(view - top)
        book = np.log(rows * (e / e.groupby(rk).transform("sum")).to_numpy(float))
    book = np.where(ok, book, np.nan)
    cols = {"db_pred": pred, "db_vs_mr": pred - mr, "db_x": book, "db_x_vs_mr": book - mr, "db_unexp": share_u}
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
