"""Sire seasons and class: a young sire judged against his own cohort, and a class index with the opportunity behind it.

Two ideas from the owner's reading of 12 Nov 2026 that the Bayesian pedigree block does not hold:

1. Cohort priors. A first-season sire's stock is not an average sire's stock with few runs: first-season sires as a
   group are priced and perform in their own way, as do second- and third-season sires. So a sire's record on a
   residual is pulled toward the record of his cohort (sires in the same season of their careers, over earlier
   days), by a prior strength K chosen from GRID_K as the one that best predicts a sire's NEXT season from his
   seasons so far, on the seasons before FIT_UNTIL (the default where they are too few). A sire's first season is
   the year of his first runner in the data held; a sire already running in the data's first year counts as
   established (4+).

       ss2_season                  the sire's season: 1, 2, 3, or 4 for established
       ss2_m  ss2_s                his posterior on m (the price against the market rating) and s (finishing
                                   position against the price), with the cohort as the prior
       ss2_cohort_m  ss2_cohort_s  the cohort's own record: what the prior is
       ss2_trend_s                 this season's record on s less last season's (the stock improving or declining)
       ss2_volume                  log of his runs this season so far over his runs by the same day last season

2. A class index with opportunity. The winners of black-type races weighted by the grade's par (Group or Grade 1
   120, 2 113, 3 109, Listed and other black type 104), over the sire's runs: sc = sum of pars of black-type wins /
   runs. And the opportunity: the mean par of the black-type races his stock ran in, and the share of his runs that
   were in black type. For the sire and the dam sire, decayed over HALF days.

       ss2_class  ss2_ds_class             the class index
       ss2_opp  ss2_opp_share              the sire's opportunity: mean par of his black-type runs; their share
       ss2_class_vs_opp                    the index over what the opportunity alone would give (wins per
                                           black-type run), shrunk: achievement for the chances given

Earlier days only for every record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import _resid
from model.connections import is_black_type
from model.freshness_features import _col
from model.race_shape import asof_decayed_mean, day_index, race_key

FEATURES = ["ss2_season", "ss2_m", "ss2_s", "ss2_cohort_m", "ss2_cohort_s", "ss2_trend_s", "ss2_volume", "ss2_class",
            "ss2_ds_class", "ss2_opp", "ss2_opp_share", "ss2_class_vs_opp"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "stallion", "dam_stallion", "placing_numerical",
         "number_of_runners", "bfsp", "major", "race_name", "race_class", "mr_lshare"]
AFTER = ("market_rating",)
FIT_UNTIL = "2022-01-01"
GRID_K = (10.0, 30.0, 60.0, 120.0, 250.0)
K_DEFAULT = 60.0
HALF = 4 * 365.0
K_OPP = 20.0
PARS = ((r"(?:group|grade|gr\.?|g)\s*(?:1|i)\b", 120.0), (r"(?:group|grade|gr\.?|g)\s*(?:2|ii)\b", 113.0),
        (r"(?:group|grade|gr\.?|g)\s*(?:3|iii)\b", 109.0))
TEST_REACH = {"YEAR_DAYS": 15}
YEAR_DAYS = 365


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _key(a, b):
    bad = (a < 0) | (b < 0)
    return np.where(bad, -1, pd.factorize(a.astype(np.int64) * 100003 + np.where(b < 0, 0, b), sort=True)[0]).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    race = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    sire, ds = _codes(_col(df, "stallion")), _codes(_col(df, "dam_stallion"))
    ok = day >= 0
    d0 = int(day[ok].min()) if ok.any() else 0
    yr = np.where(ok, (day - d0) // YEAR_DAYS, -1).astype(np.int64)        # seasons counted from the data's first day
    first = pd.Series(np.where((sire >= 0) & ok, yr, np.nan)).groupby(sire).transform("min").to_numpy(float)
    season = np.where((sire >= 0) & ok, np.where(first == 0, 4.0, np.minimum(yr - first + 1, 4.0)), np.nan)
    sea = np.where(np.isfinite(season), season, -1).astype(np.int64)

    s, _ = _resid.price_residual(df, day, race)
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(float)
    priced = np.isfinite(bsp) & (bsp > 1.0)
    inv = pd.Series(np.where(priced, 1.0 / np.where(priced, bsp, 1.0), 0.0), index=df.index)
    tot = inv.groupby(race).transform("sum").to_numpy()
    rows = pd.Series(1.0, index=df.index).groupby(race).transform("sum").to_numpy()
    npr = pd.Series(priced.astype(float), index=df.index).groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(priced & (tot > 0) & (npr >= 0.9 * rows), np.log(rows * inv.to_numpy() / np.where(tot > 0, tot, 1.0)), np.nan)
    m = np.clip(x - pd.to_numeric(_col(df, "mr_lshare"), errors="coerce").to_numpy(float), -3.0, 3.0)

    cols: dict[str, np.ndarray] = {"ss2_season": season}
    cut = int(np.datetime64(FIT_UNTIL, "D").astype(np.int64))
    sy = _key(sire, yr)
    for tag, v in (("m", m), ("s", s)):
        sm, sn = asof_decayed_mean(sire, day, v, sire, day, halflife_days=HALF)
        cm, cn = asof_decayed_mean(sea, day, v, sea, day, halflife_days=HALF)
        sm, sn, cm = np.nan_to_num(sm), np.nan_to_num(sn), np.nan_to_num(cm)
        # K: which prior strength best predicts a sire-season's mean from the sire's record at the season's start
        g = pd.DataFrame({"sy": sy, "v": v, "day": day, "sm": sm, "sn": sn, "cm": cm})[(sy >= 0) & np.isfinite(v) & (day < cut)]
        k = K_DEFAULT
        if len(g) > 5000:
            a = g.sort_values("day").groupby("sy").agg(y=("v", "mean"), n=("v", "size"), sm=("sm", "first"), sn=("sn", "first"),
                                                       cm=("cm", "first"))
            a = a[a.n >= 20]
            if len(a) >= 30:
                err = {kk: float(np.average(((a.sn * a.sm + kk * a.cm) / (a.sn + kk) - a.y) ** 2, weights=a.n)) for kk in GRID_K}
                k = min(err, key=err.get)
        cols[f"ss2_{tag}"] = np.where(sire >= 0, (sn * sm + k * cm) / (sn + k), np.nan)
        cols[f"ss2_cohort_{tag}"] = np.where(sea >= 0, np.where(cn > 0, cm, np.nan), np.nan)
    this_m, this_n = asof_decayed_mean(sy, day, s, sy, day)
    prev_key = _key(sire, np.where(yr > 0, yr - 1, -1))
    last_m, last_n = asof_decayed_mean(sy, day, s, prev_key, day)
    cols["ss2_trend_s"] = np.where((this_n >= 10) & (last_n >= 10), this_m - last_m, np.nan)
    one = np.ones(len(df))
    _, runs_now = asof_decayed_mean(sy, day, one, sy, day)
    _, runs_then = asof_decayed_mean(sy, day, one, prev_key, np.where(ok, day - YEAR_DAYS, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["ss2_volume"] = np.where((np.nan_to_num(runs_now) > 0) & (np.nan_to_num(runs_then) > 0),
                                      np.log(runs_now / runs_then), np.nan)

    # the class index and the opportunity
    text = (_col(df, "major").fillna("").astype(str) + " " + _col(df, "race_name").fillna("").astype(str)).str.lower()
    bt = is_black_type(df).to_numpy()
    par = np.where(bt, 104.0, 0.0)
    for rx, value in reversed(PARS):
        par = np.where(bt & text.str.contains(rx, regex=True).to_numpy(), value, par)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    won_par = np.where(pos == 1, par, 0.0)
    for tag, key in (("ss2_class", sire), ("ss2_ds_class", ds)):
        cmean, n_all = asof_decayed_mean(key, day, won_par, key, day, halflife_days=HALF)
        cols[tag] = np.where((key >= 0) & (np.nan_to_num(n_all) > 0), cmean, np.nan)
    opp, n_bt = asof_decayed_mean(sire, day, np.where(bt, par, np.nan), sire, day, halflife_days=HALF)
    share, n_all = asof_decayed_mean(sire, day, bt.astype(float), sire, day, halflife_days=HALF)
    wins_bt, _ = asof_decayed_mean(sire, day, np.where(bt, (pos == 1).astype(float), np.nan), sire, day, halflife_days=HALF)
    all_w, all_n = asof_decayed_mean(np.zeros(len(df), dtype=np.int64), day, np.where(bt, (pos == 1).astype(float), np.nan),
                                     np.zeros(len(df), dtype=np.int64), day, halflife_days=HALF)
    cols["ss2_opp"] = np.where((sire >= 0) & (np.nan_to_num(n_all) > 0), np.nan_to_num(opp), np.nan)   # 0: no black-type runs
    cols["ss2_opp_share"] = np.where((sire >= 0) & (np.nan_to_num(n_all) > 0), share, np.nan)
    nb = np.nan_to_num(n_bt)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (np.nan_to_num(wins_bt) * nb + K_OPP * np.nan_to_num(all_w)) / (nb + K_OPP)
        cols["ss2_class_vs_opp"] = np.where(sire >= 0, np.where(np.nan_to_num(all_w) > 0, rate / all_w, 1.0), np.nan)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
