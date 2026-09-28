"""The sire's and damsire's progeny over recent windows: how their runners are going now, as connection windows read the yard.

Connection windows (connection_windows.py) passed on the 944 at the served
recipe (iteration 68): the trainer's and jockey's recent runners carry what
career rates do not. The engine reads the sire's progeny only as career rates
and aptitudes (sire_avg_nfp, sire_wiv, sire_dist_nfp, ...), and the sire
aptitude block (iteration 52) found nothing beyond them. What a race reader and
the market weigh for a lightly raced horse is how the sire's runners are going
now: a first-crop sire having a good year. For the sire (sr) and the damsire
(ds), over their progeny's runs on days before today, most recent last:

    sw_<e>_n          progeny runs so far
    sw_<e>_nfp_car    mean normalised finishing position of all of them (a
                      non-finisher 0), shrunk to 0.5 by fifty runs
    sw_<e>_nfp_l100   the same over the last 100 runs (NaN with none)
    sw_<e>_nfp_l500   the same over the last 500
    sw_<e>_ae_l300    wins less BSP chances over the last 300, per run, shrunk
                      to 0 by fifty runs
    sw_sr_young_nfp   the sire's two- and three-year-old runners' last 200 runs,
                      NFP, shrunk to the sire's own last-500 NFP by twenty
    sw_sr_nfp_rank    the sire's last-500 NFP ranked in today's race, 1 the best

Today's results never enter: runs count only from days before the row's day,
summed in a fixed order (by day, race, horse) with running sums that restart at
each key, so a 06:00 card and the same rows with results in agree to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.connection_grains import _Keyed, _pair
from model.blocks.connection_windows import NFP_PRIOR, _codes, _runner_values
from model.freshness_features import _col
from model.race_shape import day_index, race_key

ENTITIES = {"sr": "stallion", "ds": "dam_stallion"}
FEATURES = ([f"sw_{e}_{s}" for e in ENTITIES for s in ("n", "nfp_car", "nfp_l100", "nfp_l500", "ae_l300")]
            + ["sw_sr_young_nfp", "sw_sr_nfp_rank"])
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "stallion", "dam_stallion", "horse_age",
         "number_of_runners", "placing_numerical", "bfsp", "LB", "total_dst_bt", "dist_furlongs"]
K_CAR = 50.0        # shrinkage of the career NFP and the A-E, in runs
K_YOUNG = 20.0      # shrinkage of the young progeny's NFP to the sire's own level, in runs


def build(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    day = day_index(df)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    horse = _codes(_col(df, "horse_name"))
    tie = race * (int(horse.max()) + 2) + (horse + 1)          # a fixed order within a day
    vals = _runner_values(df)
    nfp, ae = vals["nfp"], vals["ae"]
    ones = np.ones(n_rows)
    age = pd.to_numeric(_col(df, "horse_age"), errors="coerce").to_numpy(dtype=float)
    cols = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for e, column in ENTITIES.items():
            K = _Keyed(_codes(_col(df, column)), day, tie)
            cols[f"sw_{e}_n"] = K.window(ones, K.last_runners(None))[1]
            s, c = K.window(nfp, K.last_runners(None))
            cols[f"sw_{e}_nfp_car"] = (s + K_CAR * NFP_PRIOR) / (c + K_CAR)
            for w in (100, 500):
                s, c = K.window(nfp, K.last_runners(w))
                cols[f"sw_{e}_nfp_l{w}"] = np.where(c > 0, s / c, np.nan)
            s, c = K.window(ae, K.last_runners(300))
            cols[f"sw_{e}_ae_l300"] = s / (c + K_CAR)
        sire = _codes(_col(df, "stallion"))
        young = np.where(np.isfinite(age) & (age <= 3), 1, 0).astype(np.int64)
        Y = _Keyed(_pair(sire, young), day, tie)
        s, c = Y.window(nfp, Y.last_runners(200))
        level = np.where(np.isfinite(cols["sw_sr_nfp_l500"]), cols["sw_sr_nfp_l500"], NFP_PRIOR)
        young_nfp = (s + K_YOUNG * level) / (c + K_YOUNG)
        cols["sw_sr_young_nfp"] = np.where(young == 1, young_nfp, np.nan)   # read for the young runners only
        rank = pd.Series(cols["sw_sr_nfp_l500"]).groupby(race).rank(ascending=False, method="min",
                                                                      na_option="bottom")
        cols["sw_sr_nfp_rank"] = rank.to_numpy(dtype=float)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
