"""The official rating at the weights, in every race (the chase finding of 20 Nov 2026, chase-error-1120).

The stack blend is further from the Betfair SP than the morning price only in chases, and there in the better races
(Class 1-3, non-handicaps) of exposed horses at short prices: where the market reads the horses' class off their
marks and the weights they carry. The model has the mark against the weights in handicaps only
(handicap_angles.hca_wt_vs_mark, its largest single handicap feature). In a non-handicap the weights are set by age,
sex and penalties, not by the mark, so the horse with the best mark need not be best at the weights.

A pound of weight is a pound of rating (the handicapper's own scale), so each rated runner's rating at the weights is

    atw = official rating - (weight carried - the lightest weight carried in the race)

with the rating 0 read as unrated and the weight as the card's (pounds, after any claim). Within the race, over the
rated runners:

    atw_or        the rating at the weights           atw_gap     it less the best rival's
    atw_z         (atw - mean) / sd, 3+ rated          atw_rank    its rank, 1 the best
    atw_vs_or     atw_rank less the rank by the bare rating: where the weights reorder the field
    atw_hcap      1 in a handicap (where atw is near level by design), 0 otherwise

Today's card only: the block reads no result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import race_key

FEATURES = ["atw_or", "atw_gap", "atw_z", "atw_rank", "atw_vs_or", "atw_hcap"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "official_rating", "pounds", "race_type",
         "race_name"]
READS_RESULTS = False


def build(df: pd.DataFrame) -> pd.DataFrame:
    race = pd.factorize(race_key(df), sort=True)[0]
    orr = pd.to_numeric(df["official_rating"], errors="coerce").where(lambda s: s > 0).to_numpy(float)
    wt = pd.to_numeric(df["pounds"], errors="coerce").where(lambda s: s > 0).to_numpy(float)
    text = (df["race_type"].fillna("").astype(str) + " " + df["race_name"].fillna("").astype(str)).str.lower()
    hcap = text.str.contains("handicap|nursery").to_numpy().astype(float)
    light = pd.Series(wt).groupby(race).transform("min").to_numpy(float)
    atw = orr - (wt - light)
    v = pd.Series(atw)
    g = v.groupby(race)
    cnt, mean, sd = g.transform("count").to_numpy(float), g.transform("mean").to_numpy(float), g.transform("std").to_numpy(float)
    top = g.transform("max").to_numpy(float)
    second = v.where(v < top).groupby(race).transform("max").to_numpy(float)
    ties = (v == top).groupby(race).transform("sum").to_numpy(float)
    rival = np.where((atw == top) & (ties < 2), second, top)
    rank = g.rank(ascending=False, method="min").to_numpy(float)
    rank_or = pd.Series(np.where(np.isfinite(atw), orr, np.nan)).groupby(race).rank(ascending=False, method="min")
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where((cnt >= 3) & (sd > 0), (atw - mean) / sd, np.where((cnt >= 3) & (sd == 0) & np.isfinite(atw), 0.0, np.nan))
    cols = {"atw_or": atw, "atw_gap": np.where(cnt >= 2, atw - rival, np.nan), "atw_z": z, "atw_rank": rank,
            "atw_vs_or": rank - rank_or.to_numpy(float), "atw_hcap": hcap}
    new = pd.DataFrame({k: np.asarray(x, dtype=float) for k, x in cols.items()}, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
