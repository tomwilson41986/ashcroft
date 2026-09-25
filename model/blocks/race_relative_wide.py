"""The rest of the field readings: every input the engine ranks, as a distance from this field.

race_relative (iteration 35: -0.0074 on the price forecast) put thirty inputs
against the field as z-scores and gaps to the best. The engine ranks some
seventy inputs within the race (custom_metrics rank_configs), and a rank keeps
the order and throws away the distance: first by a street and first by a
whisker rank the same. This block gives the other sixty-two the same two
readings, the strongest form-variant windows too, and the shape of the field:

    rw_<m>_z            (value - the race's mean) / the race's standard deviation
    rw_<m>_gap          value - the race's highest value
    rw_fv_<m>_z, _gap   the same for form_variants' strongest windows (built after it)
    rw_field_top2_<m>   the race's best less its second best (how clear the top is),
                        the same for every runner
    rw_field_sd_<m>     the race's standard deviation, the same for every runner

over the runners with a value. Pre-race inputs only, so a card row gets what the
same row gets with its results in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import race_key

#: The engine's ranked inputs race_relative leaves out.
SOURCES = [
    "preracehorsecareerNFP", "LR3NFPtotal", "LR5NFPtotal", "LR10NFPtotal", "preracehorsecareerRB",
    "preracehorsecareerFSARB", "preracehorsecareerFSARB2", "preracehorsecareerWIV", "preracehorsecareerWAX",
    "preracehorsecareerWOA", "preracehorsecareerCWO", "preracehorsecareerRuns", "preracehorsecareerWins",
    "preracehorsecareerPlaces", "LR_EPF", "LR_EPF2", "LR_EPF3", "Jockey_Career_EPF", "trainer_Career_EPF",
    "Horse_Career_EPF", "FinalDSLR", "FSS", "FCS", "PFD3", "PFD5", "PFD10", "WPMRF3", "WPMRF5", "WPMRF10",
    "PMW5", "OFS10", "trainerjockeycareerWIV", "preracetrainercareerWIV", "preracetrainercareerWAX",
    "preracetrainercareerWOA", "preracetrainercareerCWO", "preracejockeycareerWIV", "preracejockeycareerWAX",
    "preracejockeycareerWOA", "preracejockeycareerCWO", "totalLRPjockeyindex", "EXP_NFP5", "EXP_RB5",
    "career_residual", "form_slope_3", "career_nfp_std", "dist_from_preferred", "going_from_preferred",
    "weight_vs_avg", "unexposure_score", "sire_avg_nfp", "sire_wiv", "sire_going_nfp", "sire_dist_nfp",
    "damsire_avg_nfp", "preracehorsecareerRSR", "preracehorsecareerLB", "surface_nfp", "horse_track_nfp",
    "or_change", "trainer_sr_14d", "jockey_sr_14d",
]

#: form_variants' strongest windows (iteration 34), read against the field.
FV_SOURCES = ["fv_nfp_e3", "fv_nfp_e6", "fv_nfp_s8", "fv_nfp_fsw_e6", "fv_lbw_e6", "fv_lbc_e6",
              "fv_lbpf_e6", "fv_lblog_e3"]

#: The field's shape: how clear its best is, how spread it is.
TOP2 = ["LR_ORR2", "official_rating", "PMW3", "fv_nfp_e3"]
SPREAD = ["LR_ORR2", "official_rating"]

FEATURES = (
    [f"rw_{m}_{k}" for m in SOURCES for k in ("z", "gap")]
    + [f"rw_fv_{m[3:]}_{k}" for m in FV_SOURCES for k in ("z", "gap")]
    + [f"rw_field_top2_{m}" for m in TOP2]
    + [f"rw_field_sd_{m}" for m in SPREAD]
)
POST_RACE: set[str] = set()
READS = list(dict.fromkeys(["raceid", "race_date", "race_time", "track"] + SOURCES + FV_SOURCES + TOP2 + SPREAD))
AFTER = ("form_variants",)


def _z_gap(v: pd.Series, rk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = v.groupby(rk)
    mean, sd, top = g.transform("mean"), g.transform("std"), g.transform("max")
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 0, (v - mean) / sd, np.where(v.notna() & (sd == 0), 0.0, np.nan))
    return z, (v - top).to_numpy(dtype=float)


def _second(v: pd.Series, rk: np.ndarray) -> np.ndarray:
    """The race's second-highest value (NaN with fewer than two values)."""
    order = pd.DataFrame({"r": rk, "v": v.to_numpy(dtype=float)}).dropna()
    order = order.sort_values(["r", "v"], ascending=[True, False], kind="mergesort")
    second = order.groupby("r")["v"].nth(1)
    second.index = order.loc[second.index, "r"].to_numpy()
    return pd.Series(rk).map(second).to_numpy(dtype=float)


def build(df: pd.DataFrame) -> pd.DataFrame:
    rk = race_key(df).to_numpy()

    def num(m: str) -> pd.Series:
        return pd.to_numeric(df[m], errors="coerce").astype(float) if m in df.columns \
            else pd.Series(np.nan, index=df.index)

    cols: dict[str, np.ndarray] = {}
    for m in SOURCES:
        cols[f"rw_{m}_z"], cols[f"rw_{m}_gap"] = _z_gap(num(m), rk)
    for m in FV_SOURCES:
        cols[f"rw_fv_{m[3:]}_z"], cols[f"rw_fv_{m[3:]}_gap"] = _z_gap(num(m), rk)
    for m in TOP2:
        v = num(m)
        cols[f"rw_field_top2_{m}"] = v.groupby(rk).transform("max").to_numpy(dtype=float) - _second(v, rk)
    for m in SPREAD:
        cols[f"rw_field_sd_{m}"] = num(m).groupby(rk).transform("std").to_numpy(dtype=float)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
