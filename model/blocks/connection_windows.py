"""Form windows for the connections: the trainer's and the jockey's recent runners, as the horse's form windows read the horse.

The form windows (model/form_windows.py) gave the largest gain of any block,
reading each horse's past runs over a ladder of windows. The connections have
only career strike rates, 14- and 30-day strike rates and career win indices,
and the specification's trainerNFPrank / jockeyNFPrank (normalised finishing
position of the connection's runners, ranked in the race) were never built.
For the trainer (tr) and the jockey (jk), over their runners on days before
today, most recent last:

    cw_<e>_n            runners so far
    cw_<e>_nfp_car      mean normalised finishing position of all of them (a
                        non-finisher 0), shrunk to 0.5 by twenty runners
    cw_<e>_nfp_l20      the same over the last 20 runners (NaN with none)
    cw_<e>_nfp_l100     the same over the last 100
    cw_<e>_lbs_l100     the last 100 runners' mean pounds behind the winner
                        (capped at 20 lb, finishers only)
    cw_<e>_ae_l100      the last 100 runners' wins less their BSP chances, per
                        run, shrunk to 0 by twenty runners
    cw_<e>_nfp_rank     cw_<e>_nfp_car ranked in today's race, 1 the best,
                        missing last (the specification's trainerNFPrank and
                        jockeyNFPrank; RB is NFP in this data)

Today's results never enter: a connection's runners count only from the days
before the row's day, summed in a fixed order (by day, race, horse), so a
06:00 card and the same rows with results in agree to the bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import form_lines
from model.form_windows import run_measures
from model.freshness_features import _col
from model.race_shape import day_index, race_key

ENTITIES = {"tr": "trainer", "jk": "jockey_name"}
STATS = ("n", "nfp_car", "nfp_l20", "nfp_l100", "lbs_l100", "ae_l100", "nfp_rank")
FEATURES = [f"cw_{e}_{s}" for e in ENTITIES for s in STATS]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "jockey_name", "number_of_runners",
         "placing_numerical", "bfsp", "LB", "total_dst_bt", "dist_furlongs"]
K = 20.0            # shrinkage, in runners
NFP_PRIOR = 0.5


def _codes(s: pd.Series) -> np.ndarray:
    """Order-independent codes; -1 for a missing name."""
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _runner_values(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each run's normalised finishing position (a non-finisher 0 where the race has a result), pounds behind the
    winner and wins less the BSP chance (post-race: they feed later days only)."""
    m = run_measures(df)
    race = pd.factorize(race_key(df))[0]
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    fin = pos > 0
    has_result = pd.Series(fin).groupby(race).transform("any").to_numpy()
    nfp0 = np.where(fin, m["nfp"], np.where(has_result, 0.0, np.nan))
    ae = form_lines._values(df)["ae"]
    ae = np.where(has_result, ae, np.nan)
    return {"nfp": nfp0, "lbs": m["lbs"], "ae": ae}


def _windows(key: np.ndarray, day: np.ndarray, tie: np.ndarray, vals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """For every row, sums and counts of each value over the key's runners on earlier days: all of them, and the last
    20 and 100 (by date, then the fixed tiebreak)."""
    n = len(key)
    ok = (key >= 0) & (day >= 0)
    order = np.lexsort((tie, day, key))
    k_s, d_s = key[order], day[order]
    idx = np.arange(n)
    new_key = np.r_[True, k_s[1:] != k_s[:-1]]
    new_day = new_key | np.r_[True, d_s[1:] != d_s[:-1]]
    key_start = np.maximum.accumulate(np.where(new_key, idx, 0))
    day_start = np.maximum.accumulate(np.where(new_day, idx, 0))
    out = {"n": np.empty(n)}
    out["n"][order] = (day_start - key_start).astype(float)
    # running sums restart at each key, so a key's sums read its own runners only: a global running sum
    # differenced would move in the last bits whenever another key's values changed
    def running(x):
        return pd.Series(x).groupby(k_s, sort=False).cumsum().to_numpy(dtype=float)

    def upto(cs, i, start):
        """The key's running sum over its runners before position i (0 when there are none)."""
        return np.where(i > start, cs[np.maximum(i - 1, 0)], 0.0)

    for name, v in vals.items():
        v_s = v[order]
        f = np.isfinite(v_s)
        cx, cc = running(np.where(f, v_s, 0.0)), running(f.astype(float))
        for w, lo in (("car", key_start), ("l20", np.maximum(key_start, day_start - 20)),
                      ("l100", np.maximum(key_start, day_start - 100))):
            s = upto(cx, day_start, key_start) - upto(cx, lo, key_start)
            c = upto(cc, day_start, key_start) - upto(cc, lo, key_start)
            out[f"{name}_{w}_sum"] = np.empty(n)
            out[f"{name}_{w}_cnt"] = np.empty(n)
            out[f"{name}_{w}_sum"][order] = s
            out[f"{name}_{w}_cnt"][order] = c
    for c in out:
        out[c] = np.where(ok, out[c], np.nan)
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    day = day_index(df)
    race = pd.factorize(race_key(df), sort=True)[0].astype(np.int64)
    horse = _codes(_col(df, "horse_name"))
    tie = race * (int(horse.max()) + 2) + (horse + 1)          # a fixed order within a day
    vals = _runner_values(df)
    cols = {}
    for e, column in ENTITIES.items():
        w = _windows(_codes(_col(df, column)), day, tie, vals)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"cw_{e}_n"] = w["n"]
            cols[f"cw_{e}_nfp_car"] = (w["nfp_car_sum"] + K * NFP_PRIOR) / (w["nfp_car_cnt"] + K)
            cols[f"cw_{e}_nfp_l20"] = np.where(w["nfp_l20_cnt"] > 0, w["nfp_l20_sum"] / w["nfp_l20_cnt"], np.nan)
            cols[f"cw_{e}_nfp_l100"] = np.where(w["nfp_l100_cnt"] > 0, w["nfp_l100_sum"] / w["nfp_l100_cnt"], np.nan)
            cols[f"cw_{e}_lbs_l100"] = np.where(w["lbs_l100_cnt"] > 0, w["lbs_l100_sum"] / w["lbs_l100_cnt"], np.nan)
            cols[f"cw_{e}_ae_l100"] = w["ae_l100_sum"] / (w["ae_l100_cnt"] + K)
        rank = pd.Series(cols[f"cw_{e}_nfp_car"]).groupby(race).rank(ascending=False, method="min",
                                                                      na_option="bottom")
        cols[f"cw_{e}_nfp_rank"] = rank.to_numpy(dtype=float)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n_rows
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
