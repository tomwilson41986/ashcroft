"""A rating on normalised lengths beaten, adjusted for field size and class (the owner, 18 Nov 2026: "try a new model
target, which is normalised lengths beaten. Ensure it is field size and class (prize money) adjusted").

Each run's figure (post-race; it feeds later days only):

    rel    the race's mean pounds beaten less the horse's (lengths beaten x pounds a length at the trip, capped at
           30 lb; a winner 0): above zero, better than the race's average
    s(n)   the usual within-race spread of pounds beaten in a field of n (the standard deviation by field size over
           the races before FIT_UNTIL, smoothed): rel / s(n) puts a 4-runner and a 20-runner race on one scale
    class  log(1 + prize to the winner) less its median for the code (flat or jumps) over the races before FIT_UNTIL
    figure = rel / s(n) + GAMMA x class

GAMMA, the weight of class, is chosen from GAMMAS on the races before FIT_UNTIL only (the same in every fold and on
the live path): the one whose figures, as the mean of the horse's last three, best predict the horse's next run's
rel / s(n) within its race (Spearman's rank correlation, race-centred), so class counts as much as it predicts.
Non-finishers have no figure. Prize money is as recorded (Irish races in euro, unconverted).

Features, from EARLIER DAYS only:

    nb_l1, nb_l2, nb_l3   the last three figures
    nb_m3, nb_m5          their means over the last three and five
    nb_w10                the last ten weighted 0.85^k (k runs back)
    nb_best5              the best of the last five
    nb_trend              nb_l1 - nb_m5
    nb_n                  figures held
    nb_m3_rel, nb_best5_rel   nb_m3 and nb_best5 less the mean of the rest of today's field (who have one)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _Days, _col
from model.race_shape import _codes, day_index, race_code, race_key

FEATURES = ["nb_l1", "nb_l2", "nb_l3", "nb_m3", "nb_m5", "nb_w10", "nb_best5", "nb_trend", "nb_n", "nb_m3_rel",
            "nb_best5_rel"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners", "placing_numerical", "LB",
         "total_dst_bt", "dist_furlongs", "prize_money", "race_type", "surface_type"]
FIT_UNTIL = "2023-01-01"
GAMMAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
LBS_CAP = 30.0
DECAY = 0.85
#: chosen when the frame holds too few races before FIT_UNTIL to choose (a small test frame)
GAMMA_DEFAULT = 0.5
MIN_RACES_TO_FIT = 500


def run_parts(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each run's rel, field size, class and whether it falls before FIT_UNTIL (post-race readings)."""
    from model.perf_figures import lbs_per_length, parse_beaten_lengths
    race = pd.factorize(race_key(df))[0]
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce")
    n = n.fillna(pd.Series(race).map(pd.Series(race).value_counts())).to_numpy(dtype=float)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    pos = np.where(pos > 0, pos, np.nan)
    fin = np.isfinite(pos)
    lb = (pd.to_numeric(df["LB"], errors="coerce").to_numpy(dtype=float) if "LB" in df.columns
          else _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(dtype=float))
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(dtype=float)
    lbs = np.where(pos == 1, 0.0, np.where(fin, np.minimum(lb * lbs_per_length(np.nan_to_num(dist, nan=8.0)), LBS_CAP),
                                           np.nan))
    rel = pd.Series(lbs).groupby(race).transform("mean").to_numpy(dtype=float) - lbs
    code = race_code(df).to_numpy()
    jumps = np.isin(code, ["hurdle", "chase", "nhflat"])
    prize = np.log1p(np.clip(pd.to_numeric(_col(df, "prize_money"), errors="coerce").to_numpy(dtype=float), 0, None))
    early = (pd.to_datetime(df["race_date"], errors="coerce") < pd.Timestamp(FIT_UNTIL)).to_numpy()
    return {"race": race, "n": n, "rel": rel, "jumps": jumps, "prize": prize, "early": early}


def spread_by_field(rel: np.ndarray, n: np.ndarray, race: np.ndarray, early: np.ndarray) -> dict[int, float]:
    """s(n): the within-race standard deviation of pounds beaten by field size, races before FIT_UNTIL, made
    non-decreasing in n and carried to the ends."""
    ok = early & np.isfinite(rel) & np.isfinite(n)
    if ok.sum() < 50:
        return {}
    sd = pd.DataFrame({"race": race[ok], "rel": rel[ok], "n": n[ok]}).groupby("race").agg(sd=("rel", "std"),
                                                                                          n=("n", "first"))
    by_n = sd.dropna().groupby(sd.n.clip(2, 30).round()).sd.median()
    by_n = by_n.cummax()
    return {int(k): float(v) for k, v in by_n.items() if v > 0}


def _s_of(n: np.ndarray, table: dict[int, float]) -> np.ndarray:
    if not table:
        return np.full(len(n), 8.0)
    keys = np.array(sorted(table))
    vals = np.array([table[k] for k in keys])
    return np.interp(np.clip(np.nan_to_num(n, nan=keys.min()), keys.min(), keys.max()), keys, vals)


def build(df: pd.DataFrame) -> pd.DataFrame:
    p = run_parts(df)
    s = _s_of(p["n"], spread_by_field(p["rel"], p["n"], p["race"], p["early"]))
    rel_s = p["rel"] / s
    med = {j: np.nanmedian(p["prize"][p["early"] & (p["jumps"] == j)]) if (p["early"] & (p["jumps"] == j)).any()
           else np.nanmedian(p["prize"]) for j in (False, True)}
    cls = p["prize"] - np.where(p["jumps"], med[True], med[False])
    cls = np.where(np.isfinite(cls), cls, 0.0)

    day = day_index(df)
    horse = _codes(df["horse_name"].fillna("").astype(str).str.strip().str.lower())
    horse = np.where(df["horse_name"].isna().to_numpy(), -1, horse)
    H = _Days(horse, day)
    nb = len(H.u)

    def lags(values, k_max):
        b = H.per_block(values)
        out = np.full((len(df), k_max), np.nan)
        for k in range(1, k_max + 1):
            idx = np.arange(nb) - k
            ok = H.valid & (H.pos >= k)
            out[:, k - 1] = H.to_rows(np.where(ok, b[np.clip(idx, 0, None)], np.nan))
        return out

    gamma = GAMMA_DEFAULT
    early_races = len(np.unique(p["race"][p["early"]]))
    if early_races >= MIN_RACES_TO_FIT:
        best = -np.inf
        for g in GAMMAS:
            fig = rel_s + g * cls
            m3 = np.nanmean(lags(fig, 3), axis=1) if len(df) else np.array([])
            d = pd.DataFrame({"race": p["race"], "m3": m3, "y": rel_s})[p["early"]].dropna()
            d = d[d.groupby("race").race.transform("size") >= 3]
            if d.empty:
                continue
            d["m3"] -= d.groupby("race").m3.transform("mean")
            score = d.m3.rank().corr(d.y.rank())
            if score > best:
                best, gamma = score, g
    fig = rel_s + gamma * cls

    L = lags(fig, 10)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m3 = np.nanmean(L[:, :3], axis=1)
            m5 = np.nanmean(L[:, :5], axis=1)
            b5 = np.nanmax(np.where(np.isfinite(L[:, :5]), L[:, :5], -np.inf), axis=1)
    b5 = np.where(np.isfinite(b5), b5, np.nan)
    w = DECAY ** np.arange(1, 11)
    okL = np.isfinite(L)
    w10 = np.where(okL.any(axis=1), (np.where(okL, L, 0) * w).sum(axis=1) / np.where(okL.any(axis=1),
                                                                                       (okL * w).sum(axis=1), 1), np.nan)
    n_held = np.cumsum(np.isfinite(H.per_block(fig)).astype(float))
    first = H.start
    held_before = np.where(H.valid & (H.pos > 0), n_held[np.clip(np.arange(nb) - 1, 0, None)]
                           - np.where(first > 0, n_held[np.clip(first - 1, 0, None)], 0.0), 0.0)
    race = p["race"]

    def rel_to_field(v):
        sm = pd.Series(np.nan_to_num(v)).groupby(race).transform("sum").to_numpy()
        cnt = pd.Series(np.isfinite(v).astype(float)).groupby(race).transform("sum").to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            others = (sm - np.nan_to_num(v)) / (cnt - np.isfinite(v))
        return np.where(np.isfinite(v) & (cnt - np.isfinite(v) > 0), v - others, np.nan)

    cols = {"nb_l1": L[:, 0], "nb_l2": L[:, 1], "nb_l3": L[:, 2], "nb_m3": m3, "nb_m5": m5, "nb_w10": w10,
            "nb_best5": b5, "nb_trend": L[:, 0] - m5,
            "nb_n": np.where(horse >= 0, H.to_rows(held_before), np.nan),
            "nb_m3_rel": rel_to_field(m3), "nb_best5_rel": rel_to_field(b5)}
    new = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=df.index)[FEATURES]
    out = pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
    out.attrs["nb_gamma"] = gamma
    return out
