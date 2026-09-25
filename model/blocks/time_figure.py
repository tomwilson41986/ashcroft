"""A speed figure for each horse, from the clock: the race's time rated against the course's standard and the day's going, less the horse's distance behind.

Every paper in the syndicate reading puts times first (Bolton & Chapman's
largest standardised effect is the mean speed rating of the last four runs;
Benter lists "normalized times"; Silverman models speed directly). The engine's
RSR is not that: it rates the race's winning time against a standard, so every
runner in a race carries the same number, and it takes the going from the
official description only. This rates the horse:

1. standard time per furlong for the course, trip (to the furlong) and code,
   from the winning times of EARLIER days (at least five races);
2. the going allowance for the meeting: how much faster or slower than
   standard the day's winners ran, per furlong, over the day's races at the
   course (at least two, each capped at +-1 s/f, so one freak time does not
   rate the card); a meeting of one race takes the course's average allowance
   on that going from earlier days;
3. the race figure in pounds: seconds faster than standard less the
   allowance, into lengths at the race's own speed (a length is 2.4 m), into
   pounds at the trip (model/perf_figures.lbs_per_length);
4. the horse's figure: the race figure less its pounds beaten (capped at
   30 lbs), plus the weight it carried over 130 lbs.

A run's figure is post-race; features read the horse's earlier runs only:
    tf_l1, tf_m3, tf_m5, tf_w5, tf_e3, tf_car   the figure over windows
    tf_best5, tf_best                          best of the last five, and ever
    tf_trend                                   last run less the mean of five
    tf_n                                       how many figures it has
    tf_best5_rel, tf_w5_rel                    against the best in today's field
    tf_ga_l1                                   the going allowance of its last run
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.race_shape import _codes, asof_decayed_mean, day_index, horse_decayed_prior, race_code, race_key

LENGTH_METRES = 2.4
FURLONG_METRES = 201.168
MIN_STANDARD_RACES = 5
GA_CAP = 1.0            # seconds per furlong
LBS_BEATEN_CAP = 30.0
WEIGHT_BASE = 130.0

FEATURES = ["tf_l1", "tf_m3", "tf_m5", "tf_w5", "tf_e3", "tf_car", "tf_best5", "tf_best", "tf_trend", "tf_n",
            "tf_best5_rel", "tf_w5_rel", "tf_ga_l1"]
POST_RACE: set[str] = set()


def _num(df, name):
    return pd.to_numeric(_col(df, name), errors="coerce").to_numpy(dtype=float)


def figures(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each run's going allowance, race figure and horse figure (post-race: they
    feed later days' windows and are never features themselves).

    The standard and the allowance are built on one row per race, so a race
    counts once however many runners it had and whatever order they arrive in."""
    from model.blocks.draw_v2 import going_group
    from model.perf_figures import lbs_per_length, parse_beaten_lengths
    rk = race_key(df)
    race = pd.factorize(rk)[0]
    day = day_index(df)
    dist = _num(df, "dist_furlongs")
    t = _num(df, "comptime_numeric")
    R = pd.DataFrame({
        "race": race, "day": day, "dist": dist, "t": np.where(t > 0, t, np.nan),
        "track": _col(df, "track").fillna("").astype(str).str.lower().str.strip().to_numpy(),
        "code": race_code(df).to_numpy(), "going": going_group(df).to_numpy(),
    }).groupby("race").agg(day=("day", "min"), dist=("dist", "first"), t=("t", "min"),
                           track=("track", "first"), code=("code", "first"), going=("going", "first"))
    ok = np.isfinite(R["t"]) & np.isfinite(R["dist"]) & (R["dist"] > 0)
    tpf = np.where(ok, R["t"] / R["dist"].where(R["dist"] > 0), np.nan)          # seconds per furlong
    rday = R["day"].to_numpy(dtype=np.int64)
    key = _codes(R["track"], R["dist"].round().fillna(-1), R["code"])
    std, n_std = asof_decayed_mean(key, rday, tpf, key, rday, halflife_days=np.inf)
    std = np.where(n_std >= MIN_STANDARD_RACES, std, np.nan)
    dev = tpf - std                                                              # + = slower than standard

    # the meeting's going allowance: the day's races at the course, each capped
    meet = _codes(R["track"], pd.Series(rday))
    s = pd.Series(np.clip(dev, -GA_CAP, GA_CAP)).groupby(meet)
    ga_day = s.transform("mean").to_numpy(dtype=float)
    n_day = s.transform("count").to_numpy(dtype=float)
    # a meeting of one rated race: the course's average allowance on this going, earlier days
    gkey = _codes(R["track"], R["code"], R["going"])
    ga_hist, n_hist = asof_decayed_mean(gkey, rday, np.where(n_day >= 2, ga_day, np.nan), gkey, rday,
                                        halflife_days=730.0)
    ga = np.where(n_day >= 2, ga_day, np.where(n_hist > 0, ga_hist, 0.0))
    ga = np.where(np.isfinite(dev), ga, np.nan)

    # seconds faster than standard after the going, into lengths at this race's speed, into pounds
    rdist = R["dist"].to_numpy(dtype=float)
    secs = -(dev - ga) * rdist
    speed = np.where(ok, rdist * FURLONG_METRES / R["t"].to_numpy(dtype=float), np.nan)   # metres a second
    race_fig = secs * speed / LENGTH_METRES * lbs_per_length(np.nan_to_num(rdist, nan=8.0))

    pos = _num(df, "placing_numerical")
    fin = np.isfinite(pos) & (pos > 0)
    if "LB" in df.columns:
        lb = pd.to_numeric(df["LB"], errors="coerce").to_numpy(dtype=float)
    else:
        lb = _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(dtype=float)
    lb = np.where(pos == 1, 0.0, lb)
    lbs_bt = np.minimum(lb * lbs_per_length(np.nan_to_num(dist, nan=8.0)), LBS_BEATEN_CAP)
    wt = _num(df, "pounds")
    wadj = np.where(np.isfinite(wt), wt - WEIGHT_BASE, 0.0)
    row_race_fig = race_fig[race]
    fig = np.where(fin, row_race_fig - lbs_bt + wadj, np.nan)
    return {"fig": fig, "race_fig": row_race_fig, "ga": ga[race]}


def build(df: pd.DataFrame) -> pd.DataFrame:
    f = figures(df)
    day = day_index(df)
    name = _col(df, "horse_name").fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    horse = np.where(bad, -1, _codes(name))
    H = _Days(horse, day)
    v = H.per_block(f["fig"])
    lad = _ladder(H, v)
    cols = {f"tf_{w}": H.to_rows(lad[w]) for w in ("l1", "m3", "m5", "w5", "car")}
    s, c, _ = horse_decayed_prior(np.where(bad, "", name), day, {"f": f["fig"]}, halflife_runs=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["tf_e3"] = np.where(~bad & (c["f"] > 0), s["f"] / c["f"], np.nan)

    # best of the last five, and ever, from earlier blocks only
    nb = len(H.u)
    blocks = np.arange(nb)
    best5 = np.full(nb, -np.inf)
    for j in range(1, 6):
        okj = H.valid & (H.pos >= j)
        x = np.where(okj, v[np.clip(blocks - j, 0, None)], np.nan)
        best5 = np.where(np.isfinite(x), np.maximum(best5, x), best5)
    best5 = np.where(np.isfinite(best5), best5, np.nan)
    vv = np.where(np.isfinite(v) & H.valid, v, -np.inf)
    cummax = pd.Series(vv).groupby(H.key).cummax().to_numpy()
    prev = np.clip(blocks - 1, 0, None)
    best = np.where(H.valid & (H.pos >= 1), cummax[prev], -np.inf)
    best = np.where(np.isfinite(best), best, np.nan)
    cols["tf_best5"], cols["tf_best"] = H.to_rows(best5), H.to_rows(best)
    cols["tf_trend"] = cols["tf_l1"] - cols["tf_m5"]
    cols["tf_n"] = H.to_rows(lad["_cnt"])
    ga_b = H.per_block(f["ga"])
    cols["tf_ga_l1"] = H.to_rows(np.where(H.valid & (H.pos >= 1), ga_b[prev], np.nan))

    rk = race_key(df).to_numpy()
    for m in ("best5", "w5"):
        x = pd.Series(cols[f"tf_{m}"])
        cols[f"tf_{m}_rel"] = (x - x.groupby(rk).transform("max")).to_numpy(dtype=float)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
