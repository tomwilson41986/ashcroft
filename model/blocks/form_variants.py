"""NFP and lengths beaten in every variant: windows, normalisations, shrinkage.

The owner's ask (25 Sep): the horse's normalised finishing position and its
lengths beaten by career, last run, last 3, 5 and 10, recency-weighted and scaled
across all runs; lengths beaten normalised; every variant tried.

Already built, and not repeated here:
  NFP      career (preracehorsecareerNFP), last run (LRNFP), mean of the last
           3/5/10 (LR*NFPtotal), exponentially weighted over the last 3/5/10
           (EXP_NFP*), and fw_nfp_{car,l1,m3,m5,w3,w5,w10} (form windows)
  lengths  career, last run, mean of the last 3 and 5 (preracehorsecareerLB,
           LR_LB, LR3_LB, LR5_LB); pounds beaten, raw and race-centred, over the
           seven form windows (fw_lbs_*, fw_lbsc_*)

New here (prefix fv_):
  NFP, scaled across all runs: exponentially in runs (half-lives 3 and 6 runs)
      and in time (half-life a year); weighted by the size of the field beaten;
      and shrunk toward the trainer's runners' record, empirical-Bayes style, with
      the prior worth 3 or 8 runs (model/shrinkage.py), so one win from one run
      no longer reads like twenty from twenty.
  NFP with non-finishers read as last (nfp0): a pulled-up run is a bad run, not
      a missing one. Every window.
  Lengths beaten, six ways, every window: capped at 20 (lbc), log(1 + lengths)
      (lblog), per furlong of the trip (lbpf), against the race's own spread (the
      mean margin of its beaten finishers; lbrel), per place behind the winner
      (lbpp), and signed, with a winner's margin as a negative (lbw).
Windows: career (car), last run (l1), mean of the last 3, 5 and 10 runs (m3, m5,
m10), the last 5 weighted 5..1 (w5), exponential in runs (e3, e6), exponential in
days (d365); a window counts runs, not known values.

Earlier days only: every window reads the horse's earlier racing days, and the
trainer prior reads the trainer's earlier days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.form_windows import _winning_margin
from model.freshness_features import _Days, _col
from model.race_shape import _codes, asof_decayed_mean, day_index, field_size, horse_decayed_prior, race_key
from model.shrinkage import shrunk_mean

PREFIX = "fv_"
LB_CAP = 20.0
MARGIN_CAP = 10.0
#: The trainer's record, a year's half-life, is itself pulled toward an average
#: runner's NFP (0.5) with the weight of this many runs.
TRAINER_PRIOR_K = 30.0
SHRINK_K = (3, 8)

ALL_WINDOWS = ("car", "l1", "m3", "m5", "m10", "w5", "e3", "e6", "d365")
#: What each measure gets: every window unless the engine or the form windows
#: already carry it.
NFP_WINDOWS = ("e3", "e6", "d365")
LB_WINDOWS = ("m10", "w5", "e3", "e6", "d365")
LB_MEASURES = ("lbc", "lblog", "lbpf", "lbrel", "lbpp", "lbw")

FEATURES = (
    [f"fv_nfp_{w}" for w in NFP_WINDOWS]
    + [f"fv_nfp_s{k}" for k in SHRINK_K]
    + ["fv_nfp_fsw_car", "fv_nfp_fsw_e6"]
    + [f"fv_nfp0_{w}" for w in ALL_WINDOWS] + [f"fv_nfp0_s{k}" for k in SHRINK_K]
    + [f"fv_lb_{w}" for w in LB_WINDOWS]
    + [f"fv_{m}_{w}" for m in LB_MEASURES for w in ALL_WINDOWS]
)

#: A run's own readings, kept for inspection by nothing: the block drops them.
POST_RACE: set[str] = set()

#: Every column build() reads (the live path copies only these).
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer", "number_of_runners",
         "placing_numerical", "LB", "total_dst_bt", "dist_furlongs"]

MEAN_WINDOWS = (3, 5, 10)
WEIGHTED = 5


def _num(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(_col(df, name), errors="coerce").to_numpy(dtype=float)


def run_measures(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each measure of each run, from the run's own race (post-race: they feed
    later days' windows and are never features themselves)."""
    from model.perf_figures import parse_beaten_lengths

    race = pd.factorize(race_key(df))[0].astype(np.int64)
    n = field_size(df).to_numpy(dtype=float)
    pos = _num(df, "placing_numerical")
    pos = np.where(pos > 0, pos, np.nan)
    fin = np.isfinite(pos)
    # a non-finisher is a runner without a placing in a race that has a result;
    # a card race has none, so its runners are unknown, not last
    has_result = pd.Series(fin).groupby(race).transform("any").to_numpy()
    nf = ~fin & has_result
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.clip((n - pos) / np.where(n > 1, n - 1, np.nan), 0.0, 1.0)
    nfp0 = np.where(fin, nfp, np.where(nf, 0.0, np.nan))

    if "LB" in df.columns:
        lb = pd.to_numeric(df["LB"], errors="coerce").to_numpy(dtype=float)
    else:
        lb = _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(dtype=float)
    lb = np.where(pos == 1, 0.0, np.where(fin, lb, np.nan))
    dist = _num(df, "dist_furlongs")
    beaten = np.where(fin & (pos > 1), lb, np.nan)
    spread = pd.Series(beaten).groupby(race).transform("mean").to_numpy(dtype=float)
    margin = _winning_margin(race, pos, lb)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "nfp": nfp, "nfp0": nfp0, "lb": lb,
            "lbc": np.minimum(lb, LB_CAP),
            "lblog": np.log1p(lb),
            "lbpf": lb / np.where(dist > 0, dist, np.nan),
            "lbrel": np.where(pos == 1, 0.0, lb / np.where(spread > 0, spread, np.nan)),
            "lbpp": np.where(pos == 1, 0.0, lb / np.where(pos > 1, pos - 1, np.nan)),
            "lbw": np.where(pos == 1, -np.minimum(margin, MARGIN_CAP), np.minimum(lb, LB_CAP)),
            "rivals": np.where(n > 1, n - 1, np.nan),
        }
    out["lbw"] = np.where(fin, out["lbw"], np.nan)
    return out


def _ladder(H: _Days, v: np.ndarray) -> dict[str, np.ndarray]:
    """car, l1, m3, m5, m10 and w5 of one per-block measure, per block, from the
    key's earlier blocks only, summed in a fixed order."""
    nb = len(H.u)
    v = np.where(H.valid, v, np.nan)
    blocks = np.arange(nb)
    out = {}
    msum = {k: np.zeros(nb) for k in MEAN_WINDOWS}
    mcnt = {k: np.zeros(nb) for k in MEAN_WINDOWS}
    wsum, wcnt = np.zeros(nb), np.zeros(nb)
    for j in range(1, max(MEAN_WINDOWS) + 1):
        ok = H.valid & (H.pos >= j)
        x = np.where(ok, v[np.clip(blocks - j, 0, None)], np.nan)
        f = np.isfinite(x)
        x0 = np.where(f, x, 0.0)
        if j == 1:
            out["l1"] = x
        for k in MEAN_WINDOWS:
            if j <= k:
                msum[k] += x0
                mcnt[k] += f
        if j <= WEIGHTED:
            wt = float(WEIGHTED - j + 1)
            wsum += wt * x0
            wcnt += wt * f
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in MEAN_WINDOWS:
            out[f"m{k}"] = np.where(mcnt[k] > 0, msum[k] / mcnt[k], np.nan)
        out[f"w{WEIGHTED}"] = np.where(wcnt > 0, wsum / wcnt, np.nan)
    s, c = career_totals(H, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["car"] = np.where(c > 0, s / c, np.nan)
    out["_sum"], out["_cnt"] = s, c
    return out


def career_totals(H: _Days, v: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sum of w*v and of w over the key's earlier blocks (w = 1 by default)."""
    nb = len(H.u)
    f = np.isfinite(v) & H.valid
    w = np.ones(nb) if w is None else np.where(np.isfinite(w), w, 0.0)
    key = pd.Series(H.key)
    cs = pd.Series(np.where(f, w * v, 0.0)).groupby(key).cumsum().to_numpy()
    cn = pd.Series(np.where(f, w, 0.0)).groupby(key).cumsum().to_numpy()
    prev = np.clip(np.arange(nb) - 1, 0, None)
    has = H.valid & (H.pos >= 1)
    return np.where(has, cs[prev], 0.0), np.where(has, cn[prev], 0.0)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    horse = np.where(bad, -1, _codes(name))
    H = _Days(horse, day)
    m = run_measures(df)
    cols: dict[str, np.ndarray] = {}

    def ratio(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(b > 0, a / b, np.nan)

    # exponential in runs and in days, for every measure that takes them
    decayed = {}
    for half in (3, 6):
        sums, cnts, _ = horse_decayed_prior(np.where(bad, "", name), day,
                                            {k: m[k] for k in ("nfp", "nfp0", "lb", *LB_MEASURES)},
                                            halflife_runs=half)
        for k in sums:
            decayed[(k, f"e{half}")] = np.where(bad, np.nan, ratio(sums[k], cnts[k]))
    for k in ("nfp", "nfp0", "lb", *LB_MEASURES):
        mean, _ = asof_decayed_mean(horse, day, m[k], horse, day, halflife_days=365.0)
        decayed[(k, "d365")] = mean

    # the trainer's runners' NFP over earlier days, itself pulled toward 0.5: the
    # prior a lightly raced horse's career NFP is shrunk toward
    tname = _col(df, "trainer").fillna("").astype(str).str.strip().str.lower()
    trainer = np.where(tname.isin(["", "nan", "none"]).to_numpy(), -1, _codes(tname))
    priors = {}
    for k in ("nfp", "nfp0"):
        tm, tn = asof_decayed_mean(trainer, day, m[k], trainer, day, halflife_days=365.0)
        priors[k] = shrunk_mean(np.nan_to_num(tm) * tn, tn, 0.5, TRAINER_PRIOR_K)

    for k in ("nfp", "nfp0", "lb", *LB_MEASURES):
        lad = _ladder(H, H.per_block(m[k]))
        rows = {w: H.to_rows(lad[w]) for w in ("car", "l1", "m3", "m5", "m10", "w5")}
        rows.update({w: decayed[(k, w)] for w in ("e3", "e6", "d365")})
        wanted = NFP_WINDOWS if k == "nfp" else LB_WINDOWS if k == "lb" else ALL_WINDOWS
        for w in wanted:
            cols[f"{PREFIX}{k}_{w}"] = rows[w]
        if k in priors:
            s, c = H.to_rows(lad["_sum"]), H.to_rows(lad["_cnt"])
            for kk in SHRINK_K:
                cols[f"{PREFIX}{k}_s{kk}"] = np.where(bad, np.nan, shrunk_mean(s, c, priors[k], kk))

    # NFP weighted by the rivals beaten: a win in a field of twenty says more than one in four
    rv = H.per_block(m["rivals"])
    s, c = career_totals(H, H.per_block(m["nfp"]), rv)
    cols["fv_nfp_fsw_car"] = H.to_rows(ratio(s, c))
    sums, _, _ = horse_decayed_prior(np.where(bad, "", name), day,
                                     {"num": m["nfp"] * m["rivals"],
                                      "den": np.where(np.isfinite(m["nfp"]), m["rivals"], np.nan)},
                                     halflife_runs=6)
    cols["fv_nfp_fsw_e6"] = np.where(bad, np.nan, ratio(sums["num"], sums["den"]))

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
