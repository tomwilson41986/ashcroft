"""Form windows: every per-run measure over the same seven windows.

The served features summarise a horse's past runs unevenly. NFP has a career
mean, the last run, the means of the last 3, 5 and 10 and exponential weights;
speed and lengths beaten stop at the mean of the last five; the win-over-chance
family (WAX, WOA, CWO, WIV) has a career figure only; the market's view has
harmonic sums. This block gives every measure the same ladder, from the horse's
runs on EARLIER DAYS:

    fw_<m>_car   career: the mean over every earlier run held
    fw_<m>_l1    the last run
    fw_<m>_m3    the mean of the last 3 runs
    fw_<m>_m5    the mean of the last 5 runs
    fw_<m>_w3    the last 3, 5 or 10 runs weighted linearly by recency: the last
    fw_<m>_w5    run weighs k, the one before k - 1, ... the k-th back 1
    fw_<m>_w10

A window counts runs, not known values. A run whose measure is unknown (no time
taken, no price, no margin returned) keeps its place in the window and adds
nothing to it; a window with no known value is unknown.

The measures, each describing one past run from that run's own race only:

    win    won (1/0)
    plc    placed: in the first three (1/0)
    wax    won less 1/N, a random runner's chance (the served WAX, windowed)
    nfp    normalised finishing position, 1 the winner .. 0 last
    lbs    pounds behind the winner at the trip (lengths x lbs per length), capped
    lbsc   pounds better than the race's average runner
    perf   a performance figure on the official-rating scale: the horse's rating
           that day (the race's median where it had none), less the pounds it was
           beaten, or plus the winning margin (model/perf_figures.py's rule)
    rsr    the engine's speed rating (RSR, standard times from earlier days)
    mkt    the market's view: log(N x the BSP's normalised win probability)
    ae     won less the BSP's normalised win probability (+ = beat the market)
    nres   normalised finishing position less the one the market's order implied

And today's official rating against the performance figures: fw_perf_l1_vs_or,
fw_perf_w5_vs_or, fw_perf_car_vs_or (positive = has run to more than its mark;
today's rating is on the 06:00 card).

A non-finisher (no finishing position) is unknown for every measure, as the
engine's NFP is. Nothing of the day being predicted enters any window: a 06:00
card and the same day with its results in get identical values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _Days, _col
from model.race_shape import _codes, day_index, field_size, race_key

#: The measures, in feature order.
MEASURES = ["win", "plc", "wax", "nfp", "lbs", "lbsc", "perf", "rsr", "mkt", "ae", "nres"]

#: The windows: career, the last run, plain means of the last 3 and 5, and the
#: last 3, 5 and 10 weighted linearly by recency.
MEAN_WINDOWS = (3, 5)
WEIGHTED_WINDOWS = (3, 5, 10)
WINDOWS = ["car", "l1", *[f"m{k}" for k in MEAN_WINDOWS], *[f"w{k}" for k in WEIGHTED_WINDOWS]]

#: Pounds beaten are capped here: a tailed-off run is a bad day, not a measure of
#: how bad (race_shape.LBS_CAP, perf_figures' floor).
LBS_CAP = 20.0

#: A winner is credited with its margin in pounds up to this, or WINNER_UNKNOWN_CREDIT
#: when the margin is not returned (model/perf_figures.py).
WINNER_CREDIT_CAP = 5.0
WINNER_UNKNOWN_CREDIT = 2.0

PERF_VS_OR = ["l1", "w5", "car"]

FORM_WINDOW_FEATURES = (
    [f"fw_{m}_{w}" for m in MEASURES for w in WINDOWS]
    + [f"fw_perf_{w}_vs_or" for w in PERF_VS_OR]
)


def _num(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(_col(df, name), errors="coerce").to_numpy(dtype=float)


def _winning_margin(race: np.ndarray, pos: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """The winning margin in lengths on every runner of the race: the cumulative
    margin of the second finishing position held, 0 after a dead heat for the win
    (model/perf_figures.winning_margin's rule, on lengths already parsed)."""
    ok = np.isfinite(pos) & np.isfinite(lb)
    u = (pd.DataFrame({"r": race[ok], "p": pos[ok], "b": lb[ok]})
         .groupby(["r", "p"], sort=True)["b"].max().reset_index())
    u["i"] = u.groupby("r", sort=False).cumcount()
    second = u[u["i"] == 1].set_index("r")["b"]
    margin = pd.Series(race).map(second).to_numpy(dtype=float)
    winners = np.bincount(race[ok & (pos == 1)], minlength=race.max() + 1 if len(race) else 0)
    return np.where(winners[race] > 1, 0.0, margin)


def run_measures(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each measure of each run, from the run's own race. Post-race: they feed the
    windows of later days and are never features themselves."""
    from model.perf_figures import lbs_per_length, parse_beaten_lengths

    race = pd.factorize(race_key(df))[0].astype(np.int64)
    n = field_size(df).to_numpy(dtype=float)
    pos = _num(df, "placing_numerical")
    pos = np.where(pos > 0, pos, np.nan)
    fin = np.isfinite(pos)
    with np.errstate(invalid="ignore", divide="ignore"):
        won = np.where(fin, (pos == 1).astype(float), np.nan)
        placed = np.where(fin, (pos <= 3).astype(float), np.nan)
        nfp = np.clip((n - pos) / np.where(n > 1, n - 1, np.nan), 0.0, 1.0)
        wax = won - 1.0 / np.where(n > 0, n, np.nan)

    # lengths beaten: the engine's parsed figure (LB) where it has one; a winner is
    # beaten by nothing, a non-finisher is unknown
    if "LB" in df.columns:
        lb = pd.to_numeric(df["LB"], errors="coerce").to_numpy(dtype=float)
    else:
        lb = _col(df, "total_dst_bt").map(parse_beaten_lengths).to_numpy(dtype=float)
    lb = np.where(pos == 1, 0.0, np.where(fin, lb, np.nan))
    lpl = lbs_per_length(np.nan_to_num(_num(df, "dist_furlongs"), nan=8.0))
    lbs = np.minimum(lb * lpl, LBS_CAP)
    by_race = pd.Series(lbs).groupby(race)
    lbsc = by_race.transform("mean").to_numpy(dtype=float) - lbs

    # performance figure: the rating that day less the pounds beaten, or plus the
    # winning margin in pounds
    orr = pd.Series(_num(df, "official_rating"))
    orr = orr.where(orr > 0)
    base = orr.fillna(orr.groupby(race).transform("median")).to_numpy(dtype=float)
    margin = _winning_margin(race, pos, lb)
    credit = np.where(np.isfinite(margin), np.minimum(margin * lpl, WINNER_CREDIT_CAP), WINNER_UNKNOWN_CREDIT)
    perf = np.where(pos == 1, base + credit, base - lbs)

    rsr = _num(df, "RSR")

    # the market's view from the BSP, normalised within the race over the runners priced
    bsp = _num(df, "bfsp")
    p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
    g = p.groupby(race)
    priced = g.transform("count").to_numpy(dtype=float)
    pn = (p / g.transform("sum")).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mkt = np.log(pn * priced)
        ae = won - pn
        rank = g.rank(ascending=False, method="average").to_numpy(dtype=float)
        exp_nfp = (priced - rank) / np.where(priced > 1, priced - 1, np.nan)
        nres = nfp - exp_nfp

    return {"win": won, "plc": placed, "wax": wax, "nfp": nfp, "lbs": lbs, "lbsc": lbsc,
            "perf": perf, "rsr": rsr, "mkt": mkt, "ae": ae, "nres": nres}


def window_ladder(H: _Days, v: np.ndarray) -> dict[str, np.ndarray]:
    """The seven windows of one per-block measure, per block.

    Every window reads blocks strictly before the block (the key's earlier days),
    summed in a fixed order, so a block's own value never enters its windows, not
    even as rounding."""
    nb = len(H.u)
    v = np.where(H.valid, v, np.nan)
    blocks = np.arange(nb)
    out = {}
    msum = {k: np.zeros(nb) for k in MEAN_WINDOWS}
    mcnt = {k: np.zeros(nb) for k in MEAN_WINDOWS}
    wsum = {k: np.zeros(nb) for k in WEIGHTED_WINDOWS}
    wcnt = {k: np.zeros(nb) for k in WEIGHTED_WINDOWS}
    for j in range(1, max(WEIGHTED_WINDOWS + MEAN_WINDOWS) + 1):
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
        for k in WEIGHTED_WINDOWS:
            if j <= k:
                wt = float(k - j + 1)
                wsum[k] += wt * x0
                wcnt[k] += wt * f
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in MEAN_WINDOWS:
            out[f"m{k}"] = np.where(mcnt[k] > 0, msum[k] / mcnt[k], np.nan)
        for k in WEIGHTED_WINDOWS:
            out[f"w{k}"] = np.where(wcnt[k] > 0, wsum[k] / wcnt[k], np.nan)
        # career: running totals within the key, read at the block before
        f = np.isfinite(v)
        key = pd.Series(H.key)
        cs = pd.Series(np.where(f, v, 0.0)).groupby(key).cumsum().to_numpy()
        cn = pd.Series(f.astype(float)).groupby(key).cumsum().to_numpy()
        prev = np.clip(blocks - 1, 0, None)
        has = H.valid & (H.pos >= 1) & (cn[prev] > 0)
        out["car"] = np.where(has, cs[prev] / np.where(has, cn[prev], 1.0), np.nan)
    return out


def add_form_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The form-window block on a frame of history plus (optionally) today's card.

    Needs race_date, race_time, track, horse_name; uses placing_numerical,
    number_of_runners, total_dst_bt (or the engine's LB), dist_furlongs,
    official_rating, RSR and bfsp where present. Card rows need no result: they
    read the history before their day and add nothing to it."""
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    horse = np.where(name.isin(["", "nan", "none"]).to_numpy(), -1, _codes(name))
    H = _Days(horse, day)
    measures = run_measures(df)
    cols = {}
    for m in MEASURES:
        ladder = window_ladder(H, H.per_block(measures[m]))
        for w in WINDOWS:
            cols[f"fw_{m}_{w}"] = H.to_rows(ladder[w])
    orr = _num(df, "official_rating")
    orr = np.where(orr > 0, orr, np.nan)
    for w in PERF_VS_OR:
        cols[f"fw_perf_{w}_vs_or"] = cols[f"fw_perf_{w}"] - orr
    new = pd.DataFrame(cols, index=df.index)[FORM_WINDOW_FEATURES]
    df = pd.concat([df.drop(columns=[c for c in FORM_WINDOW_FEATURES if c in df.columns]), new], axis=1)
    return df, list(FORM_WINDOW_FEATURES)
