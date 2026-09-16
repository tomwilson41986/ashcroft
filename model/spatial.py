"""
Spatial statistics for draw bias and going surfaces.

Why: stalls are not independent categories — adjacent stalls share
ground, and going changes smoothly across a card. Kriging (= Gaussian-
process regression) turns sparse, noisy per-stall win rates into a
smooth surface with uncertainty; Moran's I tests whether the pattern is
spatially structured at all.

    gp_smooth              Generic 1-D / 2-D GP smoother with count-
                           weighted heteroscedastic noise
    smooth_draw_bias       Stall -> (mean, sd, n) surface for a track/
                           distance bucket (optionally by going)
    stall_bias_surface     One fitted per-stall surface per course (or any
                           key), from data strictly before an as-of date
    walk_forward_stall_surface
                           The same surface refitted period by period and
                           mapped back onto rows — the lag-safe feature form
    going_drift_surface    Smoothed race-to-race pace-of-ground drift
                           within a meeting (from speed residuals)
    stall_adjacency        Neighbour weights for Moran's I
    morans_i               Spatial autocorrelation with permutation p-value

**As-of dates.** A smoother fitted on the whole file describes the file. Used
as a feature it tells a row in 2019 what stall 3 went on to do in 2024, which
is not a small leak: the draw surface is exactly the kind of slow-moving
quantity a model will lean on. `smooth_draw_bias` therefore takes an `as_of`
cut, and `walk_forward_stall_surface` applies one automatically — each period's
rows are scored by a surface fitted only on races run before that period began.

**Thin cells.** A stall in one course-distance bucket can carry a handful of
runs. Every raw cell mean here is shrunk toward its group mean by
``n / (n + shrink_k)`` before the GP sees it, and the GP's own noise term is
scaled by ``1 / n``, so a thin stall is pulled twice: once toward the course
and once toward its neighbours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def gp_smooth(x, y, counts=None, length_scale=2.0, length_scale_bounds=(0.5, 20.0),
              noise_level=0.05, grid=None, random_state=0):
    """Fit a GP to (x, y) and return (grid, mean, sd, fitted kernel).

    ``counts`` (observations behind each y) scale the per-point noise as
    noise_level / counts — the kriging analogue of weighting bins by n.
    x may be (n,) or (n, d).
    """
    X = np.asarray(x, float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    yv = np.asarray(y, float)
    c = np.ones(len(yv)) if counts is None else np.asarray(counts, float)
    alpha = noise_level / np.clip(c, 1, None)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale, length_scale_bounds) \
        + WhiteKernel(1e-3, (1e-6, 1.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=2, random_state=random_state)
    gp.fit(X, yv)
    if grid is None:
        if X.shape[1] == 1:
            grid = np.linspace(X.min(), X.max(), 50).reshape(-1, 1)
        else:
            grid = X
    G = np.asarray(grid, float)
    G = G.reshape(-1, 1) if G.ndim == 1 else G
    mean, sd = gp.predict(G, return_std=True)
    return G, mean, sd, gp.kernel_


#: Cell size at which a raw per-stall mean is believed half and pooled half.
SHRINK_K_STALL = 30.0


def smooth_draw_bias(df: pd.DataFrame, track: str, dist_furlongs: float, value_col: str = "won",
                     stall_col: str = "stall", dist_tol: float = 0.5, min_runners: int = 8,
                     by_going: bool = False, going_col: str = "going_description",
                     as_of=None, date_col: str = "race_date") -> pd.DataFrame:
    """GP-smoothed win rate by stall for one track / distance bucket.

    Returns a DataFrame with stall, raw_rate, n, smooth_mean, smooth_sd and
    the field-size-normalised expected rate (1/runners) for comparison.

    ``as_of`` restricts the fit to races run strictly before that date. Leave it
    None only for a report about a finished period: the table then describes
    every race in the frame, including races that have not happened yet from the
    point of view of any row you might attach it to.
    """
    d = df[(df["track"].astype(str).str.lower() == str(track).lower())
           & ((df["dist_furlongs"] - dist_furlongs).abs() <= dist_tol)
           & (df["number_of_runners"] >= min_runners)].copy()
    if as_of is not None and date_col in d.columns:
        d = d[pd.to_datetime(d[date_col], errors="coerce") < pd.Timestamp(as_of)]
    d[stall_col] = pd.to_numeric(d[stall_col], errors="coerce")
    d = d.dropna(subset=[stall_col, value_col])
    if d.empty:
        return pd.DataFrame(columns=["stall", "raw_rate", "n", "smooth_mean", "smooth_sd", "expected_rate"])
    d["_y"] = d[value_col].astype(float)
    d["_exp"] = 1.0 / d["number_of_runners"]
    d["_excess"] = d["_y"] - d["_exp"]  # remove field-size effect
    keys = [stall_col] + ([going_col] if by_going else [])
    agg = d.groupby(keys).agg(raw_rate=("_y", "mean"), n=("_y", "size"),
                              excess=("_excess", "mean"), expected_rate=("_exp", "mean")).reset_index()
    if by_going:
        agg["_g"] = agg[going_col].astype("category").cat.codes
        G, mean, sd, _ = gp_smooth(agg[[stall_col, "_g"]].values, agg["excess"].values, agg["n"].values,
                                   grid=agg[[stall_col, "_g"]].values)
    else:
        G, mean, sd, _ = gp_smooth(agg[stall_col].values, agg["excess"].values, agg["n"].values,
                                   grid=agg[stall_col].values)
    agg["smooth_mean"] = mean + agg["expected_rate"].values
    agg["smooth_sd"] = sd
    return agg.rename(columns={stall_col: "stall"}).drop(columns=["_g"], errors="ignore")


def stall_bias_surface(df: pd.DataFrame, value_col: str = "draw_resid", stall_col: str = "draw_adj",
                       keys=("track",), as_of=None, date_col: str = "race_date",
                       min_cell_n: int = 3, min_group_n: int = 50, shrink_k: float = SHRINK_K_STALL,
                       length_scale: float = 2.0, max_stall: int = 24) -> pd.DataFrame:
    """Per-stall surface of `value_col`, one fit per key group.

    ``value_col`` should already be a *residual* — what the runner did minus
    what its rating said it would do — because a mean of raw finishing positions
    by stall measures the horses drawn there as much as the stalls (framework
    §6.3). `model.draw_metrics.draw_resid` is that column.

    Returns one row per (keys..., stall) with the raw mean, the count, the
    shrunk mean the GP was fitted on, and the smoothed mean and sd. Groups with
    fewer than ``min_group_n`` rows, or fewer than three usable stalls, are
    returned unsmoothed rather than fitted on noise.
    """
    keys = list(keys)
    d = df.copy()
    if as_of is not None and date_col in d.columns:
        d = d[pd.to_datetime(d[date_col], errors="coerce") < pd.Timestamp(as_of)]
    d["_stall"] = pd.to_numeric(d[stall_col], errors="coerce").clip(1, max_stall).round()
    d["_v"] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["_stall", "_v"])
    if d.empty:
        return pd.DataFrame(columns=keys + ["stall", "raw", "n", "shrunk", "smooth_mean", "smooth_sd"])

    agg = (d.groupby(keys + ["_stall"], observed=True)["_v"]
             .agg(raw="mean", n="size").reset_index())
    grp_mean = d.groupby(keys, observed=True)["_v"].mean().rename("_gm")
    grp_n = d.groupby(keys, observed=True)["_v"].size().rename("_gn")
    agg = agg.merge(grp_mean, on=keys, how="left").merge(grp_n, on=keys, how="left")
    w = agg["n"] / (agg["n"] + shrink_k)
    agg["shrunk"] = w * agg["raw"] + (1 - w) * agg["_gm"]

    out = []
    for key, g in agg.groupby(keys, observed=True):
        g = g.sort_values("_stall").copy()
        usable = g[g["n"] >= min_cell_n]
        if len(usable) < 3 or g["_gn"].iloc[0] < min_group_n or usable["shrunk"].nunique() < 2:
            g["smooth_mean"] = g["shrunk"]
            g["smooth_sd"] = np.nan
        else:
            _, mean, sd, _ = gp_smooth(usable["_stall"].values, usable["shrunk"].values,
                                       usable["n"].values, length_scale=length_scale,
                                       grid=g["_stall"].values)
            g["smooth_mean"] = mean
            g["smooth_sd"] = sd
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res.rename(columns={"_stall": "stall"})[keys + ["stall", "raw", "n", "shrunk",
                                                          "smooth_mean", "smooth_sd"]]


def walk_forward_stall_surface(df: pd.DataFrame, value_col: str = "draw_resid",
                               stall_col: str = "draw_adj", keys=("track",), freq: str = "Y",
                               date_col: str = "race_date", min_history_rows: int = 500,
                               **surface_kw) -> pd.DataFrame:
    """The per-stall surface as a lag-safe feature, aligned to ``df.index``.

    Rows are bucketed by period (`freq`, a pandas *Period* alias — "Y", "Q",
    "M"; calendar years by default) and every
    row in a period is scored by a surface fitted only on races run *before*
    that period started. Refitting every period rather than every race is what
    makes a Gaussian process affordable here; the cost is that a row is scored
    by a surface up to one period stale, which for a quantity with a two-year
    half-life is a rounding error next to the leak that fitting on everything
    would introduce.

    Returns columns ``gp_draw_bias``, ``gp_draw_sd`` and ``gp_draw_n`` — the
    smoothed per-stall value, its GP standard deviation and the number of runs
    behind the cell.
    """
    keys = list(keys)
    idx = df.index
    out = pd.DataFrame(index=idx, columns=["gp_draw_bias", "gp_draw_sd", "gp_draw_n"], dtype=float)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().all():
        return out
    period_start = dates.dt.to_period(freq).dt.start_time
    stall = pd.to_numeric(df[stall_col], errors="coerce").clip(
        1, surface_kw.get("max_stall", 24)).round()

    for start in sorted(period_start.dropna().unique()):
        rows = period_start == start
        hist = dates < start
        if hist.sum() < min_history_rows:
            continue
        surf = stall_bias_surface(df[hist], value_col=value_col, stall_col=stall_col,
                                  keys=keys, date_col=date_col, **surface_kw)
        if surf.empty:
            continue
        surf = surf.set_index(keys + ["stall"])
        want = pd.MultiIndex.from_frame(
            pd.concat([df.loc[rows, keys].astype(object), stall[rows].rename("stall")], axis=1)
        )
        for col, src in (("gp_draw_bias", "smooth_mean"), ("gp_draw_sd", "smooth_sd"),
                         ("gp_draw_n", "n")):
            out.loc[rows, col] = surf[src].reindex(want).values
    return out


def going_drift_surface(df: pd.DataFrame, meeting_cols=("race_date", "track"), order_col: str = "race_time",
                        value_col: str = "RSR") -> pd.DataFrame:
    """Smoothed drift of a speed residual across the races of each meeting.

    Positive drift = ground riding faster as the card goes on (or the
    reverse). Emits one row per race with the smoothed value and sd.
    """
    rows = []
    for key, g in df.groupby(list(meeting_cols)):
        g = g.sort_values(order_col)
        per_race = g.groupby(order_col)[value_col].mean().dropna()
        if len(per_race) < 3:
            continue
        x = np.arange(len(per_race), dtype=float)
        G, mean, sd, _ = gp_smooth(x, per_race.values, length_scale=1.5, grid=x)
        for (rt, raw), m, s in zip(per_race.items(), mean, sd):
            rows.append({**dict(zip(meeting_cols, key)), order_col: rt, "raw": raw, "drift_smooth": m, "drift_sd": s})
    return pd.DataFrame(rows)


def stall_adjacency(stalls, k: int = 1) -> np.ndarray:
    """Binary neighbour weights: stalls within k of each other."""
    s = np.asarray(stalls, float)
    W = (np.abs(s[:, None] - s[None, :]) <= k).astype(float)
    np.fill_diagonal(W, 0.0)
    return W


def morans_i(values, W: np.ndarray, n_perm: int = 999, random_state: int = 0) -> dict:
    """Moran's I with a permutation p-value (two-sided)."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(values, float)
    z = x - x.mean()
    n = len(x); S0 = W.sum()
    if S0 == 0 or np.sum(z**2) == 0:
        return {"I": np.nan, "expected": -1 / (n - 1), "p_value": np.nan}
    def _I(zz):
        return (n / S0) * (zz @ W @ zz) / np.sum(zz**2)
    I = _I(z)
    perms = np.array([_I(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(np.abs(perms) >= abs(I)) + 1) / (n_perm + 1)
    return {"I": float(I), "expected": -1 / (n - 1), "p_value": float(p), "z_score": float((I - perms.mean()) / perms.std())}
