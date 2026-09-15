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
    going_drift_surface    Smoothed race-to-race pace-of-ground drift
                           within a meeting (from speed residuals)
    stall_adjacency        Neighbour weights for Moran's I
    morans_i               Spatial autocorrelation with permutation p-value
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


def smooth_draw_bias(df: pd.DataFrame, track: str, dist_furlongs: float, value_col: str = "won",
                     stall_col: str = "stall", dist_tol: float = 0.5, min_runners: int = 8,
                     by_going: bool = False, going_col: str = "going_description") -> pd.DataFrame:
    """GP-smoothed win rate by stall for one track / distance bucket.

    Returns a DataFrame with stall, raw_rate, n, smooth_mean, smooth_sd and
    the field-size-normalised expected rate (1/runners) for comparison.
    """
    d = df[(df["track"].astype(str).str.lower() == str(track).lower())
           & ((df["dist_furlongs"] - dist_furlongs).abs() <= dist_tol)
           & (df["number_of_runners"] >= min_runners)].copy()
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
