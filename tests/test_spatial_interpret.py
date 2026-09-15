"""Tests for GP smoothing, Moran's I, ALE and conditional permutation importance."""

import numpy as np
import pandas as pd

from model import interpret, spatial


def test_gp_smooth_tracks_trend():
    rng = np.random.default_rng(0)
    stalls = np.arange(1, 21); truth = 0.1 - 0.003 * stalls
    rate = truth + rng.normal(0, 0.01, 20); nobs = rng.integers(50, 200, 20)
    _, mean, sd, _ = spatial.gp_smooth(stalls, rate, nobs, grid=stalls)
    assert np.corrcoef(mean, truth)[0, 1] > 0.95
    assert (sd > 0).all()


def test_smooth_draw_bias_returns_all_stalls():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"track": ["Chester"] * 3000, "dist_furlongs": 5.0, "number_of_runners": 12,
                      "stall": rng.integers(1, 13, 3000)})
    d["won"] = (rng.random(3000) < (0.15 - 0.008 * d["stall"])).astype(int)
    t = spatial.smooth_draw_bias(d, "chester", 5)
    assert set(t["stall"]) == set(range(1, 13))
    assert t["smooth_mean"].iloc[0] > t["smooth_mean"].iloc[-1]  # low draw favoured


def test_morans_i_detects_autocorrelation():
    stalls = np.arange(1, 21)
    trend = 0.1 - 0.003 * stalls
    res = spatial.morans_i(trend, spatial.stall_adjacency(stalls))
    assert res["I"] > 0.5 and res["p_value"] < 0.01
    rng = np.random.default_rng(2)
    res2 = spatial.morans_i(rng.normal(0, 1, 20), spatial.stall_adjacency(stalls))
    assert res2["p_value"] > 0.01 or abs(res2["I"]) < 0.4


def test_ale_recovers_linear_slope():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(0, 1, (800, 3)), columns=list("abc"))
    fn = lambda D: 2 * D["a"].values + 0.5 * D["c"].values ** 2
    a = interpret.accumulated_local_effects(fn, X, "a", n_bins=10)
    assert abs(np.polyfit(a["x"], a["ale"], 1)[0] - 2.0) < 0.05
    b = interpret.accumulated_local_effects(fn, X, "b", n_bins=10)
    assert np.abs(b["ale"]).max() < 1e-9


def test_conditional_permutation_importance_ignores_correlated_proxy():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(0, 1, (600, 3)), columns=list("abc"))
    X["b"] = X["a"] * 0.9 + rng.normal(0, 0.4, 600)
    fn = lambda D: 2 * D["a"].values
    y = fn(X)
    imp = interpret.conditional_permutation_importance(fn, X, y, lambda yy, p: -np.mean((yy - p) ** 2), n_repeats=2)
    imp = imp.set_index("feature")
    assert imp.loc["a", "importance"] > 1.0
    assert imp.loc["b", "importance"] == 0.0
