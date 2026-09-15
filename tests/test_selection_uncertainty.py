"""Tests for FDR-controlled selection and uncertainty tools."""

import numpy as np
import pandas as pd

from model import selection, uncertainty as unc


def test_benjamini_hochberg_controls_threshold():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 1, 200), rng.uniform(0, 1e-4, 10)])
    r = selection.benjamini_hochberg(p, q=0.1)
    assert r["n_rejected"] >= 10
    assert r["reject"][-10:].all()
    assert r["n_rejected"] < 40


def test_knockoffs_select_true_features():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (1500, 30)); beta = np.zeros(30); beta[:5] = [1, -1, 0.8, 0.7, -0.6]
    y = X @ beta + rng.normal(0, 1, 1500)
    res = selection.knockoff_filter(pd.DataFrame(X, columns=[f"f{i}" for i in range(30)]), y, fdr=0.2, random_state=0)
    sel = set(res["selected"][res["selected"]].index)
    assert {"f0", "f1", "f2", "f3"}.issubset(sel)
    assert len(sel - {"f0", "f1", "f2", "f3", "f4"}) <= 2


def test_knockoff_threshold_inf_when_no_signal():
    W = np.array([0.1, -0.1, 0.05, -0.05, 0.02, -0.02])
    assert selection.knockoff_threshold(W, fdr=0.1) == np.inf


def test_univariate_pvalues_race_demeaned():
    rng = np.random.default_rng(2)
    race = np.repeat(np.arange(300), 8)
    x = rng.normal(0, 1, 2400); y = x + rng.normal(0, 1, 2400)
    X = pd.DataFrame({"sig": x, "noise": rng.normal(0, 1, 2400)})
    pv = selection.univariate_pvalues(X, y, race_id=race)
    assert pv.loc["sig", "p_value"] < 1e-6
    assert pv.loc["noise", "p_value"] > 0.001


def test_split_conformal_coverage():
    rng = np.random.default_rng(3)
    yp = rng.normal(0, 1, 4000); y = yp + rng.normal(0, 0.5, 4000)
    c = unc.SplitConformalRegressor(alpha=0.1).fit(y[:2000], yp[:2000])
    cov = c.coverage(y[2000:], yp[2000:])
    assert 0.87 < cov < 0.93


def test_prob_interval_orders_and_normalises():
    lo, pt, hi = unc.prob_interval_from_log_bfsp(np.log([3, 5, 10]), 0.3, race_id=[1, 1, 1])
    assert (lo < pt).all() and (pt < hi).all()
    assert abs(pt.sum() - 1) < 1e-9


def test_conformal_kelly_zero_without_edge():
    stake = unc.conformal_kelly_stake(p_lo=[0.2], p_point=[0.3], odds=[3.0])
    assert stake[0] == 0.0
    stake = unc.conformal_kelly_stake(p_lo=[0.4], p_point=[0.5], odds=[3.0])
    assert stake[0] > 0


def test_block_bootstrap_resamples_groups():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"g": np.repeat(np.arange(50), 4), "v": rng.normal(1, 1, 200)})
    s = unc.block_bootstrap(df, "g", lambda d: d["v"].mean(), n_boot=200, random_state=0)
    assert len(s) == 200 and abs(s.mean() - df["v"].mean()) < 0.1
    lo, hi = unc.bootstrap_ci(s)
    assert lo < df["v"].mean() < hi


def test_stationary_bootstrap_shapes():
    idx = unc.stationary_bootstrap_indices(100, mean_block=10, n_boot=5, random_state=0)
    assert idx.shape == (5, 100) and idx.min() >= 0 and idx.max() < 100
