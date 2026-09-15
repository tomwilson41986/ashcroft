"""Recovery tests for the cross-classified effects model and causal designs."""

import numpy as np
import pandas as pd
import pytest

from model import causal, effects


def _entangled_frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    J = rng.integers(0, 25, n); T = rng.integers(0, 20, n)
    # make jockeys ride mostly for one trainer (collinearity)
    T = np.where(rng.random(n) < 0.6, J % 20, T)
    je = rng.normal(0, 2.0, 25); te = rng.normal(0, 2.0, 20)
    y = 70 + je[J] + te[T] + rng.normal(0, 3, n)
    return pd.DataFrame({"jockey_name": [f"j{i}" for i in J], "trainer": [f"t{i}" for i in T], "perf": y}), je, te


def test_cross_classified_ridge_recovers_entangled_effects():
    df, je, te = _entangled_frame()
    m = effects.CrossClassifiedRidge(["jockey_name", "trainer"], lambdas=5.0).fit(df, "perf", eb_iterations=3)
    est_j = m.effects_["jockey_name"].reindex([f"j{i}" for i in range(25)]).values
    est_t = m.effects_["trainer"].reindex([f"t{i}" for i in range(20)]).values
    assert np.corrcoef(est_j, je)[0, 1] > 0.9
    assert np.corrcoef(est_t, te)[0, 1] > 0.9
    pred = m.predict(df)
    assert np.corrcoef(pred, df["perf"])[0, 1] > 0.6


def test_prior_column_shifts_target():
    df, _, _ = _entangled_frame(2000)
    df["prior"] = 70.0
    m = effects.CrossClassifiedRidge(["jockey_name"], lambdas=5.0, prior_col="prior").fit(df, "perf")
    assert abs(m.intercept_) < 1.0  # explained target is perf - 70


def test_walk_forward_effects_is_lag_safe():
    df, _, _ = _entangled_frame(3000)
    df["race_date"] = pd.date_range("2022-01-01", periods=len(df), freq="6h")
    out = effects.walk_forward_effects(df, "perf", ["jockey_name", "trainer"], min_train_days=60, freq="MS", eb_iterations=0)
    first = out["rapm_pred"].first_valid_index()
    assert first is not None
    assert out.loc[: first - 1, "rapm_pred"].isna().all()  # nothing scored before the first window
    assert out["rapm_jockey_name"].notna().sum() > 0


def test_aipw_removes_confounding():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (4000, 3))
    ps = 1 / (1 + np.exp(-(1.2 * X[:, 0] - 0.3)))
    t = (rng.random(4000) < ps).astype(int)
    y = 0.5 * t + 1.0 * X[:, 0] + rng.normal(0, 1, 4000)
    naive = y[t == 1].mean() - y[t == 0].mean()
    res = causal.aipw_effect(y, t, X)
    assert abs(naive - 0.5) > 0.5
    assert abs(res["estimate"] - 0.5) < 3 * res["se"] + 0.05
    ipw = causal.ipw_effect(y, t, causal.fit_propensity(X, t))
    assert abs(ipw["estimate"] - 0.5) < 0.15


def test_did_recovers_treatment_effect():
    rng = np.random.default_rng(2)
    units, times = 400, 4
    d = pd.DataFrame({"unit": np.repeat(np.arange(units), times), "time": np.tile(np.arange(times), units)})
    d["treated"] = (d["unit"] < units // 2).astype(int); d["post"] = (d["time"] >= 2).astype(int)
    d["D"] = d["treated"] * d["post"]
    d["y"] = 0.1 * d["unit"] + 0.5 * d["time"] + 2.0 * d["D"] + rng.normal(0, 1, len(d))
    panel = causal.did_panel(d, "unit", "time", "D", "y")
    assert abs(panel["estimate"] - 2.0) < 3 * panel["se"] + 0.05
    assert abs(causal.did_2x2(d["y"], d["treated"], d["post"])["estimate"] - 2.0) < 0.3


def test_rd_estimate_detects_jump():
    rng = np.random.default_rng(3)
    x = rng.uniform(60, 100, 5000)
    y = 0.02 * x + 1.5 * (x >= 80) + rng.normal(0, 0.5, 5000)
    res = causal.rd_estimate(x, y, cutoff=80, bandwidth=8)
    assert abs(res["estimate"] - 1.5) < 0.3


def test_e_value_known_values():
    assert abs(causal.e_value(2.0)["e_value"] - (2 + np.sqrt(2))) < 1e-9
    assert causal.e_value(1.0)["e_value"] == 1.0
    assert abs(causal.e_value(0.5)["e_value"] - causal.e_value(2.0)["e_value"]) < 1e-9
    assert causal.e_value(1.5, ci_limit=0.9)["e_value_ci"] == 1.0


def test_first_time_flag_lag_safe():
    df = pd.DataFrame({"horse_name": ["a", "a", "a", "b", "b"], "headgear": ["", "b", "b", "v", ""],
                       "race_date": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-01-01", "2025-02-01"],
                       "race_time": ["1"] * 5})
    assert causal.first_time_flag(df, "headgear").tolist() == [0, 1, 0, 1, 0]
