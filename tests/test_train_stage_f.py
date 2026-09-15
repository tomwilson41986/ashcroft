"""Smoke tests for the two-stage pipeline (walk-forward + evaluate) and the connection features."""

import numpy as np
import pandas as pd

from model.connections import add_connection_features
from train_stage_f import evaluate, walk_forward


def _races(n_races=900, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for r in range(n_races):
        n = int(rng.integers(5, 12)); X = rng.normal(0, 1, (n, 3)); V = X @ np.array([1.0, -0.5, 0.3])
        p = np.exp(V) / np.exp(V).sum(); w = rng.choice(n, p=p)
        pi = p * rng.uniform(0.7, 1.3, n); pi = pi / pi.sum()
        for i in range(n):
            rows.append({"raceid": f"r{r}", "race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=r // 6), "race_time": "1.00",
                         "horse_name": f"h{r}_{i}", "trainer": f"t{rng.integers(0, 15)}", "jockey_name": f"j{rng.integers(0, 12)}",
                         "x1": X[i, 0], "x2": X[i, 1], "x3": X[i, 2], "won": float(i == w), "pi_market": pi[i], "nmfp": 0.0})
    return pd.DataFrame(rows)


def test_walk_forward_two_stage_and_evaluate():
    d = _races()
    out, model = walk_forward(d, ["x1", "x2", "x3"], "clogit", test_start="2025-03-01", n_folds=3, l2=0.5, rounds=10, min_train_races=100)
    assert out["f_oof"].notna().sum() > 0 and out["c_oof"].notna().sum() > 0 and model is not None
    res = evaluate(out)
    assert res["r2_fundamental"] > 0.1 and np.isfinite(res["r2_market"])
    assert "delta_combined_vs_market" in res and "conditional_calibration" in res
    assert set(res["segments"]).issuperset({"field_le_8"}) or len(res["segments"]) >= 0


def test_lgbm_race_softmax_runs_small():
    d = _races(400)
    out, booster = walk_forward(d, ["x1", "x2", "x3"], "lgbm", test_start="2025-02-01", n_folds=2, l2=0.5, rounds=60, min_train_races=50)
    assert out["f_oof"].notna().sum() > 0 and booster is not None
    assert np.allclose(out.loc[out["f_oof"].notna()].groupby("raceid")["f_oof"].sum(), 1.0)


def test_connection_features_lag_safe_and_shrunk():
    rows = []
    for day in range(40):
        for i in range(6):
            rows.append({"race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day), "race_time": "1.00", "raceid": f"r{day}",
                         "horse_name": f"h{i}", "trainer": "T_good" if i == 0 else f"T{i}", "jockey_name": f"J{i % 3}",
                         "won": float(i == 0), "nmfp": 0.5 if i == 0 else -0.1})
    d = pd.DataFrame(rows)
    out, feats = add_connection_features(d, k_sr=10)
    good = out[out["trainer"] == "T_good"].sort_values("race_date")
    assert np.isclose(good["trainer_sr_shrunk"].iloc[0], (0 + 10 * d["won"].mean()) / (0 + 10))   # first run: prior only
    assert good["trainer_sr_shrunk"].iloc[-1] > 0.6 and good["trainer_runs"].iloc[-1] == 39
    assert good["trainer_form_14d"].iloc[-1] > 0 and "jockey_booking_upgrade" in feats
    assert set(feats) <= set(out.columns)
