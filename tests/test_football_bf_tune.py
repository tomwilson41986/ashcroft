"""Tuning inside the walk-forward: choices from the training months only, on the registered grids."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_bf_tune as tu


def frame(n_months=6, per=900, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_months):
        m = f"2026-{k + 1:02d}"
        x = rng.normal(size=(per, 2))
        y = 0.3 * x[:, 0] - 0.2 * np.tanh(3 * x[:, 1]) + rng.normal(scale=0.5, size=per)
        for role in ("home", "draw", "away"):
            rows.append(pd.DataFrame({"month": m, "role": role, "a": x[:, 0], "b": x[:, 1], "y_logit": y,
                                      "usable": True, "kickoff": pd.Timestamp(m + "-15", tz="UTC")}))
    return pd.concat(rows, ignore_index=True)


def test_choices_are_on_the_grids_and_ignore_the_month_forecast():
    d = frame()
    tr = d[(d["role"] == "home") & (d["month"] < "2026-06")]
    s = tu.choose(tr, ["a", "b"])
    assert s["leaves"] in tu.LEAVES and s["alpha"] in tu.ALPHAS and 0 <= s["w_gbm"] <= 1
    assert 50 <= s["trees"] <= tu.MAX_TREES
    # the forecast month's outcomes are never read: scrambling them leaves the forecasts' settings unchanged
    d2 = d.copy()
    d2.loc[d2["month"] == "2026-06", "y_logit"] = 99.0
    _, c1 = tu.walk_forward_tuned(d, ["a", "b"], min_train_months=5)
    _, c2 = tu.walk_forward_tuned(d2, ["a", "b"], min_train_months=5)
    assert c1 == c2 and len(c1) == 3
