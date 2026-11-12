"""The closing-price efficiency study: its cells and its slope say what they should on made-up selections."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_bf_efficiency as ef


def test_a_calibrated_market_has_slope_one_and_an_overconfident_one_less():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 40_000)
    won = (rng.uniform(size=p.size) < p).astype(float)
    fair = ef.logistic_slope(p, won)
    assert abs(fair["slope"] - 1) < 3 * fair["slope_se"] + 0.02 and abs(fair["intercept"]) < 0.05
    stretched = 1 / (1 + np.exp(-1.5 * np.log(p / (1 - p))))          # prices more extreme than the truth
    assert ef.logistic_slope(stretched, won)["slope"] < 0.75


def test_a_cell_reports_the_gap_and_the_return_to_backing_at_the_close():
    n = 400
    d = pd.DataFrame({"day": [f"d{i % 40}" for i in range(n)], "p": 0.5, "p_raw": 0.5,
                      "won": [1.0] * 220 + [0.0] * 180})
    c = ef.cell(d)
    assert c["n"] == n and c["mean_chance"] == 0.5 and c["rate"] == 0.55 and c["gap"] == 0.05
    assert abs(c["back_return"] - 0.10) < 1e-9                         # 0.55 / 0.5 - 1
    assert ef.cell(d.head(50)) == {"n": 50}


def test_the_slopes_standard_error_counts_a_match_once_not_once_a_selection():
    # each match's row three times over: no more information, so the clustered error must not shrink
    rng = np.random.default_rng(2)
    p = rng.uniform(0.1, 0.9, 6_000)
    won = (rng.uniform(size=p.size) < p).astype(float)
    match = np.arange(p.size)
    once = ef.logistic_slope(p, won, match)
    naive = ef.logistic_slope(np.tile(p, 3), np.tile(won, 3))
    clustered = ef.logistic_slope(np.tile(p, 3), np.tile(won, 3), np.tile(match, 3))
    assert clustered["slope"] == once["slope"] == naive["slope"]
    assert abs(clustered["slope_se"] - once["slope_se"]) < 0.1 * once["slope_se"]
    assert naive["slope_se"] < 0.65 * once["slope_se"]                 # 1 / sqrt(3): what clustering puts right
