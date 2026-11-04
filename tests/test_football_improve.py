"""The improvement experiments' mechanics: the trade a forecast points to, rank selection, the cost filter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_improve as fi


def test_a_forecast_rise_in_chance_backs_first_and_the_profit_follows_the_close():
    te = pd.DataFrame({"match_id": ["m1", "m2"], "match_date": ["2026-11-07", "2026-11-07"], "week": "w",
                       "test_season": 2025, "fd_code": "E0",
                       "bfe_open_h": [2.0, 2.0], "bfe_open_d": [3.6, 3.6], "bfe_open_a": [4.4, 4.4],
                       "bfe_close_h": [1.8, 2.3], "bfe_close_d": [3.7, 3.5], "bfe_close_a": [5.0, 3.8],
                       "p_h": [0.55, 0.40], "p_a": [0.20, 0.25]})
    c = fi.candidates(te, {"H": np.array([0.05, 0.05]), "A": np.array([-0.05, -0.05])}, "MO")
    h = c[c["leg"] == "H"].set_index("match_id")
    assert (h["side"] == 1).all()                                  # chance up: back first
    assert h.loc["m1", "profit_sym"] > 0 and h.loc["m2", "profit_sym"] < 0   # shortened / drifted
    assert h.loc["m1", "right_way"] and not h.loc["m2", "right_way"]
    a = c[c["leg"] == "A"].set_index("match_id")
    assert (a["side"] == -1).all() and a.loc["m1", "profit_sym"] > 0         # laid first, drifted to 5.0


def test_rank_selection_takes_the_strongest_k_a_week():
    c = pd.DataFrame({"week": ["a"] * 4 + ["b"] * 2, "apred": [0.01, 0.09, 0.05, 0.03, 0.02, 0.07]})
    got = fi.by_rank(c, 2)
    assert sorted(got["apred"]) == [0.02, 0.05, 0.07, 0.09]


def test_the_cost_filter_scores_every_test_candidate():
    rng = np.random.default_rng(0)
    n = 3000
    train = pd.DataFrame({"apred": rng.uniform(0, 0.1, n), "side": rng.choice([-1, 1], n), "price": rng.uniform(1.5, 6, n),
                          "over": rng.uniform(1.0, 1.05, n), "leg": rng.choice(["H", "A"], n),
                          "league": rng.choice(["E0", "D1"], n)})
    train["profit_sym"] = rng.normal(0.0, 0.05, n) + 0.3 * (train["apred"] - 0.05)
    test = train.sample(500, random_state=1)
    score, info = fi.meta_filter(train, test)
    assert len(score) == 500 and ((score >= 0) & (score <= 1)).all() and info["train_rows"] > 0


def test_daily_top_k_and_the_model_gap_trade_toward_the_model():
    c = pd.DataFrame({"day": ["d1"] * 3 + ["d2"] * 2, "apred": [0.01, 0.05, 0.03, 0.02, 0.04],
                      "gap": [0.30, -0.10, 0.05, -0.20, 0.01], "sym_b": [0.1] * 5, "sym_l": [-0.1] * 5,
                      "tick_b": [0.05] * 5, "tick_l": [-0.05] * 5, "move_dir": [-1, 1, -1, 1, -1]})
    assert sorted(fi.by_day(c, 1)["apred"]) == [0.04, 0.05]
    g = fi.by_day_gap(c, 1).set_index("gap")
    assert sorted(g.index) == [-0.2, 0.3]
    assert g.loc[0.3, "side"] == 1 and g.loc[0.3, "profit_sym"] == 0.1 and bool(g.loc[0.3, "right_way"])
    assert g.loc[-0.2, "side"] == -1 and g.loc[-0.2, "profit_sym"] == -0.1 and bool(g.loc[-0.2, "right_way"])
