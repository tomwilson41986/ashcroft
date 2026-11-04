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


def _toy_matrix(n_seasons=4, per=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2021, 2021 + n_seasons):
        for i in range(per):
            x = rng.normal()
            rows.append({"season": s, "test_season": s, "x": x, "y_h": 0.05 * x + rng.normal(0, 0.02),
                         "same_book": True, "fd_code": "E0" if i % 3 == 0 else "E1",
                         "league": "E0" if i % 3 == 0 else "E1"})
    d = pd.DataFrame(rows)
    d["league"] = pd.Categorical(d["league"])
    return d


def test_the_cost_classifier_signs_follow_the_move(monkeypatch):
    from research.open_close import football_stack_trial as st
    monkeypatch.setattr(st, "TRAIN_FROM", 2021)
    d = _toy_matrix()
    p = fi.walk_forward_clf(d, ["x"], "y_h", "same_book", 2024)
    te = d[d["test_season"] == 2024]
    assert len(p) == len(te) and np.corrcoef(p, te["x"])[0, 1] > 0.8 and (np.abs(p) <= 1).all()


def test_per_tier_fits_cover_every_test_match_and_outside_top5_drops_them(monkeypatch):
    from research.open_close import football_stack_trial as st
    monkeypatch.setattr(st, "TRAIN_FROM", 2021)
    d = _toy_matrix()
    p = fi.walk_forward_tier(d, ["x"], "y_h", "same_book", 2024)
    assert np.isfinite(p).all() and np.corrcoef(p, d[d["test_season"] == 2024]["x"])[0, 1] > 0.8
    c = pd.DataFrame({"league": ["E0", "E1", "D1", "N1"]})
    assert fi.outside_top5(c)["league"].tolist() == ["E1", "N1"]


def test_the_ev_rule_takes_every_trade_whose_expected_move_beats_its_cost():
    te = pd.DataFrame({"match_id": ["m1", "m2", "m3"], "match_date": ["2026-11-07"] * 3, "week": "w",
                       "test_season": 2025, "fd_code": "SP2",
                       "bfe_open_h": [2.0, 2.0, 2.0], "bfe_open_d": [3.6, 3.6, 3.6], "bfe_open_a": [4.4, 4.4, 4.4],
                       "bfe_close_h": [1.8, 2.3, 2.0], "bfe_close_d": [3.7, 3.5, 3.6], "bfe_close_a": [5.0, 3.8, 4.4],
                       "p_h": [0.55, 0.40, 0.5], "p_a": [0.20, 0.25, 0.25]})
    c = fi.candidates(te, {"H": np.array([0.10, -0.10, 0.001]), "A": np.array([0.0, 0.0, 0.0])}, "MO")
    assert (c["cost_tick_b"] > 0).all() and (c["cost_sym_l"] > 0).all()   # a round trip with no move loses
    cal = {"a": 0.0, "b": 1.0}
    got = fi.by_ev(c, cal, "tick").set_index(["match_id", "leg"])
    assert ("m1", "H") in got.index and got.loc[("m1", "H"), "side"] == 1     # chance up 10%: back first
    assert ("m2", "H") in got.index and got.loc[("m2", "H"), "side"] == -1    # down 10%: lay first
    assert ("m3", "H") not in got.index and ("m1", "A") not in got.index     # no edge past the cost: no trade
    assert (got["ev"] > 0).all()
    # a calibration that halves the forecast keeps fewer trades, never more
    assert len(fi.by_ev(c, {"a": 0.0, "b": 0.5}, "tick")) <= len(got)
