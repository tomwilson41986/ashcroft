"""The expected-value paper trader's mechanics: the pre-registration and frozen models, the expected value, stakes,
the bank."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from research.open_close import football_ev_trial as ev


def test_the_preregistered_hashes_match_the_frozen_models():
    from research.open_close.football_exchange import read_prereg
    rule, _ = read_prereg(ev.PREREG)
    assert rule["models_dir"] == ev.MODELS and rule["ev_margin"] == 0.02
    for name, want in rule["model_sha256"].items():
        assert hashlib.sha256((Path(ev.MODELS) / name).read_bytes()).hexdigest() == want


def test_expected_value_backs_a_rising_chance_and_lays_a_falling_one_net_of_the_round_trip():
    cal = {"a": 0.0, "b": 1.0}
    side, val = ev.expected_value([0.10, -0.10, 0.0], cal, [2.5, 2.5, 2.5], 0.02)
    assert side[0] == 1 and side[1] == -1
    assert val[0] > 0.02 and val[1] > 0.02 and val[2] < 0                    # no move: the round trip is a loss
    assert np.isnan(ev.expected_value([0.1], cal, [np.nan], 0.02)[1][0])      # no early price: no value


def test_capital_shares_follow_expected_value_within_the_budget_and_the_cap():
    f = ev.capital_shares(np.array([0.06, 0.03, 0.03]), budget=0.25, cap=0.10)
    assert abs(f.sum() - 0.25) < 1e-9 and f[0] == 0.10 and abs(f[1] - 0.075) < 1e-9
    assert ev.capital_shares(np.array([0.05]), 0.25, 0.10)[0] == 0.10           # one signal: capped, rest unused


def test_the_bank_moves_by_each_window_and_a_lay_is_staked_on_its_liability():
    rule = {"commission": 0.02, "staking": {"bank_gbp": 1000.0, "window_budget": 0.25, "max_per_trade": 0.10}}
    sig = pd.DataFrame({"event_id": ["m1", "m1", "m2"], "selection_id": ["H", "A", "H"], "side": [1, -1, 1],
                        "ev": [0.04, 0.04, 0.04], "collection": ["2026-11-06", "2026-11-06", "2026-11-10"]})
    trades = pd.DataFrame({"match_id": ["m1", "m1"], "selection": ["H", "A"], "side": [1, -1],
                           "fill_open": [2.0, 5.0], "fill_close": [1.9, 5.5], "profit": [2.0 / 1.9 - 1, 1 - 5.0 / 5.5],
                           "clv": [0.05, 0.09]})                              # m2 unfilled: its capital comes back
    staked, bank = ev.bank_path(sig, trades, rule)
    lay = staked[staked["selection"] == "A"].iloc[0]
    assert lay["capital_gbp"] == 100.0 and abs(lay["stake_gbp"] - 25.0) < 1e-9     # liability 100 at 5.0
    want = (100 * (2.0 / 1.9 - 1) + 25 * (1 - 5.0 / 5.5)) * 0.98
    assert abs(bank["pnl_gbp"] - round(want, 2)) < 0.02 and bank["windows"] == 2


def test_score_waits_for_the_football_book_record(tmp_path, monkeypatch):
    from research.open_close import football_paper_trial as pt
    monkeypatch.setattr(pt, "first_recorded_day", lambda *a, **k: None)
    res = ev.score(str(tmp_path), ev.MODELS, None, str(tmp_path), None, "2026-11-04")
    assert res["status"] == "no football book record found" and res["rule"]["ev_margin"] == 0.02


def test_scorers_wait_while_the_first_collection_is_after_as_of(tmp_path, monkeypatch):
    """A record that began midweek opens the trial at the next Friday: before then each scorer says so (it must not
    fit an empty table, which crashed the stacked rule's scorer in its first scheduled run)."""
    from datetime import date
    from research.open_close import football_paper_trial as pt
    from research.open_close import football_stack_resting as rs
    from research.open_close import football_stack_trial as st
    monkeypatch.setattr(pt, "first_recorded_day", lambda *a, **k: date(2026, 11, 4))
    for res in (ev.score(str(tmp_path), ev.MODELS, None, str(tmp_path), None, "2026-11-05"),
                st.score(str(tmp_path), "models/football/stack_v1", None, str(tmp_path), None, "2026-11-05"),
                rs.score(str(tmp_path), "models/football/stack_v1", None, str(tmp_path), None, "2026-11-05")):
        assert res["status"] == "not started" and res["start"] == "2026-11-06"
