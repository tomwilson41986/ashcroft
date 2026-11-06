"""The price-discovery scan's mechanics: costs, held bets, and that rules are chosen on the first half only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_price_discovery as pd_


def _frame(n=1400, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.choice([1.8, 2.5, 4.0], n)
    k = pd.date_range("2026-08-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"market_id": [f"1.{i}" for i in range(n)], "market_type": "MATCH_ODDS", "country": "GB",
                         "market_time": k.astype(str), "won": (rng.random(n) < 1 / p).astype(float),
                         "ltp_first": p, "ltp_t_1440": p, "ltp_t_60": p, "ltp_close": p, "kickoff": k})


def test_a_held_back_pays_the_tick_and_the_result():
    df = _frame(4)
    r = {"_h": df, "mark": "ltp_close", "side": "back"}
    prof = pd_._held_profit(r, np.ones(len(df), bool))
    from research.open_close.backtest import tick_under
    exp = np.where(df["won"] == 1, [tick_under(x) - 1 for x in df["ltp_close"]], -1.0)
    assert np.allclose(prof, exp)


def test_fair_prices_with_no_moves_yield_no_confirmed_rule():
    out = pd_.run(_frame())
    s = out["rule_scan"]
    assert s["rules_tried"] > 0 and s["confirmed"] == []


def test_the_summary_charges_commission_on_wins_only():
    s = pd_._summ(np.array([1.0, -1.0]))
    assert abs(s["roi"] - (0.98 - 1) / 2) < 1e-9


def test_markets_with_handicap_lines_are_left_out():
    df = pd.DataFrame({"market_id": ["a", "a", "b", "c"], "market_type": ["MATCH_ODDS", "MATCH_ODDS",
                                                                          "ASIAN_HANDICAP", "OVER_UNDER_25"],
                       "handicap": [None, None, -0.5, 0.0]})
    assert sorted(pd_.drop_line_markets(df)["market_id"].unique()) == ["a", "c"]
