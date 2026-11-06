"""The cross-league price test's mechanics: markets made wide, linked to fixtures, both targets priced right."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_cross_league_price as cp


def _runners(mid, home, away, t, early, close, winner):
    rows = []
    for name, leg in ((home, "h"), (away, "a"), ("The Draw", "d")):
        rows.append({"market_id": mid, "event_id": mid, "event_name": f"{home} v {away}", "market_type": "MATCH_ODDS",
                     "country": "GB", "market_time": t, "runner_name": name, "ltp_t_1440": early[leg],
                     "ltp_first": 9.0, "ltp_close": close[leg], "won": float(leg == winner)})
    return rows


def test_wide_uses_the_24h_price_normalises_and_keeps_the_result():
    m = pd.DataFrame(_runners("1.1", "Arsenal", "Chelsea", "2025-09-01T15:00:00Z",
                              {"h": 2.0, "d": 4.0, "a": 4.0}, {"h": 1.8, "d": 4.0, "a": 5.0}, "h"))
    w = cp.wide(m)
    assert len(w) == 1
    r = w.iloc[0]
    assert r["p_early_h"] == 2.0 and r["p_close_a"] == 5.0 and r["won_h"] == 1
    assert abs(r["qe_h"] + r["qe_d"] + r["qe_a"] - 1) < 1e-12 and abs(r["qe_h"] - 0.5) < 1e-12


def test_link_needs_both_sides_and_the_kickoff_and_checks_the_result():
    m = pd.DataFrame(_runners("1.1", "Man Utd", "Wolves", "2025-09-01T15:00:00Z",
                              {"h": 2.0, "d": 4.0, "a": 4.0}, {"h": 2.0, "d": 4.0, "a": 4.0}, "h")
                     + _runners("1.2", "Leeds", "Burnley", "2025-09-01T15:00:00Z",
                                {"h": 2.0, "d": 4.0, "a": 4.0}, {"h": 2.0, "d": 4.0, "a": 4.0}, "a"))
    d = pd.DataFrame({"kickoff": pd.to_datetime(["2025-09-01T15:00:00Z", "2025-09-01T18:00:00Z"], utc=True),
                      "home": ["Manchester United", "Leeds"], "away": ["Wolverhampton Wanderers", "Burnley"],
                      "group": "league_matches", "diff_v2": 0.0, "diff_league_only": 0.0, "gd": [1, -1],
                      "competition": "England | Premier League"})
    lk = cp.link(cp.wide(m), d)
    assert lk["market_id"].tolist() == ["1.1"]                 # Leeds kick off 3 h later: no link
    assert bool(lk["result_agrees"].iloc[0])


def test_targets_price_the_green_and_the_held_bet():
    p = pd.DataFrame({"market_id": ["1"], "p_early_h": [3.0], "p_close_h": [2.5], "won_h": [1.0], "pm_h": [0.5],
                      "p_early_a": [3.0], "p_close_a": [3.0], "won_a": [0.0], "pm_a": [0.1]})
    t = cp.targets(p)
    green = t["closing_price_target"]["backed_early_closed_at_close_ltp"]
    assert green["bets"] == 1 and abs(green["roi"] - (3.0 / 2.5 - 1) * (1 - cp.COMMISSION)) < 1e-4
    held = t["winning_the_bet_target"]["backed_early_held_to_result"]
    assert abs(held["roi"] - 2.0 * (1 - cp.COMMISSION)) < 1e-4 and held["strike"] == 1.0
    assert t["winning_the_bet_target"]["backed_at_close_held_to_result"]["bets"] == 1   # 0.5 x 2.5 - 1 > 2%
    assert np.isfinite(t["closing_price_target"]["same_a_tick_worse_each_end"]["roi"])
