"""cricket/model/benter.py: markets to matches, the regression's standard errors, the indicative strategy; and the
Phase 0 gate (cricket/model/gate.py).

The Betfair rows are hand-written in the historic table's shape to drive the matching rules; they are not market
data and nothing is scored on them."""

import numpy as np
import pandas as pd
import pytest

from cricket.model.benter import logistic, market_table, match_markets, same_side, strategy
from cricket.model.gate import model_gate


def _bf(mid, t, r1, r2, o1, o2, won1):
    base = {"market_id": mid, "market_type": "MATCH_ODDS", "event_name": f"{r1} v {r2}", "market_time": t,
            "went_in_play_utc": t, "ltp_t_60": None, "ltp_t_1": None}
    return [dict(base, selection_id=1, runner_name=r1, sort_priority=1, runner_status="WINNER" if won1 else "LOSER",
                 won=int(won1), ltp_close=o1),
            dict(base, selection_id=2, runner_name=r2, sort_priority=2, runner_status="LOSER" if won1 else "WINNER",
                 won=int(not won1), ltp_close=o2)]


def _pre(mid, day, bat, bowl, bat_won, p=0.5, year=2024):
    return {"match_id": mid, "start_date": day, "year": year, "event_name": "IPL", "bat_first": bat,
            "bowl_first": bowl, "bat_won": bat_won, "p_bat_first": p}


def test_names():
    assert same_side("Royal Challengers Bengaluru", "Royal Challengers Bengaluru")
    assert same_side("Chennai Super Kings", "chennai super kings")
    assert not same_side("India W", "India")                  # Betfair's women's sides: never the men's
    assert not same_side("India Legends", "India")
    assert not same_side("Pakistan A", "Pakistan")
    assert not same_side("Mumbai Indians", "Delhi Capitals")


def test_markets_match_by_day_and_both_sides_in_either_order():
    bf = pd.DataFrame(_bf("1.1", "2024-04-01T14:00:00.000Z", "Delhi Capitals", "Mumbai Indians", 2.0, 1.9, True)
                      + _bf("1.2", "2024-04-02T14:00:00.000Z", "Kolkata Knight Riders", "Punjab Kings", 1.5, 2.8, False)
                      + _bf("1.3", "2024-04-09T14:00:00.000Z", "Gujarat Titans", "Lucknow Super Giants", 1.8, 2.1,
                            True))
    pre = pd.DataFrame([_pre("m1", "2024-04-01", "Mumbai Indians", "Delhi Capitals", 0),     # runner 1 bowls first
                        _pre("m2", "2024-04-03", "Kolkata Knight Riders", "Punjab Kings", 0),  # a day later: ok
                        _pre("m3", "2024-04-09", "Gujarat Titans", "Lucknow Super Giants", 0)])  # results disagree
    j, c = match_markets(market_table(bf), pre)
    assert c["markets"] == 3 and c["matched"] == 2 and c["unmatched"] == 1
    m1 = j.set_index("match_id").loc["m1"]
    assert m1.o_bat == 1.9 and m1.o_bowl == 2.0
    assert m1.p_market == pytest.approx((1 / 1.9) / (1 / 1.9 + 1 / 2.0))
    assert c["unmatched_sample"][0]["why"] == "Betfair's result differs from Cricsheet's"


def test_logistic_standard_errors_match_the_numerical_hessian():
    # outcomes from the cricket fixtures' own results order; regressors spread over a range
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1] * 5, dtype=float)
    x = np.c_[np.linspace(-2, 2, len(y)), np.cos(np.arange(len(y)))]
    b, se = logistic(x, y)
    X = np.c_[np.ones(len(y)), x]

    def nll(beta):
        z = X @ beta
        return np.sum(np.log1p(np.exp(z)) - y * z)
    h, H = 1e-4, np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            e_i, e_k = np.eye(3)[i] * h, np.eye(3)[k] * h
            H[i, k] = (nll(b + e_i + e_k) - nll(b + e_i - e_k) - nll(b - e_i + e_k) + nll(b - e_i - e_k)) / (4 * h * h)
    assert np.allclose(se, np.sqrt(np.diag(np.linalg.inv(H))), rtol=1e-3)


def test_strategy_settles_after_commission():
    w = pd.DataFrame({"p_comb": [0.6, 0.3, 0.5], "o_bat": [2.0, 2.0, 2.0], "o_bowl": [2.0, 1.6, 2.0],
                      "bat_won": [1, 1, 0], "year": [2024, 2024, 2025]})
    s = strategy(w, threshold=0.02, c=0.05)
    # 1: back bat at 2.0, value 0.6 x 0.95 - 0.4 = 0.17, wins 0.95; 2: back bowl at 1.6, value 0.7 x 0.6 x 0.95 - 0.3
    # = 0.099, loses 1; 3: no side clears 0.02 (0.5 x 0.95 - 0.5 < 0)
    assert s["bets"] == 2 and s["pnl_after_commission"] == pytest.approx(0.95 - 1.0)
    assert s["by_year"] == {2024: pytest.approx(-0.05), 2025: 0.0}


def test_gate_needs_the_whole_interval_below_zero():
    t = pd.DataFrame({"bat_won": [1, 0] * 50, "match_id": np.repeat(np.arange(25), 4)})
    better = model_gate(t.assign(p_cal=np.where(t.bat_won == 1, 0.99, 0.01), p_base=0.5))
    assert better["pass"] and better["bootstrap"]["hi"] < 0
    sharp_but_off = model_gate(t.assign(p_cal=np.where(t.bat_won == 1, 0.8, 0.2), p_base=0.5))
    assert sharp_but_off["checks"]["beats the same-rows baseline (95% interval of the difference below 0)"]
    assert not sharp_but_off["pass"]                                   # ECE 0.2: over the 0.131 limit
    same = model_gate(t.assign(p_cal=0.5, p_base=0.5))
    assert not same["pass"]
