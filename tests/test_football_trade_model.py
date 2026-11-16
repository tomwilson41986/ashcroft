"""The football trade model's arithmetic: the trade-out and win-bet results a unit staked, and their summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.open_close import football_bf_close_model as cm
from research.open_close import football_trade_model as tm


def test_a_trade_out_is_the_same_result_whatever_the_match_does():
    # backed at 2.20 (taken a little worse), laid off at an SP of 2.00: stake 1 at 2.2c, lay 2.2c / 2.0 at 2.0
    cost = 0.0115
    got = tm.tradeout_cash([2.2, 2.0], [2.0, 2.2], cost)
    eff = 2.2 * np.exp(-cost)
    lay = eff / 2.0
    wins, loses = (eff - 1) - lay * (2.0 - 1), -1 + lay
    assert wins == pytest.approx(loses) and got[0] == pytest.approx(wins)
    assert got[1] < 0                                     # the price drifted: a loss either way
    assert tm.net(np.array([0.10, -0.10])).tolist() == pytest.approx([0.10 * (1 - cm.COMMISSION), -0.10])


def test_a_win_bet_pays_the_price_taken_or_loses_the_stake():
    got = tm.win_cash([3.0, 3.0], [1, 0], 0.0)
    assert got.tolist() == [2.0, -1.0]
    assert tm.win_cash([3.0], [1], 0.0115)[0] == pytest.approx(3.0 * np.exp(-0.0115) - 1)


def test_the_summary_is_a_mean_a_bet_with_a_day_interval_and_says_when_there_are_too_few():
    day = np.repeat([f"d{i}" for i in range(40)], 2)
    cash = np.tile([0.3, -0.1], 40)
    s = tm.summary(day, cash)
    assert (s["bets"], s["days"]) == (80, 40) and s["return"] == pytest.approx(0.1) and s["won_share"] == 0.5
    assert s["lo90"] == pytest.approx(0.1) and s["hi90"] == pytest.approx(0.1)       # every day is the same
    assert tm.summary(day[:10], cash[:10]) == {"bets": 10}
    d = tm.summary(day, cash + np.random.default_rng(0).normal(0, 0.5, 80), n_trials=300)["deflated_sharpe"]
    assert d["n_obs"] == 40 and d["n_trials"] == 300 and d["passed"] in (True, False)


def test_log_loss_is_read_against_the_market_on_the_same_rows():
    won = np.array([1, 0, 1, 0] * 20)
    day = np.repeat([f"d{i}" for i in range(20)], 4)
    r = tm.log_loss_read(day, won, np.where(won == 1, 0.7, 0.3), np.full(80, 0.5))
    assert r["market"] == pytest.approx(np.log(2)) and r["model"] == pytest.approx(-np.log(0.7))
    assert r["model_less_market"] < 0 and r["hi95"] < 0


def test_the_win_model_starts_from_the_market_and_forecasts_only_later_months():
    rng = np.random.default_rng(1)
    n = 9000
    p = rng.uniform(0.15, 0.7, n)
    d = pd.DataFrame({"role": "home", "p_cp": p, "x_cp": np.log(p / (1 - p)), "f": rng.normal(size=n),
                      "won": (rng.uniform(size=n) < p).astype(float),
                      "kickoff": pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 150, n), "D")})
    d["month"] = d["kickoff"].dt.strftime("%Y-%m")
    got = tm.win_walk_forward(d, ["f"])
    early = d["month"] <= "2026-03"
    assert np.isnan(got[early.to_numpy()]).all() and np.isfinite(got[~early.to_numpy()]).all()
    # a feature that is noise adds little to the market's chance
    assert np.abs(got[~early.to_numpy()] - p[~early.to_numpy()]).mean() < 0.03
