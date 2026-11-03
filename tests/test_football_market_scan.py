"""The market scan's settlement and costing (synthetic numbers only)."""

from __future__ import annotations

import numpy as np
import pytest

from research.open_close import football_market_scan as ms


def test_asian_handicap_settles_whole_half_and_quarter_lines():
    gd = np.array([1, 0, 0, 1, 1, 0, 2])
    line = np.array([-0.5, -0.5, 0.0, -1.0, -0.75, -0.25, -1.25])
    got = ms.ah_home_units(gd, line)
    assert got[0].tolist() == [1, 0, 0]          # won by one, gave 0.5: win
    assert got[1].tolist() == [0, 0, 1]          # draw, gave 0.5: lose
    assert got[2].tolist() == [0, 1, 0]          # draw, level: push
    assert got[3].tolist() == [0, 1, 0]          # won by one, gave 1: push
    assert got[4].tolist() == [0.5, 0.5, 0]      # won by one, gave 0.75: half win, half push
    assert got[5].tolist() == [0, 0.5, 0.5]      # draw, gave 0.25: half lose, half push
    assert got[6].tolist() == [1, 0, 0]          # won by two, gave 1.25: both halves win


def test_back_lay_and_green_returns():
    px, won = np.array([3.0, 3.0]), np.array([1.0, 0.0])
    assert ms.back_return(px, won, 0.02).tolist() == pytest.approx([2 * 0.98, -1])
    assert ms.lay_return(px, won, 0.02).tolist() == pytest.approx([-2, 0.98])
    half = ms.back_return(np.array([2.0]), np.array([0.5]), 0.0, np.array([0.5]))
    assert half.tolist() == pytest.approx([0.5])             # half win, half push
    assert ms.green(np.array([3.0]), np.array([2.5]), 1)[0] == pytest.approx(0.2)
    assert ms.green(np.array([3.0]), np.array([3.6]), -1)[0] == pytest.approx(1 / 6)


def test_symmetric_lay_sits_above_the_back_and_the_model_handicap_sums_to_one():
    back = np.array([2.0, 3.5])
    lay = ms.sym_lay(back, np.array([2.06, 3.6]))
    assert (lay > back).all() and lay[0] == pytest.approx(2.14)          # 2.06^2/2.0 = 2.1218 -> tick 2.14
    p = ms.ah_probability(np.array([1.6]), np.array([1.1]), np.array([-0.25]))
    assert p.sum() == pytest.approx(1.0) and p[0, 0] > p[0, 2]


def test_summary_and_bands():
    r = np.array([1.0, -1.0] * 40)
    day = np.repeat(np.arange(40), 2).astype(str)
    s = ms.summary(r, day, n_boot=200)
    assert s["bets"] == 80 and s["roi"] == 0.0 and s["lo90"] <= 0 <= s["hi90"]
    b = ms.by_band(r, np.array([1.2, 4.0] * 40), day)
    assert b["1.0-1.5"]["roi"] == 1.0 and b["3-5"]["roi"] == -1.0
