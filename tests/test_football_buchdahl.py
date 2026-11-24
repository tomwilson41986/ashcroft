"""Buchdahl's tests: the five ways of removing the margin, and move retention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.open_close import football_buchdahl as bd

ODDS = np.array([[1.50, 4.20, 7.00], [2.60, 3.30, 2.90], [1.20, 7.50, 15.0]])


@pytest.mark.parametrize("m", bd.METHODS)
def test_every_method_gives_chances_that_sum_to_one_in_the_price_order(m):
    p = bd.devig(ODDS, m)
    assert np.allclose(p.sum(1), 1.0)
    assert (np.argsort(p, 1) == np.argsort(1 / ODDS, 1)).all()


@pytest.mark.parametrize("m", bd.METHODS)
def test_a_book_without_a_margin_is_left_as_it_is(m):
    fair = 1 / np.array([[0.5, 0.3, 0.2]])
    assert bd.devig(fair, m) == pytest.approx(np.array([[0.5, 0.3, 0.2]]), abs=1e-6)


def test_shin_and_power_take_more_of_the_margin_from_the_outsider_than_proportional():
    prop, shin, power = (bd.devig(ODDS, m)[2, 2] for m in ("proportional", "shin", "power"))
    assert shin < prop and power < prop


def test_retention_of_a_move_that_continues_by_half():
    x = np.linspace(-1, 1, 200)
    r = bd.retention_fit(x, 0.5 * x, np.repeat(np.arange(20), 10))
    assert r["retention"] == pytest.approx(1.5) and r["lo95"] == pytest.approx(1.5)


def test_margins_reads_the_closing_columns_and_reports_each_method():
    rng = np.random.default_rng(0)
    n = 1500
    p = rng.dirichlet([4, 2.5, 3], size=n)
    o = 1 / (p * 1.05)
    res = rng.random(n)
    ftr = np.where(res < p[:, 0], "H", np.where(res < p[:, 0] + p[:, 1], "D", "A"))
    s = pd.DataFrame({"PSCH": o[:, 0], "PSCD": o[:, 1], "PSCA": o[:, 2], "FTR": ftr,
                      "Date": pd.date_range("2025-01-01", periods=n, freq="6h").strftime("%d/%m/%Y")})
    got = bd.margins(s)
    assert got["pinnacle_close"]["matches"] == n and set(got["pinnacle_close"]["methods"]) == set(bd.METHODS)
    assert "status" in got["average_close"]
