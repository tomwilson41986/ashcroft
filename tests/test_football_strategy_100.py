"""The 100-strategy scan: the payoffs and the fixed catalogue."""

from __future__ import annotations

import numpy as np
import pytest

from research.open_close import football_strategy_100 as s100


def test_the_catalogue_is_fixed_at_72_new_cells_plus_the_25_of_the_12_nov_scan():
    names = s100.catalogue()
    assert len(names) == 72 and len(set(names)) == 72


def test_back_and_lay_held_to_the_result():
    assert s100.back_ret([3.0, 3.0], [True, False]).tolist() == pytest.approx([2 * 0.98, -1.0])
    # a lay at 3.0 risks 2.0 to win 1.0: per unit of liability +0.49 or -1
    assert s100.lay_ret([3.0, 3.0], [False, True]).tolist() == pytest.approx([0.98 / 2, -1.0])


def test_trade_outs_lock_the_same_profit_whatever_happens():
    # back at 3.0, lay at 2.5: 3.0 / 2.5 - 1 = 0.2 a unit staked, less commission
    assert s100.back_to_lay([3.0], [2.5])[0] == pytest.approx(0.2 * 0.98)
    assert s100.back_to_lay([2.5], [3.0])[0] == pytest.approx(2.5 / 3.0 - 1)
    # lay at 3.0 (liability 2), back at 3.5: (1 - 3/3.5) = 0.1429 per unit, over the liability
    assert s100.lay_to_back([3.0], [3.5])[0] == pytest.approx((1 - 3 / 3.5) * 0.98 / 2)


def test_a_worse_lay_price_is_one_tick_longer():
    assert s100.tick_longer(np.array([1.5, 2.0, 3.5, 10.0])).tolist() == pytest.approx([1.51, 2.02, 3.55, 10.5])
