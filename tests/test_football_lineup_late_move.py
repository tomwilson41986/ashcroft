"""Line-up late move: the price path and the trade's sign."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_lineup_late_move as lm


def test_trades_back_the_favoured_side_and_close():
    p = pd.DataFrame({"market_id": ["1"], "gpm_surprise": [0.3],
                      "kickoff": pd.to_datetime(["2026-09-01T15:00:00Z"], utc=True),
                      "ltp_t_60_h": [2.2], "ltp_t_60_a": [3.5], "ltp_close_h": [2.0], "ltp_close_a": [4.0]})
    t = lm.trades(p, "ltp_t_60", 0.1)
    assert t["ltp"]["bets"] == 1 and abs(t["ltp"]["roi"] - (2.2 / 2.0 - 1) * 0.98) < 1e-9
    assert t["ltp"]["right_way"] == 1.0


def test_boot_slope_recovers_a_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    y = 0.5 * x + rng.normal(scale=0.1, size=400)
    b, lo, hi = lm._boot_slope(x, y, None)
    assert lo < 0.5 < hi and abs(b - 0.5) < 0.05
