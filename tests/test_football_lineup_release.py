"""Line-up release: the price series from recorded books."""

from __future__ import annotations

import pandas as pd

from research.open_close import football_lineup_release as lr


def test_price_series_normalised_log_ratio():
    m = pd.DataFrame({"market_id": ["1"] * 3, "selection_id": [1, 2, 3], "event_name": ["A v B"] * 3,
                      "runner_name": ["A", "B", "The Draw"], "market_type": "MATCH_ODDS"})
    b = pd.DataFrame({"market_id": ["1"] * 3, "selection_id": [1, 2, 3], "polled_utc": ["2026-11-07T14:00:00Z"] * 3,
                      "inplay": [0] * 3, "back1": [2.0, 4.0, 4.0], "lay1": [2.0, 4.0, 4.0]})
    s = lr.price_series(m, b)
    assert len(s) == 1 and abs(s["lr"].iloc[0] - __import__("math").log(2.0)) < 1e-9
