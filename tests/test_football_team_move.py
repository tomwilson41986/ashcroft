"""The team-feature move trade on Betfair's prices: the costing at the recorded quotes (synthetic rows only)."""

from __future__ import annotations

import pandas as pd
import pytest

from research.open_close import football_team_move as tm


def test_quoted_trades_fill_at_the_quotes():
    r = pd.DataFrame({"market_id": ["m1:MO", "m2:MO"], "day": "2025-01-04", "month": "2025-01", "sport": "football",
                      "country": "ENG", "market_type": "MATCH_ODDS",
                      "open_raw": [2.0, 3.0], "close_raw": [1.8, 3.5],
                      "open_price": [2.04, 3.06], "close_price": [1.83, 3.57], "y": [-0.1, 0.15],
                      "pred": [-0.05, 0.05]})                       # the first to shorten (back), the second to drift (lay)
    t = tm.quoted_trades(r, 0.03).sort_values("market_id").reset_index(drop=True)
    assert t.loc[0, "fill_open"] == 2.0 and t.loc[0, "fill_close"] == pytest.approx(1.81)
    assert t.loc[0, "profit"] == pytest.approx(2.0 / 1.81 - 1)
    assert t.loc[1, "fill_open"] == pytest.approx(3.05) and t.loc[1, "fill_close"] == 3.5
    assert t.loc[1, "profit"] == pytest.approx(1 - 3.05 / 3.5)
