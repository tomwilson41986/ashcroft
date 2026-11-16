from datetime import date

import pandas as pd
import pytest

from research.greyhound import green_audit as ga

D = date(2026, 11, 13)


def _bet(bid, side, ref, price, size, profit, outcome, asked=None):
    return {"marketId": "1.1", "selectionId": 7, "betId": bid, "side": side, "customerOrderRef": ref,
            "priceMatched": price, "priceRequested": asked or price, "sizeSettled": size, "profit": profit,
            "betOutcome": outcome, "orderType": "LIMIT"}


def test_an_over_covered_green_up_is_read_off_betfairs_bets():
    """A lay of 31.25 at 4.2 and two green-up backs worth more than it: the cover above 1, Betfair's profit against
    the flat green-up at the same back price, and other traders' bets left out."""
    cleared = [_bet("1", "LAY", "el-1113-00000001-7-l", 4.2, 31.25, 31.25, "WIN"),
               _bet("2", "BACK", "el-1113-00000001-7-g", 3.65, 20.0, -20.0, "LOSE", asked=3.6),
               _bet("3", "BACK", "el-1113-00000001-7-g", 3.6, 22.69, -22.69, "LOSE"),
               _bet("4", "BACK", "qm-1113-00000001-7-g", 3.6, 50.0, -50.0, "LOSE")]
    bets = ga.bets_frame(cleared, D)
    assert len(bets) == 3 and set(bets.kind) == {"lay", "green"}
    led = pd.DataFrame({"action": ["green"], "market_id": ["1.1"], "selection_id": [7], "locked_pnl": [-11.44]})
    s = ga.selections(bets, led).iloc[0]
    G = (20 * 3.65 + 22.69 * 3.6) / 42.69
    assert s.green_bets == 2 and s.won == 0
    assert s.cover == pytest.approx(42.69 * G / (31.25 * 4.2), abs=1e-4) and s.cover > 1
    assert s.betfair_profit == pytest.approx(31.25 - 42.69)
    assert s.flat_pnl == pytest.approx(31.25 * (1 - 4.2 / G), abs=0.01)
    assert s.cover_effect == pytest.approx(s.betfair_profit - s.flat_pnl, abs=0.01)
    assert s.green_better_than_asked == pytest.approx(0.05 * 20 / 42.69, abs=1e-4)
    assert s.ledger_locked == -11.44
    out = ga.summary(ga.selections(bets, led), bets, 0.02)
    assert out["over_covered_1pct"] == 1 and out["betfair_profit_gross"] == pytest.approx(-11.44)
    assert "Betfair's settled bets" in ga.report_md({"summary": out})


def test_no_bets_says_no_data_found():
    assert "no data found" in ga.summary(pd.DataFrame(), pd.DataFrame(), 0.02)["note"]
