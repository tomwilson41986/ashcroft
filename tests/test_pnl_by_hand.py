"""P&L reconciled by hand on 20 bets (the owner's model brief, 18 Nov 2026, Phase 1).

Twenty back positions in five races, some laid at the BSP (the trade-out), one non-runner in two of the races, are
settled by the trader's own paper settlement (trading/session.py Session.settle) at the live commission (2% of a
market's net winnings, trading/config_live.json). Every figure below is worked by hand in the comments:

    back:  won  -> matched x (average price - 1);  lost -> -matched;  non-runner -> 0
    lay at the BSP with liability L:  won -> -L;  lost -> L / (BSP - 1)
    commission: 2% of the market's net, when the net is positive

Nothing here trades: a paper exchange with no data, positions set directly.
"""
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from trading import config as tc
from trading.exchange import Book, Market, PaperExchange, Quote
from trading.session import Position, Session

DAY = date(2026, 11, 18)
NOW = datetime(2026, 11, 18, 18, 0, tzinfo=timezone.utc)

# (market, selection, result, matched, average price, lay liability, BSP, hand-worked P&L before commission)
BETS = [
    ("1.1", 1, "WINNER", 10.0, 6.0, 40.0, 5.0, 10.00),    # +50 back, -40 lay
    ("1.1", 2, "LOSER", 20.0, 4.0, 30.0, 4.0, -10.00),    # -20 back, +30/3 = +10 lay
    ("1.1", 3, "LOSER", 5.0, 10.0, 0.0, 9.0, -5.00),      # -5 back, no lay
    ("1.1", 4, "REMOVED", 8.0, 3.0, 0.0, None, 0.00),     # a non-runner: void
    ("1.2", 1, "LOSER", 10.0, 3.0, 15.0, 2.5, 0.00),      # -10 back, +15/1.5 = +10 lay
    ("1.2", 2, "WINNER", 4.0, 11.0, 30.0, 7.0, 10.00),    # +40 back, -30 lay
    ("1.2", 3, "LOSER", 6.0, 5.0, 20.0, 5.0, -1.00),      # -6 back, +20/4 = +5 lay
    ("1.2", 4, "LOSER", 2.0, 21.0, 0.0, 20.0, -2.00),     # -2 back
    ("1.3", 1, "WINNER", 50.0, 2.0, 0.0, 2.1, 50.00),     # +50 back
    ("1.3", 2, "LOSER", 25.0, 3.0, 40.0, 3.0, -5.00),     # -25 back, +40/2 = +20 lay
    ("1.3", 3, "LOSER", 10.0, 9.0, 64.0, 9.0, -2.00),     # -10 back, +64/8 = +8 lay
    ("1.3", 4, "LOSER", 3.0, 26.0, 0.0, 30.0, -3.00),     # -3 back
    ("1.4", 1, "LOSER", 15.0, 4.0, 45.0, 4.0, 0.00),      # -15 back, +45/3 = +15 lay
    ("1.4", 2, "LOSER", 12.0, 6.0, 50.0, 6.0, -2.00),     # -12 back, +50/5 = +10 lay
    ("1.4", 3, "WINNER", 7.0, 8.0, 49.0, 8.0, 0.00),      # +49 back, -49 lay
    ("1.4", 4, "REMOVED", 9.0, 5.0, 20.0, None, 0.00),    # a non-runner: back and lay void
    ("1.5", 1, "WINNER", 30.0, 3.5, 50.0, 3.0, 25.00),    # +75 back, -50 lay
    ("1.5", 2, "LOSER", 20.0, 4.5, 70.0, 5.0, -2.50),     # -20 back, +70/4 = +17.50 lay
    ("1.5", 3, "LOSER", 8.0, 13.0, 0.0, 12.0, -8.00),     # -8 back
    ("1.5", 4, "LOSER", 1.0, 51.0, 0.0, 60.0, -1.00),     # -1 back
]
# by market: net, commission at 2% when positive, settled
#   1.1: 10 - 10 - 5 + 0      = -5.00, commission 0.00 -> -5.00
#   1.2: 0 + 10 - 1 - 2       =  7.00, commission 0.14 ->  6.86
#   1.3: 50 - 5 - 2 - 3       = 40.00, commission 0.80 -> 39.20
#   1.4: 0 - 2 + 0 + 0        = -2.00, commission 0.00 -> -2.00
#   1.5: 25 - 2.5 - 8 - 1     = 13.50, commission 0.27 -> 13.23
COMMISSION = {"1.1": 0.00, "1.2": 0.14, "1.3": 0.80, "1.4": 0.00, "1.5": 0.27}
DAY_TOTAL = 52.29                                           # -5.00 + 6.86 + 39.20 - 2.00 + 13.23


@pytest.fixture
def settled():
    cfg = tc.load_config("trading/config_live.json", env={})
    assert cfg.commission == 0.02
    s = Session(PaperExchange(None), pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"]), cfg, DAY,
                clock=lambda: NOW, sleep=lambda _: None)
    for mid, sid, _, matched, price, liability, _, _ in BETS:
        s.positions[(mid, sid)] = Position(mid, sid, f"h{sid}", matched=matched, cost=matched * price,
                                           hedged=liability > 0, hedge_liability=liability)
    for mid in sorted({b[0] for b in BETS}):
        rows = [b for b in BETS if b[0] == mid]
        market = Market(mid, "York", "GB", NOW, runners={b[1]: f"h{b[1]}" for b in rows})
        book = Book(market_id=mid, status="CLOSED", inplay=False, total_matched=None,
                    runners={b[1]: Quote(b[1], status=b[2], bsp=b[6]) for b in rows})
        s.settle(market, book, NOW)
    return s


def test_twenty_bets_settle_to_the_hand_worked_figures(settled):
    rows = {(r["market_id"], r["selection_id"]): r for r in settled.ledger if r["event"] == "settle"}
    assert len(rows) == 20
    for mid, sid, _, _, _, _, _, pnl in BETS:
        assert rows[(mid, sid)]["pnl"] == pytest.approx(pnl, abs=0.005), (mid, sid)


def test_commission_is_two_percent_of_each_markets_positive_net(settled):
    got = {r["market_id"]: r["commission"] for r in settled.ledger if r["event"] == "commission"}
    assert got == pytest.approx(COMMISSION, abs=0.005)


def test_the_days_settled_figure_is_the_hand_worked_total(settled):
    assert settled.state.settled_pnl == pytest.approx(DAY_TOTAL, abs=0.005)
    assert all(p.settled for p in settled.positions.values())
