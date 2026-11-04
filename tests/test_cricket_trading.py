"""cricket/trading, cricket/ledger, cricket/reports/phase_gate: the paper trader's rules.

The books here are hand-built to drive each rule (a delay, a suspension, a price moving through an order); they
are not market data and nothing is scored on them."""

import datetime as dt

import pandas as pd
import pytest

from cricket.ledger.fees import Transactions, commission, expert_fee_bands
from cricket.market.book import Book, Runner
from cricket.reports.phase_gate import feed_lag, gate
from cricket.trading.paper_trader import PaperTrader
from cricket.trading.signals import Rules, Signal, back_value, lay_value, signals
from cricket.trading.sizing import Sizer, kelly_back, kelly_lay, position_after, worst_case

T0 = dt.datetime(2026, 11, 4, 14, 0, tzinfo=dt.timezone.utc)


def book(sec, back_a, lay_a, back_b=2.5, lay_b=2.6, status="OPEN", inplay=True, delay=5, size=100.0, winner=None):
    ra = Runner(back=[(back_a, size), (round(back_a - 0.01, 2), size)], lay=[(lay_a, size)],
                status="WINNER" if winner == 1 else ("LOSER" if winner == 2 else "ACTIVE"))
    rb = Runner(back=[(back_b, size)], lay=[(lay_b, size)],
                status="WINNER" if winner == 2 else ("LOSER" if winner == 1 else "ACTIVE"))
    return Book("1.1", T0 + dt.timedelta(seconds=sec), status, inplay, delay, {1: ra, 2: rb})


def test_values_after_commission():
    assert back_value(0.5, 2.2, 0.05) == pytest.approx(0.5 * 1.2 * 0.95 - 0.5)
    assert lay_value(0.4, 2.0, 0.05) == pytest.approx(0.6 * 0.95 - 0.4)


def test_signals_respect_the_quiet_window_band_and_threshold():
    b = book(0, 2.2, 2.24)
    r = Rules(threshold=0.03)
    s = signals({1: 0.55, 2: 0.45}, b, r, seconds_since_event=None)
    # 1 at 2.2: 0.55 x 1.2 x 0.95 - 0.45 = 0.177; 2 at 2.5: 0.45 x 1.5 x 0.95 - 0.55 = 0.091 (this book is generous)
    assert [(x.selection_id, x.side) for x in s] == [(1, "BACK"), (2, "BACK")]
    assert s[0].value == pytest.approx(0.177) and s[1].value == pytest.approx(0.09125)
    assert signals({1: 0.55}, b, r, seconds_since_event=3.0) == []    # inside 8 s of a wicket or boundary
    assert signals({1: 0.55}, b, Rules(p_band=(0.6, 1.0)), None) == []
    assert signals({1: 0.46}, b, r, None) == []                       # 0.46 x 1.2 x 0.95 - 0.54 < 0.03
    sus = book(0, 2.2, 2.24, status="SUSPENDED")
    assert signals({1: 0.55}, sus, r, None) == []


def test_kelly_and_the_match_cap():
    assert kelly_back(0.55, 2.2, 0.0) == pytest.approx((0.55 * 1.2 - 0.45) / 1.2)
    assert kelly_lay(0.3, 2.0, 0.0) == pytest.approx((0.7 * 1.0 - 0.3) / 1.0)
    sz = Sizer(bank=1000.0, match_cap=0.05, commission=0.0)
    sig = Signal("1.1", 1, "BACK", 2.2, 0.2, 0.55)
    stake = sz.stake(sig, {1: 0.0, 2: 0.0}, [1, 2])
    assert 0 < stake <= 50.0                                           # one third Kelly is 58.3, capped near 50
    assert worst_case(position_after({1: 0.0, 2: 0.0}, [1, 2], sig, stake)) <= 50.0
    sz.update_bank(1000.0)
    sz.bank = 790.0                                                    # 21% below the peak: stopped
    assert sz.stake(sig, {}, [1, 2]) == 0.0


def _trader():
    return PaperTrader(Rules(threshold=0.03, commission=0.05), Sizer(bank=1000.0, commission=0.05, match_cap=0.5))


def test_order_waits_for_the_delay_and_fills_from_the_book_then():
    tr = _trader()
    b0 = book(0, 2.2, 2.24)
    tr.on_book(b0)
    o = tr.submit(signals({1: 0.55, 2: 0.45}, b0, tr.rules, None)[0], b0, "m", T0)
    assert o.status == "pending" and o.due == T0 + dt.timedelta(seconds=5)
    tr.on_book(book(3, 2.2, 2.24))                       # inside the delay: nothing
    assert o.matched == 0
    tr.on_book(book(6, 2.24, 2.26, size=10.0))           # after it: 2.24 (better) then 2.23, 10 each
    assert o.matched == pytest.approx(20.0) and o.status == "resting"
    assert o.avg_price == pytest.approx((10 * 2.24 + 10 * 2.23) / 20)
    tr.on_book(book(8, 2.2, 2.22))                       # at the price, not through it: no fill
    assert o.matched == pytest.approx(20.0)
    tr.on_book(book(9, 2.3, 2.32))                       # traded through: the rest fills at its price
    assert o.status == "matched" and o.matched == pytest.approx(o.stake)
    assert tr.tx.max_hour() == 1


def test_a_suspension_during_the_delay_lapses_the_order():
    tr = _trader()
    b0 = book(0, 2.2, 2.24)
    tr.on_book(b0)
    o = tr.submit(signals({1: 0.55, 2: 0.45}, b0, tr.rules, None)[0], b0, "m", T0)
    tr.on_book(book(6, 2.2, 2.24, status="SUSPENDED"))
    assert o.status == "lapsed" and o.matched == 0
    # one order at a time per selection: a new signal is taken once the last one is done
    assert tr.submit(signals({1: 0.55, 2: 0.45}, b0, tr.rules, None)[0], b0, "m", T0) is not None


def test_settlement_commission_and_clv():
    tr = _trader()
    pre = book(-600, 2.0, 2.04, inplay=False)            # the last pre-play book: the close
    tr.on_book(pre)
    b0 = book(0, 2.2, 2.24)
    tr.on_book(b0)
    o = tr.submit(signals({1: 0.55, 2: 0.45}, b0, tr.rules, None)[0], b0, "m", T0)
    tr.on_book(book(6, 2.2, 2.24, size=500.0))
    tr.on_book(book(40, 2.1, 2.12))                       # 30 s after the delay: the mid has shortened
    net = tr.settle("1.1", 1)
    gross = o.matched * (2.2 - 1)
    assert o.pnl == pytest.approx(gross)
    assert net == pytest.approx(gross - commission(gross, 0.05))
    assert o.clv_30 > 0 and o.clv_close > 0
    assert tr.sizer.bank == pytest.approx(1000.0 + net)


def test_fees():
    tx = Transactions()
    for i in range(5003):
        tx.add(T0 + dt.timedelta(seconds=i % 3000 / 10))
    assert tx.max_hour() == 5003 and tx.charge() == pytest.approx(3 * 0.002)
    bands = expert_fee_bands([("2026-01", 20_000.0), ("2026-02", 0.0), ("2026-03", 6_000.0)])
    assert bands["weeks"] == 2 and bands["bands_reached"] == ["GBP 25,000 (20%)"]


def test_feed_lag_and_gate():
    books = [book(0, 2.2, 2.24), book(2, 2.3, 2.34), book(4, 2.3, 2.34), book(9, 2.5, 2.54)]
    receipts = [T0 + dt.timedelta(seconds=3), T0 + dt.timedelta(seconds=10)]
    lag = feed_lag(receipts, books)
    assert lag.lag_s.tolist() == [1.0, 1.0]               # moves at 2 s and 9 s, told at 3 s and 10 s
    res = gate(pd.DataFrame({"matched": [10.0], "clv_30": [0.02], "clv_close": [0.01], "pnl": [5.0]}), 10, lag,
               [5], 0.25)
    assert not res["pass"] and not res["checks"]["at least 500 paper bets"]
    assert res["checks"]["feed lag measured and under the bet delay"]
    assert res["pnl_after_commission"] == pytest.approx(4.75)


def test_paper_session_end_to_end_on_a_replayed_match():
    """A fixture match replayed on a test timeline (a delivery every 30 s) against books every 10 s that never move,
    with a fixed-probability stand-in model: orders are placed after the quiet window, matched after the delay,
    and settled on the result; the bank moves by the ledger's net."""
    import sys
    from dataclasses import replace
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cricket_live import T20, _bundle

    from cricket.ingest.ball_feed import CricsheetReplay
    from cricket.live.engine import Engine
    from cricket.live.paper import PaperSession, run_recorded

    mid, raw = T20[0]
    teams = raw["info"]["teams"]
    bundle = _bundle()
    bundle["bundle"]["calibration"] = {"chase": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                                       "first_innings": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
    events = [replace(e, receipt_utc=(T0 + dt.timedelta(seconds=30 * i)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
              for i, e in enumerate(CricsheetReplay(mid, raw).events())]
    n = len(events)
    books = [replace(book(10 * k, 1.5, 1.52, back_b=2.9, lay_b=3.0), market_id="1.9") for k in range(3 * n)]
    winner = 1 if raw["info"]["outcome"].get("winner") == teams[0] else 2
    books.append(replace(book(30 * n, 1.5, 1.52, status="CLOSED", winner=winner), market_id="1.9"))
    tr = PaperTrader(Rules(threshold=0.0, commission=0.05), Sizer(bank=1000.0, commission=0.05, match_cap=0.2))
    s = PaperSession(Engine(bundle, paths=20), tr, "1.9", {1: teams[0], 2: teams[1]})
    run_recorded(s, events, books)
    led = tr.ledger()
    assert len(led) > 0 and s.notes == []
    assert set(led.status) <= {"matched", "partial", "lapsed"}
    assert (pd.to_datetime(led.due) - pd.to_datetime(led.submitted)).dt.total_seconds().eq(5).all()
    assert tr.sizer.bank == pytest.approx(1000.0 + led.pnl.sum() - tr.commission_paid)
    assert len(s.quotes) == sum(e.kind == "delivery" for e in events)
