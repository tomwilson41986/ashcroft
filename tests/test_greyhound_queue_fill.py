"""Queue-aware fills (greyhound/queue_fill.py) and the live-key recorder's traded ladder (greyhound/book_record.py)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from greyhound import queue_fill as qf

T0 = pd.Timestamp("2026-11-03 14:00:00", tz="UTC")


def ladder(rows):
    return pd.DataFrame([{"polled_utc": (T0 + pd.Timedelta(seconds=s)).isoformat(), "price": p, "traded": v}
                         for s, p, v in rows])


def test_the_queue_ahead():
    backs = [(4.0, 30.0), (3.9, 50.0), (3.8, 80.0)]          # the layers' offers a backer takes, best (highest) first
    assert qf.queue_ahead("lay", 4.1, backs) == 0.0           # a lay at 4.1 improves on the best: first in line
    assert qf.queue_ahead("lay", 3.9, backs) == 50.0
    assert qf.queue_ahead("lay", 3.5, backs) is None          # deeper than the book shows


def test_a_lay_is_matched_only_past_the_queue():
    book = {"back": [(4.0, 30.0), (3.9, 50.0)], "lay": [(4.2, 20.0)]}
    tr = ladder([(-60, 3.9, 100.0), (10, 3.9, 130.0), (20, 3.9, 155.0), (30, 3.9, 160.0)])
    f = qf.fill("lay", 3.9, 10.0, T0, book, tr)
    # GBP50 ahead: GBP60 traded at 3.9 after we joined, so GBP10 of ours by the third read, in full
    assert f.queue_ahead == 50.0 and f.taken == 0.0 and f.matched == 10.0
    assert f.full_at == T0 + pd.Timedelta(seconds=30)
    f = qf.fill("lay", 3.9, 10.0, T0, book, tr, until=T0 + pd.Timedelta(seconds=20))
    assert f.matched == 5.0 and f.full_at is None             # GBP55 traded by then: GBP5 past the queue


def test_a_trade_beyond_our_price_fills_in_full():
    book = {"back": [(4.0, 30.0)], "lay": [(4.2, 20.0)]}
    tr = ladder([(-60, 3.5, 10.0), (15, 3.5, 12.0)])          # money traded at 3.5 after we laid at 3.6
    f = qf.fill("lay", 3.6, 10.0, T0, book, tr)
    assert f.queue_ahead is None and f.matched == 10.0 and f.full_at == T0 + pd.Timedelta(seconds=15)
    f = qf.fill("lay", 3.6, 10.0, T0, book, ladder([(-60, 3.5, 10.0), (15, 3.6, 40.0)]))
    assert f.matched == 0.0                                   # deeper than the book: only a trade beyond counts


def test_taken_at_once_where_the_book_offers_it():
    book = {"back": [(4.0, 30.0)], "lay": [(4.2, 3.0), (4.3, 50.0)]}
    f = qf.fill("lay", 4.3, 10.0, T0, book, ladder([]))
    assert f.taken == 10.0 and f.matched == 10.0 and f.full_at == T0
    f = qf.fill("lay", 4.2, 10.0, T0, book, ladder([]))
    assert f.taken == 3.0 and f.matched == 3.0 and f.full_at is None


def test_a_back_mirrors_a_lay():
    book = {"back": [(4.0, 30.0)], "lay": [(4.2, 20.0), (4.3, 40.0)]}
    tr = ladder([(-60, 4.3, 0.0), (10, 4.3, 45.0)])
    f = qf.fill("back", 4.3, 10.0, T0, book, tr)
    assert f.queue_ahead == 40.0 and f.matched == 5.0


def test_the_recorder_keeps_the_traded_ladder(tmp_path):
    from greyhound.book_record import TradedLadder

    class Rec:
        dir = tmp_path
        day = date(2026, 11, 3)
    lad = TradedLadder(Rec())
    raw = [{"marketId": "1.2", "runners": [{"selectionId": 7, "ex": {"tradedVolume": [{"price": 3.9, "size": 100.0},
                                                                                       {"price": 4.0, "size": 20.0}]}}]}]
    t = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)
    assert lad.write(raw, t) == 2
    assert lad.write(raw, t) == 0                            # unchanged: nothing written again
    raw[0]["runners"][0]["ex"]["tradedVolume"][0]["size"] = 130.0
    assert lad.write(raw, t) == 1
    d = pd.read_csv(tmp_path / "traded_greyhound_live.csv")
    assert d.traded.tolist() == [100.0, 20.0, 130.0] and set(d.columns) == set(TradedLadder.FIELDS)


def test_the_recorder_reads_in_batches_under_the_weight_limit():
    from greyhound import book_record as br

    class Data:
        def __init__(self):
            self.calls = []

        def _read(self, method, params):
            self.calls.append(len(params["marketIds"]))
            return [{"marketId": m} for m in params["marketIds"]]
    d = Data()
    out = br.read_books(d, [f"1.{i}" for i in range(10)])
    assert len(out) == 10 and max(d.calls) <= br.BATCH and br.BATCH * 25 <= 100   # 25 a market (5 + 17 + 3)


def test_the_recorder_asks_only_for_betfairs_price_data_types():
    from greyhound import book_record as br
    valid = {"SP_AVAILABLE", "SP_TRADED", "EX_BEST_OFFERS", "EX_ALL_OFFERS", "EX_TRADED"}   # Betfair's PriceData
    assert set(br.PRICE_DATA) <= valid and "EX_TRADED" in br.PRICE_DATA


def test_a_small_trade_beyond_does_not_fill_a_strict_order():
    book = {"back": [(4.0, 30.0)], "lay": [(4.2, 20.0)]}
    tr = ladder([(-60, 3.5, 10.0), (15, 3.5, 12.0), (25, 3.5, 30.0)])      # GBP2, then GBP20 traded beyond 3.6
    assert qf.fill("lay", 3.6, 10.0, T0, book, tr).full_at == T0 + pd.Timedelta(seconds=15)
    f = qf.fill("lay", 3.6, 10.0, T0, book, tr, through_min=10.0)
    assert f.matched == 10.0 and f.full_at == T0 + pd.Timedelta(seconds=25)  # only once our stake has traded there
    assert qf.fill("lay", 3.6, 10.0, T0, book, tr, until=T0 + pd.Timedelta(seconds=20), through_min=10.0).matched == 0


def test_the_money_of_the_read_an_order_was_priced_on_is_not_credited_to_it():
    """The recorder's books are stamped to the second, its traded ladder to the microsecond, from the same read: the
    ladder row of that read holds money traded before the order existed."""
    book = {"back": [(4.0, 30.0)], "lay": [(4.2, 20.0)]}
    same_read = (T0 + pd.Timedelta(milliseconds=400)).isoformat()
    tr = pd.DataFrame([{"polled_utc": (T0 - pd.Timedelta(seconds=5)).isoformat(), "price": 3.5, "traded": 10.0},
                       {"polled_utc": same_read, "price": 3.5, "traded": 40.0}])
    assert qf.fill("lay", 3.6, 10.0, T0, book, tr).matched == 0.0          # traded before we were there
