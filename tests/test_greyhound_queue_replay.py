"""The early lay's ledger replayed with queue-aware fills (research/greyhound/queue_fill.py). Hand-built frames, for the
arithmetic only."""

from __future__ import annotations

import pandas as pd
import pytest

from greyhound import early_lay as el
from research.greyhound import queue_fill as rq

T0 = pd.Timestamp("2026-11-03 14:00:00", tz="UTC")
MID, SID = "1.1", 7
BACKS = [(4.0, 30.0), (3.9, 50.0), (3.8, 80.0)]


def ladder(rows, mid=MID, sid=SID):
    return pd.DataFrame([{"polled_utc": (T0 + pd.Timedelta(seconds=s)).isoformat(), "market_id": mid,
                          "selection_id": sid, "price": p, "traded": v} for s, p, v in rows])


def book_row(s, back=BACKS, lay=((4.2, 20.0), (4.3, 40.0), (4.4, 60.0)), **kw):
    r = {"market_id": MID, "selection_id": SID, "polled": T0 + pd.Timedelta(seconds=s), "last_traded": 4.1}
    for i, (p, z) in enumerate(back, 1):
        r[f"back{i}"], r[f"back{i}_size"] = p, z
    for i, (p, z) in enumerate(lay, 1):
        r[f"lay{i}"], r[f"lay{i}_size"] = p, z
    return {**r, **kw}


def ledger(price, seconds_to_off=300):
    return pd.DataFrame([{"time_utc": T0.strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds_to_off": seconds_to_off,
                          "market_id": MID, "selection_id": SID, "price": price, "race_time": "14:05",
                          "track": "Romford", "trap": 1, "f_bsp": 5.0, "mid": 4.1}])


def test_the_queue_ahead_on_the_backers_side():
    assert rq.lay_queue_ahead(4.1, BACKS) == (0.0, False)                    # improves on the best offer
    assert rq.lay_queue_ahead(3.9, BACKS) == (50.0, False)                   # a level on show
    assert rq.lay_queue_ahead(3.85, BACKS) == (0.0, False)                   # inside the shown range: nothing there
    assert rq.lay_queue_ahead(3.5, BACKS) == (80.0, True)                    # deeper: the largest size on show
    assert rq.lay_queue_ahead(3.5, BACKS, unknown="beyond_only") == (None, True)
    assert rq.lay_queue_ahead(3.5, BACKS[:2]) == (0.0, False)                # two levels: the book shows all it has
    assert rq.lay_queue_ahead(3.5, []) == (0.0, False)


def test_matched_at_once_where_the_best_lay_offer_is_at_or_under_our_price():
    book = {"back": BACKS, "lay": [(4.2, 3.0), (4.3, 50.0)]}
    f = rq.lay_fill(4.3, 10.0, T0, book, None)
    assert f["matched"] == 10.0 and f["taken"] == 10.0 and f["full_at"] == T0 and f["first_fill_at"] == T0
    assert f["avg_price"] == pytest.approx((3.0 * 4.2 + 7.0 * 4.3) / 10.0, abs=1e-4)   # each offer at its own price
    d = rq.replay_queue(ledger(4.3), pd.DataFrame([book_row(0, lay=((4.2, 50.0),)), book_row(290)]), None)
    r = d.iloc[0]
    s = el.stake_for(4.3, el.CONFIG)
    assert r.matched and r.matched_stake == s and r.matched_at == 4.2 and r.note == "taken at once"


def test_no_fill_while_the_money_traded_at_our_price_stays_inside_the_queue():
    book = {"back": BACKS, "lay": [(4.2, 20.0)]}
    tr = ladder([(-60, 3.9, 100.0), (10, 3.9, 120.0), (30, 3.9, 150.0)])    # GBP50 traded at 3.9: the GBP50 ahead
    f = rq.lay_fill(3.9, 10.0, T0, book, tr)
    assert f["queue_ahead"] == 50.0 and f["matched"] == 0.0 and f["avg_price"] is None and f["first_fill_at"] is None


def test_a_partial_fill():
    book = {"back": BACKS, "lay": [(4.2, 20.0)]}
    tr = ladder([(-60, 3.9, 100.0), (10, 3.9, 152.0), (20, 3.9, 156.0)])    # GBP56 traded: GBP6 past the queue
    f = rq.lay_fill(3.9, 10.0, T0, book, tr)
    assert f["matched"] == 6.0 and f["filled_share"] == 0.6 and f["avg_price"] == 3.9 and f["full_at"] is None
    assert f["first_fill_at"] == T0 + pd.Timedelta(seconds=10) and f["last_fill_at"] == T0 + pd.Timedelta(seconds=20)


def test_a_trade_under_our_price_fills_in_full():
    book = {"back": BACKS, "lay": [(4.2, 20.0)]}
    tr = ladder([(-60, 3.7, 5.0), (15, 3.7, 9.0)])                           # traded beyond 3.9 after we joined
    f = rq.lay_fill(3.9, 10.0, T0, book, tr)
    assert f["matched"] == 10.0 and f["full_at"] == T0 + pd.Timedelta(seconds=15)


def test_the_fill_stops_at_the_green_up():
    # off at T0 + 300 s: the green-up at 295 s; GBP60 passes the queue at 3.9 only at 296 s
    tr = ladder([(-60, 3.9, 100.0), (100, 3.9, 152.0), (296, 3.9, 160.0)])
    books = pd.DataFrame([book_row(0), book_row(290, back=((4.5, 30.0), (4.4, 10.0), (4.3, 10.0)))])
    r = rq.replay_queue(ledger(3.9), books, tr).iloc[0]
    assert r.matched_stake == 2.0 and r.last_fill_at == T0 + pd.Timedelta(seconds=100)
    f = rq.lay_fill(3.9, 10.0, T0, {"back": BACKS, "lay": []}, tr, until=T0 + pd.Timedelta(seconds=300))
    assert f["matched"] == 10.0                                              # had it run to the off


def test_the_green_up_and_sp_pnl():
    s = el.stake_for(3.9, el.CONFIG)
    tr = ladder([(-60, 3.9, 100.0), (100, 3.9, 200.0)])                      # clears the queue: the whole stake
    books = pd.DataFrame([book_row(0), book_row(290, back=((4.5, 30.0), (4.4, 10.0), (4.3, 10.0))),
                          book_row(298, back=((5.0, 30.0),))])               # past the green-up: not used
    r = rq.replay_queue(ledger(3.9), books, tr, bsp={(MID, float(SID)): 5.2}).iloc[0]
    assert r.matched_stake == s and r.greened_at == 4.5
    assert r.pnl == round(s * (1 - 3.9 / 4.5), 2)
    assert r.sp == 5.2 and r.sp_source == "bsp" and r.pnl_sp == round(s * (1 - 3.9 / 5.2), 2)
    r = rq.replay_queue(ledger(3.9), books.assign(sp_near=6.0), tr).iloc[0]
    assert r.sp == 6.0 and r.sp_source == "sp_near"                          # no BSP: the last projected SP


def test_money_traded_before_the_ladder_began_is_not_credited():
    # the order at T0; the recorder's first read of the market at T0 + 60 s already shows GBP200 traded at 3.9
    tr = ladder([(60, 3.9, 200.0), (90, 3.9, 230.0)])
    f = rq.lay_fill(3.9, 10.0, T0, {"back": BACKS, "lay": []}, tr, ladder_from=T0 + pd.Timedelta(seconds=60))
    assert f["matched"] == 0.0 and f["ladder_base"] == "first read after placement"   # GBP30 since: inside the queue


def test_the_two_replays_side_by_side():
    tr = ladder([(-60, 3.9, 100.0), (100, 3.9, 120.0)])
    books = pd.DataFrame([book_row(0), book_row(100, last_traded=3.9),
                          book_row(290, back=((4.5, 30.0), (4.4, 10.0), (4.3, 10.0)))])
    side, tot = rq.compare(ledger(3.9), books, tr)
    o, q = tot.iloc[0], tot.iloc[1]
    s = el.stake_for(3.9, el.CONFIG)
    # the optimistic replay: matched once a trade reached 3.9; the queue-aware: GBP20 traded, GBP50 ahead
    assert o.lays == q.lays == 1 and o.matched == 1 and o.matched_stake == s and o.pnl == round(s * (1 - 3.9 / 4.5), 2)
    assert q.matched == 0 and q.matched_stake == 0.0 and q.pnl == 0.0
    assert bool(side.opt_matched.iloc[0]) and not bool(side.q_matched.iloc[0]) and side.q_queue_ahead.iloc[0] == 50.0
