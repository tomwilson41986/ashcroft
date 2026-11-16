"""The in-play leader trial's scorer: the order, its fill, its lapse, and the reading withheld until it is due."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from research.open_close import football_inplay_leader_trial as lt

T0 = datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)                    # the first book in play


def book(minute, lead=(1.10, 1.11, 500.0), status="OPEN", traded="", inplay=1, result=None):
    """Three rows at ``minute`` after the turn in play: the leader (selection 1) and two outsiders."""
    t = (T0 + timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ")
    quotes = {1: lead, 2: (12.0, 13.0, 50.0), 3: (30.0, 34.0, 20.0)}
    rows = []
    for sid, (b, lay, size) in quotes.items():
        rows.append({"polled_utc": t, "market_id": "1.1", "status": status, "inplay": inplay, "selection_id": sid,
                     "runner_status": (result or {}).get(sid, "ACTIVE"), "back1": b, "lay1": lay,
                     "lay1_size": size, "traded_delta": traded if sid == 1 else ""})
    return rows


def frame(*books):
    return pd.DataFrame([r for b in books for r in b])


RESULT = book(120, status="CLOSED", result={1: "WINNER", 2: "LOSER", 3: "LOSER"})


def test_an_order_rests_behind_the_queue_and_fills_when_enough_trades_at_its_price():
    got = lt.score_market(frame(book(0), book(49), book(51), book(52, traded="1.11:300|1.1:900"),
                                book(53, traded="1.11:211|1.12:5"), RESULT))
    assert len(got) == 1
    o = got[0]
    assert (o["window"], o["clock_min"], o["selection_id"], o["price"], o["queue_ahead"]) == \
        ("half_time", 51.0, 1, 1.11, 500.0)
    assert o["chance"] == pytest.approx(2 / (1.10 + 1.11), abs=1e-4)
    assert o["filled"] and o["won"]                           # 300 + 211 + 5 = 516 > 500 + 10; 1.10 does not count
    short = lt.score_market(frame(book(0), book(51), book(52, traded="1.11:300"), book(53, traded="1.11:210"),
                                  RESULT))
    assert not short[0]["filled"]                             # 510 is not more than the queue plus the stake


def test_it_lapses_after_five_minutes_or_at_a_suspension_whose_interval_still_counts():
    late = lt.score_market(frame(book(0), book(51), book(57, traded="1.11:900"), RESULT))
    assert not late[0]["filled"]                              # the money came six minutes on
    susp = lt.score_market(frame(book(0), book(51), book(52, status="SUSPENDED", traded="1.11:900"),
                                 book(53, traded="1.11:900"), RESULT))
    assert susp[0]["filled"]                                  # traded before the suspension: an adverse fill counts
    gone = lt.score_market(frame(book(0), book(51), book(52, status="SUSPENDED"), book(53, traded="1.11:900"),
                                 RESULT))
    assert not gone[0]["filled"]                              # cancelled at the suspension


def test_the_first_book_of_a_window_decides_and_each_window_has_its_own_order():
    none = lt.score_market(frame(book(0), book(51, lead=(1.40, 1.42, 100.0)), book(53), RESULT))
    assert none == []                                         # a chance of 0.71 at the first book: no order
    both = lt.score_market(frame(book(0), book(51), book(93, lead=(1.03, 1.04, 80.0)), RESULT))
    assert [o["window"] for o in both] == ["half_time", "late"] and both[1]["price"] == 1.04
    skipped = lt.score_market(frame(book(0), book(50, status="SUSPENDED"), book(52), RESULT))
    assert skipped[0]["clock_min"] == 52.0                    # a book that is not OPEN is not the first book
    assert lt.score_market(frame(book(0, inplay=0), book(51, inplay=0))) == []     # never in play


def test_no_return_is_shown_before_the_registered_reading():
    orders = pd.DataFrame({"day": ["2026-11-14"] * 40, "window": ["late"] * 40, "filled": [True] * 40,
                           "won": [True] * 40, "price": [1.1] * 40, "back_price": [1.09] * 40})
    early = lt.reading(orders, date(2026, 12, 1), n_trials=300)
    assert early["gate"] == "running" and early["filled"] == 40 and "headline" not in early
    assert lt.reading(orders, lt.READ_BY, n_trials=300)["gate"] == "not decided"
    days = [f"2026-11-{14 + k % 15:02d}" for k in range(320)]
    full = pd.DataFrame({"day": days, "window": ["late"] * 320, "filled": [True] * 320,
                         "won": [k % 10 != 0 for k in range(320)], "price": [1.1] * 320,
                         "back_price": [1.09] * 320})
    done = lt.reading(full, date(2026, 12, 20), n_trials=300)
    assert done["gate"] == "failed" and done["headline"]["bets"] == 320       # 90% winners at 1.10 lose
    assert done["headline"]["return"] == pytest.approx(0.9 * 0.1 * 0.98 - 0.1)
    assert lt.settle(1.5, True) == pytest.approx(0.49) and lt.settle(1.5, False) == -1.0
