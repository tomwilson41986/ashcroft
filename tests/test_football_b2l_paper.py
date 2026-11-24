"""The back-to-lay paper test at the recorded quotes: which books, the fills, the hedge, the counts-only rule."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research.open_close import football_b2l_paper as bp

KO = pd.Timestamp("2026-11-28 15:00", tz="UTC")


def book(minutes_before, back=((3.0, 50.0),), lay=((3.05, 50.0),), inplay=0, status="OPEN", runner="ACTIVE"):
    r = {"polled_utc": (KO - pd.Timedelta(minutes=minutes_before)).isoformat(), "market_id": "1.5",
         "selection_id": 7, "status": status, "inplay": inplay, "runner_status": runner}
    for side, lv in (("back", back), ("lay", lay)):
        for i in range(1, 6):
            p, s = lv[i - 1] if i <= len(lv) else (np.nan, np.nan)
            r[f"{side}{i}"], r[f"{side}{i}_size"] = p, s
    return r


def frame(*rows):
    return bp.books_of(pd.DataFrame(list(rows)), "1.5", 7)


def test_entry_is_the_first_book_at_or_after_the_checkpoint_and_exit_the_last_before_the_turn():
    r = frame(book(370), book(358), book(348), book(1, lay=((2.5, 100.0),)), book(-1, inplay=1))
    e, x = bp.entry_and_exit(r, KO - pd.Timedelta(minutes=360), 360)
    assert e["t"] == KO - pd.Timedelta(minutes=358)                   # not the book before the checkpoint
    assert x["t"] == KO - pd.Timedelta(minutes=1) and x["lay1"] == 2.5
    late = frame(book(300), book(-1, inplay=1))                          # nothing within the tolerance after 360
    assert bp.entry_and_exit(late, KO - pd.Timedelta(minutes=360), 360)[0] is None
    never = frame(book(358), book(1))                                     # no turn in play recorded: no exit
    assert bp.entry_and_exit(never, KO - pd.Timedelta(minutes=360), 360)[1] is None


def test_a_full_hedge_locks_the_same_profit_whatever_the_result():
    e = pd.Series(book(358, back=((3.0, 50.0),)))
    x = pd.Series(book(1, lay=((2.5, 100.0),)))
    w, l = bp.trade(e, x, 10.0, True), bp.trade(e, x, 10.0, False)
    # back 10 at 3.0, lay 12 at 2.5: +2 either way, less 2% commission
    assert w["ret"] == pytest.approx(0.2 * 0.98) and l["ret"] == pytest.approx(0.2 * 0.98)
    assert w["hedged"] == pytest.approx(1.0)


def test_a_back_takes_the_levels_in_order_and_a_thin_lay_leaves_the_rest_held_to_the_result():
    e = pd.Series(book(358, back=((3.0, 4.0), (2.9, 100.0))))
    t = bp.trade(e, None, 10.0, False)
    assert t["back_price"] == pytest.approx((4 * 3.0 + 6 * 2.9) / 10) and t["ret"] == -1.0   # no exit: held, lost
    x = pd.Series(book(1, lay=((2.5, 6.0),)))
    p = bp.trade(pd.Series(book(358)), x, 10.0, False)                  # wants a lay of 12, gets 6
    assert p["hedged"] == pytest.approx(0.5) and p["ret"] == pytest.approx((-10 + 6) / 10)
    assert bp.trade(pd.Series(book(358, back=())), x, 10.0, True) == {"filled": 0.0}


def test_no_return_is_shown_before_the_reading_is_due():
    sig = pd.DataFrame({"cell": ["360 over 0.05"] * 5, "day": ["2026-11-28"] * 5, "ltp_cp": 3.0, "ltp_close": 2.8})
    tr = pd.DataFrame({"cell": "360 over 0.05", "event_id": range(5), "selection_id": 7, "stake": 10.0,
                       "status": "traded", "filled": 10.0, "hedged": 1.0, "ret": 0.05, "day": "2026-11-28"})
    got = bp.reading(tr, sig, date(2026, 12, 1), n_rules=471)["360 over 0.05"]
    assert got["gate"] == "running" and got["trades_at_10"] == 5 and "headline" not in got
    assert bp.reading(tr, sig, bp.READ_BY, n_rules=471)["360 over 0.05"]["gate"] == "not decided"
