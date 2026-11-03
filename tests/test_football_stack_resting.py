"""The resting-order variant: where it posts, when the recorded book proves a fill, the deadline, the exit (synthetic
books)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research.open_close import football_stack_resting as rs
from research.open_close import football_stack_trial as st
from research.open_close.football_exchange import read_prereg

T0 = pd.Timestamp("2026-11-06 17:05", tz="UTC")
START = pd.Timestamp("2026-11-07 15:00", tz="UTC")


def book(rows, sid="1"):
    """Rows of (minutes after T0, back1, lay1[, inplay]) for one selection, as clean_books leaves them."""
    out = []
    for r in rows:
        m, b, l, *rest = r
        out.append({"market_id": "1.1", "selection_id": sid, "polled": T0 + pd.Timedelta(minutes=m),
                    "back1": b, "back1_size": 500.0, "lay1": l, "lay1_size": 500.0, "status": "OPEN",
                    "inplay": bool(rest[0]) if rest else False})
    return pd.DataFrame(out)


def test_it_posts_one_tick_inside_the_spread_or_joins_the_queue():
    assert rs.post_price(1, 2.00, 2.06) == 2.02        # back: one tick above the best back
    assert rs.post_price(1, 2.00, 2.02) == 2.02        # one-tick spread: joins the best lay
    assert rs.post_price(-1, 2.00, 2.06) == 2.04       # lay: one tick below the best lay
    assert rs.post_price(-1, 2.00, 2.02) == 2.00
    assert rs.post_price(1, 1.99, 2.04) == 2.00        # across the 2.0 band edge
    assert rs.post_price(-1, 3.0, 3.1) == 3.05
    assert rs.post_price(1, float("nan"), 2.0) is None and rs.post_price(-1, 2.0, float("nan")) is None


def test_a_fill_needs_the_book_to_reach_the_resting_price_before_the_deadline():
    deadline = START - pd.Timedelta(minutes=rs.CANCEL_BEFORE_MIN)
    b = book([(30, 2.00, 2.04), (60, 2.02, 2.06), (90, 2.04, 2.08)])
    assert rs.rest_fill(1, 2.02, b, T0, deadline) == T0 + pd.Timedelta(minutes=60)     # best back reached 2.02
    assert rs.rest_fill(1, 2.06, b, T0, deadline) is None                               # never that long
    assert rs.rest_fill(-1, 2.04, b, T0, deadline) == T0 + pd.Timedelta(minutes=30)    # best lay at 2.04
    late = (deadline - T0).total_seconds() / 60 + 2
    assert rs.rest_fill(1, 2.02, book([(30, 2.00, 2.04), (late, 2.10, 2.12)]), T0, deadline) is None
    assert rs.rest_fill(1, 2.00, book([(0, 2.00, 2.04)]), T0, deadline) is None          # not the posting snapshot


def test_a_filled_back_is_crossed_at_the_close_and_an_unfilled_one_kept_for_the_adverse_read():
    near = (START - T0).total_seconds() / 60 - 10
    b = book([(0, 2.00, 2.04), (60, 2.02, 2.06), (near, 1.90, 1.92)])
    early = b.iloc[0]
    r = rs.resting_trade(1, early, b, b, T0, START, "1")
    assert r["skip"] is None and r["fill_open"] == 2.02 and r["fill_close"] == 1.92
    assert r["profit"] == pytest.approx(2.02 / 1.92 - 1) and r["right_way"] is True
    b2 = book([(0, 2.00, 2.04), (60, 1.98, 2.02), (near, 1.90, 1.92)])
    r2 = rs.resting_trade(1, b2.iloc[0], b2, b2, T0, START, "1")
    assert r2["skip"] == "unfilled by the deadline" and r2["right_way"] is True        # shortened without us


def test_the_95_interval_is_wider_than_the_90():
    tr = pd.DataFrame({"day": [f"2026-11-{d:02d}" for d in range(1, 29)] * 3, "market_id": range(84),
                       "profit": np.random.default_rng(1).normal(0.01, 0.3, 84), "stake": 1.0})
    from research.open_close.backtest import summarise
    tr["right_way"], tr["side"] = True, 1
    lo95, hi95 = rs.interval95(tr, 0.02)
    s = summarise(tr, 0.02)
    assert lo95 < s["roi_lo90"] < s["roi_hi90"] < hi95


def test_the_preregistration_uses_the_stack_rule_frozen_models():
    rule, _ = read_prereg(rs.PREREG)
    stack, _ = read_prereg(st.PREREG)
    for k in ("model_sha256", "threshold", "leagues_fd", "book_width", "arm", "models_dir", "commission"):
        assert rule[k] == stack[k], k
    st.verify_models(rule["models_dir"], rule["model_sha256"])
    assert rule["cancel_before_start_min"] == rs.CANCEL_BEFORE_MIN
    assert all(rule["gate"][k] == v for k, v in rs.GATE.items())


def test_trade_resting_end_to_end_on_a_recorded_day(tmp_path):
    """A recorded MATCH_ODDS day: the home back fills at 2.02 when the book drifts there and is crossed at the close;
    the away lay never fills and is kept, with its close, for the adverse-selection read."""
    from types import SimpleNamespace

    from research.open_close import football_paper_trial as pt
    from tests.test_football_paper_trial import book as pbook
    from tests.test_football_paper_trial import trial_inputs
    gold, _panel, _store, _rd, _dc = trial_inputs(tmp_path)
    markets = [{"market_id": "1.9", "market_type": "MATCH_ODDS", "event_name": "Arsenal v Chelsea", "country": "GB",
                "selection_id": sid, "runner_name": name, "market_start_utc": "2025-11-08T15:00:00Z"}
               for sid, name in ((11, "Arsenal"), (12, "Chelsea"), (13, "The Draw"))]
    q = {"2025-11-07T17:05:00Z": {11: (2.00, 2.04), 12: (4.0, 4.1), 13: (3.4, 3.5)},
         "2025-11-08T10:00:00Z": {11: (2.02, 2.06), 12: (4.0, 4.1), 13: (3.4, 3.5)},
         "2025-11-08T14:55:00Z": {11: (1.90, 1.92), 12: (4.2, 4.3), 13: (3.5, 3.6)}}
    books = [pbook(at, sid, b, l, 50.0, market="1.9") for at, r in q.items() for sid, (b, l) in r.items()]
    books += [pbook("2025-11-08T15:01:00Z", sid, 2.0, 2.02, 50.0, inplay=True, market="1.9") for sid in (11, 12, 13)]
    files = {(date(2025, 11, 8), "markets_football_eve_GB.csv"): pd.DataFrame(markets),
             (date(2025, 11, 8), "books_football_eve_GB.csv"): pd.DataFrame(books)}
    coll, moment = pt.collection_of("2025-11-08")
    sig = pd.DataFrame({"event_id": "t0", "market_type": "MATCH_ODDS", "selection_id": ["H", "A"],
                        "yhat": [-0.06, 0.05], "side": [1, -1], "fd_code": "E0", "tier": "top five",
                        "match_day": "2025-11-08", "collection": coll.isoformat(), "moment_utc": moment})
    t = SimpleNamespace(gold=gold, read_day=lambda d, name: files.get((d, name)), tolerance_min=pt.TOLERANCE_MIN,
                        aliases=None)
    tr, sk = rs.trade_resting(t, sig, ["tight <1.02", "medium 1.02-1.05"])
    assert len(tr) == 1 and tr.iloc[0]["selection"] == "H"
    assert tr.iloc[0]["fill_open"] == 2.02 and tr.iloc[0]["fill_close"] == 1.92
    assert tr.iloc[0]["fill_utc"].startswith("2025-11-08T10:00")
    assert sk["reason"].tolist() == ["unfilled by the deadline"] and sk.iloc[0]["post_price"] == 4.0
    a = rs.adverse_selection(tr, sk)
    assert a["fill_rate"] == 0.5 and a["right_way_filled"] == 1.0 and a["right_way_unfilled"] == 1.0
