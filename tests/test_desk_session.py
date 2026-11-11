"""The desk session (desk/session.py) on the strategy arms' made-up day (tests/test_strategy_arms.py's fixture):
the lay book as a desk strategy reproduces the lay arm's fills, runs beside the live rule on one book, and every
order goes through the kernel under the strategy's own reference. Fixtures only: the strategy is evaluated on the
recorded books (research/queries)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from desk import registry  # noqa: E402
from desk.algos.lay_book import LayBook  # noqa: E402
from desk.session import DeskSession  # noqa: E402
from research.arms import core, sessions  # noqa: E402
from tests.test_strategy_arms import B, MID, day  # noqa: E402,F401  (the fixture)


def _reg(**limits):
    s = {x.id: x for x in registry.load()[0]}["horse-lay-book"]
    if limits:
        s = registry._strategy({**s.raw, "limits": {**s.raw["limits"], **limits}})
    return s


def _cfg():
    core.set_plan_cfg(core.make_cfg(clv_draws=2000), reset=True)
    return core.make_cfg(clv_draws=2000)


def _desk(D, **limits):
    reg = _reg(**limits)
    policy = {"algos": [LayBook(**reg.params)], "registry": [reg], "envelope": None, "_D": D}
    led, s, x = core.run_session(D, DeskSession, _cfg(), policy, core.session_start(D, "win"))
    return s


def _fills(rows, key="matched"):
    return [(r["selection_id"], r[key], r["avg_price"]) for r in rows if r.get("event") in ("lay", "intent")
            and (r.get("matched") or 0) > 0]


def test_the_lay_book_reproduces_the_lay_arm(day):
    led, arm, _ = core.run_session(day, sessions.LaySession, _cfg(), {"lay_bar": 0.03, "max_lay_price": 20.0, "_D": day},
                                core.session_start(day, "win"))
    desk = _desk(day)
    assert _fills(arm.lay_rows) and _fills(desk.desk_rows) == _fills(arm.lay_rows)
    a = core.score_side(arm.lay_positions_frame(), day, "arm", side="lay")
    d = core.score_side(desk.positions_frame("horse-lay-book", "LAY"), day, "desk", side="lay")
    assert {k: d[k] for k in ("horses", "staked", "clv_x_stake", "result")} == \
        {k: a[k] for k in ("horses", "staked", "clv_x_stake", "result")}


def test_every_desk_order_carries_its_own_reference(day):
    desk = _desk(day)
    sent = [r for r in desk.desk_rows if r["event"] == "intent" and r["status"] != "REFUSED"]
    assert sent and all(r["ref"].startswith("dlb1102-") for r in sent)
    assert not any(r["ref"].startswith("ash") for r in sent)


def test_the_kernel_cuts_the_desk_order(day):
    desk = _desk(day, max_order=50.0)
    fills = _fills(desk.desk_rows)
    assert fills and fills[0][1] == 50.0                              # the arm's GBP200 cut to the order cap
    assert any(r.get("reason") == "the order's cap" for r in desk.desk_rows if r["event"] == "intent")


def test_the_kernel_refuses_a_stale_book(day):
    desk = _desk(day, max_book_age_s=-1)                               # every book is too old
    rows = [r for r in desk.desk_rows if r["event"] == "intent"]
    assert rows and all(r["status"] == "REFUSED" and "stale" in r["reason"] for r in rows)


def test_the_desk_runs_beside_the_live_rule_on_one_book(day):
    cfg = _cfg()
    reg = _reg()
    specs = [("live", sessions.ArmSession, cfg, {"_D": day}, core.BALANCE, True),
             ("desk", DeskSession, _cfg(), {"algos": [LayBook(**reg.params)], "registry": [reg], "envelope": None,
                                            "_D": day}, core.BALANCE, False)]
    out = core.run_agents(day, specs, core.session_start(day, "win"))
    led, live, _ = out["live"]
    _, desk, _ = out["desk"]
    backs = led[(led.event == "back") & (led.matched > 0)]
    assert len(backs) and B not in set(backs.selection_id)             # the live rule backs the long horses
    lays = _fills(desk.desk_rows)
    assert lays and all(sid == B for sid, _, _ in lays)                # the desk lays the short one, on the same book


def test_two_lays_in_one_minute_share_the_days_room_as_the_arm_sizes_them():
    """The cause of desk-lay-book-1111's failed machinery check: the port sized every intent at the start of the
    minute, so two orders shared the same day-liability headroom; the arm sizes each order as it sends it."""
    from datetime import datetime, timezone

    import numpy as np

    from desk.strategy import MarketView
    from trading.exchange import Book, Market, Quote
    from trading.strategy import RunnerView

    now = datetime(2026, 11, 2, 7, 0, tzinfo=timezone.utc)
    m = Market("1.1", "Ascot", "GB", datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc), runners={1: "A", 2: "B"})
    book = Book("1.1", "OPEN", False, 5000.0, {1: Quote(1, back=[(2.9, 500.0)], lay=[(3.0, 500.0)]),
                                               2: Quote(2, back=[(3.9, 500.0)], lay=[(4.0, 500.0)])})
    views = [RunnerView(selection_id=s, name=n, model_price=6.0, back=b, back_size=500.0, lay=L, last_traded=L,
                        status="ACTIVE", traded=0.0) for s, n, b, L in ((1, "A", 2.9, 3.0), (2, "B", 3.9, 4.0))]
    draws = np.full((100, 2), 1 / 5.0)                     # the closing model's E[BSP] = 5 for both
    view = MarketView(market=m, book=book, runners=views, now=now, closing_fn=lambda v: (None, None, None, None, draws))
    lb = LayBook(day_liability=300.0)
    first, second = sorted(lb.intents(view, []), key=lambda i: -i.expected_value)
    assert (first.selection_id, first.size) == (1, 150.0)    # GBP300 of room at 3.0: GBP150 of stake
    assert second.size == 100.0                              # sized alone, B would take the same GBP300
    lb.on_fill(first, first.size, first.price)               # A fills: the day's GBP300 is used
    assert lb.revise(second) is None                         # as sent, B has no room left: the arm sends nothing
