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
