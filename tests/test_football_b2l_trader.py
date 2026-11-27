"""The back-to-lay strategy in the trader: one back a runner, the GBP 50 cap, STOP, the lay's size, paper sends nothing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from football import b2l_trader as bt

KO = datetime(2026, 11, 28, 15, 0, tzinfo=timezone.utc)


def meta(mid="1.5", event="Arsenal v Spurs"):
    return {"marketId": mid, "description": {"marketType": "MATCH_ODDS"}, "event": {"id": "99", "name": event,
                                                                                      "countryCode": "GB"},
            "runners": [{"selectionId": 1, "runnerName": "Arsenal", "sortPriority": 1},
                        {"selectionId": 2, "runnerName": "Spurs", "sortPriority": 2},
                        {"selectionId": 3, "runnerName": "The Draw", "sortPriority": 3}]}


def row(t, sid, ltp, back=None, lay=None, mid="1.5", matched=50000.0):
    r = {"polled_utc": t.isoformat(), "market_id": mid, "status": "OPEN", "inplay": 0, "selection_id": sid,
         "last_traded": ltp, "market_total_matched": matched}
    for side, (p, s) in (("back", back or (ltp, 100.0)), ("lay", lay or (round(ltp + 0.02, 2), 100.0))):
        r[f"{side}1"], r[f"{side}1_size"] = p, s
    return r


class Feed:
    def __init__(self):
        self.markets = {"1.5": {"start": KO, "meta": meta()}}
        self.last_rows = []


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def strategy(forecast, stop=lambda: False):
    clock = Clock(KO - timedelta(minutes=16 * 60))
    s = bt.B2L(ex=object(), store=None, mode="paper", stop=stop, model={"features": [], "roles": {}}, now=clock)
    s.load = lambda day: None
    import football.b2l as b2l
    b2l_predict = b2l.predict
    b2l.predict = lambda model, rows: np.array([forecast.get(r, -1.0) for r in rows["role"]])
    return s, clock, b2l, b2l_predict


def run_to(s, clock, feed, until_min, prices):
    """Poll every 10 minutes from 16 hours out to ``until_min`` before the off (then every minute), stepping."""
    t = clock.t
    while (KO - t).total_seconds() / 60 > until_min:
        feed.last_rows = [row(t, sid, p) for sid, p in prices.items()]
        clock.t = t
        s.step(feed)
        mins = (KO - t).total_seconds() / 60
        t += timedelta(minutes=10 if mins > 20 else 1)


def test_the_roles_follow_the_event_name_and_the_draw_by_name():
    assert bt.roles_of(meta()) == {1: "home", 3: "draw", 2: "away"}
    swapped = meta(event="Spurs v Arsenal")
    assert bt.roles_of(swapped) == {2: "home", 3: "draw", 1: "away"}


def test_one_back_a_runner_at_its_first_cell_and_a_lay_for_the_same_profit():
    s, clock, b2l, keep = strategy({"home": 0.06})
    try:
        feed = Feed()
        run_to(s, clock, feed, 0.5, {1: 2.5, 2: 3.2, 3: 3.4})
    finally:
        b2l.predict = keep
    backs = [r for r in s.log if r["kind"] == "back"]
    assert len(backs) == 1 and backs[0]["selection_id"] == 1 and backs[0]["cp_min"] == 360    # not again at 15
    assert backs[0]["status"] == "SUCCESS" and backs[0]["matched"] == 2.0
    lays = [r for r in s.log if r["kind"] == "lay"]
    assert len(lays) == 1 and lays[0]["size"] == pytest.approx(2.0 * 2.5 / 2.52, abs=0.01)


def test_the_day_cap_and_the_stop():
    s, clock, b2l, keep = strategy({"home": 0.06})
    s.spent["2026-11-28"] = 50.0
    try:
        run_to(s, clock, Feed(), 300, {1: 2.5, 2: 3.2, 3: 3.4})
    finally:
        b2l.predict = keep
    assert [r["outcome"] for r in s.log if r["kind"] == "back"] == ["day cap GBP 50 reached"]
    s2, clock2, b2l2, keep2 = strategy({"home": 0.06}, stop=lambda: True)
    try:
        run_to(s2, clock2, Feed(), 300, {1: 2.5, 2: 3.2, 3: 3.4})
    finally:
        b2l2.predict = keep2
    assert [r["outcome"] for r in s2.log if r["kind"] == "back"] == ["STOP"]


def test_paper_sends_nothing_to_an_exchange():
    class Live:
        def _place(self, *a, **k):
            raise AssertionError("paper must not place")
    # the paper strategy is given a plain object; only an exchange with _place is ever called
    s, clock, b2l, keep = strategy({"home": 0.06})
    try:
        run_to(s, clock, Feed(), 0.5, {1: 2.5, 2: 3.2, 3: 3.4})
    finally:
        b2l.predict = keep
    assert all(r.get("mode") == "paper" for r in s.log)
    m, avg, status = bt.lay_fok(object(), "1.5", 1, 2.52, 1.98, "b2lx", row(KO, 1, 2.5))
    assert status == "SUCCESS" and avg == pytest.approx(2.52)


def test_the_model_is_read_again_hourly_so_a_refit_after_the_start_is_picked_up(monkeypatch):
    """The trader starts at 05:40 UTC; the nightly refit can land after it. A model loaded at the start is read again
    within the hour, not only at the next UK day (26 Nov 2026)."""
    from football import b2l
    models = iter([{"v": 1}, {"v": 2}])
    calls = []
    monkeypatch.setattr(b2l, "load", lambda store: (calls.append(1), next(models))[1])
    t = [1000.0]
    monkeypatch.setattr(bt.time, "monotonic", lambda: t[0])

    class Store:
        def get(self, key):
            return None
    x = bt.B2L(None, Store(), "paper", lambda: False)
    x.load("2026-11-26")
    assert x.model == {"v": 1}
    t[0] += 60
    x.load("2026-11-26")                                   # within the hour: not read again
    assert len(calls) == 1
    t[0] += bt.RELOAD_SECONDS
    x.load("2026-11-26")
    assert x.model == {"v": 2} and len(calls) == 2


def test_a_restarted_run_keeps_the_days_backs_lays_the_open_one_and_counts_the_spend(monkeypatch):
    """A restart mid-day (26 Nov 2026) reads the day's trades: the open back is held for its lay, the laid one is not
    laid again, the spend counts toward the day cap, a paper back is not laid by a live run, the log is kept."""
    import json
    from football import b2l
    monkeypatch.setattr(b2l, "load", lambda store: {"m": 1})
    day = "2026-11-26"
    recs = [
        {"kind": "back", "market_id": "1.1", "selection_id": 7, "mode": "live", "price": 3.0, "status": "SUCCESS",
         "matched": 2.0, "avg_price": 3.0, "ref": "b2l-1.1-7-360"},
        {"kind": "back", "market_id": "1.2", "selection_id": 8, "mode": "live", "price": 2.0, "status": "SUCCESS",
         "matched": 2.0, "avg_price": 2.0, "ref": "b2l-1.2-8-360"},
        {"kind": "lay", "market_id": "1.2", "selection_id": 8, "mode": "live", "price": 1.9, "size": 2.1,
         "matched": 2.1, "status": "SUCCESS"},
        {"kind": "back", "market_id": "1.3", "selection_id": 9, "mode": "paper", "price": 4.0, "status": "SUCCESS",
         "matched": 2.0, "avg_price": 4.0, "ref": "b2l-1.3-9-15"},
        {"kind": "decide", "market_id": "1.1", "selection_id": 7, "mode": "live"},
    ]

    class Store:
        def __init__(self):
            self.put_body = None

        def get(self, key):
            return ("\n".join(json.dumps(r) for r in recs) + "\n").encode() if key.endswith(f"{day}.jsonl") else None

        def put(self, key, body):
            self.put_body = body
    st = Store()
    x = bt.B2L(None, st, "live", lambda: False)
    x.load(day)
    assert x.backed[("1.1", 7)]["lay"] is None                      # open: laid at its close
    assert isinstance(x.backed[("1.2", 8)]["lay"], dict)             # laid already
    assert ("1.3", 9) not in x.backed                                # paper, not this run's
    assert x.spent[day] == 4.0
    x.save(day)
    assert st.put_body.decode().count("\n") == len(recs)             # the log kept, not overwritten
    x.load(day)                                                      # the same day again: not restored twice
    assert x.spent[day] == 4.0 and len(x.log) == len(recs)


def test_no_back_in_a_thin_book():
    """The owner, 26 Nov 2026: no back where the market has under GBP 5,000 matched."""
    x = bt.B2L(None, None, "paper", lambda: False)
    t = KO - timedelta(minutes=15)
    for sid in (1, 2, 3):
        x.observe([row(t, sid, 3.0, mid="1.5", matched=4999.0)])
    x.back("1.5", 1, 15, "15 over 0.05", "2026-11-26")
    assert ("1.5", 1) not in x.backed and x.log[-1]["outcome"].startswith("thin book")
    x.observe([row(t, 1, 3.0, mid="1.5", matched=5000.0)])
    x.back("1.5", 1, 15, "15 over 0.05", "2026-11-26")
    assert ("1.5", 1) in x.backed
