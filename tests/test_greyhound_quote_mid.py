"""The live quotes around the mid (greyhound/quote_mid.py) on a fake exchange: the prices, the quotes and their
requotes, the green-up to a locked P&L, the SP close, paper sending nothing, and the stops."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from greyhound import quote_mid as qm
from sources.common import Store
from trading.exchange import Book, LiveExchange, Quote

DAY = date(2026, 11, 4)
OFF = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc)            # 14:00 UK (GMT)
MID = "1.234567890"


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class FakeX:
    _report = staticmethod(LiveExchange._report)
    _reason = staticmethod(LiveExchange._reason)

    def __init__(self, book):
        self.book, self.ords, self.placed, self.cancelled = book, [], [], []

    def books(self, mids):
        return {m: self.book for m in mids if m == MID}

    def orders(self, mids, refs=None):
        return [o for o in self.ords if o["marketId"] in mids]

    def funds(self):                                           # the gate reads the account's funds on live quotes
        return {"available": 10_000.0}

    def _place(self, mid, ins, ref):
        self.placed.append(ins)
        bid = str(len(self.placed))
        lim = ins.get("limitOrder") or {}
        o = {"betId": bid, "marketId": mid, "selectionId": ins["selectionId"], "side": ins["side"],
             "orderType": ins["orderType"], "customerOrderRef": ins["customerOrderRef"],
             "priceSize": {"price": lim.get("price"), "size": lim.get("size")}}
        if lim.get("timeInForce") == "FILL_OR_KILL":           # a green-up: matched in full at its price
            o.update(status="EXECUTION_COMPLETE", sizeMatched=lim["size"], sizeRemaining=0.0,
                     averagePriceMatched=lim["price"])
        elif ins["orderType"] == "MARKET_ON_CLOSE":
            o.update(status="EXECUTABLE", sizeMatched=0.0, sizeRemaining=0.0)
        else:
            o.update(status="EXECUTABLE", sizeMatched=0.0, sizeRemaining=lim["size"])
        self.ords.append(o)
        return {"status": "SUCCESS", "instructionReports": [{"status": "SUCCESS", "betId": bid,
                                                             "sizeMatched": o["sizeMatched"]}]}

    def _call(self, method, params):
        assert method == "cancelOrders" and params.get("instructions"), "never the whole market"
        for i in params["instructions"]:
            self.cancelled.append(i["betId"])
            for o in self.ords:
                if o["betId"] == i["betId"]:
                    o.update(status="EXECUTION_COMPLETE", sizeRemaining=0.0)
        return {"status": "SUCCESS"}

    def match(self, bet_id, size):
        o = next(o for o in self.ords if o["betId"] == bet_id)
        o.update(sizeMatched=size, averagePriceMatched=o["priceSize"]["price"],
                 sizeRemaining=max(o["priceSize"]["size"] - size, 0.0))


def book(prices, traded=True, status="OPEN"):
    return Book(market_id=MID, status=status, inplay=False, total_matched=500.0, runners={
        sid: Quote(selection_id=sid, back=[(b, 100.0), (round(b - 0.1, 2), 100.0)], lay=[(l, 100.0)],
                   last_traded=b if traded else None, traded=50.0)
        for sid, (b, l) in prices.items()})


def trader(tmp_path, x, live=True, clock=None, **cfg):
    store = Store(root=tmp_path)
    preds = pd.DataFrame({"market_id": [MID], "race_date": ["2026-11-04"], "race_time": ["14:00"],
                          "selection_id": [7]})
    store.put(f"greyhound/live/predictions/{DAY}.csv", preds.to_csv(index=False).encode())
    return qm.QuoteMid(store, x, dict(qm.CONFIG, **cfg), DAY, live, tmp_path / "ledger.csv",
                       now=clock or Clock(OFF - timedelta(minutes=4)))


def test_the_quotes_rest_either_side_of_the_mid():
    lay, back, m = qm.targets(4.0, 4.2, qm.CONFIG)
    assert lay < 4.0 < 4.2 < back and abs(m - 4.1) < 0.05
    assert lay == 3.85 and back == 4.4                                    # 5% either side, on the ladder
    assert qm.targets(19.5, 20.0, qm.CONFIG)[1] is None                   # over the top price: no back


def test_both_sides_quoted_on_every_dog_once_all_have_traded(tmp_path):
    x = FakeX(book({7: (4.0, 4.2), 8: (2.0, 2.04)}, traded=False))
    t = trader(tmp_path, x)
    t.step()
    assert x.placed == []                                                 # not every dog has traded yet
    x.book = book({7: (4.0, 4.2), 8: (2.0, 2.04)})
    t.step()
    got = sorted((i["selectionId"], i["side"], i["limitOrder"]["price"]) for i in x.placed)
    assert got == [(7, "BACK", 4.4), (7, "LAY", 3.85), (8, "BACK", 2.14), (8, "LAY", 1.91)]   # 2.02 x 1.05, x 0.95
    assert all(i["limitOrder"]["persistenceType"] == "LAPSE" and "timeInForce" not in i["limitOrder"]
               for i in x.placed)                                         # resting orders
    assert all(i["customerOrderRef"].startswith("qm-") and len(i["customerOrderRef"]) <= 32 for i in x.placed)
    n = len(x.placed)
    t.step()
    assert len(x.placed) == n                                             # not due again yet: 5 s apart near the off


def test_a_moved_mid_requotes_and_the_old_quote_comes_off(tmp_path):
    clock = Clock(OFF - timedelta(minutes=4))
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x, clock=clock)
    t.step()
    old_lay = next(o["betId"] for o in x.ords if o["side"] == "LAY")
    x.book = book({7: (4.4, 4.6)})
    clock.t += timedelta(seconds=5)
    t.step()
    assert old_lay in x.cancelled
    assert {i["limitOrder"]["price"] for i in x.placed[2:]} >= {4.2}       # the new lay at 4.5 x 0.95, down


def test_a_matched_lay_is_greened_up_to_a_locked_pnl(tmp_path):
    clock = Clock(OFF - timedelta(minutes=4))
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x, clock=clock)
    t.step()
    lay = next(o for o in x.ords if o["side"] == "LAY")
    x.match(lay["betId"], 10.0)                                          # GBP10 laid at 3.85
    x.book = book({7: (5.0, 5.1)})                                       # the dog drifted
    clock.t = OFF - timedelta(seconds=4)
    t.step()                                                              # quotes pulled, the hedge sent
    hedge = x.placed[-1]
    assert hedge["side"] == "BACK" and hedge["limitOrder"]["timeInForce"] == "FILL_OR_KILL"
    assert hedge["limitOrder"]["price"] == 5.0 and hedge["limitOrder"]["size"] == 7.7      # 10 x 3.85 / 5.0
    clock.t += timedelta(seconds=1)
    t.step()
    assert t.locked == pytest.approx(10 * (1 - 3.85 / 5.0), abs=0.02)     # the same whatever wins
    clock.t += timedelta(seconds=1)
    t.step()
    assert t.locked == pytest.approx(2.3, abs=0.02)                       # counted once
    assert MID in t.greened


def test_what_is_still_open_at_the_last_moment_goes_to_the_sp(tmp_path):
    clock = Clock(OFF - timedelta(minutes=4))
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x, clock=clock)
    t.step()
    lay = next(o for o in x.ords if o["side"] == "LAY")
    x.match(lay["betId"], 10.0)
    clock.t = OFF - timedelta(seconds=1)                                 # past the give-up point
    t.step()
    sp = x.placed[-1]
    assert sp["orderType"] == "MARKET_ON_CLOSE" and sp["side"] == "BACK"
    assert sp["marketOnCloseOrder"]["liability"] == 10.0


def test_paper_sends_nothing(tmp_path):
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x, live=False)
    t.step()
    assert x.placed == [] and t.orders.paper                              # decided, kept in memory, not sent
    assert (tmp_path / "ledger.csv").exists()


def test_the_stops_pull_the_quotes(tmp_path):
    clock = Clock(OFF - timedelta(minutes=4))
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x, clock=clock)
    t.step()
    t.locked = -100.0                                                      # the day's loss limit
    clock.t += timedelta(seconds=5)
    t.step()
    assert len(x.cancelled) == 2 and len(x.placed) == 2                    # pulled, nothing new
    t2 = trader(tmp_path / "b", FakeX(book({7: (4.0, 4.2)})), max_orders_hour=0)
    t2.step()
    assert t2.x.placed == []                                               # the hourly cap


def test_switch_and_table(tmp_path):
    assert qm.switch_on({"GREYHOUND_QUOTE_MID": "live"}) and not qm.switch_on({})
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x)
    t.step()
    t._write([], force=True)
    out = qm.table(Store(root=tmp_path), DAY)
    assert "Races quoted 1; quotes placed 2" in out


def test_the_live_staking_matches_the_simulators():
    from research.greyhound import profit_sim as ps
    q = Quote(selection_id=7, back=[(4.0, 40.0)], lay=[(4.2, 8.0)], last_traded=4.0)
    for kind, sim in (("liability", {"type": "liability", "size": 20.0, "min": 2.0, "max": 10.0}),
                      ("stake", {"type": "stake", "size": 20.0, "min": 2.0, "max": 10.0}),
                      ("liquidity", {"type": "liquidity", "size": 10.0, "frac": 0.25, "min": 2.0, "max": 10.0}),
                      ("late", {"type": "late", "size": 20.0, "x": 2.5, "late_secs": 120, "min": 2.0, "max": 10.0})):
        cfg = dict(qm.CONFIG, stake_type=kind)
        for side, price in (("LAY", 3.85), ("BACK", 4.4)):
            for secs in (300.0, 60.0):
                book = {"back": q.back, "lay": q.lay}
                assert qm.stake_for(price, cfg, side, q, secs) == \
                    ps.stake_q(price, {"stake": sim}, cfg, book, side.lower(), secs), (kind, side, secs)



# --- the audit's fixes, as the early lay's (the owner, 5 Nov 2026: "fix the quote_mid trader the same way") ---------

def test_live_needs_a_daily_loss_limit():
    assert qm.daily_loss_limit({}) is None
    assert qm.daily_loss_limit({"GREYHOUND_QUOTE_MID_DAILY_LOSS": "0"}) is None
    assert qm.daily_loss_limit({"GREYHOUND_QUOTE_MID_DAILY_LOSS": "40"}) == 40.0


def test_no_quote_on_a_race_after_the_trader_stops(tmp_path):
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x)
    t.until = OFF - timedelta(minutes=1)                       # the race goes off after the trader stops
    t.step()
    assert x.placed == []
    t.until = OFF + timedelta(minutes=30)
    t.step()
    assert len(x.placed) == 2


def test_resting_quotes_are_pulled_when_the_run_ends_or_is_killed(tmp_path):
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x)

    def sleep(s):                                              # the job is cancelled after the first poll
        raise SystemExit("signal 15")

    with pytest.raises(SystemExit):
        t.run(OFF + timedelta(minutes=30), sleep=sleep)
    assert len(x.placed) == 2 and sorted(x.cancelled) == ["1", "2"]
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert (led[led.action == "cancel"].error == "trader stopped").all() and (led.action == "cancel").any()


def test_the_stop_switch_pulls_every_resting_quote_once(tmp_path):
    clock = Clock(OFF - timedelta(minutes=4))
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x, clock=clock)
    t.step()
    state = {"stop": True}
    t.kill = lambda: state["stop"]
    clock.t += timedelta(seconds=5)
    t.step()
    clock.t += timedelta(seconds=5)
    t.step()
    assert sorted(x.cancelled) == ["1", "2"] and len(x.placed) == 2
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert (led.action == "cancel").sum() == 1


def test_the_gate_refuses_what_breaks_a_limit(tmp_path):
    x = FakeX(book({7: (4.0, 4.2)}))
    t = trader(tmp_path, x)
    assert t.gate("LAY", 4.0, 5.0) == ""
    assert "price" in t.gate("LAY", 25.0, 2.0)
    assert "stake" in t.gate("BACK", 4.0, 50.0)
    assert "worst outcome" in t.gate("LAY", 4.0, 5.0, win=-50.0)       # -50 less a GBP15 liability
    t.locked = -95.0
    assert "loss limit" in t.gate("BACK", 4.0, 6.0)
    t.locked = 0.0
    t._sent_min = [pd.Timestamp(OFF)] * t.cfg["max_orders_per_minute"]
    t.now = lambda: OFF
    assert "a minute" in t.gate("LAY", 4.0, 5.0)
    t._sent_min = []
    x.funds = lambda: None
    t._funds = None
    assert "funds" in t.gate("LAY", 4.0, 5.0)
    t.kill = lambda: True
    assert "STOP" in t.gate("LAY", 4.0, 5.0)


def test_a_refused_quote_is_logged_not_sent(tmp_path):
    x = FakeX(book({7: (4.0, 4.2)}))
    x.funds = lambda: None
    t = trader(tmp_path, x)
    t.step()
    assert x.placed == []
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert led.error.astype(str).str.contains("risk gate").all() and (led.action == "no_quote").sum() == 2


def test_quoted_markets_rebuilt_from_betfairs_orders(tmp_path):
    x = FakeX(book({7: (4.0, 4.2)}))
    x.ords = [{"betId": "9", "marketId": MID, "selectionId": 7, "side": "LAY", "status": "EXECUTION_COMPLETE",
               "sizeMatched": 5.0, "averagePriceMatched": 4.0, "sizeRemaining": 0.0,
               "customerOrderRef": f"qm-{DAY:%m%d}-x-7-q1", "priceSize": {"price": 4.0, "size": 5.0}},
              {"betId": "8", "marketId": MID, "selectionId": 7, "side": "BACK", "status": "EXECUTABLE",
               "customerOrderRef": "el-1104-x-7-l"}]
    t = trader(tmp_path, x)
    t.load_predictions()
    assert t.recover() == {"markets": 1, "new": 1} and MID in t.quoted
    assert trader(tmp_path / "b", x, live=False).recover() == {}


def test_reconcile_reads_the_quoters_ledger(tmp_path):
    from greyhound import early_lay as el
    led = pd.DataFrame([{"market_id": MID, "action": "quote", "locked_pnl": None},
                        {"market_id": MID, "action": "green", "locked_pnl": 2.0}])

    class S:
        def get(self, key):
            return led.to_csv(index=False).encode() if key == f"{qm.PREFIX}/live/{DAY}/ledger.csv" else None

        def put(self, key, data):
            self.key = key

    class X:
        def cleared(self, mids):
            return [{"marketId": MID, "customerOrderRef": f"qm-{DAY:%m%d}-x-7-q1", "profit": 2.5},
                    {"marketId": MID, "customerOrderRef": "el-1104-x-7-l", "profit": -9.0}]

    st = S()
    r = el.reconcile(st, X(), DAY, comm=0.02, prefix=qm.PREFIX, ref=qm.REF, laid_action="quote")
    assert r["betfair_profit_gross"] == 2.5 and r["difference"] == 0.5 and st.key.startswith(qm.PREFIX)
