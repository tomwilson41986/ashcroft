"""The take-and-close trader (greyhound/take_close.py): the take down the ladder while each price clears the bar, a
fill-or-kill back closed at once by a lay of its winnings at the SP, the guards (switch, daily loss limit, STOP), no
second close from a lagging order list, and the settlement. A fake exchange; constructed books for the unit tests."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from greyhound import take_close as tc
from trading.exchange import Book, Fill, Quote

CFG = dict(tc.CONFIG, daily_loss_limit=500.0)


def test_plan_takes_down_the_ladder_while_the_bar_clears():
    lv = [(5.0, 30.0), (4.8, 40.0), (4.6, 100.0)]
    p = tc.plan(lv, 4.4, CFG)                  # 5.0/4.4 = +13.6%, 4.8/4.4 = +9.1%, 4.6/4.4 = +4.5% (under 8%)
    assert p["limit"] == 4.8 and p["stake"] == 70.0 and p["min_fill"] == 2.64        # rounded up: 2.64 x 3.8 >= GBP10
    assert tc.plan(lv, 4.7, CFG) is None       # 5.0/4.7 = +6.4%: nothing clears
    assert tc.plan([(25.0, 50.0)], 10.0, CFG) is None                     # over the study's 20
    assert tc.plan([(5.0, 500.0)], 4.0, CFG)["stake"] == 100.0             # capped at GBP100
    assert tc.plan([(1.5, 10.0)], 1.3, CFG) is None                        # GBP10 at 1.5 cannot be closed at SP


def test_close_pnl_and_positions():
    assert tc.close_pnl(100, 5.0, 4.0, True) == 0.0
    assert tc.close_pnl(100, 5.0, 4.0, False) == pytest.approx(-100 + 100 * 4 / 3)
    assert tc.close_pnl(100, 5.0, None, False) is None and tc.close_pnl(100, 5.0, 4.0, None) is None
    pos = tc.positions([
        {"selectionId": 7, "side": "BACK", "customerOrderRef": "tc-1118-1.2-7-b", "sizeMatched": 50,
         "averagePriceMatched": 5.0},
        {"selectionId": 7, "side": "LAY", "customerOrderRef": "tc-1118-1.2-7-s", "bspLiability": 150},
        {"selectionId": 8, "side": "LAY", "customerOrderRef": "el-1118-1.2-8-x", "sizeMatched": 9}])
    assert set(pos) == {7} and pos[7]["to_close"] == 50.0


@dataclass
class FakeX:
    book: Book
    orders_list: list
    sent: list

    def books(self, mids, with_sp=False):
        return {m: self.book for m in mids}

    def orders(self, mids, refs=None):
        return list(self.orders_list)

    def back(self, mid, sid, price, size, ref, min_fill=2.0):
        self.sent.append(("back", sid, price, size, min_fill, ref))
        q = self.book.runners[sid]
        got = sum(s for p, s in q.back if p >= price)
        m = min(size, got)
        avg = sum(p * min(s, m) for p, s in q.back if p >= price) / m if m else 0.0   # one level in these books
        return Fill(market_id=mid, selection_id=sid, side="BACK", order_type="LIMIT", price=price, size=size,
                    matched=m, avg_price=avg, bet_id="1", status="SUCCESS" if m >= min_fill else "FAILURE")

    def lay_at_bsp(self, mid, sid, liability, ref):
        self.sent.append(("sp_lay", sid, liability, ref))
        return Fill(market_id=mid, selection_id=sid, side="LAY", order_type="MARKET_ON_CLOSE", price=0.0,
                    size=liability, bet_id="2", status="PENDING")


class Store:
    def __init__(self, preds):
        self.d = {f"greyhound/live/predictions/2026-11-18.csv": preds.to_csv(index=False).encode()}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v


NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)      # 12:00 UK, the race off at 12:10


def setup(tmp_path, monkeypatch, f_bsp, live=True, limit=500.0, kill=False):
    preds = pd.DataFrame({"market_id": "1.5", "selection_id": [1, 2, 3], "p_model": [0.5, 0.3, 0.2],
                          "race_date": "2026-11-18", "race_time": "12:10", "track": "Romford", "trap": [1, 2, 3],
                          "dog_name": ["A", "B", "C"]})
    book = Book(market_id="1.5", status="OPEN", inplay=False, total_matched=500.0, runners={
        1: Quote(1, back=[(2.2, 300.0)], lay=[(2.24, 50.0)], last_traded=2.2, traded=100.0),
        2: Quote(2, back=[(4.0, 60.0)], lay=[(4.1, 20.0)], last_traded=4.0, traded=50.0),
        3: Quote(3, back=[(6.0, 80.0)], lay=[(6.4, 10.0)], last_traded=6.0, traded=20.0)})
    x = FakeX(book, [], [])
    monkeypatch.setattr(tc, "forecast", lambda d, fit: d.assign(f_bsp=d.selection_id.map(f_bsp), mid=d.back1))
    t = tc.TakeClose(Store(preds), x, dict(CFG, daily_loss_limit=limit), date(2026, 11, 18),
                     tmp_path / "ledger.csv", {}, live, kill=lambda: kill, now=lambda: NOW)
    t.until = NOW + timedelta(hours=9)
    return t, x


def led(t):
    return pd.read_csv(io.BytesIO(t.store.d["greyhound/take_close/%s/2026-11-18/ledger.csv" % t.mode]))


def test_takes_what_clears_and_closes_at_the_sp(tmp_path, monkeypatch):
    t, x = setup(tmp_path, monkeypatch, {1: 2.3, 2: 3.6, 3: 6.0})     # only dog 2 clears: 4.0/3.6 = +11%
    t.step()
    assert [s[0] for s in x.sent] == ["back", "sp_lay"]
    assert x.sent[0][1:4] == (2, 4.0, 60.0) and x.sent[0][5].startswith("tc-1118-1.5-2-b")
    assert x.sent[1][1:3] == (2, 180.0)                                # the winnings 60 x 3 laid at the SP
    t.step()                                                           # decided once; the lagging list: no 2nd close
    assert len(x.sent) == 2 and t.open_stake() == 60.0 and t.day_pnl() == -60.0
    # the race settles: dog 2 lost at an SP of 3.0: -60 + 60 x 3 / 2 = +30, less 2% commission
    x.book.status = "CLOSED"
    x.book.runners[2].status, x.book.runners[2].bsp = "LOSER", 3.0
    t.step()
    assert t.settled["1.5"] == pytest.approx(30 * 0.98) and t.open_stake() == 0.0
    assert {"decided", "back_matched", "sp_lay", "settled"} <= set(led(t).action)


def test_a_restart_closes_what_betfair_lists_unclosed(tmp_path, monkeypatch):
    t, x = setup(tmp_path, monkeypatch, {1: 2.3, 2: 4.0, 3: 6.0})
    t.decided.add("1.5")
    t.held["1.5"] = {2: {"stake": 50.0, "value": 200.0}}
    x.orders_list = [{"selectionId": 2, "side": "BACK", "customerOrderRef": "tc-1118-1.5-2-b", "sizeMatched": 50,
                      "averagePriceMatched": 4.0}]
    t.step()
    assert x.sent == [("sp_lay", 2, 150.0, "tc-1118-1.5-2-s")]


@pytest.mark.parametrize("kw, note", [(dict(kill=True), "stopped (S3 STOP)"), (dict(limit=None), "no daily loss limit"),
                                      (dict(limit=30.0), "the day's loss limit")])
def test_the_guards_send_nothing(tmp_path, monkeypatch, kw, note):
    t, x = setup(tmp_path, monkeypatch, {1: 2.3, 2: 3.6, 3: 6.0}, **kw)
    t.step()
    assert x.sent == [] and note in set(led(t).note.dropna())


def test_paper_sends_nothing_and_switches(tmp_path, monkeypatch):
    t, x = setup(tmp_path, monkeypatch, {1: 2.3, 2: 3.6, 3: 6.0}, live=False)
    t.step()
    assert x.sent == [] and "back_paper" in set(led(t).action)
    assert tc.switch_on({"GREYHOUND_TAKE_CLOSE": "live"}) and not tc.switch_on({"GREYHOUND_TAKE_CLOSE": "paper"})
    assert tc.daily_loss_limit({"GREYHOUND_TAKE_CLOSE_DAILY_LOSS": "250"}) == 250.0
    assert tc.daily_loss_limit({"GREYHOUND_TAKE_CLOSE_DAILY_LOSS": ""}) is None
    assert "greyhound/STOP" not in tc.STOP_KEYS and "trading/STOP" in tc.STOP_KEYS
