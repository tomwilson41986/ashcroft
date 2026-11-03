"""The early lay traded live (greyhound/early_lay.py): the lays once every runner has traded, sized by liability, the
green-up at the best back price, the SP close near the off, nothing sent on paper, the stop switch, and a restart."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from greyhound import early_lay as el
from research.greyhound import tradeout_study as ts
from trading.exchange import Book, LiveExchange, Quote

T0 = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)
MID = "1.250000001"


class FakeX(LiveExchange):
    """Records every instruction; a lay matches ``lay_fill`` of its size at its price, a back all of it."""

    def __init__(self, lay_fill=1.0):
        self.sent, self.cancels, self.current, self.book = [], [], [], None
        self.lay_fill, self._n = lay_fill, 0

    def books(self, mids, with_sp=False):
        return {MID: self.book} if self.book else {}

    def _place(self, market_id, ins, ref):
        self.sent.append(ins)
        lim = ins.get("limitOrder") or {}
        size = lim.get("size") or (ins.get("marketOnCloseOrder") or {}).get("liability")
        m = round(size * (self.lay_fill if ins["side"] == "LAY" else 1.0), 2) if ins["orderType"] == "LIMIT" else 0.0
        self.current.append({"selectionId": ins["selectionId"], "side": ins["side"], "sizeMatched": m,
                             "averagePriceMatched": lim.get("price", 0.0), "customerOrderRef": ins["customerOrderRef"],
                             "betId": str(len(self.sent))})
        return {"status": "SUCCESS", "instructionReports": [{"status": "SUCCESS", "betId": str(len(self.sent)),
                                                             "sizeMatched": m,
                                                             "averagePriceMatched": lim.get("price", 0.0)}]}

    def _call(self, method, params):
        if method == "cancelOrders":
            self.cancels.append(params["marketId"])
            return {"status": "SUCCESS"}
        raise AssertionError(method)

    def orders(self, market_ids, refs=None):
        return list(self.current)


def book(traded=True, back=(4.0, 6.0, 9.0), lay=(4.2, 6.4, 9.6), status="OPEN"):
    runners = {}
    for i, (b, l) in enumerate(zip(back, lay)):
        runners[i + 1] = Quote(selection_id=i + 1, back=[(b, 50.0)], lay=[(l, 50.0)],
                               last_traded=(b + l) / 2 if traded else None, traded=100.0)
    return Book(market_id=MID, status=status, inplay=False, total_matched=600.0, runners=runners)


def preds(off=T0 + timedelta(minutes=10)):
    off_uk = pd.Timestamp(off).tz_convert("Europe/London")
    return pd.DataFrame({"market_id": [MID] * 3, "selection_id": [1, 2, 3], "p_model": [0.30, 0.15, 0.10],
                         "race_date": [off_uk.strftime("%Y-%m-%d")] * 3, "race_time": [off_uk.strftime("%H:%M")] * 3,
                         "track": ["Romford"] * 3, "trap": [1, 2, 3], "dog_name": ["A", "B", "C"]})


class Store:
    def __init__(self, p):
        self.p, self.put_keys = p, []

    def get(self, key):
        return self.p.to_csv(index=False).encode() if key.endswith(".csv") else None

    def put(self, key, data):
        self.put_keys.append(key)


@pytest.fixture
def fit():
    """A forecast that leans on our price: the outsiders our model rates low are forecast to drift."""
    cols = ts.CL_SETS["micro"]
    beta = np.zeros(len(cols))
    beta[cols.index("lq_market")] = 0.3
    beta[cols.index("lq_model")] = 1.5
    return {"kind": "cl", "cols": cols, "beta": beta, "mu": np.zeros(len(cols)), "sd": np.ones(len(cols)),
            "book": 1.0}


def trader(tmp_path, x, fit, live=True, now=T0, kill=False, p=None):
    clock = {"t": now}
    t = el.EarlyLay(Store(p if p is not None else preds()), x, dict(el.CONFIG), date(2026, 11, 3), live,
                    tmp_path / "ledger.csv", fit, kill=lambda: kill, now=lambda: clock["t"])
    return t, clock


def test_the_stake_is_sized_by_liability():
    cfg = dict(el.CONFIG)
    from greyhound import tradeout as to
    for price in (1.5, 2.0, 3.3, 5.0, 11.0, 20.0, 60.0):          # the paper's stake, the forward test's size
        assert np.isclose(el.stake_for(price, cfg), el._floor2(float(to.target_stake(price))))
    assert el.stake_for(5.0, cfg) == 25.0 and el.stake_for(1.5, cfg) == 50.0


def test_a_race_is_laid_once_every_runner_has_traded(tmp_path, fit):
    x = FakeX()
    x.book = book(traded=False)
    t, clock = trader(tmp_path, x, fit)
    t.step()
    assert not x.sent and MID not in t.priced                  # nobody has traded yet
    x.book = book(traded=True)
    clock["t"] += timedelta(seconds=15)
    t.step()
    lays = [i for i in x.sent if i["side"] == "LAY"]
    assert lays and MID in t.priced
    for i in lays:
        lim = i["limitOrder"]
        assert lim["persistenceType"] == "LAPSE" and "timeInForce" not in lim and i["orderType"] == "LIMIT"
        assert lim["price"] <= el.CONFIG["max_price"] and lim["size"] == el.stake_for(lim["price"], el.CONFIG)
        assert i["customerOrderRef"].startswith("el-")
    n = len(x.sent)
    clock["t"] += timedelta(seconds=15)
    t.step()
    assert len(x.sent) == n                                    # one decision a race
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert set(led.action) <= {"lay", "no_lay"} and (led.action == "lay").sum() == len(lays)


def test_the_drifters_only_at_80_percent_of_the_forecast(tmp_path, fit):
    from greyhound import tradeout as to
    from research.greyhound import limit_study as ls
    x = FakeX()
    x.book = book()
    t, _ = trader(tmp_path, x, fit)
    t.step()
    led = pd.read_csv(tmp_path / "ledger.csv")
    for r in led.itertuples():
        if r.action == "lay":
            assert r.f_bsp >= r.mid and r.price == ls.ladder([r.f_bsp * 0.8], up=False)[0]
        else:
            assert r.f_bsp < r.mid or ls.ladder([r.f_bsp * 0.8], up=False)[0] > el.CONFIG["max_price"]


def test_green_up_at_the_best_back_price(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    lays = {i["selectionId"]: i["limitOrder"] for i in x.sent if i["side"] == "LAY"}
    assert lays
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=4)   # inside the green-up
    t.step(near_only=True)
    assert x.cancels == [MID]
    backs = {i["selectionId"]: i["limitOrder"] for i in x.sent if i["side"] == "BACK"}
    assert set(backs) == set(lays)
    for sid, b in backs.items():
        L = lays[sid]
        B = x.book.runners[sid].best_back[0]
        assert b["price"] == B and b["timeInForce"] == "FILL_OR_KILL"
        assert b["size"] == el._floor2(L["size"] * L["price"] / B)
    led = pd.read_csv(tmp_path / "ledger.csv")
    g = led[led.action == "green"]
    assert len(g) == len(lays) and MID in t.greened
    exp = sum(round(lays[s]["size"] * (1 - lays[s]["price"] / x.book.runners[s].best_back[0]), 2) for s in lays)
    assert np.isclose(g.locked_pnl.sum(), exp) and np.isclose(t.locked, exp)


def test_near_the_off_the_lay_is_closed_at_the_sp(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=1)    # past the give-up
    t.step(near_only=True)
    sp = [i for i in x.sent if i["orderType"] == "MARKET_ON_CLOSE"]
    lays = [i for i in x.sent if i["side"] == "LAY"]
    assert sp and all(i["side"] == "BACK" for i in sp)
    assert {i["selectionId"] for i in sp} == {i["selectionId"] for i in lays if i["limitOrder"]["size"] >= 2.0}


def test_paper_sends_nothing(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit, live=False)
    t.step()
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=4)
    t.step(near_only=True)
    assert not x.sent and not x.cancels
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert (led.action == "lay").any() and set(led.status.dropna()) <= {"PAPER"}


def test_the_stop_switch_lays_nothing_new(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, _ = trader(tmp_path, x, fit, kill=True)
    t.step()
    assert not x.sent
    assert pd.read_csv(tmp_path / "ledger.csv").action.tolist() == ["skip"]


def test_a_restart_does_not_lay_again(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, _ = trader(tmp_path, x, fit)
    t.step()
    n = len(x.sent)
    t2, _ = trader(tmp_path, x, fit, now=T0 + timedelta(seconds=30))
    t2.step()
    assert len(x.sent) == n and MID in t2.priced


def test_positions_from_betfairs_orders():
    p = el.positions([
        {"selectionId": 1, "side": "LAY", "sizeMatched": 2.0, "averagePriceMatched": 5.0, "customerOrderRef": "el-a-l"},
        {"selectionId": 1, "side": "LAY", "sizeMatched": 2.0, "averagePriceMatched": 6.0, "customerOrderRef": "el-a-l"},
        {"selectionId": 2, "side": "LAY", "sizeMatched": 3.0, "averagePriceMatched": 4.0, "customerOrderRef": "el-b-l"},
        {"selectionId": 2, "side": "BACK", "sizeMatched": 2.5, "averagePriceMatched": 4.8, "customerOrderRef": "el-b-g"}])
    assert p[1]["lay_matched"] == 4.0 and p[1]["lay_price"] == 5.5 and not p[1]["closed"]
    assert p[2]["closed"]


def test_the_switch():
    assert el.switch_on({"GREYHOUND_EARLY_LAY": "live"}) and not el.switch_on({"GREYHOUND_EARLY_LAY": "paper"})
    assert not el.switch_on({})


def test_the_volume_is_used_only_where_the_fit_saw_it(fit):
    assert not el.vol_in_fit(fit)                              # constant in training: mu 0, sd 1
    cols = list(fit["cols"])
    fit2 = dict(fit, mu=np.where(np.array(cols) == "lq_market_x_vol", 2.5, 0.0))
    assert el.vol_in_fit(fit2)


def test_the_live_table_from_the_ledger(tmp_path, fit):
    from sources.common import Store as S3Store
    store = S3Store(root=str(tmp_path / "s3"))
    x = FakeX()
    x.book = book()
    clock = {"t": T0}
    t = el.EarlyLay(store, x, dict(el.CONFIG), date(2026, 11, 3), True, tmp_path / "ledger.csv", fit,
                    now=lambda: clock["t"])
    t.load_predictions = lambda: None
    t.preds = preds().assign(off=pd.Timestamp(T0 + timedelta(minutes=10)))
    t.step()
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=4)
    t.step(near_only=True)
    md = el.table(store, date(2026, 11, 3))
    assert "| Romford |" in md and "locked by green-ups" in md
    assert f"{t.locked:+.2f}" in md


def test_a_race_off_late_is_still_closed(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    clock["t"] = T0 + timedelta(minutes=10) + timedelta(seconds=20)    # past the scheduled off, the market still open
    t.step(near_only=True)
    assert [i for i in x.sent if i["orderType"] == "MARKET_ON_CLOSE"] and MID in t.greened


def test_the_heartbeat_reaches_the_table(tmp_path, fit):
    from sources.common import Store as S3Store
    store = S3Store(root=str(tmp_path / "s3"))
    x = FakeX()
    x.book = book()
    t = el.EarlyLay(store, x, dict(el.CONFIG), date(2026, 11, 3), True, tmp_path / "ledger.csv", fit,
                    now=lambda: T0)
    t.load_predictions = lambda: None
    t.preds = preds().assign(off=pd.Timestamp(T0 + timedelta(minutes=10)))
    t.step()
    t.heartbeat(force=True)
    md = el.table(store, date(2026, 11, 3))
    assert "Trader: last seen 2026-11-03T14:00:00Z" in md and "priced 1" in md


def test_no_green_up_before_5_seconds_out(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    n = len(x.sent)
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=20)
    t.step()
    t.step(near_only=True)
    assert len(x.sent) == n and not x.cancels and MID not in t.greened    # the lays still rest at 20 s out


def test_the_table_says_why_a_lay_was_not_greened(tmp_path, fit):
    from sources.common import Store as S3Store
    store = S3Store(root=str(tmp_path / "s3"))
    x = FakeX()
    x.book = book()
    clock = {"t": T0}
    t = el.EarlyLay(store, x, dict(el.CONFIG), date(2026, 11, 3), True, tmp_path / "ledger.csv", fit,
                    now=lambda: clock["t"])
    t.load_predictions = lambda: None
    t.preds = preds().assign(off=pd.Timestamp(T0 + timedelta(minutes=10)))
    t.step()
    x.book = book(status="SUSPENDED")                          # the market off before the green-up
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=4)
    t.step(near_only=True)
    md = el.table(store, date(2026, 11, 3))
    assert "hold (the market closed before the green-up)" in md
