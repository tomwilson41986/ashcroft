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
            self.cancel_params = getattr(self, "cancel_params", []) + [params]
            return {"status": "SUCCESS"}
        raise AssertionError(method)

    def orders(self, market_ids, refs=None):
        return list(self.current)

    def funds(self):                                          # the gate reads the account's funds on live orders
        return {"available": 10_000.0}


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
    cfg = dict(el.CONFIG, liability=20.0, min_stake=2.0, max_stake=10.0)      # the stakes of 3-11 Nov
    assert el.stake_for(5.0, cfg) == 5.0                       # GBP20 liability at 5.0
    assert el.stake_for(11.0, cfg) == 2.0                      # GBP2 at the smallest
    assert el.stake_for(1.5, cfg) == 10.0                      # GBP10 at the largest
    live = dict(el.CONFIG)                                     # from 11 Nov 2026: the forward test's GBP100 a lay
    assert el.stake_for(5.0, live) == 25.0 and el.stake_for(1.5, live) == 50.0 and el.stake_for(60.0, live) == 2.0
    assert live["max_market_liability"] == 8 * live["liability"]
    from greyhound import tradeout as to                       # the paper's GBP100, by the same rule
    big = dict(cfg, liability=to.TO_WIN, min_stake=to.MIN_STAKE, max_stake=to.MAX_STAKE)
    for price in (1.5, 2.0, 3.3, 5.0, 11.0, 20.0, 60.0):
        assert np.isclose(el.stake_for(price, big), el._floor2(float(to.target_stake(price))))


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


def test_the_drifters_only_at_the_margin_under_the_forecast(tmp_path, fit):
    from research.greyhound import limit_study as ls
    x = FakeX()
    x.book = book()
    t, _ = trader(tmp_path, x, fit)
    t.step()
    led = pd.read_csv(tmp_path / "ledger.csv")
    m = 1 - el.CONFIG["margin"]                                   # 0.8 (0.9 on 4 Nov 2026 only)
    for r in led.itertuples():
        if r.action == "lay":
            assert r.f_bsp >= r.mid and r.price == ls.ladder([r.f_bsp * m], up=False)[0]
        else:
            assert r.f_bsp < r.mid or ls.ladder([r.f_bsp * m], up=False)[0] > el.CONFIG["max_price"] \
                or ls.ladder([r.f_bsp * m], up=False)[0] >= r.mid - 1e-9


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
    exp = 0.0
    for s in lays:                                               # the worse of the two outcomes, stakes in pennies
        S, m, B = lays[s]["size"], lays[s]["price"], x.book.runners[s].best_back[0]
        b = el._floor2(S * m / B)
        exp += round(min(-S * (m - 1) + b * (B - 1), S - b), 2)
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
    assert p[2]["to_cover"] == 0.0 and not p[2]["closed"]          # greened: nothing left to cover
    assert el.outcomes(p[2]) == (pytest.approx(-9.0 + 9.5), pytest.approx(0.5))


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


def test_a_green_up_takes_the_ladder_and_keeps_a_part_filled(tmp_path, fit):
    """The best back price holds 20p: the back takes the next prices too, and what is not matched is tried again
    the next second for the value still open."""
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    lays = {i["selectionId"]: i["limitOrder"] for i in x.sent if i["side"] == "LAY"}
    sid = next(iter(lays))
    q = x.book.runners[sid]
    q.back = [(q.back[0][0], 0.2), (el.tick_down(q.back[0][0]) if hasattr(el, "tick_down") else q.back[0][0] - 0.1, 500.0)]
    full = x._place                                            # the first green-up back matches GBP1 only
    calls = {"n": 0}

    def part(mid, ins, ref):
        if ins["side"] == "BACK" and ins["customerOrderRef"].endswith("-g") and calls["n"] == 0:
            calls["n"] += 1
            ins = dict(ins, limitOrder=dict(ins["limitOrder"], size=1.0))
        return full(mid, ins, ref)
    x._place = part
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=4)
    t.step(near_only=True)
    g = [i for i in x.sent if i["side"] == "BACK" and i["selectionId"] == sid]
    assert g[0]["limitOrder"]["price"] == q.back[1][0]          # down the ladder to where the value is covered
    assert g[0]["limitOrder"]["minFillSize"] == el.CONFIG["min_order"]
    assert MID not in t.greened                                  # a part matched: the rest next second
    clock["t"] += timedelta(seconds=1)
    t.step(near_only=True)
    g = [i for i in x.sent if i["side"] == "BACK" and i["selectionId"] == sid]
    assert len(g) == 2
    L = lays[sid]
    rest = L["size"] * L["price"] - 1.0 * g[0]["limitOrder"]["price"]
    lim, st = el.ladder_stake(q.back, rest)
    assert lim == g[1]["limitOrder"]["price"] and np.isclose(g[1]["limitOrder"]["size"], el._floor2(st))
    assert MID in t.greened


def test_the_green_up_stake_covers_the_value_filled_best_price_first():
    """Betfair fills a limit back at the best prices first: the stake is sized level by level, so the backs' value is
    the lay's, not more (sized at the lowest price alone it backed more: the over-cover of 3-13 Nov)."""
    levels = [(4.0, 10.0), (3.9, 10.0), (3.8, 500.0)]
    lim, st = el.ladder_stake(levels, 100.0)
    assert lim == 3.8
    got, cost, left = 0.0, 0.0, st                              # Betfair's fill, best price first
    for price, size in levels:
        take = min(size, left)
        got, cost, left = got + take, cost + take * price, left - take
    assert np.isclose(cost, 100.0)
    assert el.ladder_stake(levels, 30.0) == (4.0, 7.5)           # covered at the best price alone
    lim, st = el.ladder_stake([(4.0, 1.0)], 40.0)                # the levels hold less: the rest at the last price
    assert lim == 4.0 and np.isclose(st, 10.0)
    assert el.ladder_stake([], 40.0) == (None, 0.0)


def test_a_fill_betfair_does_not_list_yet_is_not_backed_again(tmp_path, fit):
    """A green-up back matched but missing from listCurrentOrders on the next poll is counted from the fill itself."""
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    lays = {i["selectionId"]: i["limitOrder"] for i in x.sent if i["side"] == "LAY"}
    sid = next(iter(lays))
    full = x._place

    def part(mid, ins, ref):                                     # the first back matches GBP1 only, and is not listed
        if ins["side"] == "BACK" and not any(i["side"] == "BACK" for i in x.sent):
            ins = dict(ins, limitOrder=dict(ins["limitOrder"], size=1.0))
            r = full(mid, ins, ref)
            x.current.pop()
            return r
        return full(mid, ins, ref)
    x._place = part
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=4)
    t.step(near_only=True)
    clock["t"] += timedelta(seconds=1)
    t.step(near_only=True)
    backs = [i for i in x.sent if i["side"] == "BACK" and i["selectionId"] == sid]
    value = sum(b["limitOrder"]["size"] * b["limitOrder"]["price"] for b in backs)
    L = lays[sid]
    assert value <= L["size"] * L["price"] + 0.05                 # never more than the lay's value


def test_the_table_lists_the_matched_lays_dog_by_dog(tmp_path, fit):
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
    assert "#### Live lays that matched, dog by dog" in md
    n = sum(1 for i in x.sent if i["side"] == "LAY")
    rows = [ln for ln in md.split("dog by dog")[1].splitlines() if ln.startswith("| ") and "Romford" in ln]
    assert len(rows) == n and all("| green |" in r for r in rows)


def test_the_guard_is_off_by_default_as_the_tested_rule():
    """The owner, 11 Nov 2026: "lose the guard. I want to be exactly like the rule that was profitable in test": the
    forward test's rule (greyhound/laytest.py PRIMARY) lays every forecast drifter at 80% of the forecast SP."""
    from greyhound import laytest
    assert el.CONFIG["rest_only"] is False
    assert el.CONFIG["margin"] == laytest.PRIMARY["param"] and el.CONFIG["gate"] == laytest.PRIMARY["gate"]


def test_never_a_lay_at_or_above_the_mid(tmp_path, fit):
    """With the guard on (rest_only, the owner's rule of 3 Nov to 11 Nov): a lay priced at or above the market's mid
    is not placed."""
    from model.race_book import market_now
    x = FakeX()
    x.book = book()
    t = el.EarlyLay(Store(preds()), x, dict(el.CONFIG, rest_only=True), date(2026, 11, 3), True,
                    tmp_path / "ledger.csv", fit, now=lambda: T0)
    t.step()
    led = pd.read_csv(tmp_path / "ledger.csv")
    for i in (i for i in x.sent if i["side"] == "LAY"):
        q = x.book.runners[i["selectionId"]]
        assert i["limitOrder"]["price"] < float(market_now([q.best_back[0]], [q.best_lay[0]])[0])
    above = led[led.error.astype(str).str.contains("at or above the mid", na=False)]
    assert (above.action == "no_lay").all()
    # off, the old rule lays the same drifters whatever the mid
    x2 = FakeX()
    x2.book = book()
    t2 = el.EarlyLay(Store(preds()), x2, dict(el.CONFIG, rest_only=False), date(2026, 11, 3), True,
                     tmp_path / "ledger2.csv", fit, now=lambda: T0)
    t2.step()
    assert len([i for i in x2.sent if i["side"] == "LAY"]) >= len([i for i in x.sent if i["side"] == "LAY"])


def test_the_blocked_lays_replayed():
    t = pd.Timestamp("2026-11-03 15:00:00", tz="UTC")
    rows = pd.DataFrame([
        {"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds_to_off": 300, "market_id": "1.1", "selection_id": 7,
         "price": 5.0, "race_time": "15:05", "track": "Romford", "trap": 1, "f_bsp": 6.25, "mid": 4.9},
        {"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds_to_off": 300, "market_id": "1.1", "selection_id": 8,
         "price": 5.0, "race_time": "15:05", "track": "Romford", "trap": 2, "f_bsp": 6.25, "mid": 5.2}])
    books = pd.DataFrame([
        {"market_id": "1.1", "selection_id": 7, "polled": t, "back1": 4.8, "lay1": 5.0, "last_traded": 4.9},
        {"market_id": "1.1", "selection_id": 7, "polled": t + pd.Timedelta(seconds=290), "back1": 6.0, "lay1": 6.2,
         "last_traded": 6.0},
        {"market_id": "1.1", "selection_id": 8, "polled": t, "back1": 5.1, "lay1": 5.4, "last_traded": 5.2},
        {"market_id": "1.1", "selection_id": 8, "polled": t + pd.Timedelta(seconds=290), "back1": 5.6, "lay1": 5.8,
         "last_traded": 5.7}])
    d = el.replay(rows, books)
    a, b = d.iloc[0], d.iloc[1]
    s = el.stake_for(5.0, el.CONFIG)
    # trap 1: the best lay price 5.0 met our 5.0 at once; greened at 6.0
    assert a.matched and a.matched_at == 5.0 and a.greened_at == 6.0 and a.pnl == round(s * (1 - 5.0 / 6.0), 2)
    # trap 2: resting at 5.0, the price drifted away and never traded there: not matched
    assert b.matched is False or b.matched == False                                  # noqa: E712
    assert "never reached" in b.note


def test_the_live_rule_lays_at_80_percent_of_the_forecast():
    """The owner, 4 Nov 2026: "revert the early lay margin to 20%"; the 10% rule kept on paper."""
    assert el.CONFIG["margin"] == 0.20 and el.CONFIG["gate"] == "drift"
    assert el.PAPER_VARIANTS["m10"] == {"margin": 0.10}
    assert el.PAPER_VARIANTS["short20"] == {"gate": "short", "margin": 0.20}
    assert el.PAPER_VARIANTS["short10"] == {"gate": "short", "margin": 0.10}


def test_the_short_gate_lays_only_dogs_under_3(tmp_path, fit):
    x = FakeX()
    x.book = book(back=(1.9, 6.0, 9.0), lay=(2.0, 6.4, 9.6))
    cfg = {**el.CONFIG, **el.PAPER_VARIANTS["short20"]}
    t = el.EarlyLay(Store(preds()), x, cfg, date(2026, 11, 3), False, tmp_path / "ledger.csv", fit,
                    now=lambda: T0, book="paper-short20")
    t.step()
    assert not x.sent
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert set(led["mode"]) == {"paper-short20"} and len(led) == 3
    assert (led[led.action == "lay"].mid < el.SHORT_PRICE).all()
    longs = led[led.mid >= el.SHORT_PRICE]
    assert len(longs) == 2 and (longs.action == "no_lay").all() and longs.error.str.contains("not short-priced").all()
    with pytest.raises(ValueError):
        el.decide(pd.DataFrame(), fit, {**cfg, "gate": "fav"})


def test_a_paper_book_never_trades_live(tmp_path, fit):
    with pytest.raises(ValueError):
        el.EarlyLay(Store(preds()), FakeX(), dict(el.CONFIG), date(2026, 11, 3), True, tmp_path / "l.csv", fit,
                    book="paper-m10")
    with pytest.raises(SystemExit):
        el.main(["--trade", "--live", "--paper-variants", "m10"])
    with pytest.raises(SystemExit):
        el.main(["--trade", "--paper-variants", "nope"])


def test_the_paper_books_share_one_read_of_the_books(tmp_path, fit):
    class Counting(FakeX):
        reads = 0

        def books(self, mids, with_sp=False):
            Counting.reads += 1
            return super().books(mids)

    x = Counting()
    x.book = book()
    clock = {"t": T0, "mono": 0.0}
    shared = el.SharedBooks(x, clock=lambda: clock["mono"])
    traders = [el.EarlyLay(Store(preds()), shared, {**el.CONFIG, **el.PAPER_VARIANTS[n]}, date(2026, 11, 3), False,
                           tmp_path / n / "ledger.csv", fit, now=lambda: clock["t"], book=f"paper-{n}")
               for n in el.PAPER_VARIANTS]

    def sleep(s):                     # one full poll, then past the race (a stop time after it, so it is laid)
        clock["t"] += timedelta(minutes=25)
        clock["mono"] += 25 * 60

    out = el.run_many(traders, T0 + timedelta(minutes=30), sleep=sleep)
    assert [o["mode"] for o in out] == [f"paper-{n}" for n in el.PAPER_VARIANTS]
    assert Counting.reads == 1                                 # one read a poll, whatever the number of books
    assert not x.sent and not x.cancels
    for n in el.PAPER_VARIANTS:
        assert (tmp_path / n / "ledger.csv").exists()
    with pytest.raises(AssertionError):
        el.run_many([el.EarlyLay(Store(preds()), x, dict(el.CONFIG), date(2026, 11, 3), True, tmp_path / "l.csv",
                                 fit)], T0)


def test_a_paper_book_is_scored_on_the_books():
    t = pd.Timestamp("2026-11-03 15:00:00", tz="UTC")
    led = pd.DataFrame([
        {"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": "paper-short20", "action": "lay", "seconds_to_off": 300,
         "market_id": "1.1", "selection_id": 7, "price": 2.0, "race_time": "15:05", "track": "Romford", "trap": 1,
         "f_bsp": 2.5, "mid": 2.1, "status": "PAPER"},
        {"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": "paper-short20", "action": "no_lay",
         "seconds_to_off": 300, "market_id": "1.1", "selection_id": 8, "price": 5.0, "race_time": "15:05",
         "track": "Romford", "trap": 2, "f_bsp": 6.0, "mid": 5.2, "error": "not short-priced (mid 3 or more)"}])
    books = pd.DataFrame([
        {"market_id": "1.1", "selection_id": 7, "polled": t, "back1": 2.08, "lay1": 2.12, "last_traded": 2.1},
        {"market_id": "1.1", "selection_id": 7, "polled": t + pd.Timedelta(seconds=120), "back1": 1.98, "lay1": 2.0,
         "last_traded": 2.0},
        {"market_id": "1.1", "selection_id": 7, "polled": t + pd.Timedelta(seconds=290), "back1": 2.2, "lay1": 2.24,
         "last_traded": 2.2}])

    class S:
        def get(self, key):
            if key.endswith("paper-short20/2026-11-03/ledger.csv"):
                return led.to_csv(index=False).encode()
            return None

    md = el.paper_table(S(), date(2026, 11, 3), "short20", [("books_greyhound_live.csv", books)])
    stake = el.stake_for(2.0, {**el.CONFIG, **el.PAPER_VARIANTS["short20"]})
    assert "1 lays in 1 races; 1 would have matched; greened up 1" in md
    assert f"GBP{round(stake * (1 - 2.0 / 2.2), 2):+.2f}" in md and "live key" in md
    assert "no ledger yet" in el.paper_table(S(), date(2026, 11, 3), "m10", [])


# --- the audit's safety fixes (the owner, 5 Nov 2026: "Yes do these") -------------------------------------------------

def test_live_needs_a_daily_loss_limit():
    assert el.daily_loss_limit({}) is None
    assert el.daily_loss_limit({"GREYHOUND_EARLY_LAY_DAILY_LOSS": ""}) is None
    assert el.daily_loss_limit({"GREYHOUND_EARLY_LAY_DAILY_LOSS": "abc"}) is None
    assert el.daily_loss_limit({"GREYHOUND_EARLY_LAY_DAILY_LOSS": "-5"}) is None
    assert el.daily_loss_limit({"GREYHOUND_EARLY_LAY_DAILY_LOSS": "50"}) == 50.0


def test_no_lay_on_a_race_after_the_trader_stops(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, _ = trader(tmp_path, x, fit)
    t.until = T0 + timedelta(minutes=10)                       # the race goes off at T0 + 10 min: too late to close
    t.step()
    assert not x.sent and MID not in t.priced
    t.until = T0 + timedelta(minutes=30)
    t.step()
    assert any(i["side"] == "LAY" for i in x.sent)


def _cancels(x):
    return [m for m in x.cancels if m == MID]


def test_resting_lays_are_cancelled_when_the_run_ends_or_is_killed(tmp_path, fit):
    x = FakeX(lay_fill=0.0)                                   # nothing matches: every lay rests
    x.book = book()
    t, clock = trader(tmp_path, x, fit)

    def sleep(s):                                             # the job is cancelled after the first poll
        raise SystemExit("signal 15")

    with pytest.raises(SystemExit):
        t.run(T0 + timedelta(minutes=30), sleep=sleep)
    assert any(i["side"] == "LAY" for i in x.sent) and _cancels(x)
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert (led.action == "cancel").any() and set(led[led.action == "cancel"].error) == {"trader stopped"}


def test_the_stop_switch_cancels_every_resting_lay_once(tmp_path, fit):
    x = FakeX(lay_fill=0.0)
    x.book = book()
    state = {"stop": False}
    clock = {"t": T0}
    t = el.EarlyLay(Store(preds()), x, dict(el.CONFIG), date(2026, 11, 3), True, tmp_path / "ledger.csv", fit,
                    kill=lambda: state["stop"], now=lambda: clock["t"])
    t.step()
    assert any(i["side"] == "LAY" for i in x.sent) and not x.cancels
    state["stop"] = True
    clock["t"] += timedelta(seconds=15)
    t.step()
    n = len(_cancels(x))
    assert n >= 1
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert (led[led.action == "cancel"].error == "stopped (S3 STOP)").sum() == 1
    clock["t"] += timedelta(seconds=15)
    t.step()
    assert (pd.read_csv(tmp_path / "ledger.csv").query("action == 'cancel'").shape[0]) == 1   # once a market


def test_the_kill_switch_fails_safe():
    from greyhound import live as gl

    class Err(Exception):
        def __init__(self, code):
            super().__init__(code)
            self.response = {"Error": {"Code": code}}

    class S3:
        def __init__(self, exc):
            self.exc = exc

        def head_object(self, **kw):
            if self.exc:
                raise self.exc
            return {}

    class St:
        root, bucket = None, "b"

        def __init__(self, exc):
            self.s3 = S3(exc)

    assert gl._exists(St(None), "greyhound/STOP") is True
    assert gl._exists(St(Err("404")), "greyhound/STOP") is False
    assert gl._exists(St(Err("NoSuchKey")), "greyhound/STOP") is False
    assert gl._exists(St(Err("AccessDenied")), "greyhound/STOP") is True      # unreadable: stop
    assert gl._exists(St(Exception("timeout")), "greyhound/STOP") is True
    assert gl.KILL_CHECK_SECONDS <= 5



# --- the audit's next items (the owner, 5 Nov 2026: "Work on the next items") ------------------------------------------

def test_the_gate_refuses_what_breaks_a_limit(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, _ = trader(tmp_path, x, fit)
    assert t.gate(MID, 5.0, el.stake_for(5.0, t.cfg)) == ""
    assert "price" in t.gate(MID, 25.0, 2.0)                  # over max_price
    assert "stake" in t.gate(MID, 5.0, 60.0)                  # over max_stake
    assert "stopped" in t.gate(MID, 5.0, 2.0, stopped=True)
    t.laid[MID] = t.cfg["max_market_liability"]
    assert "race's liability" in t.gate(MID, 5.0, 2.0)
    t.laid = {"a": t.cfg["max_open_liability"]}
    assert "open liability" in t.gate(MID, 5.0, 2.0)
    t.laid = {}
    t.cfg["daily_loss_limit"] = 10.0
    t.locked = -5.0
    assert "loss limit" in t.gate(MID, 5.0, 2.0)            # -5 less this lay's GBP8 passes -10
    t.cfg["daily_loss_limit"], t.locked = None, 0.0
    t._sent = [pd.Timestamp(T0)] * t.cfg["max_orders_per_minute"]
    assert "order rate" in t.gate(MID, 5.0, 2.0)
    t._sent = []
    x.funds = lambda: None                                    # funds unreadable: no live order
    t._funds = None
    assert "funds" in t.gate(MID, 5.0, 2.0)
    x.funds = lambda: {"available": 3.0}
    t._funds = None
    assert "funds GBP3.00" in t.gate(MID, 5.0, 2.0)


def test_a_lay_the_gate_refuses_is_logged_not_sent(tmp_path, fit):
    x = FakeX()
    x.book = book()
    x.funds = lambda: None
    t, _ = trader(tmp_path, x, fit)
    t.step()
    assert not [i for i in x.sent if i["side"] == "LAY"]
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert led.error.astype(str).str.contains("risk gate").any()


def test_cancel_touches_only_our_orders():
    x = FakeX()
    x.current = [{"betId": "1", "customerOrderRef": "el-1103-x-1-l", "side": "LAY", "status": "EXECUTABLE"},
                 {"betId": "2", "customerOrderRef": "qm-1103-x-1-b", "side": "BACK", "status": "EXECUTABLE"},
                 {"betId": "3", "customerOrderRef": "el-1103-x-2-l", "side": "LAY", "status": "EXECUTION_COMPLETE"}]
    o = el.Orders(x, live=True)
    o.cancel(MID)
    assert [i["betId"] for i in x.cancel_params[-1]["instructions"]] == ["1"]
    x.current = [x.current[1]]
    assert o.cancel(MID) == "NONE"                          # nothing of ours: no call


def test_state_rebuilt_from_betfairs_orders(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, _ = trader(tmp_path, x, fit)
    t.load_predictions()
    x.current = [  # an open lay on MID, matched and not greened; nothing on another market of ours
        {"marketId": MID, "selectionId": 3, "side": "LAY", "sizeMatched": 2.0, "averagePriceMatched": 9.0,
         "sizeRemaining": 0.0, "customerOrderRef": "el-1103-50000001-3-l", "betId": "9"},
        {"marketId": MID, "selectionId": 1, "side": "BACK", "sizeMatched": 5.0, "averagePriceMatched": 4.0,
         "customerOrderRef": "qm-1103-x", "betId": "8"}]
    r = t.recover()
    assert r == {"markets": 1, "open": 1} and MID in t.priced and MID not in t.greened
    assert np.isclose(t.laid[MID], 16.0)                     # 2 x (9 - 1)
    x.current.append({"marketId": MID, "selectionId": 3, "side": "BACK", "sizeMatched": 2.0,
                      "averagePriceMatched": 9.0, "customerOrderRef": "el-1103-50000001-3-g", "betId": "10"})
    t2, _ = trader(tmp_path / "b", x, fit)
    t2.load_predictions()
    assert t2.recover() == {"markets": 1, "open": 0} and MID in t2.greened


def test_reconcile_against_betfairs_settled_bets():
    led = pd.DataFrame([
        {"market_id": MID, "action": "lay", "locked_pnl": None},
        {"market_id": MID, "action": "green", "locked_pnl": 1.50},
        {"market_id": "1.2", "action": "lay", "locked_pnl": None}])

    class S:
        def __init__(self):
            self.put_keys = []

        def get(self, key):
            return led.to_csv(index=False).encode() if key.endswith("ledger.csv") else None

        def put(self, key, data):
            self.put_keys.append(key)

    class X:
        def cleared(self, mids):
            return [{"marketId": MID, "customerOrderRef": "el-1103-a-1-l", "profit": -4.0},
                    {"marketId": MID, "customerOrderRef": "el-1103-a-1-g", "profit": 6.0},
                    {"marketId": MID, "customerOrderRef": "qm-1103-a", "profit": 100.0}]

    st = S()
    r = el.reconcile(st, X(), date(2026, 11, 3), comm=0.02)
    assert r["settled"] == 1 and r["unsettled"] == ["1.2"] and r["betfair_profit_gross"] == 2.0
    assert r["betfair_profit_net"] == round(2.0 - 0.02 * 2.0, 2) and r["difference"] == 0.5
    assert any(k.endswith("reconcile.json") for k in st.put_keys)


def test_commission_rate():
    assert el.commission({}) == el.CONFIG["commission"]
    assert el.commission({"GREYHOUND_COMMISSION": "0.05"}) == 0.05
    assert el.commission({"GREYHOUND_COMMISSION": "x"}) == el.CONFIG["commission"]


def test_the_running_page_shows_the_rule_on_paper(tmp_path):
    class S:
        def __init__(self, keys):
            self.keys = keys

        def get(self, key):
            return b"time_utc,action\n" if key in self.keys else None

    day = date(2026, 11, 5)
    assert el.modes_to_show(S(set()), day) == ["live"]
    assert el.modes_to_show(S({f"{el.PREFIX}/paper/2026-11-05/ledger.csv"}), day) == ["paper"]
    both = {f"{el.PREFIX}/{m}/2026-11-05/ledger.csv" for m in ("live", "paper")}
    assert el.modes_to_show(S(both), day) == ["live", "paper"]
    assert "nothing sent" in el.table(S(set()), day, "paper")
