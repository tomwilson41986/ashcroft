"""The favourite back on paper (greyhound/fav_back.py): the favourite only, once a race, at the forecast x 1.20 on the
ladder, staked to win GBP100; never an order sent; the fills by the registered rule; the SP close and the back to win;
the stop switch; a restart; the running table."""

from __future__ import annotations

import ast
import inspect
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from greyhound import fav_back as fb
from greyhound import tradeout as to
from research.greyhound import limit_study as ls
from tests.test_greyhound_early_lay import MID, T0, FakeX, Store, book, preds  # noqa: F401
from tests.test_greyhound_early_lay import fit  # noqa: F401  (the fixture)


def trader(tmp_path, x, fit, now=T0, kill=False, p=None):
    clock = {"t": now}
    t = fb.FavBack(Store(p if p is not None else preds()), x, dict(fb.CONFIG), date(2026, 11, 3),
                   tmp_path / "ledger.csv", fit, kill=lambda: kill, now=lambda: clock["t"])
    return t, clock


def test_the_favourite_only_once_a_race_and_nothing_sent(tmp_path, fit):
    x = FakeX()
    x.book = book(traded=False)
    t, clock = trader(tmp_path, x, fit)
    t.step()
    assert MID not in t.priced                                  # nobody has traded yet
    x.book = book(traded=True)
    clock["t"] += timedelta(seconds=15)
    t.step()
    clock["t"] += timedelta(seconds=15)
    t.step()
    assert not x.sent and not x.cancels                        # paper: never an order or a cancel
    led = pd.read_csv(tmp_path / "ledger.csv")
    backs = led[led.action == "back"]
    assert len(backs) == 1 and int(backs.selection_id.iloc[0]) == 1    # trap 1, the shortest mid (back 4.0)
    r = backs.iloc[0]
    want = ls.ladder(np.array([r.f_bsp * 1.20]), up=True)[0]
    assert np.isclose(r["limit"], want) and np.isclose(r.stake, round(float(to.target_stake(want)), 2))


def test_the_registered_fill_needs_a_trade_beyond_the_limit():
    assert fb.fill_now(5.0, 5.0) and not fb.fill_now(5.0, 4.9)
    assert fb.fill_later(5.0, 4.8, 5.0) == {"strict": False, "through": False}
    assert fb.fill_later(5.0, 5.0, 5.0) == {"strict": False, "through": True}
    assert fb.fill_later(5.0, 4.8, 5.1) == {"strict": True, "through": False}


def test_fills_logged_once_and_closed_at_the_off(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    o = t.open[MID]
    o["fills"].clear()                                          # a resting back: the limit above the price on offer
    o["limit"] = lim = 6.0
    # the favourite drifts: it trades beyond our limit, twice
    for _ in range(2):
        clock["t"] += timedelta(seconds=20)
        x.book = book(back=(lim + 0.5, 6.0, 9.0), lay=(lim + 0.7, 6.4, 9.6))
        x.book.runners[1].last_traded = lim + 0.6
        t.step()
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=3)
    t.step(near_only=True)
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert (led.action == "fill_strict").sum() == 1 and (led.action == "fill_through").sum() == 1
    assert (led.action == "sp_close").sum() == 1 and MID in t.closed and MID not in t.open
    assert not x.sent


def test_an_unmatched_back_lapses(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    t.open[MID]["fills"].clear()                                # never matched: a limit the market never reaches
    t.open[MID]["limit"] = 20.0
    clock["t"] = T0 + timedelta(minutes=10) - timedelta(seconds=3)
    t.step(near_only=True)
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert (led.action == "lapsed").sum() == 1 and "sp_close" not in set(led.action)


def test_a_back_already_on_offer_is_taken(tmp_path, fit):
    x = FakeX()
    x.book = book(back=(9.0, 12.0, 15.0), lay=(9.4, 12.6, 15.8))   # the favourite's back price above any forecast x 1.2
    t, _ = trader(tmp_path, x, fit)
    t.step()
    led = pd.read_csv(tmp_path / "ledger.csv")
    r = led[led.action == "back"].iloc[0]
    if r["limit"] <= 9.0:
        assert (led.action == "fill_taken").sum() == 1 and "fill_taken" in t.open[MID]["fills"]


def test_the_stop_switch_backs_nothing(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, _ = trader(tmp_path, x, fit, kill=True)
    t.step()
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert set(led.action) == {"no_back"} and MID in t.priced and not t.open


def test_a_restart_resumes_from_the_ledger(tmp_path, fit):
    x = FakeX()
    x.book = book()
    t, clock = trader(tmp_path, x, fit)
    t.step()
    t2, _ = trader(tmp_path, x, fit, now=clock["t"] + timedelta(seconds=30))
    assert MID in t2.priced and MID in t2.open and t2.open[MID]["limit"] == t.open[MID]["limit"]
    t2.step()
    assert (pd.read_csv(tmp_path / "ledger.csv").action == "back").sum() == 1


def test_settlement_closed_at_the_sp_and_held_to_win():
    led = pd.DataFrame([
        {"market_id": "1.1", "selection_id": 7, "track": "Hove", "race_time": "19:00", "trap": 1, "dog": "A",
         "action": "back", "limit": 3.0, "mid": 2.6, "f_bsp": 2.5, "stake": 50.0},
        {"market_id": "1.1", "selection_id": 7, "action": "fill_strict", "fill_price": 3.0},
        {"market_id": "1.2", "selection_id": 8, "track": "Hove", "race_time": "19:15", "trap": 2, "dog": "B",
         "action": "back", "limit": 4.0, "mid": 3.2, "f_bsp": 3.3, "stake": 33.33},
        {"market_id": "1.3", "selection_id": 9, "track": "Hove", "race_time": "19:30", "trap": 3, "dog": "C",
         "action": "back", "limit": 3.0, "mid": 2.4, "f_bsp": 2.5, "stake": 50.0},
        {"market_id": "1.3", "selection_id": 9, "action": "fill_through", "fill_price": 3.0}])
    close = pd.DataFrame({"market_id": ["1.1", "1.2", "1.3"], "selection_id": [7, 8, 9], "bsp": [2.5, 3.0, 2.0],
                          "won": [0, 1, 0]})
    s = fb.settle_frame(led, close, comm=0.02)
    a = s.set_index("market_id")
    # 1.1: backed 50 at 3.0, lost; closed at the SP 2.5: -50 + 50 x 2 / 1.5 = 16.67, less 2%
    assert np.isclose(a.loc["1.1", "pnl_sp"], (-50 + 50 * 2 / 1.5) * 0.98) and np.isclose(a.loc["1.1", "pnl_held"], -50)
    assert a.loc["1.2", "pnl_sp"] == 0 and not a.loc["1.2", "strict"]           # never filled
    assert not a.loc["1.3", "strict"] and a.loc["1.3", "through"] and a.loc["1.3", "pnl_sp"] == 0
    assert a.loc["1.3", "pnl_sp_through"] > 0
    sm = fb.summary(s)
    assert sm["backs"] == 3 and sm["filled_strict"] == 1 and sm["filled_through"] == 2
    assert np.isclose(sm["clv_mean%"], 100 * (3.0 / 2.5 - 1))


def test_the_table_renders(tmp_path, monkeypatch):
    from sources.common import Store as S3
    store = S3(root=tmp_path / "s3")
    day = date(2026, 11, 3)
    assert "No status yet" in fb.table(store, day)
    pd.DataFrame([{"market_id": "1.1", "selection_id": 7, "track": "Hove", "race_time": "19:00", "trap": 1,
                   "dog": "A", "action": "back", "limit": 3.0, "mid": 2.6, "f_bsp": 2.5, "stake": 50.0}]
                 ).to_csv(tmp_path / "l.csv", index=False)
    store.put(f"{fb.PREFIX}/paper/{day}/ledger.csv", (tmp_path / "l.csv").read_bytes())
    monkeypatch.setattr(fb, "closing", lambda d, s3=None: pd.DataFrame(columns=["market_id", "selection_id", "bsp",
                                                                                 "won"]))
    md = fb.table(store, day)
    assert "| 19:00 Hove | 1 | A |" in md and "Backs 1" in md


def test_the_module_cannot_send_an_order():
    src = inspect.getsource(fb)
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert not names & {"_place", "place", "back_at_sp", "lay_at_bsp", "back", "lay", "cancelOrders", "_call"}
    assert "placeOrders" not in src and "--live" not in src
