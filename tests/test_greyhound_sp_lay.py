"""greyhound/sp_lay.py: the served picks laid once at the SP in the last minutes before the off, never past the
day's loss limit at its worst, stopped by its own STOP, and nothing sent unless switched on."""

import io
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from greyhound import sp_lay as sl
from sources.common import Store
from trading.exchange import Book, Fill, Quote

DAY = date(2026, 11, 21)
T0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
CUT = -0.04


class FakeX:
    def __init__(self, removed=(), status="OPEN"):
        self.sent, self.cancelled, self.cleared_rows = [], [], []
        self.removed, self.mstatus = set(removed), status

    def books(self, mids, with_sp=False):
        out = {}
        for m in mids:
            sids = range(101, 107) if m.startswith("2.") else range(1, 7)      # a PLACE market's own selection ids
            out[m] = Book(market_id=m, status=self.mstatus, inplay=False, total_matched=0.0,
                          runners={s: Quote(selection_id=s, status="REMOVED" if (m, s) in self.removed else "ACTIVE")
                                   for s in sids})
        return out

    def lay_at_bsp(self, mid, sid, liability, ref):
        self.sent.append((mid, sid, liability, ref))
        return Fill(market_id=mid, selection_id=sid, side="LAY", order_type="MARKET_ON_CLOSE", price=0.0,
                    size=liability, bet_id=f"b{len(self.sent)}", status="PENDING")

    def cleared(self, mids):
        return [c for c in self.cleared_rows if c["marketId"] in mids]

    def _call(self, method, params):
        self.cancelled.append((method, params))
        return {"status": "SUCCESS"}


def served(n_races=8, minutes_apart=3, first_off_min=5):
    """Races of six runners; in each, trap 6 is under the cut (a pick) and the rest are not."""
    rows = []
    for i in range(n_races):
        off = T0 + timedelta(minutes=first_off_min + i * minutes_apart)
        for t in range(1, 7):
            rows.append({"day": f"{DAY}", "race_id": f"r{i}", "market_id": f"1.{100 + i}", "selection_id": t,
                         "trap": t, "dog_id": 1000 * i + t, "dog_name": f"dog{t}", "track": "Romford",
                         "race_time": off.strftime("%H:%M"), "off_utc": off.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "field": 6, "s": -0.08 if t == 6 else 0.016, "served_utc": "x", "code_hash": "h"})
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path):
    st = Store(root=tmp_path)
    st.put("greyhound/sp_offset/meta.json", json.dumps({"cuts": {"lay_bottom10": CUT}}).encode())
    buf = io.StringIO()
    served().to_csv(buf, index=False)
    st.put(f"greyhound/sp_offset/served/{DAY}.csv", buf.getvalue().encode())
    return st


def catalogue(n_races=8, minutes_apart=3, first_off_min=5, skip=()):
    """The recorder's catalogue: each race's WIN market (1.1xx, selections 1-6) and PLACE market (2.1xx, selections
    101-106), the runners named '<trap>. dog<trap>'; ``skip``: (race, trap) left out of the PLACE market."""
    rows = []
    for i in range(n_races):
        off = (T0 + timedelta(minutes=first_off_min + i * minutes_apart)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for t in range(1, 7):
            rows.append({"market_id": f"1.{100 + i}", "market_type": "WIN", "event_id": f"e{i}", "market_start_utc": off,
                         "selection_id": t, "runner_name": f"{t}. dog{t}"})
            if (i, t) not in skip:
                rows.append({"market_id": f"2.{100 + i}", "market_type": "PLACE", "event_id": f"e{i}",
                             "market_start_utc": off, "selection_id": 100 + t, "runner_name": f"{t}. Dog{t}"})
    return pd.DataFrame(rows)


def trader(store, x, live=True, clock=None, kill=lambda: False, cfg=None, cat=None):
    t = sl.SpLay(store, x, dict(sl.CONFIG, **(cfg or {})), DAY, live, kill=kill, comm=0.02,
                 clock=clock or (lambda: T0), catalogue=lambda day: cat)
    t.load()
    return t


def test_the_loss_room_is_the_day_at_its_worst():
    assert sl.loss_room(0.0, 0.0, 50.0, 300.0)[0]
    assert sl.loss_room(0.0, 250.0, 50.0, 300.0)[0]                 # 300 at the worst: exactly the limit
    assert not sl.loss_room(0.0, 260.0, 50.0, 300.0)[0]
    assert not sl.loss_room(-300.0, 0.0, 50.0, 300.0)[0]            # the limit reached: nothing more today
    assert sl.loss_room(120.0, 350.0, 50.0, 300.0)[0]               # a winning day buys room
    assert sl.net_after_commission({"a": 10.0, "b": -50.0}, 0.02) == pytest.approx(9.8 - 50.0)


def test_only_the_picks_due_in_the_window_are_laid_once_each_with_the_owners_stake(store):
    x = FakeX()
    t = trader(store, x)
    assert t.step() == 2                                            # races off at +5 and +8 min (the window is 10 min)
    assert [s[:3] for s in x.sent] == [("1.100", 6, 50.0), ("1.101", 6, 50.0)]
    assert x.sent[0][3] == "spl-1121-1.100-6" and len(x.sent[0][3]) <= 32
    assert t.step() == 0                                            # never twice
    led = pd.read_csv(io.BytesIO(store.get(f"greyhound/sp_lay/live/{DAY}/ledger.csv")))
    assert led.action.tolist() == ["lay", "lay"] and set(led.status) == {"PENDING"}


def test_a_removed_runner_in_the_race_or_a_closed_market_skips_the_race(store):
    x = FakeX(removed={("1.100", 3)})
    t = trader(store, x)
    t.step()
    assert [s[0] for s in x.sent] == ["1.101"]
    led = pd.read_csv(io.BytesIO(store.get(f"greyhound/sp_lay/live/{DAY}/ledger.csv")))
    assert "runner 3 REMOVED" in led[led.action == "skip"].note.iloc[0]
    x2 = FakeX(status="SUSPENDED")
    t2 = trader(store, x2)
    t2.mode = "live2"                                               # a separate ledger for this half
    t2.step()
    assert x2.sent == []


def test_the_limit_holds_the_seventh_open_lay_and_frees_room_as_races_settle(store):
    clock = {"now": T0}                                             # all 8 races due (the window widened to an hour)
    t = trader(store, FakeX(), clock=lambda: clock["now"], cfg={"place_from_seconds": 3600, "daily_loss_limit": 300.0})
    t.step()
    assert len(t.x.sent) == 6 and t.open_liability() == 300.0       # 6 x 50 open: -300 at worst, exactly the limit
    led = pd.read_csv(io.BytesIO(store.get(f"greyhound/sp_lay/live/{DAY}/ledger.csv")))
    assert (led.action == "hold").sum() == 2                         # the other 2 held, each logged once
    t.step()
    led = pd.read_csv(io.BytesIO(store.get(f"greyhound/sp_lay/live/{DAY}/ledger.csv")))
    assert (led.action == "hold").sum() == 2 and len(t.x.sent) == 6
    # two of them settle as winners for us (the laid dogs lost): room for more
    t.x.cleared_rows = [{"marketId": t.x.sent[0][0], "customerOrderRef": t.x.sent[0][3], "profit": 10.0},
                        {"marketId": t.x.sent[1][0], "customerOrderRef": t.x.sent[1][3], "profit": 12.0}]
    t.refresh_settled(force=True)
    assert t.settled["net"] == pytest.approx(22.0 * 0.98) and t.open_liability() == 200.0
    t.step()
    assert len(t.x.sent) == 8                                       # 21.56 - 200 - 50 - 50 = -278.44: both held picks go


def test_a_stop_lays_nothing_and_cancels_the_sp_lays_not_yet_off(store):
    flag = {"on": False}
    t = trader(store, FakeX(), kill=lambda: flag["on"])
    t.step()
    assert len(t.x.sent) == 2
    flag["on"] = True
    t.served = served(n_races=12)
    assert t.step() == 0
    assert [c[1]["instructions"][0]["betId"] for c in t.x.cancelled] == ["b1", "b2"]
    t.step()                                                        # cancelled once, not every poll
    assert len(t.x.cancelled) == 2


def test_paper_sends_nothing_and_a_restart_does_not_lay_again(store):
    x = FakeX()
    t = trader(store, x, live=False)
    t.step()
    assert x.sent == [] and sum(1 for r in t.rows if r["action"] == "paper_lay") == 2
    x2 = FakeX()
    t1 = trader(store, x2)
    t1.step()
    t2 = trader(store, x2)                                          # a restarted session, same day
    t2.resume()
    assert t2.step() == 0 and len(x2.sent) == 2 and t2.open_liability() == 100.0
    assert sl.switch_on({"GREYHOUND_SP_LAY": "live"}) and not sl.switch_on({"GREYHOUND_SP_LAY": "paper"})
    assert not sl.switch_on({})


def test_the_reconcile_reads_betfairs_settled_bets_on_our_references(store):
    x = FakeX()
    t = trader(store, x)
    t.step()
    x.cleared_rows = [{"marketId": "1.100", "customerOrderRef": "spl-1121-1.100-6", "profit": -50.0},
                      {"marketId": "1.101", "customerOrderRef": "spl-1121-1.101-6", "profit": 7.5},
                      {"marketId": "1.101", "customerOrderRef": "el-1121-x", "profit": 99.0}]
    r = sl.reconcile(store, x, DAY, 0.02)
    assert r["lays"] == 2 and r["settled"] == 2 and r["profit_gross"] == -42.5
    assert r["profit_net"] == pytest.approx(-50.0 + 7.5 * 0.98)


def test_the_owners_settings_win_and_place_at_gbp50_and_one_gbp1000_limit():
    assert sl.CONFIG["liability"] == 50.0 and sl.CONFIG["place_liability"] == 50.0 and sl.CONFIG["place"]
    assert sl.CONFIG["daily_loss_limit"] == 1000.0
    assert sl.ref_for(DAY, "2.100", 106, "place") == "spl-p1121-2.100-106" and sl.is_ours("spl-p1121-2.1-1", DAY)
    assert sl.is_ours("spl-1121-1.1-1", DAY) and not sl.is_ours("spl-1120-1.1-1", DAY) and not sl.is_ours("el-1121", DAY)


def test_the_place_market_is_the_races_and_the_runner_is_matched_by_name():
    pairs = sl.place_pairs(catalogue(2))
    got = pairs.set_index(["market_id", "selection_id"])
    assert got.loc[("1.100", 6)].tolist() == ["2.100", 106] and len(pairs) == 12
    two = pd.concat([catalogue(1), catalogue(1).query("market_type == 'PLACE'").assign(market_id="2.999")])
    assert sl.place_pairs(two).empty                                # two PLACE markets in a race: which is unknown
    assert sl.place_pairs(None).empty


def test_each_pick_is_laid_to_win_and_to_be_placed_under_one_limit(store):
    x = FakeX()
    t = trader(store, x, cat=catalogue())
    t.load()
    assert t.step() == 4                                            # two races in the window, win and place each
    assert [(m, s, r) for m, s, _, r in x.sent] == [("1.100", 6, "spl-1121-1.100-6"), ("2.100", 106, "spl-p1121-2.100-106"),
                                                    ("1.101", 6, "spl-1121-1.101-6"), ("2.101", 106, "spl-p1121-2.101-106")]
    assert t.open_liability() == 200.0 and t.step() == 0           # never twice
    led = pd.read_csv(io.BytesIO(store.get(f"greyhound/sp_lay/live/{DAY}/ledger.csv")))
    assert led.kind.tolist() == ["win", "place", "win", "place"]
    # the one limit counts both markets: GBP150 lets three GBP50 lays through, the fourth is held
    x2 = FakeX()
    t2 = trader(store, x2, cat=catalogue(), cfg={"daily_loss_limit": 150.0})
    t2.mode = "live-b"
    t2.load()
    t2.step()
    assert len(x2.sent) == 3 and t2.open_liability() == 150.0


def test_a_place_lay_is_skipped_when_its_race_changed_and_the_win_lay_still_goes(store):
    x = FakeX(removed={("2.100", 103)})                             # trap 3 out of race 0's PLACE market only
    t = trader(store, x, cat=catalogue(skip={(1, 2)}))             # trap 2 of race 1 missing from its PLACE market
    t.load()
    t.step()
    assert [m for m, *_ in x.sent] == ["1.100", "1.101"]
    led = pd.read_csv(io.BytesIO(store.get(f"greyhound/sp_lay/live/{DAY}/ledger.csv")), dtype={"market_id": str})
    sk = led[(led.action == "skip") & (led.kind == "place")].set_index("market_id")
    # race 0: trap 3 removed from the PLACE market; race 1: trap 2 not in it, so the race cannot be checked whole
    assert sorted(sk.index) == ["2.100", "2.101"] and "runner 103 REMOVED" in sk.loc["2.100", "note"]


def test_the_reconcile_splits_win_and_place(store):
    x = FakeX()
    t = trader(store, x, cat=catalogue())
    t.load()
    t.step()
    x.cleared_rows = [{"marketId": "1.100", "customerOrderRef": "spl-1121-1.100-6", "profit": 5.0},
                      {"marketId": "2.100", "customerOrderRef": "spl-p1121-2.100-106", "profit": -50.0}]
    r = sl.reconcile(store, x, DAY, 0.02)
    assert r["by_kind"]["win"]["profit_net"] == pytest.approx(4.9) and r["by_kind"]["place"]["profit_net"] == -50.0
    assert r["lays"] == 4 and r["settled"] == 2
