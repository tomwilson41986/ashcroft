"""The morning models live (morning/live.py; the owner, 4 Nov 2026: "Create two live models; Steamers, Quick
Returners", WIN only, "half stakes", the live trader's horses left out, no order before 10:00 UK). Fixtures only: a
stand-in exchange records every order; nothing reaches Betfair."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from morning import live as ML
from morning import select as S
from trading.exchange import Book, Fill, Market, Quote, tick_up

DAY = date(2026, 11, 5)                                   # GMT: 10:00 UK is 10:00 UTC
OFF = datetime(2026, 11, 5, 13, 30, tzinfo=timezone.utc)
MID = "1.900"
A, B, C = 11, 12, 13


class Stand:
    """An exchange that fills every back at the best price offered at or above the limit, and records the orders."""

    def __init__(self, backs=None, status="OPEN"):
        self.orders, self.calls = [], []
        self.backs = backs or {A: [(6.0, 500.0)], B: [(2.0, 500.0)], C: [(80.0, 50.0)]}
        self.status = status

    def books(self, ids, with_sp=False):
        self.calls.append(("books", tuple(ids)))
        return {MID: Book(MID, self.status, False, 1e5, {s: Quote(s, back=list(v)) for s, v in self.backs.items()})}

    def back(self, market_id, selection_id, price, size, ref, min_fill=2.0):
        self.orders.append({"mid": market_id, "sid": selection_id, "price": price, "size": size, "ref": ref,
                            "min_fill": min_fill})
        best = self.backs[selection_id][0][0]
        return Fill(market_id, selection_id, "BACK", "LIMIT", price, size, matched=size, avg_price=best,
                    bet_id=f"b{len(self.orders)}", status="SUCCESS")


def _sel(rows):
    return pd.DataFrame([{"model": m, "market_id": MID, "selection_id": s, "horse_name": f"H{s}", "pred": 0.1}
                         for m, s in rows])


def _session(rows, x=None, live_ledger=None, stopped=lambda: False, off=OFF):
    m = Market(MID, "Kempton", "GB", off, runners={A: "HA", B: "HB", C: "HC"})
    return ML.MorningSession(x or Stand(), DAY, _sel(rows), [m], "live",
                             live_ledger=(lambda: live_ledger), stopped=stopped)


def _clock(start):
    t = {"now": start}

    def now():
        return t["now"]

    def sleep(s):
        t["now"] += timedelta(seconds=s)
    return now, sleep


def test_no_order_before_ten_uk():
    s = _session([("steamer", A)])
    now, sleep = _clock(datetime(2026, 11, 5, 8, 5, tzinfo=timezone.utc))
    led = ML.run(s, now=now, sleep=sleep)
    assert s.x.orders and led[0]["event"] == "back"
    assert pd.Timestamp(led[0]["ts"]) == pd.Timestamp("2026-11-05T10:00:00Z")
    # and nothing at all, not even a book read, at any minute before
    s2 = _session([("steamer", A)])
    t = datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc)
    while t < ML.uk_at(DAY, ML.ENTRY_UK):
        s2.step(t)
        t += timedelta(minutes=1)
    assert s2.x.calls == [] and s2.x.orders == [] and s2.ledger == []


def test_ten_uk_is_nine_utc_in_summer():
    assert ML.uk_at(date(2026, 7, 1), ML.ENTRY_UK) == datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)


def test_half_stakes_to_win_fifty_at_most_a_hundred_at_least_one():
    assert ML.TO_WIN == 50.0 and ML.MAX_STAKE == 100.0 and ML.DAY_LIMIT == 2000.0
    assert ML.to_win_stake(6.0) == pytest.approx(10.0)
    assert ML.to_win_stake(1.3) == pytest.approx(100.0)            # 166.67 capped
    assert ML.to_win_stake(80.0) == pytest.approx(0.63)
    s = _session([("steamer", A), ("steamer", B), ("steamer", C)])
    s.step(ML.uk_at(DAY, ML.ENTRY_UK))
    ev = {r["selection_id"]: r for r in s.ledger}
    assert ev[A]["event"] == "back" and ev[A]["asked"] == pytest.approx(10.0)
    assert ev[B]["event"] == "back" and ev[B]["asked"] == pytest.approx(50.0)
    assert ev[C]["event"] == "below_min"                            # GBP0.63: under Betfair's GBP1, not placed
    assert {o["sid"] for o in s.x.orders} == {A, B}


def test_the_limit_walks_the_ladder_to_95_percent_on_a_betfair_tick():
    assert ML.limit_price(6.0) == pytest.approx(5.7)
    assert ML.limit_price(2.0) == pytest.approx(1.9)
    assert ML.limit_price(13.5) == pytest.approx(tick_up(13.5 * 0.95)) == pytest.approx(13.0)
    assert ML.limit_price(1.01) == pytest.approx(1.01)
    for p in (1.37, 2.46, 3.35, 4.7, 7.8, 15.5, 26.0, 44.0, 95.0, 310.0):
        lim = ML.limit_price(p)
        assert lim == tick_up(lim) and p * 0.95 - 1e-9 <= lim <= p


def test_the_live_traders_horses_are_left_alone():
    led = pd.DataFrame({"event": ["back", "back", "skip"], "market_id": [MID, MID, MID], "selection_id": [A, B, C],
                        "matched": [12.5, 0.0, 0.0]}).to_csv(index=False).encode()
    s = _session([("steamer", A), ("quick_return", A), ("steamer", B)], live_ledger=led)
    s.step(ML.uk_at(DAY, ML.ENTRY_UK))
    ev = {(r["model"], r["selection_id"]): r["event"] for r in s.ledger}
    assert ev[("steamer", A)] == ev[("quick_return", A)] == "skipped_live"
    assert ev[("steamer", B)] == "back"                             # an unmatched live order does not count
    assert [o["sid"] for o in s.x.orders] == [B]


def test_references_are_the_models_own_never_the_live_traders():
    s = _session([("quick_return", A), ("steamer", A)])
    s.step(ML.uk_at(DAY, ML.ENTRY_UK))
    refs = [o["ref"] for o in s.x.orders]
    assert refs == [f"mes1105{A}", f"meq1105{A}"]                   # the steamer's order first
    assert not any(r.startswith(ML.LIVE_TRADER_PREFIX) for r in refs)
    assert all(len(r) <= 32 for r in refs)
    assert ML.ours({"customerOrderRef": f"meq1105{A}"}, "quick_return", DAY)
    assert not ML.ours({"customerOrderRef": f"ash1105{A}"}, "quick_return", DAY)


def test_the_day_limit_is_per_model():
    s = _session([("steamer", A), ("quick_return", A)])
    s.staked["steamer"] = ML.DAY_LIMIT - 5.0
    s.step(ML.uk_at(DAY, ML.ENTRY_UK))
    ev = {r["model"]: r["event"] for r in s.ledger}
    assert ev == {"steamer": "day_limit", "quick_return": "back"}


def test_no_book_is_retried_to_quarter_past_then_dropped():
    x = Stand(status="SUSPENDED")
    s = _session([("steamer", A)], x=x)
    now, sleep = _clock(ML.uk_at(DAY, ML.ENTRY_UK))
    led = ML.run(s, now=now, sleep=sleep)
    assert x.orders == [] and [r["event"] for r in led] == ["no_book"]
    assert sum(c[0] == "books" for c in x.calls) == 16                # 10:00 to 10:15, each minute


def test_stop_places_nothing_and_a_race_about_to_go_is_not_bet():
    s = _session([("steamer", A)], stopped=lambda: True)
    s.step(ML.uk_at(DAY, ML.ENTRY_UK))
    assert s.x.orders == [] and s.ledger[0]["event"] == "stopped"
    s = _session([("steamer", A)], off=ML.uk_at(DAY, ML.ENTRY_UK) + timedelta(minutes=1))
    s.step(ML.uk_at(DAY, ML.ENTRY_UK))
    assert s.x.orders == [] and s.ledger[0]["event"] == "too_late"


def test_nothing_after_quarter_past_ten():
    s = _session([("steamer", A)])
    s.step(ML.uk_at(DAY, ML.RETRY_UNTIL_UK) + timedelta(minutes=1))
    assert s.x.calls == [] and s.ledger == []


def test_live_needs_the_owners_switch():
    assert ML.live_switch_on({"MORNING_LIVE": "yes"})
    assert not ML.live_switch_on({"MORNING_LIVE": "no"}) and not ML.live_switch_on({})
    assert not ML.live_switch_on({"TRADING_LIVE": "yes"})           # the live trader's switch is not this one


def test_win_only_no_place_orders_and_no_lay():
    src = Path(ML.__file__).read_text()
    assert "lay_at_bsp(" not in src and "MARKET_ON_CLOSE" not in src and '"place"' not in src


def test_the_card_maps_to_the_tables_shape():
    assert S.split_claim("Young, Lauren(5)") == ("Young, Lauren", 5.0)
    assert S.split_claim("Hart, Jason") == ("Hart, Jason", 0.0)
    assert S.time12("14:05") == "2:05" and S.time12("12:05") == "12:05" and S.time12("1:30") == "1:30"
    names = pd.Series(["Racing TV Handicap Hurdle", "Novices' Chase", "Open NH Flat Race", "Maiden Stakes",
                       "Classified Stakes", "Bumper", "WTW Willis Flat Race", "Fri 30th Oct 2026"])
    going = pd.Series(["Soft", "Good", "Heavy", "Good To Firm", "Standard To Slow", "Soft", "Yielding", "Good"])
    assert S.card_code(names, going).tolist() == ["hurdle", "chase", "nhflat", "flat", "aw", "nhflat", "nhflat",
                                                  "unknown"]
    rows = pd.DataFrame({"horse_name": ["X", "Y", "Z", "W"], "rk": ["r1", "r1", "r1", "r2"],
                         "code": ["unknown", "unknown", "unknown", "unknown"]})
    hist = pd.DataFrame({"horse_name": ["X", "Y", "Z", "X"], "code": ["chase", "chase", "flat", "hurdle"],
                         "date": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-09"])})
    # last runs: X hurdle (its later run), Y chase, Z flat: an even split stays unknown; W has no history
    assert S.settle_unknown_codes(rows, hist).tolist() == ["unknown", "unknown", "unknown", "unknown"]
    hist.loc[3, "code"] = "chase"
    assert S.settle_unknown_codes(rows, hist).tolist() == ["chase", "chase", "chase", "unknown"]


def test_the_card_joins_the_history_in_the_tables_shape_and_order():
    from research.paper import morning_features as MF
    hist = pd.DataFrame({
        "race_date": ["2026-11-01", "2026-11-03"], "race_time": ["2:05", "1:30"], "track": ["Kempton", "Lingfield"],
        "race_name": ["Handicap", "Maiden Stakes"], "race_class": ["Class 5", "Class 4"], "horse_name": ["HA", "HB"],
        "trainer": ["T, A", "T, B"], "jockey_name": ["J, A", "J, B"], "jockeys_claim": [0.0, 3.0],
        "horse_age": [4.0, 3.0], "official_rating": [70.0, 0.0], "days_since_lr": [10.0, 20.0],
        "number_of_runners": [9.0, 8.0], "headgear": [None, "TT"], "placing_numerical": [1.0, 4.0],
        "bsp": [5.0, 9.0], "won": [1.0, 0.0], "placed": [1.0, 0.0], "morningwap": [6.0, 8.0],
        "morning_vol": [100.0, 80.0], "code": ["aw", "aw"], "hcap": [True, False], "race_has_winner": [True, True]})
    hist["date"] = pd.to_datetime(hist["race_date"])
    hist["rk"] = hist["race_date"] + "|" + hist["track"] + "|" + hist["race_time"]
    hist["fin"] = hist["placing_numerical"]
    hist["bsp_rank"] = 1.0
    hist["field"] = 1
    card = pd.DataFrame({"race_time": ["14:05"], "track": ["Kempton"], "race_name": ["Racing TV Handicap"],
                         "race_class": ["Class 5"], "horse_name": ["HA"], "trainer": ["T, A"],
                         "jockey_name": ["Young, Lauren(5)"], "horse_age": ["4"], "official_rating": ["72"],
                         "days_since_lr": ["3.0"], "number_of_runners": ["9"], "headgear": [None],
                         "going_description": ["Standard"]})
    fr = S.history_with_card(hist, card, "2026-11-04")
    assert fr["date"].dtype == hist["date"].dtype and fr["hcap"].dtype == bool
    t = fr[fr["race_date"] == "2026-11-04"].iloc[0]
    assert t["race_time"] == "2:05" and t["jockey_name"] == "Young, Lauren" and t["jockeys_claim"] == 5.0
    assert t["code"] == "aw" and bool(t["hcap"]) and pd.isna(t["bsp"]) and pd.isna(t["won"])
    x = MF.build(fr)
    today = x[x["race_date"] == "2026-11-04"].iloc[0]
    assert today["dslr"] == 3 and today["prev_fin"] == 1.0 and today["prev_bsp"] == 5.0 and today["runs_held"] == 1
    assert bool(MF.rule_qr(x[x["race_date"] == "2026-11-04"]).iloc[0]) is True
