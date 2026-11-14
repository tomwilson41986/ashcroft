"""research/greyhound/fill_recon.py on a planted live ledger and planted books: a phantom fill counted as one, a common
fill's price gap put in the price part, a missed fill, a partial fill's excess, the parts summing to the gap, and days
without books or ledgers reported rather than dropped. Small hand-built fixtures, no real data."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from research.greyhound import fill_recon as fr

OFF = datetime(2026, 11, 13, 14, 0, tzinfo=timezone.utc)
T = OFF - timedelta(minutes=10)
MID = "1.900"
TS = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")


def lay(sid, price=4.0, size=10.0, matched=0.0):
    return {"time_utc": TS(T), "mode": "live", "market_id": MID, "selection_id": sid, "track": "Planted",
            "race_time": "14:00", "trap": sid, "action": "lay", "price": price, "size": size, "matched": matched,
            "avg_price": 0.0, "f_bsp": 5.0, "mid": 4.6, "status": "SUCCESS", "seconds_to_off": 600,
            "liability": size * (price - 1)}


def green(sid, lay_matched, lay_price, back_price):
    stake = lay_matched * lay_price / back_price
    locked = lay_matched * (1 - lay_price / back_price)
    return {"time_utc": TS(OFF - timedelta(seconds=5)), "mode": "live", "market_id": MID, "selection_id": sid,
            "action": "green", "price": back_price, "size": stake, "matched": stake, "avg_price": back_price,
            "lay_matched": lay_matched, "lay_price": lay_price, "locked_pnl": round(locked, 6), "seconds_to_off": 5}


def ledger():
    return pd.DataFrame([
        lay(1),                                   # live: never matched
        lay(2), green(2, 10.0, 4.0, 4.5),         # live: matched at 4.0, greened at 4.5
        lay(3), green(3, 10.0, 4.0, 5.0),         # live: matched; the replay never reaches it
        lay(4), green(4, 6.0, 4.0, 4.4),          # live: 6 of 10 matched
    ])


def book(sid, t, back1, lay1, lt):
    return {"market_id": MID, "selection_id": sid, "polled": t, "back1": back1, "lay1": lay1, "last_traded": lt}


def books():
    rows = []
    # 1: the book at placement does not offer 4.0; a later trade at 3.9 fills the replay: a phantom fill
    rows += [book(1, T - timedelta(seconds=30), 4.2, 4.3, 4.3), book(1, T + timedelta(seconds=60), 4.0, 4.1, 3.9),
             book(1, T + timedelta(seconds=300), 4.4, 4.5, 4.4)]
    # 2: the book at placement offers 3.8: the replay takes it at once, at 3.8 where live got 4.0
    rows += [book(2, T - timedelta(seconds=30), 3.7, 3.8, 3.8), book(2, T + timedelta(seconds=300), 4.5, 4.6, 4.5)]
    # 3: the price never trades down to 4.0: the replay never fills it, live did
    rows += [book(3, T - timedelta(seconds=30), 4.5, 4.6, 4.6), book(3, T + timedelta(seconds=300), 5.0, 5.1, 4.7)]
    # 4: a later trade at 4.0 fills the replay in full at 4.0
    rows += [book(4, T - timedelta(seconds=30), 4.2, 4.3, 4.3), book(4, T + timedelta(seconds=120), 3.9, 4.0, 4.0),
             book(4, T + timedelta(seconds=300), 4.4, 4.5, 4.4)]
    b = pd.DataFrame(rows)
    b["polled"] = pd.to_datetime(b.polled, utc=True)
    return b


def data(day="2026-11-13", with_books=True):
    final = {(MID, 1.0): (6.0, 0), (MID, 2.0): (4.4, 0), (MID, 3.0): (5.5, 1), (MID, 4.0): (4.6, 0)}
    return {"day": day, "ledger": ledger(), "reconcile": {"settled": 1, "markets_laid": 1, "betfair_profit_gross": 1.0},
            "recorder_books": books() if with_books else None, "live_books": None, "traded": None, "stream": None,
            "final": final, "offs": {}}


def test_the_live_orders_carry_what_live_gave_them():
    o = fr.live_orders(ledger(), data()["final"]).set_index("selection_id")
    assert list(o.index) == [1.0, 2.0, 3.0, 4.0]
    assert o.loc[1.0, "live_matched"] == 0 and o.loc[1.0, "live_pnl"] == 0
    assert o.loc[2.0, "live_matched"] == 10 and o.loc[2.0, "live_avg"] == 4.0 and o.loc[2.0, "live_green"] == 4.5
    assert o.loc[2.0, "live_pnl"] == pytest.approx(10 * (1 - 4 / 4.5), abs=1e-5)
    assert o.loc[4.0, "live_share"] == pytest.approx(0.6)
    assert o.loc[1.0, "minutes_to_off"] == pytest.approx(10.0) and o.loc[1.0, "band"] == "3-5"


def test_a_planted_phantom_fill_is_counted_as_one():
    rep, d = fr.reconcile_day(data())
    x = fr.decompose(d, "traded_retimed").set_index("selection_id")
    assert x.loc[1.0, "kind"] == "phantom"
    assert x.loc[1.0, "R"] == pytest.approx(10 * (1 - 4.0 / 4.4), abs=1e-6)
    assert x.loc[1.0, "gap_phantom"] == pytest.approx(x.loc[1.0, "R"])
    p = rep["models"]["traded_retimed"]
    assert p["phantom_fills"] == 1 and p["phantom_replay_pnl"] == pytest.approx(round(10 * (1 - 4 / 4.4), 2))
    assert p["missed_fills"] == 1 and p["missed_live_pnl"] == pytest.approx(round(10 * (1 - 4 / 5.0), 2))


def test_a_common_fills_price_gap_is_put_in_the_price_part():
    _, d = fr.reconcile_day(data())
    x = fr.decompose(d, "traded_retimed").set_index("selection_id")
    r = x.loc[2.0]
    assert r.kind == "both" and r.traded_retimed_avg == pytest.approx(3.8)
    assert r.gap_price == pytest.approx(10 * (4.0 - 3.8) / 4.5, abs=1e-6)     # the price, at live's green-up
    assert r.gap_green == pytest.approx(0.0, abs=1e-6)                          # both greened at 4.5
    assert r.gap_phantom == 0 and r.gap_missed == 0
    assert r.gap == pytest.approx(r.gap_price, abs=1e-6)


def test_a_partial_live_fill_puts_the_replays_excess_in_the_phantom_part():
    _, d = fr.reconcile_day(data())
    r = fr.decompose(d, "traded_retimed").set_index("selection_id").loc[4.0]
    assert r.kind == "both" and r.traded_retimed_matched == 10
    assert r.gap_phantom == pytest.approx(4 * (1 - 4.0 / 4.4), abs=1e-6)
    assert r.gap_price == pytest.approx(0.0, abs=1e-6) and r.gap_green == pytest.approx(0.0, abs=1e-6)


def test_the_parts_sum_to_the_whole_gap():
    rep, d = fr.reconcile_day(data())
    p = rep["models"]["traded_retimed"]
    assert p["pnl_unknown"] == 0
    assert sum(p["gap_parts"].values()) == pytest.approx(p["gap"], abs=0.02)
    assert p["gap"] == pytest.approx(p["replay_pnl"] - p["live_pnl"], abs=0.02)
    x = fr.decompose(d, "through_retimed")
    parts = x[[f"gap_{k}" for k in ("phantom", "missed", "price", "green")]].sum(axis=1)
    assert np.allclose(parts, x.gap)


def test_fill_rates_and_adverse_selection():
    rep, _ = fr.reconcile_day(data())
    p = rep["models"]["traded_retimed"]
    assert p["live_fill_rate%"] == pytest.approx(100 * 26 / 40)            # 0 + 10 + 10 + 6 of 40
    assert p["replay_fill_rate%"] == pytest.approx(100 * 30 / 40)          # 10 + 10 + 0 + 10
    a = p["adverse_selection"]
    assert a["live_unmatched"]["n"] == 1 and a["live_unmatched"]["mean"] == pytest.approx(round(np.log(6.0 / 4.0), 4))
    assert a["replay_only"]["n"] == 1


def test_days_without_books_or_ledgers_are_reported_not_dropped():
    loads = {date(2026, 11, 3): data("2026-11-03", with_books=False), date(2026, 11, 4): {"day": "2026-11-04",
                                                                                           "ledger": None},
             date(2026, 11, 5): data("2026-11-05")}
    r, orders = fr.run(None, date(2026, 11, 3), date(2026, 11, 5), loader=lambda d: loads[d])
    assert [d["day"] for d in r["days"]] == ["2026-11-03", "2026-11-04", "2026-11-05"]
    assert r["days_without_books"] == ["2026-11-03"] and r["days_without_ledger"] == ["2026-11-04"]
    assert "no recorded books" in r["days"][0]["note"]
    # the bookless day's orders stay in the count, uncovered: the model's coverage shows it
    p = r["pooled"]["traded_retimed"]
    assert p["orders"] == 8 and p["covered"] == 4 and p["coverage"] == 0.5
    assert r["verdicts"]["traded_retimed"].startswith("not judged")
    assert r["verdicts"]["queue_stream"].startswith("not judged")
    md = fr.markdown(r)
    assert "2026-11-04" in md and "no live ledger" in md


def test_the_registered_rule():
    base = {"orders": 100, "covered": 95, "coverage": 0.95}
    assert fr.verdict({**base, "gap": 49.0, "fill_gap_points": -4.9}) == "TRUSTED"
    assert fr.verdict({**base, "gap": -51.0, "fill_gap_points": 0.0}) == "NOT TRUSTED"
    assert fr.verdict({**base, "gap": 0.0, "fill_gap_points": 5.1}) == "NOT TRUSTED"
    assert fr.verdict({**base, "coverage": 0.8, "gap": 0.0, "fill_gap_points": 0.0}).startswith("not judged")


def _line(t, mc):
    import json
    return json.dumps({"rt": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                       "m": json.dumps({"op": "mcm", "pt": int(t.timestamp() * 1000), "mc": [mc]})})


def test_the_queue_aware_models_run_on_the_live_key_books_and_the_stream():
    led = pd.DataFrame([lay(2), green(2, 10.0, 4.0, 4.5)])
    lb = pd.DataFrame([{**book(2, T - timedelta(seconds=30), 4.0, 4.1, 4.1), "back1_size": 20.0, "lay1_size": 5.0},
                       {**book(2, T + timedelta(seconds=300), 4.5, 4.6, 4.0), "back1_size": 9.0, "lay1_size": 5.0}])
    lb["polled"] = pd.to_datetime(lb.polled, utc=True)
    # the ladder: 20 ahead at 4.0 when placed, then 35 traded at 4.0: 15 past the queue, our 10 filled in full
    lad = pd.DataFrame([{"polled_utc": TS(T - timedelta(seconds=30)), "market_id": MID, "selection_id": 2, "price": 4.0,
                         "traded": 100.0},
                        {"polled_utc": TS(T + timedelta(seconds=200)), "market_id": MID, "selection_id": 2,
                         "price": 4.0, "traded": 135.0}])
    defn = {"marketTime": OFF.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "inPlay": False, "status": "OPEN",
            "runners": [{"id": 2, "status": "ACTIVE"}]}
    stream = [_line(T - timedelta(seconds=30), {"id": MID, "img": True, "marketDefinition": defn,
                                                "rc": [{"id": 2, "atb": [[4.0, 20]], "atl": [[4.1, 5]],
                                                        "trd": [[4.0, 100]], "ltp": 4.0}]}),
              _line(T + timedelta(seconds=200), {"id": MID, "rc": [{"id": 2, "trd": [[4.0, 135]], "atb": [[4.5, 9]]}]}),
              _line(OFF - timedelta(seconds=4), {"id": MID, "rc": [{"id": 2, "atb": [[4.5, 9]]}]})]
    d = {"day": "2026-11-13", "ledger": led, "recorder_books": None, "live_books": lb, "traded": lad, "stream": stream,
         "final": {(MID, 2.0): (4.4, 0)}, "offs": {}}
    rep, x = fr.reconcile_day(d)
    r = x.iloc[0]
    assert r.queue_book_covered and r.queue_book_matched == pytest.approx(10.0)
    assert r.queue_stream_covered and r.queue_stream_matched == pytest.approx(10.0)
    assert r.queue_stream_green == pytest.approx(4.5)
    assert rep["models"]["queue_stream"]["gap"] == pytest.approx(0.0, abs=0.01)    # same fill, same green-up as live
    assert not r.traded_retimed_covered


def test_a_greened_lay_is_settled_by_the_result_when_known():
    """An over-covered green-up (backs worth more than the lay) is not flat: with the result it takes the outcome
    that happened, without it the worse of the two the trader locked."""
    import pandas as pd
    from research.greyhound import fill_recon as fr
    S, a = 31.25, 4.2
    close = pd.DataFrame({"action": ["green"], "matched": [42.69], "avg_price": [3.6106], "locked_pnl": [-11.44]})
    lose, how = fr._live_pnl(close, S, a, 4.4, 0)
    win, _ = fr._live_pnl(close, S, a, 4.4, 1)
    assert how == "greened" and abs(lose - (S - 42.69)) < 1e-9
    assert abs(win - (-S * (a - 1) + 42.69 * 2.6106)) < 1e-9 and win > 0
    assert fr._live_pnl(close, S, a, None, None)[0] == -11.44
