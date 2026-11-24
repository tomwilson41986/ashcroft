"""research/greyhound/paper_vs_live.py: the paper lays and the live lays through one stream, joined on the dog."""
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from research.greyhound import paper_vs_live as pv, stream_replay as sr

OFF = datetime(2026, 11, 13, 14, 0, tzinfo=timezone.utc)
T = OFF - timedelta(minutes=10)
MID = "1.900"
TS = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _line(t, mc):
    return json.dumps({"rt": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                       "m": json.dumps({"op": "mcm", "pt": int(t.timestamp() * 1000), "mc": [mc]})})


def _lay(sid):
    return {"time_utc": TS(T), "mode": "live", "market_id": MID, "selection_id": sid, "track": "Planted",
            "race_time": "14:00", "trap": sid, "action": "lay", "price": 4.0, "size": 10.0, "matched": 0.0,
            "avg_price": 0.0, "f_bsp": 5.0, "mid": 4.6, "status": "SUCCESS", "seconds_to_off": 600}


def _data():
    """Live laid dogs 2 and 3 (only 2 matched); the paper rule laid dogs 2 and 4, dog 2 thirty seconds earlier and
    a tick higher. On the stream 80 trade at 4.0 on every dog behind 20 shown; the SP is 4.4."""
    stake = 10.0 * 4.0 / 4.5
    led = pd.DataFrame([_lay(2), {"time_utc": TS(OFF - timedelta(seconds=5)), "mode": "live", "market_id": MID,
                                   "selection_id": 2, "action": "green", "price": 4.5, "size": stake, "matched": stake,
                                   "avg_price": 4.5, "lay_matched": 10.0, "lay_price": 4.0, "locked_pnl": 1.11,
                                   "seconds_to_off": 5}, _lay(3)])
    defn = lambda status, bsp=None: {
        "marketTime": OFF.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "inPlay": False, "status": status,
        "runners": [{"id": i, "status": "LOSER" if bsp else "ACTIVE", **({"bsp": bsp} if bsp else {})} for i in (2, 3, 4)]}
    rc = lambda **k: [{"id": i, **k} for i in (2, 3, 4)]
    stream = [_line(T - timedelta(seconds=60), {"id": MID, "img": True, "marketDefinition": defn("OPEN"),
                                                "rc": rc(atb=[[4.0, 20]], atl=[[4.1, 5]], trd=[[4.0, 100]])}),
              _line(T + timedelta(seconds=200), {"id": MID, "rc": rc(trd=[[4.0, 180]])}),
              _line(OFF - timedelta(seconds=4), {"id": MID, "rc": rc(atb=[[4.5, 9]])}),
              _line(OFF + timedelta(minutes=2), {"id": MID, "marketDefinition": defn("CLOSED", bsp=4.4)})]
    paper = [sr.SimOrder(MID, 2, "LAY", 4.0, 10.0, T - timedelta(seconds=30)),
             sr.SimOrder(MID, 4, "LAY", 4.0, 10.0, T - timedelta(seconds=30))]
    return {"day": "2026-11-13", "ledger": led, "final": {(MID, float(i)): (4.4, 0) for i in (2, 3, 4)}, "offs": {},
            "paper_orders": paper, "stream_all": stream}


def test_the_two_order_sets_are_read_under_one_fill_and_joined_on_the_dog():
    d = pv.run_day(None, date(2026, 11, 13), loader=lambda day: _data())
    one = 10 * (1 - 4.0 / 4.4)                                   # a GBP10 lay at 4.0 closed at an SP of 4.4
    # the calibrated fill halves the 80 printed: 40, less the 20 ahead: every order fills in full
    assert d["paper_calibrated"]["matched_orders"] == 2 and d["paper_calibrated"]["pnl_at_sp"] == pytest.approx(2 * one, abs=0.01)
    assert d["live_calibrated"]["matched_orders"] == 2                      # the replay fills dog 3; live did not
    assert d["live_actual"]["matched_orders"] == 1 and d["live_actual"]["pnl_at_sp"] == pytest.approx(one, abs=0.01)
    assert d["live_actual"]["roi_on_liability_%"] == pytest.approx(100 * one / 30.0, abs=0.01)
    s = d["sets"]
    assert s["common_as_paper"]["orders"] == 1 and s["paper_only"]["orders"] == 1 and s["live_only"]["orders"] == 1
    assert s["common_orders"]["live_seconds_after_paper_median"] == 30.0
    assert s["common_orders"]["live_price_over_paper_price_median"] == 1.0
    p = pv.pooled([d, {"day": "x", "note": "no data found: no live ledger"}])
    assert p["days"] == 1 and p["paper_own_setting"]["orders"] == 2
    md = pv.markdown({"pooled": p, "days": [{k: v for k, v in d.items() if k != "_frames"}]})
    assert "Paper only" in md and "Live lays, as the ledger settled them" in md


def test_a_day_without_a_ledger_or_paper_lays_says_so():
    assert "no data found" in pv.run_day(None, date(2026, 11, 13), loader=lambda day: {"ledger": None})["note"]
    d = _data()
    d["paper_orders"] = []
    assert "no paper lays" in pv.run_day(None, date(2026, 11, 13), loader=lambda day: d)["note"]
    assert "no data found" in pv.pooled([{"day": "x", "note": "n"}])["note"]
