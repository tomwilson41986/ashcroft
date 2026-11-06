"""research/greyhound/stream_lab.py on hand-built stream messages: the books read at the checkpoints as they stood,
each rule's trades taken at the price on offer and closed at the SP, the score."""
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from research.greyhound import stream_lab as lab

OFF = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)


def line(t, mc):
    return json.dumps({"rt": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                       "m": json.dumps({"op": "mcm", "pt": int(t.timestamp() * 1000), "mc": [mc]})})


def defn(status="OPEN", bsp=None, result=None):
    def r(i):
        d = {"id": i, "status": (result or {}).get(i, "ACTIVE")}
        if bsp:
            d["bsp"] = bsp[i]
        return d
    return {"marketTime": OFF.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "inPlay": False, "status": status,
            "marketType": "WIN", "venue": "Romford", "runners": [r(1), r(2)]}


def book(t, a, b):
    """Runner 1 and 2: (back, lay, spn)."""
    return line(t, {"id": "1.1", "img": True, "marketDefinition": defn(), "rc": [
        {"id": 1, "atb": [[a[0], 50]], "atl": [[a[1], 50]], "spn": a[2]},
        {"id": 2, "atb": [[b[0], 50]], "atl": [[b[1], 50]], "spn": b[2]}]})


def test_the_checkpoints_and_the_trades_closed_at_the_sp():
    lines = [book(OFF - timedelta(seconds=200), (2.0, 2.02, 2.0), (5.0, 5.1, 5.0)),
             # by 60 s out runner 1 has shortened 10% (mid 2.01 -> 1.81) and runner 2 shows a back above its projection
             book(OFF - timedelta(seconds=100), (1.8, 1.82, 1.8), (5.6, 5.7, 5.0)),
             line(OFF - timedelta(seconds=1), {"id": "1.1", "rc": [{"id": 1, "ltp": 1.8}]}),
             line(OFF + timedelta(minutes=3), {"id": "1.1", "marketDefinition": defn(
                 "CLOSED", bsp={1: 1.7, 2: 5.2}, result={1: "WINNER", 2: "LOSER"})})]
    s = lab.snapshots(lines)
    assert set(s.check) == {180, 120, 60, 30, 10}
    r180 = s[(s.check == 180) & (s.sid == 1)].iloc[0]
    assert r180.back1 == 2.0 and r180.bsp == 1.7 and r180.status == "WINNER"
    tr = lab.trades(s)
    g = tr[(tr.rule == "gap_back_60_0.1") & (tr.sid == 2)].iloc[0]           # 5.6 / 5.0 - 1 = 12% over the projection
    assert g.price == 5.6 and g.pnl == pytest.approx((5.6 / 5.2 - 1) * (1 - lab.COMM))
    m = tr[(tr.rule == "mom_back_60_0.05") & (tr.sid == 1)].iloc[0]          # shortened: backed at 1.8, SP 1.7
    assert m.price == 1.8 and m.pnl == pytest.approx((1.8 / 1.7 - 1) * (1 - lab.COMM))
    rl = tr[(tr.rule == "rev_lay_60_0.05") & (tr.sid == 1)].iloc[0]          # laid at 1.82, SP 1.7: a loss
    assert rl.pnl == pytest.approx(1 - 1.82 / 1.7)
    assert set(tr[tr.rule == "fav_lay_10"].sid) == {1}


def test_the_score_pools_days_and_needs_ten_for_a_candidate():
    tr = pd.DataFrame({"rule": "r", "market": [f"m{i}" for i in range(40)], "pnl": 0.05,
                       "day": [f"2026-11-{5 + i % 3:02d}" for i in range(40)]})
    sc = lab.score(tr).iloc[0]
    assert sc.trades == 40 and sc.days == 3 and sc["mean_pnl%"] == 5.0 and not sc.candidate
    tr["day"] = [f"2026-11-{5 + i % 10:02d}" for i in range(40)]
    assert lab.score(tr).iloc[0].candidate


def test_the_forward_tests_lays_are_replayed_and_scored(monkeypatch):
    """The forward test's lays, built as its own table builds them, replayed through the stream."""
    from greyhound import laytest as lt
    from research.greyhound import limit_study as ls, tradeout_study as ts
    t0 = OFF - timedelta(minutes=5)
    rows = pd.DataFrame({"market_id": ["1.1"], "selection_id": [2], "polled": [t0 + timedelta(seconds=1)]})
    monkeypatch.setattr(lt, "day_fit", lambda store, day: {"fit": 1})
    monkeypatch.setattr(lt, "today_rows", lambda store, day: (rows, "test"))
    monkeypatch.setattr(ts, "predict", lambda r, f: ([0.2], 1.0))
    monkeypatch.setattr(ls, "orders", lambda r: r.assign(limit=5.0))
    monkeypatch.setattr(lt, "_select", lambda o, rule: o)
    monkeypatch.setattr(lt, "intraday", lambda store, day: {"races": 1, "lays_placed": 1, "matched": 1,
                                                            "liability": 20.0, "pnl": 1.0, "roi%": 5.0})
    lines = [book(t0, (1.5, 1.52, 1.5), (5.0, 5.2, 5.0)),
             line(t0 + timedelta(seconds=5), {"id": "1.1", "rc": [{"id": 2, "trd": [[4.9, 3]]}]}),   # through: filled
             line(OFF - timedelta(seconds=4), {"id": "1.1", "rc": [{"id": 2, "atb": [[5.5, 9]]}]}),
             line(OFF + timedelta(minutes=3), {"id": "1.1", "marketDefinition": defn(
                 "CLOSED", bsp={1: 1.5, 2: 6.0}, result={1: "WINNER", 2: "LOSER"})})]
    r = lab.forward_check(lines, None, date(2026, 11, 5))
    x = r["stream_x1.0"]
    assert r["forward_test_own"]["roi%"] == 5.0 and x["matched"] == 1 and x["filled_through"] == 1
    stake = 20.0 / 4.0                                                  # GBP20 liability at 5.0
    assert x["matched_liability"] == pytest.approx(20.0)
    assert x["roi_green%"] == pytest.approx(100 * stake * (1 - 5.0 / 5.5) * (1 - lab.COMM) / 20.0, abs=0.01)
    assert x["roi_sp%"] == pytest.approx(100 * stake * (1 - 5.0 / 6.0) * (1 - lab.COMM) / 20.0, abs=0.01)
