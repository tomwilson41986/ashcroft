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
    pnl = [0.04, 0.06] * 20                                   # a steady 5% a race, with a little spread
    tr = pd.DataFrame({"rule": "r", "market": [f"m{i}" for i in range(40)], "pnl": pnl,
                       "day": [f"2026-11-{5 + i % 3:02d}" for i in range(40)]})
    sc = lab.score(tr).iloc[0]
    assert sc.trades == 40 and sc.days == 3 and sc["mean_pnl%"] == 5.0 and not sc.candidate
    tr["day"] = [f"2026-11-{5 + i % 10:02d}" for i in range(40)]
    sc = lab.score(tr).iloc[0]
    assert sc.candidate and sc.dsr >= 0.95 and sc.n_trials == 1


def test_a_candidate_must_clear_the_deflated_sharpe_against_every_rule_scored():
    """The lower bound alone lets about one rule in twenty through by luck; the DSR against all the rules is the bar."""
    import numpy as np
    from research.overfit import TrialLedger
    rng = np.random.default_rng(7)
    days = [f"2026-11-{5 + i % 10:02d}" for i in range(60)]
    rows = []
    for k in range(43):                                       # 43 rules, none with an edge
        rows.append(pd.DataFrame({"rule": f"r{k}", "market": [f"m{i}" for i in range(60)], "day": days,
                                  "pnl": rng.normal(0.0, 0.3, 60)}))
    sc = lab.score(pd.concat(rows, ignore_index=True))
    assert (sc.n_trials == 43).all() and not sc.candidate.any()
    led = TrialLedger(None)
    for k in range(100):                                      # a ledger that has seen more rules raises the count
        led.record(f"old{k}", sharpe=float(rng.normal(0, 0.1)))
    sc = lab.score(rows[0], ledger=led)
    assert sc.iloc[0].n_trials == 101
