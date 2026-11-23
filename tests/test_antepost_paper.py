"""The ante-post resting-order paper test (antepost/paper.py). The frames are format fixtures written for the test;
no figure from them is used anywhere."""

import numpy as np
import pandas as pd

from antepost import paper as pp


def test_the_ladder():
    assert pp.tick_down(2.0) == 1.99 and pp.tick_down(3.0) == 2.98 and pp.tick_down(10.0) == 9.8
    assert pp.tick_down(5.0) == 4.9 and pp.tick_down(1.01) == 1.01 and pp.tick_down(1000.0) == 990.0
    assert pp.to_tick(7.03) == 7.0 and pp.to_tick(23.7) == 23.0 and pp.to_tick(4.07) == 4.0 and pp.to_tick(0.5) == 1.01


def _snap(day, back1, lay1, ltp, matched, days_out, p_ran=0.8, active=True, marked=0):
    return {"day": day, "market_id": "m", "selection_id": 1, "back1": back1, "lay1": lay1,
            "log_ltp": np.log(ltp) if ltp else np.nan, "has_price": int(bool(ltp)), "market_total_matched": matched,
            "days_out": days_out, "p_ran": p_ran, "active": active, "marked_out": marked}


def test_a_resting_back_fills_only_when_traded_through_and_the_lay_rests_at_the_target():
    snap = pd.DataFrame([
        _snap("2026-12-01", 9.6, 10.5, 10.0, 100.0, 40),         # rest a back at 10.0 (a tick under 10.5)
        _snap("2026-12-02", 9.8, 10.5, 10.2, 100.0, 39),         # a new price but no new money matched: no fill
        _snap("2026-12-03", 10.5, 11.0, 10.5, 180.0, 38),        # traded above 10.0, money matched: filled at 10.0
        _snap("2026-12-10", 6.6, 6.8, 6.6, 300.0, 31),           # traded below the 7.0 target: closed at 7.0
    ])
    t = pp.replay(snap)
    [r] = t.to_dict("records")
    assert r["price"] == 10.0 and r["filled_day"] == "2026-12-03" and r["status"] == "target" and r["exit_price"] == 7.0


def test_no_target_means_crossing_at_three_days_out_and_a_mark_out_means_stuck():
    base = [_snap("2026-12-01", 9.6, 10.5, 10.0, 100.0, 40), _snap("2026-12-03", 10.5, 11.0, 10.5, 180.0, 38)]
    crossed = pp.replay(pd.DataFrame(base + [_snap("2026-12-20", 11.0, 12.0, 11.5, 400.0, 2.5)])).iloc[0]
    assert crossed["status"] == "crossed" and crossed["exit_price"] == 12.0
    stuck = pp.replay(pd.DataFrame(base + [_snap("2026-12-10", None, None, None, 400.0, 31, marked=1)])).iloc[0]
    assert stuck["status"] == "stuck"


def test_an_unfilled_order_is_cancelled_after_its_days():
    snap = pd.DataFrame([_snap("2026-12-01", 9.6, 10.5, 10.0, 100.0, 40),
                         _snap("2026-12-08", 9.6, 10.5, 10.0, 100.0, 33)])
    assert pp.replay(snap).iloc[0]["status"] == "cancelled"


def test_settlement_greens_closed_trades_and_settles_stuck_ones_by_the_race():
    t = pd.DataFrame([
        {"market_id": "m", "selection_id": 1, "status": "target", "price": 10.0, "exit_price": 7.0},
        {"market_id": "m", "selection_id": 2, "status": "stuck", "price": 10.0, "exit_price": np.nan},
        {"market_id": "m", "selection_id": 3, "status": "stuck", "price": 10.0, "exit_price": np.nan},
        {"market_id": "m", "selection_id": 4, "status": "stuck", "price": 10.0, "exit_price": np.nan}])
    o = pd.DataFrame([{"market_id": "m", "selection_id": 2, "label": "no_run", "bsp": np.nan, "removed": False},
                      {"market_id": "m", "selection_id": 3, "label": "ran", "bsp": 5.0, "removed": False},
                      {"market_id": "m", "selection_id": 4, "label": "no_run", "bsp": np.nan, "removed": True}])
    s = pp.settle(t, o).set_index("selection_id")["ret"]
    assert np.isclose(s[1], (10 / 7 - 1) * 0.98) and s[2] == -1.0 and np.isclose(s[3], 0.98) and s[4] == 0.0


def test_the_reading_waits_for_its_date():
    assert pp.reading(pd.DataFrame(), "2027-01-01")["verdict"] == "NOT YET"
    few = pd.DataFrame({"ret": [0.1] * 10, "market_id": list("ab") * 5})
    assert pp.reading(few, "2027-07-02")["verdict"] == "NOT ENOUGH TRADES"
    assert pp.reading(few, "2028-01-01")["verdict"] == "FAIL"
