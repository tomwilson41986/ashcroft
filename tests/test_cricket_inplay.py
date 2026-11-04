"""cricket/market/inplay_panel.py: Betfair's ladder ticks and the per-minute grid."""

import datetime as dt

from cricket.market.inplay_panel import market_grid, tick_down, tick_up


def test_ticks_follow_betfair_ladder():
    assert tick_up(1.5) == 1.51 and tick_down(1.5) == 1.49
    assert tick_up(2.0) == 2.02 and tick_down(2.0) == 1.99      # at an edge: up in the new band, down in the old
    assert tick_up(3.0) == 3.05 and tick_down(3.0) == 2.98
    assert tick_down(4.0) == 3.95 and tick_up(5.9) == 6.0 and tick_up(6.0) == 6.2
    assert tick_down(10.0) == 9.8 and tick_up(10.0) == 10.5
    assert tick_down(1.01) == 1.01 and tick_up(1000.0) == 1000.0


def test_grid_forward_fills_and_marks_suspension():
    t0 = dt.datetime(2026, 7, 1, 14, 0, tzinfo=dt.timezone.utc)
    m = {"inplay_at": t0,
         "status": [(t0, "OPEN", True), (t0 + dt.timedelta(minutes=2), "SUSPENDED", True),
                    (t0 + dt.timedelta(minutes=3), "OPEN", True), (t0 + dt.timedelta(minutes=5), "CLOSED", True)],
         "ltp": {(t0, 1): 2.0, (t0, 2): 2.0, (t0 + dt.timedelta(minutes=3, seconds=10), 1): 1.5}}
    g = market_grid(m, 1, 2)
    assert list(g.minute) == [0, 1, 2, 3, 4, 5]
    assert g.p_a.iloc[0] == 0.5 and g.ltp_a.iloc[3] == 2.0 and g.ltp_a.iloc[4] == 1.5   # 3:10 shows from minute 4
    assert list(g.suspended) == [False, False, True, False, False, True]
    assert g.age_b.iloc[5] == 5.0
