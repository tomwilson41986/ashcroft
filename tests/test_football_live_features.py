"""The price path as the live feed sees it: the feed's polling times, and the same ticks from history and from polls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football import live_features as lf

KO = 1_764_000_000_000                       # a kick-off, ms
MIN = 60_000


def test_the_polls_follow_the_feeds_cadence_and_start_24_hours_out():
    g = lf.poll_times(KO, KO - 48 * 60 * MIN)
    gaps = np.diff(g) / MIN
    mins = (KO - g[1:]) / MIN                  # the step back is set by the later poll's minutes to the off
    assert g[-1] == KO - 1 and g[0] >= KO - 1440 * MIN
    assert set(np.round(gaps[mins > 361])) == {30.0}       # every 30 minutes beyond six hours
    assert set(np.round(gaps[mins <= 15])) == {1.0}        # every minute in the last quarter hour


def test_a_poll_is_a_tick_only_when_the_price_moved_since_the_last_poll():
    pt = np.array([KO - 50 * MIN, KO - 3 * MIN, KO - 2.5 * MIN], dtype=np.int64)
    ltp = np.array([2.0, 2.1, 2.2])
    grid = np.array([KO - 4 * MIN, KO - 3 * MIN, KO - 2 * MIN, KO - 1 * MIN], dtype=np.int64)
    t, p = lf.sample_runner(pt, ltp, grid)
    assert list(p) == [2.0, 2.1, 2.2] and list(t) == [KO - 4 * MIN, KO - 3 * MIN, KO - 2 * MIN]


def test_history_thinned_and_the_live_polls_give_the_same_ticks():
    grid = lf.poll_times(KO, KO - 30 * MIN)
    rng = np.random.default_rng(0)
    prices = np.round(2.0 + np.cumsum(rng.choice([-0.02, 0, 0.02], size=len(grid))), 2)
    # history: a tick at every poll time (and finer ones between, which a poll cannot see)
    hist = pd.DataFrame({"market_id": "1.1", "market_type": "MATCH_ODDS", "selection_id": 7, "handicap": 0.0,
                         "pt": np.r_[grid, grid[:-1] + 20_000], "seq": 0,
                         "ltp": np.r_[prices, prices[:-1] + 0.5], "pre_off": True})
    thin = lf.thin_ticks(hist, {"1.1": KO})
    polls = pd.DataFrame({"polled_utc": pd.to_datetime(grid, unit="ms", utc=True).astype(str), "market_id": "1.1",
                          "selection_id": 7, "last_traded": prices, "inplay": 0, "status": "OPEN"})
    live = lf.ticks_from_polls(polls, {"1.1": "MATCH_ODDS"})
    assert list(thin["pt"]) == list(live["pt"]) and list(thin["ltp"]) == list(live["ltp"])
