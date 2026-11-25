"""The price path as the live feed sees it: the feed's polling times, and the same ticks from history and from polls."""

from __future__ import annotations

import numpy as np
import pytest
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


def market(ko=KO):
    """A match-odds market's ticks (home shortening) and an over/under market's, every minute for 15 hours."""
    t = np.arange(ko - 900 * MIN, ko, MIN, dtype=np.int64)
    mo = pd.concat([pd.DataFrame({"selection_id": sid, "pt": t, "ltp": np.round(p0 + slope * np.arange(len(t)) / 2, 2)})
                    for sid, p0, slope in ((1, 2.6, -0.001), (2, 3.4, 0.0), (3, 3.0, 0.0005))], ignore_index=True)
    ou = pd.concat([pd.DataFrame({"selection_id": sid, "pt": t, "ltp": p}) for sid, p in ((10, 1.9), (11, 2.0))],
                   ignore_index=True)
    return mo, ou


def test_the_checkpoint_rows_carry_every_live_feature_and_agree_with_the_warehouse_formulas():
    from football import b2l
    from football.warehouse.gold_implied import supremacy_from_1x2, total_from_over
    mo, ou = market()
    thin_mo = lf.thin_ticks(mo.assign(market_id="1.5", market_type="MATCH_ODDS", handicap=0.0, seq=0, pre_off=True),
                            {"1.5": KO})
    thin_ou = lf.thin_ticks(ou.assign(market_id="1.6", market_type="OVER_UNDER_25", handicap=0.0, seq=0,
                                      pre_off=True), {"1.6": KO})
    rows = lf.checkpoint_rows(thin_mo[["selection_id", "pt", "ltp"]], thin_ou[["selection_id", "pt", "ltp"]],
                              {1: "home", 2: "draw", 3: "away"}, 10, "1.5", 99, KO, 360, 720)
    assert len(rows) == 3
    missing = [c for c in b2l.features() if c not in rows.columns and not c.startswith("lv_")]
    assert missing == []                                   # memory is the caller's
    p_over = (1 / 1.9) / (1 / 1.9 + 1 / 2.0)
    tot = total_from_over(np.array([p_over]))
    home = rows.set_index("role").loc["home"]
    sup = supremacy_from_1x2(np.array([home["p_cp"]]), np.array([rows.set_index("role").loc["away", "p_cp"]]), tot)
    assert home["imp_total"] == pytest.approx(tot[0]) and home["imp_supremacy"] == pytest.approx(sup[0])
    assert home["idl_own_dp"] == pytest.approx(home["idl_d_p_home"]) and home["idl_d_p_home"] > 0   # home shortening
    assert home["idl_d_p_over"] == pytest.approx(0.0) and home["imp_ou_age_s"] >= 0       # the over/under never moved
    assert home["pp_x"] == pytest.approx(np.log((1 / home["ltp_cp"]) / (1 - 1 / home["ltp_cp"])))
