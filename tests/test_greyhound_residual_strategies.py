"""research/greyhound/residual_strategies.py: each route settles as a bet on the exchange would, and reads only what it
may (the morning price as a limit, never the SP as a choice)."""

import numpy as np
import pandas as pd
import pytest

from research.greyhound import residual_strategies as rs

CUTS = {"back_top10": 0.1, "lay_bottom10": -0.1}


def test_a_held_bet_and_a_greened_bet_pay_what_the_exchange_pays():
    risk, gross = rs.hold("back", [5.0, 5.0], [1, 0], [1.0, 2.0])
    assert risk.tolist() == [1.0, 2.0] and gross.tolist() == [4.0, -2.0]
    risk, gross = rs.hold("lay", [5.0, 5.0], [1, 0], [1.0, 1.0])
    assert gross.tolist() == [-1.0, 0.25]                       # a lay risking 1 at 5 is a stake of 0.25
    # backed at 6, closed at an SP of 4: 6/4 - 1 = 0.5 whatever wins; laid at 4, closed at 6: 0.25 * (1 - 4/6)
    _, g = rs.green("back", [6.0], [4.0], [1.0])
    _, l = rs.green("lay", [4.0], [6.0], [1.0])
    assert np.isclose(g[0], 0.5) and np.isclose(l[0], (1 / 3) * (1 - 4 / 6))
    # greening is the hold's two legs: back 1 at 6, lay 6/4 at 4: won 5 - 1.5*3 = 0.5, lost -1 + 1.5 = 0.5
    r = rs.per_race(["d", "d", "d"], ["a", "a", "b"], [1, 1, 1], [0.5, -0.2, -1.0], 0.02)
    a = r.set_index("race_id")
    assert np.isclose(a.loc["a", "net"], 0.3 * 0.98) and a.loc["b", "net"] == -1.0     # commission off a race's gain


def _runners():
    return pd.DataFrame({
        "day": ["d1"] * 4 + ["d2"] * 4, "race_id": ["a"] * 4 + ["b"] * 4,
        "s": [0.2, 0.0, -0.05, -0.15, 0.3, -0.12, 0.0, -0.18],
        "won": [1, 0, 0, 0, 0, 1, 0, 0],
        "bsp": [3.0, 4.0, 6.0, 8.0, 5.0, 4.0, 3.0, 12.0],
        "morningwap": [4.0, 4.0, 5.0, 7.0, np.nan, 5.0, 3.0, 10.0],
        "ppwap": [3.5, 4.0, 5.5, 9.0, 4.5, 4.5, 3.0, 11.0],
        "morning_vol": [10.0] * 8, "pp_vol": [100.0] * 8,
        "place_bsp": [1.5, 2.0, 2.5, 3.0, 2.0, 1.8, 1.6, 4.0],
        "placed": [1, 1, 0, 0, 0, 1, 1, 0]})


def test_each_route_picks_the_rows_it_may_and_settles_them():
    r = _runners()
    # the back at the morning price: a's top dog (s 0.2) and b's (s 0.3) has no morning price, so one bet at 4.0 that won
    rule, at_sp, b = rs.rule_bets(r, CUTS, "back", "hold", "morningwap", 0.0)
    assert len(b) == 1 and rule.gross.sum() == 3.0 and at_sp.gross.sum() == 2.0
    # the lay limit on close: a lay at the SP only where the SP is at or under the morning price
    rule, _, b = rs.rule_bets(r, CUTS, "lay", "loc", "morningwap", 0.0)
    assert sorted(b.bsp.tolist()) == [4.0]                       # 8 > 7 and 12 > 10 are not matched; 4 <= 5 is
    assert rule.gross.sum() == -1.0                              # the 4.0 lay won the race
    # a fixed cap of 10: the lays at 8 and 4 (12 is over), and the stakes scaled past the cut
    _, _, b = rs.rule_bets(r, CUTS, "lay", "cap", 10.0, 0.0)
    assert sorted(b.bsp.tolist()) == [4.0, 8.0]
    rule, _, b = rs.rule_bets(r, CUTS, "back", "scaled", None, 0.0)
    assert np.allclose(sorted(rule.risk), [2.0, 3.0])            # s / cut: 0.2 / 0.1 and 0.3 / 0.1
    # the place market: the same dogs at the place SP and the place result
    rule, _, b = rs.rule_bets(r, CUTS, "lay", "place", None, 0.0)
    assert len(b) == 3 and np.isclose(rule.gross.sum(), 1 / 2.0 - 1.0 + 1 / 3.0)
    # greened at the SP from the pre-play price: back at 3.5 closed at 3.0 and back at 4.5 closed at 5.0
    rule, _, _ = rs.rule_bets(r, CUTS, "back", "out", "ppwap", 0.0)
    assert np.isclose(rule.gross.sum(), (3.5 / 3.0 - 1) + (4.5 / 5.0 - 1))


def test_the_screen_reads_every_registered_rule_and_the_two_reference_rules():
    rng = np.random.default_rng(1)
    frames = []
    for i, day in enumerate(pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d")):
        x = _runners().assign(day=day, race_id=lambda f: f.race_id + str(i),
                              s=lambda f: f.s + rng.normal(0, 0.01, len(f)))
        frames.append(x)
    r = pd.concat(frames, ignore_index=True)
    folds = [{"_runners": r[r.day < "2024-01-31"], "cuts": CUTS, "test": ["a", "b"]},
             {"_runners": r[r.day >= "2024-01-31"], "cuts": CUTS, "test": ["c", "d"]}]
    res = rs.screen(folds, 0.02, 1)
    assert [x["rule"] for x in res["rules"]] == [n for n, *_ in rs.RULES] and len(res["rules"]) == 16
    assert len(res["reference"]) == 2
    by = {x["rule"]: x for x in res["rules"]}
    # the back limit on close never matches here (each top dog's SP is under its morning price, or it has none)
    assert by["back_loc_morning"]["pooled"] == {"bets": 0, "note": "no data found"} and not by["back_loc_morning"]["pass"]
    assert all("roi_%" in x["pooled"] for n, x in by.items() if n != "back_loc_morning")
    md = rs.markdown({"screen": res, "coverage": {"runners": len(r)}, "sp_pool": {"note": "x"}})
    assert "| lay_loc_10 |" in md and "no data found" in md


def test_the_join_takes_the_win_and_place_files_and_a_place_market_must_price_the_whole_race():
    runners = pd.DataFrame({"race_id": ["a", "a"], "race_date": ["2024-01-01"] * 2, "race_time": ["12:00:00"] * 2,
                            "track": ["Romford", "Romford"], "trap": [1.0, 2.0], "s": [0.0, 0.0]})
    prices = pd.DataFrame({"market": ["win", "win", "place"], "country": ["GB"] * 3, "race_date": ["2024-01-01"] * 3,
                           "race_time": ["12:00:00"] * 3, "track": ["Romford"] * 3, "trap": [1, 2, 1],
                           "bsp": [3.0, 2.0, 1.5], "ppwap": [3.2, 2.1, np.nan], "morningwap": [3.4, np.nan, np.nan],
                           "morning_vol": [5.0, 0.0, np.nan], "pp_vol": [50.0, 60.0, np.nan], "win_lose": [1, 0, 1]})
    j = rs.join_prices(runners, prices)
    assert j.morningwap.tolist()[0] == 3.4 and np.isnan(j.morningwap.tolist()[1]) and j.ppwap.tolist() == [3.2, 2.1]
    assert j.place_bsp.isna().all()                              # trap 2 has no place price: the race is not counted


def test_the_sp_pool_takes_the_most_waiting_before_the_off_and_the_sp():
    books = pd.DataFrame({
        "source": ["recorder", "recorder", "final", "recorder"], "status": ["OPEN", "OPEN", "CLOSED", "OPEN"],
        "inplay": ["False", "False", "False", "True"], "market_id": ["1.1"] * 4, "selection_id": [7, 7, 7, 7],
        "sp_back_taken": [10.0, 25.0, np.nan, 99.0], "sp_lay_taken": [5.0, 3.0, np.nan, 99.0],
        "sp_actual": [np.nan, np.nan, 6.0, np.nan]})
    pool = rs.sp_pool(books)
    assert pool.sp_back_taken.tolist() == [25.0] and pool.sp_lay_taken.tolist() == [5.0] and pool.sp_actual.tolist() == [6.0]
    t = rs.sp_pool_table(pool)
    assert t == [{"sp_band": "5-10", "runners": 1, "sp_back_stake_median": 25.0,
                  "sp_back_stake_as_liability_median": 125.0, "sp_lay_liability_median": 5.0, "share_with_sp_backs": 1.0}]
