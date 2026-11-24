"""The 100-strategy harness's core on toy frames: the money arithmetic (K2, K3, K6), the windows and the bootstrap
(K1, K10), the verdict (K10) and the runner's promise that one failure never stops the run. No real data is read."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from research.strat100 import core, run


def bets(rows):
    base = {"race_date": "2026-10-05", "race_key": "m1", "selection": "a", "side": "back", "stake": 100.0,
            "entry_price": 5.0, "exit": "bsp", "exit_price": 4.0, "won": np.nan, "void": False}
    return pd.DataFrame([{**base, **r} for r in rows])


def strat(dataset="b_live", **kw):
    kw = {"id": "T1", "name": "toy", "family": "toy", "side": "back", "market": "win",
          "build": lambda frames: core.empty_bets(), **kw}
    return core.Strategy(dataset=dataset, **kw)


def money(rows, **kw):
    b = core.check_bets(bets(rows))
    return core.settle(b, **kw), core.market_net(core.settle(b, **kw))


# ---- the money arithmetic ----------------------------------------------------------------------------------------

def test_a_back_closed_at_the_bsp_makes_the_move_on_every_outcome():
    s, m = money([{"won": 1}, {"selection": "b", "won": 0}, {"selection": "c"}])     # the result is not needed
    assert s["pnl"].tolist() == pytest.approx([25.0, 25.0, 25.0])                     # 100 x (5 / 4 - 1)
    assert m["gross"].iloc[0] == pytest.approx(75.0) and m["net"].iloc[0] == pytest.approx(75.0 * 0.98)
    s, m = money([{"entry_price": 4.0, "exit_price": 5.0}])                           # the price drifted: a loss
    assert s["pnl"].iloc[0] == pytest.approx(-20.0) and m["commission"].iloc[0] == 0.0


def test_a_lay_closed_at_the_bsp_and_its_liability():
    s, m = money([{"side": "lay", "entry_price": 4.0, "exit_price": 5.0}])
    assert s["pnl"].iloc[0] == pytest.approx(20.0)                                    # 100 x (1 - 4 / 5)
    assert s["risk"].iloc[0] == pytest.approx(300.0)                                  # liability s x (L - 1)
    assert m["net"].iloc[0] == pytest.approx(19.6)
    s, _ = money([{"side": "lay", "stake": np.nan, "liability": 300.0, "entry_price": 4.0, "exit_price": 5.0}])
    assert s["stake"].iloc[0] == pytest.approx(100.0) and s["pnl"].iloc[0] == pytest.approx(20.0)


def test_a_held_winner_and_loser():
    held = {"exit": "result", "exit_price": np.nan, "stake": 10.0, "entry_price": 6.0}
    s, _ = money([{**held, "won": 1}, {**held, "won": 0, "race_key": "m2"},
                  {**held, "side": "lay", "won": 1, "race_key": "m3"}, {**held, "side": "lay", "won": 0, "race_key": "m4"}])
    assert s["pnl"].tolist() == pytest.approx([50.0, -10.0, -50.0, 10.0])
    assert s["risk"].tolist() == pytest.approx([10.0, 10.0, 50.0, 50.0])
    s, _ = money([{**held, "won": 0.5}])                                              # a dead heat: half the stake wins
    assert s["pnl"].iloc[0] == pytest.approx(0.5 * 50.0 - 0.5 * 10.0)


def test_commission_is_two_percent_of_a_markets_net_winnings_only():
    held = {"exit": "result", "exit_price": np.nan, "stake": 10.0, "entry_price": 6.0}
    _, m = money([{**held, "won": 1}, {**held, "selection": "b", "won": 0}])          # one market: +50 - 10 = 40
    assert len(m) == 1 and m["commission"].iloc[0] == pytest.approx(0.8) and m["net"].iloc[0] == pytest.approx(39.2)
    _, m = money([{**held, "won": 1, "market": "win"}, {**held, "won": 0, "market": "place"}])   # they do not net
    assert len(m) == 2 and m["commission"].sum() == pytest.approx(1.0) and m["net"].sum() == pytest.approx(39.0)
    _, m = money([{**held, "won": 0}])
    assert m["commission"].iloc[0] == 0.0 and m["net"].iloc[0] == pytest.approx(-10.0)


def test_a_void_is_no_bet():
    s, m = money([{"void": True}, {"selection": "b"}, {"selection": "c", "exit_price": np.nan},   # no BSP: unsettled
                  {"selection": "d", "exit": "result", "exit_price": np.nan}])                    # no result: unsettled
    assert s["pnl"].tolist() == pytest.approx([0.0, 25.0, 0.0, 0.0]) and s["risk"].tolist() == [0.0, 100.0, 0.0, 0.0]
    assert s["void"].tolist() == [True, False, True, True] and s["unsettled"].tolist() == [False, False, True, True]
    assert m["bets"].iloc[0] == 1


def test_a_half_hedge_keeps_half_the_result():
    half = {"hedge_fraction": 0.5}                    # the closing lay is 0.5 x 100 x 5 / 4 = 62.5 at 4.0
    s, _ = money([{**half, "won": 1}, {**half, "won": 0, "selection": "b"},
                  {**half, "side": "lay", "won": 1, "selection": "c"}, {**half, "side": "lay", "won": 0, "selection": "d"}])
    assert s["pnl"].tolist() == pytest.approx([400 - 62.5 * 3, -100 + 62.5, -(400 - 62.5 * 3), 100 - 62.5])
    full = core.settle(core.check_bets(bets([{"won": 1}])))["pnl"].iloc[0]
    held = core.settle(core.check_bets(bets([{"won": 1, "exit": "result"}])))["pnl"].iloc[0]
    assert s["pnl"].iloc[0] == pytest.approx(0.5 * full + 0.5 * held)


def test_a_closing_bet_of_a_given_size():
    row = {"entry_price": 9.0, "exit": "inplay", "exit_price": 5.0, "exit_stake": 100.0, "entry_type": "sp"}
    s, _ = money([{**row, "won": 1}, {**row, "won": 0, "selection": "b"}, {**row, "selection": "c"}])
    assert s["pnl"].tolist() == pytest.approx([800.0 - 400.0, 0.0, 0.0])      # the stake laid back: a free bet
    assert s["unsettled"].tolist() == [False, False, True]                    # its result is needed


def test_the_non_runner_cut_and_the_worse_price():
    s, _ = money([{"cut_factor": 0.9}, {"selection": "b", "entry_price": 1.05, "cut_factor": 0.5, "exit_price": 1.01}])
    assert s["price"].tolist() == pytest.approx([4.5, 1.01]) and s["pnl"].iloc[0] == pytest.approx(12.5)
    rows = [{"entry_type": "book"}, {"selection": "b", "entry_type": "sp"},
            {"selection": "c", "side": "lay", "entry_type": "wap", "entry_price": 4.0, "exit_price": 5.0}]
    s, _ = money(rows, haircut=0.02)
    assert s["pnl"].tolist() == pytest.approx([100 * (4.9 / 4 - 1), 25.0, 100 * (1 - 4.08 / 5)])
    assert s["risk"].iloc[2] == pytest.approx(100 * 3.08)


def test_a_bad_frame_is_refused():
    with pytest.raises(ValueError):
        core.check_bets(bets([{"side": "buy"}]))
    with pytest.raises(ValueError):
        core.check_bets(bets([{"entry_price": np.nan}]))
    with pytest.raises(ValueError):
        core.check_bets(bets([{}]).drop(columns="exit"))
    assert len(core.check_bets(core.empty_bets())) == 0


# ---- windows, the daily series, the bootstrap --------------------------------------------------------------------

def test_the_windows_are_the_registered_dates():
    w = core.window_of(["2024-12-27", "2024-12-28", "2025-09-30", "2025-10-01", "2026-03-27", "2026-03-28"], "a_model")
    assert w.tolist() == ["", "select", "select", "confirm", "confirm", ""]
    w = core.window_of(["2018-01-01", "2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01"], "a_market",
                       select_from=core.FITTED_SELECT_FROM)
    assert w.tolist() == ["", "", "select", "select", "confirm"]
    assert core.window_of(["2026-10-31", "2026-11-01", "2026-11-24"], "b_live").tolist() == ["select", "confirm", ""]


def test_a_day_with_no_bet_is_zero_and_the_split_is_by_day():
    b = bets([{"race_date": "2026-10-05"}, {"race_date": "2026-11-03", "race_key": "m2", "entry_price": 4.0,
                                            "exit_price": 5.0}])
    days = ["2026-10-05", "2026-10-06", "2026-11-03", "2026-11-04", "2026-11-30"]
    r = core.score(strat(), b, days, draws=200)
    d = r["_daily"]
    assert d["race_date"].tolist() == days[:4] and d["window"].tolist() == ["select", "select", "confirm", "confirm"]
    assert d["pnl"].tolist() == pytest.approx([24.5, 0.0, -20.0, 0.0])
    assert (r["select_days"], r["confirm_days"], r["all_days"]) == (2, 2, 4)
    assert r["select_ret"] == pytest.approx(0.245) and r["confirm_ret"] == pytest.approx(-0.2)
    assert r["all_ret"] == pytest.approx(4.5 / 200.0) and r["all_bets"] == 2
    assert r["confirm_ret_2pct"] == pytest.approx(-0.216) and r["confirm_ret_5pct"] is None   # K5: 2% only on live


def test_the_bootstrap_is_by_day_a_ratio_of_sums_and_repeatable():
    rng = np.random.default_rng(1)
    pnl, stake = rng.normal(5, 20, 80), rng.uniform(50, 150, 80)
    est, lo, hi = core.day_bootstrap(pnl, stake, draws=2000)
    assert est == pytest.approx(pnl.sum() / stake.sum()) and lo < est < hi
    assert core.day_bootstrap(pnl, stake, draws=2000) == (est, lo, hi)
    mean, lo2, hi2 = core.day_bootstrap(pnl, None, draws=2000)
    assert mean == pytest.approx(pnl.mean()) and lo2 < mean < hi2
    assert np.isnan(core.day_bootstrap([], [])[0]) and np.isnan(core.day_bootstrap([1.0, 2.0], [0.0, 0.0])[0])


def test_the_holdout_is_refused():
    b = bets([{"race_date": "2026-03-28"}])
    with pytest.raises(core.HoldoutError):
        core.score(strat("a_model"), b, ["2026-03-27"], draws=50)
    kept = core.drop_holdout(pd.DataFrame({"race_date": ["2026-03-27", "2026-03-28", "2026-10-01"]}))
    assert kept["race_date"].tolist() == ["2026-03-27"]



def test_the_bsp_close_can_be_settled_as_the_order_betfair_takes():
    """K3 locks s x (p / b - 1); a MARKET_ON_CLOSE lay is a liability fixed before the SP is known, so the order
    that can be placed pays 0 on a winner and s (p - b) / (b - 1) on a loser: the same mean, a wider spread."""
    rows = [{"won": 1}, {"selection": "b", "won": 0}, {"selection": "c"}]
    b = core.check_bets(bets(rows))
    k3, real = core.settle(b), core.settle(b, sp_hedge="liability")
    assert k3["pnl"].tolist() == pytest.approx([25.0, 25.0, 25.0])                      # the registered headline
    assert real["pnl"].tolist() == pytest.approx([0.0, 100.0 / 3.0, 25.0])              # no result: K3's figure kept
    assert real["placeable"].tolist() == [True, True, False]
    assert 0.25 * 0.0 + 0.75 * 100.0 / 3.0 == pytest.approx(25.0)                       # the same mean at a fair SP 4.0
    cut = core.settle(core.check_bets(bets([{"won": 1, "cut_factor": 0.9}, {"won": 0, "cut_factor": 0.9}])),
                      sp_hedge="liability")                                              # the lay was sized before the cut
    assert cut["pnl"].tolist() == pytest.approx([100 * 3.5 - 400.0, -100 + 400.0 / 3.0])
    lay = core.settle(core.check_bets(bets([{"side": "lay", "entry_price": 4.0, "exit_price": 5.0, "won": 1},
                                            {"side": "lay", "entry_price": 4.0, "exit_price": 5.0, "won": 0}])),
                      sp_hedge="liability")                                              # an SP back of the lay's size
    assert lay["pnl"].tolist() == pytest.approx([100.0, 0.0])
    held = core.settle(core.check_bets(bets([{"exit": "result", "exit_price": np.nan, "won": 1}])), sp_hedge="liability")
    assert held["pnl"].iloc[0] == pytest.approx(400.0) and not held["placeable"].iloc[0]
    with pytest.raises(ValueError):
        core.settle(b, sp_hedge="stake")


def test_a_pass_stands_only_if_the_placeable_close_passes_too():
    """A small steady CLV on odds-on prices: locked as K3 writes it the series is smooth and passes; settled as the
    liability order a winner pays nothing and the whole gain rides on the rare loser, so the same bets swing with
    the results and the deflated Sharpe fails: 'PASS on K3 only'."""
    days = pd.date_range("2026-10-01", "2026-11-23").strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(3)
    rows = [{"race_date": d, "race_key": f"{d}|{i}", "entry_price": 1.05 + rng.normal(0, 0.0005), "exit_price": 1.04,
             "won": float(rng.random() < 1 / 1.04)} for d in days for i in range(5)]
    r = core.score(strat(), bets(rows), days, n_trials=486, draws=500)
    assert r["confirm_lo"] > 0 and r["dsr_passed"] and r["placeable_bets"] == len(rows) and r["placeable_no_result"] == 0
    assert r["all_sharpe_placeable"] < r["all_sharpe"] and not r["dsr_placeable_passed"]
    assert r["verdict"] == "PASS on K3 only" and core.verdict(r) == r["verdict"]
    assert {"pnl", "pnl_placeable"} <= set(r["_daily"].columns)
    none = core.score(strat(), bets([{**x, "won": np.nan} for x in rows]), days, n_trials=486, draws=500)
    assert none["verdict"] == "PASS" and none["placeable_no_result"] == len(rows)       # no result: K3 alone, counted


# ---- the verdict ---------------------------------------------------------------------------------------------------

def live_days():
    d = pd.date_range("2026-10-01", "2026-11-23").strftime("%Y-%m-%d").tolist()
    return d


def daily_bets(days, per_day, entry, bsp=4.0, seed=0):
    """Closed backs: ``per_day`` a day, the entry price a little noisy around ``entry``."""
    rng = np.random.default_rng(seed)
    rows = [{"race_date": d, "race_key": f"{d}|{i}", "entry_price": entry + rng.normal(0, 0.05), "exit_price": bsp}
            for d in days for i in range(per_day)]
    return bets(rows)


def test_a_steady_edge_passes_and_the_qualifiers_apply():
    days = live_days()
    r = core.score(strat(), daily_bets(days, 5, 4.4), days, n_trials=486, draws=500)
    assert r["confirm_lo"] > 0 and r["select_ret"] > 0 and r["dsr"] >= 0.95 and r["n_trials"] == 486
    assert r["dsr_window"] == "all" and core.verdict(r) == r["verdict"] == "PASS"
    r = core.score(strat(blind=False), daily_bets(days, 5, 4.4), days, n_trials=486, draws=500)
    assert r["verdict"] == "PASS (not blind)"


def test_no_edge_fails_and_too_little_is_not_read():
    days = live_days()
    r = core.score(strat(), daily_bets(days, 5, 3.9), days, n_trials=486, draws=500)
    assert r["verdict"] == "FAIL" and "a" in r["fails"] and "c" in r["fails"]
    r = core.score(strat(), daily_bets(days, 1, 4.4), days, n_trials=486, draws=500)      # 23 CONFIRM bets
    assert r["verdict"] == "not read: too few"
    r = core.score(strat(), daily_bets(days[-25:], 6, 4.4), days[-25:], n_trials=486, draws=500)   # 25 race days
    assert r["verdict"] == "not read"
    confirm_only = [d for d in days if d >= "2026-11-01"] + [f"2026-10-{i:02d}" for i in range(1, 11)]
    b = daily_bets([d for d in confirm_only if d >= "2026-11-01"], 6, 4.4)                # nothing staked in SELECT
    r = core.score(strat(), b, sorted(confirm_only), n_trials=486, draws=500)
    assert r["verdict"] == "FAIL" and "b" in r["fails"]
    r = core.score(strat(), daily_bets(days, 5, 4.4), days, draws=500)                    # not deflated yet
    assert r["verdict"] == "FAIL" and r["fails"] == "c"


def test_a_price_file_pass_that_dies_at_the_worse_price_is_wap_only():
    sel = pd.date_range("2023-10-01", "2023-12-31").strftime("%Y-%m-%d").tolist()
    con = pd.date_range("2024-01-01", "2024-04-30").strftime("%Y-%m-%d").tolist()
    b = daily_bets(sel + con, 4, 4.04)                                                    # 1% over the BSP
    r = core.score(strat("a_market", entry="wap"), b, sel + con, n_trials=486, draws=500)
    assert r["confirm_ret"] > 0 and r["confirm_ret_2pct"] < 0 and r["confirm_ret_5pct"] < r["confirm_ret_2pct"]
    assert r["dsr_window"] == "confirm" and r["verdict"] == "PASS at the WAP only" and r["flags"] == "no cuts"
    r = core.score(strat("a_market", entry="sp"), b, sel + con, n_trials=486, draws=500)  # SP entries: no worse price
    assert r["confirm_ret_2pct"] == r["confirm_ret"] and r["verdict"] == "PASS" and r["flags"] == ""


def test_a_paired_strategy_is_read_against_its_benchmark():
    days = live_days()
    bench = core.score(strat(id="BASE"), daily_bets(days, 5, 4.2, seed=1), days, n_trials=486, draws=500)
    better = core.score(strat(id="T2", paired="BASE"), daily_bets(days, 5, 4.4, seed=2), days, n_trials=486, draws=500)
    core.pair(better, bench, 486, draws=500)
    assert better["pair_confirm_lo"] > 0 and better["pair_verdict"] == "better than its benchmark"
    same = core.score(strat(id="T3", paired="BASE"), daily_bets(days, 5, 4.2, seed=3), days, n_trials=486, draws=500)
    core.pair(same, bench, 486, draws=500)
    assert same["pair_verdict"] == "not better than its benchmark"
    seen = core.score(strat(id="T4", paired="BASE", blind=False), daily_bets(days, 5, 4.4, seed=2), days,
                      n_trials=486, draws=500)
    core.pair(seen, bench, 486, draws=500)                 # K10: CONFIRM seen before, the pair says so too
    assert seen["pair_verdict"] == "better than its benchmark (not blind)"
    seen_same = core.score(strat(id="T5", paired="BASE", blind=False), daily_bets(days, 5, 4.2, seed=3), days,
                           n_trials=486, draws=500)
    core.pair(seen_same, bench, 486, draws=500)
    assert seen_same["pair_verdict"] == "not better than its benchmark"


def held_backs(days, n, edge, seed):
    """``n`` held GBP20 backs a day whose chance of winning is ``edge`` / price: every bet returns edge - 1 a GBP."""
    rng = np.random.default_rng(seed)
    k = len(days) * n
    price = rng.uniform(1.5, 4.0, k)
    return pd.DataFrame({"race_date": np.repeat(np.asarray(days, dtype=object), n), "race_key": [f"m{i}" for i in range(k)],
                         "selection": "1", "side": "back", "stake": 20.0, "entry_price": price, "exit": "result",
                         "exit_price": np.nan, "won": (rng.random(k) < edge / price).astype(float), "void": False})


def test_a_benchmark_of_another_size_is_compared_by_return_not_by_pounds():
    """A benchmark that is a superset (or another set of races) at level stakes: in pounds a random twentieth of a
    losing benchmark 'beats' it by losing less money; pair_mode 'return' reads the return on the same stake."""
    days = pd.date_range("2021-01-01", "2026-03-27").strftime("%Y-%m-%d").tolist()
    base = held_backs(days, 40, 0.97, seed=3)                              # no skill anywhere: -3% before commission
    random_part = base[np.random.default_rng(4).random(len(base)) < 0.05]
    kw = {"n_trials": 486, "draws": 500}
    bench = core.score(strat("a_market", id="BASE", entry="sp", benchmark=True), base, days, **kw)
    assert bench["confirm_ret"] < -0.02
    gbp = core.pair(core.score(strat("a_market", id="T1", entry="sp", paired="BASE"), random_part, days, **kw),
                    bench, 486, draws=500)
    assert gbp["pair_mode"] == "gbp" and gbp["pair_confirm_lo"] > 0       # the flattering reading: minus the benchmark's loss
    ret = core.pair(core.score(strat("a_market", id="T2", entry="sp", paired="BASE", pair_mode="return"), random_part,
                               days, **kw), bench, 486, draws=500)
    assert ret["pair_mode"] == "return" and ret["pair_verdict"] == "not better than its benchmark"
    assert ret["pair_confirm_lo"] < 0 < ret["pair_confirm_hi"] and abs(ret["pair_confirm_mean"]) < 0.1 * gbp["pair_confirm_mean"]
    d = ret["_pair"].merge(ret["_daily"], on="race_date").merge(bench["_daily"], on="race_date", suffixes=("", "_b"))
    day = d[(d["stake"] > 0) & (d["stake_b"] > 0)].iloc[0]                 # one day by hand
    assert day["diff"] == pytest.approx(day["pnl"] - day["stake"] * day["pnl_b"] / day["stake_b"])
    assert (d.loc[d["stake"] == 0, "diff"] == 0).all()                     # no bet, no difference
    assert core.pair_sharpe(ret, bench) == pytest.approx(ret["pair_sharpe"], abs=1e-6)
    # real skill on other races than the benchmark's: +12% a GBP against -3%
    good = core.pair(core.score(strat("a_market", id="T3", entry="sp", paired="BASE", pair_mode="return"),
                                held_backs(days, 10, 1.12, seed=5), days, **kw), bench, 486, draws=500)
    assert good["pair_confirm_lo"] > 0 and good["pair_verdict"] == "better than its benchmark"
    # a benchmark on OTHER days (S073's): its window return stands in on the strategy's days
    odd = core.score(strat("a_market", id="BASE2", entry="sp", benchmark=True), base[base["race_date"].isin(days[1::2])],
                     days, **kw)
    even = core.pair(core.score(strat("a_market", id="T4", entry="sp", paired="BASE2", pair_mode="return"),
                                random_part[random_part["race_date"].isin(days[::2])], days, **kw), odd, 486, draws=500)
    assert even["pair_verdict"] == "not better than its benchmark" and even["pair_confirm_lo"] < 0 < even["pair_confirm_hi"]
    none = core.score(strat("a_market", id="BASE3", entry="sp", benchmark=True), core.empty_bets(), days, **kw)
    lone = core.pair(core.score(strat("a_market", id="T5", entry="sp", paired="BASE3", pair_mode="return"), random_part,
                                days, **kw), none, 486, draws=500)
    assert lone["pair_verdict"].startswith("not read") and lone["pair_confirm_mean"] is None
    with pytest.raises(ValueError, match="pair_mode"):
        strat(pair_mode="ratio")


# ---- helpers -------------------------------------------------------------------------------------------------------

def test_the_tick_ladder():
    assert len(core.TICKS) == 350 and core.TICKS[0] == 1.01 and core.TICKS[-1] == 1000.0
    assert core.tick_floor(3.52) == 3.5 and core.tick_ceil(3.52) == 3.55 and core.tick_floor(2.0) == 2.0
    assert core.tick_move(2.0, 1) == 2.02 and core.tick_move(2.0, -1) == 1.99 and core.tick_move(3.52, -1) == 3.5
    assert core.tick_under(3.5) == 3.45 and core.tick_under(3.52) == 3.45 and core.tick_over(3.52) == 3.6
    assert np.isnan(core.tick_under(1.01)) and np.isnan(core.tick_over(1000.0)) and np.isnan(core.tick_floor(np.nan))
    assert core.tick_under(np.array([10.0, 20.0, 100.0])).tolist() == [9.8, 19.5, 95.0]


def test_probabilities_and_capacity():
    p = core.normalise([2.0, 4.0, 4.0, 3.0, np.nan], [1, 1, 1, 2, 2])
    assert p[:3].tolist() == pytest.approx([0.5, 0.25, 0.25]) and np.isnan(p[3:]).all()
    assert core.normalise([3.0, np.nan], [2, 2], whole=False)[0] == pytest.approx(1.0)
    q = np.array([0.5, 0.3, 0.2])
    assert core.harville_place(q, 2)[0] == pytest.approx(0.5 + 0.3 * 0.5 / 0.7 + 0.2 * 0.5 / 0.8)
    assert core.harville_place(q, 2).sum() == pytest.approx(2.0)
    q = np.array([0.45, 0.2, 0.12, 0.1, 0.06, 0.04, 0.02, 0.01])
    hen, har, k9 = core.henery_place(q, 3), core.harville_place(q, 3), core.place_prob(q, 3)
    assert hen.sum() == pytest.approx(3.0) and k9.sum() == pytest.approx(3.0)
    assert hen[0] < har[0] and hen[-1] > har[-1] and k9[0] < har[0]        # the favourite places less than Harville says
    assert core.henery_place(q, 1) == pytest.approx(q, abs=0.01)
    assert core.ordering_exponents(8) == (0.804, 0.640) and core.ordering_exponents(20) == (0.829, 0.735)
    assert core.capacity([10.0, 50.0, 1.0], [100.0, 100.0, 100.0], 0.10).tolist() == [10.0, 10.0, 0.0]
    assert core.to_win_stake(400.0, [1.5, 3.0, 41.0], 15.0, 200.0).tolist() == pytest.approx([200.0, 200.0, 15.0])


def test_the_trial_count_is_never_under_the_registered_floor():
    t = core.n_trials_for(100)
    assert t["n_trials"] >= 486 and t["n_trials"] >= t["ledger_rules"]
    if not t["registration_in_ledger"]:
        assert t["n_trials"] >= 100 + t["ledger_rules"]


def test_the_registered_strategies_are_counted_once(tmp_path):
    import json as _json
    base = [{"when": "2026-11-01T00:00Z", "kind": "query", "tag": f"t{i}", "sport": "horse_racing"} for i in range(500)]
    led = tmp_path / "ledger.jsonl"
    led.write_text("\n".join(_json.dumps(d) for d in base) + "\n")
    before = core.n_trials_for(100, ledger_path=led)
    assert before["n_trials"] == 100 + before["ledger_rules"] and not before["registration_in_ledger"]
    reg = {"when": "2026-11-24T18:00Z", "kind": "query", "tag": core.REGISTRATION_TAG, "sport": "horse_racing",
           "n_rules": 100}
    led.write_text(led.read_text() + _json.dumps(reg) + "\n")
    after = core.n_trials_for(100, ledger_path=led)
    assert after["registration_in_ledger"] and after["n_trials"] == before["n_trials"]


# ---- the runner ----------------------------------------------------------------------------------------------------

def test_one_failure_never_stops_the_run(tmp_path, monkeypatch, capsys):
    days = live_days()

    def boom(frames):
        raise RuntimeError("no such column")

    strategies = types.ModuleType("fake_strategies")
    strategies.STRATEGIES = [
        strat(id="BASE", build=lambda f: daily_bets(f["days"], 5, 4.2, seed=1), benchmark=True),
        strat(id="T1", build=lambda f: daily_bets(f["days"], 5, 4.4, seed=2), paired="BASE"),
        strat(id="T2", build=boom),
        strat(id="T3", build=lambda f: (daily_bets(f["days"][-20:], 6, 4.4), f["days"][-20:])),   # its own 20 days
    ]
    data = types.ModuleType("fake_data")
    data.load_b_live = lambda: {"days": days, "marks": pd.DataFrame({"race_date": days})}
    monkeypatch.setitem(sys.modules, "research.strat100.fake_strategies", strategies)
    monkeypatch.setitem(sys.modules, "research.strat100.fake_data", data)
    monkeypatch.setattr(run, "DATASETS", {"b_live": ("fake_strategies", "fake_data", "load_b_live")})
    assert run.main(["--out", str(tmp_path), "--draws", "300", "--only", "T1,T2,T3"]) == 0     # BASE comes with T1
    res = json.loads((tmp_path / "strat100_results.json").read_text())
    by = {r["id"]: r for r in res["strategies"]}
    assert set(by) == {"BASE", "T1", "T2", "T3"} and res["n_trials"] >= 486
    assert by["T2"]["verdict"] == "error" and "no such column" in by["T2"]["error"]
    assert by["T1"]["verdict"] in ("PASS", "FAIL") and by["T1"]["dsr"] is not None     # deflated by the run's own spread
    assert by["T1"]["confirm_lo"] > 0 and by["T1"]["pair_verdict"] == "better than its benchmark"
    assert by["BASE"]["verdict"].startswith("benchmark (") and by["T3"]["verdict"] == "not read"
    daily = pd.read_csv(tmp_path / "strat100_daily.csv")
    assert set(daily["id"]) == {"BASE", "T1", "T3"} and len(daily[daily["id"] == "T1"]) == len(days)
    assert daily.loc[daily["id"] == "T1", "pnl_less_benchmark"].notna().all()
    out = capsys.readouterr().out
    assert "ERROR T2" in out and "n_trials used for the deflated Sharpe" in out


def test_a_data_set_that_will_not_load_fails_its_strategies_only(tmp_path, monkeypatch):
    strategies = types.ModuleType("fake_strategies")
    strategies.STRATEGIES = [strat("a_model", id="A1")]
    data = types.ModuleType("fake_data")
    data.load_a_model = lambda: {"oos": pd.DataFrame({"race_date": ["2026-03-27", "2026-04-02"]})}   # holdout rows
    monkeypatch.setitem(sys.modules, "research.strat100.fake_strategies", strategies)
    monkeypatch.setitem(sys.modules, "research.strat100.fake_data", data)
    monkeypatch.setattr(run, "DATASETS", {"a_model": ("fake_strategies", "fake_data", "load_a_model")})
    assert run.main(["--out", str(tmp_path), "--draws", "100"]) == 0
    r = json.loads((tmp_path / "strat100_results.json").read_text())["strategies"][0]
    assert r["verdict"] == "error" and "HoldoutError" in r["error"]
