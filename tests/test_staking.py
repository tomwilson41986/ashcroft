"""Tests for the market-blind staking analysis (Kelly sizing, rank selection)."""

import numpy as np
import pandas as pd

from model.bet_analysis import prepare_bets
from model.staking import (add_kelly, bankroll_path, growth_stats, kelly_by_rank, rank_focus,
                           rank_staking, shrinkage_scan, staking_report, time_split_table, topk_staking)


def _frame(n_races=400, seed=7, price_noise=0.0):
    """Fair-priced races: BSP is exactly 1/p_true, so a truthful model has no edge."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = int(rng.integers(6, 14))
        s = rng.normal(0, 1, n)
        p = np.exp(s) / np.exp(s).sum()
        winner = rng.choice(n, p=p)
        rest = [i for i in range(n) if i != winner]
        rng.shuffle(rest)
        pos = {winner: 1, **{h: k + 2 for k, h in enumerate(rest)}}
        for i in range(n):
            price = 1.0 / p[i] * float(np.exp(rng.normal(0, price_noise))) if price_noise else 1.0 / p[i]
            day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=r // 4)
            rows.append({"race_date": day.strftime("%Y-%m-%d"), "race_time": "%02d:00" % (12 + r % 4),
                         "track": "T%d" % (r % 3), "horse_name": "h%d" % i,
                         "predicted_win_prob_norm": p[i], "predicted_bfsp": 1.0 / p[i],
                         "bfsp": price, "won": i == winner, "placing_numerical": pos[i]})
    return pd.DataFrame(rows)


def _prep(df, commission=0.0):
    return add_kelly(prepare_bets(df, commission=commission, blend_lambda=None), commission=commission)


def test_kelly_matches_closed_form():
    d = pd.DataFrame({"p_model": [0.5], "bsp": [3.0], "predicted_bfsp": [2.0]})
    k = add_kelly(d, commission=0.0)
    assert np.isclose(k["b_net"].iloc[0], 2.0)
    assert np.isclose(k["kelly_full"].iloc[0], 0.25)          # (0.5*2 - 0.5)/2
    assert np.isclose(k["edge_model"].iloc[0], 0.5)
    assert np.isclose(k["model_price"].iloc[0], 2.0)          # the model's own forecast, not the BSP
    # commission eats the edge
    assert add_kelly(d, commission=0.5)["kelly_full"].iloc[0] == 0.0


def test_bankroll_arithmetic_and_compounding():
    d = pd.DataFrame({
        "raceid": ["r1", "r1", "r2"], "race_date": ["2025-01-01"] * 3, "race_time": ["13:00", "13:00", "14:00"],
        "won": [True, False, True], "bsp": [3.0, 5.0, 2.0], "b_net": [1.90, 3.80, 0.95],
        "kelly_full": [0.10, 0.20, 0.50], "ret_back": [1.90, -1.0, 0.95],
    })
    p = bankroll_path(d, fraction=1.0)
    assert np.isclose(p["pnl"].iloc[0], 0.10 * 1.90 - 0.20)           # +0.19 - 0.20
    assert np.isclose(p["bank"].iloc[0], 0.99)
    assert np.isclose(p["bank"].iloc[1], 0.99 * (1 + 0.5 * 0.95))     # stakes are a fraction of the *current* bank
    assert p["when"].is_monotonic_increasing


def test_bankroll_stays_in_logs_through_a_long_losing_run():
    n = 4000
    d = pd.DataFrame({"raceid": [f"r{i}" for i in range(n)], "race_date": ["2025-01-01"] * n,
                      "race_time": ["12:00"] * n, "won": False, "bsp": 5.0, "b_net": 3.8,
                      "kelly_full": 0.5, "ret_back": -1.0})
    p = bankroll_path(d, fraction=1.0)
    # halving every race: a cumprod would have flattened to 0 around race 1080
    assert p["log_bank"].is_monotonic_decreasing
    assert p["log_bank"].iloc[-1] < -2000
    s = growth_stats(p)
    assert np.isclose(s["log_growth_per_race"], np.log(0.5))
    assert s["races_to_halve_bank"] == 1 and s["max_drawdown_pct"] > 99.9


def test_total_loss_is_recorded_as_ruin_and_stops_the_path():
    d = pd.DataFrame({"raceid": ["r1", "r2"], "race_date": ["2025-01-01"] * 2, "race_time": ["12:00", "13:00"],
                      "won": [False, True], "bsp": [4.0, 4.0], "b_net": [2.85, 2.85],
                      "kelly_full": [1.0, 1.0], "ret_back": [-1.0, 2.85]})
    p = bankroll_path(d, fraction=1.0)
    assert len(p) == 1 and bool(p["ruin"].iloc[0]) and growth_stats(p)["ruined"]


def test_fair_prices_leave_no_edge_and_no_stake():
    d = _prep(_frame(), commission=0.0)
    assert d["kelly_full"].max() < 1e-9          # p == 1/price exactly: nothing to bet
    rt = rank_staking(d, max_rank=4, n_boot=50)
    assert len(rt) == 4 and rt["win_rate_pct"].is_monotonic_decreasing
    assert rt["avg_bsp"].is_monotonic_increasing
    assert abs(rt.loc[rt["model_rank"] == 1, "roi_flat_pct"].iloc[0]) < 15     # fair game, finite sample


def test_mispriced_market_gives_the_model_a_measurable_edge():
    d = _prep(_frame(price_noise=0.35, seed=3), commission=0.0)
    bets = d[d["kelly_full"] > 0]
    assert 0.2 < len(bets) / len(d) < 0.8
    # the model is the truth here, so realised return on its Kelly bets is positive
    assert np.average(bets["ret_back"], weights=bets["kelly_full"]) > 0
    ladder = kelly_by_rank(d, ranks=(1, 2), fraction=0.25)
    assert set(ladder["rank"]) == {"1", "2"} and ladder["log10_terminal_bank"].iloc[0] > 0


def test_topk_strike_rate_rises_and_shrinkage_scan_runs():
    d = _prep(_frame(), commission=0.05)
    tk = topk_staking(d, ks=(1, 2, 3), n_boot=50)
    assert tk["strike_rate_pct"].is_monotonic_increasing and (tk["bets"] == tk["races"] * tk["top_k"]).all()
    # fair prices plus commission: the model as trained finds nothing, while flattening
    # it to 1/N manufactures "edge" on outsiders that is not there
    sc0 = shrinkage_scan(d, lambdas=(1.0, 0.0), fraction=0.25).set_index("lambda")
    assert sc0.loc[1.0, "bets"] == 0 and sc0.loc[0.0, "bets"] > 0
    assert sc0.loc[0.0, "log_growth_per_race"] < 0
    sc = shrinkage_scan(_prep(_frame(price_noise=0.35, seed=5), commission=0.0), lambdas=(1.0, 0.5, 0.0))
    assert len(sc) == 3 and sc["bets"].min() > 0 and sc["mean_stake_pct"].notna().all()


def test_focus_filters_on_the_forecast_price_not_the_settled_price():
    d = _prep(_frame(price_noise=0.4, seed=11), commission=0.0)
    f = rank_focus(d, rank=1, price_min=3.0, n_boot=50)
    sub = d[(d["model_rank"] == 1) & (d["model_price"] >= 3.0)]
    assert f["n"] == len(sub) > 80
    assert (sub["bsp"] < 3.0).any()          # some settle shorter: the filter is not look-ahead
    assert "stability" in f and len(f["stability"]) == 4


def test_time_split_is_chronological_and_report_assembles():
    d = _prep(_frame(), commission=0.05)
    ts = time_split_table(d[d["model_rank"] == 1])
    assert list(ts["slice"]) == [1, 2, 3, 4] and ts["from"].is_monotonic_increasing
    rep = staking_report(_frame(n_races=200), commission=0.05, max_rank=3)
    for key in ("kelly_ladder", "rank_staking", "topk", "shrinkage", "price_forecast", "kelly_by_rank"):
        assert len(rep[key]) > 0
    assert rep["n_races"] == 200 and len(rep["focus"]) == 6
