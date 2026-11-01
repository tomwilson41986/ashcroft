"""The open-close drift package: the mechanics on a synthetic panel (never a reported figure)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.open_close import backtest, drift, features, panels, run


def synthetic_panel(n_markets: int = 600, seed: int = 0, drift_strength: float = 0.15,
                    with_model: bool = False) -> pd.DataFrame:
    """Markets of 4-8 runners from Jan 2024, the close a favourite-longshot unwind of the open plus noise (and,
    with a model, part of the model's view). Built to exercise the code, not to look like a market."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for m in range(n_markets):
        n = int(rng.integers(4, 9))
        strength = rng.dirichlet(np.ones(n) * 1.5)
        open_p = np.clip(1.0 / (strength * 0.92), 1.05, 400.0)            # the open's book about 1.09
        lq = np.log(open_p)
        model_p = np.exp(lq + rng.normal(0, 0.3, n))
        move = -drift_strength * (lq - lq.mean()) + rng.normal(0, 0.12, n)   # favourites shorten, long shots drift
        if with_model:
            move += 0.4 * np.log(open_p / model_p)                         # the close leans to the model
        close_p = np.clip(open_p * np.exp(move), 1.02, 900.0)
        off = start + pd.Timedelta(days=int(m * 600 / n_markets), hours=13 + int(rng.integers(0, 6)))
        win = rng.choice(n, p=strength)
        for i in range(n):
            rows.append({"sport": "horse", "country": "uk" if m % 3 else "ire", "market_type": "WIN",
                         "market_id": f"m{m}", "event_id": m, "event_name": f"race {m}", "selection_id": i,
                         "selection_name": f"h{i}", "off_time_utc": off, "n_runners": n, "open_price": open_p[i],
                         "open_time_utc": off - pd.Timedelta(hours=6), "open_matched": float(rng.integers(5, 3000)),
                         "p_t24h": np.nan, "p_t60": np.nan, "p_t15": np.nan, "p_t1": np.nan,
                         "close_price": close_p[i], "bsp": close_p[i], "pre_play_matched": np.nan,
                         "result": float(i == win), "source": "test",
                         **({"model_price": model_p[i]} if with_model else {})})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def prepared():
    return features.prepare(synthetic_panel())


def test_prepare_targets_and_features(prepared):
    d = prepared
    assert set(features.TARGETS) <= set(d.columns)
    assert set(features.FEATURES) <= set(d.columns)
    np.testing.assert_allclose(d["y"], np.log(d["close_price"] / d["open_price"]))
    # the share change sums to zero in each market; the open's shares to one
    assert np.allclose(d.groupby("market_id")["y_share"].sum(), 0.0, atol=1e-9)
    assert np.allclose(d.groupby("market_id")["p_share"].sum(), 1.0)
    assert (d.groupby("market_id")["rank_open"].min() == 1).all()
    assert np.allclose(d["hours_to_off"], 6.0)
    assert d["month"].str.match(r"\d{4}-\d{2}").all()


def test_clean_drops_unpriced_and_single_rows():
    p = synthetic_panel(n_markets=20)
    p.loc[0, "close_price"] = np.nan
    p.loc[p["market_id"] == "m1", "open_price"] = np.nan
    p.loc[p["market_id"] == "m2", "open_price"] = 0.5
    d = features.clean(p)
    assert "m1" not in set(d["market_id"]) and "m2" not in set(d["market_id"])
    assert len(d) == len(p) - 1 - (p["market_id"] == "m1").sum() - (p["market_id"] == "m2").sum()


def test_fair_prices_make_a_book_of_one_at_both_ends():
    p = synthetic_panel(n_markets=30)
    d = features.prepare(p, normalise_book=True)
    assert np.allclose((1 / d["open_price"]).groupby(d["market_id"]).sum(), 1.0)
    assert np.allclose((1 / d["close_price"]).groupby(d["market_id"]).sum(), 1.0)
    assert np.allclose(d["overround_open"], 1.0)
    raw = features.prepare(p)
    assert np.allclose(d["y"] - raw["y"], np.log(d["overround_close_raw"] / d["overround_open_raw"]))


def test_walk_forward_never_scores_training_rows(prepared):
    models = [drift.NoChange(), drift.BandShrinkage(min_rows=20)]
    scored = drift.walk_forward(prepared, models, min_train_months=3, refit_every=1)
    assert not scored.empty
    assert (scored["fit_month"] < scored["month"]).all()
    months = sorted(prepared["month"].unique())
    assert sorted(scored["month"].unique()) == months[3:]
    assert np.allclose(scored["pred_no_change"], 0.0)
    # the band model catches the built-in unwind: better than no change out of sample
    acc = backtest.accuracy_table(scored, ["pred_no_change", "pred_band_ols"])
    assert acc.loc["band_ols", "r2_vs_no_change"] > 0.05
    assert acc.loc["no_change", "r2_vs_no_change"] == pytest.approx(0.0)


def test_refit_every_fits_before_the_scored_month(prepared):
    scored = drift.walk_forward(prepared, [drift.BandShrinkage(min_rows=20)], min_train_months=3, refit_every=4)
    assert (scored["fit_month"] < scored["month"]).all()
    assert scored["fit_month"].nunique() < scored["month"].nunique()


def test_gradient_boosting_and_book_models(prepared):
    models = [drift.GradientBoosting(n_estimators=60, n_jobs=1), drift.ConditionalLogit(),
              drift.BookCoherent(drift.GradientBoosting(n_estimators=60, n_jobs=1))]
    scored = drift.walk_forward(prepared, models, min_train_months=6, refit_every=6)
    cols = [c for c in scored.columns if c.startswith("pred_")]
    assert set(cols) == {"pred_gbm", "pred_clogit", "pred_gbm_book"}
    acc = backtest.accuracy_table(scored, cols)
    assert (acc["r2_vs_no_change"] > 0).all()
    # the coherent forecasts form a book: each market's implied closing probabilities sum to the overround the book
    # model gives it (a near-constant across markets), where the raw per-selection forecast's sum wanders
    g = features.market_codes(scored)
    close_book = np.bincount(g, weights=1.0 / scored["close_price"].to_numpy())
    raw = np.bincount(g, weights=np.exp(-scored["pred_gbm"].to_numpy()) / scored["open_price"].to_numpy())
    for c in ("pred_clogit", "pred_gbm_book"):
        p_hat = np.exp(-scored[c].to_numpy()) / scored["open_price"].to_numpy()
        book = np.bincount(g, weights=p_hat)
        assert np.std(book) < 0.5 * np.std(raw)
        assert abs(np.median(book) - np.median(close_book)) < 0.03
    assert "log_open" in drift.GradientBoosting(n_estimators=5, n_jobs=1).fit(prepared).importance()


def test_renormalise_book_keeps_the_book():
    g = np.array([0, 0, 0, 1, 1])
    open_p = np.array([2.0, 3.0, 6.0, 1.5, 3.0])
    yhat = np.array([-0.2, 0.1, 0.3, -0.05, 0.1])
    book = drift.BookModel(a=0.0, b=1.0)                    # the close's overround = the open's
    out = drift.renormalise_book(yhat, open_p, g, book)
    p = np.exp(-out) / open_p
    np.testing.assert_allclose(np.bincount(g, weights=p), np.bincount(g, weights=1 / open_p))


def test_shuffle_test_kills_the_signal(prepared):
    models = [drift.BandShrinkage(min_rows=20), drift.GradientBoosting(n_estimators=60, n_jobs=1)]
    sh = drift.shuffle_test(prepared, models, min_train_months=6, refit_every=6)
    acc = backtest.accuracy_table(sh, ["pred_band_ols", "pred_gbm"])
    assert (acc["r2_vs_no_change"] < 0.02).all()


def test_with_model_arm_uses_the_forecast():
    d = features.prepare(synthetic_panel(n_markets=500, with_model=True, seed=1))
    assert set(features.MODEL_FEATURES) <= set(d.columns)
    models = [drift.ConditionalLogit(), drift.ConditionalLogit(with_model=True),
              drift.GradientBoosting(n_estimators=80, n_jobs=1), drift.GradientBoosting(with_model=True, n_estimators=80, n_jobs=1)]
    scored = drift.walk_forward(d, models, min_train_months=6, refit_every=6)
    acc = backtest.accuracy_table(scored, ["pred_clogit", "pred_clogit_model", "pred_gbm", "pred_gbm_model"])
    assert acc.loc["clogit_model", "rmse"] < acc.loc["clogit", "rmse"]
    assert acc.loc["gbm_model", "rmse"] < acc.loc["gbm", "rmse"]


def test_hedged_trade_profit_is_greened_and_commission_on_the_market_net():
    scored = pd.DataFrame({"market_id": ["a", "a", "b"], "day": ["2024-01-01"] * 3, "month": ["2024-01"] * 3,
                           "sport": "horse", "country": "uk", "market_type": "WIN",
                           "open_price": [4.0, 10.0, 3.0], "close_price": [2.0, 20.0, 3.0],
                           "y": np.log([0.5, 2.0, 1.0]), "pred": [-0.3, 0.3, 0.0]})
    tr = backtest.hedged_trades(scored, "pred", 0.1)
    assert list(tr["side"]) == [1, -1]                       # back the steamer, lay the drifter; nothing at 0
    # back 1 at 4, lay 2 at 2: wins 3 - 2 = 1, loses -1 + 2 = 1: o/c - 1 = 1 either way
    assert tr.loc[0, "profit"] == pytest.approx(4.0 / 2.0 - 1.0)
    # lay 1 at 10 (liability 9), back 0.5 at 20: horse wins -9 + 9.5 = 0.5, loses 1 - 0.5 = 0.5: 1 - o/c
    assert tr.loc[1, "profit"] == pytest.approx(1.0 - 10.0 / 20.0)
    assert tr["right_way"].all()
    net = backtest.net_by_market(tr, 0.02)
    assert net["net"].sum() == pytest.approx(1.5 * 0.98)      # one market, commission on its net of 1.5
    s = backtest.summarise(tr, 0.02, n_boot=50)
    assert s["bets"] == 2 and s["roi"] == pytest.approx(1.5 * 0.98 / 2)
    # a losing market pays no commission
    scored2 = scored.assign(close_price=[8.0, 5.0, 3.0], y=np.log([2.0, 0.5, 1.0]))
    tr2 = backtest.hedged_trades(scored2, "pred", 0.1)
    assert backtest.net_by_market(tr2, 0.02)["net"].sum() == pytest.approx((4 / 8 - 1) + (1 - 10 / 5))
    assert not tr2["right_way"].any()


def test_spread_costs_a_tick_each_end():
    scored = pd.DataFrame({"market_id": ["a", "a"], "day": ["2024-01-01"] * 2, "month": ["2024-01"] * 2,
                           "sport": "horse", "country": "uk", "market_type": "WIN",
                           "open_price": [4.0, 10.0], "close_price": [3.0, 12.0],
                           "y": np.log([0.75, 1.2]), "pred": [-0.3, 0.3]})
    mid = backtest.hedged_trades(scored, "pred", 0.1)
    sp = backtest.hedged_trades(scored, "pred", 0.1, spread=True)
    assert list(sp["fill_open"]) == [3.95, 10.5] and list(sp["fill_close"]) == [3.05, 11.5]
    assert (sp["profit"] < mid["profit"]).all()


def test_sweep_and_groups_and_drawdown(prepared):
    scored = drift.walk_forward(prepared, [drift.BandShrinkage(min_rows=20)], min_train_months=3)
    sw = backtest.sweep(scored, "pred_band_ols", thresholds=(0.0, 0.05, 0.5), n_boot=100)
    assert list(sw["threshold"]) == [0.0, 0.05, 0.5]
    assert sw.loc[0, "bets"] >= sw.loc[1, "bets"]
    assert sw.loc[1, "roi_spread"] < sw.loc[1, "roi"]
    assert sw.loc[1, "roi_alt_commission"] <= sw.loc[1, "roi"]
    by = backtest.by_group(scored, "pred_band_ols", 0.05, "band", n_boot=20)
    assert by["bets"].sum() == sw.loc[1, "bets"]
    by_c = backtest.by_group(scored, "pred_band_ols", 0.05, "country", n_boot=20)
    assert set(by_c.index) == {"uk", "ire"}
    assert backtest.drawdown(np.array([1.0, -2.0, 0.5, -1.0])) == pytest.approx(2.5)   # 1 -> -1.5
    assert backtest.drawdown(np.array([1.0, 1.0])) == 0.0


def test_run_panel_holdout_is_last_months_and_thresholds_are_fixed_there(tmp_path):
    p = synthetic_panel(n_markets=500, seed=2)
    res = run.run_panel(p, "synthetic", holdout_months=3, min_train_months=4, refit_every=6, n_jobs=1,
                        n_boot=50, shuffle=True)
    months = sorted(features.prepare(p)["month"].unique())
    assert res["holdout_months"] == [months[-3], months[-1]]
    assert res["development_months"][1] == months[-4]
    assert res["development"]["months"][1] <= months[-4]
    assert res["holdout"]["months"] == [months[-3], months[-1]]
    for name, e in res["holdout"]["models"].items():
        assert e["threshold"] == res["development"]["models"][name]["threshold"]
    assert "shuffle_test" in res and "band_ols" in res["shuffle_test"]["accuracy"]
    md = run.report_md(res)
    assert "Holdout (read once)" in md and "Leak test" in md
    # the CLI end to end
    path = tmp_path / "panel.csv"
    p.to_csv(path, index=False)
    out, rep = tmp_path / "o.json", tmp_path / "r.md"
    assert run.main(["--panel", str(path), "--out", str(out), "--report", str(rep), "--min-train-months", "4",
                     "--refit-every", "6", "--n-boot", "20", "--n-jobs", "1", "--no-shuffle"]) == 0
    assert out.exists() and "Open-close drift model" in rep.read_text()


def test_split_months_from_a_date():
    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
    assert run.split_months(months, 3, None) == (["2026-01", "2026-02"], ["2026-03", "2026-04", "2026-05"])
    assert run.split_months(months, 3, "2026-04") == (["2026-01", "2026-02", "2026-03"], ["2026-04", "2026-05"])


def test_football_panel_from_gold_rows():
    gold = pd.DataFrame({
        "match_id": ["x1", "x2", "x3"], "status": ["FINISHED", "FINISHED", "SCHEDULED"], "country": "ENG",
        "competition_id": "ENG1", "season": [2024, 2025, 2025], "home_team": ["A", "B", "C"], "away_team": ["D", "E", "F"],
        "kickoff_utc": ["2024-08-17T14:00:00Z", "2025-08-16T14:00:00Z", "2025-08-23T14:00:00Z"],
        "ft_home": [2, 0, np.nan], "ft_away": [1, 0, np.nan],
        "open_h": [2.1, 3.0, 2.0], "open_d": [3.4, 3.3, 3.0], "open_a": [3.6, 2.5, 4.0],
        "close_h": [2.0, 3.2, np.nan], "close_d": [3.5, 3.3, np.nan], "close_a": [3.9, 2.4, np.nan],
        "open_1x2_src": "PS", "close_1x2_src": "PSC",
        "open_o25": [1.9, np.nan, 1.9], "open_u25": [2.0, 1.9, 2.0], "close_o25": [1.85, 2.0, 1.9], "close_u25": [2.05, 1.9, 2.0],
        "open_ou_src": "P", "close_ou_src": "PC",
        "open_ah_line": [-0.5, 0.25, 0.0], "close_ah_line": [-0.5, 0.0, 0.0], "open_ah_h": [1.95, 1.9, 1.9],
        "open_ah_a": [1.95, 2.0, 2.0], "close_ah_h": [1.9, 1.95, 1.9], "close_ah_a": [2.0, 1.95, 2.0],
        "open_ah_src": "PAH", "close_ah_src": "PCAH",
        "bfe_open_h": [2.2, 3.1, 2.0], "bfe_open_d": [3.5, 3.4, 3.0], "bfe_open_a": [3.7, 2.6, 4.0],
        "bfe_close_h": [2.1, 3.3, np.nan], "bfe_close_d": [3.6, 3.4, np.nan], "bfe_close_a": [4.0, 2.5, np.nan],
        "bfe_open_o25": np.nan, "bfe_open_u25": np.nan, "bfe_close_o25": np.nan, "bfe_close_u25": np.nan,
    })
    p = panels.football_panel(gold)
    # x1: 1X2 (3), O/U (2), AH (2, same line); x2: 1X2 (3), no O/U open on one side, AH line moved; x3 not finished
    assert p.groupby("market_type").size().to_dict() == {"ASIAN_HANDICAP": 2, "MATCH_ODDS": 6, "OVER_UNDER_25": 2}
    h = p[(p["market_id"] == "x1:MATCH_ODDS") & (p["selection_id"] == "H")].iloc[0]
    assert h["result"] == 1.0 and h["open_price"] == 2.1 and h["close_price"] == 2.0
    assert str(h["off_time_utc"].tz) == "UTC"
    b = panels.football_panel(gold, venue="bfe")
    assert b.groupby("market_type").size().to_dict() == {"MATCH_ODDS": 6}


def test_horses_panel_times_and_model_join():
    prices = pd.DataFrame({"country": ["uk", "uk", "ire"], "market_type": "win", "race_date": "2026-06-01",
                           "race_time": ["14:30", "14:30", "17:05"], "event_id": [1, 1, 2], "event_name": "r",
                           "selection_id": [11, 12, 21], "selection_name": ["a", "b", "c"], "win_lose": [1, 0, 0],
                           "bsp": [2.5, 4.0, 3.0], "ppwap": [2.6, 3.9, 3.1], "morningwap": [3.0, 3.5, 2.9],
                           "morning_vol": [500, 20, 0], "pp_vol": [5000, 800, 100], "race_results_id": [5, 6, None],
                           "track": ["Ascot", "Ascot", None], "rr_time": ["2.30.", "2.30.", None],
                           "horse_name": ["A (IRE)", "B", None]})
    fc = pd.DataFrame({"race_date": ["2026-06-01"], "track": ["Ascot"], "race_time": ["2.30."],
                       "horse_name": ["A (IRE)"], "predicted_bfsp": [2.2]})
    p = panels.horses_panel(prices, fc)
    assert p["off_time_utc"].iloc[0] == pd.Timestamp("2026-06-01 13:30", tz="UTC")       # BST is UTC+1
    assert list(p["model_price"].fillna(-1)) == [2.2, -1, -1]
    assert list(p["n_runners"]) == [2, 2, 1] and p["market_type"].eq("WIN").all()
    assert p["open_price"].tolist() == [3.0, 3.5, 2.9] and p["close_price"].tolist() == [2.5, 4.0, 3.0]


def test_greyhound_panel():
    bf = pd.DataFrame({"market_id": ["1.1", "1.1", "1.2"], "event_id": [1, 1, 2], "event_name": "e",
                       "market_type": ["WIN", "WIN", "PLACE"], "country": "GB", "market_time": "2026-03-01T19:04:00.000Z",
                       "selection_id": [1, 2, 3], "runner_name": ["d1", "d2", "d3"],
                       "runner_status": ["WINNER", "LOSER", "LOSER"], "won": [1, 0, 0], "bsp": [2.0, 5.0, 2.0],
                       "ltp_first": [2.5, 4.0, 2.0], "ltp_close": [2.1, 4.8, 2.0], "ltp_t_1440": np.nan,
                       "ltp_t_60": [2.4, 4.2, 2.0], "ltp_t_15": [2.2, 4.6, 2.0], "ltp_t_1": [2.1, 4.8, 2.0]})
    p = panels.greyhound_panel(bf)
    assert len(p) == 2 and p["sport"].eq("greyhound").all()
    assert p["p_t60"].tolist() == [2.4, 4.2] and p["open_price"].tolist() == [2.5, 4.0]
    assert p["close_price"].tolist() == [2.1, 4.8]
    assert panels.greyhound_panel(bf, close="bsp")["close_price"].tolist() == [2.0, 5.0]


# ---------------------------------------------------------------------------
# the CI query (research/queries/open_close_horses_1101.py): its logic on synthetic tables
# ---------------------------------------------------------------------------

def _query_module():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "research" / "queries" / "open_close_horses_1101.py"
    if not path.exists():
        path = path.parent / "done" / path.name
    spec = importlib.util.spec_from_file_location("open_close_horses_1101", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_query_lay_cell_logic():
    q = _query_module()
    marks = pd.DataFrame({
        "race_date": ["2026-10-05"] * 5 + ["2026-10-06"] * 2,
        "market_id": ["1.1"] * 5 + ["1.2"] * 2, "selection_id": [1, 1, 1, 2, 2, 3, 3],
        "mark": ["clock_09:00", "clock_10:00", "final", "clock_09:00", "final", "clock_11:00", "final"],
        "minutes_to_off": [300, 240, -5, 300, -5, 200, -5],
        "back1": [28, 30, None, 60, None, 24, None], "back1_size": [10, 10, None, 5, None, 30, None],
        "lay1": [30.0, 32.0, None, 70.0, None, 26.0, None], "lay1_size": [25.0, 10.0, None, 30.0, None, 50.0, None],
        "bsp": [None, None, 40.0, None, 100.0, None, 20.0], "country": "GB", "market_type": "WIN", "venue": "v"})
    t = q.lay_trades(marks)
    assert len(t) == 4 and set(t["band"]) == {"21-51", "51+"}
    r1 = t[(t["selection_id"] == 1) & (t["mark"] == "clock_09:00")].iloc[0]
    assert r1["profit"] == pytest.approx(1 - 30 / 40) and r1["liability"] == 29.0 and r1["first_mark"]
    assert not t[(t["selection_id"] == 1) & (t["mark"] == "clock_10:00")].iloc[0]["first_mark"]
    r3 = t[t["selection_id"] == 3].iloc[0]
    assert r3["profit"] == pytest.approx(1 - 26 / 20)                 # the lay lost: the dog shortened
    s = q.lay_summary(t, commission=0.02, n_boot=50)
    assert s["bets"] == 4 and s["days"] == 2
    # market 1.1: profits 0.25 + 0.2 + 0.3 = 0.75, net 0.98 of it; market 1.2: -0.3 (no commission on a loss)
    net = 0.75 * 0.98 - 0.3
    assert s["return_on_liability"] == pytest.approx(net / (29 + 31 + 69 + 25))
    cells = q.lay_cells(t)
    pre = cells[cells["pre_registered"]]
    assert len(pre) == 1 and pre.iloc[0]["band"] == "21-51" and pre.iloc[0]["min_size"] == 20.0
    assert pre.iloc[0]["bets"] == 2                                   # sel 1 at 09:00 (25 offered), sel 3 at 11:00


def test_query_runs_end_to_end_on_a_synthetic_db(tmp_path, monkeypatch, capsys):
    import sqlite3
    q = _query_module()
    panel = synthetic_panel(n_markets=300, seed=3)
    off_uk = panel["off_time_utc"].dt.tz_convert("Europe/London")
    prices = pd.DataFrame({"country": panel["country"], "market_type": "win",
                           "race_date": off_uk.dt.strftime("%Y-%m-%d"), "race_time": off_uk.dt.strftime("%H:%M"),
                           "event_id": panel["event_id"], "event_name": panel["event_name"],
                           "selection_id": panel["selection_id"], "selection_name": panel["selection_name"],
                           "win_lose": panel["result"], "bsp": panel["close_price"], "ppwap": panel["close_price"],
                           "morningwap": panel["open_price"], "morning_vol": panel["open_matched"], "pp_vol": 1000.0,
                           "race_results_id": np.arange(len(panel)) + 1})
    rr = pd.DataFrame({"id": prices["race_results_id"], "race_date": prices["race_date"], "track": "T",
                       "race_time": "2.30.",
                       "horse_name": "h" + prices["selection_id"].astype(str) + "_" + prices["event_id"].astype(str)})
    fc = rr.assign(predicted_bfsp=prices["morningwap"] * 0.9)[["race_date", "race_time", "track", "horse_name",
                                                                "predicted_bfsp"]]
    fc_path = tmp_path / "oos.csv"
    fc.to_csv(fc_path, index=False)
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    # one October 2026 price row, the BSP the lay cell's fill reads for runner 2 (a lone row: not a market to model)
    extra = prices.iloc[:1].copy()
    extra["race_date"], extra["race_results_id"], extra["bsp"], extra["event_id"] = "2026-10-05", 999999, 50.0, 999999
    pd.concat([prices, extra], ignore_index=True).to_sql("betfair_prices", conn, index=False)
    rr.to_sql("race_results", conn, index=False)
    # runner 1's final mark holds its BSP; runner 2's does not (the record before 7 Oct): the price files fill it
    marks = pd.DataFrame({"race_date": "2026-10-05", "market_id": "1.9", "selection_id": [1, 1, 2, 2],
                          "mark": ["clock_09:00", "final", "clock_10:00", "final"], "minutes_to_off": [200, -5, 140, -5],
                          "back1": [28, None, 24, None], "back1_size": [10, None, 10, None],
                          "lay1": [30.0, None, 25.0, None], "lay1_size": [25.0, None, 30.0, None],
                          "bsp": [None, 40.0, None, None]})
    marks.to_sql("betfair_live_marks", conn, index=False)
    pd.DataFrame({"market_id": ["1.9", "1.9"], "selection_id": [1, 2], "country": "GB", "market_type": "WIN",
                  "venue": "v", "race_results_id": [None, 999999]}).to_sql("betfair_live_markets", conn, index=False)
    conn.commit()
    conn.close()
    monkeypatch.setattr(q, "DB", str(db))
    monkeypatch.setattr(q, "FORECASTS", fc_path)
    monkeypatch.setattr(q, "FROM", "2024-01-01")
    monkeypatch.setattr(q, "HOLDOUT_FROM", "2025-06")
    monkeypatch.setattr(q, "N_BOOT", 20)
    monkeypatch.setattr(q, "N_JOBS", 1)
    assert q.main(["--sport", "horse,lay"]) == 0
    text = capsys.readouterr().out
    j = json.loads(text.split("=== JSON ===")[1])
    assert j["errors"] == {}
    assert [r["name"] for r in j["results"]] == ["horse_uk_win", "horse_ire_win"]
    uk = j["results"][0]
    assert uk["with_model"] and uk["holdout_months"][0] == "2025-06"
    assert "gbm_model" in uk["development"]["accuracy"] and "holdout" in uk
    assert j["lay_cell"]["pre_registered"]["all_runner_marks"]["bets"] == 2
    assert "final marks with a BSP: 1 from the record, 1 more from the price files" in text
