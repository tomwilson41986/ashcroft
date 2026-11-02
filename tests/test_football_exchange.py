"""The football exchange panel, models and scoring: mechanics on synthetic rows only (never a reported figure)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.open_close import drift
from research.open_close import football_exchange as fx


def synthetic_gold(n_weeks: int = 60, per_week: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gold rows (bfe early and close, scores) and silver rows (bookmakers' early prices) for made-up matches from
    Aug 2024; the bookmakers' fair price is the truth, Betfair's early price a noisy copy that closes on it."""
    rng = np.random.default_rng(seed)
    gold, silver = [], []
    start = pd.Timestamp("2024-08-02")
    for w in range(n_weeks):
        for k in range(per_week):
            mid = f"m{w}_{k}"
            day = (start + pd.Timedelta(days=7 * w + int(rng.integers(0, 4)))).strftime("%Y-%m-%d")
            p = rng.dirichlet([4, 2.5, 3])
            q = p * np.exp(rng.normal(0, 0.15, 3))
            q /= q.sum()
            bfe_open = 1.0 / (q * 1.02)
            bfe_close = 1.0 / (p * 1.005)
            po = rng.uniform(0.4, 0.6)
            qo = np.clip(po * np.exp(rng.normal(0, 0.08)), 0.05, 0.95)
            div = "E0" if k % 2 else "E1"
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            gold.append({"match_id": mid, "status": "FINISHED", "kickoff_utc": pd.Timestamp(day + " 15:00", tz="UTC"),
                         "match_date": day, "country": "ENG", "home_team": f"h{k}", "away_team": f"a{k}",
                         "ft_home": hg, "ft_away": ag, "season": 2024 + (w >= 40), "competition_id": div,
                         "fd_code": div,
                         **{f"bfe_open_{o}": round(float(v), 2) for o, v in zip("hda", bfe_open)},
                         **{f"bfe_close_{o}": round(float(v), 2) for o, v in zip("hda", bfe_close)},
                         "bfe_open_o25": round(1 / (qo * 1.02), 2), "bfe_open_u25": round(1 / ((1 - qo) * 1.02), 2),
                         "bfe_close_o25": round(1 / (po * 1.005), 2), "bfe_close_u25": round(1 / ((1 - po) * 1.005), 2)})
            row = {"match_id": mid}
            for o, v in zip("HDA", p):
                row[f"PS{o}"] = 1 / (v * 1.05)
                row[f"Avg{o}"] = 1 / (v * 1.07)
                row[f"Max{o}"] = 1 / (v * 1.02)
                row[f"B365{o}"] = 1 / (v * 1.06)
            for b in ("P", "Avg", "Max", "B365"):
                row[f"{b}>2.5"], row[f"{b}<2.5"] = 1 / (po * 1.05), 1 / ((1 - po) * 1.05)
            silver.append(row)
    return pd.DataFrame(gold), pd.DataFrame(silver)


@pytest.fixture(scope="module")
def panel():
    gold, silver = synthetic_gold()
    dc = pd.DataFrame({"match_id": gold["match_id"], "p_h": 0.45, "p_d": 0.27, "p_a": 0.28, "p_o25": 0.5})
    d, prov = fx.exchange_panel(gold, silver, dc)
    return d, prov, gold


def test_panel_features_from_the_early_books(panel):
    d, prov, gold = panel
    assert prov["panel_rows"] == len(gold) * 5 and prov["duplicate_selection_rows_dropped"] == 0
    mo = d[d.market_type == "MATCH_ODDS"]
    # each book sums over its own market; the fair prices make the gap a log ratio of fair prices
    np.testing.assert_allclose(mo["ps_over"], 1.05, rtol=1e-9)
    over = (1 / mo["open_raw"]).groupby(mo["market_id"]).transform("sum")
    np.testing.assert_allclose(mo["bfe_over_open"], over)
    fair_ps = mo["ps"] * mo["ps_over"]
    np.testing.assert_allclose(mo["gap_ps"], np.log(mo["open_raw"] * over / fair_ps))
    # Pinnacle's goals prices come from P>2.5 / P<2.5, its 1X2 from PS*; never a closing column
    ou = d[d.market_type == "OVER_UNDER_25"]
    assert ou["ps"].notna().all() and np.allclose(ou["ps_over"], 1.05)
    # the model's chance per leg, the under the complement of the over
    assert np.allclose(d.loc[d.selection_id == "Under 2.5", "dc_p"], 0.5)
    assert np.allclose(d.loc[d.selection_id == "H", "dc_p"], 0.45)
    assert set(d["pin_period"]) == {"fresh", "stale"}
    assert (d.loc[d.match_date >= fx.PIN_STALE_FROM, "pin_period"] == "stale").all()
    assert set(d["tier"]) == {"top five", "rest"}


def test_book_width_labels():
    lab = fx.width_label([1.0, 1.019, 1.02, 1.05, 1.0501, 1.4])
    assert list(lab) == [fx.WIDTH_LABELS[0], fx.WIDTH_LABELS[0], fx.WIDTH_LABELS[1], fx.WIDTH_LABELS[1],
                         fx.WIDTH_LABELS[2], fx.WIDTH_LABELS[2]]


def test_days_to_kickoff_follow_the_collection_rule():
    days = pd.Series(["2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04",
                      "2026-11-05"])                                  # Fri .. Thu
    assert list(fx.days_to_kickoff(days)) == [0, 1, 2, 3, 0, 1, 2]


def test_season_stage_counts_earlier_days_only():
    g = pd.DataFrame({"match_id": list("abcd"), "competition_id": "E0", "season": 2024,
                      "match_date": ["2024-08-01", "2024-08-01", "2024-08-08", "2024-08-15"]})
    st = fx.season_stage(g)
    assert st["a"] == 0 and st["b"] == 0 and st["c"] == 0.5 and st["d"] == 0.75


def test_gap_rule_and_ols(panel):
    d = fx.prepared(panel[0])
    rule = fx.GapRule("gap_ps", "gap_rule_ps")
    np.testing.assert_allclose(rule.fit(d).predict(d), -d["gap_ps"].to_numpy())
    ols = fx.GapOLS("gap_ps", "gap_ols_ps").fit(d)
    a, b = ols.coef
    assert b < 0                                                     # Betfair longer than the bookmaker: it shortens
    x = d.assign(gap_ps=np.nan)
    assert np.all(ols.predict(x) == 0.0) and np.all(rule.predict(x) == 0.0)


def test_gbm_uses_the_named_features_only(panel):
    d = fx.prepared(panel[0])
    m = drift.GradientBoosting(name="g", features=["log_open", "gap_ps", "not_a_column"], n_estimators=10).fit(d)
    assert m.columns == ["log_open", "gap_ps"]
    assert len(m.predict(d)) == len(d)


def test_walk_forward_scores_after_the_fit_and_holdout_is_never_developed(panel):
    d = fx.prepared(panel[0])
    mo = d[d.market_type == "MATCH_ODDS"]
    sc = drift.walk_forward(mo, fx.exchange_models(n_jobs=1)[:6], min_train_months=fx.MIN_TRAIN_MONTHS)
    assert (sc["fit_month"] < sc["month"]).all()
    assert {"pred_gap_rule_ps", "pred_gap_ols_ps", "pred_gap_rule_avg"} <= set(sc.columns)


def test_quoted_back_takes_the_recorded_back_and_lays_a_tick_over(panel):
    d = fx.prepared(panel[0])
    d["pred_x"] = -d["gap_ps"]
    q = fx.quoted_back_trades(d, "pred_x", 0.05)
    assert (q["side"] > 0).all() and len(q)
    rows = q["row"].to_numpy()
    np.testing.assert_allclose(q["fill_open"], d["open_raw"].to_numpy()[rows])
    assert (q["fill_close"].to_numpy() > d["close_raw"].to_numpy()[rows]).all()
    np.testing.assert_allclose(q["profit"], q["fill_open"] / q["fill_close"] - 1)


def test_clv_is_the_value_at_the_price_taken():
    t = pd.DataFrame({"side": [1, -1], "fill_open": [3.0, 2.0], "close_price": [2.5, 2.5]})
    np.testing.assert_allclose(fx.clv(t), [0.2, 0.2])


def test_rule_rows_filters_and_the_gate(panel):
    d = fx.prepared(panel[0])
    rule = {"model": "x", "market_types": ["MATCH_ODDS"], "leagues": "top5", "book_width": [fx.WIDTH_LABELS[0]],
            "threshold": 0.05, "gate": {"min_bets": 3, "min_days": 2}}
    r = fx.rule_rows(d, rule)
    assert (r.market_type == "MATCH_ODDS").all() and (r.top5 > 0).all() and (r.book_width == fx.WIDTH_LABELS[0]).all()
    d["pred_x"] = -d["gap_ps"]
    res = fx.score_rule(d, rule, n_boot=50)
    g = fx.gate(res, rule)
    assert set(g["checks"]) == {"bets", "days", "roi_lo90_above_0", "clv_above_0"}
    assert g["passed"] == all(g["checks"].values())


def test_read_prereg_takes_the_json_block(tmp_path):
    p = tmp_path / "pre.md"
    p.write_text("# rule\n\ntext\n\n```json\n" + json.dumps({"model": "gap_rule_ps", "threshold": 0.1}) + "\n```\n")
    rule, digest = fx.read_prereg(str(p))
    assert rule == {"model": "gap_rule_ps", "threshold": 0.1} and len(digest) == 64
    (tmp_path / "bad.md").write_text("no block")
    with pytest.raises(SystemExit):
        fx.read_prereg(str(tmp_path / "bad.md"))


def test_holdout_scores_only_the_registered_rules_after_the_boundary(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(fx, "HOLDOUT_FROM", "2025-06")
    rule = {"model": "gap_rule_avg", "market_types": ["OVER_UNDER_25"], "leagues": "all",
            "book_width": list(fx.WIDTH_LABELS[:2]), "threshold": 0.05, "gate": {"min_bets": 5, "min_days": 2},
            "secondary": [{"model": "gap_rule_ps", "market_types": ["MATCH_ODDS"], "leagues": "all",
                           "book_width": list(fx.WIDTH_LABELS[:2]), "threshold": 0.10}]}
    p = tmp_path / "pre.md"
    p.write_text("```json\n" + json.dumps(rule) + "\n```\n")
    res = fx.holdout(panel[0], str(p), n_jobs=1, n_boot=20, log=None)
    for r in res["markets"].values():
        assert r["holdout"]["months"][0] >= "2025-06"
    assert res["rule_result"]["rule"]["model"] == "gap_rule_avg"
    assert len(res["secondary"]) == 1 and res["secondary"][0]["rule"]["model"] == "gap_rule_ps"
    assert "passed" in res["gate"] and "passed" in res["secondary"][0]["gate"]
    md = fx.tables(json.loads(fx._json(res)))
    assert "The pre-registered rule" in md and "Secondary rule 1" in md
