"""The rank-1 evaluation harness: betting arithmetic and walk-forward plumbing.

Synthetic races test arithmetic only; the model is trained and scored on the
real database (.github/workflows/research-loop.yml)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("outcome_model", ROOT / "scripts" / "outcome_model.py")
om = importlib.util.module_from_spec(spec)
sys.modules["outcome_model"] = om
spec.loader.exec_module(om)
rs = om.rs


def test_bet_returns_charge_commission_on_winnings_only():
    r = om.bet_returns(np.array([1, 0, 1.0]), np.array([3.0, 3.0, 1.5]), commission=0.05)
    assert r == pytest.approx([2.0 * 0.95, -1.0, 0.5 * 0.95])


def test_expected_value_zero_at_fair_price_without_commission():
    assert om.expected_value(np.array([0.25]), np.array([4.0]), commission=0.0)[0] == pytest.approx(0.0)
    assert om.expected_value(np.array([0.25]), np.array([4.0]), commission=0.05)[0] == pytest.approx(0.25 * 3 * 0.95 - 0.75)


def test_pick_rank1_one_per_race_first_on_ties():
    race = np.array(["a", "a", "a", "b", "b"])
    score = np.array([0.2, 0.5, 0.5, 0.9, 0.1])
    assert om.pick_rank1(race, score).tolist() == [False, True, False, True, False]


def _oos(n_races=400, size=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        p = rng.dirichlet(np.ones(size) * 2)
        w = rng.choice(size, p=p)
        for i in range(size):
            rows.append({"race": f"2025-{1 + r % 12:02d}-01|X|{r}", "date": pd.Timestamp(f"2025-{1 + r % 12:02d}-01"),
                         "horse_name": f"h{r}_{i}", "bsp": 1.0 / p[i], "won": float(i == w),
                         "p_model": p[i], "p_mkt": p[i]})
    return pd.DataFrame(rows)


def test_score_strategies_on_a_fair_market():
    oos = _oos()
    s = om.score_strategies(oos, commission=0.0)
    assert s["rank1"]["bets"] == 400 and s["fav"]["bets"] == 400
    assert s["rank1_is_fav_share"] == 1.0                     # same probabilities, same pick
    assert s["rank1"]["lo90"] <= s["rank1"]["roi"] <= s["rank1"]["hi90"]
    assert set(s["rank1_by_year"]) == {2025}
    assert len(s["rank1_halves"]) == 2
    # at fair prices with no commission, no rank-1 bet has positive EV
    assert s["rank1_ev>0"]["bets"] == 0


def test_walk_forward_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n_races, size = 900, 7
    codes = np.repeat(np.arange(n_races), size)
    dates = pd.to_datetime("2022-06-01") + pd.to_timedelta((np.arange(n_races) * 1.2).astype(int)[codes], unit="D")
    x = rng.normal(size=len(codes))
    strength = rng.normal(size=len(codes))
    starts, seg = rs.race_blocks(codes)
    p_true = rs.softmax_blocks(strength + 0.6 * x, starts, seg)
    p_mkt = rs.softmax_blocks(strength, starts, seg)                 # the market misses x
    won = np.zeros(len(codes))
    for s in starts:
        won[s + rng.choice(size, p=p_true[s:s + size] / p_true[s:s + size].sum())] = 1
    from model.bfsp_features import ALL_FEATURE_COLS
    f0, f1 = [c for c in dict.fromkeys(ALL_FEATURE_COLS) if not c.endswith("_cat")][:2]
    df = pd.DataFrame({"race_date": dates, "track": "York", "race_time": (codes % 3 + 1).astype(str) + ".30",
                       "horse_name": [f"h{i}" for i in range(len(codes))], "won": won, "bfsp": 1.0 / p_mkt,
                       f0: x, f1: rng.normal(size=len(codes))})
    monkeypatch.setattr(rs, "load_frame", lambda *a, **k: df.copy())
    args = om.argparse.Namespace(db="x", feature_cache=None, start_date="2021-01-01", first_fold="2023-06-01",
                                 fold_months=6, val_months=4, lockbox_from="2025-01-01", final=False, drop="",
                                 blocks="", params=json.dumps({"min_data_in_leaf": 50, "learning_rate": 0.1}),
                                 tag="t", out_dir=str(tmp_path), mode="offset")
    res = om.run(args)
    assert res["folds"] and all(f["races"] > 0 for f in res["folds"])
    oos = pd.read_csv(tmp_path / "oos_t.csv.gz")
    assert pd.to_datetime(oos["date"]).max() < pd.Timestamp("2025-01-01")    # lockbox respected
    assert pd.to_datetime(oos["date"]).min() >= pd.Timestamp("2023-06-01")
    assert res["loglik"]["dll_mnats"] > 0                                    # it finds x
    entry = json.loads((tmp_path / "ledger_entry_t.json").read_text())
    assert entry["kind"] == "dev" and entry["config"]["features"] == 2
    assert (tmp_path / "summary_t.md").read_text().startswith("# Outcome model `t`")


def test_kelly_path_grows_on_a_real_edge_and_never_bets_without_one():
    rng = np.random.default_rng(4)
    n = 4000
    p_true = rng.uniform(0.2, 0.5, n)
    price = 1.0 / (p_true * 0.9)                                  # the price is 10% too long
    won = (rng.random(n) < p_true).astype(float)
    k = om.kelly_path(p_true, price, won, np.ones(n, bool), commission=0.0)
    assert k["final_bank"] > 1.5 and 0 < k["max_drawdown"] < 1
    none = om.kelly_path(p_true, 1.0 / p_true * 0.95, won, np.ones(n, bool), commission=0.0)
    assert none["turnover"] == 0.0 and none["final_bank"] == 1.0   # negative edge: no stake


def test_segments_report_each_level():
    oos = _oos(n_races=900)
    oos["race_type"] = np.where(oos["race"].str.endswith(("0", "2", "4", "6", "8")), "Handicap", "Maiden")
    oos["number_of_runners"] = 6
    s = om.score_strategies(oos, commission=0.0)
    assert set(s["segments"]["race_type"]) == {"Handicap", "Maiden"}
    assert s["segments"]["field"]["6-8"]["races"] == 900


def test_free_mode_runs_and_finds_the_signal(tmp_path, monkeypatch):
    """Benter's two stages: a market-free fundamental model, then Stage C."""
    rng = np.random.default_rng(2)
    n_races, size = 900, 7
    codes = np.repeat(np.arange(n_races), size)
    dates = pd.to_datetime("2022-06-01") + pd.to_timedelta((np.arange(n_races) * 1.2).astype(int)[codes], unit="D")
    x = rng.normal(size=len(codes))
    strength = rng.normal(size=len(codes))
    starts, seg = rs.race_blocks(codes)
    p_true = rs.softmax_blocks(strength + 0.8 * x, starts, seg)
    p_mkt = rs.softmax_blocks(strength, starts, seg)
    won = np.zeros(len(codes))
    for s in starts:
        won[s + rng.choice(size, p=p_true[s:s + size] / p_true[s:s + size].sum())] = 1
    from model.bfsp_features import ALL_FEATURE_COLS
    f0 = [c for c in dict.fromkeys(ALL_FEATURE_COLS) if not c.endswith("_cat")][0]
    df = pd.DataFrame({"race_date": dates, "track": "York", "race_time": (codes % 3 + 1).astype(str) + ".30",
                       "horse_name": [f"h{i}" for i in range(len(codes))], "won": won, "bfsp": 1.0 / p_mkt, f0: x})
    monkeypatch.setattr(rs, "load_frame", lambda *a, **k: df.copy())
    args = om.argparse.Namespace(db="x", feature_cache=None, start_date="2021-01-01", first_fold="2023-06-01",
                                 fold_months=6, val_months=4, lockbox_from="2025-01-01", final=False, drop="",
                                 blocks="", params=json.dumps({"min_data_in_leaf": 50, "learning_rate": 0.1}),
                                 tag="free", out_dir=str(tmp_path), mode="free")
    res = om.run(args)
    assert res["config"]["mode"] == "free"
    assert res["loglik"]["dll_mnats"] > 0
