"""The residual screen's estimator and plumbing.

Synthetic races here test the arithmetic only: that the conditional logit
recovers coefficients it was given, that a feature carrying information the
price lacks scores above zero out of sample and noise does not. No model is
trained or evaluated on generated data; the screen itself runs on the real
database (.github/workflows/residual-screen.yml).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("residual_screen", ROOT / "scripts" / "residual_screen.py")
rs = importlib.util.module_from_spec(spec)
sys.modules["residual_screen"] = rs
spec.loader.exec_module(rs)


def _races(n_races=3000, size=8, seed=0, beta_mkt=1.0, beta_x=0.5, noise_col=True):
    """Races whose winner is drawn from softmax(beta_mkt·ln π + beta_x·x), where
    the 'market' π deliberately ignores x."""
    rng = np.random.default_rng(seed)
    n = n_races * size
    codes = np.repeat(np.arange(n_races), size)
    strength = rng.normal(size=n)
    x = rng.normal(size=n)
    ln_pi_raw = strength
    starts, seg = rs.race_blocks(codes)
    ln_pi = ln_pi_raw - np.log(np.add.reduceat(np.exp(ln_pi_raw), starts))[seg]
    eta = beta_mkt * ln_pi + beta_x * x
    p = rs.softmax_blocks(eta, starts, seg)
    y = np.zeros(n)
    u = rng.random(n_races)
    cum = np.add.accumulate(p)
    base = np.r_[0, np.add.accumulate(np.add.reduceat(p, starts))[:-1]]
    for r in range(n_races):
        s = starts[r]
        c = np.cumsum(p[s:s + size])
        y[s + int(np.searchsorted(c, u[r] * c[-1]))] = 1
    noise = rng.normal(size=n) if noise_col else None
    return codes, y, ln_pi, x, noise


def test_race_loglik_matches_brute_force():
    codes = np.array([0, 0, 0, 1, 1])
    eta = np.array([0.2, -1.0, 0.5, 2.0, 0.1])
    y = np.array([0, 0, 1, 0, 1.0])
    starts, seg = rs.race_blocks(codes)
    got = rs.race_loglik(eta, y, starts, seg)
    want = [0.5 - np.log(np.exp([0.2, -1.0, 0.5]).sum()), 0.1 - np.log(np.exp([2.0, 0.1]).sum())]
    assert np.allclose(got, want)
    assert np.allclose(rs.uniform_loglik(starts, len(codes)), [-np.log(3), -np.log(2)])


def test_fit_clogit_recovers_coefficients():
    codes, y, ln_pi, x, _ = _races(n_races=4000, beta_mkt=1.0, beta_x=0.5)
    starts, seg = rs.race_blocks(codes)
    b = rs.fit_clogit(np.column_stack([ln_pi, x]), y, starts, seg)
    assert b[0] == pytest.approx(1.0, abs=0.08)
    assert b[1] == pytest.approx(0.5, abs=0.06)


def test_chunked_hessian_equals_direct():
    rng = np.random.default_rng(3)
    codes = np.repeat(np.arange(500), 6)
    starts, seg = rs.race_blocks(codes)
    X = rng.normal(size=(len(codes), 45))
    p = rs.softmax_blocks(X @ rng.normal(size=45) * 0.1, starts, seg)
    direct = rs._neg_hessian(X, p, starts, None)
    old = rs.CHUNK_ROWS
    try:
        rs.CHUNK_ROWS = 700
        chunked = rs._neg_hessian(X, p, starts, rs._chunk_starts(starts, len(codes)))
    finally:
        rs.CHUNK_ROWS = old
    assert np.allclose(direct, chunked)


def test_ridge_shrinks_only_the_penalised_terms():
    codes, y, ln_pi, x, _ = _races(n_races=1500)
    starts, seg = rs.race_blocks(codes)
    X = np.column_stack([ln_pi, x])
    free = rs.fit_clogit(X, y, starts, seg)
    pen = rs.fit_clogit(X, y, starts, seg, ridge=[0.0, 1e5])
    assert abs(pen[1]) < 0.05 * abs(free[1])
    assert pen[0] == pytest.approx(free[0], abs=0.15)


def _screen_for(codes, y, ln_pi, split_at):
    n_races = codes.max() + 1
    race_train = np.arange(n_races) < split_at
    tr = race_train[codes]
    pos = np.arange(len(codes))
    M = rs.market_columns(ln_pi)
    return rs.OutcomeScreen([(rs.Part(pos[tr], codes[tr], y[tr], M[tr]),
                              rs.Part(pos[~tr], codes[~tr], y[~tr], M[~tr]))]), tr


def test_outcome_screen_finds_information_the_price_lacks_and_not_noise():
    codes, y, ln_pi, x, noise = _races(n_races=4000, beta_x=0.5)
    screen, tr = _screen_for(codes, y, ln_pi, 2500)
    F_x, _ = rs.feature_design(x, tr)
    F_n, _ = rs.feature_design(noise, tr)
    g_x = rs.gain_summary(screen.gains(F_x), screen.null)
    g_n = rs.gain_summary(screen.gains(F_n), screen.null)
    assert g_x["t"] > 5 and g_x["dll_mnats"] > 10
    assert g_n["t"] < 2.5 and g_n["dll_mnats"] < 2
    # a feature that is just the price again adds nothing either
    F_p, _ = rs.feature_design(ln_pi, tr)
    assert rs.gain_summary(screen.gains(F_p), screen.null)["dll_mnats"] < 2


def test_alone_scores_against_random():
    codes, y, ln_pi, x, _ = _races(n_races=3000, beta_x=0.5)
    screen, tr = _screen_for(codes, y, ln_pi, 2000)
    F_x, _ = rs.feature_design(x, tr)
    alone = screen.gains(F_x, with_market=False)
    assert alone.sum() > 0                          # x alone beats picking at random
    assert screen.market_r2() > 0.05


def test_base_extra_absorbs_what_it_already_explains():
    codes, y, ln_pi, x, _ = _races(n_races=4000, beta_x=0.6)
    screen, tr = _screen_for(codes, y, ln_pi, 2500)
    F_x, _ = rs.feature_design(x, tr, quadratic=False)
    over_market = screen.gains(F_x).mean()
    over_index = screen.gains(F_x, base_extra=x).mean()   # the index already is x
    assert over_market > 0.005
    assert abs(over_index) < 0.002


def test_feature_design_learns_constants_on_training_rows_only():
    v = np.r_[np.arange(1000.0), np.full(1000, 1e6)]       # the test half is wild
    tr = np.r_[np.ones(1000, bool), np.zeros(1000, bool)]
    F, kind = rs.feature_design(v, tr)
    assert kind == "numeric" and F.shape[1] == 2
    assert np.allclose(F[tr, 0].mean(), 0, atol=1e-9)
    assert F[~tr, 0].max() < 2.0                        # clipped at the training 99.5 %
    v2 = v.copy(); v2[:300] = np.nan
    F2, _ = rs.feature_design(v2, tr)
    assert F2.shape[1] == 3 and F2[:300, 2].sum() == 300   # missing flag
    assert rs.feature_design(np.ones(2000), tr)[1] == "constant"
    Fb, kb = rs.feature_design(np.r_[np.zeros(1000), np.ones(1000)] [np.random.default_rng(0).permutation(2000)], np.ones(2000, bool))
    assert kb == "binary" and Fb.shape[1] == 1


def test_categorical_levels():
    v = np.array(["a"] * 600 + ["b"] * 300 + ["c"] * 250 + ["d"] * 5, dtype=object)
    F, kind = rs.feature_design(v, np.ones(len(v), bool), categorical=True)
    assert kind == "category" and F.shape[1] == 2          # b and c; a is the base, d too rare


def test_within_race_variation():
    starts, _ = rs.race_blocks(np.repeat(np.arange(100), 5))
    race_level = np.repeat(np.arange(100.0), 5)
    assert rs.within_race_varies(race_level, starts) == 0.0
    assert rs.within_race_varies(np.arange(500.0), starts) == 1.0


def test_move_screen_detects_a_feature_that_moves_the_price():
    rng = np.random.default_rng(5)
    n_races, size = 2000, 8
    codes = np.repeat(np.arange(n_races), size)
    pos = np.arange(len(codes))
    ln_pi_m = rng.normal(size=len(codes))
    x, noise = rng.normal(size=len(codes)), rng.normal(size=len(codes))
    move = 0.3 * x + rng.normal(scale=0.5, size=len(codes))
    M = rs.market_columns(ln_pi_m)
    month_a = (np.arange(n_races) % 2 == 0)[codes]
    a = rs.Part(pos[month_a], codes[month_a], np.zeros(month_a.sum()), M[month_a])
    b = rs.Part(pos[~month_a], codes[~month_a], np.zeros((~month_a).sum()), M[~month_a])
    ms = rs.MoveScreen([(a, b), (b, a)], move)
    gx = ms.gains(rs.feature_design(x, month_a)[0])
    gn = ms.gains(rs.feature_design(noise, month_a)[0])
    assert gx.sum() / ms.sst > 0.1
    assert gn.sum() / ms.sst < 0.01


def test_boosted_gain_starts_from_the_market():
    codes, y, ln_pi, x, noise = _races(n_races=3000, beta_x=0.7)
    n_races = codes.max() + 1
    pos = np.arange(len(codes))
    M = rs.market_columns(ln_pi)
    r = np.arange(n_races)[codes]
    parts = [rs.Part(pos[m], codes[m], y[m], M[m]) for m in (r < 1500, (r >= 1500) & (r < 2000), r >= 2000)]
    bm = rs.fit_clogit(parts[0].M, parts[0].y, *parts[0].blk)
    X = np.column_stack([x, noise]).astype(np.float32)
    d, used = rs.boosted_gain(X, parts[0], parts[1], parts[2], bm,
                              params={"min_data_in_leaf": 200, "learning_rate": 0.1}, rounds=300, early=30)
    assert len(d) == 1000 and used >= 1
    assert d.mean() > 0
    d2, used2 = rs.boosted_gain(X, parts[0], None, parts[2], bm, params={"min_data_in_leaf": 200}, rounds=5)
    assert used2 == 5


def test_markdown_table():
    t = rs.markdown_table(pd.DataFrame({"a": ["x"], "b": [0.12345], "c": [np.nan], "d": [12.3456]}))
    assert t.splitlines()[2] == "| x | 0.1235 |  | 12.35 |"


def test_run_end_to_end_on_a_tiny_frame(tmp_path, monkeypatch):
    """Plumbing only: every stage runs and writes its outputs."""
    rng = np.random.default_rng(11)
    n_races, size = 600, 7
    codes = np.repeat(np.arange(n_races), size)
    dates = pd.to_datetime("2024-06-01") + pd.to_timedelta((np.arange(n_races) // 2)[codes], unit="D")
    strength = rng.normal(size=len(codes))
    p = rs.softmax_blocks(strength, *rs.race_blocks(codes))
    won = np.zeros(len(codes))
    starts, _ = rs.race_blocks(codes)
    for i, s in enumerate(starts):
        won[s + rng.choice(size, p=p[s:s + size] / p[s:s + size].sum())] = 1
    from model.bfsp_features import ALL_FEATURE_COLS
    feats = list(dict.fromkeys(ALL_FEATURE_COLS))[:6]
    df = pd.DataFrame({"race_date": dates, "track": "Ascot", "race_time": (codes % 2 + 1).astype(str) + ".00",
                       "horse_name": [f"h{i}" for i in range(len(codes))], "won": won,
                       "bfsp": 1.0 / p * rng.uniform(0.97, 1.03, size=len(codes))})
    for j, f in enumerate(feats):
        df[f] = rng.normal(size=len(codes)) if not f.endswith("_cat") else rng.integers(0, 4, size=len(codes))
    mp = pd.DataFrame({"race_date": df["race_date"].dt.strftime("%Y-%m-%d"), "track": df["track"],
                       "race_time": df["race_time"], "horse_name": df["horse_name"],
                       "morningwap": df["bfsp"] * np.exp(rng.normal(scale=0.2, size=len(df))),
                       "morning_vol": 500.0, "bf_bsp": df["bfsp"]})
    monkeypatch.setattr(rs, "load_frame", lambda *a, **k: df.copy())
    monkeypatch.setattr(rs, "morning_prices", lambda *a, **k: mp)
    monkeypatch.setattr(rs, "MIN_RACES_TO_SCORE", 50)
    args = rs.argparse.Namespace(db="x", feature_cache=None, start_date="2021-01-01", split="2025-01-01",
                                 lockbox_from="2026-04-01",
                                 val_months=3, ridge_grid=[1.0, 10.0], blocks="", limit=0, no_alone=False,
                                 no_boost=False, no_morning=False, out_dir=str(tmp_path))
    res = rs.run(args)
    per = pd.read_csv(tmp_path / "per_feature.csv")
    assert set(per["feature"]) == set(feats)
    assert {"bsp_dll_mnats", "morning_dll_mnats", "move_dR2_x1000", "alone_R2"} <= set(per.columns)
    assert "bsp_all" in res["joint_linear"] and "boosted_bsp" in res
    assert (tmp_path / "summary.md").read_text().startswith("# Residual information screen")
    json.loads((tmp_path / "screen.json").read_text())
