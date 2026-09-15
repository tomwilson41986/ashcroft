"""Structural tests for the race ABM (engine behaviour, not model evaluation).

The synthetic fields here exercise the simulator's mechanics; they are
not used to train or evaluate anything (see CLAUDE.md rule on sample data).
"""

import numpy as np
import pandas as pd
import pytest

from model.abm import (AbilityConfig, PatternTargets, RaceSimulator, SimConfig, Track, abc_smc, build_field,
                       pattern_loss, simulate_race, simulated_patterns, summarise)
from model.abm.features import ABM_FEATURES, add_abm_features, merge_abm_features, place_terms
from model.abm.track import going_speed_factor


def _field(n=8, seed=0, **over):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "race_date": ["2026-03-12"] * n, "race_time": ["2.30"] * n, "track": ["Kempton"] * n,
        "horse_name": [f"H{i}" for i in range(n)],
        "official_rating": np.linspace(90, 70, n).round(), "pounds": np.linspace(140, 120, n).round(),
        "horse_perf_lbs_ewm": np.linspace(95, 65, n), "dist_furlongs": [8.0] * n,
        "going_description": ["Standard"] * n, "stall": list(range(1, n + 1)),
        "pred_early_pos": rng.uniform(1.5, 5.8, n), "career_runs": [10] * n, "horse_age": [4] * n,
        "race_type": ["Handicap"] * n, "bfsp": np.linspace(3, 40, n),
    })
    for k, v in over.items():
        d[k] = v
    return d


def test_track_geometry_and_going():
    t = Track.from_race(8.0, "Good To Soft", "Kempton")
    assert not t.is_straight and 0 < t.bend_fraction < 1
    assert t.curvature_at(np.array([0.0, t.bend_start_m + 1, t.length_m - 1]))[1] > 0
    s = Track.from_race(6.0, "Good", "Ascot")
    assert s.is_straight and s.curvature_at(np.array([500.0]))[0] == 0
    assert going_speed_factor("Heavy") < going_speed_factor("Good") < going_speed_factor("Firm")


def test_build_field_uses_perf_figures_and_weight():
    d = _field()
    f = build_field(d)
    assert f.n == 8 and abs(f.ability_lbs.mean()) < 1e-9
    assert f.ability_lbs[0] > f.ability_lbs[-1]
    assert f.v_cp[0] > f.v_cp[-1]
    assert (f.w_prime > 0).all() and (f.kick_dist_m > 0).all()


def test_probabilities_sum_to_one_and_follow_ability():
    feats, res, res_solo = simulate_race(_field(), n_sims=400, seed=1)
    assert abs(feats["abm_win_prob"].sum() - 1) < 1e-9
    assert abs(feats["abm_win_prob_solo"].sum() - 1) < 1e-9
    assert abs(feats["abm_place_prob"].sum() - place_terms(8, True)) < 1e-9
    # solo reference is monotone in ability (30 lbs range)
    assert feats["abm_win_prob_solo"].iloc[0] > feats["abm_win_prob_solo"].iloc[-1]
    assert feats["abm_exp_beaten_l"].iloc[0] < feats["abm_exp_beaten_l"].iloc[-1]
    assert set(ABM_FEATURES).issubset(feats.columns)
    assert (res.times > 0).all() and (res.positions.min(axis=1) == 1).all()


def test_solo_reference_ignores_draw_and_style():
    d = _field(n=6, official_rating=[80] * 6, pounds=[130] * 6, horse_perf_lbs_ewm=[80] * 6,
               pred_early_pos=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], track=["Chester"] * 6, dist_furlongs=[5.0] * 6)
    feats, _, _ = simulate_race(d, n_sims=1500, seed=2)
    assert feats["abm_win_prob_solo"].max() - feats["abm_win_prob_solo"].min() < 0.08


def test_interactions_change_outcomes_and_report_pace():
    feats, res, _ = simulate_race(_field(n=10), n_sims=300, seed=3)
    assert feats["abm_pace_delta"].abs().sum() > 0
    assert 0 <= feats["abm_p_trouble"].max() <= 1
    assert np.isfinite(feats["abm_pace_contest"].iloc[0]) and np.isfinite(feats["abm_race_entropy"].iloc[0])
    assert (res.led_early.sum(axis=1) == 1).all()


def test_deterministic_with_seed():
    a, _, _ = simulate_race(_field(), n_sims=100, seed=9)
    b, _, _ = simulate_race(_field(), n_sims=100, seed=9)
    pd.testing.assert_frame_equal(a, b)


def test_market_abilities_rank_like_the_market():
    d = _field(n=8)
    feats, _, _ = simulate_race(d, n_sims=800, seed=4, abilities="market", ability_cfg=AbilityConfig(market_beta=0.16))
    mkt = (1 / d["bfsp"]) / (1 / d["bfsp"]).sum()
    assert np.corrcoef(np.log(mkt), np.log(feats["abm_win_prob_solo"].clip(1e-3)))[0, 1] > 0.9


def test_add_and_merge_abm_features():
    d = pd.concat([_field(n=6, seed=0), _field(n=7, seed=1, race_time=["3.00"] * 7)], ignore_index=True)
    out = add_abm_features(d, n_sims=60, seed=0, log_every=0)
    assert out["abm_win_prob"].notna().all()
    keyed = out[["race_date", "race_time", "track", "horse_name"] + ABM_FEATURES]
    merged = merge_abm_features(d, keyed.assign(race_date=pd.to_datetime(keyed["race_date"])))
    assert merged["abm_win_prob"].notna().all() and len(merged) == len(d)


def test_calibration_harness_runs():
    d = _field(n=8)
    fld = build_field(d); tr = Track.from_race(8.0, "Standard", "Kempton")
    res = RaceSimulator(SimConfig()).run(fld, tr, n_sims=100, seed=0)
    pat = simulated_patterns([res])
    obs = PatternTargets(0.1, 0.4, 0.3, 1.2, 4.0, -0.05)
    assert np.isfinite(pattern_loss(pat, obs))
    table = abc_smc(lambda th: (th["a"] - 0.3) ** 2 + (th["b"] - 2.0) ** 2, {"a": (0, 1), "b": (0, 5)},
                    n_particles=30, n_rounds=4, seed=0, verbose=False)
    best = table.iloc[0]
    assert abs(best["a"] - 0.3) < 0.1 and abs(best["b"] - 2.0) < 0.5


def test_place_terms():
    assert place_terms(4) == 1 and place_terms(7) == 2 and place_terms(12) == 3
    assert place_terms(16, is_handicap=True) == 4 and place_terms(16, is_handicap=False) == 3
