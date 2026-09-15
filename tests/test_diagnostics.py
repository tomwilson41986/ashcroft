"""Known-answer tests for the forecast diagnostics (model/diagnostics.py)."""

import numpy as np
import pandas as pd
import pytest

from model import diagnostics as dg


@pytest.fixture
def calibrated():
    rng = np.random.default_rng(0)
    p = rng.beta(2, 8, 20000)
    y = (rng.random(20000) < p).astype(float)
    return y, p


def test_murphy_identity_holds_exactly(calibrated):
    y, p = calibrated
    d = dg.murphy_decomposition(y, p, n_bins=10)
    assert abs(d["identity_gap"]) < 1e-10
    assert d["reliability"] < 0.001          # calibrated by construction
    assert d["resolution"] > 0.005           # informative forecasts
    assert abs(d["uncertainty"] - y.mean() * (1 - y.mean())) < 1e-12


def test_murphy_bias_correction_shrinks_reliability(calibrated):
    y, p = calibrated
    d = dg.murphy_decomposition(y[:500], p[:500], n_bins=10)
    assert d["reliability_c"] < d["reliability"]
    assert d["uncertainty_c"] > d["uncertainty"]


def test_skill_scores_sign():
    rng = np.random.default_rng(1)
    p = rng.beta(2, 8, 5000); y = (rng.random(5000) < p).astype(float)
    worse = np.clip(p + rng.normal(0, 0.15, 5000), 0.01, 0.99)
    assert dg.brier_skill_score(y, p, worse) > 0
    assert dg.log_loss_skill_score(y, p, worse) > 0
    assert abs(dg.brier_skill_score(y, p, p)) < 1e-12


def test_rps_prefers_mass_near_outcome():
    close = np.array([[0.6, 0.3, 0.1]]); far = np.array([[0.6, 0.1, 0.3]])
    assert dg.ranked_probability_score(close, [0]) < dg.ranked_probability_score(far, [0])


def test_crps_ensemble_matches_gaussian_closed_form():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (2000, 400))
    assert abs(dg.crps_ensemble(X, np.zeros(2000)) - dg.crps_gaussian(0.0, 1.0, np.zeros(2000))) < 0.01


def test_concordance_index_extremes():
    race = np.repeat(np.arange(200), 8); pos = np.tile(np.arange(1, 9), 200)
    assert dg.concordance_index(-pos, pos, race) == 1.0
    assert dg.concordance_index(pos, pos, race) == 0.0
    rnd = dg.concordance_index(np.random.default_rng(3).random(1600), pos, race)
    assert 0.45 < rnd < 0.55


def test_js_divergence_symmetric_and_bounded():
    p, q = [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]
    assert abs(dg.js_divergence(p, q) - dg.js_divergence(q, p)) < 1e-12
    assert 0 < dg.js_divergence(p, q) <= np.log(2)
    assert dg.js_divergence(p, p) < 1e-12


def test_scoring_report_on_race_frame():
    rng = np.random.default_rng(4)
    rows = []
    for r in range(300):
        n = rng.integers(6, 12)
        strength = rng.normal(0, 1, n)
        p_true = np.exp(strength) / np.exp(strength).sum()
        winner = rng.choice(n, p=p_true)
        for i in range(n):
            rows.append({"race_date": "2025-01-01", "race_time": f"{r}.00", "track": "T", "horse_name": f"h{i}",
                         "predicted_win_prob_norm": p_true[i] * rng.uniform(0.8, 1.2), "bfsp": 1 / p_true[i] * 1.05,
                         "won": float(i == winner), "placing_numerical": 1 if i == winner else 2})
    rep = dg.scoring_report(pd.DataFrame(rows))
    assert rep["n_races"] == 300
    assert 0.4 < rep["model_concordance"] < 1.0
    assert "brier_skill_vs_market" in rep and np.isfinite(rep["brier_skill_vs_market"])
    assert isinstance(rep["reliability"], pd.DataFrame)
