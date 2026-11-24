"""research/overfit.pbo_cscv: noise picks look like coin flips, a real edge survives (constructed returns; a code
test, not an evaluation)."""
import numpy as np

from research.overfit import pbo_cscv


def test_pure_noise_gives_a_pbo_near_a_half():
    r = np.random.default_rng(1).normal(0.0, 1.0, size=(400, 30))
    out = pbo_cscv(r, n_splits=10)
    assert out["splits"] == 252 and 0.3 < out["pbo"] < 0.7


def test_one_real_edge_among_noise_gives_a_pbo_near_zero():
    r = np.random.default_rng(2).normal(0.0, 1.0, size=(400, 30))
    r[:, 7] += 0.5
    assert pbo_cscv(r, n_splits=10)["pbo"] < 0.05


def test_too_little_to_split_says_so():
    assert pbo_cscv(np.zeros((10, 1)))["pbo"] is None
    assert pbo_cscv(np.zeros((10, 5)), n_splits=16)["pbo"] is None
