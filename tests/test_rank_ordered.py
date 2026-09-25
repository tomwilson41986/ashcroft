"""The rank-ordered (Plackett-Luce) likelihood in scripts/outcome_model.py.

Depth 1 must be the race softmax the model always fitted; the objective's
gradient must be the likelihood's; the stage discount must recover a known one;
non-finishers and dead heats must be handled. Simulated orders test the
mechanics only; nothing is trained for use on them.
"""
import numpy as np
import pytest

from scripts import outcome_model as om
from scripts import residual_screen as rs


def _races(n_races=300, runners=8, seed=0, discount=(1.0, 0.7, 0.5)):
    """Finishing orders drawn from a Plackett-Luce model on utility u, each place
    below the first at its discount."""
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(n_races), runners)
    u = rng.normal(size=n_races * runners)
    pos = np.full(len(u), np.nan)
    for r in range(n_races):
        idx = np.arange(r * runners, (r + 1) * runners)
        left = list(idx)
        for k, lam in enumerate(discount, start=1):
            w = np.exp(lam * u[left])
            pick = rng.choice(len(left), p=w / w.sum())
            pos[left.pop(pick)] = k
        for j, i in enumerate(left):          # the rest, in any order
            pos[i] = len(discount) + 1 + j
    starts, seg = rs.race_blocks(codes)
    return u, pos, starts, seg


def test_depth_one_is_the_race_softmax():
    u, pos, starts, seg = _races()
    (in_set, label), = om.pl_stages(pos, starts, seg, 1)
    assert in_set.all() and np.array_equal(label, (pos == 1).astype(float))
    np.testing.assert_allclose(om.masked_softmax(u, in_set, starts, seg), rs.softmax_blocks(u, starts, seg))


def test_the_objective_is_the_likelihoods_gradient():
    u, pos, starts, seg = _races(n_races=5, runners=5)
    stages = om.pl_stages(pos, starts, seg, 3)
    lams = [1.0, 0.8, 0.6]

    def nll(eta):
        return -sum(om.stage_loglik(l * eta, st, starts, seg) for l, st in zip(lams, stages))

    g = np.zeros_like(u)
    for lam, (in_set, label) in zip(lams, stages):
        p = om.masked_softmax(lam * u, in_set, starts, seg)
        g += lam * (p - label)
    eps = 1e-6
    num = np.array([(nll(u + eps * np.eye(len(u))[i]) - nll(u - eps * np.eye(len(u))[i])) / (2 * eps)
                    for i in range(len(u))])
    np.testing.assert_allclose(g, num, atol=1e-6)


def test_the_stage_discount_is_recovered():
    u, pos, starts, seg = _races(n_races=4000, discount=(1.0, 0.7, 0.5), seed=3)
    lams = om.fit_stage_discounts(u, om.pl_stages(pos, starts, seg, 3), starts, seg)
    assert lams[0] == 1.0
    assert abs(lams[1] - 0.7) <= 0.1 and abs(lams[2] - 0.5) <= 0.1


def test_non_finishers_stay_in_every_set_and_dead_heats_share():
    codes = np.array([0, 0, 0, 0, 1, 1, 1])
    pos = np.array([1, 1, 3, np.nan, 2, 1, np.nan])        # race 0: a dead heat for first; one pulled up
    starts, seg = rs.race_blocks(codes)
    st = om.pl_stages(pos, starts, seg, 3)
    in1, lab1 = st[0]
    assert lab1.tolist() == [0.5, 0.5, 0, 0, 0, 1, 0]
    in2, lab2 = st[1]
    # race 0 has no second (the dead heat took 1 and 2): no stage there; race 1: chosen from the unplaced
    assert not in2[:4].any() and lab2[:4].sum() == 0
    assert in2[4:].tolist() == [True, False, True] and lab2[4:].tolist() == [1, 0, 0]
    in3, lab3 = st[2]
    assert in3[:4].tolist() == [False, False, True, True] and lab3[:4].tolist() == [0, 0, 1, 0]
    assert not in3[4:].any()                                # race 1 has no third finisher
