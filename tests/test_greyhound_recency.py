"""Recency row weights for the share trees and the networks (review section 5: ``recency_half_life_days``): off by
default and the fit unchanged without it; when set, the trees and the networks get the same weights, rising to the
test's start, and the live stack's fit mirrors the fast loop's. Constructed frames for the unit tests only."""
from __future__ import annotations

import numpy as np
import pytest

from greyhound import fastloop as fl

torch = pytest.importorskip("torch")

from greyhound import nn, stack_model as sm  # noqa: E402
from tests.test_greyhound_fastloop import QUICK, synthetic_cache  # noqa: E402

EXP = {"name": "s", "features": "card_safe", "objective": "stack_share", "seeds": 1, "nn_seeds": 1, "nn_seconds": 60}


@pytest.fixture()
def small(monkeypatch):
    monkeypatch.setattr(nn, "MAX_EPOCHS", 3)
    df, meta = synthetic_cache(n_days=60, races_a_day=10)
    return df, meta


def spy(monkeypatch):
    seen = {"nn": [], "obj": []}
    real_fit, real_obj = nn.fit, fl.race_soft_objective

    def fit(*a, **k):
        seen["nn"].append(k.get("weight"))
        return real_fit(*a, **k)

    def obj(race, target, weight=None):
        seen["obj"].append(weight)
        return real_obj(race, target, weight)
    monkeypatch.setattr(nn, "fit", fit)
    monkeypatch.setattr(fl, "race_soft_objective", obj)
    return seen


def test_the_spec_is_checked():
    fl.check_spec({"experiments": [dict(EXP, recency_half_life_days=180)]})
    for bad in (dict(EXP, recency_half_life_days=0), dict(EXP, recency_half_life_days=90, halflife_days=90),
                dict(EXP, objective="binary", recency_half_life_days=90)):
        with pytest.raises(ValueError):
            fl.check_spec({"experiments": [bad]})


def test_off_by_default_no_weight_reaches_either_fit(small, monkeypatch):
    df, meta = small
    seen = spy(monkeypatch)
    fl.run_experiment(df, meta, dict(EXP), dict(QUICK, train_from="2026-03-01"), test_from="2026-04-15")
    assert seen["nn"] == [None] and seen["obj"][0] is None


def test_uniform_weights_leave_the_network_bit_identical():
    rng = np.random.default_rng(0)
    race = np.repeat(np.arange(60), 5)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    t = np.tile([0.4, 0.3, 0.15, 0.1, 0.05], 60)
    a = nn.fit(X[:250], race[:250], t[:250], X[250:], race[250:], t[250:], seconds=30, seed=1)
    b = nn.fit(X[:250], race[:250], t[:250], X[250:], race[250:], t[250:], seconds=30, seed=1, weight=np.ones(250))
    for k, v in a["net"].state_dict().items():
        assert torch.equal(v, b["net"].state_dict()[k])
    c = nn.fit(X[:250], race[:250], t[:250], X[250:], race[250:], t[250:], seconds=30, seed=1,
               weight=np.linspace(0.1, 1.0, 250))
    assert not all(torch.equal(v, c["net"].state_dict()[k]) for k, v in a["net"].state_dict().items())


def test_set_the_trees_and_networks_share_the_weights_and_the_live_fit_mirrors_it(small, monkeypatch):
    df, meta = small
    seen = spy(monkeypatch)
    q = dict(QUICK, train_from="2026-03-01")
    res, _ = fl.run_experiment(df, meta, dict(EXP, recency_half_life_days=20), q, test_from="2026-04-15")
    w_nn, w_tree = seen["nn"][0], seen["obj"][0]
    assert w_nn is not None and w_tree is not None and np.array_equal(w_nn, w_tree)
    assert w_nn.max() <= 1.0 and w_nn.min() < 0.5 and np.all(np.diff(w_nn) >= -1e-12)   # rows in time order
    feats = fl.feature_set(meta, "card_safe")
    recipe = {"learning_rate": q["learning_rate"], "early_stop": q["early_stop"], "rounds": q["rounds"],
              "max_bin": q["max_bin"], "seeds": 1, "nn_seeds": 1, "nn_seconds": 60.0, "recency_half_life_days": 20}
    bundle = sm.fit(df, feats, before="2026-04-15", recipe=recipe, train_from=q["train_from"], log=lambda m: None)
    assert bundle["stack"] == res["stack"]
    assert np.array_equal(seen["nn"][-1], w_nn)
