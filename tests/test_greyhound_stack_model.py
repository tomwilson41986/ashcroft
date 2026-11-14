"""greyhound/stack_model.py, the live price model from 10 Nov 2026: it prices the rows the fast loop's ``stack_share``
recipe priced, the same; it survives a save and a load; it refuses rows not grouped by race."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound import fastloop as fl

torch = pytest.importorskip("torch")

from greyhound import nn, stack_model as sm  # noqa: E402
from tests.test_greyhound_fastloop import QUICK, synthetic_cache  # noqa: E402


@pytest.fixture()
def small(monkeypatch):
    monkeypatch.setattr(nn, "MAX_EPOCHS", 4)
    df, meta = synthetic_cache()
    feats = fl.feature_set(meta, "card_safe")
    recipe = {"learning_rate": QUICK["learning_rate"], "early_stop": QUICK["early_stop"], "rounds": QUICK["rounds"],
              "max_bin": QUICK["max_bin"], "seeds": 2, "nn_seeds": 2, "nn_seconds": 120.0}
    return df, meta, feats, recipe


def test_the_live_model_prices_as_the_fast_loop_scored_it(small):
    df, meta, feats, recipe = small
    exp = {"name": "s", "features": "card_safe", "objective": "stack_share", "seeds": 2, "nn_seeds": 2,
           "nn_seconds": 120, "calibrate": ["stretch_track"], "serve_calibrated": "stretch_track"}
    res, pred = fl.run_experiment(df, meta, exp, QUICK, test_from="2026-06-01")
    bundle = sm.fit(df, feats, before="2026-06-01", recipe=recipe, train_from=QUICK["train_from"], log=lambda m: None)
    assert bundle["stack"] == res["stack"]
    test = df[df.t >= "2026-06-01"].reset_index(drop=True)
    got = sm.price(bundle, test)
    assert np.allclose(got.p_model.to_numpy(), pred.p_model.to_numpy(), atol=1e-6)
    assert np.allclose(got.pred_bsp.to_numpy(), pred.pred_bsp.to_numpy(), rtol=1e-5)
    assert np.allclose(got.pred_bsp_raw.to_numpy(), pred.pred_bsp_raw.to_numpy(), rtol=1e-5)
    assert np.allclose(got.groupby(test.race_id).p_model.sum(), 1.0)


def test_a_saved_model_prices_the_same(small):
    df, meta, feats, recipe = small
    bundle = sm.fit(df, feats, before="2026-06-01", recipe=recipe, train_from=QUICK["train_from"], log=lambda m: None)
    again = sm.from_bytes(sm.to_bytes(bundle))
    test = df[df.t >= "2026-06-01"].reset_index(drop=True)
    a, b = sm.price(bundle, test), sm.price(again, test)
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-6)
    assert again["meta"]["recipe"] == "stack_ml" and len(again["trees"]) == 2 and len(again["nets"]) == 2


def test_rows_must_be_grouped_by_race(small):
    df, meta, feats, recipe = small
    bundle = sm.fit(df, feats, before="2026-06-01", recipe=dict(recipe, seeds=1, nn_seeds=1),
                    train_from=QUICK["train_from"], log=lambda m: None)
    test = df[df.t >= "2026-06-01"].reset_index(drop=True)
    shuffled = test.iloc[np.r_[np.arange(0, len(test), 2), np.arange(1, len(test), 2)]]
    with pytest.raises(ValueError):
        sm.price(bundle, shuffled)


def test_the_window_rolls_back_29_months():
    assert sm.window_start("2026-11-10", 29) == "2024-06-10"
