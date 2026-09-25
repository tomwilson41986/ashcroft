"""The fast retrain: one fit at a known round count, from the cached matrix, no stage 2.

Synthetic frames test the mechanics only; nothing is trained for use on them.
"""
import json

import numpy as np
import pandas as pd
import pytest

import train_bfsp
from model.bfsp_model import DEFAULT_PARAMS, TrainConfig, fit_bfsp


def _matrix(days=200, per_day=8, seed=0):
    """A frame shaped like the prepared matrix: served feature names, a price, a race."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("2024-01-01", periods=days, freq="D")
    n = days * per_day
    x = rng.normal(size=(n, 3))
    df = pd.DataFrame({
        "race_date": np.repeat(d, per_day), "race_time": "2.30", "track": "york",
        "horse_name": [f"h{i}" for i in range(n)], "raceid": np.repeat(np.arange(days), per_day).astype(str),
        "rPMW3": x[:, 0], "LR_ORR2": x[:, 1], "or_num": x[:, 2],
        "placing_numerical": np.tile(np.arange(1, per_day + 1), days).astype(float),
    })
    df["bfsp"] = np.exp(1.5 + 0.6 * x[:, 0] + 0.2 * rng.normal(size=n)).clip(1.1, 500)
    df["log_bfsp"] = np.log(df["bfsp"])
    df["won"] = (df["placing_numerical"] == 1).astype(float)
    return df


SMALL = {**DEFAULT_PARAMS, "num_leaves": 7, "min_child_samples": 10}


def test_a_fixed_round_fit_is_one_booster_at_exactly_that_count():
    df = _matrix()
    fit = fit_bfsp(df, ["rPMW3", "LR_ORR2", "or_num"], TrainConfig(fixed_rounds=37, params=SMALL))
    assert fit.booster.num_trees() == 37 and fit.best_iteration == 37
    assert fit.refit and not fit.early_stopped and fit.n_train == len(df)
    assert fit.holdout_metrics["in_sample"] is True
    # the same thing twice: deterministic
    again = fit_bfsp(df, ["rPMW3", "LR_ORR2", "or_num"], TrainConfig(fixed_rounds=37, params=SMALL))
    X = df[["rPMW3", "LR_ORR2", "or_num"]].astype(float)
    np.testing.assert_array_equal(fit.booster.predict(X), again.booster.predict(X))


def test_the_holdout_protocol_is_unchanged_without_it():
    df = _matrix()
    fit = fit_bfsp(df, ["rPMW3", "LR_ORR2", "or_num"],
                   TrainConfig(num_boost_round=60, params=SMALL, refit_on_full=True))
    assert fit.n_train < len(df) and "in_sample" not in fit.holdout_metrics


def test_training_from_the_prepared_matrix_skips_the_engine_and_stage_2(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(train_bfsp.CustomMetricsEngine, "calculate_all",
                        lambda self, df: called.append("engine") or df)
    trainer = train_bfsp.BFSPTrainer(cfg=TrainConfig(fixed_rounds=20, params=SMALL), prob_model=False)
    summary = trainer.train(_matrix(), output_dir=str(tmp_path), prepared=True)
    assert "error" not in summary and not called
    assert summary["has_prob_model"] is False
    assert set(trainer.feature_cols) == {"rPMW3", "LR_ORR2", "or_num"}
    meta = json.loads((tmp_path / "bfsp_model_meta.json").read_text())
    assert meta["best_iteration"] == 20 and meta["fixed_rounds"] == 20
    assert not (tmp_path / "probability_model.lgb").exists()
