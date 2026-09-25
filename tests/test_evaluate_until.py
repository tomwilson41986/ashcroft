"""evaluate_oos.walk_forward_predict with --eval-until never scores a row on or after the date.

The model is replaced by a stub: this tests the fold walk, not the booster.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

import evaluate_oos


def _stub(monkeypatch):
    def fit(train_df, feature_cols, cfg):
        return SimpleNamespace(booster=None, best_iteration=1, early_stopped=True, n_holdout=0,
                               holdout_start=train_df["race_date"].max(), holdout_metrics={"mae": 0.0},
                               init_offset=0.0)

    def predict(model, df, feature_cols, **kw):
        df["predicted_bfsp"] = 5.0
        return df

    monkeypatch.setattr(evaluate_oos, "fit_bfsp", fit)
    monkeypatch.setattr(evaluate_oos, "predict_prices", predict)


def test_no_fold_scores_past_the_end_date(monkeypatch):
    _stub(monkeypatch)
    days = pd.date_range("2024-01-01", "2026-09-20", freq="D")
    df = pd.DataFrame({"race_date": np.repeat(days, 3), "raceid": np.repeat(np.arange(len(days)), 3),
                       "x": np.random.default_rng(0).normal(size=3 * len(days))})
    out = evaluate_oos.walk_forward_predict(df, ["x"], min_train_days=365, val_window_days=91, step_days=91,
                                            eval_from="2025-06-01", eval_until="2026-04-01")
    assert len(out) > 0
    assert out["race_date"].max() < pd.Timestamp("2026-04-01")
    assert out["race_date"].min() >= pd.Timestamp("2025-06-01")
    # without the end date the same walk runs into the holdout
    full = evaluate_oos.walk_forward_predict(df, ["x"], min_train_days=365, val_window_days=91, step_days=91,
                                             eval_from="2025-06-01")
    assert full["race_date"].max() >= pd.Timestamp("2026-04-01")
