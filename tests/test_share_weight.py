"""Row weights by market share (TrainConfig.share_weight): each training row weighted by its
share of its race's Betfair SP book to a power, normalised to a mean of 1.

Synthetic frames test mechanics only; nothing is trained or evaluated on them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.bfsp_model import TrainConfig, build_target, fit_bfsp, predict_prices, row_weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_race_xent import SMALL, _history  # noqa: E402


def test_off_by_default_every_row_counts_alike():
    df = _history(n_days=30)
    assert TrainConfig().share_weight == 0.0
    assert row_weights(df, TrainConfig()) is None


def test_a_row_counts_for_its_share_of_the_race_to_the_power():
    df = _history(n_days=30)
    share = build_target(df, "race_xent")
    for power in (0.5, 1.0):
        w = row_weights(df, TrainConfig(share_weight=power))
        assert w.mean() == pytest.approx(1.0)
        # within a race, the weights stand in the ratio of the shares to the power
        np.testing.assert_allclose(w / w[0], (share / share[0]) ** power, rtol=1e-9)


def test_it_multiplies_the_recency_weight():
    df = _history(n_days=30)
    both = row_weights(df, TrainConfig(share_weight=0.5, decay_rate=1.0))
    alone = row_weights(df, TrainConfig(share_weight=0.5))
    days = (df["race_date"].max() - pd.to_datetime(df["race_date"])).dt.days.to_numpy(float)
    np.testing.assert_allclose(both, alone * np.exp(-days / 365.0), rtol=1e-9)


def test_race_xent_already_counts_a_runner_for_its_share():
    with pytest.raises(ValueError, match="race_xent"):
        row_weights(_history(n_days=10), TrainConfig(target="race_xent", share_weight=0.5))


def test_a_weighted_fit_prices_books_of_one_and_records_the_power():
    df = _history(n_days=120)
    cfg = TrainConfig(target="demeaned_log", share_weight=0.5, params=SMALL, num_boost_round=60,
                      holdout_days=20, early_stopping_rounds=20)
    fit = fit_bfsp(df, ["f1", "f2"], cfg)
    out = predict_prices(fit.booster, df, ["f1", "f2"], target="demeaned_log")
    book = out.groupby("raceid")["predicted_win_prob_norm"].sum()
    np.testing.assert_allclose(book.to_numpy(), 1.0, atol=1e-9)
    assert cfg.describe()["sample_weighting"]["share_power"] == 0.5
