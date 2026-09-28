"""The within-race cross-entropy target: its objective, metric and prices.

Synthetic frames test mechanics only; nothing is trained or evaluated on them for use.
"""
import numpy as np
import pandas as pd
import pytest

from model.bfsp_model import (
    DEFAULT_PARAMS,
    RaceIndex,
    TrainConfig,
    build_target,
    fit_bfsp,
    predict_prices,
    race_xent_metric,
    race_xent_objective,
)


class _D:
    """What LightGBM hands a custom objective: the label, the weight, and the races."""

    def __init__(self, label, races, weight=None):
        self._y, self._w = np.asarray(label, float), weight
        self.race_index = RaceIndex(races)

    def get_label(self):
        return self._y

    def get_weight(self):
        return self._w


def _loss(score, p, races):
    q = RaceIndex(races).softmax_neg(score)
    return -np.sum(p * np.log(q))


def test_the_softmax_is_per_race_whatever_order_the_rows_arrive_in():
    races = np.array(["b", "a", "b", "c", "a", "b"])
    score = np.array([0.5, 1.0, 2.0, 0.3, 0.0, 1.0])
    q = RaceIndex(races).softmax_neg(score)
    for r in "abc":
        m = races == r
        e = np.exp(-score[m])
        np.testing.assert_allclose(q[m], e / e.sum())
    assert q[races == "c"][0] == pytest.approx(1.0)       # a one-runner race holds its whole book


def test_the_gradient_and_curvature_are_the_losss():
    rng = np.random.default_rng(0)
    races = np.repeat(np.arange(5), 6)[rng.permutation(30)]
    p = rng.dirichlet(np.ones(6), size=5)
    label = np.empty(30)
    for r in range(5):
        label[races == r] = p[r]
    score = rng.normal(size=30)
    grad, hess = race_xent_objective(score, _D(label, races))
    eps = 1e-6
    for i in (0, 7, 19):
        up, dn = score.copy(), score.copy()
        up[i] += eps
        dn[i] -= eps
        fd = (_loss(up, label, races) - _loss(dn, label, races)) / (2 * eps)
        assert grad[i] == pytest.approx(fd, rel=1e-5, abs=1e-8)
        h = 1e-4                                   # a second difference needs a wider step than 1e-6
        up2, dn2 = score.copy(), score.copy()
        up2[i] += h
        dn2[i] -= h
        fd2 = (_loss(up2, label, races) - 2 * _loss(score, label, races) + _loss(dn2, label, races)) / h ** 2
        assert hess[i] == pytest.approx(fd2, rel=1e-4)
    w = np.linspace(0.5, 2.0, 30)
    gw, hw = race_xent_objective(score, _D(label, races, weight=w))
    np.testing.assert_allclose(gw, grad * w)
    np.testing.assert_allclose(hw, hess * w)


def test_the_metric_is_the_mean_cross_entropy_per_race():
    races = np.array([0, 0, 1, 1, 1])
    label = np.array([0.75, 0.25, 0.5, 0.25, 0.25])
    score = np.zeros(5)
    name, value, higher_better = race_xent_metric(score, _D(label, races))
    assert name == "race_xent" and higher_better is False
    assert value == pytest.approx((np.log(2) + np.log(3)) / 2)


def _history(n_days=300, races_per_day=3, runners=8, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for r in range(races_per_day):
            skill = rng.normal(size=runners)
            ip = np.exp(1.2 * skill)
            ip = 1.08 * ip / ip.sum()
            for i in range(runners):
                rows.append({"race_date": day, "race_time": f"{1 + r}.30", "track": "York",
                             "horse_name": f"h{d}_{r}_{i}", "raceid": f"{day.date()}_{r}",
                             "f1": skill[i] + rng.normal(0, 0.2), "f2": rng.normal(),
                             "bfsp": 1.0 / ip[i]})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)   # races arrive interleaved


SMALL = {**DEFAULT_PARAMS, "num_leaves": 7, "min_child_samples": 20, "learning_rate": 0.1}


def test_a_fit_prices_the_races_like_the_market_and_records_its_objective():
    df = _history()
    cfg = TrainConfig(target="race_xent", params=SMALL, num_boost_round=300, holdout_days=40,
                      early_stopping_rounds=20)
    fit = fit_bfsp(df, ["f1", "f2"], cfg)
    assert 0 < fit.best_iteration <= 300
    out = predict_prices(fit.booster, df, ["f1", "f2"], target="race_xent")
    book = out.groupby("raceid")["predicted_win_prob_norm"].sum()
    np.testing.assert_allclose(book.to_numpy(), 1.0, atol=1e-9)
    truth = build_target(df, "race_xent")
    corr = np.corrcoef(np.log(out["predicted_win_prob_norm"]), np.log(truth))[0, 1]
    assert corr > 0.95
    assert cfg.describe()["lightgbm_objective"] == "custom:race_xent"


def test_race_xent_cannot_be_combined_with_the_profit_weighted_objective():
    with pytest.raises(ValueError, match="race_xent"):
        fit_bfsp(_history(n_days=40), ["f1", "f2"],
                 TrainConfig(target="race_xent", objective="profit_weighted", params=SMALL, num_boost_round=5))


def test_a_fitted_booster_serves_from_its_file_as_it_predicted_in_memory():
    """The 06:00 path reads a booster from its file and prices it by its metadata's target.
    A booster fitted with the custom objective is written as "custom"; read back, it must
    give the same raw score, so the served price is the fitted one."""
    import lightgbm as lgb
    from model.bfsp_model import assert_meta_is_servable, attach_serving_rule
    df = _history(n_days=120)
    cfg = TrainConfig(target="race_xent", params=SMALL, num_boost_round=60, holdout_days=20,
                      early_stopping_rounds=20)
    fit = fit_bfsp(df, ["f1", "f2"], cfg)
    meta = {"feature_cols": ["f1", "f2"], "objective": "l2", "target": "race_xent"}
    assert_meta_is_servable(meta)
    loaded = attach_serving_rule(lgb.Booster(model_str=fit.booster.model_to_string()), meta)
    want = predict_prices(fit.booster, df, ["f1", "f2"], target="race_xent")
    got = predict_prices(loaded, df, ["f1", "f2"])
    np.testing.assert_allclose(got["predicted_bfsp"].to_numpy(), want["predicted_bfsp"].to_numpy(), rtol=1e-12)
    book = got.groupby("raceid")["predicted_win_prob_norm"].sum()
    np.testing.assert_allclose(book.to_numpy(), 1.0, atol=1e-9)
