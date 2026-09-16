"""The price and the probability must be the same number.

Three scripts used to derive them separately and a calibrator overwrote one of
them, so `1 / predicted_bfsp` and `predicted_win_prob_norm` disagreed by a
median of 13% on the last evaluation and the per-race book ran from 0.16 to
1.85. Downstream code reads whichever it happens to want — the Kelly simulation
takes the price, the bet analysis takes the probability — so the disagreement
changed the answer depending on which report you read.
"""

import numpy as np
import pandas as pd
import pytest

from model.bfsp_model import ensure_race_id, normalise_prices, predict_prices


class _Booster:
    """Returns fixed log-prices, so the test is about the arithmetic around it."""

    def __init__(self, log_prices):
        self._v = np.asarray(log_prices, dtype=float)

    def predict(self, X, num_iteration=None):
        return self._v[: len(X)]


def _card(n_races=4, runners=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        for i in range(runners):
            rows.append({
                "race_date": pd.Timestamp("2026-04-01"),
                "race_time": f"{1 + r}.30",
                "track": "Ascot",
                "horse_name": f"h{r}_{i}",
                "f1": rng.normal(),
                "f2": rng.normal(),
            })
    return pd.DataFrame(rows)


def test_price_and_probability_are_one_number():
    df = _card()
    rng = np.random.default_rng(1)
    booster = _Booster(rng.uniform(0.3, 4.0, len(df)))

    out = predict_prices(booster, df, ["f1", "f2"])

    assert np.allclose(1.0 / out["predicted_bfsp"], out["predicted_win_prob_norm"], atol=1e-12)
    assert np.allclose(out["predicted_win_prob"], out["predicted_win_prob_norm"], atol=1e-12)
    assert np.allclose(out["predicted_log_bfsp"], np.log(out["predicted_bfsp"]), atol=1e-12)


def test_every_race_book_is_one():
    df = _card(n_races=5, runners=6)
    rng = np.random.default_rng(2)
    out = predict_prices(_Booster(rng.uniform(0.3, 5.0, len(df))), df, ["f1", "f2"])

    book = (1.0 / out["predicted_bfsp"]).groupby(out["raceid"]).sum()
    assert np.allclose(book.values, 1.0, atol=1e-9), book.describe()


def test_the_raw_price_is_kept_for_offline_calibration():
    """`research_lab.py price-cal` must fit against the booster's own output.

    The previous calibration report compared calibrated output with calibrated
    output, because the raw column was never exported, and concluded the
    calibrator "adds nothing"."""
    df = _card(n_races=2, runners=4)
    log_prices = np.array([0.5, 1.0, 1.5, 2.0, 0.7, 1.2, 1.7, 2.2])
    out = predict_prices(_Booster(log_prices), df, ["f1", "f2"])

    assert np.allclose(out["predicted_bfsp_raw"], np.exp(log_prices))
    assert np.allclose(out["predicted_log_bfsp_raw"], log_prices)
    # The raw book is whatever the booster produced; the served one is not.
    raw_book = (1.0 / out["predicted_bfsp_raw"]).groupby(out["raceid"]).sum()
    assert not np.allclose(raw_book.values, 1.0)


def test_normalisation_preserves_the_within_race_ordering():
    """Scaling a race is a constant in log space, so ranks cannot move."""
    df = _card(n_races=3, runners=7, seed=5)
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.2, 4.5, len(df))
    out = predict_prices(_Booster(raw), df, ["f1", "f2"])

    for _, g in out.groupby("raceid"):
        assert list(g.sort_values("predicted_bfsp_raw").horse_name) == \
               list(g.sort_values("predicted_bfsp").horse_name)


def test_a_target_book_other_than_one_is_exact():
    df = _card(n_races=3, runners=5)
    rng = np.random.default_rng(4)
    prob, price = normalise_prices(
        np.exp(rng.uniform(0.3, 4.0, len(df))), ensure_race_id(df), target_book=1.0015
    )
    book = pd.Series(prob).groupby(ensure_race_id(df).to_numpy()).sum()
    assert np.allclose(book.values, 1.0015, atol=1e-9)
    assert np.allclose(price, 1.0 / prob, atol=1e-12)


def test_race_id_is_rebuilt_when_absent_and_respected_when_present():
    df = _card(n_races=2, runners=3)
    rebuilt = ensure_race_id(df)
    assert rebuilt.nunique() == 2

    df2 = df.copy()
    df2["raceid"] = "fixed"
    assert ensure_race_id(df2).nunique() == 1
    # A single race id means one book over every runner.
    out = predict_prices(_Booster(np.full(len(df2), 1.0)), df2, ["f1", "f2"])
    assert np.isclose((1.0 / out["predicted_bfsp"]).sum(), 1.0, atol=1e-9)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_runner_with_no_prediction_does_not_poison_its_race(bad):
    """A NaN price makes its own row NaN; the guard is that it raises nothing."""
    df = _card(n_races=1, runners=4)
    out = predict_prices(_Booster([1.0, 2.0, bad, 1.5]), df, ["f1", "f2"])
    assert out["predicted_bfsp"].notna().sum() >= 1


# ---------------------------------------------------------------------------
# A feature that did not build must not become a silent NaN column.
# ---------------------------------------------------------------------------

from model.bfsp_model import (  # noqa: E402
    MissingFeatureColumns,
    all_nan_columns,
    resolve_feature_columns,
)


def test_missing_feature_columns_fail_loudly():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(MissingFeatureColumns) as e:
        resolve_feature_columns(df, ["a", "b", "c", "d"])
    assert "c" in str(e.value) and "d" in str(e.value)
    assert "2 of 4" in str(e.value)


def test_missing_can_be_allowed_deliberately():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    cols = resolve_feature_columns(df, ["a", "c", "b"], strict=False)
    assert cols == ["a", "b"]


def test_drop_prefixes_and_names_both_apply():
    df = pd.DataFrame({"td_x": [1.0], "td_y": [1.0], "keep": [1.0], "byname": [1.0]})
    cols = resolve_feature_columns(
        df, ["td_x", "td_y", "keep", "byname"], drop_prefixes=("td_",), drop_names=("byname",)
    )
    assert cols == ["keep"]


def test_a_column_that_built_empty_is_reported():
    df = pd.DataFrame({"ok": [1.0, 2.0], "empty": [np.nan, np.nan]})
    assert all_nan_columns(df, ["ok", "empty", "absent"]) == ["empty"]


# ---------------------------------------------------------------------------
# The evaluation must measure the model production ships.
#
# It did not. `evaluate_oos.py` trained plain L2 over every runner with no
# sample weights; `train_bfsp.py` defaulted to a profit-weighted objective that
# weights a runner by 1/sqrt(BFSP) and penalises predicting too short by 1.5x,
# plus recency decay giving a three-year-old race 5% of today's weight.
# ---------------------------------------------------------------------------

from dataclasses import replace  # noqa: E402

from model.bfsp_model import (  # noqa: E402
    DEFAULT_PARAMS,
    TrainConfig,
    build_target,
    fit_bfsp,
    fold_masks,
    invert_target,
    model_meta,
    assert_meta_is_servable,
    profit_weighted_objective,
    sample_weights,
)


def _history(n_days=400, races_per_day=3, runners=8, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for r in range(races_per_day):
            for i in range(runners):
                skill = rng.normal()
                rows.append({
                    "race_date": day, "race_time": f"{1 + r}.30", "track": "Ascot",
                    "horse_name": f"h{d}_{r}_{i}",
                    "raceid": f"{day.date()}_{r}",
                    "f1": skill + rng.normal(0, 0.3), "f2": rng.normal(),
                    "bfsp": float(np.exp(1.5 - 0.8 * skill + rng.normal(0, 0.25))),
                })
    return pd.DataFrame(rows)


def test_defaults_are_the_recipe_the_evaluation_measures():
    cfg = TrainConfig()
    assert cfg.objective == "l2"
    assert cfg.decay_rate == 0.0
    assert cfg.target == "log_bfsp"
    d = cfg.describe()
    assert d["sample_weighting"] == {"type": "none", "decay_rate": 0.0}
    assert d["lightgbm_objective"] == "regression"


def test_the_objective_and_weighting_are_recorded():
    cfg = TrainConfig(objective="profit_weighted", decay_rate=1.0)
    d = cfg.describe()
    assert d["lightgbm_objective"] == "custom:profit_weighted"
    assert d["sample_weighting"]["type"] == "exponential_decay"
    # 1.0/year is a half-life a bit over eight months; worth seeing in the meta.
    assert 250 < d["sample_weighting"]["half_life_days"] < 260


def test_early_stopping_never_sees_the_rows_it_will_score():
    """The iteration count must not be chosen on the test fold."""
    df = _history(n_days=200)
    cfg = TrainConfig(holdout_days=30, num_boost_round=40, early_stopping_rounds=5)
    fit = fit_bfsp(df, ["f1", "f2"], cfg)

    assert fit.n_holdout > 0 and fit.n_train > 0
    # The holdout is the tail of the TRAINING window, so a fold starting after
    # the training window ends cannot overlap it.
    assert pd.Timestamp(fit.holdout_start) <= df["race_date"].max()
    assert fit.n_train + fit.n_holdout == len(df)


def test_purge_keeps_the_fold_clear_of_the_training_window():
    df = _history(n_days=150)
    cfg = TrainConfig(purge_days=30, embargo_days=7)
    val_start, val_end = pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-01")
    tr, va = fold_masks(df["race_date"], val_start, val_end, cfg)

    assert df.loc[tr, "race_date"].max() < val_start - pd.Timedelta(days=30)
    assert df.loc[va, "race_date"].min() >= val_start + pd.Timedelta(days=7)
    assert df.loc[va, "race_date"].max() < val_end


def test_recency_weights_are_off_unless_asked_for():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    assert sample_weights(dates, 0.0) is None
    w = sample_weights(dates, 1.0)
    assert w[-1] == pytest.approx(1.0)      # today
    assert w[0] < w[-1]                      # older rows count for less


@pytest.mark.parametrize("target", ["log_bfsp", "demeaned_log", "logit_norm_prob"])
def test_every_target_describes_the_same_normalised_price(target):
    """The three targets are reparameterisations, not different models.

    Because the served price is race-normalised, a race-level term cancels --
    which is why `demeaned_log` needs no separate race-level model."""
    df = _history(n_days=20)
    y = build_target(df, target)
    back = invert_target(y, df, target)

    ip = 1.0 / back
    q = ip / pd.Series(ip).groupby(df["raceid"].to_numpy()).transform("sum").to_numpy()
    truth = 1.0 / df["bfsp"].to_numpy()
    truth = truth / pd.Series(truth).groupby(df["raceid"].to_numpy()).transform("sum").to_numpy()
    assert np.allclose(q, truth, atol=1e-9)


def test_profit_weighted_gradient_favours_favourites_and_punishes_short_quotes():
    class _D:
        def __init__(self, y): self._y = np.asarray(y, float)
        def get_label(self): return self._y

    labels = np.log(np.array([2.0, 2.0, 50.0, 50.0]))
    preds = labels + np.array([0.1, -0.1, 0.1, -0.1])     # over, under, over, under
    grad, hess = profit_weighted_objective(preds, _D(labels))

    # A 2.0 shot carries far more gradient than a 50.0 shot for the same error.
    assert abs(grad[0]) > 4 * abs(grad[2])
    # Under-predicting (quoting shorter than it settles) is penalised harder.
    assert abs(grad[1]) > abs(grad[0])


def test_a_model_trained_on_the_race_result_cannot_be_served():
    cfg = TrainConfig()
    meta = model_meta(cfg, ["or_num", "win_surprise", "NFP_residual"])
    with pytest.raises(ValueError, match="describe the race being predicted"):
        assert_meta_is_servable(meta)


def test_an_artefact_with_no_recipe_cannot_be_served():
    """The deployed model recorded no objective, so nobody could tell which
    of the two models it was."""
    with pytest.raises(ValueError, match="records no objective"):
        assert_meta_is_servable({"feature_cols": ["or_num", "dist_furlongs"]})


def test_meta_records_what_was_trained():
    df = _history(n_days=120)
    cfg = TrainConfig(num_boost_round=30, early_stopping_rounds=5, holdout_days=20)
    fit = fit_bfsp(df, ["f1", "f2"], cfg)
    meta = model_meta(cfg, ["f1", "f2"], fit=fit, vocab={"track": ["Ascot"]})

    assert meta["objective"] == "l2"
    assert meta["target"] == "log_bfsp"
    assert meta["sample_weighting"]["type"] == "none"
    assert meta["n_features"] == 2
    assert meta["categorical_vocab"] == {"track": ["Ascot"]}
    assert meta["best_iteration"] >= 1
    assert meta["feature_code_hash"]
    assert_meta_is_servable(meta)       # a clean one is servable


def test_refit_on_full_uses_every_row():
    df = _history(n_days=120)
    cfg = TrainConfig(num_boost_round=25, early_stopping_rounds=5, holdout_days=20)
    plain = fit_bfsp(df, ["f1", "f2"], cfg)
    refit = fit_bfsp(df, ["f1", "f2"], replace(cfg, refit_on_full=True))

    assert plain.refit is False and refit.refit is True
    # Same stopping point, different fitted booster: the refit saw the tail.
    assert refit.best_iteration == plain.best_iteration
    x = df[["f1", "f2"]].astype(float).head(50)
    assert not np.allclose(plain.booster.predict(x), refit.booster.predict(x))
