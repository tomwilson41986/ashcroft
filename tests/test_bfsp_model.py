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

    def predict(self, X):
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
