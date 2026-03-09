"""Tests for FundamentalModel — win probability prediction."""

import numpy as np
import pandas as pd
import pytest

from src.models.probability_model import FundamentalModel


def _make_training_data(n_races=50, n_runners=8):
    """Create synthetic training data."""
    rows = []
    for race_idx in range(n_races):
        race_id = f"race_{race_idx}"
        winner = np.random.randint(0, n_runners)
        for i in range(n_runners):
            rows.append({
                "raceid": race_id,
                "won": int(i == winner),
                "preracehorsecareerNFP": np.random.uniform(0.3, 0.8),
                "preracehorsecareerRB": np.random.uniform(0.3, 0.8),
                "preracehorsecareerFSARB": np.random.uniform(0.3, 0.8),
                "preracehorsecareerWIV": np.random.uniform(0.5, 2.0),
                "preracehorsecareerWAX": np.random.uniform(-0.1, 0.2),
                "preracehorsecareerCWO": np.random.uniform(-0.5, 1.0),
                "LR3NFPtotal": np.random.uniform(0.3, 0.8),
                "LR5NFPtotal": np.random.uniform(0.3, 0.8),
                "LR10NFPtotal": np.random.uniform(0.3, 0.8),
                "last_finish_pos": np.random.randint(1, n_runners + 1),
                "last_bsp": np.random.uniform(2, 50),
                "days_since_last_run": np.random.randint(7, 90),
                "number_of_runners": n_runners,
                "distance_furlongs": 8.0,
                "going_numeric": 4.0,
                "race_class_numeric": 3,
                "weight_lbs": 126,
                "draw_position": i + 1,
                "age": 4,
                "is_debut": 0,
                "bfsp": 2.0 + i * 2.0,
            })
    return pd.DataFrame(rows)


class TestFundamentalModel:
    def test_logistic_fit_predict(self):
        """Logistic model can fit and predict."""
        data = _make_training_data(n_races=30)
        model = FundamentalModel(model_type="logistic")
        model.fit(data, race_id_col="raceid", target_col="won")
        preds = model.predict(data, race_id_col="raceid")

        assert "p_model_raw" in preds.columns
        assert "p_model" in preds.columns
        assert len(preds) == len(data)

    def test_probability_normalisation(self):
        """Probabilities must sum to 1.0 within each race."""
        data = _make_training_data(n_races=20)
        model = FundamentalModel(model_type="logistic")
        model.fit(data, race_id_col="raceid", target_col="won")
        preds = model.predict(data, race_id_col="raceid")

        # Check per-race sums
        race_sums = preds.groupby("raceid")["p_model"].sum()
        np.testing.assert_allclose(race_sums.values, 1.0, atol=1e-6)

    def test_probabilities_positive(self):
        """All probabilities should be positive."""
        data = _make_training_data(n_races=20)
        model = FundamentalModel(model_type="logistic")
        model.fit(data, race_id_col="raceid", target_col="won")
        preds = model.predict(data, race_id_col="raceid")

        assert (preds["p_model"] > 0).all()
        assert (preds["p_model"] < 1).all()

    def test_feature_importance_logistic(self):
        """Feature importance should return non-empty dict for logistic."""
        data = _make_training_data(n_races=30)
        model = FundamentalModel(model_type="logistic")
        model.fit(data, race_id_col="raceid", target_col="won")

        imp = model.get_feature_importance()
        assert isinstance(imp, dict)
        assert len(imp) > 0

    def test_predict_without_fit_raises(self):
        """Predicting before fitting should raise RuntimeError."""
        model = FundamentalModel(model_type="logistic")
        data = _make_training_data(n_races=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(data)

    def test_single_race_normalisation(self):
        """When no race_id_col, normalise the entire batch."""
        data = _make_training_data(n_races=20)
        model = FundamentalModel(model_type="logistic")
        model.fit(data, race_id_col="raceid", target_col="won")

        single_race = data[data["raceid"] == "race_0"].drop(columns=["raceid"])
        preds = model.predict(single_race)

        total = preds["p_model"].sum()
        assert abs(total - 1.0) < 1e-6

    def test_handles_extra_columns(self):
        """Model should work when prediction data has extra columns."""
        data = _make_training_data(n_races=20)
        model = FundamentalModel(model_type="logistic")
        model.fit(data, race_id_col="raceid", target_col="won")

        # Add extra column — should not break prediction
        data_extra = data.copy()
        data_extra["extra_col"] = 999
        preds = model.predict(data_extra, race_id_col="raceid")
        assert len(preds) == len(data_extra)


class TestXGBoostModel:
    def test_xgboost_fit_predict(self):
        """XGBoost model can fit and predict."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost not installed")

        data = _make_training_data(n_races=30)
        model = FundamentalModel(model_type="xgboost")
        model.fit(data, race_id_col="raceid", target_col="won")
        preds = model.predict(data, race_id_col="raceid")

        assert "p_model" in preds.columns
        race_sums = preds.groupby("raceid")["p_model"].sum()
        np.testing.assert_allclose(race_sums.values, 1.0, atol=1e-6)

    def test_xgboost_feature_importance(self):
        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost not installed")

        data = _make_training_data(n_races=30)
        model = FundamentalModel(model_type="xgboost")
        model.fit(data, race_id_col="raceid", target_col="won")

        imp = model.get_feature_importance()
        assert isinstance(imp, dict)
        assert len(imp) > 0

    def test_xgboost_with_validation(self):
        """XGBoost should use validation set for early stopping."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost not installed")

        data = _make_training_data(n_races=40)
        train = data[data["raceid"].str.extract(r"(\d+)")[0].astype(int) < 30]
        val = data[data["raceid"].str.extract(r"(\d+)")[0].astype(int) >= 30]

        model = FundamentalModel(model_type="xgboost")
        model.fit(train, val_df=val, race_id_col="raceid", target_col="won")
        preds = model.predict(val, race_id_col="raceid")
        assert len(preds) == len(val)
