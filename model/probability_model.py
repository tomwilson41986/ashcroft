"""
Stage 1: Fundamental Win Probability Model.

Predicts P(win) for each runner in a race using pre-race features.
This is a binary classification problem where we want calibrated
probabilities, not just rankings.

Probabilities are normalised within each race to sum to 1.0
(exactly one horse wins each race).
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


class FundamentalModel:
    """Predict win probabilities from pre-race features.

    Uses LightGBM binary classification trained to predict P(win).
    After prediction, probabilities are normalised per race so they
    sum to 1.0.

    The model can also perform Benter-style feature selection: keeping
    only features that improve log-loss beyond what public odds capture.
    """

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
    }

    def __init__(self, params: dict | None = None):
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.model: lgb.Booster | None = None
        self.feature_cols: list[str] = []
        self.metadata: dict = {}

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "won",
        raceid_col: str = "raceid",
        num_boost_round: int = 2000,
        early_stopping: int = 50,
    ) -> dict:
        """Train the win probability model.

        Args:
            train_df: Training data with features and target.
            val_df: Validation data for early stopping.
            feature_cols: List of feature column names.
            target_col: Binary target column (1=win, 0=lose).
            raceid_col: Column identifying each race (for grouping).
            num_boost_round: Maximum boosting iterations.
            early_stopping: Stop after N rounds without improvement.

        Returns:
            Dict of training metrics.
        """
        self.feature_cols = feature_cols

        X_train = train_df[feature_cols].astype(float)
        y_train = train_df[target_col].astype(float)
        X_val = val_df[feature_cols].astype(float)
        y_val = val_df[target_col].astype(float)

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=early_stopping),
        ]

        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Compute validation metrics
        val_probs = self.model.predict(X_val)
        val_ll = log_loss(y_val, val_probs)

        # Normalise and evaluate per-race calibration
        if raceid_col in val_df.columns:
            val_copy = val_df[[raceid_col, target_col]].copy()
            val_copy["p_raw"] = val_probs
            val_copy["p_norm"] = val_copy.groupby(raceid_col)[
                "p_raw"
            ].transform(lambda x: x / x.sum())
            norm_ll = log_loss(val_copy[target_col], val_copy["p_norm"])
        else:
            norm_ll = val_ll

        metrics = {
            "val_logloss_raw": round(val_ll, 6),
            "val_logloss_normalised": round(norm_ll, 6),
            "best_iteration": self.model.best_iteration,
            "n_features": len(feature_cols),
            "train_size": len(train_df),
            "val_size": len(val_df),
        }

        self.metadata = metrics
        return metrics

    def predict(
        self,
        df: pd.DataFrame,
        raceid_col: str = "raceid",
        normalise: bool = True,
    ) -> pd.DataFrame:
        """Predict win probabilities for runners.

        Args:
            df: DataFrame with feature columns.
            raceid_col: Column to group runners by race.
            normalise: If True, normalise probabilities to sum to 1 per race.

        Returns:
            DataFrame with added columns: p_model, p_model_norm.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        X = df[self.feature_cols].astype(float)
        raw_probs = self.model.predict(X)

        result = df.copy()
        result["p_model"] = raw_probs

        if normalise and raceid_col in result.columns:
            result["p_model"] = result.groupby(raceid_col)[
                "p_model"
            ].transform(lambda x: x / x.sum())

        # Convert to implied BFSP
        result["predicted_bfsp"] = 1.0 / result["p_model"].replace(0, np.nan)

        return result

    def select_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        candidate_features: list[str],
        target_col: str = "won",
        bfsp_col: str = "bfsp",
    ) -> list[str]:
        """Benter-style feature selection.

        Starts with log(market_prob) as baseline, then adds features
        one at a time, keeping only those that improve log-loss.

        Args:
            train_df: Training data.
            val_df: Validation data.
            candidate_features: All candidate feature columns.
            target_col: Binary target.
            bfsp_col: BFSP column for computing market probability baseline.

        Returns:
            List of selected feature names.
        """
        # Baseline: market probability only
        base_feature = "log_market_prob"
        train_df = train_df.copy()
        val_df = val_df.copy()

        if bfsp_col in train_df.columns:
            train_df[base_feature] = np.log(
                1.0 / train_df[bfsp_col].replace(0, np.nan)
            )
            val_df[base_feature] = np.log(
                1.0 / val_df[bfsp_col].replace(0, np.nan)
            )
        else:
            # No market data — start with empty baseline
            selected = candidate_features[:20]  # take first 20 as baseline
            return selected

        # Train baseline model with just market prob
        selected = [base_feature]
        baseline_model = self._quick_train(
            train_df, val_df, selected, target_col
        )
        baseline_ll = baseline_model["val_logloss"]

        # Try adding each feature
        improvements = []
        for feat in candidate_features:
            if feat == bfsp_col or feat not in train_df.columns:
                continue
            if train_df[feat].isna().all() or val_df[feat].isna().all():
                continue

            test_features = selected + [feat]
            result = self._quick_train(
                train_df, val_df, test_features, target_col
            )
            improvement = baseline_ll - result["val_logloss"]
            if improvement > 0.0001:  # meaningful improvement threshold
                improvements.append((feat, improvement))

        # Sort by improvement and add features greedily
        improvements.sort(key=lambda x: -x[1])
        selected_features = [base_feature]

        for feat, _ in improvements:
            candidate = selected_features + [feat]
            result = self._quick_train(
                train_df, val_df, candidate, target_col
            )
            # Keep if it still improves
            if result["val_logloss"] < baseline_ll:
                selected_features.append(feat)
                baseline_ll = result["val_logloss"]

        # Remove the market prob from selected (it will be used in blending)
        selected_features = [
            f for f in selected_features if f != base_feature
        ]

        return selected_features if selected_features else candidate_features[:20]

    def _quick_train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        features: list[str],
        target_col: str,
    ) -> dict:
        """Quick LightGBM train for feature selection."""
        X_train = train_df[features].astype(float)
        y_train = train_df[target_col].astype(float)
        X_val = val_df[features].astype(float)
        y_val = val_df[target_col].astype(float)

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = self.params.copy()
        params["num_leaves"] = 31  # smaller for speed

        model = lgb.train(
            params,
            train_set,
            num_boost_round=200,
            valid_sets=[val_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_val)
        ll = log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7))
        return {"val_logloss": ll}

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Get feature importance rankings."""
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.feature_importance(
            importance_type=importance_type
        )
        return pd.DataFrame(
            {
                "feature": self.model.feature_name(),
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

    def save(self, directory: str):
        """Save model and metadata to directory."""
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, "probability_model.lgb")
        meta_path = os.path.join(directory, "probability_model_meta.json")

        self.model.save_model(model_path)

        meta = {
            "feature_cols": self.feature_cols,
            "params": self.params,
            "metrics": self.metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, directory: str) -> "FundamentalModel":
        """Load a saved model."""
        model_path = os.path.join(directory, "probability_model.lgb")
        meta_path = os.path.join(directory, "probability_model_meta.json")

        instance = cls()
        instance.model = lgb.Booster(model_file=model_path)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            instance.feature_cols = meta.get("feature_cols", [])
            instance.params = meta.get("params", cls.DEFAULT_PARAMS)
            instance.metadata = meta.get("metrics", {})

        return instance
