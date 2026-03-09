"""Stage 1: Fundamental win probability model.

Predicts P(win) for each runner in a race. Supports both conditional logit
(Benter's original approach) and XGBoost. Probabilities are normalised within
each race so they sum to 1.0.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# Default feature set — starting point, to be pruned by select_features()
DEFAULT_FEATURES = [
    # Core horse form
    "preracehorsecareerNFP", "preracehorsecareerRB", "preracehorsecareerFSARB",
    "preracehorsecareerWIV", "preracehorsecareerWAX", "preracehorsecareerCWO",
    "LR3NFPtotal", "LR5NFPtotal", "LR10NFPtotal",
    "last_finish_pos", "last_bsp", "days_since_last_run",
    # Market-derived
    "LR3_RWO", "LR5_RWO", "LR10_RWO",
    "PFD3", "PFD5", "PFD10",
    "OFS3", "OFS5", "OFS10",
    # Stability / environment
    "FSS", "FCS", "FinalDSLR", "CIL3", "CIL5", "CIL10",
    "going_preference", "distance_preference", "class_change", "weight_change",
    "course_win_pct", "course_runs",
    # Prize money
    "WPMRF3", "WPMRF5", "WPMRF10",
    "PMW3", "PMW5", "PMW10",
    # Jockey / trainer
    "preracejockeycareerWIV", "preracejockeycareerNFP",
    "preracetrainercareerWIV", "preracetrainercareerNFP",
    "trainerjockeycareerWIV", "trainerjockeycareerNFP",
    "Jockey_Career_EPF", "trainer_Career_EPF", "Horse_Career_EPF",
    # Race shape
    "horsepaceindex", "jockeypaceindex", "trainerpaceindex",
    # Today's race
    "number_of_runners", "distance_furlongs", "going_numeric",
    "race_class_numeric", "weight_lbs", "draw_position", "age", "is_debut",
    # Within-race ranks
    "rNFP", "rNFPLR5", "horseWIVrank", "horseRBrank",
    "rRWOLR5", "rRWOLR10", "rPFD5", "rFSS", "rFCS",
]


def _normalise_within_race(df: pd.DataFrame, prob_col: str, race_id_col: str) -> pd.Series:
    """Normalise probabilities to sum to 1.0 within each race."""
    return df.groupby(race_id_col)[prob_col].transform(lambda x: x / x.sum())


class FundamentalModel:
    """Fundamental win probability model (Stage 1 of Benter pipeline).

    Predicts P(win) for each runner. Supports:
      - 'logistic': Conditional logit (sklearn LogisticRegression)
      - 'xgboost': XGBClassifier with race-level normalisation
    """

    def __init__(self, model_type: str = "xgboost", features: list[str] | None = None):
        """
        Args:
            model_type: 'logistic' or 'xgboost'
            features: Feature column names to use. Defaults to DEFAULT_FEATURES.
        """
        if model_type == "xgboost" and not HAS_XGB:
            raise ImportError("xgboost is required for model_type='xgboost'")
        self.model_type = model_type
        self.features = features or DEFAULT_FEATURES.copy()
        self.model = None
        self.feature_importance_ = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None,
            race_id_col: str = "raceid", target_col: str = "won") -> "FundamentalModel":
        """Train the model.

        Args:
            train_df: Training data with feature columns, target, and race IDs.
            val_df: Optional validation data for early stopping (XGBoost).
            race_id_col: Column identifying unique races.
            target_col: Binary target column (1=win, 0=lose).

        Returns:
            self
        """
        available = [f for f in self.features if f in train_df.columns]
        self.features = available

        X_train = train_df[self.features].fillna(0).values
        y_train = train_df[target_col].values.astype(int)

        if self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                class_weight="balanced",
            )
            self.model.fit(X_train, y_train)

        elif self.model_type == "xgboost":
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_estimators": 1000,
                "use_label_encoder": False,
                "verbosity": 0,
                "random_state": 42,
            }
            self.model = xgb.XGBClassifier(**params)

            fit_kwargs = {}
            if val_df is not None:
                X_val = val_df[self.features].fillna(0).values
                y_val = val_df[target_col].values.astype(int)
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["verbose"] = False

            self.model.fit(X_train, y_train, **fit_kwargs)

            # Store feature importance
            self.feature_importance_ = dict(
                zip(self.features, self.model.feature_importances_)
            )

        return self

    def predict(self, df: pd.DataFrame, race_id_col: str = "raceid") -> pd.DataFrame:
        """Predict win probabilities for each runner.

        Args:
            df: DataFrame with feature columns and race IDs.
            race_id_col: Column identifying unique races.

        Returns:
            Input DataFrame with added columns:
                - p_model_raw: raw predicted probability
                - p_model: race-normalised probability
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = df[self.features].fillna(0).values
        result = df.copy()

        if self.model_type == "logistic":
            result["p_model_raw"] = self.model.predict_proba(X)[:, 1]
        elif self.model_type == "xgboost":
            result["p_model_raw"] = self.model.predict_proba(X)[:, 1]

        # Normalise within race
        if race_id_col in result.columns:
            result["p_model"] = _normalise_within_race(result, "p_model_raw", race_id_col)
        else:
            # Single race — just normalise the whole batch
            total = result["p_model_raw"].sum()
            result["p_model"] = result["p_model_raw"] / total if total > 0 else 1.0 / len(result)

        return result

    def select_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        race_id_col: str = "raceid",
                        target_col: str = "won") -> list[str]:
        """Benter's variable selection: keep features that improve log-loss
        beyond what market odds already capture.

        1. Fit a baseline model using only log(1/BFSP) as a feature.
        2. Add candidate features one at a time.
        3. Keep those that improve log-loss on validation set.

        Args:
            train_df: Training data.
            val_df: Validation data.
            race_id_col: Race identifier column.
            target_col: Binary target column.

        Returns:
            List of selected feature names.
        """
        # Baseline: log of market-implied probability
        baseline_feature = "log_market_prob"
        for dset in [train_df, val_df]:
            if "bfsp" in dset.columns:
                dset = dset.copy()
                bfsp = dset["bfsp"].clip(lower=1.01)
                dset[baseline_feature] = np.log(1.0 / bfsp)

        # We need bfsp for baseline
        if "bfsp" not in train_df.columns:
            return self.features  # Can't do selection without market prices

        train_work = train_df.copy()
        val_work = val_df.copy()
        train_work[baseline_feature] = np.log(1.0 / train_work["bfsp"].clip(lower=1.01))
        val_work[baseline_feature] = np.log(1.0 / val_work["bfsp"].clip(lower=1.01))

        # Fit baseline
        baseline_model = FundamentalModel(model_type="logistic", features=[baseline_feature])
        baseline_model.fit(train_work, target_col=target_col)
        baseline_preds = baseline_model.predict(val_work, race_id_col=race_id_col)
        baseline_loss = log_loss(val_work[target_col], baseline_preds["p_model"].clip(1e-10, 1 - 1e-10))

        # Test each candidate feature
        selected = [baseline_feature]
        candidates = [f for f in self.features if f in train_work.columns and f != baseline_feature]

        for feat in candidates:
            trial_features = selected + [feat]
            trial_model = FundamentalModel(model_type="logistic", features=trial_features)
            try:
                trial_model.fit(train_work, target_col=target_col)
                trial_preds = trial_model.predict(val_work, race_id_col=race_id_col)
                trial_loss = log_loss(
                    val_work[target_col],
                    trial_preds["p_model"].clip(1e-10, 1 - 1e-10),
                )
                if trial_loss < baseline_loss - 1e-6:
                    selected.append(feat)
                    baseline_loss = trial_loss
            except Exception:
                continue

        # Remove baseline feature from final list
        selected = [f for f in selected if f != baseline_feature]
        self.features = selected if selected else self.features
        return self.features

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        if self.feature_importance_ is not None:
            return dict(sorted(self.feature_importance_.items(), key=lambda x: -x[1]))

        if self.model_type == "logistic" and self.model is not None:
            coeffs = dict(zip(self.features, np.abs(self.model.coef_[0])))
            return dict(sorted(coeffs.items(), key=lambda x: -x[1]))

        return {}

    def get_shap_values(self, df: pd.DataFrame) -> np.ndarray | None:
        """Compute SHAP values for feature contributions (XGBoost only)."""
        if not HAS_SHAP or self.model_type != "xgboost" or self.model is None:
            return None
        X = df[self.features].fillna(0).values
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)

    def save(self, path: str) -> None:
        """Save model artifacts to directory."""
        os.makedirs(path, exist_ok=True)
        meta = {
            "model_type": self.model_type,
            "features": self.features,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if self.model_type == "xgboost":
            self.model.save_model(os.path.join(path, "model.xgb"))
        else:
            joblib.dump(self.model, os.path.join(path, "model.joblib"))

        if self.feature_importance_:
            with open(os.path.join(path, "feature_importance.json"), "w") as f:
                json.dump(self.feature_importance_, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FundamentalModel":
        """Load a saved model from directory."""
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)

        instance = cls(model_type=meta["model_type"], features=meta["features"])

        if meta["model_type"] == "xgboost":
            instance.model = xgb.XGBClassifier()
            instance.model.load_model(os.path.join(path, "model.xgb"))
        else:
            instance.model = joblib.load(os.path.join(path, "model.joblib"))

        imp_path = os.path.join(path, "feature_importance.json")
        if os.path.exists(imp_path):
            with open(imp_path) as f:
                instance.feature_importance_ = json.load(f)

        return instance
