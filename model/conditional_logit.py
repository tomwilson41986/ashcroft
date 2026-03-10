"""
Benter-Inspired Conditional Logit Model for Horse Racing.

Adapts Benter (1994) to predict win probabilities WITHOUT using BFSP
as a feature (since BFSP isn't available at prediction time).

For each race r with runners i = 1, ..., N_r:

    V_i = X_i @ beta

    P(horse i wins race r) = exp(V_i) / sum_j exp(V_j)

Key Benter principles retained:
1. Within-race softmax (probabilities sum to 1 per race)
2. Race-relative features (within-race ranks)
3. Optimised via multinomial log-likelihood (not binary)
4. Fair BFSP derived from model probability: BFSP_fair = 1 / P(win)
5. Overlays identified when actual BFSP > fair BFSP

Reference:
    Benter, W. (1994). "Computer Based Horse Race Handicapping and
    Wagering Systems: A Report."
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def _softmax_by_race(V: np.ndarray, race_starts: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities within each race.

    Args:
        V: Utility values, shape (n_total_runners,).
        race_starts: Array of start indices for each race.
            race_starts[i] to race_starts[i+1] gives runners in race i.

    Returns:
        Probabilities, shape (n_total_runners,).
    """
    probs = np.empty_like(V)
    n_races = len(race_starts) - 1
    for r in range(n_races):
        s, e = race_starts[r], race_starts[r + 1]
        v = V[s:e]
        v_max = v.max()
        exp_v = np.exp(v - v_max)
        probs[s:e] = exp_v / exp_v.sum()
    return probs


def _neg_log_likelihood(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    race_starts: np.ndarray,
    alpha: float,
) -> float:
    """Negative log-likelihood + L2 penalty."""
    V = X @ beta
    probs = _softmax_by_race(V, race_starts)

    # Log-likelihood: sum of log(p) for winners only
    winner_probs = probs[y == 1]
    ll = np.sum(np.log(np.clip(winner_probs, 1e-15, None)))

    # L2 regularisation
    penalty = alpha * np.sum(beta ** 2)

    return -ll + penalty


def _neg_log_likelihood_grad(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    race_starts: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Gradient of negative log-likelihood + L2 penalty."""
    V = X @ beta
    probs = _softmax_by_race(V, race_starts)

    # Residuals: y - p (positive for winners, negative for losers)
    residuals = y - probs

    # Gradient w.r.t. beta
    grad_beta = -X.T @ residuals + 2 * alpha * beta

    return grad_beta


class BenterConditionalLogit:
    """Benter-inspired conditional logit for horse race prediction.

    Models P(horse i wins | race field) using multinomial logit on
    pre-race features only (no BFSP input). The model directly learns
    within-race competition structure.

    Fair BFSP is derived as 1 / P(win). Overlays are identified when
    actual BFSP > fair BFSP.

    Args:
        alpha: L2 regularisation strength.
        max_iter: Maximum L-BFGS-B iterations.
        scale_features: Whether to standardise features before fitting.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 500,
        scale_features: bool = True,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.scale_features = scale_features

        self.beta_: np.ndarray | None = None
        self.feature_cols: list[str] = []
        self.scaler: StandardScaler | None = None
        self.metadata: dict = {}

    def _prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        raceid_col: str = "raceid",
        target_col: str = "won",
        fit_scaler: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare arrays for conditional logit.

        Returns:
            X: Feature matrix (n_runners, n_features).
            y: Binary win indicator.
            race_starts: Race boundary indices.
        """
        # Sort by race to ensure contiguous groups
        df = df.sort_values([raceid_col]).reset_index(drop=True)

        # Feature matrix
        X = df[feature_cols].astype(float).values
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0.0)

        # Scale features
        if self.scale_features:
            if fit_scaler:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            elif self.scaler is not None:
                X = self.scaler.transform(X)

        # Target
        y = df[target_col].values.astype(float)

        # Race boundaries
        race_ids = df[raceid_col].values
        _, first_idx, counts = np.unique(
            race_ids, return_index=True, return_counts=True
        )
        order = np.argsort(first_idx)
        first_idx = first_idx[order]
        race_starts = np.concatenate([first_idx, [len(df)]])

        return X, y, race_starts

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "won",
        raceid_col: str = "raceid",
    ) -> dict:
        """Fit the conditional logit model.

        Args:
            train_df: Training data with features and outcomes.
            val_df: Validation data for evaluation.
            feature_cols: Feature column names (pre-race only, no BFSP).
            target_col: Binary win indicator column.
            raceid_col: Race identifier column.

        Returns:
            Dict of training metrics.
        """
        self.feature_cols = list(feature_cols)

        # Filter to valid races
        train_clean = self._filter_valid_races(
            train_df, feature_cols, raceid_col, target_col
        )
        val_clean = self._filter_valid_races(
            val_df, feature_cols, raceid_col, target_col
        )

        if len(train_clean) < 100 or len(val_clean) < 10:
            log.warning("Insufficient data for conditional logit training")
            return {"error": "insufficient_data"}

        n_features = len(feature_cols)
        log.info(
            f"  Conditional logit: {len(train_clean):,} train, "
            f"{len(val_clean):,} val, {n_features} features"
        )

        # Prepare training data
        X_train, y_train, rs_train = self._prepare_data(
            train_clean, feature_cols, raceid_col, target_col,
            fit_scaler=True,
        )

        # Initial parameters
        beta0 = np.zeros(n_features)

        # Optimise
        result = minimize(
            _neg_log_likelihood,
            beta0,
            args=(X_train, y_train, rs_train, self.alpha),
            jac=_neg_log_likelihood_grad,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "disp": False},
        )

        self.beta_ = result.x

        log.info(f"  Converged: {result.success}, iterations: {result.nit}")

        # Top features
        feat_importance = sorted(
            zip(feature_cols, np.abs(self.beta_)),
            key=lambda x: -x[1],
        )
        log.info("  Top 10 features by |beta|:")
        for fname, fval in feat_importance[:10]:
            idx = feature_cols.index(fname)
            log.info(f"    {fname}: {self.beta_[idx]:+.4f}")

        # Evaluate on validation set
        from sklearn.metrics import log_loss as sk_log_loss

        X_val, y_val, rs_val = self._prepare_data(
            val_clean, feature_cols, raceid_col, target_col,
        )
        V_val = X_val @ self.beta_
        val_probs = _softmax_by_race(V_val, rs_val)
        val_ll = sk_log_loss(y_val, np.clip(val_probs, 1e-7, 1 - 1e-7))

        # Random baseline (1/N)
        n_runners = val_clean["number_of_runners"].values.astype(float)
        random_probs = np.clip(1.0 / n_runners, 1e-7, 1 - 1e-7)
        random_ll = sk_log_loss(y_val, random_probs)

        metrics = {
            "val_logloss": round(val_ll, 6),
            "random_logloss": round(random_ll, 6),
            "improvement_vs_random": round(random_ll - val_ll, 6),
            "n_features": n_features,
            "train_size": len(train_clean),
            "val_size": len(val_clean),
            "converged": result.success,
            "n_iterations": result.nit,
            "alpha": self.alpha,
        }

        self.metadata = metrics
        return metrics

    def predict(
        self,
        df: pd.DataFrame,
        raceid_col: str = "raceid",
    ) -> pd.DataFrame:
        """Predict win probabilities and derive fair BFSP.

        Returns DataFrame with:
            p_model: predicted win probability (sums to 1 per race)
            predicted_bfsp: fair BFSP (1 / p_model)
        """
        if self.beta_ is None:
            raise ValueError("Model not trained. Call train() first.")

        clean = df.copy()
        clean = clean.sort_values([raceid_col]).reset_index(drop=True)

        X, _, race_starts = self._prepare_data(
            clean, self.feature_cols, raceid_col,
            target_col="won" if "won" in clean.columns else "placing_numerical",
        )

        V = X @ self.beta_
        probs = _softmax_by_race(V, race_starts)

        result = clean.copy()
        result["p_model"] = probs
        result["predicted_bfsp"] = 1.0 / np.clip(probs, 1e-10, None)

        return result

    def _filter_valid_races(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        raceid_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Filter to races with valid targets and at least 2 runners."""
        clean = df.copy()
        clean = clean[clean[target_col].notna()]

        # At least 2 runners per race
        race_sizes = clean.groupby(raceid_col).size()
        valid_races = race_sizes[race_sizes >= 2].index
        clean = clean[clean[raceid_col].isin(valid_races)]

        # Exactly one winner per race
        race_winners = clean.groupby(raceid_col)[target_col].sum()
        single_winner = race_winners[race_winners == 1].index
        clean = clean[clean[raceid_col].isin(single_winner)]

        return clean.reset_index(drop=True)

    def save(self, directory: str):
        """Save model parameters."""
        os.makedirs(directory, exist_ok=True)

        params = {
            "beta": self.beta_.tolist(),
            "feature_cols": self.feature_cols,
            "alpha": self.alpha,
            "metrics": self.metadata,
        }
        with open(os.path.join(directory, "conditional_logit.json"), "w") as f:
            json.dump(params, f, indent=2)

        if self.scaler is not None:
            scaler_params = {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            }
            with open(
                os.path.join(directory, "conditional_logit_scaler.json"), "w"
            ) as f:
                json.dump(scaler_params, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "BenterConditionalLogit":
        """Load a saved model."""
        with open(os.path.join(directory, "conditional_logit.json")) as f:
            params = json.load(f)

        instance = cls(alpha=params.get("alpha", 1.0))
        instance.beta_ = np.array(params["beta"])
        instance.feature_cols = params["feature_cols"]
        instance.metadata = params.get("metrics", {})

        scaler_path = os.path.join(directory, "conditional_logit_scaler.json")
        if os.path.exists(scaler_path):
            with open(scaler_path) as f:
                scaler_params = json.load(f)
            instance.scaler = StandardScaler()
            instance.scaler.mean_ = np.array(scaler_params["mean"])
            instance.scaler.scale_ = np.array(scaler_params["scale"])
            instance.scaler.var_ = instance.scaler.scale_ ** 2
            instance.scaler.n_features_in_ = len(instance.scaler.mean_)

        return instance

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance (absolute coefficient values)."""
        if self.beta_ is None:
            raise ValueError("Model not trained.")
        return pd.DataFrame({
            "feature": self.feature_cols,
            "coefficient": self.beta_,
            "abs_coefficient": np.abs(self.beta_),
        }).sort_values("abs_coefficient", ascending=False)
