"""Stage 2: Benter (1994) log-linear probability blending.

Combines the fundamental model's probabilities with market-implied
probabilities (from BFSP) using a log-linear blending formula:

    P_combined(i) = [P_model(i)^(1-lambda) * P_public(i)^lambda] / Z

Where Z is a normalisation constant so probabilities sum to 1 within a race.

lambda ~ 0.80 means the market provides ~80% of the signal. The model's value
is in the remaining 20% where it systematically disagrees with the market.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


class BenterBlender:
    """Benter (1994) two-stage log-linear probability blending."""

    def __init__(self, lambda_: float = 0.80):
        """
        Args:
            lambda_: Blending weight. 0 = pure model, 1 = pure market.
                     Benter's typical value is ~0.80.
        """
        self.lambda_ = lambda_

    def blend(self, race_df: pd.DataFrame, lambda_: float | None = None,
              race_id_col: str = "raceid") -> pd.DataFrame:
        """Blend model and market probabilities.

        Args:
            race_df: DataFrame with columns 'p_model' and 'bfsp' for runners.
                     Can contain multiple races identified by race_id_col.
            lambda_: Override blending weight. Uses self.lambda_ if None.
            race_id_col: Column identifying unique races.

        Returns:
            DataFrame with added columns:
                - p_public: market implied probability (1/BFSP)
                - p_combined: blended probability
                - model_fair_bfsp: 1/p_combined
                - edge: (p_combined / p_public) - 1
                - edge_pct: edge * 100
                - blend_available: whether blending was applied
        """
        lam = lambda_ if lambda_ is not None else self.lambda_
        result = race_df.copy()

        # Market-implied probability
        has_bfsp = result["bfsp"].notna() & (result["bfsp"] > 1.0)
        result["p_public"] = np.where(has_bfsp, 1.0 / result["bfsp"], np.nan)
        result["blend_available"] = has_bfsp

        # For runners with BFSP: log-linear blend
        # P_combined(i) = P_model(i)^(1-lambda) * P_public(i)^lambda
        # Then normalise within race
        result["p_combined"] = result["p_model"].copy()

        if has_bfsp.any():
            mask = has_bfsp
            p_model = result.loc[mask, "p_model"].clip(lower=1e-10)
            p_public = result.loc[mask, "p_public"].clip(lower=1e-10)
            blended_raw = (p_model ** (1 - lam)) * (p_public ** lam)
            result.loc[mask, "p_combined"] = blended_raw.values

        # Normalise within each race
        if race_id_col in result.columns:
            result["p_combined"] = result.groupby(race_id_col)["p_combined"].transform(
                lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x)
            )
        else:
            total = result["p_combined"].sum()
            if total > 0:
                result["p_combined"] = result["p_combined"] / total

        # Derived columns
        result["model_fair_bfsp"] = 1.0 / result["p_combined"].clip(lower=1e-10)
        result["edge"] = np.where(
            result["p_public"].notna() & (result["p_public"] > 0),
            (result["p_combined"] / result["p_public"]) - 1.0,
            np.nan,
        )
        result["edge_pct"] = result["edge"] * 100

        return result

    def optimise_lambda(self, val_df: pd.DataFrame,
                        race_id_col: str = "raceid",
                        target_col: str = "won",
                        lambda_range: tuple[float, float] = (0.0, 1.0),
                        step: float = 0.05) -> float:
        """Find optimal lambda by minimising log-loss on validation data.

        Args:
            val_df: Validation data with 'p_model', 'bfsp', and target columns.
            race_id_col: Race identifier column.
            target_col: Binary outcome column.
            lambda_range: (min, max) range for lambda search.
            step: Step size for grid search.

        Returns:
            Optimal lambda value.
        """
        # Only use races where BFSP is available
        mask = val_df["bfsp"].notna() & (val_df["bfsp"] > 1.0)
        if mask.sum() < 10:
            return self.lambda_

        val_subset = val_df[mask].copy()

        best_lambda = self.lambda_
        best_loss = float("inf")

        lam = lambda_range[0]
        while lam <= lambda_range[1] + 1e-9:
            blended = self.blend(val_subset, lambda_=lam, race_id_col=race_id_col)
            p = blended["p_combined"].clip(1e-10, 1 - 1e-10)
            y = val_subset[target_col].values
            try:
                loss = log_loss(y, p)
                if loss < best_loss:
                    best_loss = loss
                    best_lambda = lam
            except Exception:
                pass
            lam += step

        self.lambda_ = round(best_lambda, 3)
        return self.lambda_

    def blend_single_race(self, p_model: np.ndarray, bfsp: np.ndarray,
                          lambda_: float | None = None) -> dict:
        """Convenience method for blending a single race from arrays.

        Args:
            p_model: Array of model probabilities (one per runner).
            bfsp: Array of Betfair Starting Prices.
            lambda_: Optional override.

        Returns:
            Dict with 'p_combined', 'edge', 'model_fair_bfsp' arrays.
        """
        lam = lambda_ if lambda_ is not None else self.lambda_
        p_model = np.asarray(p_model, dtype=float).clip(min=1e-10)
        bfsp = np.asarray(bfsp, dtype=float)

        p_public = 1.0 / np.clip(bfsp, 1.01, None)

        blended = (p_model ** (1 - lam)) * (p_public ** lam)
        total = blended.sum()
        p_combined = blended / total if total > 0 else np.ones_like(blended) / len(blended)

        edge = (p_combined / p_public) - 1.0
        fair_bfsp = 1.0 / np.clip(p_combined, 1e-10, None)

        return {
            "p_combined": p_combined,
            "p_public": p_public,
            "edge": edge,
            "model_fair_bfsp": fair_bfsp,
        }
