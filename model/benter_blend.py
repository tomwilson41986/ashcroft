"""
Stage 2: Benter (1994) Log-Linear Probability Blending.

Combines the fundamental model's win probabilities with the market's
implied probabilities (from BFSP) using a log-linear formula:

    P_combined(i) = [P_model(i)^(1-λ) * P_public(i)^λ] / Z

Where λ ≈ 0.80 means the public odds provide ~80% of the signal.
The model's value is in the remaining ~20% where it systematically
disagrees with the market.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


class BenterBlender:
    """Benter two-stage log-linear probability blending.

    Blends model probabilities with market probabilities to produce
    a combined estimate that leverages both the model's insights and
    the market's collective wisdom.

    Attributes:
        optimal_lambda: Best lambda found by optimise_lambda().
    """

    def __init__(self):
        self.optimal_lambda: float | None = None

    def blend(
        self,
        race_df: pd.DataFrame,
        lambda_: float = 0.80,
        p_model_col: str = "p_model",
        bfsp_col: str = "bfsp",
    ) -> pd.DataFrame:
        """Blend model and market probabilities for runners.

        Can handle a single race or multiple races (grouped by raceid).

        Args:
            race_df: DataFrame with p_model and bfsp columns.
            lambda_: Blending weight (0=pure model, 1=pure market).
            p_model_col: Column name for model probabilities.
            bfsp_col: Column name for Betfair Starting Price.

        Returns:
            DataFrame with added columns: p_public, p_combined,
            model_fair_bfsp, edge, edge_pct, blend_available.
        """
        result = race_df.copy()

        # Market implied probability
        result["p_public"] = 1.0 / result[bfsp_col].replace(0, np.nan)

        # Check if blending is possible
        has_market = result["p_public"].notna() & (result["p_public"] > 0)
        has_model = result[p_model_col].notna() & (result[p_model_col] > 0)
        result["blend_available"] = has_market & has_model

        # Log-linear blending: P_combined ∝ P_model^(1-λ) * P_public^λ
        p_model = result[p_model_col].clip(lower=1e-10)
        p_public = result["p_public"].clip(lower=1e-10)

        raw_combined = np.where(
            result["blend_available"],
            np.power(p_model, 1 - lambda_) * np.power(p_public, lambda_),
            p_model,  # Fall back to pure model when no market data
        )

        # Normalise per race so probabilities sum to 1.0
        if "raceid" in result.columns:
            result["_raw_combined"] = raw_combined
            result["p_combined"] = result.groupby("raceid")[
                "_raw_combined"
            ].transform(lambda x: x / x.sum())
            result.drop(columns=["_raw_combined"], inplace=True)
        else:
            # Single race — normalise across all rows
            total = raw_combined.sum()
            result["p_combined"] = (
                raw_combined / total if total > 0 else raw_combined
            )

        # Model's fair BFSP (what the price "should" be)
        result["model_fair_bfsp"] = 1.0 / result["p_combined"].replace(
            0, np.nan
        )

        # Edge: positive means overlay (horse underpriced by market)
        result["edge"] = np.where(
            result["p_public"] > 0,
            (result["p_combined"] / result["p_public"]) - 1,
            0,
        )
        result["edge_pct"] = result["edge"] * 100

        return result

    def optimise_lambda(
        self,
        val_df: pd.DataFrame,
        p_model_col: str = "p_model",
        bfsp_col: str = "bfsp",
        target_col: str = "won",
        lambda_range: tuple[float, float] = (0.0, 1.0),
        step: float = 0.05,
    ) -> float:
        """Find optimal lambda by minimising log-loss on validation data.

        Args:
            val_df: Validation DataFrame with model probs, BFSP, and outcomes.
            p_model_col: Column with model probabilities.
            bfsp_col: Column with BFSP.
            target_col: Binary outcome column (1=win).
            lambda_range: Range of lambda values to search.
            step: Step size for grid search.

        Returns:
            Optimal lambda value.
        """
        # Filter to rows with both model and market data
        valid = val_df[
            val_df[p_model_col].notna()
            & val_df[bfsp_col].notna()
            & (val_df[bfsp_col] > 1)
        ].copy()

        if len(valid) < 100:
            self.optimal_lambda = 0.80
            return 0.80

        best_ll = float("inf")
        best_lambda = 0.80

        lambdas = np.arange(lambda_range[0], lambda_range[1] + step, step)

        for lam in lambdas:
            blended = self.blend(
                valid,
                lambda_=lam,
                p_model_col=p_model_col,
                bfsp_col=bfsp_col,
            )
            probs = blended["p_combined"].clip(1e-7, 1 - 1e-7)
            ll = log_loss(valid[target_col], probs)

            if ll < best_ll:
                best_ll = ll
                best_lambda = lam

        self.optimal_lambda = round(best_lambda, 2)
        return self.optimal_lambda

    def blend_with_optimal(
        self,
        df: pd.DataFrame,
        p_model_col: str = "p_model",
        bfsp_col: str = "bfsp",
    ) -> pd.DataFrame:
        """Blend using the previously optimised lambda."""
        if self.optimal_lambda is None:
            raise ValueError(
                "No optimal lambda set. Call optimise_lambda() first."
            )
        return self.blend(
            df,
            lambda_=self.optimal_lambda,
            p_model_col=p_model_col,
            bfsp_col=bfsp_col,
        )
