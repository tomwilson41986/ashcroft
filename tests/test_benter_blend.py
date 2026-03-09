"""Tests for BenterBlender — log-linear probability blending."""

import numpy as np
import pandas as pd
import pytest

from src.models.benter_blend import BenterBlender


def _make_race_data(n_runners=8):
    """Create a single race with model probabilities and BFSP."""
    p_model = np.random.dirichlet(np.ones(n_runners))
    bfsp = 1.0 / p_model * 1.1  # Slightly overround market

    return pd.DataFrame({
        "raceid": ["race_1"] * n_runners,
        "horse_name": [f"Horse_{i}" for i in range(n_runners)],
        "p_model": p_model,
        "bfsp": bfsp,
        "won": [1] + [0] * (n_runners - 1),
    })


class TestBenterBlender:
    def test_lambda_zero_gives_pure_model(self):
        """lambda=0 should give pure model probabilities."""
        data = _make_race_data()
        blender = BenterBlender()
        result = blender.blend(data, lambda_=0.0)

        np.testing.assert_allclose(
            result["p_combined"].values,
            data["p_model"].values / data["p_model"].sum(),
            atol=1e-6,
        )

    def test_lambda_one_gives_pure_market(self):
        """lambda=1 should give pure market probabilities."""
        data = _make_race_data()
        blender = BenterBlender()
        result = blender.blend(data, lambda_=1.0)

        p_public = 1.0 / data["bfsp"].values
        p_public_norm = p_public / p_public.sum()

        np.testing.assert_allclose(
            result["p_combined"].values,
            p_public_norm,
            atol=1e-6,
        )

    def test_probabilities_sum_to_one(self):
        """Blended probabilities must sum to 1.0 within each race."""
        data = _make_race_data()
        blender = BenterBlender(lambda_=0.80)
        result = blender.blend(data)

        race_sum = result.groupby("raceid")["p_combined"].sum()
        np.testing.assert_allclose(race_sum.values, 1.0, atol=1e-6)

    def test_edge_calculation(self):
        """Edge should be (p_combined / p_public) - 1."""
        data = _make_race_data()
        blender = BenterBlender()
        result = blender.blend(data, lambda_=0.5)

        expected_edge = result["p_combined"] / result["p_public"] - 1.0
        np.testing.assert_allclose(result["edge"].values, expected_edge.values, atol=1e-6)

    def test_model_fair_bfsp(self):
        """model_fair_bfsp should be 1/p_combined."""
        data = _make_race_data()
        blender = BenterBlender()
        result = blender.blend(data)

        expected_fair = 1.0 / result["p_combined"]
        np.testing.assert_allclose(result["model_fair_bfsp"].values, expected_fair.values, atol=1e-4)

    def test_missing_bfsp_uses_model(self):
        """When BFSP is NaN, fall back to pure model probability."""
        data = _make_race_data()
        data.loc[0, "bfsp"] = np.nan
        blender = BenterBlender()
        result = blender.blend(data)

        # The row with missing BFSP should have blend_available = False
        assert result.loc[0, "blend_available"] == False

    def test_optimise_lambda(self):
        """optimise_lambda should return a float between 0 and 1."""
        np.random.seed(42)
        # Create multi-race validation data
        races = []
        for i in range(20):
            r = _make_race_data(n_runners=6)
            r["raceid"] = f"race_{i}"
            # Randomly assign a winner
            r["won"] = 0
            r.loc[r.index[np.random.randint(0, 6)], "won"] = 1
            races.append(r)
        val_df = pd.concat(races, ignore_index=True)

        blender = BenterBlender()
        opt_lam = blender.optimise_lambda(val_df)

        assert 0.0 <= opt_lam <= 1.0

    def test_blend_single_race(self):
        """Convenience method for single race from arrays."""
        p_model = np.array([0.3, 0.25, 0.2, 0.15, 0.10])
        bfsp = np.array([3.5, 4.0, 5.0, 7.0, 10.0])

        blender = BenterBlender(lambda_=0.80)
        result = blender.blend_single_race(p_model, bfsp)

        # Probabilities should sum to 1
        assert abs(result["p_combined"].sum() - 1.0) < 1e-6
        assert len(result["edge"]) == 5

    def test_multiple_races(self):
        """Blending works across multiple races in one DataFrame."""
        r1 = _make_race_data(n_runners=5)
        r1["raceid"] = "race_A"
        r2 = _make_race_data(n_runners=6)
        r2["raceid"] = "race_B"
        data = pd.concat([r1, r2], ignore_index=True)

        blender = BenterBlender()
        result = blender.blend(data)

        for rid in ["race_A", "race_B"]:
            race = result[result["raceid"] == rid]
            assert abs(race["p_combined"].sum() - 1.0) < 1e-6

    def test_edge_pct(self):
        """edge_pct should be edge * 100."""
        data = _make_race_data()
        blender = BenterBlender()
        result = blender.blend(data)

        np.testing.assert_allclose(
            result["edge_pct"].values,
            result["edge"].values * 100,
            atol=1e-6,
        )
