"""
Stage 3: Overlay Detection and Kelly Staking.

Identifies overlays (underpriced runners) where the blended model
probability exceeds the market probability. Sizes bets using fractional
Kelly criterion with safety constraints.
"""

import numpy as np
import pandas as pd


class OverlayDetector:
    """Identify value bets and calculate stakes using fractional Kelly.

    An overlay exists when P_combined > P_market, meaning our blended
    model thinks the horse has a higher chance of winning than the
    market implies.

    Args:
        min_edge: Minimum edge (%) required to bet (default 5%).
        kelly_fraction: Fraction of full Kelly to stake (default 0.25).
        bankroll: Starting bankroll in GBP.
        max_stake_pct: Maximum stake as fraction of bankroll (default 2%).
        min_stake: Minimum stake in GBP.
        max_stake: Maximum stake in GBP.
        commission_rate: Betfair commission on net winnings (default 5%).
    """

    def __init__(
        self,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.25,
        bankroll: float = 1000.0,
        max_stake_pct: float = 0.02,
        min_stake: float = 2.0,
        max_stake: float = 500.0,
        commission_rate: float = 0.05,
    ):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll
        self.max_stake_pct = max_stake_pct
        self.min_stake = min_stake
        self.max_stake = max_stake
        self.commission_rate = commission_rate

    def kelly_stake(
        self, p_combined: float, bfsp: float
    ) -> dict:
        """Calculate stake using fractional Kelly criterion.

        Kelly formula: f* = (p * b - q) / b
        where p = win probability, b = decimal odds - 1, q = 1 - p

        Args:
            p_combined: Blended win probability.
            bfsp: Betfair Starting Price (decimal odds).

        Returns:
            Dict with kelly_f, adjusted_kelly_f, stake_gbp, expected_value.
        """
        if pd.isna(p_combined) or pd.isna(bfsp) or bfsp <= 1:
            return {
                "kelly_f": 0,
                "adjusted_kelly_f": 0,
                "stake_gbp": 0,
                "expected_value": 0,
            }

        p = p_combined
        q = 1 - p
        b = bfsp - 1  # net decimal odds

        # Full Kelly fraction
        kelly_f = (p * b - q) / b if b > 0 else 0

        if kelly_f <= 0:
            return {
                "kelly_f": 0,
                "adjusted_kelly_f": 0,
                "stake_gbp": 0,
                "expected_value": 0,
            }

        # Fractional Kelly
        adjusted_f = kelly_f * self.kelly_fraction

        # Calculate stake with constraints
        stake = adjusted_f * self.bankroll
        stake = min(stake, self.bankroll * self.max_stake_pct)
        stake = min(stake, self.max_stake)
        stake = max(stake, self.min_stake) if stake >= self.min_stake else 0

        # Expected value accounting for commission
        ev = stake * (
            p * b * (1 - self.commission_rate) - q
        )

        return {
            "kelly_f": round(kelly_f, 6),
            "adjusted_kelly_f": round(adjusted_f, 6),
            "stake_gbp": round(stake, 2),
            "expected_value": round(ev, 2),
        }

    def generate_bet_card(
        self, predictions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate the day's bet card from blended predictions.

        Args:
            predictions_df: DataFrame with columns: horse_name, race_date,
                track, race_time, p_model, p_public, p_combined, bfsp,
                model_fair_bfsp, edge, edge_pct.

        Returns:
            DataFrame of qualifying bets sorted by expected value descending.
        """
        df = predictions_df.copy()

        # Filter to runners meeting minimum edge threshold
        qualifying = df[df["edge"] >= self.min_edge].copy()

        if len(qualifying) == 0:
            return pd.DataFrame()

        # Calculate Kelly stakes
        stakes = qualifying.apply(
            lambda row: self.kelly_stake(row["p_combined"], row["bfsp"]),
            axis=1,
            result_type="expand",
        )

        qualifying = pd.concat(
            [qualifying.reset_index(drop=True), stakes.reset_index(drop=True)],
            axis=1,
        )

        # Filter out zero stakes
        qualifying = qualifying[qualifying["stake_gbp"] > 0].copy()

        if len(qualifying) == 0:
            return pd.DataFrame()

        # Bet type: BSP (bet at starting price)
        qualifying["bet_type"] = "BSP"

        # Select output columns
        output_cols = [
            "race_date",
            "track",
            "race_time",
            "horse_name",
            "p_model",
            "p_public",
            "p_combined",
            "bfsp",
            "model_fair_bfsp",
            "edge",
            "edge_pct",
            "kelly_f",
            "adjusted_kelly_f",
            "stake_gbp",
            "expected_value",
            "bet_type",
        ]
        available = [c for c in output_cols if c in qualifying.columns]

        result = qualifying[available].sort_values(
            "expected_value", ascending=False
        )

        return result.reset_index(drop=True)

    def simulate_settlement(
        self, bet_card: pd.DataFrame, outcomes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Settle bets against actual race outcomes.

        Args:
            bet_card: Output from generate_bet_card().
            outcomes_df: DataFrame with horse_name, race_date, track, won columns.

        Returns:
            Bet card with added columns: won, pnl, running_pnl.
        """
        result = bet_card.copy()

        # Merge outcomes
        merge_keys = ["horse_name"]
        if "race_date" in result.columns and "race_date" in outcomes_df.columns:
            merge_keys.append("race_date")
        if "track" in result.columns and "track" in outcomes_df.columns:
            merge_keys.append("track")

        result = result.merge(
            outcomes_df[merge_keys + ["won"]],
            on=merge_keys,
            how="left",
            suffixes=("", "_outcome"),
        )

        won_col = "won_outcome" if "won_outcome" in result.columns else "won"

        # Calculate P&L per bet
        result["pnl"] = np.where(
            result[won_col] == 1,
            result["stake_gbp"]
            * (result["bfsp"] - 1)
            * (1 - self.commission_rate),
            -result["stake_gbp"],
        )

        result["running_pnl"] = result["pnl"].cumsum()
        result["roi_pct"] = (
            result["pnl"].cumsum() / result["stake_gbp"].cumsum() * 100
        )

        return result
