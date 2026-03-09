"""Stage 3: Overlay detection and Kelly staking.

Identifies runners where our blended probability exceeds market-implied
probability (overlays), and sizes bets using fractional Kelly criterion.
"""

import numpy as np
import pandas as pd


class OverlayDetector:
    """Identify overlays (underpriced runners) and size bets using fractional Kelly.

    An overlay exists when P_combined > P_market, meaning our blended model
    thinks the horse has a higher chance of winning than the market implies.

    Edge = P_combined / P_market - 1

    We only bet when edge exceeds a minimum threshold (default 5%).
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
        """
        Args:
            min_edge: Minimum edge (as fraction) required to place a bet.
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly).
            bankroll: Starting bankroll in GBP.
            max_stake_pct: Maximum single bet as fraction of bankroll.
            min_stake: Minimum stake in GBP.
            max_stake: Maximum stake in GBP.
            commission_rate: Betfair commission rate on net winnings.
        """
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.bankroll = bankroll
        self.max_stake_pct = max_stake_pct
        self.min_stake = min_stake
        self.max_stake = max_stake
        self.commission_rate = commission_rate

    def kelly_stake(self, p_combined: float, bfsp: float) -> float:
        """Calculate stake using fractional Kelly criterion.

        f* = (p * b - q) / b
        where p = p_combined, b = bfsp - 1, q = 1 - p

        Actual stake = kelly_fraction * f* * bankroll

        Constraints:
        - No bet if edge < min_edge
        - No single bet > max_stake_pct of bankroll
        - Minimum stake of min_stake
        - Maximum stake of max_stake
        """
        if not np.isfinite(bfsp) or bfsp <= 1.0 or not np.isfinite(p_combined):
            return 0.0

        p = p_combined
        q = 1.0 - p
        b = bfsp - 1.0  # decimal odds profit per unit

        # Edge check
        edge = (p * bfsp) - 1.0  # expected return per unit
        if edge / 1.0 < self.min_edge:
            return 0.0

        # Kelly fraction
        f_star = (p * b - q) / b
        if f_star <= 0:
            return 0.0

        stake = self.kelly_fraction * f_star * self.bankroll

        # Apply constraints
        max_by_pct = self.max_stake_pct * self.bankroll
        stake = min(stake, max_by_pct, self.max_stake)

        if stake < self.min_stake:
            return 0.0

        return round(stake, 2)

    def generate_bet_card(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate the day's bet card from blended predictions.

        Args:
            predictions_df: DataFrame from BenterBlender.blend() output with columns:
                p_model, p_public, p_combined, bfsp, edge, edge_pct,
                horse_name, and optionally date/course/race_time.

        Returns:
            DataFrame of qualified bets sorted by expected value descending.
            Columns: date, course, race_time, horse_name, p_model, p_public,
            p_combined, bfsp, model_fair_bfsp, edge, edge_pct, kelly_f,
            adjusted_kelly_f, stake_gbp, expected_value_gbp, bet_type.
        """
        df = predictions_df.copy()

        # Filter to runners with positive edge above threshold
        mask = df["edge"].notna() & (df["edge"] >= self.min_edge)
        bets = df[mask].copy()

        if bets.empty:
            return self._empty_bet_card()

        # Calculate Kelly and stakes for each qualifying runner
        bets["kelly_f"] = bets.apply(
            lambda r: self._full_kelly(r["p_combined"], r["bfsp"]), axis=1
        )
        bets["adjusted_kelly_f"] = bets["kelly_f"] * self.kelly_fraction
        bets["stake_gbp"] = bets.apply(
            lambda r: self.kelly_stake(r["p_combined"], r["bfsp"]), axis=1
        )

        # Remove rows where stake is 0 (below minimum)
        bets = bets[bets["stake_gbp"] > 0].copy()

        if bets.empty:
            return self._empty_bet_card()

        # Expected value per bet
        bets["expected_value_gbp"] = bets.apply(
            lambda r: self._expected_value(r["p_combined"], r["bfsp"], r["stake_gbp"]),
            axis=1,
        )

        # Bet type
        bets["bet_type"] = "BSP"

        # Select and order output columns
        output_cols = [
            "horse_name", "p_model", "p_public", "p_combined",
            "bfsp", "model_fair_bfsp", "edge", "edge_pct",
            "kelly_f", "adjusted_kelly_f", "stake_gbp",
            "expected_value_gbp", "bet_type",
        ]
        # Include optional context columns if present
        for col in ["date", "course", "race_time", "raceid"]:
            if col in bets.columns:
                output_cols.insert(0, col)

        available_cols = [c for c in output_cols if c in bets.columns]
        bets = bets[available_cols].sort_values("expected_value_gbp", ascending=False)

        return bets.reset_index(drop=True)

    def _full_kelly(self, p: float, bfsp: float) -> float:
        """Full Kelly fraction (before applying kelly_fraction multiplier)."""
        if bfsp <= 1.0 or not np.isfinite(p) or not np.isfinite(bfsp):
            return 0.0
        b = bfsp - 1.0
        q = 1.0 - p
        f = (p * b - q) / b
        return max(f, 0.0)

    def _expected_value(self, p: float, bfsp: float, stake: float) -> float:
        """Expected value of a bet accounting for commission."""
        if bfsp <= 1.0 or stake <= 0:
            return 0.0
        win_return = stake * (bfsp - 1.0) * (1 - self.commission_rate)
        ev = p * win_return - (1 - p) * stake
        return round(ev, 2)

    def _empty_bet_card(self) -> pd.DataFrame:
        """Return an empty DataFrame with the bet card schema."""
        return pd.DataFrame(columns=[
            "horse_name", "p_model", "p_public", "p_combined",
            "bfsp", "model_fair_bfsp", "edge", "edge_pct",
            "kelly_f", "adjusted_kelly_f", "stake_gbp",
            "expected_value_gbp", "bet_type",
        ])

    def update_bankroll(self, result: str, stake: float, bfsp: float) -> float:
        """Update bankroll after a bet settles.

        Args:
            result: 'win' or 'lose'.
            stake: Amount staked.
            bfsp: Betfair SP the bet settled at.

        Returns:
            Net P&L from this bet.
        """
        if result == "win":
            gross_profit = stake * (bfsp - 1.0)
            commission = gross_profit * self.commission_rate
            net_pl = gross_profit - commission
        else:
            net_pl = -stake

        self.bankroll += net_pl
        return round(net_pl, 2)
