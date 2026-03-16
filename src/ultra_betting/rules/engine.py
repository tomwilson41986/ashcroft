"""Rules engine — reads rules.yaml, outputs bet/no-bet decisions.

Pure function: (predictions, live_prices, rules_config) -> list[BetInstruction]
No side effects. No API calls. Just applies rules to data.
"""

import logging

from ultra_betting.config import load_rules
from ultra_betting.data.schemas import BetInstruction, Prediction, SkippedBet
from ultra_betting.betfair.pricing import MarketPrices
from ultra_betting.rules.edge import calculate_edge
from ultra_betting.rules.staking import calculate_stake

log = logging.getLogger(__name__)


class RulesEngine:
    """Evaluate predictions against live prices using configured rules."""

    def __init__(self, rules_config: dict | None = None):
        self.config = rules_config or load_rules()

    def evaluate(
        self,
        predictions: list[Prediction],
        live_prices: dict[str, MarketPrices],
    ) -> tuple[list[BetInstruction], list[SkippedBet]]:
        """Evaluate all predictions and return bet instructions + skipped reasons.

        Args:
            predictions: Model predictions for today's runners.
            live_prices: {market_id: MarketPrices} with live exchange prices.

        Returns:
            (instructions, skipped) — bets to place and reasons for skips.
        """
        instructions = []
        skipped = []
        entry = self.config.get("entry_conditions", {})
        side_config = self.config.get("side_logic", {})
        staking_config = self.config.get("staking", {})

        for pred in predictions:
            # Find live prices for this market
            prices = live_prices.get(pred.market_id)
            if prices is None:
                skipped.append(SkippedBet(
                    prediction_id=pred.prediction_id,
                    runner_name=pred.runner_name,
                    venue=pred.venue,
                    race_time=pred.race_time,
                    reason="no_market_prices",
                    detail=f"No live prices for market {pred.market_id}",
                ))
                continue

            # Check market status
            if prices.status != "OPEN":
                skipped.append(SkippedBet(
                    prediction_id=pred.prediction_id,
                    runner_name=pred.runner_name,
                    venue=pred.venue,
                    race_time=pred.race_time,
                    reason="market_not_open",
                    detail=f"Market status: {prices.status}",
                ))
                continue

            # Get live price
            side = self._determine_side(pred, prices, side_config)
            if side == "BACK":
                live_price = prices.get_best_back(pred.selection_id)
            else:
                live_price = prices.get_best_lay(pred.selection_id)

            if live_price is None:
                skipped.append(SkippedBet(
                    prediction_id=pred.prediction_id,
                    runner_name=pred.runner_name,
                    venue=pred.venue,
                    race_time=pred.race_time,
                    reason="no_price_available",
                    detail=f"No {side} price for selection {pred.selection_id}",
                ))
                continue

            # Calculate edge
            edge = calculate_edge(pred.predicted_bfsp, live_price, side)

            # Check entry conditions
            skip_reason = self._check_entry_conditions(pred, edge, entry)
            if skip_reason:
                skipped.append(SkippedBet(
                    prediction_id=pred.prediction_id,
                    runner_name=pred.runner_name,
                    venue=pred.venue,
                    race_time=pred.race_time,
                    reason="entry_condition_failed",
                    detail=skip_reason,
                ))
                continue

            # Check market liquidity
            min_matched = entry.get("min_market_total_matched", 0)
            if min_matched and prices.total_matched < min_matched:
                skipped.append(SkippedBet(
                    prediction_id=pred.prediction_id,
                    runner_name=pred.runner_name,
                    venue=pred.venue,
                    race_time=pred.race_time,
                    reason="illiquid_market",
                    detail=f"Total matched {prices.total_matched:.0f} < {min_matched}",
                ))
                continue

            # Calculate stake
            stake = calculate_stake(
                pred.predicted_win_prob, edge, live_price, staking_config
            )

            if stake <= 0:
                skipped.append(SkippedBet(
                    prediction_id=pred.prediction_id,
                    runner_name=pred.runner_name,
                    venue=pred.venue,
                    race_time=pred.race_time,
                    reason="zero_stake",
                    detail=f"Calculated stake: £{stake:.2f}",
                ))
                continue

            persistence = self.config.get("persistence", {}).get("type", "LAPSE")

            instructions.append(BetInstruction(
                prediction_id=pred.prediction_id,
                market_id=pred.market_id,
                selection_id=pred.selection_id,
                runner_name=pred.runner_name,
                side=side,
                stake=round(stake, 2),
                price=live_price,
                persistence=persistence,
                reasoning=f"Edge: {edge:.1f}%, Pred BFSP: {pred.predicted_bfsp:.2f}, "
                          f"Live: {live_price:.2f}, P(win): {pred.predicted_win_prob:.3f}",
            ))

        log.info(
            f"Rules engine: {len(instructions)} bets, {len(skipped)} skipped "
            f"from {len(predictions)} predictions"
        )
        return instructions, skipped

    def _determine_side(
        self, pred: Prediction, prices: MarketPrices, side_config: dict
    ) -> str:
        """Determine whether to BACK or LAY."""
        mode = side_config.get("mode", "back")
        if mode == "back":
            return "BACK"
        elif mode == "lay":
            return "LAY"
        else:
            # "both" — back if predicted shorter, lay if predicted longer
            best_back = prices.get_best_back(pred.selection_id)
            if best_back and pred.predicted_bfsp < best_back:
                return "BACK"
            return "LAY"

    def _check_entry_conditions(
        self, pred: Prediction, edge: float, conditions: dict
    ) -> str | None:
        """Check if entry conditions are met. Returns skip reason or None."""
        min_edge = conditions.get("min_edge_percent", 0)
        if edge < min_edge:
            return f"Edge {edge:.1f}% < min {min_edge}%"

        min_prob = conditions.get("min_win_probability", 0)
        if pred.predicted_win_prob < min_prob:
            return f"P(win) {pred.predicted_win_prob:.3f} < min {min_prob}"

        max_bfsp = conditions.get("max_predicted_bfsp", float("inf"))
        if pred.predicted_bfsp > max_bfsp:
            return f"Predicted BFSP {pred.predicted_bfsp:.2f} > max {max_bfsp}"

        min_bfsp = conditions.get("min_predicted_bfsp", 0)
        if pred.predicted_bfsp < min_bfsp:
            return f"Predicted BFSP {pred.predicted_bfsp:.2f} < min {min_bfsp}"

        return None
