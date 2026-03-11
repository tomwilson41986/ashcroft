"""Bet placement engine with safety controls.

Wraps :class:`BetfairClient` with daily loss limits, exposure caps,
bet count limits, and price bounds.  Supports a dry-run mode where
everything runs except the actual API call to place orders.
"""

import logging

import pandas as pd

from betfair.audit import BetAuditLog
from betfair.client import BetfairClient
from betfair.config import BettingConfig

log = logging.getLogger(__name__)

# Betfair price increments (inclusive lower bound -> tick size)
_TICK_RANGES = [
    (1.01, 2.00, 0.01),
    (2.00, 3.00, 0.02),
    (3.00, 4.00, 0.05),
    (4.00, 6.00, 0.10),
    (6.00, 10.0, 0.20),
    (10.0, 20.0, 0.50),
    (20.0, 30.0, 1.00),
    (30.0, 50.0, 2.00),
    (50.0, 100.0, 5.00),
    (100.0, 1000.0, 10.0),
]


def round_to_betfair_tick(price: float) -> float:
    """Round a price down to the nearest valid Betfair tick."""
    if price < 1.01:
        return 1.01
    for low, high, tick in _TICK_RANGES:
        if low <= price < high:
            return round(int(price / tick) * tick, 2)
    return round(int(price / 10.0) * 10.0, 2)


class BettingEngine:
    """Place bets from an overlay detector bet card with safety controls.

    Args:
        client: Authenticated :class:`BetfairClient` instance.
        config: :class:`BettingConfig` with limits and parameters.
        audit: :class:`BetAuditLog` for recording placed bets.
    """

    def __init__(
        self,
        client: BetfairClient,
        config: BettingConfig,
        audit: BetAuditLog,
    ):
        self.client = client
        self.config = config
        self.audit = audit

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def check_kill_switch(self) -> bool:
        """Return ``True`` if betting is enabled."""
        if not self.config.enabled:
            log.warning("KILL SWITCH: betting is disabled (BETTING_ENABLED=false)")
            return False
        return True

    def check_daily_limits(self, proposed_stake: float = 0.0) -> bool:
        """Return ``True`` if daily limits allow another bet."""
        exposure = self.audit.get_daily_exposure()
        bet_count = self.audit.get_daily_bet_count()

        if exposure + proposed_stake > self.config.max_daily_exposure:
            log.warning(
                f"Daily exposure limit reached: "
                f"{exposure:.2f} + {proposed_stake:.2f} > "
                f"{self.config.max_daily_exposure:.2f}"
            )
            return False

        if bet_count >= self.config.max_daily_bets:
            log.warning(
                f"Daily bet count limit reached: "
                f"{bet_count} >= {self.config.max_daily_bets}"
            )
            return False

        daily_pnl = self.audit.get_daily_pnl()
        if daily_pnl < -self.config.max_daily_loss:
            log.warning(
                f"Daily loss limit reached: "
                f"P&L={daily_pnl:.2f} < -{self.config.max_daily_loss:.2f}"
            )
            return False

        return True

    def check_race_limits(self, market_id: str) -> bool:
        """Return ``True`` if we haven't hit the per-race bet limit."""
        race_bets = self.audit.get_race_bet_count(market_id)
        if race_bets >= self.config.max_single_race_bets:
            log.info(
                f"Race bet limit for {market_id}: "
                f"{race_bets} >= {self.config.max_single_race_bets}"
            )
            return False
        return True

    def check_price_bounds(self, price: float) -> bool:
        """Return ``True`` if the price is within configured bounds."""
        return self.config.min_back_price <= price <= self.config.max_back_price

    # ------------------------------------------------------------------
    # Bet placement
    # ------------------------------------------------------------------

    def place_bet_card(self, bet_card: pd.DataFrame) -> pd.DataFrame:
        """Place all qualifying bets from an overlay detector bet card.

        Expected columns in *bet_card*: ``horse_name``, ``market_id``,
        ``selection_id``, ``stake_gbp``, ``live_back_price``,
        ``p_combined``, ``edge``, ``edge_pct``, ``track``, ``race_time``.

        Returns a copy of the bet card with added columns:
        ``bet_id``, ``bet_status``, ``size_matched``, ``placed``.
        """
        result = bet_card.copy()
        result["bet_id"] = None
        result["bet_status"] = "SKIPPED"
        result["size_matched"] = 0.0
        result["placed"] = False

        if not self.check_kill_switch():
            log.warning("All bets skipped: kill switch active")
            return result

        placed_count = 0

        for idx, row in result.iterrows():
            market_id = row.get("market_id")
            selection_id = row.get("selection_id")
            stake = row.get("stake_gbp", 0)
            price = row.get("live_back_price", 0)

            if pd.isna(selection_id) or pd.isna(market_id):
                result.at[idx, "bet_status"] = "NO_MATCH"
                continue

            # Round to valid Betfair tick
            tick_price = round_to_betfair_tick(price)

            if not self.check_price_bounds(tick_price):
                result.at[idx, "bet_status"] = "PRICE_OUT_OF_RANGE"
                log.info(
                    f"Skipping {row.get('horse_name')}: "
                    f"price {tick_price} outside [{self.config.min_back_price}, "
                    f"{self.config.max_back_price}]"
                )
                continue

            if not self.check_daily_limits(stake):
                result.at[idx, "bet_status"] = "DAILY_LIMIT"
                continue

            if not self.check_race_limits(str(market_id)):
                result.at[idx, "bet_status"] = "RACE_LIMIT"
                continue

            # Ensure minimum stake
            if stake < self.config.min_stake:
                result.at[idx, "bet_status"] = "BELOW_MIN_STAKE"
                continue

            instruction = {
                "selection_id": int(selection_id),
                "side": "BACK",
                "price": tick_price,
                "size": round(stake, 2),
            }

            if self.config.dry_run:
                log.info(
                    f"[DRY RUN] Would place: BACK {row.get('horse_name')} "
                    f"@ {tick_price} for {stake:.2f} GBP "
                    f"(edge={row.get('edge_pct', 0):.1f}%)"
                )
                result.at[idx, "bet_status"] = "DRY_RUN"
                result.at[idx, "placed"] = False
                self.audit.log_bet(
                    row, tick_price, stake, bet_id=None, status="DRY_RUN"
                )
                placed_count += 1
                continue

            # Place the actual order
            try:
                response = self.client.place_orders(
                    market_id=str(market_id),
                    instructions=[instruction],
                )
                report = response["instruction_reports"][0]

                result.at[idx, "bet_id"] = report["bet_id"]
                result.at[idx, "bet_status"] = report["status"]
                result.at[idx, "size_matched"] = report.get("size_matched", 0)
                result.at[idx, "placed"] = report["status"] == "SUCCESS"

                self.audit.log_bet(
                    row,
                    tick_price,
                    stake,
                    bet_id=report["bet_id"],
                    status=report["status"],
                )

                log.info(
                    f"Placed: BACK {row.get('horse_name')} "
                    f"@ {tick_price} for {stake:.2f} GBP -> "
                    f"{report['status']} (bet_id={report['bet_id']})"
                )
                placed_count += 1

            except Exception as exc:
                log.error(
                    f"Failed to place bet on {row.get('horse_name')}: {exc}"
                )
                result.at[idx, "bet_status"] = "ERROR"
                self.audit.log_bet(
                    row, tick_price, stake, bet_id=None, status=f"ERROR: {exc}"
                )

        mode = "DRY RUN" if self.config.dry_run else "LIVE"
        log.info(
            f"Bet placement complete [{mode}]: "
            f"{placed_count}/{len(result)} bets processed"
        )

        return result
