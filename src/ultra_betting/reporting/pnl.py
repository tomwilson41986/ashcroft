"""P&L calculation and cumulative tracking."""

import logging

from ultra_betting.data.schemas import DailyPnL, Execution, Settlement
from ultra_betting.data import s3

log = logging.getLogger(__name__)

BETFAIR_COMMISSION_RATE = 0.05  # 5% standard commission on winnings


def calculate_settlement_pnl(execution: Execution, result: str) -> tuple[float, float, float]:
    """Calculate P&L for a single settled bet.

    Args:
        execution: The original execution record.
        result: "WON", "LOST", or "VOID".

    Returns:
        (gross_pnl, commission, net_pnl)
    """
    if result == "VOID":
        return 0.0, 0.0, 0.0

    if execution.side == "BACK":
        if result == "WON":
            gross = execution.size_matched * (execution.price_matched - 1)
            commission = gross * BETFAIR_COMMISSION_RATE
            return gross, -commission, gross - commission
        else:
            return -execution.size_matched, 0.0, -execution.size_matched
    else:
        # LAY
        if result == "WON":
            # Lay bet wins when the horse loses
            gross = execution.size_matched
            commission = gross * BETFAIR_COMMISSION_RATE
            return gross, -commission, gross - commission
        else:
            # Lay bet loses when the horse wins
            liability = execution.size_matched * (execution.price_matched - 1)
            return -liability, 0.0, -liability


def calculate_daily_pnl(
    settlements: list[Settlement],
    bank_start: float = 1000.0,
) -> DailyPnL:
    """Calculate daily P&L summary from settlements."""
    if not settlements:
        return DailyPnL(date="", bank_start=bank_start, bank_end=bank_start)

    target_date = settlements[0].date
    bets_won = sum(1 for s in settlements if s.result == "WON")
    bets_lost = sum(1 for s in settlements if s.result == "LOST")
    gross_pnl = sum(s.pnl for s in settlements)
    commission = sum(s.commission for s in settlements)
    net_pnl = sum(s.net_pnl for s in settlements)
    total_staked = sum(s.stake for s in settlements)
    roi = (net_pnl / total_staked * 100) if total_staked > 0 else 0.0

    return DailyPnL(
        date=target_date,
        bets_placed=len(settlements),
        bets_won=bets_won,
        bets_lost=bets_lost,
        gross_pnl=round(gross_pnl, 2),
        commission=round(commission, 2),
        net_pnl=round(net_pnl, 2),
        roi_percent=round(roi, 2),
        bank_start=round(bank_start, 2),
        bank_end=round(bank_start + net_pnl, 2),
    )


def update_cumulative_pnl(daily: DailyPnL) -> list[dict]:
    """Append today's P&L to the cumulative tracker in S3."""
    cumulative = s3.read_json("cumulative_pnl.json")
    if not isinstance(cumulative, list):
        cumulative = []

    # Don't duplicate
    existing_dates = {entry["date"] for entry in cumulative}
    if daily.date not in existing_dates:
        cumulative.append(daily.model_dump())
        s3.write_json("cumulative_pnl.json", cumulative)
        log.info(f"Updated cumulative P&L: {len(cumulative)} days tracked")

    return cumulative


def get_current_bank(starting_bank: float = 1000.0) -> float:
    """Get the current bank balance from cumulative P&L."""
    cumulative = s3.read_json("cumulative_pnl.json")
    if not isinstance(cumulative, list) or not cumulative:
        return starting_bank

    # Last entry's bank_end
    return cumulative[-1].get("bank_end", starting_bank)
