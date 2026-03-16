"""Bet placement and cancellation via Betfair API."""

import logging

from ultra_betting.audit import audit_event
from ultra_betting.betfair.client import get_client, BetfairAPIError
from ultra_betting.data.schemas import BetInstruction, Execution

log = logging.getLogger(__name__)

# Betfair API endpoint for placing orders
PLACE_ORDERS_URL = "placeOrders"
CANCEL_ORDERS_URL = "cancelOrders"
LIST_CURRENT_ORDERS_URL = "listCurrentOrders"


def place_bet(instruction: BetInstruction, dry_run: bool = False) -> Execution:
    """Place a bet on Betfair or simulate it in dry-run mode.

    Args:
        instruction: The bet instruction from the rules engine.
        dry_run: If True, log the bet but don't actually place it.

    Returns:
        Execution record with the result.
    """
    execution = Execution(
        date=instruction.prediction_id.split("_")[0] if "_" in instruction.prediction_id else "",
        venue="",
        race_time="",
        market_id=instruction.market_id,
        selection_id=instruction.selection_id,
        runner_name=instruction.runner_name,
        side=instruction.side,
        stake=instruction.stake,
        price_requested=instruction.price,
        dry_run=dry_run,
        reasoning=instruction.reasoning,
        prediction_id=instruction.prediction_id,
    )

    if dry_run:
        execution.status = "DRY_RUN"
        execution.price_matched = instruction.price
        execution.size_matched = instruction.stake
        audit_event("bet_dry_run", **instruction.model_dump())
        log.info(
            f"[DRY RUN] {instruction.side} £{instruction.stake:.2f} "
            f"@ {instruction.price:.2f} on {instruction.runner_name}"
        )
        return execution

    client = get_client()

    order_type = "LIMIT"
    persistence_type = instruction.persistence or "LAPSE"

    params = {
        "marketId": instruction.market_id,
        "instructions": [{
            "selectionId": instruction.selection_id,
            "side": instruction.side,
            "orderType": order_type,
            "limitOrder": {
                "size": round(instruction.stake, 2),
                "price": instruction.price,
                "persistenceType": persistence_type,
            },
        }],
    }

    try:
        result = client._api_call(PLACE_ORDERS_URL, params)
        audit_event("bet_placed", request=params, response=result)

        if result.get("status") == "SUCCESS":
            report = result["instructionReports"][0]
            execution.bet_id = str(report.get("betId", ""))
            execution.status = report.get("status", "MATCHED")
            execution.price_matched = report.get("averagePriceMatched", instruction.price)
            execution.size_matched = report.get("sizeMatched", 0)
            log.info(
                f"BET PLACED: {instruction.side} £{instruction.stake:.2f} "
                f"@ {execution.price_matched:.2f} on {instruction.runner_name} "
                f"(bet_id={execution.bet_id})"
            )
        else:
            error_code = result.get("errorCode", "UNKNOWN")
            execution.status = "FAILED"
            execution.reasoning += f" | API error: {error_code}"
            log.error(f"Bet placement failed: {error_code}")

    except BetfairAPIError as e:
        execution.status = "FAILED"
        execution.reasoning += f" | Exception: {e}"
        log.error(f"Bet placement exception: {e}")

    return execution


def cancel_bet(market_id: str, bet_id: str) -> dict:
    """Cancel an unmatched bet."""
    client = get_client()
    params = {
        "marketId": market_id,
        "instructions": [{"betId": bet_id}],
    }
    result = client._api_call(CANCEL_ORDERS_URL, params)
    audit_event("bet_cancelled", market_id=market_id, bet_id=bet_id, result=result)
    return result


def list_current_bets(market_id: str | None = None) -> list[dict]:
    """List current (unmatched/partially matched) bets."""
    client = get_client()
    params = {"orderProjection": "EXECUTABLE"}
    if market_id:
        params["marketIds"] = [market_id]
    return client._api_call(LIST_CURRENT_ORDERS_URL, params).get("currentOrders", [])
