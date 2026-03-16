"""Settled bet retrieval from Betfair."""

import logging
from datetime import date, datetime, timedelta, timezone

from ultra_betting.betfair.client import get_client

log = logging.getLogger(__name__)

LIST_CLEARED_ORDERS_URL = "listClearedOrders"


def get_settled_bets(
    settled_date: date | None = None,
    bet_ids: list[str] | None = None,
) -> list[dict]:
    """Fetch settled/cleared bets from Betfair.

    Args:
        settled_date: Date to fetch settlements for. Defaults to today.
        bet_ids: Optional list of specific bet IDs to query.

    Returns:
        List of cleared order records.
    """
    client = get_client()

    if settled_date is None:
        settled_date = date.today()

    from_dt = datetime(
        settled_date.year, settled_date.month, settled_date.day,
        tzinfo=timezone.utc,
    )
    to_dt = from_dt + timedelta(days=1)

    params = {
        "betStatus": "SETTLED",
        "settledDateRange": {
            "from": from_dt.isoformat(),
            "to": to_dt.isoformat(),
        },
        "recordCount": 1000,
    }

    if bet_ids:
        params["betIds"] = bet_ids

    try:
        result = client._api_call(LIST_CLEARED_ORDERS_URL, params)
        orders = result.get("clearedOrders", [])
        log.info(f"Retrieved {len(orders)} settled bets for {settled_date}")
        return orders
    except Exception as e:
        log.error(f"Failed to fetch settled bets: {e}")
        return []


def match_settlements_to_executions(
    settlements: list[dict],
    executions_bet_ids: set[str],
) -> list[dict]:
    """Filter settlements to only those matching our execution bet IDs."""
    matched = [s for s in settlements if str(s.get("betId", "")) in executions_bet_ids]
    log.info(f"Matched {len(matched)} settlements to our executions")
    return matched
