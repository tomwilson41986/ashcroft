"""Market discovery helpers."""

import logging
from datetime import date, datetime, timezone

from ultra_betting.betfair.client import get_client
from ultra_betting.config import load_guardrails

log = logging.getLogger(__name__)


def list_todays_markets(
    target_date: date | None = None,
    country_codes: list[str] | None = None,
) -> list[dict]:
    """List today's horse racing WIN markets, filtered by allowed countries."""
    if target_date is None:
        target_date = date.today()

    if country_codes is None:
        guardrails = load_guardrails()
        country_codes = guardrails.get("allowed_countries", ["GB", "IE"])

    client = get_client()
    return client.list_horse_racing_markets(target_date, country_codes=country_codes)


def get_market_status(market_ids: list[str]) -> dict[str, str]:
    """Get status for a batch of markets. Returns {market_id: status}."""
    client = get_client()
    result = {}

    for i in range(0, len(market_ids), 40):
        batch = market_ids[i:i + 40]
        try:
            books = client._api_call("listMarketBook", {
                "marketIds": batch,
                "priceProjection": {"priceData": ["SP_AVAILABLE"]},
            })
            for book in books:
                result[book["marketId"]] = book.get("status", "UNKNOWN")
        except Exception as e:
            log.warning(f"Failed to get status for batch: {e}")

    return result


def is_market_open(market_id: str) -> bool:
    """Check if a specific market is open."""
    statuses = get_market_status([market_id])
    return statuses.get(market_id) == "OPEN"


def minutes_to_start(market_start_time: str) -> float:
    """Calculate minutes until market start from an ISO timestamp."""
    try:
        start = datetime.fromisoformat(market_start_time.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (start - now).total_seconds() / 60
    except (ValueError, AttributeError):
        return -1
