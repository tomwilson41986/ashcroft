"""Price fetching and formatting."""

import logging
from datetime import date

from ultra_betting.betfair.client import get_client
from ultra_betting.data.schemas import Prediction

log = logging.getLogger(__name__)


class MarketPrices:
    """Live prices for all runners in a single market."""

    def __init__(self, market_id: str, runners: list[dict], status: str = "UNKNOWN",
                 total_matched: float = 0):
        self.market_id = market_id
        self.status = status
        self.total_matched = total_matched
        # {selection_id: runner_data}
        self._runners = {r["selection_id"]: r for r in runners}

    def get_best_back(self, selection_id: int) -> float | None:
        r = self._runners.get(selection_id)
        return r.get("best_back_price") if r else None

    def get_best_lay(self, selection_id: int) -> float | None:
        r = self._runners.get(selection_id)
        return r.get("best_lay_price") if r else None

    def get_last_traded(self, selection_id: int) -> float | None:
        r = self._runners.get(selection_id)
        return r.get("last_traded_price") if r else None


def fetch_market_prices(market_id: str) -> MarketPrices:
    """Fetch live prices for a market."""
    client = get_client()
    book = client.get_market_odds(market_id)

    runners = []
    for runner in book.get("runners", []):
        back = runner.get("ex", {}).get("availableToBack", [])
        lay = runner.get("ex", {}).get("availableToLay", [])
        best_back = back[0] if back else {}
        best_lay = lay[0] if lay else {}

        runners.append({
            "selection_id": runner["selectionId"],
            "runner_status": runner.get("status", "ACTIVE"),
            "best_back_price": best_back.get("price"),
            "best_back_size": best_back.get("size"),
            "best_lay_price": best_lay.get("price"),
            "best_lay_size": best_lay.get("size"),
            "last_traded_price": runner.get("lastPriceTraded"),
            "sp_actual": runner.get("sp", {}).get("actualSP"),
        })

    return MarketPrices(
        market_id=market_id,
        runners=runners,
        status=book.get("status", "UNKNOWN"),
        total_matched=book.get("totalMatched", 0),
    )


def fetch_all_prices(target_date: date | None = None) -> dict[str, MarketPrices]:
    """Fetch prices for all markets on a date. Returns {market_id: MarketPrices}."""
    client = get_client()
    if target_date is None:
        target_date = date.today()

    odds_data = client.get_live_odds_for_date(target_date)
    # Group by market_id
    by_market: dict[str, list[dict]] = {}
    for row in odds_data:
        mid = row["market_id"]
        if mid not in by_market:
            by_market[mid] = []
        by_market[mid].append({
            "selection_id": row["selection_id"],
            "best_back_price": row.get("best_back_price"),
            "best_back_size": row.get("best_back_size"),
            "best_lay_price": row.get("best_lay_price"),
            "best_lay_size": row.get("best_lay_size"),
            "last_traded_price": row.get("last_traded_price"),
        })

    result = {}
    for mid, runners in by_market.items():
        status = next(
            (r.get("market_status", "UNKNOWN") for r in odds_data if r["market_id"] == mid),
            "UNKNOWN"
        )
        total_matched = next(
            (r.get("total_matched", 0) for r in odds_data if r["market_id"] == mid),
            0
        )
        result[mid] = MarketPrices(mid, runners, status, total_matched)

    return result
