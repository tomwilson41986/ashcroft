"""Market discovery and pricing tools."""

import json
from datetime import date

from mcp.server import Server
from mcp.types import Tool, TextContent


def register(server: Server):
    @server.tool()
    async def betfair_list_events(
        target_date: str | None = None,
    ) -> list[TextContent]:
        """List today's horse racing meetings/events on Betfair.

        Args:
            target_date: Optional date (YYYY-MM-DD). Defaults to today.
        """
        from ultra_betting.betfair.auth import ensure_session
        from ultra_betting.betfair.markets import list_todays_markets

        ensure_session()
        dt = date.fromisoformat(target_date) if target_date else date.today()
        markets = list_todays_markets(dt)

        # Group by venue
        venues = {}
        for m in markets:
            venue = m.get("event", {}).get("venue", "Unknown")
            if venue not in venues:
                venues[venue] = []
            venues[venue].append({
                "market_id": m["marketId"],
                "start_time": m.get("marketStartTime", ""),
                "runners": len(m.get("runners", [])),
            })

        return [TextContent(type="text", text=json.dumps(venues, indent=2))]

    @server.tool()
    async def betfair_list_markets(
        venue: str,
        target_date: str | None = None,
    ) -> list[TextContent]:
        """List markets at a specific venue.

        Args:
            venue: Venue name (e.g. "Cheltenham").
            target_date: Optional date (YYYY-MM-DD).
        """
        from ultra_betting.betfair.auth import ensure_session
        from ultra_betting.betfair.markets import list_todays_markets

        ensure_session()
        dt = date.fromisoformat(target_date) if target_date else date.today()
        markets = list_todays_markets(dt)

        venue_lower = venue.lower()
        filtered = [
            {
                "market_id": m["marketId"],
                "name": m.get("marketName", ""),
                "start_time": m.get("marketStartTime", ""),
                "runners": [
                    {"name": r["runnerName"], "selection_id": r["selectionId"]}
                    for r in m.get("runners", [])
                ],
            }
            for m in markets
            if venue_lower in m.get("event", {}).get("venue", "").lower()
        ]

        return [TextContent(type="text", text=json.dumps(filtered, indent=2))]

    @server.tool()
    async def betfair_list_runners(
        market_id: str,
    ) -> list[TextContent]:
        """Get live exchange prices for all runners in a market.

        Args:
            market_id: Betfair market ID.
        """
        from ultra_betting.betfair.auth import ensure_session
        from ultra_betting.betfair.pricing import fetch_market_prices

        ensure_session()
        prices = fetch_market_prices(market_id)

        runners = []
        for sel_id, data in prices._runners.items():
            runners.append({
                "selection_id": sel_id,
                "status": data.get("runner_status", ""),
                "best_back": data.get("best_back_price"),
                "best_back_size": data.get("best_back_size"),
                "best_lay": data.get("best_lay_price"),
                "best_lay_size": data.get("best_lay_size"),
                "last_traded": data.get("last_traded_price"),
            })

        result = {
            "market_id": market_id,
            "status": prices.status,
            "total_matched": prices.total_matched,
            "runners": runners,
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]
