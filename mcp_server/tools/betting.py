"""Bet placement, confirmation, and cancellation tools."""

import json

from mcp.server import Server
from mcp.types import Tool, TextContent

# Pending bets awaiting confirmation (confirmation flow)
_pending_bets: dict[str, dict] = {}


def register(server: Server):
    @server.tool()
    async def betfair_place_bet(
        market_id: str,
        selection_id: int,
        side: str,
        stake: float,
        price: float,
    ) -> list[TextContent]:
        """Stage a manual bet for confirmation. Call betfair_confirm_bet to execute.

        Args:
            market_id: Betfair market ID.
            selection_id: Runner selection ID.
            side: "BACK" or "LAY".
            stake: Stake in GBP.
            price: Odds to request.
        """
        import uuid
        pending_id = str(uuid.uuid4())[:8]

        _pending_bets[pending_id] = {
            "market_id": market_id,
            "selection_id": selection_id,
            "side": side.upper(),
            "stake": stake,
            "price": price,
        }

        liability = stake * (price - 1) if side.upper() == "LAY" else stake
        msg = (
            f"Bet staged (ID: {pending_id}):\n"
            f"  {side.upper()} £{stake:.2f} @ {price:.2f}\n"
            f"  Market: {market_id}, Selection: {selection_id}\n"
            f"  Liability: £{liability:.2f}\n\n"
            f"Call betfair_confirm_bet('{pending_id}') to execute, "
            f"or betfair_cancel_staged('{pending_id}') to cancel."
        )
        return [TextContent(type="text", text=msg)]

    @server.tool()
    async def betfair_confirm_bet(pending_id: str) -> list[TextContent]:
        """Confirm and execute a previously staged bet.

        Args:
            pending_id: The pending bet ID from betfair_place_bet.
        """
        if pending_id not in _pending_bets:
            return [TextContent(type="text", text=f"No pending bet with ID {pending_id}")]

        bet = _pending_bets.pop(pending_id)

        from ultra_betting.betfair.auth import ensure_session
        from ultra_betting.betfair.execution import place_bet
        from ultra_betting.data.schemas import BetInstruction

        ensure_session()

        instruction = BetInstruction(
            prediction_id=f"manual_{pending_id}",
            market_id=bet["market_id"],
            selection_id=bet["selection_id"],
            runner_name="Manual bet",
            side=bet["side"],
            stake=bet["stake"],
            price=bet["price"],
            reasoning="Manual bet via MCP",
        )

        execution = place_bet(instruction, dry_run=False)
        return [TextContent(type="text", text=json.dumps(execution.model_dump(), indent=2, default=str))]

    @server.tool()
    async def betfair_cancel_bet(
        market_id: str,
        bet_id: str,
    ) -> list[TextContent]:
        """Cancel an unmatched bet on Betfair.

        Args:
            market_id: Betfair market ID.
            bet_id: The bet ID to cancel.
        """
        from ultra_betting.betfair.auth import ensure_session
        from ultra_betting.betfair.execution import cancel_bet

        ensure_session()
        result = cancel_bet(market_id, bet_id)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    @server.tool()
    async def betfair_list_current_bets(
        market_id: str | None = None,
    ) -> list[TextContent]:
        """List current open/unmatched bets.

        Args:
            market_id: Optional market ID filter.
        """
        from ultra_betting.betfair.auth import ensure_session
        from ultra_betting.betfair.execution import list_current_bets

        ensure_session()
        bets = list_current_bets(market_id)
        return [TextContent(type="text", text=json.dumps(bets, indent=2, default=str))]

    @server.tool()
    async def betfair_list_settled_bets(
        target_date: str | None = None,
    ) -> list[TextContent]:
        """List settled bets for a date.

        Args:
            target_date: Date (YYYY-MM-DD). Defaults to today.
        """
        from datetime import date as dt
        from ultra_betting.betfair.auth import ensure_session
        from ultra_betting.betfair.settlement import get_settled_bets

        ensure_session()
        d = dt.fromisoformat(target_date) if target_date else dt.today()
        bets = get_settled_bets(d)
        return [TextContent(type="text", text=json.dumps(bets, indent=2, default=str))]
