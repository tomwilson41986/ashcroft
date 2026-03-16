"""Account management and utility tools."""

import json

from mcp.server import Server
from mcp.types import TextContent


def register(server: Server):
    @server.tool()
    async def betfair_cancel_staged(pending_id: str) -> list[TextContent]:
        """Cancel a staged (unconfirmed) bet.

        Args:
            pending_id: The pending bet ID from betfair_place_bet.
        """
        from mcp_server.tools.betting import _pending_bets
        if pending_id in _pending_bets:
            _pending_bets.pop(pending_id)
            return [TextContent(type="text", text=f"Staged bet {pending_id} cancelled")]
        return [TextContent(type="text", text=f"No staged bet with ID {pending_id}")]
