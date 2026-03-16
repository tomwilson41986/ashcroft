"""Authentication and account tools."""

import json

from mcp.server import Server
from mcp.types import Tool, TextContent


def register(server: Server):
    @server.tool()
    async def betfair_login() -> list[TextContent]:
        """Authenticate with Betfair Exchange. Auto-called on first use of any Betfair tool."""
        from ultra_betting.betfair.auth import ensure_session
        ensure_session()
        return [TextContent(type="text", text="Successfully authenticated with Betfair")]

    @server.tool()
    async def betfair_account_balance() -> list[TextContent]:
        """Check Betfair account funds and available balance."""
        from ultra_betting.betfair.client import get_client
        client = get_client()
        # Use the account API
        import requests
        resp = client._session.get(
            "https://api.betfair.com/exchange/account/rest/v1.0/getAccountFunds/",
            headers={
                "X-Application": client.app_key,
                "X-Authentication": client.session_token,
                "Accept": "application/json",
            },
        )
        data = resp.json()
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
