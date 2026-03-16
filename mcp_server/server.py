"""MCP Server — tool registration for Claude Code."""

import logging

from mcp.server import Server

log = logging.getLogger(__name__)


def create_server() -> Server:
    """Create and configure the MCP server with all tools."""
    server = Server("ultra-betting")

    # Register tools from each module
    from mcp_server.tools.auth import register as register_auth
    from mcp_server.tools.markets import register as register_markets
    from mcp_server.tools.betting import register as register_betting
    from mcp_server.tools.management import register as register_management
    from mcp_server.tools.pipeline import register as register_pipeline

    register_auth(server)
    register_markets(server)
    register_betting(server)
    register_management(server)
    register_pipeline(server)

    log.info("Ultra Betting MCP server initialized with all tools")
    return server
