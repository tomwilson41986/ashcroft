"""MCP stdio entry point.

Usage:
    python -m mcp_server

Register with Claude Code:
    claude mcp add betfair --command "python" --args "-m mcp_server"
"""

import sys
from pathlib import Path

# Ensure project root and src are importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))

from mcp_server.server import create_server


def main():
    server = create_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
