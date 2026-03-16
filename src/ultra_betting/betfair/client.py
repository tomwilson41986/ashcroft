"""Betfair client wrapper — re-exports the existing BetfairClient with session management."""

import logging
import sys
from pathlib import Path

# Add project root to path so we can import the existing betfair_client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from betfair_client import BetfairClient, BetfairAPIError  # noqa: E402

log = logging.getLogger(__name__)

_instance: BetfairClient | None = None


def get_client(login: bool = True) -> BetfairClient:
    """Get a singleton BetfairClient, logging in if needed."""
    global _instance
    if _instance is None:
        _instance = BetfairClient()
    if login and _instance.session_token is None:
        _instance.login()
    return _instance


def reset_client() -> None:
    """Reset the singleton (e.g. after session expiry)."""
    global _instance
    if _instance is not None:
        _instance.logout()
    _instance = None


__all__ = ["get_client", "reset_client", "BetfairClient", "BetfairAPIError"]
