"""Betfair authentication helpers with session keep-alive."""

import logging
import time

from ultra_betting.betfair.client import get_client, reset_client, BetfairAPIError

log = logging.getLogger(__name__)


def ensure_session(max_retries: int = 3) -> None:
    """Ensure we have a valid Betfair session, refreshing if needed."""
    client = get_client(login=False)

    for attempt in range(max_retries):
        try:
            if client.session_token is None:
                client.login()
                return

            # Test the session with a lightweight call
            client._api_call("listEventTypes", {
                "filter": {"eventTypeIds": ["7"]}
            })
            return

        except BetfairAPIError as e:
            if "INVALID_SESSION" in str(e) or "NO_SESSION" in str(e):
                log.warning(f"Session expired, re-authenticating (attempt {attempt + 1})")
                client.session_token = None
                try:
                    client.login()
                    return
                except BetfairAPIError:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
            raise

    raise BetfairAPIError("Failed to establish Betfair session after retries")
