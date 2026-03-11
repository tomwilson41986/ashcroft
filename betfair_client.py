"""
Betfair Exchange API Client.

Provides authentication, market discovery, live odds retrieval, and actual
BSP (Betfair Starting Price) data for horse racing markets.

Betfair API docs: https://docs.developer.betfair.com/

Usage:
    from betfair_client import BetfairClient

    client = BetfairClient()
    client.login()

    # Get live markets for today
    markets = client.list_horse_racing_markets("2026-03-11")

    # Get live odds for a market
    odds = client.get_market_odds(market_id)

    # Get actual BSPs for a settled market
    bsps = client.get_actual_bsps(market_id)

Environment variables:
    BETFAIR_USERNAME   - Betfair account username
    BETFAIR_PASSWORD   - Betfair account password
    BETFAIR_APP_KEY    - Betfair API application key
    BETFAIR_CERT_FILE  - (Optional) Path to client certificate .crt file
    BETFAIR_KEY_FILE   - (Optional) Path to client certificate .key file
"""

import logging
import os
import re
from datetime import date, datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# Betfair API endpoints
LOGIN_URL = "https://identitysso.betfair.com/api/login"
CERT_LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"
BETTING_URL = "https://api.betfair.com/exchange/betting/rest/v1.0/"

# Horse Racing event type ID on Betfair
HORSE_RACING_EVENT_TYPE_ID = "7"

# Market types we care about
WIN_MARKET_TYPE = "WIN"
PLACE_MARKET_TYPE = "PLACE"


class BetfairAPIError(Exception):
    """Raised when the Betfair API returns an error."""
    pass


class BetfairClient:
    """Client for the Betfair Exchange API.

    Supports both interactive (username/password) and certificate-based login.
    Certificate login is preferred for automated/production use.
    """

    def __init__(self):
        self.session_token = None
        self.app_key = os.getenv("BETFAIR_APP_KEY", "")
        self.username = os.getenv("BETFAIR_USERNAME", "")
        self.password = os.getenv("BETFAIR_PASSWORD", "")
        self.cert_file = os.getenv("BETFAIR_CERT_FILE", "")
        self.key_file = os.getenv("BETFAIR_KEY_FILE", "")
        self._session = requests.Session()

    def login(self) -> str:
        """Authenticate with Betfair and obtain a session token.

        Tries certificate-based login first, then falls back to
        interactive (username/password) login.

        Returns:
            Session token string.
        """
        if not self.username or not self.password:
            raise BetfairAPIError(
                "BETFAIR_USERNAME and BETFAIR_PASSWORD must be set"
            )
        if not self.app_key:
            raise BetfairAPIError("BETFAIR_APP_KEY must be set")

        # Try certificate login if cert files are provided
        if self.cert_file and self.key_file:
            if os.path.exists(self.cert_file) and os.path.exists(self.key_file):
                return self._cert_login()

        return self._interactive_login()

    def _cert_login(self) -> str:
        """Login using client SSL certificate (preferred for automation)."""
        log.info("Logging in to Betfair via certificate...")
        resp = self._session.post(
            CERT_LOGIN_URL,
            data={"username": self.username, "password": self.password},
            cert=(self.cert_file, self.key_file),
            headers={"X-Application": self.app_key},
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("loginStatus") != "SUCCESS":
            raise BetfairAPIError(
                f"Betfair cert login failed: {data.get('loginStatus')}"
            )

        self.session_token = data["sessionToken"]
        log.info("Betfair certificate login successful")
        return self.session_token

    def _interactive_login(self) -> str:
        """Login using username/password (simpler setup)."""
        log.info("Logging in to Betfair via username/password...")
        resp = self._session.post(
            LOGIN_URL,
            data={"username": self.username, "password": self.password},
            headers={
                "X-Application": self.app_key,
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "SUCCESS":
            raise BetfairAPIError(
                f"Betfair login failed: {data.get('error', data.get('status'))}"
            )

        self.session_token = data["token"]
        log.info("Betfair interactive login successful")
        return self.session_token

    def _api_call(self, method: str, params: dict | None = None) -> dict:
        """Make a Betfair Exchange API call.

        Args:
            method: API method name (e.g. 'listMarketCatalogue').
            params: Request parameters (filter, etc).

        Returns:
            Parsed JSON response.
        """
        if not self.session_token:
            raise BetfairAPIError("Not logged in. Call login() first.")

        url = f"{BETTING_URL}{method}/"
        headers = {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        resp = self._session.post(url, json=params or {}, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # Check for API-level errors
        if isinstance(data, dict) and "faultcode" in data:
            raise BetfairAPIError(
                f"Betfair API error: {data.get('faultstring', data)}"
            )

        return data

    # ------------------------------------------------------------------
    # Market Discovery
    # ------------------------------------------------------------------

    def list_horse_racing_markets(
        self,
        target_date: str | date,
        market_type: str = WIN_MARKET_TYPE,
        country_codes: list[str] | None = None,
    ) -> list[dict]:
        """List horse racing WIN markets for a given date.

        Args:
            target_date: Date string (YYYY-MM-DD) or date object.
            market_type: Market type filter (default: WIN).
            country_codes: Country code filter (e.g. ['GB', 'IE']).
                          Default: GB + IE.

        Returns:
            List of market catalogue entries, each containing:
            - marketId, marketName, marketStartTime
            - event (name, venue info)
            - runners (list of {selectionId, runnerName, sortPriority})
        """
        if isinstance(target_date, str):
            target_date = date.fromisoformat(target_date)

        if country_codes is None:
            country_codes = ["GB", "IE"]

        # Time range: full day in UTC
        from_time = datetime(
            target_date.year, target_date.month, target_date.day,
            tzinfo=timezone.utc,
        )
        to_time = from_time + timedelta(days=1)

        params = {
            "filter": {
                "eventTypeIds": [HORSE_RACING_EVENT_TYPE_ID],
                "marketTypeCodes": [market_type],
                "marketCountries": country_codes,
                "marketStartTime": {
                    "from": from_time.isoformat(),
                    "to": to_time.isoformat(),
                },
            },
            "marketProjection": [
                "EVENT",
                "MARKET_START_TIME",
                "RUNNER_DESCRIPTION",
                "MARKET_DESCRIPTION",
            ],
            "maxResults": "1000",
            "sort": "FIRST_TO_START",
        }

        markets = self._api_call("listMarketCatalogue", params)
        log.info(
            f"Found {len(markets)} {market_type} markets for "
            f"{target_date} ({', '.join(country_codes)})"
        )
        return markets

    # ------------------------------------------------------------------
    # Live Odds (pre-race and in-play)
    # ------------------------------------------------------------------

    def get_market_odds(
        self,
        market_id: str,
        price_depth: int = 3,
    ) -> dict:
        """Get current exchange odds for a market.

        Returns runner-level data including best back/lay prices,
        traded volume, and SP (starting price) data if available.

        Args:
            market_id: Betfair market ID.
            price_depth: Number of price levels to return (default 3).

        Returns:
            Market book dict with runners and their prices.
        """
        params = {
            "marketIds": [market_id],
            "priceProjection": {
                "priceData": [
                    "EX_BEST_OFFERS",
                    "EX_TRADED",
                    "SP_AVAILABLE",
                    "SP_TRADED",
                ],
                "exBestOffersOverrides": {
                    "bestPricesDepth": price_depth,
                },
            },
        }

        result = self._api_call("listMarketBook", params)
        if not result:
            return {}
        return result[0]

    def get_live_odds_for_date(
        self,
        target_date: str | date,
        country_codes: list[str] | None = None,
    ) -> list[dict]:
        """Get live exchange odds for all horse racing markets on a date.

        Returns a list of dicts, one per runner, with:
        - market_id, market_name, market_start_time, venue
        - selection_id, runner_name
        - best_back_price, best_back_size
        - best_lay_price, best_lay_size
        - sp_near_price, sp_far_price (SP projected price)
        - last_traded_price, total_matched
        - market_status (OPEN, SUSPENDED, CLOSED)

        Args:
            target_date: Date to query.
            country_codes: Country filter (default: GB, IE).

        Returns:
            List of runner-level dicts with live odds data.
        """
        markets = self.list_horse_racing_markets(
            target_date, country_codes=country_codes
        )

        all_runners = []

        for market in markets:
            market_id = market["marketId"]
            market_name = market.get("marketName", "")
            event = market.get("event", {})
            venue = event.get("venue", event.get("name", ""))
            start_time = market.get("marketStartTime", "")

            # Build runner name lookup from catalogue
            runner_names = {}
            for r in market.get("runners", []):
                runner_names[r["selectionId"]] = r["runnerName"]

            # Get market book with prices
            try:
                book = self.get_market_odds(market_id)
            except Exception as e:
                log.warning(f"Failed to get odds for {market_id}: {e}")
                continue

            market_status = book.get("status", "UNKNOWN")
            total_matched = book.get("totalMatched", 0)

            for runner in book.get("runners", []):
                sel_id = runner["selectionId"]
                status = runner.get("status", "ACTIVE")

                # Best back/lay prices
                back_prices = runner.get("ex", {}).get("availableToBack", [])
                lay_prices = runner.get("ex", {}).get("availableToLay", [])

                best_back = back_prices[0] if back_prices else {}
                best_lay = lay_prices[0] if lay_prices else {}

                # SP data
                sp_data = runner.get("sp", {})

                row = {
                    "market_id": market_id,
                    "market_name": market_name,
                    "venue": _normalise_venue(venue),
                    "market_start_time": start_time,
                    "market_status": market_status,
                    "selection_id": sel_id,
                    "runner_name": runner_names.get(sel_id, f"Runner {sel_id}"),
                    "runner_status": status,
                    "best_back_price": best_back.get("price"),
                    "best_back_size": best_back.get("size"),
                    "best_lay_price": best_lay.get("price"),
                    "best_lay_size": best_lay.get("size"),
                    "sp_near_price": sp_data.get("nearPrice"),
                    "sp_far_price": sp_data.get("farPrice"),
                    "sp_actual_price": sp_data.get("actualSP"),
                    "last_traded_price": runner.get("lastPriceTraded"),
                    "total_matched": total_matched,
                }
                all_runners.append(row)

        log.info(
            f"Retrieved live odds for {len(all_runners)} runners "
            f"across {len(markets)} markets"
        )
        return all_runners

    # ------------------------------------------------------------------
    # Actual BSP Data (settled markets)
    # ------------------------------------------------------------------

    def get_actual_bsps(self, market_id: str) -> list[dict]:
        """Get actual BSP (Betfair Starting Price) for a settled market.

        The actualSP field is only populated after the market has been
        settled (race completed and results confirmed).

        Args:
            market_id: Betfair market ID.

        Returns:
            List of dicts with {selection_id, runner_name, actual_bsp}.
        """
        params = {
            "marketIds": [market_id],
            "priceProjection": {
                "priceData": ["SP_TRADED"],
            },
        }

        result = self._api_call("listMarketBook", params)
        if not result:
            return []

        book = result[0]
        bsps = []
        for runner in book.get("runners", []):
            sp_data = runner.get("sp", {})
            actual_sp = sp_data.get("actualSP")
            bsps.append({
                "selection_id": runner["selectionId"],
                "actual_bsp": actual_sp,
                "runner_status": runner.get("status", ""),
            })

        return bsps

    def get_bsps_for_date(
        self,
        target_date: str | date,
        country_codes: list[str] | None = None,
    ) -> list[dict]:
        """Get actual BSPs for all settled horse racing markets on a date.

        Returns a list of dicts, one per runner, with:
        - market_id, venue, market_start_time
        - selection_id, runner_name
        - actual_bsp (the real Betfair Starting Price)
        - runner_status (WINNER, LOSER, REMOVED)

        Args:
            target_date: Date to query (should be in the past).
            country_codes: Country filter (default: GB, IE).

        Returns:
            List of runner-level dicts with actual BSP data.
        """
        markets = self.list_horse_racing_markets(
            target_date, country_codes=country_codes
        )

        all_runners = []

        for market in markets:
            market_id = market["marketId"]
            event = market.get("event", {})
            venue = event.get("venue", event.get("name", ""))
            start_time = market.get("marketStartTime", "")

            # Runner name lookup
            runner_names = {}
            for r in market.get("runners", []):
                runner_names[r["selectionId"]] = r["runnerName"]

            try:
                bsps = self.get_actual_bsps(market_id)
            except Exception as e:
                log.warning(
                    f"Failed to get BSPs for {market_id} ({venue}): {e}"
                )
                continue

            for bsp in bsps:
                sel_id = bsp["selection_id"]
                row = {
                    "market_id": market_id,
                    "venue": _normalise_venue(venue),
                    "market_start_time": start_time,
                    "selection_id": sel_id,
                    "runner_name": runner_names.get(sel_id, f"Runner {sel_id}"),
                    "actual_bsp": bsp["actual_bsp"],
                    "runner_status": bsp["runner_status"],
                }
                all_runners.append(row)

        settled = [r for r in all_runners if r["actual_bsp"] is not None]
        log.info(
            f"Retrieved BSPs for {len(settled)} runners "
            f"({len(all_runners)} total) across {len(markets)} markets"
        )
        return all_runners

    # ------------------------------------------------------------------
    # Market status helpers
    # ------------------------------------------------------------------

    def get_non_completed_markets(
        self,
        target_date: str | date,
        country_codes: list[str] | None = None,
    ) -> list[dict]:
        """Get markets that haven't completed yet (upcoming or in-play).

        Returns market catalogue entries filtered to only those with
        status OPEN or SUSPENDED (not yet settled).

        Args:
            target_date: Date to query.
            country_codes: Country filter (default: GB, IE).

        Returns:
            List of market catalogue entries for non-completed races.
        """
        markets = self.list_horse_racing_markets(
            target_date, country_codes=country_codes
        )

        # Get status for each market
        non_completed = []
        market_ids = [m["marketId"] for m in markets]

        # Batch into groups of 40 (API limit)
        for i in range(0, len(market_ids), 40):
            batch = market_ids[i:i + 40]
            params = {
                "marketIds": batch,
                "priceProjection": {
                    "priceData": ["SP_AVAILABLE"],
                },
            }
            try:
                books = self._api_call("listMarketBook", params)
            except Exception as e:
                log.warning(f"Failed to get market books: {e}")
                continue

            status_map = {b["marketId"]: b.get("status") for b in books}

            for market in markets:
                mid = market["marketId"]
                if mid in status_map and status_map[mid] in ("OPEN", "SUSPENDED"):
                    market["_status"] = status_map[mid]
                    non_completed.append(market)

        log.info(
            f"Found {len(non_completed)} non-completed markets "
            f"out of {len(markets)} total"
        )
        return non_completed

    def logout(self):
        """End the Betfair session."""
        if self.session_token:
            try:
                self._session.post(
                    "https://identitysso.betfair.com/api/logout",
                    headers={
                        "X-Application": self.app_key,
                        "X-Authentication": self.session_token,
                        "Accept": "application/json",
                    },
                )
                log.info("Betfair session closed")
            except Exception:
                pass
            self.session_token = None


def _normalise_venue(venue: str) -> str:
    """Normalise Betfair venue names to match HRB track names.

    Betfair uses formats like 'Newc' for Newcastle, 'Wolv' for
    Wolverhampton, etc. This maps common abbreviations.
    """
    venue = venue.strip()

    # Common Betfair abbreviations -> full names
    abbreviations = {
        "Newc": "Newcastle",
        "Wolv": "Wolverhampton",
        "Kemp": "Kempton",
        "Sand": "Sandown",
        "Donc": "Doncaster",
        "Chel": "Cheltenham",
        "Aint": "Aintree",
        "Newb": "Newbury",
        "York": "York",
        "Asc": "Ascot",
        "Hayd": "Haydock",
        "Leic": "Leicester",
        "Nott": "Nottingham",
        "Ling": "Lingfield",
        "Weth": "Wetherby",
        "Ffly": "Ffos Las",
        "Chep": "Chepstow",
        "Muss": "Musselburgh",
        "Carl": "Carlisle",
        "Hntg": "Huntingdon",
        "Sedg": "Sedgefield",
        "Plmp": "Plumpton",
        "Font": "Fontwell",
        "Taun": "Taunton",
        "Bang": "Bangor-On-Dee",
        "Winc": "Wincanton",
        "Stfd": "Stratford",
        "Uttx": "Uttoxeter",
        "Mark": "Market Rasen",
        "Catt": "Catterick",
        "Thir": "Thirsk",
        "Ripon": "Ripon",
        "Brig": "Brighton",
        "Bath": "Bath",
        "Ches": "Chester",
        "Good": "Goodwood",
        "Newm": "Newmarket",
        "Epso": "Epsom",
        "Warw": "Warwick",
        "Here": "Hereford",
        "Ludl": "Ludlow",
        "Kelso": "Kelso",
        "Ayr": "Ayr",
        "Perth": "Perth",
        "Hamil": "Hamilton",
        "Curr": "Curragh",
        "Naas": "Naas",
        "Leop": "Leopardstown",
        "Fair": "Fairyhouse",
        "Punches": "Punchestown",
        "Galw": "Galway",
        "Lime": "Limerick",
        "Cork": "Cork",
        "Down": "Down Royal",
        "Navan": "Navan",
        "Gowran": "Gowran Park",
        "Tipp": "Tipperary",
        "Wexf": "Wexford",
        "Tram": "Tramore",
        "Dund": "Dundalk",
        "Balli": "Ballinrobe",
        "Kill": "Killarney",
        "Sligo": "Sligo",
        "List": "Listowel",
        "Rosc": "Roscommon",
        "Thurl": "Thurles",
        "Clonm": "Clonmel",
        "Dowp": "Downpatrick",
        "Belf": "Bellewstown",
        "Layer": "Laytown",
    }

    if venue in abbreviations:
        return abbreviations[venue]

    return venue


def match_runner_name(betfair_name: str, db_name: str) -> bool:
    """Check if a Betfair runner name matches a database horse name.

    Betfair names often include country suffix like "Horse Name (IRE)".
    DB names may or may not include this.

    Args:
        betfair_name: Name from Betfair (e.g. "Tiger Roll (IRE)").
        db_name: Name from the database (e.g. "Tiger Roll").

    Returns:
        True if names match (case-insensitive, ignoring country suffix).
    """
    # Strip country suffix from Betfair name: "(IRE)", "(GB)", "(FR)", etc.
    bf_clean = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", betfair_name).strip()
    db_clean = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", db_name).strip()

    return bf_clean.lower() == db_clean.lower()
