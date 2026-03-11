"""Betfair Exchange API client.

Thin wrapper around *betfairlightweight* that exposes only the
operations required by the automated betting pipeline:

* List WIN markets for UK & Ireland horse racing
* Fetch live back/lay prices
* Place and cancel LIMIT orders
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import betfairlightweight
from betfairlightweight import filters

log = logging.getLogger(__name__)

HORSE_RACING_EVENT_TYPE_ID = "7"
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds


class BetfairClient:
    """Authenticated Betfair Exchange API client.

    Args:
        username: Betfair account username.
        password: Betfair account password.
        app_key: Betfair API application key.
        certs_dir: Directory containing ``client-2048.crt`` and
            ``client-2048.key`` SSL certificate files.
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        app_key: str | None = None,
        certs_dir: str | None = None,
    ):
        self.username = username or os.getenv("BETFAIR_USERNAME", "")
        self.password = password or os.getenv("BETFAIR_PASSWORD", "")
        self.app_key = app_key or os.getenv("BETFAIR_APP_KEY", "")
        self.certs_dir = certs_dir or os.getenv("BETFAIR_CERTS_DIR", "certs")

        self.client = betfairlightweight.APIClient(
            username=self.username,
            password=self.password,
            app_key=self.app_key,
            certs=self.certs_dir,
        )
        self._logged_in = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def login(self) -> None:
        """Authenticate with Betfair using SSL certificates."""
        self.client.login()
        self._logged_in = True
        log.info("Betfair login successful")

    def keep_alive(self) -> None:
        """Refresh the session token (call periodically for long runs)."""
        self.client.keep_alive()

    def logout(self) -> None:
        """End the Betfair session."""
        if self._logged_in:
            self.client.logout()
            self._logged_in = False
            log.info("Betfair logout successful")

    # ------------------------------------------------------------------
    # Market catalogue
    # ------------------------------------------------------------------

    def list_market_catalogue(
        self,
        target_date: datetime | None = None,
        market_countries: list[str] | None = None,
        market_type_codes: list[str] | None = None,
    ) -> list[dict]:
        """Fetch all WIN markets for UK & Ireland horse racing on *target_date*.

        Returns a list of dicts, each with keys:
            market_id, market_name, market_start_time,
            venue, runners [{selection_id, runner_name, sort_priority}]
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc)
        if market_countries is None:
            market_countries = ["GB", "IE"]
        if market_type_codes is None:
            market_type_codes = ["WIN"]

        # Time window: start of target day to end of target day (UTC)
        day_start = target_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = day_start + timedelta(days=1)

        market_filter = filters.market_filter(
            event_type_ids=[HORSE_RACING_EVENT_TYPE_ID],
            market_countries=market_countries,
            market_type_codes=market_type_codes,
            market_start_time={
                "from": day_start.isoformat(),
                "to": day_end.isoformat(),
            },
        )

        catalogues = self._retry(
            lambda: self.client.betting.list_market_catalogue(
                filter=market_filter,
                market_projection=[
                    "RUNNER_DESCRIPTION",
                    "MARKET_START_TIME",
                    "EVENT",
                ],
                max_results=200,
                sort="FIRST_TO_START",
            )
        )

        results = []
        for cat in catalogues:
            venue = cat.event.venue if cat.event else ""
            runners = []
            for runner in cat.runners:
                runners.append({
                    "selection_id": runner.selection_id,
                    "runner_name": runner.runner_name,
                    "sort_priority": runner.sort_priority,
                })
            results.append({
                "market_id": cat.market_id,
                "market_name": cat.market_name,
                "market_start_time": cat.market_start_time,
                "venue": venue,
                "runners": runners,
            })

        log.info(
            f"Fetched {len(results)} market catalogues "
            f"({sum(len(r['runners']) for r in results)} runners)"
        )
        return results

    # ------------------------------------------------------------------
    # Live prices
    # ------------------------------------------------------------------

    def get_market_books(
        self, market_ids: list[str]
    ) -> list[dict]:
        """Fetch live back/lay prices for the given markets.

        Returns a list of dicts, each with keys:
            market_id, status,
            runners [{selection_id, status, last_price_traded,
                      best_back_price, best_back_size,
                      best_lay_price, best_lay_size}]

        Betfair limits ``listMarketBook`` to 40 market IDs per call,
        so this method batches automatically.
        """
        all_results = []
        batch_size = 40

        for i in range(0, len(market_ids), batch_size):
            batch = market_ids[i : i + batch_size]
            books = self._retry(
                lambda b=batch: self.client.betting.list_market_book(
                    market_ids=b,
                    price_projection=filters.price_projection(
                        price_data=["EX_BEST_OFFERS"],
                    ),
                )
            )

            for book in books:
                runners = []
                for runner in book.runners:
                    back_prices = runner.ex.available_to_back or []
                    lay_prices = runner.ex.available_to_lay or []
                    runners.append({
                        "selection_id": runner.selection_id,
                        "status": runner.status,
                        "last_price_traded": runner.last_price_traded,
                        "best_back_price": back_prices[0].price if back_prices else None,
                        "best_back_size": back_prices[0].size if back_prices else None,
                        "best_lay_price": lay_prices[0].price if lay_prices else None,
                        "best_lay_size": lay_prices[0].size if lay_prices else None,
                    })
                all_results.append({
                    "market_id": book.market_id,
                    "status": book.status,
                    "runners": runners,
                })

        log.info(f"Fetched prices for {len(all_results)} markets")
        return all_results

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_orders(
        self,
        market_id: str,
        instructions: list[dict],
    ) -> dict:
        """Place LIMIT orders on a single market.

        Each instruction dict must contain:
            selection_id (int), side ("BACK"|"LAY"),
            price (float), size (float)

        Returns dict with keys: status, market_id, instruction_reports.
        """
        order_instructions = []
        for inst in instructions:
            order_instructions.append(
                filters.place_instruction(
                    order_type="LIMIT",
                    selection_id=inst["selection_id"],
                    side=inst["side"],
                    limit_order=filters.limit_order(
                        size=round(inst["size"], 2),
                        price=inst["price"],
                        persistence_type="LAPSE",
                    ),
                )
            )

        response = self._retry(
            lambda: self.client.betting.place_orders(
                market_id=market_id,
                instructions=order_instructions,
            )
        )

        reports = []
        for report in response.place_instruction_reports:
            reports.append({
                "bet_id": report.bet_id,
                "status": report.status,
                "size_matched": report.size_matched,
                "average_price_matched": report.average_price_matched,
                "order_status": report.order_status,
            })

        result = {
            "status": response.status,
            "market_id": response.market_id,
            "instruction_reports": reports,
        }

        log.info(
            f"Placed {len(instructions)} order(s) on {market_id}: "
            f"status={response.status}"
        )
        return result

    def cancel_orders(
        self, market_id: str, bet_ids: list[str] | None = None
    ) -> dict:
        """Cancel unmatched orders on a market.

        If *bet_ids* is ``None``, cancels all unmatched orders on the market.
        """
        cancel_instructions = None
        if bet_ids:
            cancel_instructions = [
                filters.cancel_instruction(bet_id=bid) for bid in bet_ids
            ]

        response = self._retry(
            lambda: self.client.betting.cancel_orders(
                market_id=market_id,
                instructions=cancel_instructions,
            )
        )
        log.info(f"Cancelled orders on {market_id}: status={response.status}")
        return {"status": response.status, "market_id": market_id}

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _retry(self, func, retries: int = MAX_RETRIES):
        """Execute *func* with exponential backoff on transient failures."""
        for attempt in range(retries + 1):
            try:
                return func()
            except Exception as exc:
                if attempt == retries:
                    raise
                wait = RETRY_BACKOFF * (2 ** attempt)
                log.warning(
                    f"Betfair API call failed (attempt {attempt + 1}/"
                    f"{retries + 1}): {exc}. Retrying in {wait}s..."
                )
                time.sleep(wait)
