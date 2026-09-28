"""The Betfair Exchange, read-only, and a paper exchange that simulates orders against it.

BetfairData      reads markets and prices through betfair_client.BetfairClient (listMarketCatalogue,
                 listMarketBook). It has no order methods: nothing in this package can place, change
                 or cancel a bet on the exchange.
PaperExchange    reads through BetfairData and simulates orders: a back fills at once against the book
                 last read, at the prices offered at or above its limit (as a fill-or-kill limit order
                 would), and the rest lapses; a lay at BSP is held until the market reconciles.

Placing real orders is deliberately not built here: it is the owner's decision to enable, and
TRADING.md says what it would take.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

#: Betfair's price ladder: (upper bound of the band, tick size)
_LADDER = ((2.0, 0.01), (3.0, 0.02), (4.0, 0.05), (6.0, 0.1), (10.0, 0.2), (20.0, 0.5),
           (30.0, 1.0), (50.0, 2.0), (100.0, 5.0), (1000.0, 10.0))


def tick_up(price: float) -> float:
    """The smallest price on Betfair's ladder at or above `price`."""
    price = max(1.01, float(price))
    lo = 1.0
    for top, step in _LADDER:
        if price <= top + 1e-9:
            n = math.ceil(round((price - lo) / step, 9))
            return round(min(lo + n * step, 1000.0), 2)
        lo = top
    return 1000.0


def tick_down(price: float) -> float:
    """The largest price on Betfair's ladder at or below `price`."""
    price = min(1000.0, float(price))
    lo = 1.0
    for top, step in _LADDER:
        if price <= top + 1e-9:
            n = math.floor(round((price - lo) / step, 9))
            return round(max(lo + n * step, 1.01), 2)
        lo = top
    return 1000.0


@dataclass
class Market:
    market_id: str
    venue: str
    country: str
    start: datetime                       # UTC
    name: str = ""
    runners: dict = field(default_factory=dict)       # selection_id -> runner name


@dataclass
class Quote:
    selection_id: int
    status: str = "ACTIVE"                # ACTIVE, REMOVED, WINNER, LOSER, PLACED
    back: list = field(default_factory=list)           # [(price, size)], best (highest) first
    lay: list = field(default_factory=list)            # [(price, size)], best (lowest) first
    last_traded: float | None = None
    traded: float = 0.0
    bsp: float | None = None

    @property
    def best_back(self) -> tuple[float, float] | None:
        return self.back[0] if self.back else None

    @property
    def best_lay(self) -> tuple[float, float] | None:
        return self.lay[0] if self.lay else None


@dataclass
class Book:
    market_id: str
    status: str                            # OPEN, SUSPENDED, CLOSED
    inplay: bool
    total_matched: float | None            # None when Betfair leaves it out (a delayed key may)
    runners: dict                          # selection_id -> Quote
    bsp_reconciled: bool = False


@dataclass
class Fill:
    """A simulated order and what it would have matched."""
    market_id: str
    selection_id: int
    side: str                              # BACK, LAY
    order_type: str                        # LIMIT, MARKET_ON_CLOSE
    price: float                           # the limit price (0 for a BSP order)
    size: float                            # the stake asked for (the liability for a BSP lay)
    matched: float = 0.0
    avg_price: float = 0.0
    bet_id: str = ""
    status: str = ""                       # SUCCESS, FAILURE, PENDING (a BSP order before the off)
    error: str = ""


def parse_book(raw: dict) -> Book:
    runners = {}
    for r in raw.get("runners", []):
        ex = r.get("ex", {}) or {}
        sp = r.get("sp", {}) or {}
        runners[int(r["selectionId"])] = Quote(
            selection_id=int(r["selectionId"]), status=r.get("status", "ACTIVE"),
            back=[(float(o["price"]), float(o["size"])) for o in ex.get("availableToBack", [])],
            lay=[(float(o["price"]), float(o["size"])) for o in ex.get("availableToLay", [])],
            last_traded=r.get("lastPriceTraded"), traded=float(r.get("totalMatched") or 0.0),
            bsp=sp.get("actualSP") if isinstance(sp.get("actualSP"), (int, float)) else None)
    return Book(market_id=raw["marketId"], status=raw.get("status", "OPEN"), inplay=bool(raw.get("inplay")),
                total_matched=(float(raw["totalMatched"]) if raw.get("totalMatched") is not None else None),
                runners=runners,
                bsp_reconciled=bool(raw.get("bspReconciled")))


class BetfairData:
    """Betfair's markets and prices, read-only."""

    def __init__(self, client=None):
        if client is None:
            from betfair_client import BetfairClient
            client = BetfairClient()
        self.client = client

    def login(self) -> None:
        self.client.login()

    def _read(self, method: str, params: dict):
        if method not in ("listMarketCatalogue", "listMarketBook"):
            raise ValueError(f"{method}: this client only reads markets and prices")
        try:
            return self.client._api_call(method, params)
        except Exception as exc:                       # an expired session: log in once and retry
            text = str(exc)
            if "INVALID_SESSION" in text or "NO_SESSION" in text or "Not logged in" in text:
                log.warning("Betfair session lost (%s); logging in again", text[:120])
                self.client.login()
                return self.client._api_call(method, params)
            raise

    def markets(self, start: datetime, end: datetime, countries=("GB", "IE")) -> list[Market]:
        raw = self._read("listMarketCatalogue", {
            "filter": {"eventTypeIds": ["7"], "marketCountries": list(countries), "marketTypeCodes": ["WIN"],
                       "marketStartTime": {"from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                           "to": end.strftime("%Y-%m-%dT%H:%M:%SZ")}},
            "marketProjection": ["EVENT", "MARKET_START_TIME", "RUNNER_DESCRIPTION"],
            "maxResults": 1000, "sort": "FIRST_TO_START"})
        out = []
        for m in raw or []:
            ev = m.get("event", {}) or {}
            out.append(Market(
                market_id=m["marketId"], venue=ev.get("venue", ""), country=ev.get("countryCode", ""),
                start=datetime.strptime(m["marketStartTime"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc),
                name=m.get("marketName", ""),
                runners={int(r["selectionId"]): r.get("runnerName", "") for r in m.get("runners", [])}))
        return out

    def books(self, market_ids: list[str], with_sp: bool = False) -> dict[str, Book]:
        data = ["EX_BEST_OFFERS"] + (["SP_TRADED"] if with_sp else [])
        batch = 10 if with_sp else 25
        out = {}
        for i in range(0, len(market_ids), batch):
            raw = self._read("listMarketBook", {
                "marketIds": market_ids[i:i + batch],
                "priceProjection": {"priceData": data, "exBestOffersOverrides": {"bestPricesDepth": 3}}})
            for b in raw or []:
                out[b["marketId"]] = parse_book(b)
        return out


class PaperExchange:
    """Reads Betfair's markets and prices; simulates every order; sends none."""

    def __init__(self, data: BetfairData | None, bank: float = 1000.0):
        self.data = data
        self.bank = bank
        self._books: dict[str, Book] = {}
        self._n = 0

    def login(self) -> None:
        if self.data is not None:
            self.data.login()

    def markets(self, start, end, countries=("GB", "IE")) -> list[Market]:
        return self.data.markets(start, end, countries)

    def books(self, market_ids: list[str], with_sp: bool = False) -> dict[str, Book]:
        got = self.data.books(market_ids, with_sp)
        self._books.update(got)
        return got

    def back(self, market_id, selection_id, price, size, ref, min_fill: float = 2.0) -> Fill:
        """Fill now against the book last read: every level offered at `price` or better, up to `size`."""
        self._n += 1
        fill = Fill(market_id=market_id, selection_id=int(selection_id), side="BACK", order_type="LIMIT",
                    price=float(price), size=round(float(size), 2), bet_id=f"paper-{self._n}")
        book = self._books.get(market_id)
        q = book.runners.get(int(selection_id)) if book else None
        if not book or book.status != "OPEN" or book.inplay or q is None or q.status != "ACTIVE":
            fill.status, fill.error = "FAILURE", "MARKET_NOT_OPEN_FOR_BETTING"
            return fill
        left, cost, got = fill.size, 0.0, 0.0
        for p, s in q.back:
            if p + 1e-9 < price or left <= 0:
                break
            take = min(s, left)
            got += take
            cost += take * p
            left -= take
        if got + 1e-9 < min(min_fill, fill.size):
            fill.status, fill.error = "FAILURE", "NOT_MATCHED"
            return fill
        fill.matched, fill.avg_price, fill.status = round(got, 2), round(cost / got, 4), "SUCCESS"
        return fill

    def lay_at_bsp(self, market_id, selection_id, liability, ref) -> Fill:
        self._n += 1
        return Fill(market_id=market_id, selection_id=int(selection_id), side="LAY", order_type="MARKET_ON_CLOSE",
                    price=0.0, size=round(float(liability), 2), bet_id=f"paper-{self._n}", status="PENDING")

    def cancel_all(self, market_id=None) -> None:
        return None


def uk_day_window(day, now: datetime | None = None) -> tuple[datetime, datetime]:
    """The UTC span of one UK racing day, from now (or midnight) to the next midnight UK time."""
    from zoneinfo import ZoneInfo
    uk = ZoneInfo("Europe/London")
    start = datetime(day.year, day.month, day.day, tzinfo=uk).astimezone(timezone.utc)
    end = (datetime(day.year, day.month, day.day, tzinfo=uk) + timedelta(days=1)).astimezone(timezone.utc)
    if now is not None and now > start:
        start = now
    return start, end
