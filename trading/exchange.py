"""The Betfair Exchange: read-only data, a paper exchange that simulates orders, and the live exchange.

BetfairData      reads markets and prices through betfair_client.BetfairClient (listMarketCatalogue,
                 listMarketBook), and the account's settled bets (listClearedOrders). It has no order methods.
PaperExchange    reads through BetfairData and simulates orders: a back fills at once against the book
                 last read, at the prices offered at or above its limit (as a fill-or-kill limit order
                 would), and the rest lapses; a lay at BSP is held until the market reconciles.
LiveExchange     the owner's decision (30 Sep 2026): reads through BetfairData and places real orders on
                 the owner's account, the same two kinds the paper exchange simulates. Only auto_trade.py
                 --live makes one, and only with TRADING_LIVE=yes.
"""

from __future__ import annotations

import logging
import math
import time
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
    """Betfair's markets and prices, read-only. With a ``recorder`` (betfair_recorder.DayRecorder) every catalogue
    and book read is kept for research; the record never stands in the way of the reading."""

    def __init__(self, client=None, recorder=None, source: str = "trader"):
        if client is None:
            from betfair_client import BetfairClient
            client = BetfairClient()
        self.client = client
        self.recorder = recorder
        self.source = source

    def _keep(self, kind: str, raw) -> None:
        if self.recorder is None or not raw:
            return
        try:
            if kind == "books":
                self.recorder.record_books(raw, source=self.source)
            else:
                self.recorder.record_catalogue(raw)
        except Exception as exc:                       # research data must never stop trading
            log.warning("recorder: %s not kept (%s)", kind, exc)

    def login(self) -> None:
        self.client.login()

    def funds(self) -> dict | None:
        """The account's funds (the Accounts API's getAccountFunds; read-only): available to bet, the exposure, the
        exposure limit and the retained commission. None when they cannot be read: they are for the record."""
        try:
            raw = self.client.account_funds() or {}
        except Exception as exc:
            log.warning("account funds not read (%s)", str(exc)[:200])
            return None
        return {"available": raw.get("availableToBetBalance"), "exposure": raw.get("exposure"),
                "exposure_limit": raw.get("exposureLimit"), "retained_commission": raw.get("retainedCommission")}

    def _read(self, method: str, params: dict):
        if method not in ("listMarketCatalogue", "listMarketBook", "listClearedOrders", "listEventTypes",
                          "listMarketTypes"):
            raise ValueError(f"{method}: this client only reads markets, prices and settled bets")
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
        self._keep("catalogue", raw)
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
        # with_sp, after the off: SP_TRADED alone brought the money taken at SP but never the BSP itself (6-7 Oct, the
        # live key; ledger bsp-at-off-1007), so the projection's SP_AVAILABLE is asked too. Betfair's weight a market:
        # EX_BEST_OFFERS 5, SP_AVAILABLE 3, SP_TRADED 7; ten markets a call stay under its limit of 200
        data = ["EX_BEST_OFFERS"] + (["SP_AVAILABLE", "SP_TRADED"] if with_sp else [])
        batch = 10 if with_sp else 25
        out = {}
        for i in range(0, len(market_ids), batch):
            raw = self._read("listMarketBook", {
                "marketIds": market_ids[i:i + batch],
                "priceProjection": {"priceData": data, "exBestOffersOverrides": {"bestPricesDepth": 3}}})
            self._keep("books", raw)
            for b in raw or []:
                out[b["marketId"]] = parse_book(b)
        return out

    def cleared(self, market_ids: list[str]) -> list[dict]:
        """The account's settled bets on these markets, one record per bet (listClearedOrders, SETTLED): what
        Betfair paid on each back and each lay at SP ("profit", before commission) and the price each was matched
        at (for a lay at SP, the Betfair SP). A market appears once Betfair has settled it, some minutes after the
        race; a void bet (a non-runner) never does."""
        out = []
        for i in range(0, len(market_ids), 50):
            start = 0
            while True:
                reply = self._read("listClearedOrders", {
                    "betStatus": "SETTLED", "marketIds": list(market_ids[i:i + 50]),
                    "fromRecord": start, "recordCount": 1000}) or {}
                got = reply.get("clearedOrders", []) or []
                out += got
                if not reply.get("moreAvailable") or not got:
                    break
                start += len(got)
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

    def cleared(self, market_ids: list[str]) -> list[dict]:
        """The account's settled bets (read-only): the evening job settles the live ledger through this exchange."""
        return self.data.cleared(market_ids)

    def funds(self) -> dict | None:
        return None                                        # paper money is never short

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


def _floor2(x: float) -> float:
    return math.floor(float(x) * 100 + 1e-9) / 100


class LiveExchange:
    """Reads through BetfairData and places real orders on the owner's account.

    A back is a limit order at the price read, filled at once or not at all (FILL_OR_KILL, at least `min_fill`):
    nothing rests in the book, so a price that has moved is missed, never chased. The trade-out is a lay at the
    Betfair SP for a liability (MARKET_ON_CLOSE), matched when the market reconciles at the off. Nothing is ever
    cancelled: the only orders left open are those lays, and they are the hedges. When a reply is lost (a timeout,
    a dropped connection) the order may still have been placed, so the exchange asks Betfair for the orders under
    that reference before reporting a failure: a bet is never placed twice for want of an answer.
    """

    mode = "live"

    def __init__(self, data: BetfairData):
        self.data = data
        self._n = 0

    def login(self) -> None:
        self.data.login()

    def markets(self, start, end, countries=("GB", "IE")) -> list[Market]:
        return self.data.markets(start, end, countries)

    def books(self, market_ids: list[str], with_sp: bool = False) -> dict[str, Book]:
        return self.data.books(market_ids, with_sp)

    def cleared(self, market_ids: list[str]) -> list[dict]:
        return self.data.cleared(market_ids)

    def funds(self) -> dict | None:
        return self.data.funds()

    # ------------------------------------------------------------------ orders

    def _call(self, method: str, params: dict):
        """One API call; an expired session is logged in again and the call made once more (Betfair refuses a call
        without a session before doing anything, so an order refused that way was not placed)."""
        try:
            return self.data.client._api_call(method, params)
        except Exception as exc:
            text = str(exc)
            if "INVALID_SESSION" in text or "NO_SESSION" in text or "Not logged in" in text:
                log.warning("Betfair session lost on %s (%s); logging in again", method, text[:120])
                self.data.client.login()
                return self.data.client._api_call(method, params)
            raise

    def _place(self, market_id: str, instruction: dict, ref: str) -> dict | None:
        """placeOrders for one instruction; None when the reply was lost."""
        self._n += 1
        params = {"marketId": market_id, "instructions": [instruction],
                  "customerRef": f"{ref[:22]}-{int(time.time()) % 100000}-{self._n % 100}"[:32]}
        try:
            return self._call("placeOrders", params) or {}
        except Exception as exc:
            log.error("placeOrders %s %s: no reply (%s); asking Betfair what was placed", market_id, ref, exc)
            return None

    def orders(self, market_ids: list[str], refs: list[str] | None = None) -> list[dict]:
        """Our orders on these markets as Betfair holds them (listCurrentOrders: open, and matched but not yet
        settled), optionally only those under the given references."""
        out, start = [], 0
        while True:
            params = {"marketIds": list(market_ids), "orderProjection": "ALL", "fromRecord": start,
                      "recordCount": 1000}
            if refs:
                params["customerOrderRefs"] = list(refs)
            reply = self._call("listCurrentOrders", params) or {}
            got = reply.get("currentOrders", []) or []
            out += got
            if not reply.get("moreAvailable") or not got:
                return out
            start += len(got)

    @staticmethod
    def _report(reply: dict) -> tuple[dict, bool]:
        reps = reply.get("instructionReports") or [{}]
        rep = reps[0] or {}
        ok = reply.get("status") == "SUCCESS" and rep.get("status", "SUCCESS") == "SUCCESS"
        return rep, ok

    @staticmethod
    def _reason(reply: dict, rep: dict, default: str) -> str:
        """Why Betfair refused an order. An instruction's ERROR_IN_ORDER says only that the order as a whole failed;
        the reason (INSUFFICIENT_FUNDS, MARKET_SUSPENDED, ...) is the report's own code, so that goes first."""
        codes = list(dict.fromkeys(c for c in (str(rep.get("errorCode") or ""), str(reply.get("errorCode") or ""))
                                   if c))
        telling = [c for c in codes if c != "ERROR_IN_ORDER"] or codes
        if not telling:
            return default
        rest = [c for c in codes if c != telling[0]]
        return f"{telling[0]} ({', '.join(rest)})" if rest else telling[0]

    def back(self, market_id, selection_id, price, size, ref, min_fill: float = 2.0) -> Fill:
        size = _floor2(size)
        fill = Fill(market_id=market_id, selection_id=int(selection_id), side="BACK", order_type="LIMIT",
                    price=float(price), size=size)
        instruction = {"selectionId": int(selection_id), "handicap": 0, "side": "BACK", "orderType": "LIMIT",
                       "customerOrderRef": ref[:32],
                       "limitOrder": {"size": size, "price": float(price), "persistenceType": "LAPSE",
                                      "timeInForce": "FILL_OR_KILL", "minFillSize": _floor2(min(min_fill, size))}}
        reply = self._place(market_id, instruction, ref)
        if reply is None:                                  # placed or not? Betfair's own record decides
            return self._confirm_back(fill, ref)
        rep, ok = self._report(reply)
        fill.bet_id = str(rep.get("betId") or "")
        matched = _floor2(rep.get("sizeMatched") or 0.0)
        if ok and matched > 0:
            fill.matched, fill.avg_price, fill.status = matched, float(rep.get("averagePriceMatched") or price), "SUCCESS"
        else:
            fill.status = "FAILURE"
            fill.error = self._reason(reply, rep, str(rep.get("orderStatus") or "NOT_MATCHED"))
        return fill

    def _confirm_back(self, fill: Fill, ref: str) -> Fill:
        try:
            got = [o for o in self.orders([fill.market_id], [ref[:32]])
                   if o.get("side") == "BACK" and int(o.get("selectionId", 0)) == fill.selection_id]
        except Exception as exc:
            fill.status, fill.error = "UNKNOWN", f"no reply, and the orders could not be read: {exc}"[:200]
            return fill
        # the orders under this reference may include earlier fills of the same horse today: the newest is ours
        got.sort(key=lambda o: str(o.get("placedDate", "")))
        if got and float(got[-1].get("sizeMatched") or 0) > 0:
            o = got[-1]
            fill.bet_id, fill.status = str(o.get("betId") or ""), "SUCCESS"
            fill.matched, fill.avg_price = _floor2(o["sizeMatched"]), float(o.get("averagePriceMatched") or fill.price)
            fill.error = "reply lost; confirmed from Betfair's orders"
        else:
            fill.status, fill.error = "FAILURE", "reply lost; no matched order found under the reference"
        return fill

    def lay_at_bsp(self, market_id, selection_id, liability, ref) -> Fill:
        liability = _floor2(liability)
        fill = Fill(market_id=market_id, selection_id=int(selection_id), side="LAY", order_type="MARKET_ON_CLOSE",
                    price=0.0, size=liability)
        instruction = {"selectionId": int(selection_id), "handicap": 0, "side": "LAY", "orderType": "MARKET_ON_CLOSE",
                       "customerOrderRef": ref[:32], "marketOnCloseOrder": {"liability": liability}}
        reply = self._place(market_id, instruction, ref)
        if reply is None:
            try:
                have = sum(float(o.get("bspLiability") or 0) for o in self.orders([market_id], [ref[:32]])
                           if o.get("side") == "LAY" and int(o.get("selectionId", 0)) == int(selection_id))
            except Exception as exc:
                fill.status, fill.error = "UNKNOWN", f"no reply, and the orders could not be read: {exc}"[:200]
                return fill
            fill.status = "PENDING" if have + 0.01 >= liability else "FAILURE"
            fill.error = f"reply lost; Betfair holds GBP{have:.2f} of lays at SP under the reference"
            return fill
        rep, ok = self._report(reply)
        fill.bet_id = str(rep.get("betId") or "")
        if ok:
            fill.status = "PENDING"                        # matched at the off, at the Betfair SP
        else:
            fill.status = "FAILURE"
            fill.error = self._reason(reply, rep, "FAILURE")
        return fill

    def cancel_all(self, market_id=None) -> None:
        """Nothing to cancel: the backs are fill-or-kill, and the only open orders are the lays at SP that hedge
        them. The kill switch stops new bets; it never unhedges a position."""
        log.warning("live: the kill switch stops new bets; the open lays at SP are the hedges and stay")


def uk_day_window(day, now: datetime | None = None) -> tuple[datetime, datetime]:
    """The UTC span of one UK racing day, from now (or midnight) to the next midnight UK time."""
    from zoneinfo import ZoneInfo
    uk = ZoneInfo("Europe/London")
    start = datetime(day.year, day.month, day.day, tzinfo=uk).astimezone(timezone.utc)
    end = (datetime(day.year, day.month, day.day, tzinfo=uk) + timedelta(days=1)).astimezone(timezone.utc)
    if now is not None and now > start:
        start = now
    return start, end
