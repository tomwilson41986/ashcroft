"""The paper trader (build spec 7): hypothetical orders against the recorded books, never against the last traded
price, with the market's own bet delay.

An order is submitted at the best price shown when its signal fires. It is judged on the first recorded book at
least ``betDelay`` seconds later (the delay read from the market's book at submission):

- the market is not open then (suspended for a wicket, a review, the innings break, closed): the order lapses, as
  Betfair lapses an in-play bet the market suspends during its delay;
- else it matches against what that book shows at the order's price or better, level by level, best first (a back
  at P against the backs at P or longer, at their prices; a lay against the lays at P or shorter);
- what is left rests at its price (``persistence``: LAPSE, the default, cancels it at the next suspension, PERSIST
  keeps it) and is matched only when a later open book trades through it (a back at P once the best back is longer
  than P), at P: our place in the queue at P is not known, so being at the price is never counted as a fill.

One order at a time per selection. Each matched bet keeps its closing-line values: ``clv_30`` against the mid 30
seconds after its delay ends (build spec 8: delay-plus-30-seconds), ``clv_close`` against the last pre-play mid. A
settled market's net winnings pay commission at the account's rate (cricket/ledger/fees.py).
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import asdict, dataclass, field

import pandas as pd

from cricket.ledger.fees import Transactions, commission
from cricket.market.book import Book
from cricket.trading.signals import Rules, Signal
from cricket.trading.sizing import Sizer

CLV_AFTER_S = 30.0


@dataclass
class PaperOrder:
    ref: str
    match_id: str
    market_id: str
    selection_id: int
    side: str
    price: float
    stake: float
    p: float
    value: float
    submitted: dt.datetime
    delay_s: int
    due: dt.datetime
    status: str = "pending"            # pending, resting, matched, partial, lapsed
    matched: float = 0.0
    matched_value: float = 0.0         # sum of size x price over fills
    fills: list = field(default_factory=list)
    first_fill: dt.datetime | None = None
    clv_30: float | None = None
    clv_close: float | None = None
    pnl: float | None = None

    @property
    def avg_price(self) -> float | None:
        return self.matched_value / self.matched if self.matched else None

    @property
    def remaining(self) -> float:
        return round(self.stake - self.matched, 2)

    def fill(self, when: dt.datetime, price: float, size: float) -> None:
        size = round(min(size, self.remaining), 2)
        if size <= 0:
            return
        self.matched = round(self.matched + size, 2)
        self.matched_value += size * price
        self.fills.append((when.isoformat(), price, size))
        self.first_fill = self.first_fill or when


class PaperTrader:
    def __init__(self, rules: Rules, sizer: Sizer, persistence: str = "LAPSE"):
        self.rules, self.sizer, self.persistence = rules, sizer, persistence
        self.orders: list[PaperOrder] = []
        self.tx = Transactions()
        self.history: dict[str, list[tuple[dt.datetime, bool, dict[int, float | None]]]] = {}
        self.selections: dict[str, list[int]] = {}
        self.settled: dict[str, int] = {}
        self._ids = itertools.count(1)
        self.commission_paid = 0.0

    # ----------------------------------------------------------------------------------------------------------
    def position(self, market_id: str) -> dict[int, float]:
        """Profit by winning selection from the matched bets on the market."""
        sels = self.selections.get(market_id, [])
        pos = {s: 0.0 for s in sels}
        for o in self.orders:
            if o.market_id != market_id or not o.matched:
                continue
            for s in sels:
                win = s == o.selection_id
                if o.side == "BACK":
                    pos[s] += o.matched_value - o.matched if win else -o.matched
                else:
                    pos[s] += -(o.matched_value - o.matched) if win else o.matched
        return pos

    def _open_order(self, market_id: str, sel: int) -> bool:
        return any(o.market_id == market_id and o.selection_id == sel and o.status in ("pending", "resting")
                   for o in self.orders)

    def submit(self, sig: Signal, book: Book, match_id: str, now: dt.datetime) -> PaperOrder | None:
        self.selections.setdefault(book.market_id, sorted(book.runners))
        if self._open_order(sig.market_id, sig.selection_id) or sig.market_id in self.settled:
            return None
        stake = self.sizer.stake(sig, self.position(sig.market_id), self.selections[sig.market_id])
        if stake <= 0:
            return None
        o = PaperOrder(ref=f"c{next(self._ids)}", match_id=match_id, market_id=sig.market_id,
                       selection_id=sig.selection_id, side=sig.side, price=sig.price, stake=stake, p=sig.p,
                       value=sig.value, submitted=now, delay_s=book.bet_delay,
                       due=now + dt.timedelta(seconds=book.bet_delay))
        self.orders.append(o)
        self.tx.add(now)
        return o

    # ----------------------------------------------------------------------------------------------------------
    @staticmethod
    def _levels(o: PaperOrder, book: Book):
        r = book.runners.get(o.selection_id)
        if r is None:
            return []
        if o.side == "BACK":
            return [(p, s) for p, s in r.back if p >= o.price]
        return [(p, s) for p, s in r.lay if p <= o.price]

    @staticmethod
    def _through(o: PaperOrder, book: Book) -> bool:
        r = book.runners.get(o.selection_id)
        if r is None:
            return False
        if o.side == "BACK":
            return bool(r.back) and r.back[0][0] > o.price
        return bool(r.lay) and r.lay[0][0] < o.price

    def on_book(self, book: Book) -> None:
        self.selections.setdefault(book.market_id, sorted(book.runners))
        self.history.setdefault(book.market_id, []).append(
            (book.pt, book.inplay, {s: r.mid for s, r in book.runners.items()}))
        for o in self.orders:
            if o.market_id != book.market_id or o.status not in ("pending", "resting") or book.pt < o.due:
                continue
            if o.status == "pending":
                if not book.open:
                    o.status = "lapsed"
                    continue
                for price, size in self._levels(o, book):
                    o.fill(book.pt, price, size)
                    if o.remaining <= 0:
                        break
                o.status = "matched" if o.remaining <= 0 else "resting"
                continue
            # resting
            if not book.open:
                if self.persistence == "LAPSE" or book.status == "CLOSED":
                    o.status = "partial" if o.matched else "lapsed"
                continue
            if self._through(o, book):
                o.fill(book.pt, o.price, o.remaining)
                o.status = "matched"

    # ----------------------------------------------------------------------------------------------------------
    def _mid_after(self, market_id: str, sel: int, t: dt.datetime) -> float | None:
        for pt, _, mids in self.history.get(market_id, []):
            if pt >= t and mids.get(sel):
                return mids[sel]
        return None

    def _close_mid(self, market_id: str, sel: int) -> float | None:
        pre = [m.get(sel) for pt, inplay, m in self.history.get(market_id, []) if not inplay and m.get(sel)]
        return pre[-1] if pre else None

    @staticmethod
    def _clv(o: PaperOrder, ref: float | None) -> float | None:
        if ref is None or not o.matched:
            return None
        return o.avg_price / ref - 1.0 if o.side == "BACK" else ref / o.avg_price - 1.0

    def settle(self, market_id: str, winner: int) -> float:
        """Settle a market on its winning selection: each bet's P&L, the market's commission. Returns the net."""
        if market_id in self.settled:
            return 0.0
        self.settled[market_id] = winner
        net = 0.0
        bets = [o for o in self.orders if o.market_id == market_id]
        for o in bets:
            if o.status in ("pending", "resting"):
                o.status = "partial" if o.matched else "lapsed"
            if not o.matched:
                o.pnl = 0.0
                continue
            win = o.selection_id == winner
            gain = o.matched_value - o.matched
            o.pnl = (gain if win else -o.matched) if o.side == "BACK" else (-gain if win else o.matched)
            o.clv_30 = self._clv(o, self._mid_after(market_id, o.selection_id,
                                                    o.due + dt.timedelta(seconds=CLV_AFTER_S)))
            o.clv_close = self._clv(o, self._close_mid(market_id, o.selection_id))
            net += o.pnl
        com = commission(net, self.rules.commission)
        self.sizer.update_bank(self.sizer.bank + net - com)
        self.commission_paid += com
        return net - com

    def ledger(self) -> pd.DataFrame:
        rows = []
        for o in self.orders:
            d = asdict(o)
            d["avg_price"] = o.avg_price
            d["fills"] = len(o.fills)
            rows.append(d)
        return pd.DataFrame(rows)
