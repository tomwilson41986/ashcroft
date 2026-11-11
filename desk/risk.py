"""The one path from a strategy's intent to an order: deterministic limits, checked in a fixed order.

A strategy never touches the exchange. It hands an ``Intent`` to ``RiskKernel.check``, which returns the size it
may send (possibly cut down) or 0 and the reason. Nothing is defaulted: a strategy at a money stage without its
limits, or a desk without the owner's envelope, sends nothing (paper only).

The order of the checks (the first that refuses wins):
    1. stopped (the S3 STOP objects, the warden, a reconciliation break)
    2. no envelope (paper only), and the strategy's stage (only pilot and scaled send money)
    3. the strategy's limits are all set
    4. stale data (the book the intent was priced on is too old)
    5. price collar (the order's price too far from the best price now) and Betfair's price range
    6. order rate (a strategy's orders a minute: a runaway loop)
    7. size: the order's own cap, the strategy's day turnover, the pilot budget
    8. loss: the order's worst case must fit what is left of the strategy's, the sport's and the desk's day loss
    9. the market: the worst case of every desk strategy together on the market, with this order
   10. funds: what is available, less the reserve kept for the live horse rule
   11. the minimum stake after every cut

Losses here are the day's settled P&L plus the worst case of the desk's open positions: conservative, so a strategy
cannot run past its day loss on positions not yet settled. The legacy traders keep their own limits; the warden
watches the account as a whole.

Paper (``paper=True``: replays and the paper stage) applies the same checks to orders that are never sent to
Betfair, so a strategy's paper record is the record it would have had under its limits: the stage and the envelope
do not hold it back (paper is how a strategy earns its stage), and the envelope's caps apply when there is one.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace

from desk.account import Position, worst_case
from desk.envelope import Envelope
from desk.registry import LIMIT_FIELDS, MONEY_STAGES, Strategy

MIN_PRICE, MAX_PRICE = 1.01, 1000.0     # Betfair's price ladder (trading/exchange.py tick_up/tick_down)


@dataclass(frozen=True)
class Intent:
    strategy_id: str
    market_id: str
    selection_id: int
    side: str                 # BACK or LAY
    price: float              # the limit price
    size: float               # the backer's stake (for a LAY, the backer's stake it accepts)
    best_price: float         # the best price on our side of the book when the intent was made
    book_age_s: float         # seconds since that book was read
    decision_price: float | None = None
    expected_value: float | None = None

    def position(self, size: float | None = None) -> Position:
        s = self.size if size is None else size
        if self.side == "BACK":
            return Position(self.market_id, self.selection_id, if_wins=s * (self.price - 1), if_loses=-s)
        return Position(self.market_id, self.selection_id, if_wins=-s * (self.price - 1), if_loses=s)

    def liability(self, size: float | None = None) -> float:
        s = self.size if size is None else size
        return s if self.side == "BACK" else s * (self.price - 1)


@dataclass
class Verdict:
    size: float
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.size > 0


@dataclass
class DeskState:
    """The day's running state across every desk strategy: settled P&L, turnover, open positions, order times."""
    settled: dict = field(default_factory=lambda: defaultdict(float))          # strategy id -> GBP settled today
    turnover: dict = field(default_factory=lambda: defaultdict(float))         # strategy id -> GBP sent today
    positions: dict = field(default_factory=lambda: defaultdict(list))         # strategy id -> [Position]
    sent: dict = field(default_factory=lambda: defaultdict(deque))             # strategy id -> order times (s)
    funds_available: float | None = None
    stopped: str = ""

    def note_sent(self, intent: Intent, now: float | None = None) -> None:
        """An order sent (matched or not): the order rate counts every one."""
        self.sent[intent.strategy_id].append(time.time() if now is None else now)

    def record(self, intent: Intent, size: float, price: float | None = None) -> None:
        """What matched: ``size`` at ``price`` (the average matched; the intent's price when not given)."""
        if size <= 0:
            return
        it = intent if price is None else replace(intent, price=float(price))
        self.turnover[it.strategy_id] += size
        self.positions[it.strategy_id].append(it.position(size))
        if self.funds_available is not None:
            self.funds_available -= it.liability(size)


def _floor_pence(x: float) -> float:
    """Down to the penny. Rounded first: in binary floating point 1.13 * 100 is 112.99999..., and a bare floor would
    take a penny off an exact stake (desk-lay-book-1112's machinery check)."""
    return math.floor(round(max(x, 0.0) * 100, 6)) / 100


def _open_worst(positions) -> float:
    """The worst case of open positions, market by market (each market's outcomes are independent of the others')."""
    by_market = defaultdict(list)
    for p in positions:
        by_market[p.market_id].append(p)
    return sum(worst_case(ps) for ps in by_market.values())


def _size_for_loss(intent: Intent, headroom: float) -> float:
    """The largest size whose worst case fits the headroom (a back loses its stake, a lay its liability)."""
    if headroom <= 0:
        return 0.0
    return headroom if intent.side == "BACK" else headroom / max(intent.price - 1, 1e-9)


class RiskKernel:
    def __init__(self, strategies: list[Strategy], envelope: Envelope | None, state: DeskState | None = None,
                 clock=time.time, paper: bool = False):
        self.by_id = {s.id: s for s in strategies}
        self.envelope = envelope
        self.state = state or DeskState()
        self.clock = clock
        self.paper = paper

    def _sport_ids(self, sport: str) -> list[str]:
        return [i for i, s in self.by_id.items() if s.sport == sport and s.desk_run]

    def check(self, it: Intent) -> Verdict:
        st, env = self.state, self.envelope
        s = self.by_id.get(it.strategy_id)
        if s is None:
            return Verdict(0, f"{it.strategy_id}: not in the registry")
        if st.stopped:
            return Verdict(0, f"stopped: {st.stopped}")
        if not s.desk_run or s.stage == "retired":
            return Verdict(0, f"stage {s.stage}: not a desk strategy that trades")
        if not self.paper:
            if env is None:
                return Verdict(0, "no envelope: paper only")
            if s.stage not in MONEY_STAGES:
                return Verdict(0, f"stage {s.stage}: no money")
        lim = s.limits or {}
        missing = [k for k in LIMIT_FIELDS if k not in lim]
        if missing:
            return Verdict(0, f"limits not set: {missing}")
        if it.side not in ("BACK", "LAY") or it.size <= 0 or it.price <= 0 or it.best_price <= 0:
            return Verdict(0, "malformed intent")
        if it.book_age_s > lim["max_book_age_s"]:
            return Verdict(0, f"stale book: {it.book_age_s:.0f}s old, over {lim['max_book_age_s']}s")
        if not (MIN_PRICE <= it.price <= MAX_PRICE):
            return Verdict(0, f"price {it.price} outside Betfair's {MIN_PRICE}-{MAX_PRICE}")
        if abs(math.log(it.price / it.best_price)) > lim["price_collar"]:
            return Verdict(0, f"price {it.price} outside the collar around {it.best_price}")
        now = self.clock()
        q = st.sent[s.id]
        while q and now - q[0] >= 60:
            q.popleft()
        if len(q) >= lim["max_orders_per_minute"]:
            return Verdict(0, f"order rate: {len(q)} in the last minute")

        size, why = it.size, ""

        def cut(x: float, reason: str) -> None:
            nonlocal size, why
            if x < size:
                size, why = max(x, 0.0), reason

        cut(lim["max_order"], "the order's cap")
        cut(lim["max_day_turnover"] - st.turnover[s.id], "the strategy's day turnover")
        if s.stage == "pilot" and env is not None:
            pilot = sum(st.turnover[i] for i, x in self.by_id.items() if x.stage == "pilot")
            cut(env.pilot_budget - pilot, "the pilot budget")
        # the day loss: settled today plus the worst case still open, for the strategy, its sport and the desk
        desk_ids = [i for i, x in self.by_id.items() if x.desk_run]

        def loss_so_far(ids) -> float:
            return -sum(st.settled[i] for i in ids) - _open_worst([p for i in ids for p in st.positions[i]])

        caps = [([s.id], lim["max_day_loss"], "the strategy's day loss")]
        if env is not None:
            caps += [(self._sport_ids(s.sport), env.sport_daily_loss(s.sport), f"{s.sport}'s day loss"),
                     (desk_ids, env.daily_loss, "the desk's day loss")]
        for ids, cap, name in caps:
            cut(_size_for_loss(it, cap - loss_so_far(ids)), name)
        # the market: every desk strategy's positions on it together, with this order
        mine = [p for p in st.positions[s.id] if p.market_id == it.market_id]
        here = [p for i in desk_ids for p in st.positions[i] if p.market_id == it.market_id]
        cut(_largest_fitting(it, mine, lim["max_market_exposure"], size), "the strategy's market exposure")
        if env is not None:
            cut(_largest_fitting(it, here, env.max_market_exposure, size), "the desk's market exposure")
        if st.funds_available is None:
            if not self.paper:
                return Verdict(0, "funds not read: no order")
        else:
            free = st.funds_available - (env.reserve if env is not None else 0.0)
            cut(_size_for_loss(it, free), "the funds, less the horse rule's reserve")
        size = _floor_pence(size) if why else it.size        # an order no limit cut is sent as asked
        if size < lim["min_stake"]:
            return Verdict(0, f"below the minimum stake after limits ({why or 'size'})")
        return Verdict(size, why)


def _largest_fitting(it: Intent, positions: list[Position], cap: float, upper: float) -> float:
    """The largest size (to the penny, at most ``upper``) whose market worst case with these positions stays within
    the cap. The worst case is monotone in the size, so a bisection finds it."""
    if -worst_case(positions + [it.position(upper)]) <= cap:
        return upper
    lo, hi = 0.0, upper
    if -worst_case(positions + [it.position(0.0)]) > cap:
        return 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if -worst_case(positions + [it.position(mid)]) <= cap:
            lo = mid
        else:
            hi = mid
    return lo
