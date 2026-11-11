"""The desk session: several desk strategies on one market, every intent through the risk kernel, on paper.

Built on the replay's session (research/arms/sessions.py ArmSession: the market list, the window, each race's runner
views and the closing model's view of a race), with the live rule's own backs switched off: in a replay the live rule
runs as its own session beside this one (research/arms/core.py run_agents, the mirror stepping first, so the desk
meets the book after the live trader's fills). Each minute:

1. the race books are read once; each strategy is shown each open market (``MarketView``) and what the desk already
   holds there (its own positions and every other desk strategy's);
2. every intent is revised by its strategy as it is sent (after the minute's earlier fills: the arms size each order
   that way) and goes to the kernel (``RiskKernel``, paper mode); what it allows is sent to the exchange under the
   strategy's own order reference (``<prefix><mmdd>-<market>-<selection>-<n>``, never ``ash``), the highest expected
   value first across the markets, as the arms do;
3. what matched goes back to the kernel (positions, turnover) and to the strategy (``on_fill``).

After the day, each position is settled at the BSP from Betfair's record (``_D.settled``): ``positions_frame`` gives
the rows ``research/arms/core.py score_side`` scores (closed at the BSP after the reduction factors and 2%).

Paper only: nothing here can reach Betfair. A live desk exchange is phase 2, behind the pilot stage.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from desk.risk import DeskState, RiskKernel
from desk.strategy import MarketView
from research.arms.sessions import ArmSession


class DeskSession(ArmSession):
    """Policy keys: algos ([desk.strategy.Strategy]), registry ([desk.registry.Strategy]: each algo's entry),
    envelope (desk.envelope.Envelope or None), _D (the day's record, for the settlement)."""

    def __init__(self, *a, policy=None, **k):
        super().__init__(*a, policy=policy, **k)
        p = self.policy
        self.algos = list(p.get("algos") or [])
        reg = {s.id: s for s in p.get("registry") or []}
        missing = [a.id for a in self.algos if a.id not in reg]
        if missing:
            raise ValueError(f"not in the registry: {missing}")
        self.prefix = {a.id: reg[a.id].order_ref_prefix for a in self.algos}
        self.kernel = RiskKernel(list(reg.values()), p.get("envelope"), state=DeskState(), paper=True,
                                 clock=lambda: self.clock().timestamp())
        self.desk_rows: list[dict] = []
        self.desk_positions: dict = {}                     # (strategy, market, selection, side) -> tally
        self._n = 0

    def execute(self, m, book, d, now):
        return                                             # the live rule's backs are its own session's

    def held(self, market_id: str) -> list:
        st = self.kernel.state
        return [p for sid in self.prefix for p in st.positions[sid] if p.market_id == market_id]

    def ref(self, strategy_id: str, market_id: str, selection_id: int) -> str:
        self._n += 1
        return f"{self.prefix[strategy_id]}{self.day:%m%d}-{market_id[-6:]}-{selection_id}-{self._n}"[:32]

    def step(self, now: datetime | None = None) -> None:
        now = now or self.clock()
        super().step(now)
        if not self.in_window(now) or not self.algos:
            return
        to_trade = self.tradable_markets(now)
        if not to_trade:
            return
        books = self.x.books([m.market_id for m in to_trade])
        funds = self.x.funds()
        self.kernel.state.funds_available = None if not funds else funds.get("available")
        for algo in self.algos:
            wanted = []
            for m in to_trade:
                b = books.get(m.market_id)
                if b is None or b.status != "OPEN" or b.inplay:
                    continue
                views = self.race_views(m, b)
                view = MarketView(market=m, book=b, runners=views, now=now,
                                  closing_fn=lambda v: self.closing(v, self.cfg.clv_draws))
                wanted += [(m, it) for it in algo.intents(view, self.held(m.market_id))]
            wanted.sort(key=lambda t: -(t[1].expected_value if t[1].expected_value is not None else -np.inf))
            for m, it in wanted:
                self.send(algo, m, it, now)

    def send(self, algo, m, it, now: datetime) -> None:
        it = algo.revise(it)                               # sized after the minute's earlier fills, as sent
        if it is None:
            return
        v = self.kernel.check(it)
        row = {"ts": self._ts(now), "event": "intent", "strategy": it.strategy_id, "market_id": it.market_id,
               "selection_id": it.selection_id, "side": it.side, "price": it.price, "asked": it.size,
               "allowed": v.size, "reason": v.reason, "ev": it.expected_value, "decision_price": it.decision_price}
        if not v.ok:
            self.desk_rows.append({**row, "status": "REFUSED"})
            return
        self.kernel.state.note_sent(it, now.timestamp())
        ref = self.ref(it.strategy_id, it.market_id, it.selection_id)
        order = self.x.back if it.side == "BACK" else self.x.lay
        fill = order(it.market_id, it.selection_id, it.price, v.size, ref=ref, min_fill=self.cfg.limits.min_stake)
        self.desk_rows.append({**row, "ref": ref, "matched": fill.matched, "avg_price": fill.avg_price,
                               "status": fill.status, "error": fill.error})
        if fill.matched > 0:
            self.kernel.state.record(it, fill.matched, fill.avg_price)
            algo.on_fill(it, fill.matched, fill.avg_price)
            key = (it.strategy_id, it.market_id, it.selection_id, it.side)
            pos = self.desk_positions.setdefault(key, {"matched": 0.0, "cost": 0.0, "fills": 0,
                                                       "first": self._ts(now)})
            pos["matched"] += fill.matched
            pos["cost"] += fill.matched * fill.avg_price
            pos["fills"] += 1

    def settle_extra(self, now: datetime) -> None:
        D = self.policy.get("_D")
        for (sid, mid, sel, side), pos in self.desk_positions.items():
            bsp, st = D.settled.get((mid, sel), (np.nan, "REMOVED")) if D is not None else (np.nan, "REMOVED")
            avg = pos["cost"] / pos["matched"]
            liab = pos["matched"] * (avg - 1.0) if side == "LAY" else pos["matched"]
            self.desk_rows.append({"ts": self._ts(now), "event": "settle", "strategy": sid, "market_id": mid,
                                   "selection_id": sel, "side": side, "matched": round(pos["matched"], 2),
                                   "avg_price": round(avg, 4), "liability": round(liab, 2), "bsp": bsp, "result": st,
                                   "fills": pos["fills"], "first": pos["first"]})

    def positions_frame(self, strategy_id: str, side: str) -> pd.DataFrame:
        """One strategy's settled positions on one side, in the columns score_side reads."""
        rows = [r for r in self.desk_rows if r["event"] == "settle" and r["strategy"] == strategy_id
                and r["side"] == side]
        if not rows:
            return pd.DataFrame(columns=["market_id", "selection_id", "ts", "matched", "avg_price", "bsp", "result",
                                         "liability"])
        f = pd.DataFrame(rows)
        f["ts"] = f["first"]
        return f
