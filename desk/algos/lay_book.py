"""horse-lay-book: lays where the closing model expects the horse to drift to the SP, closed by a back at the SP.

The rule of the lay arm (research/arms/sessions.py LaySession, TRADING.md:1787-1790: +GBP827 over 21 days on
GBP75,509 of liability, 15 of 21 days up), ported as a desk strategy so it runs beside the live rule through the risk
kernel. Its choices are the arm's, unchanged, so the desk's replay can be checked against the arm's own:

- the expected CLV of a lay at the best lay price L is ``1 - L / E[BSP]`` (E[BSP] = 1 / E[1/BSP] from the closing
  model's draws); a lay needs it at least ``bar`` and L no longer than the model's own price;
- ``liability`` at risk a horse (the mirror of the rule's to-win stake), the stake at least ``stake_floor`` and at
  most ``max_stake``, topped up to that target at most ``max_topups`` times;
- the order walks up the lay levels while each still clears the bar, the model's price and ``max_price``;
- ``day_liability`` across the day.
"""

from __future__ import annotations

import math

import numpy as np

from desk.risk import Intent
from desk.strategy import MarketView, Strategy

DEFAULTS = {"bar": 0.03, "max_price": 20.0, "liability": 400.0, "stake_floor": 15.0, "max_stake": 200.0,
            "min_stake": 2.0, "max_topups": 5, "day_liability": 4000.0}


class LayBook(Strategy):
    id = "horse-lay-book"

    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}")
        self.p = {**DEFAULTS, **params}
        self.day_liability_used = 0.0
        self.matched: dict = {}                          # (market, selection) -> GBP matched
        self.fills: dict = {}                            # (market, selection) -> fills

    def candidates(self, view: MarketView) -> list[dict]:
        c = view.closing()
        if c is None:
            return []
        exp_bsp = 1.0 / c[4].mean(axis=0)                # E[1/BSP] -> 1 / E[1/BSP], as the arm
        out = []
        for i, v in enumerate(view.runners):
            q = view.book.runners[v.selection_id]
            if not q.lay:
                continue
            L, size = q.lay[0]
            if not (1.01 <= L <= self.p["max_price"]):
                continue
            ev = 1.0 - L / exp_bsp[i]
            if not np.isfinite(ev) or ev < self.p["bar"] or L > v.model_price * (1 + 1e-9):
                continue
            out.append({"sid": v.selection_id, "price": L, "size": size, "ev": float(ev),
                        "exp_bsp": float(exp_bsp[i]), "model_price": float(v.model_price), "levels": list(q.lay)})
        out.sort(key=lambda r: -r["ev"])
        return out

    def intents(self, view: MarketView, held) -> list[Intent]:
        p, out = self.p, []
        for c in self.candidates(view):
            key = (view.market_id, c["sid"])
            if p["max_topups"] is not None and self.fills.get(key, 0) >= p["max_topups"] + 1:
                continue
            L = c["price"]
            target = min(max(p["liability"] / (L - 1.0), p["stake_floor"]), p["max_stake"])
            want = target - self.matched.get(key, 0.0)
            if want < p["min_stake"]:
                continue
            limit, size = L, 0.0
            for lp, s in c["levels"]:
                if 1.0 - lp / c["exp_bsp"] < p["bar"] or lp > c["model_price"] or lp > p["max_price"]:
                    break
                limit, size = lp, size + s
            offered = size if size > 0 else c["size"]
            stake = math.floor(min(want, offered) * 100) / 100
            room = p["day_liability"] - self.day_liability_used
            if stake * (limit - 1.0) > room + 1e-9:
                stake = math.floor(max(0.0, room / (limit - 1.0)) * 100) / 100
            if stake < p["min_stake"]:
                continue
            out.append(Intent(strategy_id=self.id, market_id=view.market_id, selection_id=int(c["sid"]), side="LAY",
                              price=float(limit), size=float(stake), best_price=float(L), book_age_s=view.book_age_s,
                              decision_price=float(L), expected_value=c["ev"]))
        return out

    def on_fill(self, intent: Intent, matched: float, avg_price: float) -> None:
        if matched <= 0:
            return
        key = (intent.market_id, intent.selection_id)
        self.matched[key] = self.matched.get(key, 0.0) + matched
        self.fills[key] = self.fills.get(key, 0) + 1
        self.day_liability_used += matched * (avg_price - 1.0)
