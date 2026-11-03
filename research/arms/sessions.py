"""The sessions of arms 1, 2 and 5: the live Session with a policy on top (read-only research).

ArmSession (arm 1): the rule as live, with a stake floor on long prices, a bar that rises with the clock, or a stop
after a UK time; the top-up cap is the config's. LaySession (arm 2): the rule's mirror, laying the horses the closing
model has shorter than the market (expected CLV of the lay ``1 - price / E[BSP]`` at least the bar, never at a price
over the model's own), GBP400 at risk a horse down the lay levels, closed by a back at the BSP. RaceBookSession
(arm 5): model/race_book.py's Kelly book at the close on every runner of a race, once a race, filled against the
sizes offered.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from model import race_book as rb
from research.arms.core import UK, hours_uk
from trading.session import Session, _hhmm
from trading.strategy import RunnerView, _closing_model, _market_price

LAY_LIABILITY = 400.0            # the mirror of "to win GBP400": GBP400 at risk on each lay


class ArmSession(Session):
    """Policy keys: floor_long (price, floor), bar_slope (CLV points an hour from trade_from), stop_after ('HH:MM' UK),
    _D (the day's record, for the arms that settle their own positions)."""

    def __init__(self, *a, policy=None, **k):
        super().__init__(*a, **k)
        self.policy = policy or {}
        self.base_bar = float(self.cfg.clv_bar)

    def bar_now(self, now: datetime) -> float:
        slope = self.policy.get("bar_slope")
        if not slope:
            return self.base_bar
        return self.base_bar + float(slope) * max(0.0, hours_uk(now, self.cfg.trade_from))

    def step(self, now=None):
        now = now or self.clock()
        self.cfg.clv_bar = self.bar_now(now)                # the ladder's levels read the bar too
        super().step(now)

    def execute(self, m, book, d, now):
        pol = self.policy
        stop = pol.get("stop_after")
        if stop and hours_uk(now, stop) >= 0:
            return
        if np.isfinite(d.edge) and d.edge < self.cfg.clv_bar - 1e-12:
            return
        fl = pol.get("floor_long")
        if fl and d.price and d.price >= fl[0]:
            d = replace(d, target=max(d.target, min(float(fl[1]), self.cfg.limits.max_stake)))
        super().execute(m, book, d, now)

    def _ts(self, now: datetime) -> str:
        return now.strftime("%Y-%m-%dT%H:%M:%SZ")

    def in_window(self, now: datetime) -> bool:
        uk = now.astimezone(UK)
        mins = uk.hour * 60 + uk.minute
        return _hhmm(self.cfg.trade_from) <= mins <= _hhmm(self.cfg.trade_until) and not self.state.stopped

    def tradable_markets(self, now: datetime) -> list:
        return [m for m in self.markets.values() if m.start > now
                and (m.start - now) > timedelta(minutes=self.cfg.stop_before_off)]

    def race_views(self, m, book) -> list[RunnerView]:
        views = []
        for sid, q in book.runners.items():
            if q.status != "ACTIVE":
                continue
            bb, bl = q.best_back, q.best_lay
            views.append(RunnerView(selection_id=sid, name=m.runners.get(sid, str(sid)),
                                    model_price=self.prices.get((m.market_id, sid)),
                                    back=bb[0] if bb else None, back_size=bb[1] if bb else 0.0,
                                    lay=bl[0] if bl else None, last_traded=q.last_traded, status=q.status,
                                    traded=q.traded))
        return views

    def closing(self, views: list[RunnerView], n_draws: int):
        """The closing model's view of one whole race: (market now, model price, q_hat, sigma, E[1/BSP]) or None
        when a runner lacks a price (the model's or the market's)."""
        if len(views) < 2 or any(not (v.model_price and v.model_price > 1) for v in views):
            return None
        fallback = [_market_price(v) for v in views]
        if any(p is None for p in fallback):
            return None
        back = np.array([v.back if (v.back and v.back > 1) else np.nan for v in views], float)
        lay = np.array([v.lay if (v.lay and v.lay > 1) else np.nan for v in views], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            market = np.where(np.isfinite(back), rb.market_now(back, lay), np.array(fallback, float))
        model = _closing_model(self.cfg.closing_model_novol)
        model_price = np.array([v.model_price for v in views], float)
        d = pd.DataFrame({"race": 0, "morningwap": market, "predicted_bfsp": model_price, "morning_vol": 0.0})
        inputs = rb.closing_inputs(d)
        g = np.zeros(len(d), int)
        q_hat, sigma = model.predict(inputs, g), model.sigma(inputs)
        rng = np.random.default_rng(sum(int(v.selection_id) for v in views) % (2 ** 32))
        draws = rb.closing_draws(q_hat, sigma, book=model.book, n_draws=n_draws, rng=rng)
        return market, model_price, q_hat, sigma, draws


class LaySession(ArmSession):
    """Policy keys: lays_only (the rule's backs off, default True), lay_bar (0.03), max_lay_price (20),
    lay_day_liability (4000), lay_max_topups (5)."""

    def __init__(self, *a, policy=None, **k):
        super().__init__(*a, policy=policy, **k)
        self.lay_positions: dict = {}
        self.lay_liability = 0.0
        self.lay_rows: list[dict] = []
        self._lay_cache: dict = {}
        self.lays_only = bool(self.policy.get("lays_only", True))
        self.lay_bar = float(self.policy.get("lay_bar", 0.03))
        self.max_lay_price = float(self.policy.get("max_lay_price", 20.0))
        self.lay_day_liability = float(self.policy.get("lay_day_liability", 4000.0))
        self.lay_max_topups = self.policy.get("lay_max_topups", 5)

    def execute(self, m, book, d, now):
        if self.lays_only:
            return
        super().execute(m, book, d, now)

    def step(self, now=None):
        now = now or self.clock()
        super().step(now)
        if not self.in_window(now):
            return
        to_trade = self.tradable_markets(now)
        if not to_trade:
            return
        books = self.x.books([m.market_id for m in to_trade])
        plans = []
        for m in to_trade:
            b = books.get(m.market_id)
            if b is None or b.status != "OPEN" or b.inplay:
                continue
            plans.extend((m, b, c) for c in self.lay_plan(m, b))
        plans.sort(key=lambda t: -t[2]["ev"])
        for m, b, c in plans:
            self.lay_execute(m, b, c, now)

    def lay_plan(self, m, book) -> list[dict]:
        views = self.race_views(m, book)
        key = tuple((v.selection_id, v.back, v.lay, v.last_traded, v.model_price) for v in views)
        if key not in self._lay_cache:
            c = self.closing(views, self.cfg.clv_draws)
            self._lay_cache[key] = None if c is None else 1.0 / c[4].mean(axis=0)     # E[1/BSP] -> 1 / E[1/BSP]
        exp_bsp = self._lay_cache[key]
        if exp_bsp is None:
            return []
        out = []
        for i, v in enumerate(views):
            q = book.runners[v.selection_id]
            if not q.lay:
                continue
            L, size = q.lay[0]
            if not (1.01 <= L <= self.max_lay_price):
                continue
            ev = 1.0 - L / exp_bsp[i]
            if not np.isfinite(ev) or ev < self.lay_bar or L > v.model_price * (1 + 1e-9):
                continue
            out.append({"sid": v.selection_id, "name": v.name, "price": L, "size": size, "ev": float(ev),
                        "exp_bsp": float(exp_bsp[i]), "model_price": float(v.model_price), "levels": list(q.lay)})
        return out

    def lay_execute(self, m, book, c: dict, now: datetime) -> None:
        key = (m.market_id, c["sid"])
        pos = self.lay_positions.get(key)
        if pos and self.lay_max_topups is not None and pos["fills"] >= self.lay_max_topups + 1:
            return
        L = c["price"]
        target = min(max(LAY_LIABILITY / (L - 1.0), self.cfg.stake_floor), self.cfg.limits.max_stake)
        want = target - (pos["matched"] if pos else 0.0)
        if want < self.cfg.limits.min_stake:
            return
        limit, size = L, 0.0                                   # the mirror ladder: up the lay levels while they clear
        for p, s in c["levels"]:
            if 1.0 - p / c["exp_bsp"] < self.lay_bar or p > c["model_price"] or p > self.max_lay_price:
                break
            limit, size = p, size + s
        offered = size if size > 0 else c["size"]
        stake = math.floor(min(want, offered) * 100) / 100
        room = self.lay_day_liability - self.lay_liability
        if stake * (limit - 1.0) > room + 1e-9:
            stake = math.floor(max(0.0, room / (limit - 1.0)) * 100) / 100
        if stake < self.cfg.limits.min_stake:
            return
        fill = self.x.lay(m.market_id, c["sid"], limit, stake, ref="lay", min_fill=self.cfg.limits.min_stake)
        self.lay_rows.append({"ts": self._ts(now), "event": "lay", "market_id": m.market_id, "venue": m.venue,
                              "off": m.start.astimezone(UK).strftime("%H:%M"), "selection_id": c["sid"],
                              "runner": c["name"], "price": L, "limit": limit, "asked": stake,
                              "matched": fill.matched, "avg_price": fill.avg_price, "ev": round(c["ev"], 4),
                              "exp_bsp": round(c["exp_bsp"], 3), "model_price": c["model_price"],
                              "status": fill.status, "error": fill.error})
        if fill.matched > 0:
            pos = self.lay_positions.setdefault(key, {"matched": 0.0, "cost": 0.0, "fills": 0, "name": c["name"],
                                                      "first": self._ts(now)})
            pos["matched"] += fill.matched
            pos["cost"] += fill.matched * fill.avg_price
            pos["fills"] += 1
            self.lay_liability += fill.matched * (fill.avg_price - 1.0)

    def settle_extra(self, now: datetime) -> None:
        D = self.policy.get("_D")
        for (mid, sid), pos in self.lay_positions.items():
            bsp, st = D.settled.get((mid, sid), (np.nan, "REMOVED")) if D is not None else (np.nan, "REMOVED")
            avg = pos["cost"] / pos["matched"]
            self.lay_rows.append({"ts": self._ts(now), "event": "lay_settle", "market_id": mid, "selection_id": sid,
                                  "runner": pos["name"], "matched": round(pos["matched"], 2), "avg_price": round(avg, 4),
                                  "liability": round(pos["matched"] * (avg - 1.0), 2), "bsp": bsp, "result": st,
                                  "fills": pos["fills"], "first": pos["first"]})

    def lay_positions_frame(self) -> pd.DataFrame:
        rows = [r for r in self.lay_rows if r["event"] == "lay_settle"]
        if not rows:
            return pd.DataFrame(columns=["market_id", "selection_id", "ts", "matched", "avg_price", "bsp", "result",
                                         "liability"])
        f = pd.DataFrame(rows)
        f["ts"] = f["first"]
        return f


class RaceBookSession(ArmSession):
    """Policy keys: book_from ('HH:MM' UK, default trade_from), lays (bool), exposure (a race, GBP), stake_cap,
    theta, delta, draws, day_exposure, max_lay_price, _D."""

    def __init__(self, *a, policy=None, **k):
        super().__init__(*a, policy=policy, **k)
        p = self.policy
        self.book_from = str(p.get("book_from", self.cfg.trade_from))
        self.with_lays = bool(p.get("lays", False))
        self.exposure = float(p.get("exposure", 600.0))
        self.stake_cap = float(p.get("stake_cap", 200.0))
        self.theta = float(p.get("theta", 0.03))
        self.delta = float(p.get("delta", 0.03))
        self.draws = int(p.get("draws", 4000))
        self.day_exposure = float(p.get("day_exposure", 4000.0))
        self.max_lay_price = float(p.get("max_lay_price", 20.0))
        self.booked: set = set()
        self.positions_book: dict = {}
        self.book_rows: list[dict] = []
        self.exposure_used = 0.0

    def execute(self, m, book, d, now):
        return                                               # the rule's single backs are off: the book only

    def step(self, now=None):
        now = now or self.clock()
        if not self.in_window(now) or hours_uk(now, self.book_from) < 0:
            return
        to_trade = [m for m in self.tradable_markets(now) if m.market_id not in self.booked]
        if not to_trade:
            return
        books = self.x.books([m.market_id for m in to_trade])
        for m in to_trade:
            b = books.get(m.market_id)
            if b is None or b.status != "OPEN" or b.inplay:
                continue
            self.book_race(m, b, now)

    def book_race(self, m, book, now: datetime) -> None:
        views = self.race_views(m, book)
        c = self.closing(views, self.draws)
        if c is None:
            return
        market, model_price, q_hat, sigma, draws = c
        back = np.array([v.back if (v.back and v.back > 1) else np.nan for v in views], float)
        lay = np.array([v.lay if (v.lay and v.lay > 1) else np.nan for v in views], float)
        bsize = np.array([v.back_size for v in views], float)
        lsize = np.array([book.runners[v.selection_id].lay[0][1] if book.runners[v.selection_id].lay else 0.0
                          for v in views], float)
        allow_back = np.isfinite(back) & (bsize >= self.cfg.limits.min_stake) & (back <= self.cfg.limits.max_price)
        allow_lay = (np.isfinite(lay) & (lsize >= self.cfg.limits.min_stake) & (lay <= self.max_lay_price)
                     if self.with_lays else np.zeros(len(views), bool))
        if allow_back.sum() + allow_lay.sum() == 0 or allow_back.sum() < 0.5 * len(views):
            return                                            # a book that is mostly unpriced waits
        self.booked.add(m.market_id)
        room = self.day_exposure - self.exposure_used
        if room < self.cfg.limits.min_stake:
            return
        back_f, lay_f = np.nan_to_num(back, nan=1.0), np.nan_to_num(lay, nan=1.0)
        payoffs = rb.close_payoffs(draws, back_f, lay_f)
        w = np.full(len(draws), 1.0 / len(draws))
        bk = rb.build_book(payoffs, w, lay_f, delta=self.delta, theta=self.theta,
                           exposure_cap=min(self.exposure, room), stake_cap=self.stake_cap,
                           allow_back=allow_back, allow_lay=allow_lay, commission=self.cfg.commission)
        for i, v in enumerate(views):
            sid = v.selection_id
            if bk.back[i] >= self.cfg.limits.min_stake and allow_back[i]:
                stake = math.floor(min(bk.back[i], bsize[i]) * 100) / 100
                f = self.x.back(m.market_id, sid, back[i], stake, ref="book", min_fill=self.cfg.limits.min_stake)
                self._record(m, v, now, "back", f, bk.edges[i], q_hat[i])
            if bk.lay[i] >= self.cfg.limits.min_stake and allow_lay[i]:
                stake = math.floor(min(bk.lay[i], lsize[i]) * 100) / 100
                f = self.x.lay(m.market_id, sid, lay[i], stake, ref="book", min_fill=self.cfg.limits.min_stake)
                self._record(m, v, now, "lay", f, bk.edges[len(views) + i], q_hat[i])

    def _record(self, m, v, now, side, f, edge, q_hat) -> None:
        self.book_rows.append({"ts": self._ts(now), "event": f"book_{side}", "market_id": m.market_id,
                               "venue": m.venue, "off": m.start.astimezone(UK).strftime("%H:%M"),
                               "selection_id": v.selection_id, "runner": v.name, "price": f.price, "asked": f.size,
                               "matched": f.matched, "avg_price": f.avg_price, "edge": round(float(edge), 4),
                               "q_hat": round(float(q_hat), 4), "model_price": v.model_price, "status": f.status,
                               "error": f.error})
        if f.matched > 0:
            key = (m.market_id, v.selection_id, side)
            pos = self.positions_book.setdefault(key, {"matched": 0.0, "cost": 0.0, "name": v.name,
                                                       "first": self._ts(now)})
            pos["matched"] += f.matched
            pos["cost"] += f.matched * f.avg_price
            self.exposure_used += f.matched if side == "back" else f.matched * (f.avg_price - 1.0)

    def settle_extra(self, now: datetime) -> None:
        D = self.policy.get("_D")
        for (mid, sid, side), pos in self.positions_book.items():
            bsp, st = D.settled.get((mid, sid), (np.nan, "REMOVED")) if D is not None else (np.nan, "REMOVED")
            avg = pos["cost"] / pos["matched"]
            self.book_rows.append({"ts": pos["first"], "event": "book_settle", "market_id": mid, "selection_id": sid,
                                   "side": side, "runner": pos["name"], "matched": round(pos["matched"], 2),
                                   "avg_price": round(avg, 4), "bsp": bsp, "result": st,
                                   "liability": round(pos["matched"] * (avg - 1.0), 2) if side == "lay"
                                   else round(pos["matched"], 2)})

    def book_positions_frame(self) -> pd.DataFrame:
        rows = [r for r in self.book_rows if r["event"] == "book_settle"]
        cols = ["market_id", "selection_id", "side", "ts", "matched", "avg_price", "bsp", "result", "liability"]
        return pd.DataFrame(rows, columns=cols) if not rows else pd.DataFrame(rows)[cols]


def score_book(rows: pd.DataFrame, D, name: str) -> dict:
    """A race book closed at the BSP: the hedge's value at the prices matched, 2% on each race's net, and the
    same held to the result (for information: twenty days of results are noise)."""
    base = {"day": str(D.day), "arm": name, "races": 0, "positions": 0, "backs": 0, "lays": 0, "exposure": 0.0,
            "clv_x_stake": 0.0, "result": 0.0, "result_held": 0.0, "clv_on_exposure": None}
    if rows.empty:
        return base
    r = rows[rows.result.ne("REMOVED") & (rows.bsp > 1) & (rows.matched > 0)].copy()
    if r.empty:
        return base
    sign = np.where(r.side.eq("back"), 1.0, -1.0)
    r["hedge"] = sign * r.matched * (r.avg_price / r.bsp - 1.0)
    won = r.result.eq("WINNER")
    r["held"] = np.where(r.side.eq("back"), np.where(won, r.matched * (r.avg_price - 1.0), -r.matched),
                         np.where(won, -r.matched * (r.avg_price - 1.0), r.matched))
    by = r.groupby("market_id")[["hedge", "held"]].sum()
    res = float((by.hedge - 0.02 * by.hedge.clip(lower=0)).sum())
    held = float((by.held - 0.02 * by.held.clip(lower=0)).sum())
    expo = float(r.liability.sum())
    return {**base, "races": int(by.shape[0]), "positions": int(len(r)), "backs": int(r.side.eq("back").sum()),
            "lays": int(r.side.eq("lay").sum()), "exposure": round(expo, 2),
            "clv_x_stake": round(float(r.hedge.sum()), 2), "result": round(res, 2), "result_held": round(held, 2),
            "clv_on_exposure": round(float(r.hedge.sum() / expo), 4) if expo > 0 else None}
