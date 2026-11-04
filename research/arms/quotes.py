"""Both sides of the line (the owner, 4 Nov 2026: "paper trade ... High Volume Trading each side of the line"): a
resting back above a fair price and a resting lay below it on every runner with a two-sided book, requoted each
minute, every fill closed at the Betfair SP. Read-only research on the recorded books (research/arms/core.py).

- **The fair price:** ``model``: the closing model's expected BSP, 1 / E[1/BSP] (the live rule's own view, from the
  market's price now and ours); ``mid``: the market's own mid now (model.race_book.market_now), no model at all.
- **The quotes:** a back at ``fair x (1 + margin)`` rounded up the ladder, a lay at ``fair x (1 - margin)`` rounded
  down, ``stake`` a quote, at most ``max_side`` matched a runner a side, ``day_matched`` a day across both sides; no
  lay over ``max_lay_price``, no back over the config's max_price; from the config's trade_from to 15 minutes before
  each off (the live window). A quote that crosses the book takes what is offered at it or better at once.
- **The fills, without a traded ladder** (the horse books carry each runner's total matched and last traded price a
  minute apart, not the money traded at each price): a resting back at P (offered to layers) is matched in full once
  the market trades above P or offers to lay at P or more; at P itself it is matched by half the money traded since
  the last read beyond the queue that was ahead of it (the size shown at P when placed; deeper than the book shows:
  never at P). A lay mirrors it. A requote at a new price starts at the back of the new queue. Conservative: trades
  at other prices between two reads may be counted at ours only through that half.
- **The exit:** each fill hedged at the SP when matched (a back by a lay of stake x price / BSP, and the mirror), so a
  fill locks stake x (price / BSP - 1) for a back and stake x (1 - price / BSP) for a lay; a runner taken out after a
  fill cuts its winners' matched price by the reduction factor (Betfair's rule), as the arms score; 2% on each
  market's net winnings.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from model import race_book as rb
from research.arms.core import UK
from research.arms.sessions import ArmSession
from trading.exchange import tick_down, tick_up

SIDES = ("back", "lay")


class QuoteSession(ArmSession):
    """Policy keys: fair ('model' | 'mid'), margin (0.03), stake (10), max_side (100), day_matched (4000),
    max_lay_price (50), queue_share (0.5: the share of the money traded at our price counted ours)."""

    def __init__(self, *a, policy=None, **k):
        super().__init__(*a, policy=policy, **k)
        p = self.policy
        self.fair = str(p.get("fair", "model"))
        self.margin = float(p.get("margin", 0.03))
        self.stake = float(p.get("stake", 10.0))
        self.max_side = float(p.get("max_side", 100.0))
        self.day_matched = float(p.get("day_matched", 4000.0))
        self.max_lay_price = float(p.get("max_lay_price", 50.0))
        self.queue_share = float(p.get("queue_share", 0.5))
        self.resting: dict = {}          # (market, selection, side) -> order
        self.matched: dict = {}          # (market, selection, side) -> GBP matched
        self.quote_rows: list[dict] = []
        self.total = 0.0
        self._fair_cache: dict = {}
        self._mid = None

    def execute(self, m, book, d, now):          # the rule's own backs are off: quotes only
        return

    # -- the fair prices --------------------------------------------------------------------------------------------
    def fair_prices(self, m, book) -> dict:
        views = self.race_views(m, book)
        if self.fair == "mid":
            out = {}
            for v in views:
                if v.back and v.lay and v.back > 1 and v.lay > 1:
                    out[v.selection_id] = float(rb.market_now(np.array([v.back]), np.array([v.lay]))[0])
            return out
        key = tuple((v.selection_id, v.back, v.lay, v.last_traded, v.model_price) for v in views)
        if key not in self._fair_cache:
            c = self.closing(views, self.cfg.clv_draws)
            self._fair_cache[key] = None if c is None else {
                v.selection_id: float(e) for v, e in zip(views, 1.0 / c[4].mean(axis=0))}
        return self._fair_cache[key] or {}

    # -- the orders -------------------------------------------------------------------------------------------------
    def _room(self, key) -> float:
        return max(0.0, min(self.max_side - self.matched.get(key, 0.0), self.day_matched - self.total))

    def _record(self, now, m, sid, side, price, size, how) -> None:
        if size <= 0:
            return
        key = (m.market_id, sid, side)
        self.matched[key] = self.matched.get(key, 0.0) + size
        self.total += size
        self.quote_rows.append({"ts": self._ts(now), "event": "fill", "market_id": m.market_id, "venue": m.venue,
                                "off": m.start.astimezone(UK).strftime("%H:%M"), "selection_id": sid,
                                "runner": m.runners.get(sid, str(sid)), "side": side, "price": price,
                                "matched": round(size, 2), "how": how})

    def _queue(self, side: str, price: float, q) -> float:
        """The money ahead of a new resting order: a back rests among the offers to layers (q.lay), a lay among
        the offers to backers (q.back)."""
        offers = q.lay if side == "back" else q.back
        ahead = self._others_resting(side, price, q)            # other agents' quotes at this price came first
        if not offers:
            return ahead
        best = offers[0][0]
        if (side == "back" and price < best - 1e-9) or (side == "lay" and price > best + 1e-9):
            return ahead                                        # it improves the best offer shown: behind the agents'
        for p, s in offers:
            if abs(p - price) < 1e-9:
                return float(s) + ahead
        return math.inf                                         # deeper than the book shows

    # -- the shared book: other agents' resting quotes and takes (research/arms/core.py SharedReplayData) -----------
    def _shared(self):
        return getattr(self.x.data, "resting", None)

    def _others_resting(self, side, price, q) -> float:
        r = self._shared()
        return float(r.get((self._mid, q.selection_id, side, round(price, 2)), 0.0)) if r is not None else 0.0

    def _rest(self, key, o, sign: float) -> None:
        r = self._shared()
        if r is not None:
            r[(key[0], key[1], key[2], round(o["price"], 2))] += sign * o["left"]

    def _drop(self, key) -> None:
        o = self.resting.pop(key, None)
        if o is not None:
            self._rest(key, o, -1.0)

    def _take_shown(self, m, sid, side, P, levels, amount) -> None:
        """A resting quote matched by an offer across it takes that offer from the shared book."""
        data = self.x.data
        left = amount
        for p, s in levels:
            if left <= 1e-9:
                break
            take = min(s, left)
            if side == "back" and hasattr(data, "agent_taken"):
                data.agent_taken[(m.market_id, sid, round(p, 2))] += take
            elif side == "lay":
                getattr(data, "agent_taken_lay", data.taken_lay)[(m.market_id, sid, round(p, 2))] += take
            left -= take

    def _advance(self, now, m, sid, q) -> None:
        """What each resting order on this runner matched since the last read."""
        for side in SIDES:
            key = (m.market_id, sid, side)
            o = self.resting.get(key)
            if o is None:
                continue
            want = min(o["left"], self._room(key))
            if want <= 0:
                self._drop(key)
                continue
            lt = q.last_traded
            delta = max(0.0, (q.traded or 0.0) - o["traded"])
            P = o["price"]
            got, how = 0.0, ""
            if side == "back":
                crossing = [(p, s) for p, s in q.back if p >= P - 1e-9]   # someone now offers to lay at P or more
                through = lt is not None and lt > P + 1e-9 and delta > 0
            else:
                crossing = [(p, s) for p, s in q.lay if p <= P + 1e-9]    # someone now offers to back at P or less
                through = lt is not None and lt < P - 1e-9 and delta > 0
            cross = sum(s for _, s in crossing)
            if through:
                got, how = want, "traded through"
            elif cross > 0:
                got, how = min(want, cross), "crossed"
                self._take_shown(m, sid, side, P, crossing, got)
            elif lt is not None and abs(lt - P) < 1e-9 and delta > 0:
                ours = self.queue_share * delta
                got = min(want, max(0.0, ours - o["queue"]))
                o["queue"] = max(0.0, o["queue"] - ours)
                how = "at our price"
            o["traded"] = q.traded or 0.0
            if got >= 0.01:
                got = math.floor(got * 100) / 100
                self._record(now, m, sid, side, P, got, how)
                r = self._shared()
                if r is not None:
                    r[(key[0], key[1], key[2], round(P, 2))] -= got
                o["left"] -= got
            if o["left"] < self.cfg.limits.min_stake:
                self._drop(key)

    def _place(self, now, m, sid, side, price, q) -> None:
        key = (m.market_id, sid, side)
        o = self.resting.get(key)
        if o is not None and abs(o["price"] - price) < 1e-9:
            return                                              # the same price: keep the place in the queue
        self._drop(key)
        size = min(self.stake, self._room(key))
        if size < self.cfg.limits.min_stake:
            return
        # a quote that crosses the book takes what is offered at it or better at once
        if side == "back" and q.back and q.back[0][0] >= price - 1e-9:
            f = self.x.back(m.market_id, sid, price, round(size, 2), ref="qb", min_fill=self.cfg.limits.min_stake)
            if f.matched > 0:
                self._record(now, m, sid, "back", f.avg_price, f.matched, "taken")
            return
        if side == "lay" and q.lay and q.lay[0][0] <= price + 1e-9:
            f = self.x.lay(m.market_id, sid, price, round(size, 2), ref="ql", min_fill=self.cfg.limits.min_stake)
            if f.matched > 0:
                self._record(now, m, sid, "lay", f.avg_price, f.matched, "taken")
            return
        self._mid = m.market_id
        self.resting[key] = {"price": price, "left": size, "queue": self._queue(side, price, q),
                             "traded": q.traded or 0.0, "placed": now}
        self._rest(key, self.resting[key], +1.0)

    def step(self, now=None):
        now = now or self.clock()
        super().step(now)
        live = {m.market_id for m in self.tradable_markets(now)} if self.in_window(now) else set()
        for key in [k for k in self.resting if k[0] not in live]:
            self._drop(key)                                     # 15 minutes before the off: the quotes come off
        if not live:
            return
        books = self.x.books(sorted(live))
        for m in self.tradable_markets(now):
            b = books.get(m.market_id)
            if b is None or b.status != "OPEN" or b.inplay:
                continue
            fair = self.fair_prices(m, b)
            for sid, q in b.runners.items():
                if q.status != "ACTIVE" or not q.back or not q.lay:
                    continue
                self._advance(now, m, sid, q)
                F = fair.get(sid)
                if not F or not np.isfinite(F) or F <= 1.0:
                    continue
                pb = tick_up(F * (1.0 + self.margin))
                pl = tick_down(F * (1.0 - self.margin))
                if pb <= self.cfg.limits.max_price:
                    self._place(now, m, sid, "back", pb, q)
                if 1.01 <= pl <= self.max_lay_price and pl < pb:
                    self._place(now, m, sid, "lay", pl, q)

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.quote_rows)


def fill_pnl(fills: pd.DataFrame, D) -> pd.DataFrame:
    """Each fill (ts, market_id, selection_id, side 'back' | 'lay', price, matched) hedged at the SP when matched:
    its BSP and result, the CLV (a back's price / BSP - 1, a lay's 1 - price / BSP), the P&L by the result, and the
    same after the reductions of the runners taken out after it (Betfair's rule: a winner's matched price is cut).
    A runner that did not run (no BSP) is left out: void."""
    if fills is None or fills.empty:
        return pd.DataFrame()
    f = fills.copy()
    f["selection_id"] = f.selection_id.astype(int)
    f["bsp"] = [D.settled.get((m, int(s)), (np.nan, "REMOVED"))[0] for m, s in zip(f.market_id, f.selection_id)]
    f["res"] = [D.settled.get((m, int(s)), (np.nan, "REMOVED"))[1] for m, s in zip(f.market_id, f.selection_id)]
    f["bsp"] = pd.to_numeric(f.bsp, errors="coerce")
    f = f[np.isfinite(f.bsp) & (f.bsp > 1)].copy()
    if f.empty:
        return f
    t = pd.to_datetime(f.ts, utc=True)
    rem = D.removed
    fac = []
    for r, tt in zip(f.itertuples(), t):
        x = rem[(rem.market_id == r.market_id) & (rem.selection_id != r.selection_id) & (rem.rt > tt)] \
            if len(rem) else rem
        fac.append(float(np.prod(1 - x.af.to_numpy() / 100.0)) if len(x) else 1.0)
    P, s, B = f.price.to_numpy(float), f.matched.to_numpy(float), f.bsp.to_numpy(float)
    Pr = np.maximum(1.01, P * np.array(fac))
    won = f.res.eq("WINNER").to_numpy()
    h = s * P / B                                               # the hedge's stake at the SP
    back = f.side.eq("back").to_numpy()
    f["pnl"] = np.where(back, np.where(won, s * (P - 1) - h * (B - 1), -s + h),
                        np.where(won, -s * (P - 1) + h * (B - 1), s - h))
    f["pnl_red"] = f["pnl"] + np.where(won, np.where(back, -s * (P - Pr), s * (P - Pr)), 0.0)
    f["clv"] = np.where(back, P / B - 1, 1 - P / B)
    f["locked"] = s * f["clv"]
    return f


def net_of_commission(v: pd.Series) -> pd.Series:
    """2% on each market's net winnings (Betfair's commission), a loss untouched."""
    return v - 0.02 * v.clip(lower=0)


def score_quotes(fills: pd.DataFrame, D, name: str) -> dict:
    """Every fill hedged at the SP, after the reductions of the runners taken out after it, 2% on each market's net
    winnings. A runner that did not run (no BSP) is void."""
    out = {"day": str(D.day), "arm": name, "fills": 0, "runners": 0, "back_staked": 0.0, "lay_staked": 0.0,
           "lay_liability": 0.0, "locked": 0.0, "result": 0.0, "result_reduced": 0.0, "two_sided": 0}
    f = fill_pnl(fills, D)
    if f.empty:
        return out
    back = f.side.eq("back").to_numpy()
    net = f.groupby("market_id")[["pnl", "pnl_red"]].sum()
    sides = f.groupby(["market_id", "selection_id"]).side.nunique()
    out.update({"fills": int(len(f)), "runners": int(f.groupby(["market_id", "selection_id"]).ngroups),
                "back_staked": round(float(f.loc[back, "matched"].sum()), 2),
                "lay_staked": round(float(f.loc[~back, "matched"].sum()), 2),
                "lay_liability": round(float((f.loc[~back, "matched"] * (f.loc[~back, "price"] - 1)).sum()), 2),
                "locked": round(float(f.locked.sum()), 2),
                "result": round(float(net_of_commission(net.pnl).sum()), 2),
                "result_reduced": round(float(net_of_commission(net.pnl_red).sum()), 2),
                "two_sided": int((sides > 1).sum())})
    staked = out["back_staked"] + out["lay_staked"]
    out["clv"] = round(out["locked"] / staked, 4) if staked else None
    return out
