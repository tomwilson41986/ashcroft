"""A trading day: read the markets, decide, check the limits, place or simulate, close at BSP, settle.

One `Session` runs one UK racing day against an exchange (trading/exchange.py): the paper exchange reads
Betfair's live prices and simulates every order, the live one places them. Each `step` reads the books of
the races still ahead, plans each race (trading/strategy.py), backs the difference between the plan and
what is already matched, subject to the limits (trading/risk.py), closes each fill with a lay at BSP when
`trade_out` is on, and settles the races that have finished: live, from Betfair's record of the settled
bets (what it paid on each back and each lay at SP, and the SP the lays matched at); on paper, from the
closed market (winner, loser or non-runner, and the BSP). A race whose lays at SP cannot yet be priced
waits: a lay is never counted as nothing for want of its price. Every order and every settlement is a
row in the ledger: the forward test of the edge at the prices actually on offer.
"""

from __future__ import annotations

import csv
import logging
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading.exchange import Book, Market, uk_day_window
from trading.risk import DayState, allowed_stake
from trading.strategy import RunnerView, plan_race

log = logging.getLogger(__name__)

LEDGER_FIELDS = [
    "ts", "mode", "event", "market_id", "venue", "off", "minutes_to_off", "selection_id", "runner",
    "strategy", "staking", "reason", "p_model", "p_market", "p_pool", "edge", "move", "best_back", "back_size",
    "target", "asked", "matched", "avg_price", "bet_id", "status", "error", "hedged", "hedge_liability",
    "result", "bsp", "clv", "pnl_back", "pnl_lay", "pnl", "commission",
]
MAX_HEDGE_TRIES = 5
#: the reason on a live settlement taken from Betfair's record of the bets (listClearedOrders)
SETTLED_BY_BETFAIR = "Betfair's record of the settled bets"


def _num(x) -> float:
    """A ledger value as a number (a restored ledger holds strings; a blank is NaN)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _num0(x) -> float:
    """A number from Betfair's record, a missing one as 0."""
    v = _num(x)
    return v if np.isfinite(v) else 0.0


@dataclass
class Position:
    market_id: str
    selection_id: int
    name: str
    matched: float = 0.0
    cost: float = 0.0                    # sum of matched stake x price
    hedged: bool = False
    hedge_liability: float = 0.0
    hedge_failures: int = 0              # lays at BSP refused; after MAX_HEDGE_TRIES the position is left to settle
    settled: bool = False

    @property
    def avg_price(self) -> float:
        return self.cost / self.matched if self.matched > 0 else 0.0


def _uk(dt: datetime):
    from zoneinfo import ZoneInfo
    return dt.astimezone(ZoneInfo("Europe/London"))


def _hhmm(s: str):
    h, m = str(s).split(":")
    return int(h) * 60 + int(m)


class Session:
    def __init__(self, exchange, predictions: pd.DataFrame, cfg, day: date, *, kill=None, clock=None,
                 sleep=None, ledger_path: str | None = None, on_flush=None):
        self.x = exchange
        self.cfg = cfg
        self.day = day
        self.mode = getattr(exchange, "mode", "paper")     # live: the orders are real (trading/exchange.py)
        self.on_flush = on_flush                           # called after each write of the ledger (live: to S3)
        self.kill = kill or (lambda: False)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep or _time.sleep
        self.ledger_path = ledger_path
        self.state = DayState()
        self.positions: dict[tuple[str, int], Position] = {}
        self.ledger: list[dict] = []
        self.markets: dict[str, Market] = {}
        self._skips: set = set()
        p = predictions.dropna(subset=["market_id", "selection_id"]).copy()
        p["market_id"] = p["market_id"].astype(str)
        p["selection_id"] = p["selection_id"].astype("int64")
        self.prices = {(r.market_id, int(r.selection_id)): float(r.predicted_bfsp)
                       for r in p.itertuples() if getattr(r, "predicted_bfsp", 0) and r.predicted_bfsp > 1}

    # ------------------------------------------------------------------ the day

    def load_markets(self) -> None:
        start, end = uk_day_window(self.day)
        ms = self.x.markets(start, end, self.cfg.countries)
        wanted = {mid for mid, _ in self.prices}
        self.markets = {m.market_id: m for m in ms if m.market_id in wanted}
        log.info("%d of %d markets today carry the model's prices", len(self.markets), len(ms))

    def bank(self) -> float:
        return self.cfg.limits.bank + self.state.settled_pnl

    def step(self, now: datetime | None = None) -> None:
        now = now or self.clock()
        if not self.state.stopped and self.kill():
            self.state.stopped = "kill switch"
            log.warning("Kill switch: no more bets today; cancelling anything unmatched")
            self.x.cancel_all()
        uk = _uk(now)
        mins = uk.hour * 60 + uk.minute
        in_window = _hhmm(self.cfg.trade_from) <= mins <= _hhmm(self.cfg.trade_until)
        ahead = {mid: m for mid, m in self.markets.items() if m.start > now}
        to_trade = [m for m in ahead.values()
                    if in_window and not self.state.stopped
                    and (m.start - now) > timedelta(minutes=self.cfg.stop_before_off)]
        to_hedge = [m for m in ahead.values() if self.cfg.trade_out and self.cfg.trade_out_at == "before_off"
                    and (m.start - now) <= timedelta(minutes=self.cfg.trade_out_before_off)
                    and any(p.matched > 0 and not p.hedged for (mid, _), p in self.positions.items()
                            if mid == m.market_id)]
        # closing on the fill: a lay at BSP that did not go through (or waited for a bigger liability) is sent again
        to_hedge += [m for m in ahead.values() if self.cfg.trade_out and self.cfg.trade_out_at == "fill"
                     and any(p.matched > 0 and not p.hedged and p.hedge_failures < MAX_HEDGE_TRIES
                             and self._unhedged(p) >= self.cfg.limits.min_bsp_liability
                             for (mid, _), p in self.positions.items() if mid == m.market_id)]
        to_settle = [m for mid, m in self.markets.items() if now - m.start >= timedelta(minutes=5)
                     and any(not p.settled and p.matched > 0 for (pm, _), p in self.positions.items() if pm == mid)]
        ids = list(dict.fromkeys([m.market_id for m in to_hedge + to_trade]))
        books = self.x.books(ids) if ids else {}
        for m in to_hedge:
            self.close_out(m, now)
        # every race's plan first, then the orders by expected value: when a limit binds, the best go first
        planned = [(m, books.get(m.market_id), d) for m in to_trade for d in self.plan(m, books.get(m.market_id))]
        planned.sort(key=lambda t: -(t[2].edge if np.isfinite(t[2].edge) else -1e9))
        for m, book, d in planned:
            self.execute(m, book, d, now)
        if to_settle:
            self._settle(to_settle, now)

    def run(self, until: datetime) -> dict:
        self.load_markets()
        errors = 0
        while True:
            now = self.clock()
            if now >= until:
                break
            open_positions = any(p.matched > 0 and not p.settled for p in self.positions.values())
            if not open_positions and all(m.start <= now for m in self.markets.values()):
                break                                        # every race is off and every bet settled
            try:
                self.step(now)
                errors = 0
            except Exception as exc:                         # one bad call must not end the day
                errors += 1
                log.exception("step failed (%d in a row): %s", errors, exc)
                if errors >= 10:
                    self.state.stopped = f"stopped after {errors} failed steps"
            self.flush()
            self.sleep(self.cfg.poll_seconds)
        self.shutdown()
        return self.summary()

    def shutdown(self) -> None:
        self.flush()

    # ------------------------------------------------------------------ one race

    def trade(self, m: Market, book: Book | None, now: datetime) -> None:
        for d in self.plan(m, book):
            self.execute(m, book, d, now)

    def plan(self, m: Market, book: Book | None) -> list:
        """The strategy's decisions for one open race, from its book and the model's prices."""
        if book is None or book.status != "OPEN" or book.inplay:
            return []
        views = []
        for sid, q in book.runners.items():
            bb, bl = q.best_back, q.best_lay
            views.append(RunnerView(selection_id=sid, name=m.runners.get(sid, str(sid)),
                                    model_price=self.prices.get((m.market_id, sid)),
                                    back=bb[0] if bb else None, back_size=bb[1] if bb else 0.0,
                                    lay=bl[0] if bl else None, last_traded=q.last_traded, status=q.status,
                                    traded=q.traded))
        return plan_race(views, self.cfg, self.bank())

    def execute(self, m: Market, book: Book, d, now: datetime) -> None:
        """Back the difference between a decision's target and what is matched already, within the limits."""
        key = (m.market_id, d.selection_id)
        pos = self.positions.get(key)
        want = d.target - (pos.matched if pos else 0.0)
        if self.cfg.strategy == "closing_clv" and pos and pos.matched > 0:
            # a fixed win: only what is left to win, at the price now, within the per-bet limit, so a horse
            # that has shortened since it was backed is not topped up beyond the target
            left = self.cfg.target - pos.matched * (pos.avg_price - 1.0)
            want = min(left / (d.price - 1.0), self.cfg.limits.max_stake - pos.matched)
        if want < self.cfg.limits.min_stake:
            return
        stake, why = allowed_stake(want, d.price, m.market_id, book.total_matched, d.back_size,
                                   self.state, self.cfg.limits)
        base = self._row(now, "back", m, d.selection_id, d.name, strategy=self.cfg.strategy,
                         staking=self.cfg.staking, reason=d.reason, p_model=d.p_model, p_market=d.p_market,
                         p_pool=d.p_pool, edge=d.edge, move=d.move, best_back=d.price, back_size=d.back_size,
                         target=d.target)
        if stake <= 0:
            if (key, why) not in self._skips:                # one line per runner and reason, not one per poll
                self._skips.add((key, why))
                self.ledger.append({**base, "event": "skip", "error": why})
            return
        fill = self.x.back(m.market_id, d.selection_id, d.price, stake,
                           ref=f"ash{self.day:%m%d}{d.selection_id}"[:32], min_fill=self.cfg.limits.min_stake)
        if fill.matched > 0:
            pos = self.positions.setdefault(key, Position(m.market_id, d.selection_id, d.name))
            pos.matched += fill.matched
            pos.cost += fill.matched * fill.avg_price
            self.state.record(m.market_id, fill.matched)
        if fill.status == "UNKNOWN":                         # placed or not cannot be told: no more bets today
            self.state.stopped = f"order state unknown on {m.market_id} {d.selection_id}: {fill.error}"
            log.error("%s", self.state.stopped)
        self.ledger.append({**base, "asked": stake, "matched": fill.matched, "avg_price": fill.avg_price,
                            "bet_id": fill.bet_id, "status": fill.status, "error": fill.error})
        if fill.matched > 0 and self.cfg.trade_out and self.cfg.trade_out_at == "fill":
            self._lay_at_bsp(m, pos, self._unhedged(pos), now)
        self.flush()                                         # an order on record at once, not at the poll's end

    @staticmethod
    def _unhedged(pos: Position) -> float:
        """The liability still to lay at BSP: the position's winnings less the lays already sent."""
        return round(pos.matched * (pos.avg_price - 1.0) - pos.hedge_liability, 2)

    def close_out(self, m: Market, now: datetime) -> None:
        """Lay each open position at BSP with the position's unhedged winnings as the liability."""
        for (mid, sid), pos in self.positions.items():
            if mid != m.market_id or pos.hedged or pos.matched <= 0:
                continue
            self._lay_at_bsp(m, pos, self._unhedged(pos), now)

    def _lay_at_bsp(self, m: Market, pos: Position, liability: float, now: datetime) -> None:
        """A lay at BSP for `liability` (simulated on paper, placed when live): if the runner wins it costs what
        the backs win; if it loses it wins liability / (BSP - 1). Across fills the liabilities add up to the
        position's winnings. A liability under the smallest Betfair takes waits for the next fill or poll."""
        if liability < max(1.0, self.cfg.limits.min_bsp_liability):
            return
        fill = self.x.lay_at_bsp(m.market_id, pos.selection_id, liability,
                                 ref=f"ash{self.day:%m%d}{pos.selection_id}x"[:32])
        if fill.status in ("PENDING", "SUCCESS"):
            pos.hedge_liability = round(pos.hedge_liability + liability, 2)
            pos.hedged = pos.hedge_liability >= round(pos.matched * (pos.avg_price - 1.0), 2) - 0.01
        elif fill.status == "UNKNOWN":                       # hedged or not cannot be told: never lay it twice
            pos.hedge_failures = MAX_HEDGE_TRIES
            self.state.stopped = f"trade-out state unknown on {m.market_id} {pos.selection_id}: {fill.error}"
            log.error("%s", self.state.stopped)
        else:
            pos.hedge_failures += 1
            if pos.hedge_failures >= MAX_HEDGE_TRIES:
                log.error("lay at BSP refused %d times on %s %s (%s): left to the result", pos.hedge_failures,
                          m.market_id, pos.selection_id, fill.error)
        self.ledger.append({**self._row(now, "trade_out", m, pos.selection_id, pos.name), "asked": liability,
                            "bet_id": fill.bet_id, "status": fill.status, "error": fill.error,
                            "matched": pos.matched, "avg_price": round(pos.avg_price, 4), "hedged": pos.hedged,
                            "hedge_liability": liability})

    def _settle(self, markets: list, now: datetime) -> None:
        """Settle the races that have finished: live from Betfair's record of the settled bets, on paper from the
        closed markets. A read that fails leaves the races to the next poll or the evening job; it never ends the
        day's trading."""
        ids = [m.market_id for m in markets]
        cleared = None
        if self.mode == "live":
            try:
                cleared = self.x.cleared(ids)
            except Exception as exc:
                log.warning("Betfair's settled bets not read (%s): %d races wait", str(exc)[:200], len(ids))
                return
        try:
            books = self.x.books(ids, with_sp=True)
        except Exception as exc:
            if cleared is None:
                log.warning("closed markets not read (%s): %d races wait", str(exc)[:200], len(ids))
                return
            books = {}                                       # live, the book only names the result
        for m in markets:
            if cleared is None:
                self.settle(m, books.get(m.market_id), now)
            else:
                self.settle_live(m, books.get(m.market_id), cleared, now)

    def settle(self, m: Market, book: Book | None, now: datetime) -> None:
        """Settle a race on paper from its closed market: void a non-runner, back and BSP-lay profits, then
        commission on the market's net winnings. A race with a lay at BSP on a runner the market has not priced
        (a feed without the SP) waits for the price."""
        if book is None or book.status != "CLOSED":
            return
        unpriced = [sid for (mid, sid), pos in self.positions.items()
                    if mid == m.market_id and not pos.settled and pos.matched > 0 and pos.hedge_liability > 0
                    and sid in book.runners and book.runners[sid].status != "REMOVED"
                    and not (book.runners[sid].bsp and book.runners[sid].bsp > 1)]
        if unpriced:
            if (m.market_id, "unpriced") not in self._skips:
                self._skips.add((m.market_id, "unpriced"))
                log.warning("%s: no BSP for the laid runners %s yet; the race waits for it", m.market_id, unpriced)
            return
        net, rows = 0.0, []
        for (mid, sid), pos in self.positions.items():
            if mid != m.market_id or pos.settled or pos.matched <= 0:
                continue
            q = book.runners.get(sid)
            result = q.status if q else "REMOVED"
            bsp = q.bsp if q else None
            if result == "REMOVED":
                pnl_back = pnl_lay = 0.0                     # a non-runner's bets are void
            else:
                won = result == "WINNER"
                pnl_back = pos.matched * (pos.avg_price - 1.0) if won else -pos.matched
                pnl_lay = 0.0
                if pos.hedge_liability > 0 and bsp and bsp > 1:
                    pnl_lay = -pos.hedge_liability if won else pos.hedge_liability / (bsp - 1.0)
            pos.settled = True
            clv = (pos.avg_price / bsp - 1.0) if bsp and bsp > 1 else np.nan
            net += pnl_back + pnl_lay
            rows.append({**self._row(now, "settle", m, sid, pos.name), "matched": pos.matched,
                         "avg_price": round(pos.avg_price, 4), "hedged": pos.hedged,
                         "hedge_liability": pos.hedge_liability, "result": result, "bsp": bsp,
                         "clv": clv, "pnl_back": round(pnl_back, 2), "pnl_lay": round(pnl_lay, 2),
                         "pnl": round(pnl_back + pnl_lay, 2)})
        if not rows:
            return
        commission = round(self.cfg.commission * max(net, 0.0), 2)
        self.state.settled_pnl += net - commission
        self.ledger.extend(rows)
        self.ledger.append({**self._row(now, "commission", m, 0, ""), "commission": commission,
                            "pnl": round(-commission, 2)})

    def settle_live(self, m: Market, book: Book | None, cleared: list[dict], now: datetime) -> None:
        """Settle a race from Betfair's record of the settled bets (listClearedOrders): what Betfair paid on each
        back and on each lay at SP, the SP the lays were matched at, then commission on the market's net winnings.
        Betfair settles a market some minutes after the race; until it has, the race waits, so a lay at SP is never
        counted as nothing for want of its price (the delayed feed carries no SP)."""
        held = [(sid, pos) for (mid, sid), pos in self.positions.items()
                if mid == m.market_id and not pos.settled and pos.matched > 0]
        if not held:
            return
        ours = self._bet_ids()
        mine = [o for o in cleared or [] if str(o.get("marketId")) == m.market_id and self._ours(o, ours)]
        runners = book.runners if book is not None else {}
        if not mine:
            # nothing of ours settled on the market yet, unless every horse held was a non-runner (void bets are
            # never settled)
            void = book is not None and book.status == "CLOSED" and all(
                sid not in runners or runners[sid].status == "REMOVED" for sid, _ in held)
            if not void:
                return
        net, rows = 0.0, []
        for sid, pos in held:
            bets = [o for o in mine if int(_num0(o.get("selectionId"))) == sid]
            backs = [o for o in bets if o.get("side") == "BACK"]
            lays = [o for o in bets if o.get("side") == "LAY"]
            pnl_back = sum(_num0(o.get("profit")) for o in backs)
            pnl_lay = sum(_num0(o.get("profit")) for o in lays)
            laid = sum(_num0(o.get("sizeSettled")) for o in lays)
            bsp = (sum(_num0(o.get("priceMatched")) * _num0(o.get("sizeSettled")) for o in lays) / laid
                   if laid > 0 else None)
            q = runners.get(sid)
            if bsp is None and q is not None and q.bsp and q.bsp > 1:
                bsp = q.bsp
            if q is not None and q.status in ("WINNER", "LOSER", "REMOVED"):
                result = q.status
            elif backs:
                result = "WINNER" if pnl_back > 0 else "LOSER"
            elif lays:
                result = "LOSER" if pnl_lay > 0 else "WINNER"
            else:
                result = "REMOVED"                           # nothing settled on the horse: its bets were void
            notes = []
            settled_back = sum(_num0(o.get("sizeSettled")) for o in backs)
            if result != "REMOVED" and abs(settled_back - pos.matched) >= 0.01:
                notes.append(f"Betfair settled GBP{settled_back:.2f} of backs; the ledger holds GBP{pos.matched:.2f}")
            if result != "REMOVED" and pos.hedge_liability > 0 and not lays:
                notes.append("no lay at SP settled: the back stood alone")
            pos.settled = True
            clv = (pos.avg_price / bsp - 1.0) if bsp and bsp > 1 else np.nan
            net += pnl_back + pnl_lay
            rows.append({**self._row(now, "settle", m, sid, pos.name), "reason": SETTLED_BY_BETFAIR,
                         "matched": pos.matched, "avg_price": round(pos.avg_price, 4), "hedged": pos.hedged,
                         "hedge_liability": pos.hedge_liability, "result": result,
                         "bsp": round(bsp, 4) if bsp else None, "clv": clv, "pnl_back": round(pnl_back, 2),
                         "pnl_lay": round(pnl_lay, 2), "pnl": round(pnl_back + pnl_lay, 2),
                         "error": "; ".join(notes)})
            if notes:
                log.warning("%s %s: %s", m.market_id, sid, "; ".join(notes))
        commission = round(self.cfg.commission * max(net, 0.0), 2)
        self.state.settled_pnl += net - commission
        self.ledger.extend(rows)
        self.ledger.append({**self._row(now, "commission", m, 0, ""), "reason": SETTLED_BY_BETFAIR,
                            "commission": commission, "pnl": round(-commission, 2)})

    def _bet_ids(self) -> set:
        """The bet ids of every order this ledger placed."""
        return {str(r.get("bet_id")) for r in self.ledger
                if r.get("event") in ("back", "trade_out") and str(r.get("bet_id") or "").strip()}

    def _ours(self, order: dict, bet_ids: set) -> bool:
        """A settled bet of this trader's (its reference, or a bet id in the ledger), not one the owner placed."""
        ref = str(order.get("customerOrderRef") or "")
        return ref.startswith(f"ash{self.day:%m%d}") or str(order.get("betId") or "") in bet_ids

    def _start(self, hhmm: str) -> datetime:
        from zoneinfo import ZoneInfo
        h, mi = (int(x) for x in str(hhmm).split(":"))
        return datetime(self.day.year, self.day.month, self.day.day, h, mi,
                        tzinfo=ZoneInfo("Europe/London")).astimezone(timezone.utc)

    def restore(self, rows: list[dict], markets: list[Market]) -> None:
        """Take up a day from its ledger, to settle it after the job that traded it has ended. A race whose market
        the catalogue no longer lists (closed) keeps its course and off from the ledger. Live, a race settled from
        the price feed rather than from Betfair's record of the bets (the first live day's, when the delayed feed's
        missing SP counted every lay at SP as nothing) is settled again: its rows stay in the ledger as unsettled."""
        self.ledger = list(rows)
        self.markets = {m.market_id: m for m in markets}
        stale = set()
        if self.mode == "live":
            stale = {str(r["market_id"]) for r in rows
                     if r["event"] in ("settle", "commission") and r.get("reason") != SETTLED_BY_BETFAIR}
        for r in rows:
            mid = str(r["market_id"])
            if mid not in self.markets and r.get("off"):
                try:
                    self.markets[mid] = Market(mid, r.get("venue") or "", "", self._start(r["off"]))
                except ValueError:
                    pass
        for r in rows:
            key = (str(r["market_id"]), int(float(r["selection_id"] or 0)))
            if r["event"] == "back" and float(r.get("matched") or 0) > 0:
                pos = self.positions.setdefault(key, Position(key[0], key[1], r.get("runner", "")))
                pos.matched += float(r["matched"])
                pos.cost += float(r["matched"]) * float(r["avg_price"])
                self.state.record(key[0], float(r["matched"]))
            elif r["event"] == "trade_out" and r.get("status") in ("PENDING", "SUCCESS") and key in self.positions:
                pos = self.positions[key]
                pos.hedge_liability = round(pos.hedge_liability + float(r.get("hedge_liability") or 0), 2)
                pos.hedged = pos.hedge_liability >= round(pos.matched * (pos.avg_price - 1.0), 2) - 0.01
            elif r["event"] == "settle" and key in self.positions and key[0] not in stale:
                self.positions[key].settled = True
        for r in rows:
            if r["event"] in ("settle", "commission") and str(r["market_id"]) in stale:
                r["event"] = "unsettled"
                r["error"] = "settled from the price feed with the lays at SP unpriced; settled again from " \
                             "Betfair's record"
        if stale:
            log.warning("%d races settled from the price feed are settled again from Betfair's record", len(stale))
        self.state.settled_pnl = sum(float(r.get("pnl") or 0) for r in rows if r["event"] in ("settle", "commission"))

    def settle_all(self, now: datetime | None = None) -> None:
        """Settle every position whose race has finished, from the closed markets."""
        now = now or self.clock()
        mids = sorted({pm for (pm, _), p in self.positions.items() if p.matched > 0 and not p.settled})
        if not mids:
            return
        self._settle([self.markets.get(mid) or Market(mid, "", "", now) for mid in mids], now)

    # ------------------------------------------------------------------ records

    def _row(self, now: datetime, event: str, m: Market, sid: int, name: str, **kw) -> dict:
        return {"ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode, "event": event,
                "market_id": m.market_id, "venue": m.venue, "off": _uk(m.start).strftime("%H:%M"),
                "minutes_to_off": round((m.start - now).total_seconds() / 60, 1), "selection_id": sid,
                "runner": name, **kw}

    def flush(self) -> None:
        if not self.ledger_path:
            return
        path = Path(self.ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.ledger)
        if self.on_flush is not None:
            try:
                self.on_flush(path, len(self.ledger))
            except Exception as exc:                         # a copy that failed is tried again next flush
                log.warning("ledger copy failed: %s", exc)

    def sync(self, orders: list[dict]) -> None:
        """Take today's positions from Betfair's own record of our orders (live): what is matched on each runner
        and what is laid at SP against it. Where Betfair and the ledger differ, Betfair is right, so a job that
        starts again never backs a horse a second time for want of its own notes."""
        prefix = f"ash{self.day:%m%d}"
        got: dict = {}
        for o in orders:
            if not str(o.get("customerOrderRef", "")).startswith(prefix):
                continue
            key = (str(o["marketId"]), int(o["selectionId"]))
            g = got.setdefault(key, {"matched": 0.0, "cost": 0.0, "laid": 0.0})
            if o.get("side") == "BACK":
                size = float(o.get("sizeMatched") or 0.0)
                g["matched"] += size
                g["cost"] += size * float(o.get("averagePriceMatched") or 0.0)
            elif o.get("side") == "LAY" and o.get("orderType") == "MARKET_ON_CLOSE":
                g["laid"] += float(o.get("bspLiability") or 0.0)
        now = self.clock()
        for key, g in got.items():
            if g["matched"] <= 0:
                continue
            pos = self.positions.get(key)
            if pos and abs(pos.matched - g["matched"]) < 0.01 and abs(pos.hedge_liability - g["laid"]) < 0.01:
                continue
            m = self.markets.get(key[0]) or Market(key[0], "", "", now)
            name = m.runners.get(key[1], pos.name if pos else str(key[1]))
            before = pos.matched if pos else 0.0
            pos = self.positions.setdefault(key, Position(key[0], key[1], name))
            self.state.turnover += g["matched"] - before
            self.state.race_stake[key[0]] = self.state.race_stake.get(key[0], 0.0) + g["matched"] - before
            if before <= 0:
                self.state.bets += 1
            pos.matched, pos.cost, pos.hedge_liability = g["matched"], g["cost"], g["laid"]
            pos.hedged = pos.hedge_liability >= round(pos.matched * (pos.avg_price - 1.0), 2) - 0.01
            self.ledger.append({**self._row(now, "sync", m, key[1], name), "matched": round(pos.matched, 2),
                                "avg_price": round(pos.avg_price, 4), "hedge_liability": pos.hedge_liability,
                                "hedged": pos.hedged, "error": f"from Betfair's orders (ledger had {before:.2f})"})
            log.warning("positions from Betfair: %s %s matched %.2f (the ledger had %.2f)", key[0], key[1],
                        pos.matched, before)

    def summary(self) -> dict:
        backs = [r for r in self.ledger if r["event"] == "back" and _num(r.get("matched")) > 0]
        settled = [r for r in self.ledger if r["event"] == "settle"]
        stakes = np.array([_num(r.get("matched")) for r in settled], float)
        clvs = np.array([_num(r.get("clv")) for r in settled], float)
        ok = np.isfinite(clvs) & (stakes > 0)
        return {
            "mode": self.mode, "day": str(self.day), "strategy": self.cfg.strategy, "staking": self.cfg.staking,
            "markets": len(self.markets), "bets_matched": len(backs),
            "turnover": round(self.state.turnover, 2),
            "races_settled": len({r["market_id"] for r in settled}),
            "settled_pnl": round(self.state.settled_pnl, 2),
            "stake_weighted_clv": round(float(np.sum(stakes[ok] * clvs[ok]) / stakes[ok].sum()), 4) if ok.any() else None,
            "stopped": self.state.stopped,
        }
