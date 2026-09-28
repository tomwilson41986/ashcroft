"""A paper-trading day: read the markets, decide, check the limits, simulate, close at BSP, settle.

One `Session` runs one UK racing day against the paper exchange (trading/exchange.py), which reads
Betfair's live prices and simulates every order. Each `step` reads the books of the races still
ahead, plans each race (trading/strategy.py), backs the difference between the plan and what is
already matched, subject to the limits (trading/risk.py), closes each fill with a lay at BSP when
`trade_out` is on, and settles the races that have finished from the closed market (winner, loser
or non-runner, and the BSP). Every simulated order and every settlement is a row in the ledger:
the forward test of the edge at the prices actually on offer.
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


@dataclass
class Position:
    market_id: str
    selection_id: int
    name: str
    matched: float = 0.0
    cost: float = 0.0                    # sum of matched stake x price
    hedged: bool = False
    hedge_liability: float = 0.0
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
                 sleep=None, ledger_path: str | None = None):
        self.x = exchange
        self.cfg = cfg
        self.day = day
        self.mode = "paper"
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
        to_settle = [m for mid, m in self.markets.items() if now - m.start >= timedelta(minutes=5)
                     and any(not p.settled and p.matched > 0 for (pm, _), p in self.positions.items() if pm == mid)]
        ids = list(dict.fromkeys([m.market_id for m in to_hedge + to_trade]))
        books = self.x.books(ids) if ids else {}
        for m in to_hedge:
            self.close_out(m, now)
        for m in to_trade:
            self.trade(m, books.get(m.market_id), now)
        if to_settle:
            done = self.x.books([m.market_id for m in to_settle], with_sp=True)
            for m in to_settle:
                self.settle(m, done.get(m.market_id), now)

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
        if book is None or book.status != "OPEN" or book.inplay:
            return
        views = []
        for sid, q in book.runners.items():
            bb, bl = q.best_back, q.best_lay
            views.append(RunnerView(selection_id=sid, name=m.runners.get(sid, str(sid)),
                                    model_price=self.prices.get((m.market_id, sid)),
                                    back=bb[0] if bb else None, back_size=bb[1] if bb else 0.0,
                                    lay=bl[0] if bl else None, last_traded=q.last_traded, status=q.status))
        for d in plan_race(views, self.cfg, self.bank()):
            key = (m.market_id, d.selection_id)
            pos = self.positions.get(key)
            want = d.target - (pos.matched if pos else 0.0)
            if want < self.cfg.limits.min_stake:
                continue
            stake, why = allowed_stake(want, d.price, m.market_id, book.total_matched, d.back_size,
                                       self.state, self.cfg.limits)
            base = self._row(now, "back", m, d.selection_id, d.name, strategy=self.cfg.strategy,
                             staking=self.cfg.staking, reason=d.reason, p_model=d.p_model, p_market=d.p_market,
                             p_pool=d.p_pool, edge=d.edge, move=d.move, best_back=d.price, back_size=d.back_size,
                             target=d.target)
            if stake <= 0:
                if (key, why) not in self._skips:            # one line per runner and reason, not one per poll
                    self._skips.add((key, why))
                    self.ledger.append({**base, "event": "skip", "error": why})
                continue
            fill = self.x.back(m.market_id, d.selection_id, d.price, stake,
                               ref=f"ash{self.day:%m%d}{d.selection_id}"[:32], min_fill=self.cfg.limits.min_stake)
            if fill.matched > 0:
                pos = self.positions.setdefault(key, Position(m.market_id, d.selection_id, d.name))
                pos.matched += fill.matched
                pos.cost += fill.matched * fill.avg_price
                self.state.record(m.market_id, fill.matched)
            self.ledger.append({**base, "asked": stake, "matched": fill.matched, "avg_price": fill.avg_price,
                                "bet_id": fill.bet_id, "status": fill.status, "error": fill.error})
            if fill.matched > 0 and self.cfg.trade_out and self.cfg.trade_out_at == "fill":
                self._lay_at_bsp(m, pos, round(fill.matched * (fill.avg_price - 1.0), 2), now)

    def close_out(self, m: Market, now: datetime) -> None:
        """Lay each open position at BSP with the position's unhedged winnings as the liability."""
        for (mid, sid), pos in self.positions.items():
            if mid != m.market_id or pos.hedged or pos.matched <= 0:
                continue
            self._lay_at_bsp(m, pos, round(pos.matched * (pos.avg_price - 1.0) - pos.hedge_liability, 2), now)

    def _lay_at_bsp(self, m: Market, pos: Position, liability: float, now: datetime) -> None:
        """A simulated lay at BSP for `liability`: if the runner wins it costs what the backs win; if it
        loses it wins liability / (BSP - 1). Across fills the liabilities add up to the position's winnings."""
        if liability < 1.0:
            return
        fill = self.x.lay_at_bsp(m.market_id, pos.selection_id, liability,
                                 ref=f"ash{self.day:%m%d}{pos.selection_id}x"[:32])
        if fill.status in ("PENDING", "SUCCESS"):
            pos.hedge_liability = round(pos.hedge_liability + liability, 2)
            pos.hedged = pos.hedge_liability >= round(pos.matched * (pos.avg_price - 1.0), 2) - 0.01
        self.ledger.append({**self._row(now, "trade_out", m, pos.selection_id, pos.name), "asked": liability,
                            "bet_id": fill.bet_id, "status": fill.status, "error": fill.error,
                            "matched": pos.matched, "avg_price": round(pos.avg_price, 4), "hedged": pos.hedged,
                            "hedge_liability": liability})

    def settle(self, m: Market, book: Book | None, now: datetime) -> None:
        """Settle a race from its closed market: void a non-runner, back and BSP-lay profits, then
        commission on the market's net winnings."""
        if book is None or book.status != "CLOSED":
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

    def restore(self, rows: list[dict], markets: list[Market]) -> None:
        """Take up a day from its ledger, to settle it after the job that traded it has ended."""
        self.ledger = list(rows)
        self.markets = {m.market_id: m for m in markets}
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
            elif r["event"] == "settle" and key in self.positions:
                self.positions[key].settled = True
        self.state.settled_pnl = sum(float(r.get("pnl") or 0) for r in rows if r["event"] in ("settle", "commission"))

    def settle_all(self, now: datetime | None = None) -> None:
        """Settle every position whose race has finished, from the closed markets."""
        now = now or self.clock()
        mids = sorted({pm for (pm, _), p in self.positions.items() if p.matched > 0 and not p.settled})
        if not mids:
            return
        books = self.x.books(mids, with_sp=True)
        for mid in mids:
            self.settle(self.markets.get(mid) or Market(mid, "", "", now), books.get(mid), now)

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

    def summary(self) -> dict:
        backs = [r for r in self.ledger if r["event"] == "back" and (r.get("matched") or 0) > 0]
        settled = [r for r in self.ledger if r["event"] == "settle"]
        stakes = np.array([r["matched"] for r in settled], float)
        clvs = np.array([r["clv"] for r in settled], float)
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
