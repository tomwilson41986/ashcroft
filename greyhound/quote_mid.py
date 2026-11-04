"""Quotes both sides of every dog around the market's own mid, live (the owner, 4 Nov 2026: "go live with live trading
today", after the profit simulator's quotes around the mid, research/greyhound/profit_sim.py ``mid_*``; the owner,
3 Nov: "it is allowed to execute trades on each side of the line on every dog right up to the off").

- **When:** each race from the moment every dog in it has traded, from ``start_minutes`` before the off, requoted at
  the simulator's cadence: every ``far_poll_seconds`` until 5 minutes out, then every ``poll_seconds``; no new quote in
  the last ``stop_seconds``.
- **What:** for each dog, a resting LIMIT lay at ``mid x (1 - margin)`` (rounded down the ladder) and a resting LIMIT
  back at ``mid x (1 + margin)`` (rounded up), the mid being the market's price now (model.race_book.market_now). Both
  rest away from the mid, so neither takes a price; persistence LAPSE. An order whose price no longer holds is
  cancelled (what it matched is kept) and a new one placed at the new price.
- **Size:** the simulator's: each order risks ``liability`` (a lay's liability, a back's winnings), its stake between
  ``min_stake`` and ``max_stake``. A dog takes no new order on a side once its worst outcome reaches ``max_loss``.
- **The exit:** ``green_seconds`` before the off the resting orders come off and each dog's position is greened up: one
  order of stake ``(win - lose) / G`` (a lay when positive, a back when negative) down the best three prices, fill or
  kill, kept and retried each second while partly matched; ``give_up_seconds`` before the off what is still lay-heavy
  is backed at the Betfair SP, anything else held to the result.
- **Limits:** a day stop once the locked P&L reaches ``-day_loss_limit``; at most ``max_orders_hour`` orders an hour
  (Betfair charges for more than 5,000 transactions an hour). The S3 objects ``greyhound/STOP`` and ``trading/STOP``
  stop new quotes and pull the resting ones; what is matched is still greened up.

Real orders only with ``--live`` and ``GREYHOUND_QUOTE_MID=live``; otherwise the same decisions are logged and nothing
is sent. Our orders carry the ``qm-`` reference, so positions are read from Betfair's own record and a restarted
session greens up what an earlier one matched. Ledger: ``sources/greyhound/quote_mid/<mode>/<day>/ledger.csv``.

    GREYHOUND_QUOTE_MID=live python -m greyhound.quote_mid --trade --live --until 21:30
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PREFIX = "greyhound/quote_mid"
REF = "qm-"
CONFIG = {
    "margin": 0.05,                 # the simulator's mid_m05 (the hardened run decides the live value)
    # the staking (research/greyhound/profit_sim.py stake_q): 'liability' (each order risks ``liability``: a lay's
    # liability, a back's winnings), 'stake' (``liability`` as the stake itself), 'liquidity' (``liq_frac`` of the
    # money shown at the order's price on its side, at most ``max_stake``) or 'late' (liability, ``late_x`` times as
    # much in the last ``late_secs``)
    "stake_type": "liability",
    "liability": 20.0,
    "min_stake": 2.0,
    "max_stake": 10.0,
    "liq_frac": 0.25,
    "late_x": 2.5,
    "late_secs": 120.0,
    "max_price": 20.0,
    "max_loss": 60.0,               # a dog's worst outcome at which that side takes no new order
    "start_minutes": 15.0,          # quoted from this long before the off (the live-key books' first reads)
    "far_poll_seconds": 60.0,       # requoted this often until 5 minutes out
    "poll_seconds": 5.0,            # then this often (the live-key books' cadence the simulator read)
    "near_poll_seconds": 1.0,       # the loop's tick: green-ups are tried every second
    "stop_seconds": 10.0,           # no new quote this close to the off
    "green_seconds": 5.0,
    "give_up_seconds": 2.0,
    "min_order": 1.0,               # Betfair's smallest limit stake
    "min_sp_back": 2.0,
    "day_loss_limit": 100.0,        # the day stops once the locked P&L reaches minus this
    "max_orders_hour": 4000,        # under Betfair's 5,000 an hour, above which each is charged
}
LEDGER_FIELDS = ["time_utc", "mode", "market_id", "selection_id", "action", "side", "price", "size", "matched",
                 "avg_price", "bet_id", "status", "error", "mid", "win", "lose", "locked_pnl", "seconds_to_off"]


def switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("GREYHOUND_QUOTE_MID", "")).strip().lower() == "live"


def _floor2(x: float) -> float:
    return math.floor(float(x) * 100 + 1e-9) / 100


def stake_for(price: float, cfg: dict, side: str = "LAY", quote=None, secs: float = 600.0) -> float:
    """A quote's stake by ``stake_type`` (the simulator's ``stake_q``), between ``min_stake`` and ``max_stake``."""
    kind = cfg.get("stake_type", "liability")
    if kind == "stake":
        s = cfg["liability"]
    elif kind == "liquidity":
        lvl = [] if quote is None else list(quote.back if side == "LAY" else quote.lay)
        s = cfg["liq_frac"] * (lvl[0][1] if lvl else 0.0)
    else:
        s = cfg["liability"] / max(price - 1.0, 1e-9)
        if kind == "late" and secs <= cfg["late_secs"]:
            s *= cfg["late_x"]
    return _floor2(min(max(s, cfg["min_stake"]), cfg["max_stake"]))


def mid_of(back1: float, lay1: float | None) -> float:
    from model.race_book import market_now
    return float(market_now([back1], [lay1 if lay1 else np.nan])[0])


def targets(back1: float, lay1: float | None, cfg: dict) -> tuple[float | None, float | None, float]:
    """The lay and back prices to rest at for a dog whose best prices are ``back1`` / ``lay1``: the mid less and
    plus ``margin``, on the ladder, each kept only where it rests away from the mid and under ``max_price``."""
    from trading.exchange import tick_down, tick_up
    m = mid_of(back1, lay1)
    lay = tick_down(m * (1 - cfg["margin"]))
    back = tick_up(m * (1 + cfg["margin"]))
    lay = lay if lay and 1.0 < lay < m - 1e-9 and lay <= cfg["max_price"] else None
    back = back if back and back > m + 1e-9 and back <= cfg["max_price"] else None
    return lay, back, m


def outcomes_of(fills) -> tuple[float, float]:
    """A dog's matched position, (side, price, stake) each: its P&L if it wins and if it loses."""
    win = sum(st * (p - 1) if sd == "BACK" else -st * (p - 1) for sd, p, st in fills)
    lose = sum(-st if sd == "BACK" else st for sd, p, st in fills)
    return win, lose


def positions(orders: list[dict]) -> dict:
    """Per dog from Betfair's orders under our reference: what is matched ((side, average price, stake) each) and the
    quotes still resting (side -> the orders)."""
    out: dict = {}
    for o in orders:
        sid = int(o.get("selectionId", 0))
        p = out.setdefault(sid, {"fills": [], "open": {}, "sp": False})
        side = str(o.get("side") or "")
        m = float(o.get("sizeMatched") or 0)
        ref = str(o.get("customerOrderRef") or "")
        if o.get("orderType") == "MARKET_ON_CLOSE":
            p["sp"] = True
            continue
        if m > 0:
            p["fills"].append((side, float(o.get("averagePriceMatched") or 0), m))
        rest = float(o.get("sizeRemaining") or 0)
        if o.get("status") == "EXECUTABLE" and rest > 0 and "-q" in ref:
            p["open"].setdefault(side, []).append(o)
    return out


def hedge(fills, back: list, lay: list, min_order: float) -> tuple[str, float, float] | None:
    """The order that greens a dog up: stake ``(win - lose) / G`` at the price ``G`` reached down the best three prices
    it can take (a lay when the dog's position wins more if it wins, else a back); None when it is already flat to
    within Betfair's smallest stake."""
    from greyhound.early_lay import ladder_limit
    win, lose = outcomes_of(fills)
    d = win - lose
    side = "LAY" if d > 0 else "BACK"
    levels = lay if side == "LAY" else back
    if not levels:
        return None
    limit = ladder_limit(levels, abs(d))
    if not limit or limit <= 1:
        return None
    stake = _floor2(abs(d) / limit)
    if stake < min_order:
        return None
    return side, float(limit), stake


# --------------------------------------------------------------------------------------------------------------------
# The orders
# --------------------------------------------------------------------------------------------------------------------

class Orders:
    """Limit orders, their cancellation by bet id and Betfair's record of them, on a trading.exchange.LiveExchange's
    session. ``live`` False: nothing is sent; resting quotes are kept in memory (never matched) so the decisions read as
    they would live."""

    def __init__(self, x, live: bool):
        self.x, self.live = x, live
        self.paper: dict = {}             # bet id -> order (paper only)
        self.n = 0

    def place(self, mid: str, sid: int, side: str, price: float, size: float, ref: str, fok: bool = False,
              min_fill: float | None = None) -> dict:
        self.n += 1
        if not self.live:
            bid = f"paper-{self.n}"
            if not fok:
                self.paper[bid] = {"betId": bid, "marketId": mid, "selectionId": int(sid), "side": side,
                                   "orderType": "LIMIT", "status": "EXECUTABLE", "sizeMatched": 0.0,
                                   "sizeRemaining": size, "priceSize": {"price": price, "size": size},
                                   "customerOrderRef": ref[:32]}
            return {"status": "PAPER", "matched": 0.0, "avg_price": 0.0, "bet_id": bid}
        lim = {"size": _floor2(size), "price": float(price), "persistenceType": "LAPSE"}
        if fok:
            lim.update({"timeInForce": "FILL_OR_KILL", "minFillSize": _floor2(min(min_fill or size, size))})
        ins = {"selectionId": int(sid), "handicap": 0, "side": side, "orderType": "LIMIT",
               "customerOrderRef": ref[:32], "limitOrder": lim}
        reply = self.x._place(mid, ins, ref)
        if reply is None:                                       # placed or not: Betfair's own record decides
            got = [o for o in self.mine(mid) if o.get("customerOrderRef") == ref[:32]]
            if got:
                o = got[-1]
                return {"status": "SUCCESS", "matched": float(o.get("sizeMatched") or 0),
                        "avg_price": float(o.get("averagePriceMatched") or 0), "bet_id": str(o.get("betId") or ""),
                        "error": "reply lost; found in Betfair's orders"}
            return {"status": "FAILURE", "matched": 0.0, "avg_price": 0.0, "bet_id": "",
                    "error": "reply lost; no order under the reference"}
        rep, ok = self.x._report(reply)
        return {"status": "SUCCESS" if ok else "FAILURE", "matched": float(rep.get("sizeMatched") or 0),
                "avg_price": float(rep.get("averagePriceMatched") or 0), "bet_id": str(rep.get("betId") or ""),
                "error": "" if ok else self.x._reason(reply, rep, "FAILURE")}

    def cancel(self, mid: str, bet_ids: list[str]) -> str:
        """Our resting orders by bet id (never the whole market: the early lay trades the same markets)."""
        bet_ids = [b for b in bet_ids if b]
        if not bet_ids:
            return ""
        if not self.live:
            for b in bet_ids:
                self.paper.pop(b, None)
            return "PAPER"
        out = []
        for i in range(0, len(bet_ids), 60):
            try:
                reply = self.x._call("cancelOrders", {"marketId": mid, "instructions": [
                    {"betId": str(b)} for b in bet_ids[i:i + 60]]}) or {}
                out.append(str(reply.get("status") or ""))
            except Exception as exc:
                out.append(f"FAILURE: {exc}"[:200])
        return ";".join(out)

    def mine(self, mid: str) -> list[dict]:
        if not self.live:
            return [o for o in self.paper.values() if o["marketId"] == mid]
        return [o for o in self.x.orders([mid]) if str(o.get("customerOrderRef") or "").startswith(REF)]

    def back_at_sp(self, mid: str, sid: int, stake: float, ref: str) -> dict:
        if not self.live:
            return {"status": "PAPER"}
        ins = {"selectionId": int(sid), "handicap": 0, "side": "BACK", "orderType": "MARKET_ON_CLOSE",
               "customerOrderRef": ref[:32], "marketOnCloseOrder": {"liability": _floor2(stake)}}
        reply = self.x._place(mid, ins, ref)
        if reply is None:
            return {"status": "UNKNOWN", "error": "reply lost"}
        rep, ok = self.x._report(reply)
        return {"status": "PENDING" if ok else "FAILURE", "bet_id": str(rep.get("betId") or ""),
                "error": "" if ok else self.x._reason(reply, rep, "FAILURE")}


# --------------------------------------------------------------------------------------------------------------------
# The trader
# --------------------------------------------------------------------------------------------------------------------

class QuoteMid:
    def __init__(self, store, x, cfg: dict, day: date, live: bool, ledger_path: Path, kill=lambda: False,
                 now=lambda: datetime.now(timezone.utc)):
        self.store, self.x, self.cfg, self.day, self.live = store, x, cfg, day, live
        self.mode = "live" if live else "paper"
        self.orders = Orders(x, live)
        self.ledger_path, self.kill, self.now = ledger_path, kill, now
        self.last_quote: dict = {}          # market -> when it was last requoted
        self.quoted: set = set()            # markets with a quote placed
        self.pulled: set = set()            # markets whose resting quotes came off before the green-up
        self.greened: set = set()           # markets closed out
        self.dog_done: set = set()          # (market, dog) greened up, sent to the SP or held
        self.locked = 0.0
        self.placed: deque = deque()        # times of the orders placed, for the hourly count
        self.preds: pd.DataFrame | None = None
        self._loaded_at = None
        self._beat_at = None
        self._copied_at = None
        self.last_error = ""
        self.stopped_for = ""
        self._resume()

    # the record ----------------------------------------------------------------------------------------------------
    def _resume(self) -> None:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open() as f:
            for r in csv.DictReader(f):
                if r["action"] == "quote":
                    self.quoted.add(r["market_id"])
                if r["action"] in ("closed",):
                    self.greened.add(r["market_id"])
                if r["action"] == "green" and r.get("locked_pnl"):
                    self.locked += float(r["locked_pnl"])
                if r["action"] in ("green", "sp_close", "hold") and r.get("selection_id"):
                    self.dog_done.add((r["market_id"], int(float(r["selection_id"]))))

    def _write(self, rows: list[dict], force: bool = False) -> None:
        if rows:
            new = not self.ledger_path.exists()
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerows(rows)
        t = self.now()
        if self.ledger_path.exists() and (force or not self._copied_at or (t - self._copied_at).total_seconds() >= 60):
            self._copied_at = t
            try:
                self.store.put(f"{PREFIX}/{self.mode}/{self.day:%Y-%m-%d}/ledger.csv", self.ledger_path.read_bytes())
            except Exception as exc:                            # the local ledger is the record; S3 its copy
                log.warning("ledger not copied to S3 (%s)", exc)

    def _row(self, market: str, secs: float, **kw) -> dict:
        return {"time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
                "market_id": market, "seconds_to_off": round(secs, 1), **kw}

    def orders_last_hour(self) -> int:
        t = self.now()
        while self.placed and (t - self.placed[0]).total_seconds() > 3600:
            self.placed.popleft()
        return len(self.placed)

    def load_predictions(self) -> None:
        """The day's GB races (market ids and offs) from the live model's prices: the races the simulator quoted."""
        t = self.now()
        if self._loaded_at and (t - self._loaded_at).total_seconds() < 600:
            return
        raw = self.store.get(f"greyhound/live/predictions/{self.day:%Y-%m-%d}.csv")
        self._loaded_at = t
        if raw:
            p = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
            p = p.dropna(subset=["market_id"]).drop_duplicates("market_id", keep="last")
            p["off"] = pd.to_datetime(p.race_date + " " + p.race_time.astype(str).str[:5]).dt.tz_localize(
                "Europe/London").dt.tz_convert("UTC")
            self.preds = p

    def heartbeat(self, force: bool = False, **extra) -> None:
        t = self.now()
        if not force and self._beat_at and (t - self._beat_at).total_seconds() < 60:
            return
        self._beat_at = t
        st = {"time_utc": pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
              "margin": self.cfg["margin"], "markets_quoted": len(self.quoted), "markets_closed": len(self.greened),
              "locked_pnl": round(self.locked, 2), "orders_last_hour": self.orders_last_hour(),
              "stopped": bool(self.kill()), "stopped_for": self.stopped_for, "last_error": self.last_error[:300],
              **extra}
        try:
            self.store.put(f"{PREFIX}/{self.mode}/{self.day:%Y-%m-%d}/status.json", json.dumps(st).encode())
        except Exception as exc:
            log.warning("status not written (%s)", exc)

    # one poll ------------------------------------------------------------------------------------------------------
    def step(self) -> None:
        self.load_predictions()
        if self.preds is None:
            return
        t = pd.Timestamp(self.now())
        to_off = (self.preds.off - t).dt.total_seconds()
        # a race with a position stays watched past its scheduled off until it is closed (a race off late is open)
        late = self.preds.market_id.isin(self.quoted) & (to_off > -180)
        watch = self.preds[((to_off > 0) | late) & (to_off <= self.cfg["start_minutes"] * 60)
                           & ~self.preds.market_id.isin(self.greened)]
        if not len(watch):
            return
        offs = watch.set_index(watch.market_id.astype(str)).off
        mids = sorted(offs.index, key=lambda m: offs[m])               # the nearest off first
        stopped = bool(self.kill())
        if not stopped and self.locked <= -self.cfg["day_loss_limit"]:
            stopped, self.stopped_for = True, f"the day's loss limit (locked {self.locked:.2f})"
        busy = self.orders_last_hour() >= self.cfg["max_orders_hour"]
        books = self.x.books(mids)
        for mid in mids:
            b = books.get(mid)
            if b is None:
                continue
            secs = float((pd.Timestamp(offs[mid]) - pd.Timestamp(self.now())).total_seconds())
            try:
                if secs <= self.cfg["green_seconds"]:
                    self.green(mid, b, secs)
                elif secs <= self.cfg["stop_seconds"] or stopped:
                    if mid in self.quoted and mid not in self.pulled:
                        self.pull(mid, secs, "stop" if stopped else "near the off")
                elif self.due(mid, secs) and not busy:
                    self.quote(mid, b, secs)
            except Exception as exc:                            # one market's failure never stops the others
                log.warning("%s: %s", mid, exc)
                self.last_error = f"{mid}: {exc}"

    def due(self, mid: str, secs: float) -> bool:
        last = self.last_quote.get(mid)
        gap = self.cfg["far_poll_seconds"] if secs > 300 else self.cfg["poll_seconds"]
        return last is None or (self.now() - last).total_seconds() >= gap - 0.5

    def pull(self, mid: str, secs: float, why: str) -> None:
        ords = self.orders.mine(mid)
        ids = [str(o.get("betId")) for o in ords if o.get("status") == "EXECUTABLE"
               and float(o.get("sizeRemaining") or 0) > 0 and "-q" in str(o.get("customerOrderRef") or "")]
        st = self.orders.cancel(mid, ids)
        self.pulled.add(mid)
        self._write([self._row(mid, secs, action="pull", status=st, error=why, size=len(ids))])

    def quote(self, mid: str, b, secs: float) -> None:
        if b.status != "OPEN" or b.inplay:
            return
        active = {sid: q for sid, q in b.runners.items() if q.status == "ACTIVE"}
        if not active or any(not (q.last_traded and q.last_traded > 1 and q.best_back) for q in active.values()):
            return                                              # every dog must have traded (the simulator's start)
        self.last_quote[mid] = self.now()
        self.pulled.discard(mid)
        pos = positions(self.orders.mine(mid))
        rows, cancel = [], []
        tag = f"{REF}{self.day:%m%d}-{mid[-6:]}"
        for sid, q in active.items():
            lay, back, m = targets(q.best_back[0], q.best_lay[0] if q.best_lay else None, self.cfg)
            p = pos.get(sid, {"fills": [], "open": {}})
            win, lose = outcomes_of(p["fills"])
            want = {"LAY": lay if win > -self.cfg["max_loss"] else None,       # a lay loses if the dog wins
                    "BACK": back if lose > -self.cfg["max_loss"] else None}     # a back loses if it loses
            for side, price in want.items():
                # keep one resting quote at the price wanted; any other on this side comes off
                cur = None
                for o in p["open"].get(side, []):
                    op = float((o.get("priceSize") or {}).get("price") or 0)
                    if cur is None and price is not None and abs(op - price) < 1e-9:
                        cur = o
                    else:
                        cancel.append(str(o.get("betId")))
                if price is not None and cur is None:
                    if self.orders_last_hour() >= self.cfg["max_orders_hour"]:
                        continue
                    size = stake_for(price, self.cfg, side, q, secs)
                    ref = f"{tag}-{str(sid)[-7:]}-q{self.orders.n % 1000}"
                    r = self.orders.place(mid, sid, side, price, size, ref)
                    self.placed.append(self.now())
                    self.quoted.add(mid)
                    rows.append(self._row(mid, secs, selection_id=sid, action="quote", side=side, price=price,
                                          size=size, matched=r.get("matched"), avg_price=r.get("avg_price"),
                                          bet_id=r.get("bet_id"), status=r.get("status"), error=r.get("error", ""),
                                          mid=round(m, 3), win=round(win, 2), lose=round(lose, 2)))
        if cancel:
            # the old quotes come off after the new ones go in: a dog is never left unquoted between the two
            st = self.orders.cancel(mid, cancel)
            rows.append(self._row(mid, secs, action="requote", size=len(cancel), status=st))
        self._write(rows)

    def green(self, mid: str, b, secs: float) -> None:
        """Pull the quotes, then green each dog's position up down the book; near the off, close what is left."""
        if mid not in self.quoted:
            self.greened.add(mid)
            return
        if mid not in self.pulled:
            self.pull(mid, secs, "green-up")
        pos = positions(self.orders.mine(mid))
        rows, left = [], False
        closed = b.status != "OPEN" or b.inplay
        tag = f"{REF}{self.day:%m%d}-{mid[-6:]}"
        for sid, p in pos.items():
            if not p["fills"] or p.get("sp") or (mid, sid) in self.dog_done:
                continue
            win, lose = outcomes_of(p["fills"])
            q = b.runners.get(sid)
            d = win - lose
            levels = ([] if q is None else list(q.lay if d > 0 else q.back))
            ref_price = levels[0][0] if levels else 2.0
            if abs(d) / ref_price < self.cfg["min_order"]:              # flat to within the smallest stake
                locked = round(min(win, lose), 2)
                self.locked += locked
                self.dog_done.add((mid, sid))
                rows.append(self._row(mid, secs, selection_id=sid, action="green", win=round(win, 2),
                                      lose=round(lose, 2), locked_pnl=locked))
                continue
            h = None if closed or q is None else hedge(p["fills"], list(q.back), list(q.lay), self.cfg["min_order"])
            if h is not None and secs > self.cfg["give_up_seconds"]:
                side, price, stake = h
                ref = f"{tag}-{str(sid)[-7:]}-g{self.orders.n % 1000}"
                r = self.orders.place(mid, sid, side, price, stake, ref, fok=True, min_fill=self.cfg["min_order"])
                self.placed.append(self.now())
                rows.append(self._row(mid, secs, selection_id=sid, action="hedge", side=side, price=price, size=stake,
                                      matched=r.get("matched"), avg_price=r.get("avg_price"), bet_id=r.get("bet_id"),
                                      status=r.get("status"), error=r.get("error", ""), win=round(win, 2),
                                      lose=round(lose, 2)))
                left = True                                     # read again next second: flat, or tried again
                continue
            if not closed and h is None and levels and secs > self.cfg["give_up_seconds"]:
                left = True                                     # under the smallest stake at this price: next second
                continue
            # the off is upon it, or no price to take: what is lay-heavy is backed at the SP, the rest held
            self.dog_done.add((mid, sid))
            net_lay = sum(st for sd, pr, st in p["fills"] if sd == "LAY") - \
                sum(st for sd, pr, st in p["fills"] if sd == "BACK")
            if not closed and net_lay >= self.cfg["min_sp_back"]:
                r = self.orders.back_at_sp(mid, sid, _floor2(net_lay), f"{tag}-{str(sid)[-7:]}-s")
                rows.append(self._row(mid, secs, selection_id=sid, action="sp_close", side="BACK", size=_floor2(net_lay),
                                      status=r.get("status"), bet_id=r.get("bet_id", ""), error=r.get("error", ""),
                                      win=round(win, 2), lose=round(lose, 2)))
            else:
                rows.append(self._row(mid, secs, selection_id=sid, action="hold", win=round(win, 2),
                                      lose=round(lose, 2), error="market closed" if closed else "held to the result"))
        if not left or closed or secs <= self.cfg["give_up_seconds"] - 3:
            self.greened.add(mid)
            rows.append(self._row(mid, secs, action="closed"))
        self._write(rows, force=mid in self.greened)

    def run(self, until: datetime, sleep=time.sleep) -> dict:
        while self.now() < until:
            try:
                self.step()
            except Exception as exc:                            # one failed poll: the next tries again
                log.warning("poll failed (%s)", exc)
                self.last_error = f"poll: {exc}"
            self.heartbeat()
            sleep(self.cfg["near_poll_seconds"])
        self._write([], force=True)
        self.heartbeat(force=True, finished=True)
        return {"day": f"{self.day:%Y-%m-%d}", "mode": self.mode, "margin": self.cfg["margin"],
                "markets_quoted": len(self.quoted), "markets_closed": len(self.greened),
                "locked_pnl": round(self.locked, 2)}


# --------------------------------------------------------------------------------------------------------------------
# The running table
# --------------------------------------------------------------------------------------------------------------------

def table(store, day: date, mode: str = "live") -> str:
    """The day so far, race by race, from the ledger: quotes placed, what matched, the green-ups and their locked P&L,
    what went to the SP or was held."""
    raw = store.get(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/ledger.csv")
    st = store.get(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/status.json")
    L = [f"### Quotes around the mid, {mode} ({day})", ""]
    if st:
        s = json.loads(st)
        L += [f"Trader: last seen {s.get('time_utc')} UTC; margin {s.get('margin')}; quoted {s.get('markets_quoted')}, "
              f"closed {s.get('markets_closed')}; orders in the last hour {s.get('orders_last_hour')}"
              + (f"; stopped: {s.get('stopped_for') or 'S3 STOP'}" if s.get("stopped") else "")
              + (f"; last error: {s.get('last_error')}" if s.get("last_error") else "") + ".", ""]
    if not raw:
        return "\n".join(L + ["(no orders yet)"]) + "\n"
    d = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    for c in ("size", "matched", "locked_pnl", "seconds_to_off"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    q = d[d.action == "quote"]
    g = d[d.action == "green"]
    h = d[d.action == "hedge"]
    sp = d[d.action.isin(["sp_close", "hold"])]
    L += [f"Races quoted {q.market_id.nunique()}; quotes placed {len(q)}; dogs greened up {len(g)}; locked "
          f"**GBP{g.locked_pnl.sum():+.2f}**; to the SP or held {len(sp)} (settled by the result).", "",
          "| Market | Quotes | Hedges | Dogs greened | Locked | Running | SP / held |", "|---|---|---|---|---|---|---|"]
    run = 0.0
    for mid, x in d.groupby("market_id", sort=False):
        lk = float(x[x.action == "green"].locked_pnl.sum())
        run += lk
        L.append(f"| {mid} | {int((x.action == 'quote').sum())} | {int((x.action == 'hedge').sum())} | "
                 f"{int((x.action == 'green').sum())} | {lk:+.2f} | {run:+.2f} | "
                 f"{int(x.action.isin(['sp_close', 'hold']).sum())} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from betfair_recorder import uk_time_on, uk_today
    from sources.common import Store
    from greyhound.live import kill_switch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--trade", action="store_true")
    ap.add_argument("--live", action="store_true", help="real orders (only with GREYHOUND_QUOTE_MID=live)")
    ap.add_argument("--until", default="21:30")
    ap.add_argument("--date", default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    out = {}
    if a.trade:
        from trading.exchange import BetfairData, LiveExchange
        live = a.live and switch_on()
        if a.live and not live:
            log.warning("GREYHOUND_QUOTE_MID is not 'live': the decisions are logged, nothing is sent")
        mode = "live" if live else "paper"
        cfg = dict(CONFIG)
        if a.margin is not None:
            cfg["margin"] = float(a.margin)
        x = LiveExchange(BetfairData(source="greyhound"))
        x.login()
        ledger = Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound")) / \
            f"quote_mid_{mode}" / f"{day:%Y-%m-%d}" / "ledger.csv"
        log.info("quotes around the mid %s: margin %.2f, GBP%.2f a quote (stake GBP%.2f-%.2f), day stop GBP%.2f",
                 mode, cfg["margin"], cfg["liability"], cfg["min_stake"], cfg["max_stake"], cfg["day_loss_limit"])
        out["trade"] = QuoteMid(store, x, cfg, day, live, ledger, kill=kill_switch(store)).run(uk_time_on(day, a.until))
    if a.table:
        print(table(store, day))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
