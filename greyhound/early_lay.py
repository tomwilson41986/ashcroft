"""The early lay, live (the owner, 3 Nov 2026: "lets go live now, i will monitor and stop manually if any concerns";
"use early lay"). The forward test's primary rule (greyhound/laytest.py PRIMARY), traded on the owner's account:

- each runner's first book with a trade (its ``first_traded`` mark, as research.greyhound.limit_study reads it), kept
  as the market is polled; once every runner in the race has one and the day's prices cover the field, the race is
  priced once: the SP forecast is the conditional logit fitted on every recorded day before today
  (``laytest.day_fit``, the forward test's own fit), on those books and the live model's price;
- every runner the forecast says will drift (forecast SP at or above the mid now) is laid at 80% of its forecast SP,
  on Betfair's ladder (rounded down), at a price up to ``max_price``: a resting LIMIT lay sized by its liability, persistence
  LAPSE (whatever is unmatched lapses at the off), matched at once where the book already offers it;
- ``green_seconds`` before the scheduled off, the unmatched part is cancelled and each matched lay is greened up: a back
  at the best back price for ``matched x lay price / back price`` (fill or kill), the profit or loss then the same
  whatever the result; tried again each poll while the price moves, and ``give_up_seconds`` before the off, a back at
  the Betfair SP for the lay's stake instead (the 'sp' close), or, under Betfair's smallest stake, held to the result.

Size: each lay's liability ``liability`` (the owner, 3 Nov: "Important to get matched"), its stake between
``min_stake`` and ``max_stake``. No loss limit (the owner, 3 Nov: "i will monitor and stop manually if any concerns"); a guard against a
fault, ``max_open_liability``, stops new lays while that much is laid and not closed. The S3 objects ``greyhound/STOP`` and ``trading/STOP`` stop new lays and cancel the resting ones
within a poll, and the matched lays are still greened up. Real orders only with ``--live`` and
``GREYHOUND_EARLY_LAY=live``; otherwise the same decisions are logged and nothing is sent. The ledger:
``sources/greyhound/early_lay/<mode>/<day>/ledger.csv``. The positions are read from Betfair's own orders (the
``el-`` references), so a restarted session greens up what an earlier one laid.

    GREYHOUND_EARLY_LAY=live python -m greyhound.early_lay --trade --live --until 21:30
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
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PREFIX = "greyhound/early_lay"
REF = "el-"                                   # every order of this trader: el-<mmdd>-<market>-<selection>-<kind>
CONFIG = {
    "liability": 20.0,                        # each lay's liability (stake x (price - 1)); the owner, 3 Nov: size
    "min_stake": 2.0,                         # to get matched, not the paper's GBP100 (see stake_for)
    "max_stake": 10.0,
    "margin": 0.2,                            # lay at the forecast SP less this share
    "max_price": 20.0,                        # no lay above this price (greyhound.tradeout.MAX_PRICE)
    "min_order": 1.0,                         # Betfair's smallest stake on a limit order
    "min_sp_back": 2.0,                       # the smallest back at the SP that closes a lay (greyhound.tradeout)
    "daily_loss_limit": None,                 # the owner, 3 Nov: no loss limit, watched and stopped by hand
    "max_open_liability": 1000.0,             # a guard against a fault, not a limit on the trading
    "watch_minutes": 60.0,                    # markets off within this are watched for their first trades
    "min_rest_seconds": 60.0,                 # no lay placed this close to the green-up
    "green_seconds": 30.0,                    # green up from this many seconds before the scheduled off
    "give_up_seconds": 8.0,                   # then a back at the SP instead
    "poll_seconds": 15.0,
    "near_poll_seconds": 4.0,                 # the markets near their green-up
}
LEDGER_FIELDS = ["time_utc", "mode", "market_id", "selection_id", "track", "race_time", "trap", "dog", "action",
                 "price", "size", "matched", "avg_price", "f_bsp", "mid", "p_model", "bet_id", "status", "error",
                 "locked_pnl", "liability", "lay_matched", "lay_price", "seconds_to_off"]


def switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("GREYHOUND_EARLY_LAY", "")).strip().lower() == "live"


def _floor2(x: float) -> float:
    return math.floor(float(x) * 100 + 1e-9) / 100


# --------------------------------------------------------------------------------------------------------------------
# The decision: the forecast at the first traded books, the drifters, their lay prices
# --------------------------------------------------------------------------------------------------------------------

def stake_for(price: float, cfg: dict) -> float:
    """A lay's stake: its liability ``cfg['liability']`` at ``price``, between ``min_stake`` and ``max_stake``."""
    s = cfg["liability"] / max(price - 1.0, 1e-9)
    return _floor2(min(max(s, cfg["min_stake"]), cfg["max_stake"]))


def first_rows(first: dict, prev: dict, preds: pd.DataFrame) -> pd.DataFrame:
    """One market's runners at their first traded book (``first``: selection -> that book's fields), with the book 90
    seconds or more before it (``prev``: selection -> (back1, lay1) or None) and our price, in the columns
    greyhound.tradeout.add_features reads."""
    p = preds.set_index("selection_id")
    rows = []
    for sid, b in first.items():
        pv = prev.get(sid) or (np.nan, np.nan)
        rows.append({"race_id": str(b["market_id"]), "market_id": str(b["market_id"]), "selection_id": sid,
                     "mark": "first_traded", "p_model": float(p.loc[sid, "p_model"]), "back1": b["back1"],
                     "back1_size": b["back1_size"], "lay1": b["lay1"], "lay1_size": b["lay1_size"],
                     "last_traded": b["last_traded"], "runner_total_matched": b["traded"],
                     "market_total_matched": b["market_traded"], "back1_prev": pv[0], "lay1_prev": pv[1],
                     "bsp": np.nan, "won": 0})
    d = pd.DataFrame(rows)
    d["p_model"] = d.p_model / d.p_model.sum()                # renormalised over the runners in the book
    return d


def vol_in_fit(fit: dict) -> bool:
    """Whether the fit's books carried each runner's traded volume: its volume terms (``*_x_vol``) varied in
    training. The recorder's greyhound books had none (reports/greyhound_roadmap.md M4), so the terms were constant
    there, and a live book's volume would move the forecast where the fit never learnt it."""
    cols = list(fit.get("cols", []))
    idx = [cols.index(c) for c in ("lq_market_x_vol", "lq_model_x_vol") if c in cols]
    if not idx:
        return False
    mu, sd = np.asarray(fit["mu"], float), np.asarray(fit["sd"], float)
    return bool(any(abs(mu[i]) > 1e-9 or abs(sd[i] - 1.0) > 1e-9 for i in idx))


def decide(d: pd.DataFrame, fit: dict, cfg: dict) -> pd.DataFrame:
    """The race's lays: the forecast SP for every runner, the drift gate, the lay price on the ladder (rounded down)
    and whether it is placed."""
    from greyhound import tradeout as to
    from research.greyhound import limit_study as ls, tradeout_study as ts
    if not vol_in_fit(fit):
        d = d.assign(runner_total_matched=np.nan)            # as the fit saw it: no volume in its books
    with np.errstate(divide="ignore", invalid="ignore"):
        x = to.add_features(d)
        q, book = ts.predict(x, fit)
    x = x.assign(f_bsp=1.0 / (book * np.asarray(q, float)))
    x["drift"] = ~(x.f_bsp.to_numpy(float) < x["mid"].to_numpy(float))       # limit_study.gate_flags' 'drift'
    x["lay_price"] = ls.ladder(x.f_bsp.to_numpy(float) * (1 - cfg["margin"]), up=False)
    ok = np.isfinite(x.lay_price) & (x.lay_price > 1.0) & (x.lay_price <= cfg["max_price"])
    x["lay"] = x.drift & ok
    return x


# --------------------------------------------------------------------------------------------------------------------
# The orders (the trader's own calls on the exchange's session: trading/ is not changed for another sport)
# --------------------------------------------------------------------------------------------------------------------

class Orders:
    """Resting lays, their cancellation, Betfair's record of them, and the green-up backs, on a
    trading.exchange.LiveExchange's session. ``live`` False: nothing is sent; every call says what it would do."""

    def __init__(self, x, live: bool):
        self.x, self.live = x, live

    def lay(self, mid: str, sid: int, price: float, size: float, ref: str) -> dict:
        if not self.live:
            return {"status": "PAPER", "matched": 0.0, "avg_price": 0.0, "bet_id": ""}
        ins = {"selectionId": int(sid), "handicap": 0, "side": "LAY", "orderType": "LIMIT",
               "customerOrderRef": ref[:32],
               "limitOrder": {"size": _floor2(size), "price": float(price), "persistenceType": "LAPSE"}}
        reply = self.x._place(mid, ins, ref)
        if reply is None:                                    # placed or not: Betfair's own record decides
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

    def cancel(self, mid: str) -> str:
        """Every unmatched order of ours on the market (only this trader bets on greyhound markets)."""
        if not self.live:
            return "PAPER"
        try:
            reply = self.x._call("cancelOrders", {"marketId": mid}) or {}
            return str(reply.get("status") or "")
        except Exception as exc:
            return f"FAILURE: {exc}"[:200]

    def mine(self, mid: str) -> list[dict]:
        if not self.live:
            return []
        return [o for o in self.x.orders([mid]) if str(o.get("customerOrderRef") or "").startswith(REF)]

    def back(self, mid: str, sid: int, price: float, size: float, ref: str):
        if not self.live:
            return None
        return self.x.back(mid, sid, price, size, ref, min_fill=size)

    def back_at_sp(self, mid: str, sid: int, stake: float, ref: str) -> dict:
        if not self.live:
            return {"status": "PAPER"}
        ins = {"selectionId": int(sid), "handicap": 0, "side": "BACK", "orderType": "MARKET_ON_CLOSE",
               "customerOrderRef": ref[:32], "marketOnCloseOrder": {"liability": _floor2(stake)}}
        reply = self.x._place(mid, ins, ref)
        if reply is None:
            got = [o for o in self.mine(mid) if o.get("customerOrderRef") == ref[:32]]
            return {"status": "PENDING" if got else "FAILURE", "error": "reply lost"}
        rep, ok = self.x._report(reply)
        return {"status": "PENDING" if ok else "FAILURE", "bet_id": str(rep.get("betId") or ""),
                "error": "" if ok else self.x._reason(reply, rep, "FAILURE")}


def positions(orders: list[dict]) -> dict:
    """Per selection from Betfair's orders on one market: the lays matched (stake, average price) and whether a
    green-up back (``-g``) or an SP back (``-s``) is already on."""
    out: dict = {}
    for o in orders:
        sid = int(o.get("selectionId", 0))
        p = out.setdefault(sid, {"lay_matched": 0.0, "lay_value": 0.0, "closed": False})
        ref = str(o.get("customerOrderRef") or "")
        if o.get("side") == "LAY":
            m = float(o.get("sizeMatched") or 0)
            p["lay_matched"] += m
            p["lay_value"] += m * float(o.get("averagePriceMatched") or 0)
        elif o.get("side") == "BACK" and (ref.endswith("-g") or ref.endswith("-s")):
            if float(o.get("sizeMatched") or 0) > 0 or ref.endswith("-s"):
                p["closed"] = True
    for p in out.values():
        p["lay_price"] = p["lay_value"] / p["lay_matched"] if p["lay_matched"] > 0 else None
    return out


# --------------------------------------------------------------------------------------------------------------------
# The trader
# --------------------------------------------------------------------------------------------------------------------

class EarlyLay:
    def __init__(self, store, x, cfg: dict, day: date, live: bool, ledger_path: Path, fit: dict,
                 kill=lambda: False, now=lambda: datetime.now(timezone.utc)):
        self.store, self.x, self.cfg, self.day, self.live = store, x, cfg, day, live
        self.mode = "live" if live else "paper"
        self.orders = Orders(x, live)
        self.ledger_path, self.fit, self.kill, self.now = ledger_path, fit, kill, now
        self.first: dict = {}          # market -> selection -> its first traded book
        self.hist: dict = {}           # (market, selection) -> [(time, back1, lay1)]
        self.priced: set = set()       # markets decided (laid or not)
        self.greened: set = set()      # markets closed out
        self.locked = 0.0              # the day's P&L locked by green-ups
        self.open_liability: dict = {}  # market -> liability not yet closed at a known P&L
        self.laid: dict = {}           # market -> liability of the lays placed and not yet closed
        self.preds: pd.DataFrame | None = None
        self._loaded_at = None
        self._beat_at = None
        self.last_error = ""
        self._resume()

    # the record ----------------------------------------------------------------------------------------------------
    def _resume(self) -> None:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open() as f:
            for r in csv.DictReader(f):
                if r["action"] in ("lay", "no_lay", "skip"):
                    self.priced.add(r["market_id"])
                if r["action"] in ("green", "sp_close", "hold", "flat"):
                    self.greened.add(r["market_id"])
                if r["action"] == "green" and r.get("locked_pnl"):
                    self.locked += float(r["locked_pnl"])
                if r["action"] in ("sp_close", "hold") and r.get("liability"):
                    self.open_liability[r["market_id"]] = self.open_liability.get(r["market_id"], 0.0) + \
                        float(r["liability"])

    def _write(self, rows: list[dict]) -> None:
        new = not self.ledger_path.exists()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
        try:
            self.store.put(f"{PREFIX}/{self.mode}/{self.day:%Y-%m-%d}/ledger.csv", self.ledger_path.read_bytes())
        except Exception as exc:                                # the local ledger is the record; S3 its copy
            log.warning("ledger not copied to S3 (%s)", exc)

    def day_loss(self) -> float:
        """The day's P&L as far as it can be lost: locked green-ups less every open liability."""
        return self.locked - sum(self.open_liability.values())

    def load_predictions(self) -> None:
        t = self.now()
        if self._loaded_at and (t - self._loaded_at).total_seconds() < 600:
            return
        raw = self.store.get(f"greyhound/live/predictions/{self.day:%Y-%m-%d}.csv")
        self._loaded_at = t
        if raw:
            p = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
            p["selection_id"] = pd.to_numeric(p.selection_id, errors="coerce")
            p = p.dropna(subset=["selection_id"]).drop_duplicates(["market_id", "selection_id"], keep="last")
            p["off"] = pd.to_datetime(p.race_date + " " + p.race_time.astype(str).str[:5]).dt.tz_localize(
                "Europe/London").dt.tz_convert("UTC")
            self.preds = p

    def heartbeat(self, force: bool = False, **extra) -> None:
        """The trader's state each minute to S3 (``status.json``), read by the running table: the job's own log is
        printed only when the day ends."""
        t = self.now()
        if not force and self._beat_at and (t - self._beat_at).total_seconds() < 60:
            return
        self._beat_at = t
        st = {"time_utc": pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
              "markets_watched": len(self.first), "markets_priced": len(self.priced),
              "markets_closed": len(self.greened), "locked_pnl": round(self.locked, 2),
              "open_liability": round(sum(self.open_liability.values()), 2),
              "laid_not_closed": round(sum(self.laid.values()), 2), "stopped": bool(self.kill()),
              "prices_loaded": self.preds is not None, "last_error": self.last_error[:300], **extra}
        try:
            self.store.put(f"{PREFIX}/{self.mode}/{self.day:%Y-%m-%d}/status.json", json.dumps(st).encode())
        except Exception as exc:
            log.warning("status not written (%s)", exc)

    # one poll ------------------------------------------------------------------------------------------------------
    def step(self, near_only: bool = False) -> None:
        self.load_predictions()
        if self.preds is None:
            return
        t = pd.Timestamp(self.now())
        stopped = self.kill()
        to_off = (self.preds.off - t).dt.total_seconds()
        horizon = self.cfg["green_seconds"] + 2 * self.cfg["near_poll_seconds"] + 5 if near_only \
            else self.cfg["watch_minutes"] * 60
        # a race laid and not yet closed stays watched past its scheduled off (a race off late is still open)
        late = self.preds.market_id.isin(self.priced) & (to_off > -180)
        watch = self.preds[((to_off > 0) | late) & (to_off <= horizon) & ~self.preds.market_id.isin(self.greened)]
        mids = sorted(watch.market_id.dropna().astype(str).unique())
        if not mids:
            return
        books = self.x.books(mids)
        for mid in mids:
            b = books.get(mid)
            if b is None:
                continue
            g = watch[watch.market_id.astype(str) == mid]
            secs = float((g.off.iloc[0] - t).total_seconds())
            try:
                if secs <= self.cfg["green_seconds"]:
                    self.green(mid, b, g, secs)
                elif not near_only:
                    self.watch(mid, b, g, secs, stopped)
                    if stopped and mid in self.priced:
                        self.orders.cancel(mid)             # the resting lays come off; the matched are greened later
            except Exception as exc:                         # one market's failure never stops the others
                log.warning("%s: %s", mid, exc)
                self.last_error = f"{mid}: {exc}"

    def watch(self, mid: str, b, g: pd.DataFrame, secs: float, stopped: bool) -> None:
        if b.status != "OPEN" or b.inplay:
            return
        t = self.now()
        active = {sid: q for sid, q in b.runners.items() if q.status == "ACTIVE"}
        first = self.first.setdefault(mid, {})
        for sid, q in active.items():
            bb, bl = q.best_back, q.best_lay
            h = self.hist.setdefault((mid, sid), [])
            h.append((t, bb[0] if bb else np.nan, bl[0] if bl else np.nan))
            if sid not in first and q.last_traded and q.last_traded > 1 and bb:
                first[sid] = {"market_id": mid, "polled": t, "back1": bb[0], "back1_size": bb[1],
                              "lay1": bl[0] if bl else np.nan, "lay1_size": bl[1] if bl else np.nan,
                              "last_traded": float(q.last_traded), "traded": float(q.traded or 0.0),
                              "market_traded": float(b.total_matched or 0.0)}
        if mid in self.priced or secs < self.cfg["green_seconds"] + self.cfg["min_rest_seconds"]:
            return
        if not active or any(sid not in first for sid in active):
            return                                           # not every runner has traded yet
        priced = set(g.selection_id.dropna().astype(int))
        if not set(active) <= priced:
            return                                           # a runner our prices lack: looked at again next poll
        self.priced.add(mid)
        info = g.set_index(g.selection_id.astype(int))
        base = {"time_utc": pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode, "market_id": mid,
                "seconds_to_off": round(secs)}
        limit = self.cfg.get("daily_loss_limit")
        resting = sum(self.laid.values())
        if stopped or (limit is not None and self.day_loss() <= -limit) \
                or resting >= self.cfg["max_open_liability"]:
            why = "stopped (S3 STOP)" if stopped else ("the day's loss limit" if limit is not None and
                                                       self.day_loss() <= -limit else "the open liability guard")
            self._write([{**base, "action": "skip", "error": why}])
            return
        prev = {}
        for sid, fb in first.items():
            if sid not in active:
                continue
            before = [(bk, ly) for (ts_, bk, ly) in self.hist.get((mid, sid), [])
                      if (fb["polled"] - ts_).total_seconds() >= 90]
            prev[sid] = before[-1] if before else None
        d = decide(first_rows({s: first[s] for s in active}, prev, g[g.selection_id.isin(list(active))]),
                   self.fit, self.cfg)
        rows = []
        for r in d.itertuples(index=False):
            sid = int(r.selection_id)
            row = {**base, "selection_id": sid, "track": info.loc[sid, "track"], "race_time": info.loc[sid, "race_time"],
                   "trap": info.loc[sid, "trap"], "dog": info.loc[sid, "dog_name"] if "dog_name" in info else "",
                   "f_bsp": round(float(r.f_bsp), 3), "mid": round(float(r.mid), 3), "p_model": round(float(r.p_model), 4)}
            if not r.lay:
                rows.append({**row, "action": "no_lay", "error": "forecast to shorten" if not r.drift else "price"})
                continue
            price = float(r.lay_price)
            size = stake_for(price, self.cfg)
            ref = f"{REF}{self.day:%m%d}-{mid[-8:]}-{sid}-l"
            fill = self.orders.lay(mid, sid, price, size, ref)
            self.laid[mid] = self.laid.get(mid, 0.0) + size * (price - 1)
            rows.append({**row, "action": "lay", "price": price, "size": size, "matched": fill.get("matched"),
                         "avg_price": fill.get("avg_price"), "bet_id": fill.get("bet_id"), "status": fill.get("status"),
                         "error": fill.get("error", ""), "liability": round(size * (price - 1), 2)})
        if rows:
            self._write(rows)

    def green(self, mid: str, b, g: pd.DataFrame, secs: float) -> None:
        """Close the market's matched lays: cancel what rests, then back each at the best back price, else at the SP
        near the off, else hold."""
        if mid not in self.priced:
            self.greened.add(mid)
            return
        self.orders.cancel(mid)
        if not self.live:
            self.greened.add(mid)
            self.laid.pop(mid, None)
            self._write([{"time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
                          "market_id": mid, "action": "flat", "error": "paper: no orders to close",
                          "seconds_to_off": round(secs)}])
            return
        pos = positions(self.orders.mine(mid))
        if b.status != "OPEN" or b.inplay:
            if any(p["lay_matched"] > 0 and not p["closed"] for p in pos.values()):
                self._held(mid, pos, secs, "the market closed before the green-up")
            self.greened.add(mid)
            return
        rows, left = [], False
        base = {"time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
                "market_id": mid, "seconds_to_off": round(secs)}
        for sid, p in pos.items():
            if p["lay_matched"] <= 0 or p["closed"]:
                continue
            S, m = _floor2(p["lay_matched"]), p["lay_price"]
            q = b.runners.get(sid)
            bb = q.best_back if q else None
            if bb and secs > self.cfg["give_up_seconds"]:
                B = bb[0]
                stake = _floor2(S * m / B)
                if stake >= self.cfg["min_order"]:
                    fill = self.orders.back(mid, sid, B, stake, f"{REF}{self.day:%m%d}-{mid[-8:]}-{sid}-g")
                    if fill is not None and fill.status == "SUCCESS" and fill.matched + 0.01 >= stake:
                        B = float(fill.avg_price or B)
                        locked = round(S * (1 - m / B), 2)          # the same whatever the result
                        self.locked += locked
                        p["closed"] = True
                        rows.append({**base, "selection_id": sid, "action": "green", "price": B, "size": stake,
                                     "lay_matched": S, "lay_price": round(m, 2),
                                     "matched": fill.matched, "avg_price": fill.avg_price, "bet_id": fill.bet_id,
                                     "status": fill.status, "locked_pnl": locked, "liability": round(S * (m - 1), 2)})
                        continue
                    left = True                               # the price moved: tried again next poll
                    rows.append({**base, "selection_id": sid, "action": "green_retry", "price": B, "size": stake,
                                 "status": getattr(fill, "status", ""), "error": getattr(fill, "error", "")})
                    continue
            if secs <= self.cfg["give_up_seconds"] or not bb or _floor2(S * m / (bb[0] if bb else m)) < self.cfg["min_order"]:
                rows += self._close_at_sp(mid, sid, S, m, base)
                p["closed"] = True
            else:
                left = True
        if rows:
            self._write(rows)
        if not left:
            self.greened.add(mid)
            self.laid.pop(mid, None)

    def _close_at_sp(self, mid: str, sid: int, S: float, m: float, base: dict) -> list[dict]:
        liab = round(S * (m - 1), 2)
        self.open_liability[mid] = self.open_liability.get(mid, 0.0) + liab
        if S + 1e-9 < self.cfg["min_sp_back"]:
            return [{**base, "selection_id": sid, "action": "hold", "size": S, "avg_price": m, "liability": liab,
                     "lay_matched": S, "lay_price": round(m, 2),
                     "error": "under the smallest SP back: held to the result"}]
        r = self.orders.back_at_sp(mid, sid, S, f"{REF}{self.day:%m%d}-{mid[-8:]}-{sid}-s")
        return [{**base, "selection_id": sid, "action": "sp_close", "size": S, "avg_price": m, "liability": liab,
                 "lay_matched": S, "lay_price": round(m, 2),
                 "status": r.get("status"), "bet_id": r.get("bet_id", ""), "error": r.get("error", "")}]

    def _held(self, mid: str, pos: dict, secs: float, why: str) -> None:
        self.laid.pop(mid, None)
        base = {"time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
                "market_id": mid, "seconds_to_off": round(secs)}
        rows = []
        for sid, p in pos.items():
            if p["lay_matched"] > 0 and not p["closed"]:
                liab = round(p["lay_matched"] * (p["lay_price"] - 1), 2)
                self.open_liability[mid] = self.open_liability.get(mid, 0.0) + liab
                rows.append({**base, "selection_id": sid, "action": "hold", "size": p["lay_matched"],
                             "avg_price": p["lay_price"], "liability": liab, "error": why,
                             "lay_matched": p["lay_matched"], "lay_price": round(p["lay_price"], 2)})
        if rows:
            self._write(rows)

    def run(self, until: datetime, sleep=time.sleep) -> dict:
        last_full = None
        while self.now() < until:
            try:
                if last_full is not None and (self.now() - last_full).total_seconds() < self.cfg["poll_seconds"]:
                    self.step(near_only=True)
                else:
                    last_full = self.now()
                    self.step()
            except Exception as exc:                            # one failed poll: the next tries again
                log.warning("poll failed (%s)", exc)
                self.last_error = f"poll: {exc}"
            self.heartbeat()
            sleep(self.cfg["near_poll_seconds"])
        self.heartbeat(force=True, finished=True)
        return {"day": f"{self.day:%Y-%m-%d}", "mode": self.mode, "markets_priced": len(self.priced),
                "markets_closed": len(self.greened), "locked_pnl": round(self.locked, 2),
                "open_liability": round(sum(self.open_liability.values()), 2)}


def table(store, day: date, mode: str = "live") -> str:
    """The day's live early lay so far, a row per race, from the trader's ledger in S3: the lays placed and their
    stake, what matched by the green-up, how each was closed, the P&L locked by the green-ups (the SP closes and the
    held lays settle with the result, shown as open liability)."""
    raw = store.get(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/ledger.csv")
    head = f"### Early lay, {mode} orders on the owner's account ({day:%Y-%m-%d})"
    st = store.get(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/status.json")
    if st:
        s = json.loads(st)
        head += (f"\n\nTrader: last seen {s.get('time_utc')} UTC; watching {s.get('markets_watched')} races, priced "
                 f"{s.get('markets_priced')}, closed {s.get('markets_closed')}"
                 + ("; **STOPPED** (S3 STOP)" if s.get("stopped") else "")
                 + (f"; {s['state']}" if s.get("state") else "")
                 + (f"; last error: {s['last_error']}" if s.get("last_error") else "") + ".")
    else:
        head += "\n\nTrader: no status yet (not started, or not running)."
    if not raw:
        return f"{head}\n\nno ledger yet\n"
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    lays = led[led.action == "lay"]
    closes = led[led.action.isin(["green", "sp_close", "hold"])]
    skips = led[led.action == "skip"]
    rows, run = [], 0.0
    for mid in list(dict.fromkeys(lays.market_id)):
        lg, cg = lays[lays.market_id == mid], closes[closes.market_id == mid]
        locked = float(pd.to_numeric(cg.locked_pnl, errors="coerce").fillna(0).sum())
        run += locked
        opened = cg[cg.action.isin(["sp_close", "hold"])]
        rows.append({"race": str(lg.race_time.iloc[0]), "track": str(lg.track.iloc[0]), "lays": len(lg),
                     "staked": round(float(lg["size"].sum()), 2),
                     "matched": round(float(pd.to_numeric(cg.lay_matched, errors="coerce").fillna(0).sum()), 2),
                     "greened": int((cg.action == "green").sum()), "sp_or_held": len(opened),
                     "open_liability": round(float(pd.to_numeric(opened.liability, errors="coerce").fillna(0).sum()), 2),
                     "locked": round(locked, 2), "running": round(run, 2),
                     "errors": "; ".join(sorted(set(lg.error.dropna().astype(str)) - {""}))[:80]})
    L = [head, "", f"Races laid {len(rows)}; lays {len(lays)}, GBP{lays['size'].sum():.2f} asked; matched "
         f"GBP{sum(r['matched'] for r in rows):.2f}; locked by green-ups **GBP{run:+.2f}**; still settling with the "
         f"result (closed at the SP or held) GBP{sum(r['open_liability'] for r in rows):.2f} liability."
         + (f" Skipped races: {len(skips)} ({'; '.join(sorted(set(skips.error.astype(str))))})." if len(skips) else ""),
         "", "| Race (UK) | Track | Lays | Stake asked | Matched | Greened | SP/held | Open liability | Locked | Running "
         "| Errors |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["race"]):
        L.append(f"| {r['race']} | {r['track']} | {r['lays']} | {r['staked']} | {r['matched']} | {r['greened']} | "
                 f"{r['sp_or_held']} | {r['open_liability']} | {r['locked']} | {r['running']} | {r['errors']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from betfair_recorder import uk_time_on, uk_today
    from sources.common import Store
    from greyhound import laytest
    from greyhound.live import kill_switch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--trade", action="store_true")
    ap.add_argument("--live", action="store_true", help="real orders (only with GREYHOUND_EARLY_LAY=live)")
    ap.add_argument("--until", default="21:30")
    ap.add_argument("--date", default=None)
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    out = {}
    if a.trade:
        from trading.exchange import BetfairData, LiveExchange
        live = a.live and switch_on()
        if a.live and not live:
            log.warning("GREYHOUND_EARLY_LAY is not 'live': the decisions are logged, nothing is sent")
        mode = "live" if live else "paper"
        status_key = f"{PREFIX}/{mode}/{day:%Y-%m-%d}/status.json"
        store.put(status_key, json.dumps({"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                          "mode": mode, "state": "starting: fitting the forecast"}).encode())
        fit = laytest.day_fit(store, day)
        if fit is None:
            store.put(status_key, json.dumps({"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                              "mode": mode, "state": "stopped: no forecast fit for today"}).encode())
            raise SystemExit("no forecast fit for today: nothing traded")
        x = LiveExchange(BetfairData(source="greyhound"))
        x.login()
        ledger = Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound")) / \
            f"early_lay_{mode}" / f"{day:%Y-%m-%d}" / "ledger.csv"
        cfg = dict(CONFIG)
        log.info("early lay %s: GBP%.2f liability a lay (stake GBP%.2f-%.2f), fit to %s", mode, cfg["liability"],
                 cfg["min_stake"], cfg["max_stake"], fit.get("fit_to"))
        out["trade"] = EarlyLay(store, x, cfg, day, live, ledger, fit, kill=kill_switch(store)).run(
            uk_time_on(day, a.until))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
