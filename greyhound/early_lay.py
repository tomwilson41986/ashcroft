"""The early lay, live (the owner, 3 Nov 2026: "lets go live now, i will monitor and stop manually if any concerns";
"use early lay"). The forward test's primary rule (greyhound/laytest.py PRIMARY), traded on the owner's account:

- each runner's first book with a trade (its ``first_traded`` mark, as research.greyhound.limit_study reads it), kept
  as the market is polled; once every runner in the race has one and the day's prices cover the field, the race is
  priced once: the SP forecast is the conditional logit fitted on every recorded day before today
  (``laytest.day_fit``, the forward test's own fit), on those books and the live model's price;
- every runner the forecast says will drift (forecast SP at or above the mid now) is laid at 80% of its forecast SP,
  on Betfair's ladder (rounded down), at a price up to ``max_price``: a resting LIMIT lay sized by its liability, persistence
  LAPSE (whatever is unmatched lapses at the off), matched at once where the book already offers it;
- ``green_seconds`` (5) before the scheduled off, the unmatched part is cancelled and each matched lay is greened up: a back
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound import costs

log = logging.getLogger(__name__)

PREFIX = "greyhound/early_lay"
REF = "el-"                                   # every order of this trader: el-<mmdd>-<market>-<selection>-<kind>
CONFIG = {
    "liability": 100.0,                       # GBP100 from 11 Nov 2026 (the owner: "GBP100 a lay", the forward test's
                                              # stake: greyhound.tradeout TO_WIN, stakes GBP2-50); GBP20 3-11 Nov. Each lay's liability (stake x (price - 1)). The owner, 3 Nov 14:20: cut to
    "min_stake": 2.0,                         # GBP20 while the paper is rebuilt on the live feed (live and paper laid
    "max_stake": 50.0,                        # different dogs); GBP100 again only once that shows an edge
    "margin": 0.20,                           # lay at the forecast SP less this share. 0.2 to 4 Nov, 0.10 on 4 Nov
                                              # (3 Nov's live books in the corrected simulator: +GBP3.82 on 26 fills
                                              # against 0.2's -GBP3.93, within noise); 0.2 again from 5 Nov, the owner,
                                              # 4 Nov: "revert the early lay margin to 20%" (the limit study per track
                                              # group, fast loop batch 10: 0.2 ahead of 0.1 in both groups); 0.10 kept
                                              # on paper beside it (PAPER_VARIANTS)
    "gate": "drift",                          # which dogs: 'drift' (forecast SP at or above the mid now) or 'short'
                                              # (the mid now under SHORT_PRICE, the limit study's 'short' gate)
    "max_price": 20.0,                        # no lay above this price (greyhound.tradeout.MAX_PRICE)
    "min_order": 1.0,                         # Betfair's smallest stake on a limit order
    "min_sp_back": 2.0,                       # the smallest back at the SP that closes a lay (greyhound.tradeout)
    "daily_loss_limit": None,                 # GBP; live trading needs one (the owner, 5 Nov 2026, the audit: "set a
                                              # daily loss limit"): GREYHOUND_EARLY_LAY_DAILY_LOSS, else no live orders
    "close_margin_seconds": 120.0,            # no lay on a race off later than this before the trader stops
    "overrun_minutes": 15.0,                  # past --until, the trader stays to close what it laid, then cancels
    "max_open_liability": 5000.0,             # a guard against a fault, not a limit on the trading (50 lays' worth,
                                              # as GBP1,000 was at GBP20 a lay)
    "rest_only": False,                       # the owner, 11 Nov 2026: "lose the guard. I want to be exactly like the
                                              # rule that was profitable in test" (the forward test's rule lays at 80%
                                              # of the forecast SP whatever the mid: on 11 Nov the guard blocked 76
                                              # lays, the strongest forecast drifts). True: never at or above the mid
                                              # (the owner, 3 Nov, after the first live lays above the mid lost)
    # the pre-trade gate (the owner, 5 Nov 2026, the audit A5: "Work on the next items"); guards derived from the
    # settings above, for the owner to set: a race's whole field at the lay liability, an order a runner a race at a
    # race a minute; the commission is the account's rate on greyhounds (greyhound/costs.py; GREYHOUND_COMMISSION
    # overrides it for a run)
    "max_market_liability": 800.0,            # 8 runners x GBP100
    "max_orders_per_minute": 30,
    "commission": costs.COMMISSION,
    "watch_minutes": 60.0,                    # markets off within this are watched for their first trades
    "min_rest_seconds": 60.0,                 # no lay placed this close to the green-up
    "green_seconds": 5.0,                     # green up from this many seconds before the scheduled off (the
    "give_up_seconds": 2.0,                   # owner, 3 Nov: "green up 5s before the off"); then a back at the SP
    "poll_seconds": 5.0,                      # the owner, 3 Nov: "trade as frequently as the live feed enables"
    "near_poll_seconds": 1.0,                 # the markets near their green-up, every second
}
SHORT_PRICE = 3.0                             # research.greyhound.limit_study.SHORT_PRICE
# The paper variants (the owner, 4 Nov 2026: "revert the early lay margin to 20%, run this on paper. Keep running 10%
# too"), traded beside the live rule on the same books, nothing sent: the live rule at 10% (live to 4 Nov), and the
# short-priced lay (the mid under 3 at first traded; the limit study's lead: first traded, bid, lay 0.2, short, 1,891
# dogs, +3.34% a trade at the last price, 90% 2.46 to 4.32, fast loop batch 9) at 20% and at 10%. Ledgers under
# greyhound/early_lay/paper-<name>/<day>/; each scored on the live-key books (paper_table).
PAPER_VARIANTS = {
    "m10": {"margin": 0.10},
    "short20": {"gate": "short", "margin": 0.20},
    "short10": {"gate": "short", "margin": 0.10},
}
LEDGER_FIELDS = ["time_utc", "mode", "market_id", "selection_id", "track", "race_time", "trap", "dog", "action",
                 "price", "size", "matched", "avg_price", "f_bsp", "mid", "p_model", "bet_id", "status", "error",
                 "locked_pnl", "liability", "lay_matched", "lay_price", "seconds_to_off"]


def daily_loss_limit(env=None) -> float | None:
    """The live trader's daily loss limit in GBP (GREYHOUND_EARLY_LAY_DAILY_LOSS, a positive number), else None: and
    without one main() sends no order (the owner, 5 Nov 2026: "set a daily loss limit")."""
    env = os.environ if env is None else env
    try:
        v = float(str(env.get("GREYHOUND_EARLY_LAY_DAILY_LOSS", "")).strip())
    except ValueError:
        return None
    return v if math.isfinite(v) and v > 0 else None


def _terminate(signum, frame):
    raise SystemExit(f"signal {signum}")


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
    gate = cfg.get("gate", "drift")
    if gate not in ("drift", "short"):
        raise ValueError(f"gate must be drift or short, not {gate!r}")
    from greyhound import tradeout as to
    from research.greyhound import limit_study as ls, tradeout_study as ts
    if not vol_in_fit(fit):
        d = d.assign(runner_total_matched=np.nan)            # as the fit saw it: no volume in its books
    with np.errstate(divide="ignore", invalid="ignore"):
        x = to.add_features(d)
        q, book = ts.predict(x, fit)
    x = x.assign(f_bsp=1.0 / (book * np.asarray(q, float)))
    x["drift"] = ~(x.f_bsp.to_numpy(float) < x["mid"].to_numpy(float))       # limit_study.gate_flags' 'drift'
    x["short"] = x["mid"].to_numpy(float) < SHORT_PRICE                       # limit_study.gate_flags' 'short'
    x["lay_price"] = ls.ladder(x.f_bsp.to_numpy(float) * (1 - cfg["margin"]), up=False)
    ok = np.isfinite(x.lay_price) & (x.lay_price > 1.0) & (x.lay_price <= cfg["max_price"])
    # the owner, 3 Nov 2026: never at or above the market's mid, so a lay only ever rests in the queue and never takes
    # the price on a forecast that overstates the drift (the first live lays: those above the mid lost GBP17.85 of 20.39)
    x["below_mid"] = x.lay_price.to_numpy(float) < x["mid"].to_numpy(float) - 1e-9
    x["lay"] = x[gate] & ok & (x.below_mid if cfg.get("rest_only", True) else True)
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
        """Every unmatched order of ours (``el-``) on the market, by bet id: never another trader's orders (the
        quotes around the mid trade the same markets; the audit A5)."""
        if not self.live:
            return "PAPER"
        try:
            ids = [str(o.get("betId")) for o in self.mine(mid)
                   if o.get("betId") and str(o.get("status") or "") != "EXECUTION_COMPLETE"]
            if not ids:
                return "NONE"
            reply = self.x._call("cancelOrders", {"marketId": mid,
                                                  "instructions": [{"betId": b} for b in ids]}) or {}
            return str(reply.get("status") or "")
        except Exception as exc:
            return f"FAILURE: {exc}"[:200]

    def mine(self, mid: str) -> list[dict]:
        if not self.live:
            return []
        return [o for o in self.x.orders([mid]) if str(o.get("customerOrderRef") or "").startswith(REF)]

    def back(self, mid: str, sid: int, price: float, size: float, ref: str, min_fill: float | None = None):
        if not self.live:
            return None
        return self.x.back(mid, sid, price, size, ref, min_fill=size if min_fill is None else min_fill)

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
    """Per selection from Betfair's orders on one market: the lays matched (stake, value = stake x price), the
    green-up backs matched so far (``-g``: stake, value), and whether an SP back (``-s``) is on. A lay is greened once
    the backs' value reaches the lays': ``to_cover`` is the value still open."""
    out: dict = {}
    for o in orders:
        sid = int(o.get("selectionId", 0))
        p = out.setdefault(sid, {"lay_matched": 0.0, "lay_value": 0.0, "back_matched": 0.0, "back_value": 0.0,
                                 "closed": False})
        ref = str(o.get("customerOrderRef") or "")
        m = float(o.get("sizeMatched") or 0)
        if o.get("side") == "LAY":
            p["lay_matched"] += m
            p["lay_value"] += m * float(o.get("averagePriceMatched") or 0)
        elif o.get("side") == "BACK" and ref.endswith("-g"):
            p["back_matched"] += m
            p["back_value"] += m * float(o.get("averagePriceMatched") or 0)
        elif o.get("side") == "BACK" and ref.endswith("-s"):
            p["closed"] = True
    for p in out.values():
        p["lay_price"] = p["lay_value"] / p["lay_matched"] if p["lay_matched"] > 0 else None
        p["to_cover"] = p["lay_value"] - p["back_value"]
    return out


def outcomes(p: dict) -> tuple[float, float]:
    """A selection's P&L if it wins and if it loses, from its lays and green-up backs."""
    win = -(p["lay_value"] - p["lay_matched"]) + (p["back_value"] - p["back_matched"])
    lose = p["lay_matched"] - p["back_matched"]
    return win, lose


def ladder_limit(levels: list, value: float, depth: int = 3) -> float | None:
    """The lowest back price to take, down the best ``depth`` prices, for backs worth ``value`` (stake x price):
    the price where the offered value first covers it, else the last of the levels."""
    cum, limit = 0.0, None
    for price, size in levels[:depth]:
        cum += price * size
        limit = price
        if cum + 1e-9 >= value:
            break
    return limit


def ladder_stake(levels: list, value: float, depth: int = 3) -> tuple[float | None, float]:
    """The green-up back for ``value`` (stake x price) down the best ``depth`` back prices: the lowest price to take and
    the stake that, filled best price first as Betfair fills a limit back, is worth ``value``. A stake sized at the
    lowest price alone (value / limit) is filled partly at the better prices above it and backs more than the lay's
    value: the over-cover inferred on 296 of 479 live green-ups, 1.4% over on their median, about 18% at the worst
    1% (the fill reconciliation's ledger, 3-13 Nov 2026; the owner, 14 Nov: "fix the closing order"). Where the levels hold less than the
    value, the rest is sized at the lowest price (a fill or kill takes what is there; the rest is tried next second)."""
    left, stake, limit = float(value), 0.0, None
    for price, size in levels[:depth]:
        if left <= 1e-9:
            break
        limit = price
        take = min(float(size), left / price)
        stake += take
        left -= take * price
    if left > 1e-9 and limit:
        stake += left / limit
    return limit, stake


# --------------------------------------------------------------------------------------------------------------------
# The trader
# --------------------------------------------------------------------------------------------------------------------

class EarlyLay:
    def __init__(self, store, x, cfg: dict, day: date, live: bool, ledger_path: Path, fit: dict,
                 kill=lambda: False, now=lambda: datetime.now(timezone.utc), book: str | None = None):
        if book is not None and live:
            raise ValueError("a named paper book never trades live")
        self.store, self.x, self.cfg, self.day, self.live = store, x, cfg, day, live
        self.mode = book or ("live" if live else "paper")       # the S3 folder and the ledger's mode
        self.orders = Orders(x, live)
        self.ledger_path, self.fit, self.kill, self.now = ledger_path, fit, kill, now
        self.first: dict = {}          # market -> selection -> its first traded book
        self.hist: dict = {}           # (market, selection) -> [(time, back1, lay1)]
        self.priced: set = set()       # markets decided (laid or not)
        self.greened: set = set()      # markets closed out
        self.locked = 0.0              # the day's P&L locked by green-ups
        self.open_liability: dict = {}  # market -> liability not yet closed at a known P&L
        self.laid: dict = {}           # market -> liability of the lays placed and not yet closed
        self._backs: dict = {}         # market -> bet id -> (selection, ref, matched, price): green-up backs seen
        self.preds: pd.DataFrame | None = None
        self.until: datetime | None = None            # set by run(): the time the trader stops laying
        self._sent: list = []                          # the times of the lays sent (the gate's order rate)
        self._funds: tuple | None = None               # (read at, available to bet) for the gate
        self._stop_cancelled: set = set()              # markets whose resting lays the STOP switch has cancelled
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
        if stopped:
            self.cancel_open("stopped (S3 STOP)", once=True)
        to_off = (self.preds.off - t).dt.total_seconds()
        horizon = self.cfg["green_seconds"] + 2 * self.cfg["near_poll_seconds"] + 5 if near_only \
            else self.cfg["watch_minutes"] * 60
        # a race laid and not yet closed stays watched past its scheduled off (a race off late is still open)
        late = self.preds.market_id.isin(self.priced) & (to_off > -180)
        watch = self.preds[((to_off > 0) | late) & (to_off <= horizon) & ~self.preds.market_id.isin(self.greened)]
        mids = sorted(watch.market_id.dropna().astype(str).unique())
        if not mids:
            return
        # the races nearest their off first: a slow pass over the races still forming must never hold up a green-up
        # (3 Nov, 14:49 Monmore: the 5-second window passed inside one poll and the lays went to the SP unattempted)
        offs = watch.groupby(watch.market_id.astype(str)).off.min()
        mids = sorted(mids, key=lambda m: offs.get(m, pd.Timestamp.max.tz_localize("UTC")))
        books = self.x.books(mids)
        for mid in mids:
            b = books.get(mid)
            if b is None:
                continue
            g = watch[watch.market_id.astype(str) == mid]
            secs = float((pd.Timestamp(g.off.iloc[0]) - pd.Timestamp(self.now())).total_seconds())
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
        if mid not in self.priced and self.until is not None and pd.Timestamp(g.off.iloc[0]) > \
                pd.Timestamp(self.until) - pd.Timedelta(seconds=self.cfg.get("close_margin_seconds", 120.0)):
            return                                       # never lay a race the trader would not be here to close
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
                gated = r.drift if self.cfg.get("gate", "drift") == "drift" else r.short
                why = (("forecast to shorten" if self.cfg.get("gate", "drift") == "drift" else
                        f"not short-priced (mid {SHORT_PRICE:g} or more)") if not gated else
                       ("at or above the mid" if not r.below_mid else "price"))
                rows.append({**row, "action": "no_lay", "price": float(r.lay_price) if np.isfinite(r.lay_price) else None,
                             "error": why})
                continue
            price = float(r.lay_price)
            q = b.runners.get(sid)                           # the book now, as the order goes in
            if self.cfg.get("rest_only", True) and q is not None and q.best_back:
                from model.race_book import market_now
                cur = float(market_now([q.best_back[0]], [q.best_lay[0] if q.best_lay else np.nan])[0])
                if price >= cur - 1e-9:
                    rows.append({**row, "action": "no_lay", "price": price,
                                 "error": f"at or above the mid now ({cur:.2f})"})
                    continue
            size = stake_for(price, self.cfg)
            why = self.gate(mid, price, size, stopped)
            if why:
                rows.append({**row, "action": "no_lay", "price": price, "error": f"risk gate: {why}"})
                continue
            ref = f"{REF}{self.day:%m%d}-{mid[-8:]}-{sid}-l"
            self._sent.append(pd.Timestamp(self.now()))
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
        pos = positions(self._with_local_backs(mid, self.orders.mine(mid)))
        if b.status != "OPEN" or b.inplay:
            if any(p["lay_matched"] > 0 and not p["closed"] for p in pos.values()):
                self._held(mid, pos, secs, "the market closed before the green-up")
            self.greened.add(mid)
            return
        rows, left = [], False
        base = {"time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
                "market_id": mid, "seconds_to_off": round(secs)}
        mn = self.cfg["min_order"]
        for sid, p in pos.items():
            if p["lay_matched"] <= 0 or p["closed"]:
                continue
            m = p["lay_price"]
            q = b.runners.get(sid)
            levels = list(q.back) if q else []
            if p["to_cover"] > 0 and levels and secs > self.cfg["give_up_seconds"]:
                # down the best three back prices; a part matched is kept and the rest tried next second
                limit, stake = ladder_stake(levels, p["to_cover"])
                stake = _floor2(stake)
                if stake >= mn:
                    ref = f"{REF}{self.day:%m%d}-{mid[-8:]}-{sid}-g"
                    fill = self.orders.back(mid, sid, limit, stake, ref, min_fill=mn)
                    if fill is not None and fill.status == "SUCCESS" and fill.matched > 0:
                        self._backs.setdefault(mid, {})[str(fill.bet_id or f"local-{len(self._backs.get(mid, {}))}")] = (
                            sid, ref, float(fill.matched), float(fill.avg_price or limit))
                        p["back_matched"] += fill.matched
                        p["back_value"] += fill.matched * float(fill.avg_price or limit)
                        p["to_cover"] = p["lay_value"] - p["back_value"]
                        done = p["to_cover"] / limit < mn
                        win, lose = outcomes(p)
                        row = {**base, "selection_id": sid, "action": "green" if done else "green_part",
                               "price": limit, "size": stake, "matched": fill.matched, "avg_price": fill.avg_price,
                               "bet_id": fill.bet_id, "status": fill.status, "lay_matched": round(p["lay_matched"], 2),
                               "lay_price": round(m, 2), "liability": round(p["lay_value"] - p["lay_matched"], 2),
                               "error": f"if it wins {win:+.2f}, if it loses {lose:+.2f}"}
                        if done:
                            row["locked_pnl"] = round(min(win, lose), 2)
                            self.locked += row["locked_pnl"]
                            p["closed"] = True
                        else:
                            left = True
                        rows.append(row)
                        continue
                    left = True                               # nothing matched: tried again next second
                    rows.append({**base, "selection_id": sid, "action": "green_retry", "price": limit, "size": stake,
                                 "status": getattr(fill, "status", ""), "error": getattr(fill, "error", "")})
                    continue
            if p["to_cover"] / (levels[0][0] if levels else m) < mn and p["back_matched"] > 0:
                win, lose = outcomes(p)                       # what is left is under the smallest order: done
                locked = round(min(win, lose), 2)
                self.locked += locked
                p["closed"] = True
                rows.append({**base, "selection_id": sid, "action": "green", "lay_matched": round(p["lay_matched"], 2),
                             "lay_price": round(m, 2), "locked_pnl": locked,
                             "error": f"if it wins {win:+.2f}, if it loses {lose:+.2f}"})
                continue
            if secs <= self.cfg["give_up_seconds"] or not levels:
                rows += self._close_at_sp(mid, sid, _floor2(p["to_cover"] / m), m, base)
                p["closed"] = True
            else:
                left = True
        if rows:
            self._write(rows)
        if not left:
            self.greened.add(mid)
            self.laid.pop(mid, None)

    def _with_local_backs(self, mid: str, orders: list[dict]) -> list[dict]:
        """Betfair's orders on the market with every green-up back this session saw matched and Betfair does not yet
        list (by bet id): a fill missing from listCurrentOrders a second later would otherwise be backed again."""
        seen = {str(o.get("betId")) for o in orders if o.get("betId")}
        extra = [{"selectionId": sid, "side": "BACK", "customerOrderRef": ref, "sizeMatched": m,
                  "averagePriceMatched": a, "betId": bid}
                 for bid, (sid, ref, m, a) in self._backs.get(mid, {}).items() if bid not in seen]
        return list(orders) + extra

    def _close_at_sp(self, mid: str, sid: int, S: float, m: float, base: dict) -> list[dict]:
        """The lay stake still open, ``S`` at ``m``, closed by a back at the SP of the same stake."""
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

    def gate(self, mid: str, price: float, size: float, stopped: bool = False) -> str:
        """The pre-trade risk gate (the audit A5): every lay passes it or is not sent. The reason it fails, else ''."""
        c = self.cfg
        if stopped or self.kill():
            return "stopped (S3 STOP)"
        if not (np.isfinite(price) and 1.01 <= price <= c["max_price"]):
            return f"price {price} outside 1.01-{c['max_price']}"
        if not (np.isfinite(size) and c["min_order"] <= size <= c["max_stake"] + 1e-9):
            return f"stake {size} outside {c['min_order']}-{c['max_stake']}"
        liab = size * (price - 1)
        if self.laid.get(mid, 0.0) + liab > c.get("max_market_liability", float("inf")) + 1e-9:
            return f"the race's liability would pass GBP{c['max_market_liability']:.0f}"
        if sum(self.laid.values()) + liab > c["max_open_liability"] + 1e-9:
            return f"open liability would pass GBP{c['max_open_liability']:.0f}"
        limit = c.get("daily_loss_limit")
        if limit is not None and self.day_loss() - liab <= -limit:
            return f"the day's loss limit (GBP{limit:.0f}) with this lay's liability"
        t = pd.Timestamp(self.now())
        self._sent = [s for s in self._sent if (t - s).total_seconds() < 60]
        if len(self._sent) >= c.get("max_orders_per_minute", 10 ** 9):
            return f"order rate ({c['max_orders_per_minute']} a minute)"
        if self.live:
            if self._funds is None or (t - self._funds[0]).total_seconds() > 60:
                try:
                    f = self.x.funds()
                    self._funds = (t, float(f["available"]) if f and f.get("available") is not None else None)
                except Exception:
                    self._funds = (t, None)
            avail = self._funds[1]
            if avail is None:
                return "the account's funds could not be read"
            if avail < liab:
                return f"funds GBP{avail:.2f} under the liability GBP{liab:.2f}"
        return ""

    def recover(self) -> dict:
        """Rebuild the day's state from Betfair's own orders (the audit A4): every market with an ``el-`` order of
        today's is laid (``priced``); its liability counted; closed (``greened``) where every lay is covered or
        closed at the SP. Run on start, live only, beside the local ledger's ``_resume``."""
        if not self.live or self.preds is None:
            return {}
        mids = sorted(self.preds.market_id.dropna().astype(str).unique())
        tag = f"{REF}{self.day:%m%d}-"
        got = []
        for i in range(0, len(mids), 50):
            got += [o for o in self.x.orders(mids[i:i + 50]) if str(o.get("customerOrderRef") or "").startswith(tag)]
        by = {}
        for o in got:
            by.setdefault(str(o.get("marketId")), []).append(o)
        rows = []
        for mid, orders in by.items():
            self.priced.add(mid)
            liab = sum(float(o.get("sizeMatched") or 0) * (float(o.get("averagePriceMatched") or 0) - 1)
                       + float(o.get("sizeRemaining") or 0) * (float(o.get("priceSize", {}).get("price") or 1) - 1)
                       for o in orders if o.get("side") == "LAY")
            pos = positions(orders)
            closed = all(p["closed"] or p["lay_matched"] <= 0 or p["to_cover"] <= 1e-6 for p in pos.values())
            unmatched = any(float(o.get("sizeRemaining") or 0) > 0 for o in orders)
            if closed and not unmatched:
                self.greened.add(mid)
            else:
                self.laid[mid] = max(self.laid.get(mid, 0.0), liab)
            rows.append({"time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
                         "market_id": mid, "action": "recovered", "liability": round(liab, 2),
                         "status": "closed" if mid in self.greened else "open"})
        if rows:
            self._write(rows)
        return {"markets": len(by), "open": len([m for m in by if m not in self.greened])}

    def open_markets(self) -> list[str]:
        """The markets laid and not yet closed."""
        return sorted(self.priced - self.greened)

    def cancel_open(self, why: str, once: bool = False) -> None:
        """Cancel the resting lays of every market laid and not yet closed (the matched part is closed by green(), or
        held to the result if the trader is gone). ``once``: each market once (the STOP switch's repeated calls)."""
        for mid in self.open_markets():
            if once and mid in self._stop_cancelled:
                continue
            try:
                status = self.orders.cancel(mid)
            except Exception as exc:                            # one market's failure never stops the others
                status = f"FAILURE: {exc}"[:200]
            self._stop_cancelled.add(mid)
            self._write([{"time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode,
                          "market_id": mid, "action": "cancel", "status": status, "error": why}])

    def run(self, until: datetime, sleep=time.sleep) -> dict:
        """Lay until ``until`` (no race off later than ``close_margin_seconds`` before it), then stay up to
        ``overrun_minutes`` past it while a laid market is not closed; on the way out, by the clock, an error or a
        signal, cancel every resting lay (the owner, 5 Nov 2026: the audit's end-of-run fix)."""
        self.until = until
        if self.live:
            try:
                self.load_predictions()
                log.info("recovered from Betfair's orders: %s", self.recover())
            except Exception as exc:                            # the local ledger's resume stands without it
                log.warning("recovery from Betfair's orders failed (%s)", exc)
                self.last_error = f"recover: {exc}"
        last_full = None
        hard_stop = until + timedelta(minutes=self.cfg.get("overrun_minutes", 15.0))
        try:
            while self.now() < until or (self.open_markets() and self.now() < hard_stop):
                try:
                    if last_full is not None and (self.now() - last_full).total_seconds() < self.cfg["poll_seconds"]:
                        self.step(near_only=True)
                    else:
                        last_full = self.now()
                        self.step()
                except Exception as exc:                        # one failed poll: the next tries again
                    log.warning("poll failed (%s)", exc)
                    self.last_error = f"poll: {exc}"
                self.heartbeat()
                sleep(self.cfg["near_poll_seconds"])
        finally:
            self.cancel_open("trader stopped")
        self.heartbeat(force=True, finished=True, still_open=self.open_markets())
        return {"day": f"{self.day:%Y-%m-%d}", "mode": self.mode, "markets_priced": len(self.priced),
                "markets_closed": len(self.greened), "locked_pnl": round(self.locked, 2),
                "open_liability": round(sum(self.open_liability.values()), 2)}


class SharedBooks:
    """One poll's books shared by the paper books (``run_many``): a market read within ``ttl`` seconds is not read
    again, so the paper variants add no requests beyond the first's. Everything else is the exchange's own."""

    def __init__(self, x, ttl: float = 0.8, clock=time.monotonic):
        self.x, self.ttl, self.clock, self._c = x, ttl, clock, {}

    def books(self, mids, **kw):
        t = self.clock()
        need = [m for m in mids if m not in self._c or t - self._c[m][0] > self.ttl]
        if need:
            got = self.x.books(need, **kw)
            for m in need:
                if m in got:
                    self._c[m] = (t, got[m])
                else:
                    self._c.pop(m, None)
        return {m: self._c[m][1] for m in mids if m in self._c}

    def __getattr__(self, k):
        return getattr(self.x, k)


def run_many(traders: list, until: datetime, sleep=time.sleep) -> list[dict]:
    """The paper books together, poll by poll, as ``EarlyLay.run`` runs one (the live trader runs alone)."""
    assert traders and not any(t.live for t in traders), "run_many is for the paper books only"
    now, cfg, last_full = traders[0].now, traders[0].cfg, None
    for t in traders:
        t.until = until
    while now() < until:
        full = last_full is None or (now() - last_full).total_seconds() >= cfg["poll_seconds"]
        if full:
            last_full = now()
        for t in traders:
            try:
                t.step(near_only=not full)
            except Exception as exc:                            # one failed poll: the next tries again
                log.warning("%s: poll failed (%s)", t.mode, exc)
                t.last_error = f"poll: {exc}"
            t.heartbeat()
        sleep(cfg["near_poll_seconds"])
    out = []
    for t in traders:
        t.heartbeat(force=True, finished=True)
        out.append({"day": f"{t.day:%Y-%m-%d}", "mode": t.mode, "markets_priced": len(t.priced),
                    "markets_closed": len(t.greened)})
    return out


def replay(rows: pd.DataFrame, books: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """The lays the rest-only guard blocked (ledger rows ``no_lay`` 'at or above the mid'), replayed on the books after
    them (the owner, 3 Nov: "how would the races post 14:31 have looked without the guard"): matched at once at the
    best lay price where it was at or under our price, else resting at our price and matched once a later trade
    reached it; greened up at the best back price of the last book at least ``green_seconds`` before the off, the stake
    as the trader's (``stake_for``). ``books``: rows of market_id, selection_id, polled (UTC), back1, lay1,
    last_traded, the pre-off books."""
    cfg = cfg or CONFIG
    out = []
    b = books.assign(polled=pd.to_datetime(books.polled, utc=True), market_id=books.market_id.astype(str),
                     selection_id=pd.to_numeric(books.selection_id, errors="coerce"))
    for r in rows.itertuples(index=False):
        t = pd.Timestamp(r.time_utc)
        off = t + pd.Timedelta(seconds=float(r.seconds_to_off))
        price = float(r.price)
        stake = stake_for(price, cfg)
        g = b[(b.market_id == str(r.market_id)) & (b.selection_id == float(r.selection_id))].sort_values("polled")
        row = {"race_time": r.race_time, "track": r.track, "trap": r.trap, "lay_price": price,
               "f_bsp": r.f_bsp, "mid": r.mid, "stake": stake, "matched": None, "matched_at": None, "greened_at": None,
               "pnl": None, "note": ""}
        before = g[g.polled <= t]
        after = g[(g.polled > t) & (g.polled <= off)]
        if not len(before) and not len(after):
            out.append({**row, "note": "no books for this dog"})
            continue
        now = before.iloc[-1] if len(before) else after.iloc[0]
        lay1 = float(now.lay1) if pd.notna(now.lay1) and now.lay1 > 1 else None
        if lay1 is not None and lay1 <= price + 1e-9:
            m, how = lay1, "taken at once"
        else:
            lt = pd.to_numeric(after.last_traded, errors="coerce")
            hit = after[lt <= price + 1e-9]
            if not len(hit):
                out.append({**row, "matched": False, "note": "resting: never reached"})
                continue
            m, how = price, "rested, then traded at our price"
        last = g[(g.polled <= off - pd.Timedelta(seconds=cfg["green_seconds"])) & (g.polled > t)]
        last = last[pd.to_numeric(last.back1, errors="coerce") > 1]
        if not len(last):
            out.append({**row, "matched": True, "matched_at": m, "note": f"{how}; no book to green up in"})
            continue
        B = float(last.back1.iloc[-1])
        out.append({**row, "matched": True, "matched_at": m, "greened_at": B, "pnl": round(stake * (1 - m / B), 2),
                    "note": how})
    return pd.DataFrame(out)


def replay_books(day: date) -> list[tuple[str, pd.DataFrame]]:
    """The day's pre-off books for ``replay``: the live-key recorder's (greyhound/book_record.py, every 5 s near the
    off), then the market recorder's (about 90 seconds stale, re-timed)."""
    from betfair_recorder import _read_day_csv
    from greyhound import tradeout as to
    parts = []
    for name, retimed in (("books_greyhound_live.csv", False), ("books_greyhound.csv", True)):
        bk = _read_day_csv(day, name)
        mk = _read_day_csv(day, name.replace("books", "markets"))
        if bk is None or mk is None or not len(bk):
            continue
        pre = to.pre_off_books(bk, mk, retimed=retimed)
        if len(pre):
            parts.append((name, pre[["market_id", "selection_id", "polled", "back1", "lay1", "last_traded"]]))
    return parts


def replay_on_books(rows: pd.DataFrame, parts: list, cfg: dict | None = None) -> pd.DataFrame:
    """``replay`` row by row on the first books that cover the dog (the live key's, then the recorder's)."""
    res = []
    for _, r in rows.iterrows():
        got = None
        for name, pre in parts:                                  # the live-key books first, where they cover it
            one = replay(pd.DataFrame([r]), pre, cfg)
            if len(one) and one.note.iloc[0] != "no books for this dog":
                got = one.assign(books="live key" if "live" in name else "recorder, ~90 s stale")
                break
        res.append(got if got is not None else replay(pd.DataFrame([r]), pd.DataFrame(
            columns=["market_id", "selection_id", "polled", "back1", "lay1", "last_traded"]), cfg).assign(books="none"))
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame()


def blocked_section(store, day: date, mode: str = "live", since: str | None = None) -> list[str]:
    """The guard's blocked lays, replayed (``replay``), as a section of the table: the live-key books where they cover
    the race, else the recorder's (about 90 seconds stale)."""
    raw = store.get(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/ledger.csv")
    if not raw:
        return []
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    blk = led[(led.action == "no_lay") & led.error.astype(str).str.contains("at or above the mid", na=False)
              & pd.to_numeric(led.price, errors="coerce").gt(1)]
    if since:
        blk = blk[pd.to_datetime(blk.time_utc, utc=True) >= pd.Timestamp(since)]
    if not len(blk):
        return ["", "#### Lays the rest-only guard blocked, replayed", "", "none yet"]
    d = replay_on_books(blk, replay_books(day))
    m = d[d.matched == True]                                              # noqa: E712
    tot = float(pd.to_numeric(m.pnl, errors="coerce").fillna(0).sum())
    L = ["", "#### Lays the rest-only guard blocked, replayed (what would have happened without it)", "",
         f"{len(d)} lays blocked; {len(m)} would have matched; greened up {int(m.pnl.notna().sum())}, "
         f"**GBP{tot:+.2f}** locked at GBP{CONFIG['liability']:.0f} a lay. Matched at once where the best lay price was at "
         "or under our price, else when a later trade reached it; greened up at the best back price of the last book "
         f"{CONFIG['green_seconds']:.0f} s or more before the off.", "",
         "| Race (UK) | Track | Trap | Forecast SP | Mid | Lay price | Stake | Matched at | Greened at | P&L | How | Books |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.sort_values("race_time").itertuples(index=False):
        L.append(f"| {r.race_time} | {r.track} | {r.trap} | {r.f_bsp} | {r.mid} | {r.lay_price} | {r.stake} | "
                 f"{r.matched_at if r.matched_at is not None else ''} | {r.greened_at if r.greened_at is not None else ''} | "
                 f"{r.pnl if r.pnl is not None else ''} | {r.note} | {r.books} |")
    return L


def modes_to_show(store, day: date) -> list[str]:
    """The early lay's tables for the running page: each mode with a ledger today (live, paper), else the live one
    (its status says whether it started) (the owner, 5 Nov 2026: the rule on paper was missing from the page)."""
    have = [m for m in ("live", "paper") if store.get(f"{PREFIX}/{m}/{day:%Y-%m-%d}/ledger.csv")]
    return have or ["live"]


def table(store, day: date, mode: str = "live") -> str:
    """The day's live early lay so far, a row per race, from the trader's ledger in S3: the lays placed and their
    stake, what matched by the green-up, how each was closed, the P&L locked by the green-ups (the SP closes and the
    held lays settle with the result, shown as open liability)."""
    raw = store.get(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/ledger.csv")
    head = (f"### Early lay, {mode} orders on the owner's account ({day:%Y-%m-%d})" if mode == "live" else
            f"### Early lay, the live rule on paper: nothing sent ({day:%Y-%m-%d})")
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
                     "closed": "; ".join(f"{a}" + (f" ({e})" if isinstance(e, str) and e else "") + (f" {st}" if
                                         isinstance(st, str) and st else "")
                                         for a, e, st in zip(opened.action, opened.error, opened.status))[:120],
                     "retries": int((led[(led.market_id == mid)].action == "green_retry").sum()),
                     "errors": "; ".join(sorted(set(lg.error.dropna().astype(str)) - {""}))[:80]})
    comm = commission()
    net = sum(r["locked"] - comm * max(r["locked"], 0.0) for r in rows)
    rec = store.get(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/reconcile.json")
    if rec:
        r = json.loads(rec)
        head += (f"\n\nBetfair's settled bets: {r.get('settled')} of {r.get('markets_laid')} markets, "
                 f"GBP{r.get('betfair_profit_gross', 0):+.2f} before commission, GBP{r.get('betfair_profit_net', 0):+.2f} "
                 f"after; against the ledger GBP{r.get('difference', 0):+.2f}.")
    L = [head, "", f"Races laid {len(rows)}; lays {len(lays)}, GBP{lays['size'].sum():.2f} asked; matched "
         f"GBP{sum(r['matched'] for r in rows):.2f}; locked by green-ups **GBP{run:+.2f}** (GBP{net:+.2f} after "
         f"{comm:.0%} commission); still settling with the "
         f"result (closed at the SP or held) GBP{sum(r['open_liability'] for r in rows):.2f} liability."
         + (f" Skipped races: {len(skips)} ({'; '.join(sorted(set(skips.error.astype(str))))})." if len(skips) else ""),
         "", "| Race (UK) | Track | Lays | Stake asked | Matched | Greened | SP/held | Open liability | Locked | Running "
         "| Green-up retries | Not greened: how and why | Lay errors |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["race"]):
        L.append(f"| {r['race']} | {r['track']} | {r['lays']} | {r['staked']} | {r['matched']} | {r['greened']} | "
                 f"{r['sp_or_held']} | {r['open_liability']} | {r['locked']} | {r['running']} | {r['retries']} | "
                 f"{r['closed']} | {r['errors']} |")
    # dog by dog: every lay that matched, against its forecast and the price when laid
    num = lambda v: pd.to_numeric(v, errors="coerce")
    fin = led[led.action.isin(["green", "green_part", "sp_close", "hold"])].copy()
    if len(fin):
        fin["selection_id"] = num(fin.selection_id)
        lg = lays.assign(selection_id=num(lays.selection_id))
        L += ["", "#### Live lays that matched, dog by dog", "",
              "| Race (UK) | Track | Trap | Forecast SP | Mid when laid | Lay price | Lay stake | Matched | Greened at "
              "| How closed | P&L |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (mid, sid), g in fin.groupby(["market_id", "selection_id"]):
            lr = lg[(lg.market_id == mid) & (lg.selection_id == sid)]
            if not len(lr):
                continue
            lr = lr.iloc[-1]
            gr = g[g.action.isin(["green", "green_part"])]
            bk = num(gr.matched).fillna(0)
            green_at = round(float((num(gr.avg_price).fillna(0) * bk).sum() / bk.sum()), 2) if bk.sum() > 0 else ""
            pnl = num(g.locked_pnl).dropna()
            L.append(f"| {lr.race_time} | {lr.track} | {lr.trap} | {lr.f_bsp} | {lr.mid} | {lr.price} | {lr['size']} | "
                     f"{num(g.lay_matched).max()} | {green_at} | {g.action.iloc[-1]} | "
                     f"{round(float(pnl.sum()), 2) if len(pnl) else 'settles with the result'} |")
    try:
        L += blocked_section(store, day, mode)
    except Exception as exc:                                     # the table stands without the replay
        L += ["", f"(the blocked lays' replay failed: {str(exc)[:160]})"]
    return "\n".join(L) + "\n"


def commission(env=None) -> float:
    """The commission rate on a market's net winnings: the account's rate on greyhounds (greyhound/costs.py),
    GREYHOUND_COMMISSION overriding it for a run."""
    return costs.commission(env)


def reconcile(store, x, day: date, comm: float | None = None, prefix: str = PREFIX, ref: str = REF,
              laid_action: str = "lay") -> dict:
    """The live ledger against Betfair's settled bets (the audit A6): per market, what Betfair paid on our ``el-``
    bets (listClearedOrders' profit, before commission) beside what the ledger locked by green-ups; the markets not
    yet settled; the day's net after commission on each market's winnings. Written to S3 beside the ledger."""
    comm = commission() if comm is None else comm
    raw = store.get(f"{prefix}/live/{day:%Y-%m-%d}/ledger.csv")
    if not raw:
        return {"day": f"{day}", "note": "no data found: no live ledger"}
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    mids = sorted(set(led[led.action == laid_action].market_id.dropna().astype(str)))
    if not mids:
        return {"day": f"{day}", "note": "no lays"}
    tag = f"{ref}{day:%m%d}-"
    ours = [c for c in x.cleared(mids) if str(c.get("customerOrderRef") or "").startswith(tag)]
    paid: dict = {}
    for c in ours:
        paid[str(c.get("marketId"))] = paid.get(str(c.get("marketId")), 0.0) + float(c.get("profit") or 0)
    locked = pd.to_numeric(led[led.action == "green"].set_index("market_id").locked_pnl, errors="coerce")
    locked = locked.groupby(level=0).sum()
    rows = [{"market_id": m, "betfair_profit": round(paid[m], 2) if m in paid else None,
             "ledger_locked": round(float(locked.get(m, 0.0)), 2),
             "difference": round(paid[m] - float(locked.get(m, 0.0)), 2) if m in paid else None} for m in mids]
    gross = sum(paid.values())
    net = sum(v - comm * max(v, 0.0) for v in paid.values())
    out = {"day": f"{day}", "markets_laid": len(mids), "settled": len(paid),
           "unsettled": [m for m in mids if m not in paid], "betfair_profit_gross": round(gross, 2),
           "betfair_profit_net": round(net, 2), "commission": comm,
           "ledger_locked": round(float(locked.sum()), 2), "difference": round(gross - float(locked.reindex(
               list(paid)).fillna(0).sum()), 2), "by_market": rows}
    try:
        store.put(f"{prefix}/live/{day:%Y-%m-%d}/reconcile.json", json.dumps(out, indent=1).encode())
    except Exception as exc:
        log.warning("reconciliation not written (%s)", exc)
    return out


def paper_table(store, day: date, name: str, parts: list | None = None) -> str:
    """A paper book's day (``PAPER_VARIANTS``): the lays it would have placed, replayed on the books (``replay``):
    matched at once where the best lay price was at or under ours, else when a later trade reached it (no queue: a
    paper fill is the best case), greened up at the best back price 5 s or more before the off."""
    cfg = {**CONFIG, **PAPER_VARIANTS[name]}
    book = f"paper-{name}"
    desc = (f"{'short-priced (mid under ' + format(SHORT_PRICE, 'g') + ')' if cfg['gate'] == 'short' else 'drifters'}"
            f", laid at {1 - cfg['margin']:.0%} of the forecast SP")
    head = f"### Early lay on paper, {name}: {desc} ({day:%Y-%m-%d})"
    st = store.get(f"{PREFIX}/{book}/{day:%Y-%m-%d}/status.json")
    if st:
        s = json.loads(st)
        head += (f"\n\nPaper trader: last seen {s.get('time_utc')} UTC; priced {s.get('markets_priced')} races"
                 + (f"; last error: {s['last_error']}" if s.get("last_error") else "") + ".")
    raw = store.get(f"{PREFIX}/{book}/{day:%Y-%m-%d}/ledger.csv")
    if not raw:
        return f"{head}\n\nno ledger yet\n"
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    lays = led[(led.action == "lay") & pd.to_numeric(led.price, errors="coerce").gt(1)]
    if not len(lays):
        return f"{head}\n\nno lays yet ({int((led.action == 'no_lay').sum())} dogs looked at)\n"
    d = replay_on_books(lays, replay_books(day) if parts is None else parts, cfg)
    m = d[d.matched == True]                                              # noqa: E712
    pnl = pd.to_numeric(m.pnl, errors="coerce")
    tot = float(pnl.fillna(0).sum())
    pending = int((d.note == "no books for this dog").sum())
    L = [head, "", f"{len(d)} lays in {d.race_time.nunique()} races; {len(m)} would have matched; greened up "
         f"{int(pnl.notna().sum())}, **GBP{tot:+.2f}** at GBP{cfg['liability']:.0f} a lay"
         + (f" ({tot / max(float(m.stake.sum()), 1e-9):+.1%} of the matched stake)" if len(m) else "")
         + (f"; {pending} not yet in the books" if pending else "") + ". A paper fill takes no place in the queue: "
         "the best case.", "",
         "| Race (UK) | Track | Trap | Forecast SP | Mid | Lay price | Stake | Matched at | Greened at | P&L | How | Books |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d.sort_values("race_time").itertuples(index=False):
        L.append(f"| {r.race_time} | {r.track} | {r.trap} | {r.f_bsp} | {r.mid} | {r.lay_price} | {r.stake} | "
                 f"{r.matched_at if r.matched_at is not None else ''} | {r.greened_at if r.greened_at is not None else ''} | "
                 f"{r.pnl if r.pnl is not None else ''} | {r.note} | {r.books} |")
    return "\n".join(L) + "\n"


def paper_tables(store, day: date) -> str:
    """Every paper book's table, the books read once."""
    parts = replay_books(day)
    out = []
    for name in PAPER_VARIANTS:
        try:
            out.append(paper_table(store, day, name, parts))
        except Exception as exc:                                  # one book's failure never hides the others
            out.append(f"### Early lay on paper, {name}\n\nnot scored ({str(exc)[:160]})\n")
    return "\n".join(out)


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
    ap.add_argument("--reconcile", action="store_true",
                    help="the live ledger against Betfair's settled bets for --date (default yesterday, UK)")
    ap.add_argument("--paper-variants", default=None,
                    help=f"comma-separated paper books ({', '.join(PAPER_VARIANTS)}), traded together on paper; never "
                         "with --live")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    out = {}
    if a.reconcile:
        from trading.exchange import BetfairData, LiveExchange
        rd = date.fromisoformat(a.date) if a.date else uk_today() - timedelta(days=1)
        x = LiveExchange(BetfairData(source="greyhound"))
        x.login()
        out["reconcile"] = reconcile(store, x, rd)
    if a.trade and a.paper_variants:
        from trading.exchange import BetfairData, LiveExchange
        if a.live:
            raise SystemExit("--paper-variants is paper only: no --live")
        names = [n.strip() for n in a.paper_variants.split(",") if n.strip()]
        bad = [n for n in names if n not in PAPER_VARIANTS]
        if bad or not names:
            raise SystemExit(f"unknown paper variants {bad}; known: {', '.join(PAPER_VARIANTS)}")
        fit = laytest.live_fit(store, day)
        if fit is None:
            for n in names:
                store.put(f"{PREFIX}/paper-{n}/{day:%Y-%m-%d}/status.json", json.dumps(
                    {"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": f"paper-{n}",
                     "state": "stopped: no forecast fit for today"}).encode())
            raise SystemExit("no forecast fit for today: nothing traded")
        x = LiveExchange(BetfairData(source="greyhound"))
        x.login()
        shared = SharedBooks(x)
        base = Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound"))
        traders = []
        for n in names:
            cfg = {**CONFIG, **PAPER_VARIANTS[n]}
            log.info("early lay paper-%s: gate %s, margin %.2f, GBP%.2f liability a lay", n, cfg["gate"],
                     cfg["margin"], cfg["liability"])
            traders.append(EarlyLay(store, shared, cfg, day, False, base / f"early_lay_paper-{n}" /
                                    f"{day:%Y-%m-%d}" / "ledger.csv", fit, kill=kill_switch(store), book=f"paper-{n}"))
        out["paper"] = run_many(traders, uk_time_on(day, a.until))
    elif a.trade:
        from trading.exchange import BetfairData, LiveExchange
        live = a.live and switch_on()
        if a.live and not live:
            log.warning("GREYHOUND_EARLY_LAY is not 'live': the decisions are logged, nothing is sent")
        limit = daily_loss_limit()
        if live and limit is None:
            live = False
            log.warning("no daily loss limit (GREYHOUND_EARLY_LAY_DAILY_LOSS): no live orders, the decisions are logged")
        mode = "live" if live else "paper"
        status_key = f"{PREFIX}/{mode}/{day:%Y-%m-%d}/status.json"
        store.put(status_key, json.dumps({"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                          "mode": mode, "state": "starting: fitting the forecast"}).encode())
        fit = laytest.live_fit(store, day)                    # fitted on the prices of the model that is live
        if fit is None:
            store.put(status_key, json.dumps({"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                              "mode": mode, "state": "stopped: no forecast fit for today"}).encode())
            raise SystemExit("no forecast fit for today: nothing traded")
        x = LiveExchange(BetfairData(source="greyhound"))
        x.login()
        if live:                                         # yesterday's live day against Betfair's settled bets
            try:
                out["reconcile_yesterday"] = reconcile(store, x, day - timedelta(days=1))
            except Exception as exc:
                log.warning("yesterday not reconciled (%s)", exc)
        ledger = Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound")) / \
            f"early_lay_{mode}" / f"{day:%Y-%m-%d}" / "ledger.csv"
        cfg = dict(CONFIG, daily_loss_limit=limit, commission=commission())
        import signal
        signal.signal(signal.SIGTERM, _terminate)            # a cancelled job: run()'s finally cancels the resting lays
        log.info("early lay %s: daily loss limit %s", mode, f"GBP{limit:.2f}" if limit is not None else "none")
        log.info("early lay %s: margin %.2f, gate %s, GBP%.2f liability a lay (stake GBP%.2f-%.2f), fit to %s", mode,
                 cfg["margin"], cfg["gate"], cfg["liability"], cfg["min_stake"], cfg["max_stake"], fit.get("fit_to"))
        out["trade"] = EarlyLay(store, x, cfg, day, live, ledger, fit, kill=kill_switch(store)).run(
            uk_time_on(day, a.until))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
