"""Take and close at the SP, live (the owner, 18 Nov 2026: "Run the take-and-close LIVE"; "GBP100 a bet"; its own
switch, greyhound/STOP left on; live without waiting for the backtest).

The trade-out study's one first-traded back lead (ledger greyhound-strategy-search-stack-m-1114, batch 35 arm
stackm_out: holdout +10.96%, 90% 4.41 to 18.66, 138 trades; one of 238 rules, about 2.6 pass by chance: a lead, not a
proven edge), traded on the owner's account:

- each runner's **first traded book**, kept as the market is polled (greyhound/early_lay.py's rule); once every runner
  has one and the day's prices cover the field, the race is decided once: the SP forecast is the early lay's
  (``greyhound.laytest.live_fit``: the conditional logit on the book and the live model's price);
- every runner whose **best back price is at least ``bar`` (8%) longer than its forecast SP** (the study's score for a
  back, ``price x E[1/BSP] - 1 > bar``, greyhound/tradeout.py ``positions``), at a price up to ``max_price``, is
  **taken**: a FILL_OR_KILL back down the best ``depth`` (3) back prices while each still clears the bar, for up to
  ``stake`` (GBP100); whatever is not matched at once is cancelled by Betfair, so nothing ever rests in the queue
  (the passive fills lost: ledger greyhound-lay-backtest-mlsitd-1118);
- each matched back is **closed at once by laying its winnings at the Betfair SP** (MARKET_ON_CLOSE, liability =
  matched x (price - 1)): nothing if the dog wins, ``-stake + stake (price - 1) / (BSP - 1)`` if it loses. A back is
  only sent when its smallest fill can be closed (Betfair takes no lay at SP under GBP10 of liability).

Guards: real orders only with ``--live``, ``GREYHOUND_TAKE_CLOSE=live`` and a daily loss limit
(``GREYHOUND_TAKE_CLOSE_DAILY_LOSS``, GBP); without the limit nothing is sent. The day's loss counts every open
position at its worst (its whole stake) until its race settles. The S3 objects ``greyhound/take_close/STOP`` and
``trading/STOP`` stop new backs within a poll (an unreadable switch counts as STOP); closes already placed stay.
``greyhound/STOP`` does not stop this trader (the owner, 18 Nov 2026). Orders carry ``tc-`` references, and the
positions are read back from Betfair's own orders each poll, so a restarted session closes what an earlier one backed.

    GREYHOUND_TAKE_CLOSE=live GREYHOUND_TAKE_CLOSE_DAILY_LOSS=<GBP> python -m greyhound.take_close --trade --live --until 21:30
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

PREFIX = "greyhound/take_close"
REF = "tc-"                                   # every order: tc-<mmdd>-<market>-<selection>-<b|s>
STOP_KEYS = ("greyhound/take_close/STOP", "trading/STOP")
MIN_SP_LIABILITY = 10.0                       # Betfair: no lay at SP under GBP10 of liability
CONFIG = {
    "stake": 100.0,                           # the owner, 18 Nov 2026: "GBP100 a bet" (the most backed on one dog)
    "bar": 0.08,                              # the study's lead: the price at least 8% over the forecast SP
    "depth": 3,                               # down the best three backs, each clearing the bar
    "max_price": 20.0,                        # the study's prices (greyhound.tradeout.MAX_PRICE)
    "min_stake": 2.0,
    "max_open_stake": 1000.0,                 # a guard against a fault: no new back while this much is open
    "daily_loss_limit": None,                 # GBP; needed for any real order
    "watch_minutes": 60,
    "min_seconds": 30,                        # no back with less than this to the off (the SP lay must be in)
    "close_margin_seconds": 120,
    "poll_seconds": 5,
    "near_poll_seconds": 1,
    "overrun_minutes": 20,
}
LEDGER_FIELDS = ["time_utc", "mode", "market_id", "selection_id", "track", "race_time", "trap", "dog", "action",
                 "price", "limit", "stake", "matched", "f_bsp", "edge", "back1", "mid", "p_model", "liability",
                 "bet_id", "pnl", "seconds_to_off", "note"]


def _floor2(x: float) -> float:
    return math.floor(float(x) * 100 + 1e-9) / 100


def switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("GREYHOUND_TAKE_CLOSE", "")).strip().lower() == "live"


def daily_loss_limit(env=None) -> float | None:
    env = os.environ if env is None else env
    try:
        v = float(str(env.get("GREYHOUND_TAKE_CLOSE_DAILY_LOSS", "")).strip())
    except ValueError:
        return None
    return v if math.isfinite(v) and v > 0 else None


def kill_switch(store, keys=STOP_KEYS, every: float = 5.0):
    """True while S3 holds any of ``keys``; an unreadable switch counts as STOP (greyhound.live._exists)."""
    from greyhound.live import _exists
    state = {"at": None, "on": False}

    def check() -> bool:
        now = datetime.now(timezone.utc)
        if state["at"] and (now - state["at"]).total_seconds() < every:
            return state["on"]
        state["at"] = now
        state["on"] = any(_exists(store, k) for k in keys)
        return state["on"]
    return check


# --------------------------------------------------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------------------------------------------------

def forecast(d: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """The race at its first traded book (``early_lay.first_rows``) with each runner's forecast SP, as the early lay
    and the favourite back forecast it."""
    from greyhound import tradeout as to
    from greyhound.early_lay import vol_in_fit
    from research.greyhound import tradeout_study as ts
    if not vol_in_fit(fit):
        d = d.assign(runner_total_matched=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = to.add_features(d)
        q, book = ts.predict(x, fit)
    return x.assign(f_bsp=1.0 / (book * np.asarray(q, float)))


def plan(levels: list, f_bsp: float, cfg: dict) -> dict | None:
    """The take for one runner from its back ladder now (``levels``: [(price, size)], best first): down the best
    ``depth`` prices while each is at least ``bar`` over the forecast SP and at most ``max_price``, for up to
    ``stake``. Returns the lowest price to take (the FILL_OR_KILL limit), the stake, and the smallest fill that can
    still be closed at the SP; None when nothing qualifies or what is on offer cannot be closed."""
    if not (np.isfinite(f_bsp) and f_bsp > 1):
        return None
    cum, limit = 0.0, None
    for price, size in levels[:cfg["depth"]]:
        if not (price / f_bsp - 1.0 >= cfg["bar"] - 1e-12 and 1.0 < price <= cfg["max_price"]):
            break
        cum += float(size)
        limit = float(price)
    if limit is None:
        return None
    stake = _floor2(min(cfg["stake"], cum))
    min_fill = max(cfg["min_stake"], math.ceil(MIN_SP_LIABILITY / (limit - 1.0) * 100) / 100)
    if stake < min_fill:
        return None
    return {"limit": limit, "stake": stake, "min_fill": round(min_fill, 2), "edge": levels[0][0] / f_bsp - 1.0}


def close_pnl(stake: float, price: float, bsp: float | None, won: bool | None) -> float | None:
    """A back closed by laying its winnings at the SP, before commission; None until the race settles."""
    if won is None:
        return None
    if won:
        return 0.0
    if bsp is None or not bsp > 1:
        return None
    return -stake + stake * (price - 1.0) / (bsp - 1.0)


def positions(orders: list[dict]) -> dict:
    """Per selection from Betfair's orders (``tc-``) on one market: the backs matched (stake, value) and the SP lay
    liability placed."""
    out: dict = {}
    for o in orders:
        ref = str(o.get("customerOrderRef") or "")
        if not ref.startswith(REF):
            continue
        p = out.setdefault(int(o.get("selectionId", 0)), {"stake": 0.0, "value": 0.0, "sp_liability": 0.0})
        if o.get("side") == "BACK" and ref.endswith("-b"):
            m = float(o.get("sizeMatched") or 0)
            p["stake"] += m
            p["value"] += m * float(o.get("averagePriceMatched") or 0)
        elif o.get("side") == "LAY" and ref.endswith("-s"):
            p["sp_liability"] += float(o.get("bspLiability") or 0)
    for p in out.values():
        p["price"] = p["value"] / p["stake"] if p["stake"] > 0 else None
        p["to_close"] = _floor2(max(0.0, p["value"] - p["stake"] - p["sp_liability"]))   # winnings not yet laid
    return out


# --------------------------------------------------------------------------------------------------------------------
# The trader
# --------------------------------------------------------------------------------------------------------------------

class TakeClose:
    def __init__(self, store, x, cfg: dict, day: date, ledger_path: Path, fit: dict, live: bool,
                 kill=lambda: False, now=lambda: datetime.now(timezone.utc)):
        self.store, self.x, self.cfg, self.day, self.fit, self.live = store, x, cfg, day, fit, live
        self.mode = "live" if live else "paper"
        self.ledger_path, self.kill, self.now = ledger_path, kill, now
        self.first: dict = {}             # market -> selection -> its first traded book
        self.hist: dict = {}              # (market, selection) -> [(time, back1, lay1)]
        self.decided: set = set()
        self.held: dict = {}              # market -> selection -> {stake, price} (this session's or Betfair's)
        self.settled: dict = {}           # market -> net P&L after commission
        self.laid: dict = {}              # (market, selection) -> SP lay liability this session sent (or the ledger's)
        self.preds: pd.DataFrame | None = None
        self.until: datetime | None = None
        self._loaded_at = self._beat_at = None
        self.last_error = ""
        self._resume()

    # the record ----------------------------------------------------------------------------------------------------
    def _resume(self) -> None:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open() as f:
            for r in csv.DictReader(f):
                m = r["market_id"]
                if r["action"] in ("decided", "no_back"):
                    self.decided.add(m)
                if r["action"] == "back_matched":
                    h = self.held.setdefault(m, {}).setdefault(int(float(r["selection_id"])), {"stake": 0.0, "value": 0.0})
                    h["stake"] += float(r["matched"])
                    h["value"] += float(r["matched"]) * float(r["price"])
                if r["action"] == "sp_lay":
                    k = (m, int(float(r["selection_id"])))
                    self.laid[k] = self.laid.get(k, 0.0) + float(r["liability"] or 0)
                if r["action"] == "settled":
                    self.settled[m] = float(r["pnl"] or 0)

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
        except Exception as exc:
            log.warning("ledger not copied to S3 (%s)", exc)

    def _t(self) -> str:
        return pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ")

    def ref(self, mid: str, sid: int, kind: str) -> str:
        return f"{REF}{self.day:%m%d}-{mid}-{sid}-{kind}"[:32]

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

    # the money -----------------------------------------------------------------------------------------------------
    def open_stake(self) -> float:
        return sum(h["stake"] for m, hs in self.held.items() if m not in self.settled for h in hs.values())

    def day_pnl(self) -> float:
        """Settled net, less every open position at its worst (its whole stake)."""
        return sum(self.settled.values()) - self.open_stake()

    def may_back(self, stake: float) -> str:
        lim = self.cfg.get("daily_loss_limit")
        if self.live and lim is None:
            return "no daily loss limit"
        if lim is not None and self.day_pnl() - stake < -float(lim) - 1e-9:
            return "the day's loss limit"
        if self.open_stake() + stake > self.cfg["max_open_stake"] + 1e-9:
            return "the open-stake guard"
        return ""

    def heartbeat(self, force: bool = False, **extra) -> None:
        t = self.now()
        if not force and self._beat_at and (t - self._beat_at).total_seconds() < 60:
            return
        self._beat_at = t
        st = {"time_utc": self._t(), "mode": self.mode, "markets_decided": len(self.decided),
              "positions": sum(len(v) for v in self.held.values()), "open_stake": round(self.open_stake(), 2),
              "settled_net": round(sum(self.settled.values()), 2), "day_pnl_worst": round(self.day_pnl(), 2),
              "daily_loss_limit": self.cfg.get("daily_loss_limit"), "stopped": bool(self.kill()),
              "prices_loaded": self.preds is not None, "last_error": self.last_error[:300], **extra}
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
        unsettled = self.preds.market_id.isin([m for m in self.held if m not in self.settled])
        watch = self.preds[((to_off > 0) & (to_off <= self.cfg["watch_minutes"] * 60)) | (unsettled & (to_off > -3600))]
        mids = sorted(watch.market_id.dropna().astype(str).unique())
        if not mids:
            return
        books = self.x.books(mids, with_sp=True)
        stopped = self.kill()
        for mid in mids:
            b = books.get(mid)
            if b is None:
                continue
            g = watch[watch.market_id.astype(str) == mid]
            secs = float((pd.Timestamp(g.off.iloc[0]) - pd.Timestamp(self.now())).total_seconds())
            try:
                if mid in self.held and mid not in self.settled:
                    self.reconcile(mid, b, secs)
                if b.status == "OPEN" and not b.inplay and secs > self.cfg["min_seconds"] and mid not in self.decided:
                    self.watch(mid, b, g, secs, stopped)
            except Exception as exc:                          # one market's failure never stops the others
                log.warning("%s: %s", mid, exc)
                self.last_error = f"{mid}: {exc}"

    def watch(self, mid: str, b, g: pd.DataFrame, secs: float, stopped: bool) -> None:
        from greyhound.early_lay import first_rows
        if self.until is not None and pd.Timestamp(g.off.iloc[0]) > \
                pd.Timestamp(self.until) - pd.Timedelta(seconds=self.cfg["close_margin_seconds"]):
            return
        t = self.now()
        active = {sid: q for sid, q in b.runners.items() if q.status == "ACTIVE"}
        first = self.first.setdefault(mid, {})
        for sid, q in active.items():
            bb, bl = q.best_back, q.best_lay
            self.hist.setdefault((mid, sid), []).append((t, bb[0] if bb else np.nan, bl[0] if bl else np.nan))
            if sid not in first and q.last_traded and q.last_traded > 1 and bb:
                first[sid] = {"market_id": mid, "polled": t, "back1": bb[0], "back1_size": bb[1],
                              "lay1": bl[0] if bl else np.nan, "lay1_size": bl[1] if bl else np.nan,
                              "last_traded": float(q.last_traded), "traded": float(q.traded or 0.0),
                              "market_traded": float(b.total_matched or 0.0)}
        if not active or any(sid not in first for sid in active):
            return
        if not set(active) <= set(g.selection_id.dropna().astype(int)):
            return
        self.decided.add(mid)
        base = {"time_utc": self._t(), "mode": self.mode, "market_id": mid, "seconds_to_off": round(secs)}
        if stopped:
            self._write([{**base, "action": "no_back", "note": "stopped (S3 STOP)"}])
            return
        prev = {}
        for sid in active:
            h = [e for e in self.hist.get((mid, sid), []) if (first[sid]["polled"] - e[0]).total_seconds() >= 90]
            prev[sid] = (h[-1][1], h[-1][2]) if h else None
        d = forecast(first_rows({s: first[s] for s in active}, prev, g.assign(selection_id=g.selection_id.astype(int))),
                     self.fit)
        info = g.set_index(g.selection_id.astype(int))
        rows, takes = [], []
        for _, r in d.iterrows():
            sid = int(r.selection_id)
            q = active.get(sid)
            pl = plan(list(q.back) if q is not None else [], float(r.f_bsp), self.cfg)
            meta = info.loc[sid]
            row = {**base, "selection_id": sid, "track": meta.get("track", ""), "race_time": meta.get("race_time", ""),
                   "trap": meta.get("trap", ""), "dog": meta.get("dog_name", ""), "f_bsp": round(float(r.f_bsp), 3),
                   "back1": q.best_back[0] if q is not None and q.best_back else "", "mid": round(float(r["mid"]), 3),
                   "p_model": round(float(r.p_model), 4)}
            if pl is not None:
                takes.append((sid, pl, {**row, "limit": pl["limit"], "stake": pl["stake"],
                                        "edge": round(pl["edge"], 4)}))
            rows.append(row)
        self._write([{**base, "action": "decided", "note": f"{len(takes)} of {len(rows)} runners clear the bar"}])
        for sid, pl, row in takes:
            why = self.may_back(pl["stake"])
            if why:
                self._write([{**row, "action": "no_back", "note": why}])
                continue
            self.take(mid, sid, pl, row)

    def take(self, mid: str, sid: int, pl: dict, row: dict) -> None:
        if not self.live:
            self._write([{**row, "action": "back_paper", "note": "paper: nothing sent"}])
            return
        f = self.x.back(mid, sid, pl["limit"], pl["stake"], self.ref(mid, sid, "b"), min_fill=pl["min_fill"])
        if f.status != "SUCCESS" or f.matched <= 0:
            self._write([{**row, "action": "back_unmatched", "bet_id": f.bet_id, "note": f.error or f.status}])
            return
        h = self.held.setdefault(mid, {}).setdefault(sid, {"stake": 0.0, "value": 0.0})
        h["stake"] += f.matched
        h["value"] += f.matched * f.avg_price
        self._write([{**row, "time_utc": self._t(), "action": "back_matched", "price": f.avg_price,
                      "matched": f.matched, "bet_id": f.bet_id}])
        self.close(mid, sid, _floor2(f.matched * (f.avg_price - 1.0)), row)

    def close(self, mid: str, sid: int, liability: float, row: dict) -> None:
        if liability < MIN_SP_LIABILITY:
            self._write([{**row, "time_utc": self._t(), "action": "held", "liability": liability,
                          "note": "under Betfair's GBP10 lay at SP: held to the result"}])
            return
        f = self.x.lay_at_bsp(mid, sid, liability, self.ref(mid, sid, "s"))
        if f.status in ("PENDING", "SUCCESS"):
            self.laid[(mid, sid)] = self.laid.get((mid, sid), 0.0) + liability
        self._write([{**row, "time_utc": self._t(), "action": "sp_lay" if f.status in ("PENDING", "SUCCESS")
                      else "sp_lay_failed", "liability": liability, "bet_id": f.bet_id, "note": f.error}])

    def reconcile(self, mid: str, b, secs: float) -> None:
        """Betfair's own record of the market: any back whose winnings are not yet laid at the SP is closed (while the
        market is open); once settled, each position's net."""
        if not self.live:
            return
        pos = positions(self.x.orders([mid]))
        for sid, p in pos.items():
            if p["stake"] > 0:
                self.held.setdefault(mid, {})[sid] = {"stake": p["stake"], "value": p["value"]}
            # what Betfair lists, or what this session sent if more (a lay just placed may not be listed yet)
            to_close = _floor2(p["value"] - p["stake"] - max(p["sp_liability"], self.laid.get((mid, sid), 0.0)))
            if to_close >= MIN_SP_LIABILITY and b.status == "OPEN" and not b.inplay:
                self.close(mid, sid, to_close, {"market_id": mid, "selection_id": sid, "mode": self.mode,
                                                     "seconds_to_off": round(secs), "note": "repair"})
        if b.status != "CLOSED":
            return
        net, done = 0.0, True
        for sid, h in self.held.get(mid, {}).items():
            q = b.runners.get(sid)
            won = None if q is None or q.status not in ("WINNER", "LOSER") else q.status == "WINNER"
            price = h["value"] / h["stake"] if h["stake"] > 0 else 0.0
            v = close_pnl(h["stake"], price, q.bsp if q is not None else None, won)
            if v is None:
                done = False
                break
            net += v
        if done:
            net = net - costs.commission() * max(net, 0.0)
            self.settled[mid] = net
            self._write([{"time_utc": self._t(), "mode": self.mode, "market_id": mid, "action": "settled",
                          "pnl": round(net, 2), "note": "from Betfair's result and SP"}])

    def run(self, until: datetime, sleep=time.sleep) -> dict:
        self.until = until
        hard_stop = until + timedelta(minutes=self.cfg["overrun_minutes"])
        while self.now() < until or ([m for m in self.held if m not in self.settled] and self.now() < hard_stop):
            try:
                self.step()
            except Exception as exc:
                log.warning("poll failed (%s)", exc)
                self.last_error = f"poll: {exc}"
            self.heartbeat()
            sleep(self.cfg["poll_seconds"])
        self.heartbeat(force=True, finished=True)
        return {"day": f"{self.day:%Y-%m-%d}", "mode": self.mode, "markets_decided": len(self.decided),
                "positions": sum(len(v) for v in self.held.values()), "settled_net": round(sum(self.settled.values()), 2),
                "unsettled_markets": len([m for m in self.held if m not in self.settled])}


def main(argv=None) -> int:
    from betfair_recorder import uk_time_on, uk_today
    from greyhound import laytest
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--trade", action="store_true")
    ap.add_argument("--live", action="store_true", help="real orders, with GREYHOUND_TAKE_CLOSE=live and a loss limit")
    ap.add_argument("--until", default="21:30")
    ap.add_argument("--date", default=None)
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    if not a.trade:
        return 0
    limit = daily_loss_limit()
    live = bool(a.live and switch_on() and limit is not None)
    if a.live and switch_on() and limit is None:
        log.error("GREYHOUND_TAKE_CLOSE is live but GREYHOUND_TAKE_CLOSE_DAILY_LOSS is not set: paper, nothing sent")
    mode = "live" if live else "paper"
    status_key = f"{PREFIX}/{mode}/{day:%Y-%m-%d}/status.json"
    fit = laytest.live_fit(store, day)
    if fit is None:
        store.put(status_key, json.dumps({"mode": mode, "state": "stopped: no forecast fit for today"}).encode())
        raise SystemExit("no forecast fit for today: nothing decided")
    from trading.exchange import BetfairData, LiveExchange
    x = LiveExchange(BetfairData(source="greyhound"))
    x.login()
    ledger = Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound")) / \
        f"take_close_{mode}" / f"{day:%Y-%m-%d}" / "ledger.csv"
    cfg = dict(CONFIG, daily_loss_limit=limit)
    log.info("take and close (%s): bar %.2f, GBP%.0f a bet, daily loss limit %s, fit to %s", mode, cfg["bar"],
             cfg["stake"], limit, fit.get("fit_to"))
    out = TakeClose(store, x, cfg, day, ledger, fit, live, kill=kill_switch(store)).run(uk_time_on(day, a.until))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
