"""The residual's lay at the Betfair SP, live (the owner, 21 Nov 2026: "Put any possible profitable strategies live";
then, asked: the bottom-10% SP lay only, GBP50 a lay, a GBP300 daily loss limit, live today).

The rule is the paper forward test's gate rule (ledger greyhound-sp-offset-forward-1120; research/greyhound/
sp_offset_forward.py): the frozen residual on the BSP, served from the morning card (S3 sources/greyhound/sp_offset/
served/<day>.csv), and every runner whose centred residual is at or under the frozen lay_bottom10 cut is laid at the
Betfair SP. This trader reads the served picks and nothing else of the model: it never prices a race itself.

- **When**: each pick is laid once, in the last ``place_from_seconds`` (600) before its race's scheduled off and no
  later than ``place_to_seconds`` (20) before it, while the market is open and not in play.
- **How**: a MARKET_ON_CLOSE lay for ``liability`` (GBP50) of liability (``trading.exchange.LiveExchange.lay_at_bsp``,
  the horse trader's own SP lay), matched at the SP when the market reconciles at the off.
- **What is skipped**: a pick whose runner, or any runner served in its race, is no longer active (the paper test
  counts only races where every served runner gets an SP); a market that is closed, suspended or in play.
- **The daily loss limit** (``daily_loss_limit``, GBP300): no lay whose worst case (the day's settled net after
  commission, less every placed lay not yet settled at its full liability, less this lay's) would take the day past
  minus the limit; and none at all once the settled net reaches minus the limit. So the day can never lose more than
  the limit.
- **Stop**: the S3 objects ``greyhound/sp_lay/STOP`` and ``trading/STOP`` (greyhound-stop.yml trader sp_lay) stop new
  lays within a poll and cancel the SP lays not yet matched (markets not yet off). An unreadable switch counts as
  STOP. ``greyhound/STOP`` (the early lay's) does not stop this trader.
- **Real orders** only with ``--live`` and ``GREYHOUND_SP_LAY=live``; otherwise the same decisions are logged and
  nothing is sent. Orders carry ``spl-<mmdd>-<market>-<selection>`` references. Ledger:
  ``sources/greyhound/sp_lay/<mode>/<day>/ledger.csv``; status beside it.

    GREYHOUND_SP_LAY=live python -m greyhound.sp_lay --trade --live --until 21:45
    python -m greyhound.sp_lay --reconcile --date 2026-11-21
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger(__name__)

PREFIX = "greyhound/sp_lay"
REF = "spl-"
STOP_KEYS = ("greyhound/sp_lay/STOP", "trading/STOP")
RULE = "lay_bottom10"
CONFIG = {
    "liability": 50.0,                 # the owner, 21 Nov 2026: "GBP50 a lay"
    "daily_loss_limit": 300.0,         # the owner, 21 Nov 2026: "GBP300"
    "place_from_seconds": 600.0,       # lay in the last 10 minutes before the off, so the loss limit acts race by race
    "place_to_seconds": 20.0,          # and never in the last 20 seconds (the market may already be suspending)
    "poll_seconds": 15.0,
    "reload_seconds": 120.0,           # the served file: later serves add the races listed late
    "settled_every_seconds": 120.0,    # the day's settled net from Betfair's cleared orders
}
LEDGER_FIELDS = ["time_utc", "mode", "action", "market_id", "selection_id", "race_id", "track", "race_time", "trap",
                 "dog", "s", "off_utc", "seconds_to_off", "liability", "bet_id", "status", "error", "settled_net",
                 "open_liability", "note"]


def switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("GREYHOUND_SP_LAY", "")).strip().lower() == "live"


def ref_for(day: date, market_id: str, selection_id) -> str:
    return f"{REF}{day:%m%d}-{market_id}-{int(selection_id)}"[:32]


def kill_switch(store, keys=STOP_KEYS, every: float = 5.0):
    """True while S3 holds any of ``keys``; an unreadable switch counts as STOP (greyhound.live._exists)."""
    from greyhound.live import _exists
    state = {"at": 0.0, "on": False}

    def check() -> bool:
        now = time.monotonic()
        if state["at"] and now - state["at"] < every:
            return state["on"]
        state["at"] = now
        state["on"] = any(_exists(store, k) for k in keys)
        return state["on"]
    return check


# --------------------------------------------------------------------------------------------------------------------
# The picks and the decisions (pure: tested without an exchange)
# --------------------------------------------------------------------------------------------------------------------

def picks(served: pd.DataFrame, cut: float) -> pd.DataFrame:
    """The served runners the rule lays: centred residual at or under the frozen cut, with a market and a selection."""
    s = served.assign(s=pd.to_numeric(served.s, errors="coerce"))
    p = s[(s.s <= cut) & s.market_id.notna() & s.selection_id.notna()].copy()
    p["market_id"] = p.market_id.astype(str)
    p["selection_id"] = pd.to_numeric(p.selection_id, errors="coerce").astype("int64")
    return p


def due(p: pd.DataFrame, now: datetime, done: set, cfg: dict) -> pd.DataFrame:
    """The picks to lay now: not laid or decided yet, off between ``place_to`` and ``place_from`` seconds from now."""
    if not len(p):
        return p
    secs = (pd.to_datetime(p.off_utc, utc=True, errors="coerce") - pd.Timestamp(now)).dt.total_seconds()
    key = [(m, int(s)) not in done for m, s in zip(p.market_id, p.selection_id)]
    mask = (secs <= cfg["place_from_seconds"]) & (secs >= cfg["place_to_seconds"]) & pd.Series(key, index=p.index)
    out = p[mask].copy()                       # never .assign a full Series to an empty frame: it rebuilds the rows
    out["seconds_to_off"] = secs[mask].to_numpy()
    return out


def race_intact(served_race: pd.DataFrame, book) -> tuple[bool, str]:
    """Whether a race may be laid: the market open and not in play, and every runner served in it still active."""
    if book is None:
        return False, "no book"
    if str(book.status) != "OPEN" or book.inplay:
        return False, f"market {book.status}{' in play' if book.inplay else ''}"
    for sid in pd.to_numeric(served_race.selection_id, errors="coerce").dropna().astype("int64"):
        q = book.runners.get(int(sid))
        if q is None or str(q.status) != "ACTIVE":
            return False, f"runner {int(sid)} {getattr(q, 'status', 'missing')}"
    return True, ""


def loss_room(settled_net: float, open_liability: float, liability: float, limit: float) -> tuple[bool, str]:
    """A lay is allowed only if the day, at its worst (every open lay and this one lost), stays within the limit."""
    if settled_net <= -limit:
        return False, f"the day's settled net {settled_net:.2f} has reached the limit -{limit:.2f}"
    worst = settled_net - open_liability - liability
    if worst < -limit - 1e-9:
        return False, f"worst case {worst:.2f} would pass the limit -{limit:.2f}"
    return True, ""


def net_after_commission(profit_by_market: dict, comm: float) -> float:
    """Betfair's profit on our bets, a market at a time, commission off a market's gain."""
    return float(sum(v - comm * max(v, 0.0) for v in profit_by_market.values()))


# --------------------------------------------------------------------------------------------------------------------
# The trader
# --------------------------------------------------------------------------------------------------------------------

class SpLay:
    def __init__(self, store, x, cfg: dict, day: date, live: bool, kill=lambda: False, comm: float = 0.02,
                 clock=lambda: datetime.now(timezone.utc)):
        self.store, self.x, self.cfg, self.day, self.live = store, x, cfg, day, live
        self.kill, self.comm, self.clock = kill, comm, clock
        self.mode = "live" if live else "paper"
        self.rows: list[dict] = []
        self.done: set = set()                      # (market, selection) laid or decided not to lay
        self.placed: dict = {}                      # (market, selection) -> {bet, liability, off} of each lay sent
        self.held: set = set()                      # picks held back by the loss limit (logged once)
        self._status_at = 0.0
        self.served = pd.DataFrame()
        self.cut: float | None = None
        self.settled = {"net": 0.0, "markets": set(), "at": None}
        self.stopped_cancelled = False

    # the record ------------------------------------------------------------------------------------------------
    def _key(self, name: str) -> str:
        return f"{PREFIX}/{self.mode}/{self.day:%Y-%m-%d}/{name}"

    def _write(self, row: dict) -> None:
        row = {"time_utc": self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode, **row}
        self.rows.append({k: row.get(k, "") for k in LEDGER_FIELDS})
        try:
            buf = io.StringIO()
            pd.DataFrame(self.rows, columns=LEDGER_FIELDS).to_csv(buf, index=False)
            self.store.put(self._key("ledger.csv"), buf.getvalue().encode())
        except Exception as exc:                    # the record never stops the trading
            log.warning("ledger not written: %s", exc)

    def status(self, state: str, force: bool = False, **extra) -> None:
        if not force and time.monotonic() - self._status_at < 60:
            return
        self._status_at = time.monotonic()
        try:
            self.store.put(self._key("status.json"), json.dumps({
                "time_utc": self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode, "state": state,
                "lays_sent": len(self.placed), "settled_net": round(self.settled["net"], 2),
                "open_liability": round(self.open_liability(), 2), **extra}, default=str).encode())
        except Exception as exc:
            log.warning("status not written: %s", exc)

    def resume(self) -> None:
        """A restarted session: what an earlier one decided today (its ledger) is not decided again, and the lays it
        sent count against the limit."""
        raw = self.store.get(self._key("ledger.csv"))
        if not raw:
            return
        led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
        self.rows = led.reindex(columns=LEDGER_FIELDS).fillna("").to_dict("records")
        for r in self.rows:
            k = (str(r["market_id"]), int(float(r["selection_id"]))) if str(r["selection_id"]) not in ("", "nan") else None
            if k is None:
                continue
            if r["action"] in ("lay", "paper_lay", "skip"):
                self.done.add(k)
            if r["action"] in ("lay", "paper_lay") and str(r["status"]) in ("PENDING", "PAPER", "UNKNOWN"):
                self.placed[k] = {"bet": str(r.get("bet_id") or ""),
                                  "liability": float(r.get("liability") or self.cfg["liability"]),
                                  "off": str(r.get("off_utc") or "")}
        log.info("resumed: %d decided, %d lays sent", len(self.done), len(self.placed))

    # the served picks ------------------------------------------------------------------------------------------
    def load(self) -> None:
        from research.greyhound import sp_offset_forward as fw
        if self.cut is None:
            raw = self.store.get(fw.META_KEY)
            if raw:
                self.cut = float(json.loads(raw)["cuts"][RULE])
        raw = self.store.get(f"{fw.PREFIX}/served/{self.day:%Y-%m-%d}.csv")
        if raw:
            self.served = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str, "race_id": str})

    # the day's money -------------------------------------------------------------------------------------------
    def open_liability(self) -> float:
        """Every lay sent and not yet settled, at its full liability. Live: settled once Betfair's cleared orders hold
        its market. Paper (no settlement to read): open until 10 minutes after its off."""
        if self.live:
            return float(sum(v["liability"] for (m, _), v in self.placed.items() if m not in self.settled["markets"]))
        cutoff = pd.Timestamp(self.clock()) - pd.Timedelta(minutes=10)
        return float(sum(v["liability"] for v in self.placed.values()
                         if pd.isna(pd.to_datetime(v["off"], utc=True, errors="coerce"))
                         or pd.to_datetime(v["off"], utc=True) > cutoff))

    def refresh_settled(self, force: bool = False) -> None:
        if not self.live or not self.placed:
            return
        now = time.monotonic()
        if not force and self.settled["at"] and now - self.settled["at"] < self.cfg["settled_every_seconds"]:
            return
        self.settled["at"] = now
        mids = sorted({m for m, _ in self.placed})
        tag = f"{REF}{self.day:%m%d}-"
        try:
            ours = [c for c in self.x.cleared(mids) if str(c.get("customerOrderRef") or "").startswith(tag)]
        except Exception as exc:                    # unknown: the open lays still count at their full liability
            log.warning("cleared orders not read (%s)", exc)
            return
        paid: dict = {}
        for c in ours:
            paid[str(c.get("marketId"))] = paid.get(str(c.get("marketId")), 0.0) + float(c.get("profit") or 0)
        self.settled["net"] = net_after_commission(paid, self.comm)
        self.settled["markets"] = set(paid)

    # one poll --------------------------------------------------------------------------------------------------
    def step(self) -> int:
        stopped = self.kill()
        if stopped:
            if self.live and not self.stopped_cancelled:
                self.cancel_open("stopped (S3 STOP)")
                self.stopped_cancelled = True
            return 0
        self.stopped_cancelled = False
        if self.cut is None or not len(self.served):
            return 0
        now = self.clock()
        p = picks(self.served, self.cut)
        d = due(p, now, self.done, self.cfg)
        if not len(d):
            return 0
        books = {}
        try:
            books = self.x.books(sorted(set(d.market_id)))
        except Exception as exc:
            log.warning("books not read (%s): nothing laid this poll", exc)
            return 0
        self.refresh_settled()
        sent = 0
        for r in d.itertuples(index=False):
            k = (str(r.market_id), int(r.selection_id))
            base = {"market_id": r.market_id, "selection_id": int(r.selection_id), "race_id": r.race_id,
                    "track": getattr(r, "track", ""), "race_time": getattr(r, "race_time", ""),
                    "trap": getattr(r, "trap", ""), "dog": getattr(r, "dog_name", ""), "s": round(float(r.s), 5),
                    "off_utc": r.off_utc, "seconds_to_off": round(float(r.seconds_to_off), 1),
                    "liability": self.cfg["liability"]}
            ok, why = race_intact(self.served[self.served.race_id.astype(str) == str(r.race_id)], books.get(r.market_id))
            if not ok:
                self.done.add(k)
                self._write({**base, "action": "skip", "note": why})
                continue
            ok, why = loss_room(self.settled["net"], self.open_liability(), self.cfg["liability"],
                                self.cfg["daily_loss_limit"])
            if not ok:                              # not marked done: room may come back as races settle
                if k not in self.held:
                    self.held.add(k)
                    self._write({**base, "action": "hold", "note": why, "settled_net": round(self.settled["net"], 2),
                                 "open_liability": round(self.open_liability(), 2)})
                continue
            self.done.add(k)
            if not self.live:
                self.placed[k] = {"bet": "", "liability": self.cfg["liability"], "off": str(r.off_utc)}
                self._write({**base, "action": "paper_lay", "status": "PAPER"})
                log.info("paper lay %s %s trap %s (s %.4f), %.0f s to the off", r.market_id, getattr(r, "track", ""),
                         getattr(r, "trap", ""), float(r.s), float(r.seconds_to_off))
                continue
            f = self.x.lay_at_bsp(r.market_id, int(r.selection_id), self.cfg["liability"],
                                  ref_for(self.day, r.market_id, r.selection_id))
            if f.status in ("PENDING", "UNKNOWN"):  # UNKNOWN: counted as laid, so the limit holds either way
                self.placed[k] = {"bet": f.bet_id, "liability": self.cfg["liability"], "off": str(r.off_utc)}
                sent += 1
            log.info("lay at SP %s %s trap %s: %s %s %s", r.market_id, getattr(r, "track", ""), getattr(r, "trap", ""),
                     f.status, f.bet_id, f.error)
            self._write({**base, "action": "lay", "bet_id": f.bet_id, "status": f.status, "error": f.error,
                         "settled_net": round(self.settled["net"], 2), "open_liability": round(self.open_liability(), 2)})
        return sent

    def cancel_open(self, why: str) -> None:
        """The SP lays not yet matched (their markets not off), cancelled by bet id: never another trader's orders."""
        for (m, s), v in list(self.placed.items()):
            bet = v["bet"]
            if not bet or m in self.settled["markets"]:
                continue
            try:
                reply = self.x._call("cancelOrders", {"marketId": m, "instructions": [{"betId": bet}]}) or {}
                st = str(reply.get("status") or "")
            except Exception as exc:
                st = f"FAILURE: {exc}"[:200]
            self._write({"market_id": m, "selection_id": s, "action": "cancel", "bet_id": bet, "status": st, "note": why})

    def run(self, until: datetime, sleep=time.sleep) -> dict:
        self.resume()
        last_load = 0.0
        self.status("running", force=True)
        try:
            while self.clock() < until:
                if time.monotonic() - last_load >= self.cfg["reload_seconds"] or not len(self.served):
                    self.load()
                    last_load = time.monotonic()
                try:
                    self.step()
                except Exception as exc:            # one bad poll never ends the day
                    log.exception("poll failed: %s", exc)
                self.status("stopped (S3 STOP)" if self.kill() else "running")
                sleep(self.cfg["poll_seconds"])
        finally:
            self.refresh_settled(force=True)
            self.status("finished", force=True)
        lays = [r for r in self.rows if r["action"] in ("lay", "paper_lay")]
        return {"day": f"{self.day}", "mode": self.mode, "picks_served": int(len(picks(self.served, self.cut)))
                if self.cut is not None and len(self.served) else 0, "lays": len(lays),
                "skips": sum(1 for r in self.rows if r["action"] == "skip"),
                "holds": sum(1 for r in self.rows if r["action"] == "hold"),
                "settled_net": round(self.settled["net"], 2), "open_liability": round(self.open_liability(), 2)}


def reconcile(store, x, day: date, comm: float) -> dict:
    """A live day against Betfair's settled bets: per market what Betfair paid on our ``spl-`` lays; the day's net
    after commission. Written beside the ledger."""
    raw = store.get(f"{PREFIX}/live/{day:%Y-%m-%d}/ledger.csv")
    if not raw:
        return {"day": f"{day}", "note": "no data found: no live ledger"}
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    lays = led[(led.action == "lay") & led.status.isin(["PENDING", "UNKNOWN"])]
    mids = sorted(set(lays.market_id.dropna().astype(str)))
    if not mids:
        return {"day": f"{day}", "note": "no lays"}
    tag = f"{REF}{day:%m%d}-"
    ours = [c for c in x.cleared(mids) if str(c.get("customerOrderRef") or "").startswith(tag)]
    paid: dict = {}
    for c in ours:
        paid[str(c.get("marketId"))] = paid.get(str(c.get("marketId")), 0.0) + float(c.get("profit") or 0)
    out = {"day": f"{day}", "lays": int(len(lays)), "markets": len(mids), "settled": len(paid),
           "unsettled": [m for m in mids if m not in paid], "profit_gross": round(sum(paid.values()), 2),
           "profit_net": round(net_after_commission(paid, comm), 2), "commission": comm,
           "liability_laid": round(float(pd.to_numeric(lays.liability, errors="coerce").sum()), 2)}
    try:
        store.put(f"{PREFIX}/live/{day:%Y-%m-%d}/reconcile.json", json.dumps(out, indent=1).encode())
    except Exception as exc:
        log.warning("reconciliation not written (%s)", exc)
    return out


def main(argv=None) -> int:
    from betfair_recorder import uk_time_on, uk_today
    from greyhound import costs
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--trade", action="store_true")
    ap.add_argument("--live", action="store_true", help="real orders (only with GREYHOUND_SP_LAY=live)")
    ap.add_argument("--until", default="21:45")
    ap.add_argument("--date", default=None)
    ap.add_argument("--reconcile", action="store_true", help="a live day against Betfair's settled bets")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    comm = costs.commission()
    out: dict = {}
    if a.reconcile:
        from trading.exchange import BetfairData, LiveExchange
        x = LiveExchange(BetfairData(source="greyhound"))
        x.login()
        out["reconcile"] = reconcile(store, x, date.fromisoformat(a.date) if a.date else uk_today() - timedelta(days=1), comm)
    if a.trade:
        from trading.exchange import BetfairData, LiveExchange
        day = date.fromisoformat(a.date) if a.date else uk_today()
        live = a.live and switch_on()
        if a.live and not live:
            log.warning("GREYHOUND_SP_LAY is not 'live': the decisions are logged, nothing is sent")
        cfg = dict(CONFIG)
        if not (math.isfinite(cfg["daily_loss_limit"]) and cfg["daily_loss_limit"] > 0):
            raise SystemExit("no daily loss limit: nothing traded")
        x = LiveExchange(BetfairData(source="greyhound"))
        x.login()
        if live:
            try:
                out["reconcile_yesterday"] = reconcile(store, x, day - timedelta(days=1), comm)
            except Exception as exc:
                log.warning("yesterday not reconciled (%s)", exc)
        log.info("sp lay %s: GBP%.2f liability a lay, daily loss limit GBP%.2f, commission %.3f", "live" if live
                 else "paper", cfg["liability"], cfg["daily_loss_limit"], comm)
        out["trade"] = SpLay(store, x, cfg, day, live, kill=kill_switch(store), comm=comm).run(uk_time_on(day, a.until))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
