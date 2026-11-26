"""The back-to-lay strategy inside the football trader's process: decide at the checkpoints, back, lay at the close.

racing² football, 25 Nov 2026 (the owner: back-to-lay live on all four cells, GBP 2 a bet, GBP 50 a day, below the
registered bar; ledger ``football-b2l-live-1125``). It runs beside the value bets in ``football.live.trade`` and reads
nothing from Betfair of its own: the all-games feed (``football/exchange_feed.py``) in the same process polls every
match-odds and over/under 2.5 market, and this keeps the polls it hands over (``Feed.last_rows``).

For every match-odds market the feed holds:

- **decide**: at the first poll at or after 360 and at 15 minutes before the off (no later than the feed's gap there
  plus a minute), the model's features from the polls so far (``football.live_features.checkpoint_rows`` and the
  warehouse's move memory), its forecast (``football.b2l``), and for each runner whose forecast clears a cell, a back;
- **back**: GBP 2, fill-or-kill, at the best price on offer to back, once a runner a market (whichever checkpoint
  comes first), reference ``b2l-<market>-<selection>-<checkpoint>``; nothing once GBP 50 of backs have been placed
  today, nothing while a STOP object is there;
- **lay**: at the first poll in the last ``LAY_WITHIN_MIN`` minutes before the off, a lay of stake x entry price / lay
  price, fill-or-kill, at the best price on offer to lay: the same profit whatever happens. A lay that does not fill
  (or fills in part) leaves the back, or what is not covered, to stand to the result;
- **log**: every decision, order and fill, to ``football/b2l/trades/<UK day>.jsonl`` in the capture bucket.

Orders go to the exchange the trader was given: real only when the trader runs live (``FOOTBALL_LIVE=yes`` and
``--live``), simulated against the polled book otherwise.
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STAKE = 2.0
DAY_CAP = 50.0
LAY_WITHIN_MIN = 2.0
PREV_CHECKPOINT = {360: 720, 15: 30}            # the grid point the implied changes are measured from
REF_PREFIX = "b2l-"
RELOAD_SECONDS = 3600.0                         # the model and memory read again hourly
TRADES_PREFIX = "football/b2l/trades"


def _ms(t: datetime) -> int:
    return int(t.timestamp() * 1000)


def roles_of(meta: dict) -> dict:
    """selection id -> home, draw or away, as the warehouse reads them: the sides' names from 'Home v Away', the draw
    by name; else Betfair's sort order. Empty when the market is not a three-way book."""
    runners = meta.get("runners") or []
    draw = [r for r in runners if str(r.get("runnerName", "")).strip().lower() == "the draw"]
    sides = [r for r in runners if r not in draw]
    if len(draw) != 1 or len(sides) != 2:
        return {}
    name = str((meta.get("event") or {}).get("name", ""))
    home_name = name.split(" v ")[0].strip().lower() if " v " in name else ""
    by_name = [r for r in sides if str(r.get("runnerName", "")).strip().lower() == home_name]
    if len(by_name) == 1:
        home = by_name[0]
    else:
        home = sorted(sides, key=lambda r: r.get("sortPriority", 9))[0]
    away = [r for r in sides if r is not home][0]
    return {int(home["selectionId"]): "home", int(draw[0]["selectionId"]): "draw", int(away["selectionId"]): "away"}


def levels(row: dict, side: str) -> list[tuple[float, float]]:
    out = []
    for i in range(1, 6):
        p, s = row.get(f"{side}{i}"), row.get(f"{side}{i}_size")
        try:
            p, s = float(p), float(s)
        except (TypeError, ValueError):
            continue
        if math.isfinite(p) and math.isfinite(s) and p > 1.0 and s > 0:
            out.append((p, s))
    return out


def lay_fok(ex, market_id: str, selection_id: int, price: float, size: float, ref: str, book_row: dict | None):
    """A lay of ``size`` (the backer's stake) at ``price`` or better, fill-or-kill. On the live exchange it is the same
    placeOrders call the trader's backs use (its LIMIT, LAPSE, FILL_OR_KILL); on the paper exchange it fills against
    the polled lay levels. (matched, average price, status)."""
    size = math.floor(size * 100 + 1e-9) / 100
    if hasattr(ex, "_place") and hasattr(ex, "_report"):                      # the live exchange
        instruction = {"selectionId": int(selection_id), "handicap": 0, "side": "LAY", "orderType": "LIMIT",
                       "customerOrderRef": ref[:32],
                       "limitOrder": {"size": size, "price": float(price), "persistenceType": "LAPSE",
                                      "timeInForce": "FILL_OR_KILL", "minFillSize": size}}
        reply = ex._place(market_id, instruction, ref)
        if reply is not None:
            rep, ok = ex._report(reply)
            if ok and rep.get("betId"):
                return float(rep.get("sizeMatched") or 0.0), float(rep.get("averagePriceMatched") or 0.0), "SUCCESS"
            if reply.get("status") == "FAILURE":
                return 0.0, 0.0, f"FAILURE {rep.get('errorCode') or reply.get('errorCode') or ''}".strip()
        # a lost reply: Betfair's own record of the order decides
        got = [o for o in ex.orders([market_id], [ref[:32]]) if o.get("side") == "LAY"]
        m = sum(float(o.get("sizeMatched") or 0) for o in got)
        avg = (sum(float(o.get("sizeMatched") or 0) * float(o.get("averagePriceMatched") or 0) for o in got) / m
               if m else 0.0)
        return m, avg, "SUCCESS" if m else "NOT_MATCHED (reply lost)"
    left, got, cost = size, 0.0, 0.0                                         # paper
    for p, s in levels(book_row or {}, "lay"):
        if p > price + 1e-9 or left <= 0:
            break
        take = min(s, left)
        got, cost, left = got + take, cost + take * p, left - take
    if got + 1e-9 < size:
        return 0.0, 0.0, "NOT_MATCHED"
    return got, cost / got, "SUCCESS"


def back_fok(ex, market_id: str, selection_id: int, price: float, size: float, ref: str, book_row: dict | None):
    """A back of ``size`` at ``price`` or better, fill-or-kill: the trader's own live ``back`` on the live exchange;
    on paper, filled against the polled back levels. (matched, average price, status, error)."""
    if hasattr(ex, "_place"):                                                 # the live exchange
        f = ex.back(market_id, selection_id, price, size, ref, min_fill=size)
        return float(f.matched or 0.0), float(f.avg_price or 0.0), f.status, getattr(f, "error", "")
    left, got, cost = size, 0.0, 0.0
    for p, s in levels(book_row or {}, "back"):
        if p + 1e-9 < price or left <= 0:
            break
        take = min(s, left)
        got, cost, left = got + take, cost + take * p, left - take
    if got + 1e-9 < size:
        return 0.0, 0.0, "FAILURE", "NOT_MATCHED"
    return got, cost / got, "SUCCESS", ""


class B2L:
    def __init__(self, ex, store, mode: str, stop, model: dict | None = None, memory: pd.DataFrame | None = None,
                 now=None):
        from football.paper_book import tolerance_min
        self.ex, self.store, self.mode, self.stop = ex, store, mode, stop
        self.model, self.memory = model, memory
        self.tolerance_min = tolerance_min
        self.polls: dict[str, list[dict]] = {}          # market id -> every pre-off row the feed handed over
        self.latest: dict[str, dict] = {}               # market id -> selection id -> the newest row
        self.decided: set[tuple[str, int]] = set()      # (market, checkpoint)
        self.backed: dict[tuple[str, int], dict] = {}   # (market, selection) -> the back and its lay
        self.spent: dict[str, float] = {}               # UK day -> GBP backed
        self.log: list[dict] = []
        self.loaded_day = None
        self.loaded_at = 0.0
        self.now = now or (lambda: datetime.now(timezone.utc))

    # ---- inputs --------------------------------------------------------------------------------------------------
    def load(self, day: str) -> None:
        """The model and the memory, read again every hour (the nightly refit writes them after the trader's start)."""
        mono = time.monotonic()
        if self.loaded_day == day and self.model is not None and mono - self.loaded_at < RELOAD_SECONDS:
            return
        self.loaded_at = mono
        if self.loaded_day != day:
            self.restore(day)
        from football import b2l
        self.model = b2l.load(self.store) or self.model
        raw = self.store.get(b2l.MEMORY_KEY) if self.store is not None else None
        if raw:
            self.memory = pd.read_parquet(io.BytesIO(raw))
        self.loaded_day = day

    def restore(self, day: str) -> None:
        """The day's trades already written (a restarted run, 26 Nov 2026): the log kept, the stake spent counted, and
        each back of this mode held, so a back still open is laid at its close and no runner is backed twice. A back
        of the other mode (paper before a live restart) is kept in the log only."""
        if self.store is None or day in self.spent:
            return
        try:
            raw = self.store.get(f"{TRADES_PREFIX}/{day}.jsonl")
        except Exception as exc:                                              # a read failure: start the day afresh
            log.warning("b2l trades not read (%s)", exc)
            return
        if not raw:
            return
        recs = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
        self.log = recs + self.log
        for r in recs:
            if r.get("mode") != self.mode or r.get("kind") not in ("back", "lay"):
                continue
            key = (str(r["market_id"]), int(r["selection_id"]))
            if r["kind"] == "back" and "status" in r:
                self.spent[day] = self.spent.get(day, 0.0) + (STAKE if r["status"] == "SUCCESS" else 0.0)
                matched = float(r.get("matched") or 0.0)
                self.backed[key] = {"price": r.get("price"), "matched": matched, "avg": r.get("avg_price") or r.get("price"),
                                    "status": r["status"], "ref": r.get("ref"), "lay": None if matched > 0 else "not needed"}
            elif r["kind"] == "lay" and key in self.backed:
                self.backed[key]["lay"] = {k: r.get(k) for k in ("price", "size", "matched", "status")} \
                    if "status" in r else r.get("outcome", "laid")
        self.spent.setdefault(day, 0.0)
        log.info("b2l restored %d records, %d backs, GBP %.0f spent", len(recs), len(self.backed), self.spent[day])

    def observe(self, rows: list[dict]) -> None:
        for r in rows:
            mid = str(r.get("market_id"))
            self.polls.setdefault(mid, []).append(r)
            self.latest.setdefault(mid, {})[int(r["selection_id"])] = r

    # ---- the loop ------------------------------------------------------------------------------------------------
    def step(self, feed) -> None:
        now = self.now()
        from football.exchange_feed import uk_date
        day = uk_date(now)
        self.observe(getattr(feed, "last_rows", []) or [])
        self.load(day)
        if self.model is None:
            return
        for mid, mk in list(feed.markets.items()):
            meta = mk.get("meta") or {}
            if (meta.get("description") or {}).get("marketType", meta.get("marketType")) != "MATCH_ODDS":
                continue
            mins = (mk["start"] - now).total_seconds() / 60.0
            for cp in PREV_CHECKPOINT:
                if (mid, cp) not in self.decided and cp - self.tolerance_min(cp) <= mins <= cp:
                    self.decided.add((mid, cp))
                    self.decide(feed, mid, mk, cp, day)
            if 0 < mins <= LAY_WITHIN_MIN:
                for (m, sid), pos in list(self.backed.items()):
                    if m == mid and pos.get("lay") is None and pos.get("matched", 0) > 0:
                        self.lay(mid, sid, pos, day)

    def ticks(self, mid: str) -> pd.DataFrame:
        from football.live_features import ticks_from_polls
        rows = self.polls.get(mid) or []
        if not rows:
            return pd.DataFrame(columns=["selection_id", "pt", "ltp"])
        t = ticks_from_polls(pd.DataFrame(rows), {mid: "MATCH_ODDS"})
        return t[["selection_id", "pt", "ltp"]]

    def ou_market(self, feed, event_id) -> tuple[str | None, int | None]:
        for mid, mk in feed.markets.items():
            meta = mk.get("meta") or {}
            if (str((meta.get("event") or {}).get("id")) == str(event_id)
                    and (meta.get("description") or {}).get("marketType", meta.get("marketType")) == "OVER_UNDER_25"):
                over = [r for r in meta.get("runners") or [] if str(r.get("runnerName", "")).lower().startswith("over")]
                return mid, (int(over[0]["selectionId"]) if over else None)
        return None, None

    def decide(self, feed, mid: str, mk: dict, cp: int, day: str) -> None:
        from football import b2l
        from football.live_features import checkpoint_rows
        meta = mk.get("meta") or {}
        roles = roles_of(meta)
        ev = meta.get("event") or {}
        rec = {"at": self.now().isoformat(timespec="seconds"), "kind": "decision", "market_id": mid, "cp_min": cp,
               "event": ev.get("name"), "mode": self.mode}
        if not roles:
            self.log.append({**rec, "outcome": "not a three-way book"})
            return
        ou_mid, over_id = self.ou_market(feed, ev.get("id"))
        ou_ticks = None
        if ou_mid:
            from football.live_features import ticks_from_polls
            rows = self.polls.get(ou_mid) or []
            if rows:
                ou_ticks = ticks_from_polls(pd.DataFrame(rows), {ou_mid: "OVER_UNDER_25"})[["selection_id", "pt", "ltp"]]
        ko_ms = _ms(mk["start"])
        rows = checkpoint_rows(self.ticks(mid), ou_ticks, roles, over_id, mid, int(ev.get("id") or 0), ko_ms, cp,
                               PREV_CHECKPOINT[cp])
        if rows is None or rows.empty:
            self.log.append({**rec, "outcome": "no complete book at the checkpoint"})
            return
        home = [s for s, r in roles.items() if r == "home"][0]
        away = [s for s, r in roles.items() if r == "away"][0]
        now_min = int(_ms(mk["start"]) / 60_000) - cp
        if self.memory is not None:
            mem = [b2l.memory_features(self.memory, home, away, ev.get("countryCode"), r, cp, now_min)
                   for r in rows["role"]]
            rows = pd.concat([rows.reset_index(drop=True), pd.DataFrame(mem)], axis=1)
        rows["forecast"] = b2l.predict(self.model, rows)
        for _, r in rows.iterrows():
            cell = b2l.cell_for(cp, float(r["forecast"]))
            self.log.append({**rec, "selection_id": int(r["selection_id"]), "role": r["role"],
                             "forecast": round(float(r["forecast"]), 5), "cell": cell})
            if cell and (mid, int(r["selection_id"])) not in self.backed:
                self.back(mid, int(r["selection_id"]), cp, cell, day)

    def back(self, mid: str, sid: int, cp: int, cell: str, day: str) -> None:
        rec = {"at": self.now().isoformat(timespec="seconds"), "kind": "back", "market_id": mid, "selection_id": sid,
               "cp_min": cp, "cell": cell, "mode": self.mode}
        if self.stop():
            self.log.append({**rec, "outcome": "STOP"})
            return
        if self.spent.get(day, 0.0) + STAKE > DAY_CAP + 1e-9:
            self.log.append({**rec, "outcome": f"day cap GBP {DAY_CAP:.0f} reached"})
            return
        row = (self.latest.get(mid) or {}).get(sid) or {}
        lv = levels(row, "back")
        if not lv:
            self.log.append({**rec, "outcome": "no price to back"})
            return
        price = lv[0][0]
        ref = f"{REF_PREFIX}{mid}-{sid}-{cp}"
        matched, avg, status, error = back_fok(self.ex, mid, sid, price, STAKE, ref, row)
        self.spent[day] = self.spent.get(day, 0.0) + (STAKE if status == "SUCCESS" else 0.0)
        pos = {"price": price, "matched": matched, "avg": avg or price, "status": status, "ref": ref, "lay": None}
        if pos["matched"] > 0:
            self.backed[(mid, sid)] = pos
        else:
            self.backed[(mid, sid)] = {**pos, "lay": "not needed"}          # once a runner a market, filled or not
        self.log.append({**rec, "price": price, "ref": ref, "status": status, "matched": pos["matched"],
                         "avg_price": pos["avg"], "error": error})

    def lay(self, mid: str, sid: int, pos: dict, day: str) -> None:
        row = (self.latest.get(mid) or {}).get(sid) or {}
        lv = levels(row, "lay")
        rec = {"at": self.now().isoformat(timespec="seconds"), "kind": "lay", "market_id": mid, "selection_id": sid,
               "mode": self.mode}
        if not lv:
            pos["lay"] = "no price to lay"
            self.log.append({**rec, "outcome": pos["lay"]})
            return
        price = lv[0][0]
        size = pos["matched"] * pos["avg"] / price
        ref = pos["ref"].replace(REF_PREFIX, f"{REF_PREFIX}x", 1)[:32]
        matched, avg, status = lay_fok(self.ex, mid, sid, price, size, ref, row)
        pos["lay"] = {"price": price, "size": round(size, 2), "matched": matched, "avg": avg, "status": status}
        locked = (pos["matched"] * pos["avg"] / avg - pos["matched"]) if matched >= size - 0.01 and avg else None
        self.log.append({**rec, "price": price, "size": round(size, 2), "matched": matched, "avg_price": avg,
                         "status": status, "ref": ref,
                         "locked_profit_before_commission": round(locked, 4) if locked is not None else None})

    def save(self, day: str) -> None:
        if self.store is None or not self.log:
            return
        body = "\n".join(json.dumps(r, default=str) for r in self.log) + "\n"
        try:
            self.store.put(f"{TRADES_PREFIX}/{day}.jsonl", body.encode())
        except Exception as exc:                                              # the record must not stop the trader
            log.warning("b2l trades not saved (%s)", exc)
