"""Paper trading every football market from the live feed: executable prices, depth, queues, the close, the result.

``football.exchange_feed.Feed`` hands each book read here. Nothing is sent to Betfair. For every selection of every
market recorded:

**Crossing lines.** At each checkpoint before the off (``CHECKPOINTS``, minutes), the first open pre-play read
inside the checkpoint's tolerance gives one line a side. A back walks the offered backs (best first), a lay the
offered lays, for each stake in ``STAKES``. Each line keeps the volume-weighted price and how much of each stake the
shown depth would have filled, the best prices, the spread in ticks and the book's overround.

**Resting lines.** At the checkpoints in ``REST_CHECKPOINTS``, a GBP10 back and a GBP10 lay are posted, each one tick
inside the spread, or joining the best price on the other side when the spread is one tick. Each is filled from the
recorded traded volume and books:

- The queue ahead starts at the money shown at that price (zero when the price improves on the book).
- It is cut to the money still shown there, which bounds what is left ahead of us.
- Money traded at the price fills the queue first, then the order.
- The order is filled outright once the book trades through its price: for a back at P, a best back at P or longer.
- An unfilled order is cancelled at the last checkpoint (2 minutes before the off).

**The close and the result.** The close is each selection's last open, pre-play book. The result is Betfair's own
(the runner WINNER or LOSER, read after the match). Each line gets:

- ``green_close``: the return a stake unit had it been greened at the close, crossing the spread.
- ``clv_mid``: the price against the close mid.
- ``pnl_<stake>``: held to the result, after 2% commission on a winning line.

Lines are written by match day to ``paper_lines.csv`` beside the feed's record.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

CHECKPOINTS = (1440.0, 360.0, 120.0, 60.0, 15.0, 2.0)
REST_CHECKPOINTS = (1440.0, 360.0, 120.0, 60.0)
REST_EXPIRE_MIN = 2.0
STAKES = (2.0, 10.0, 50.0, 200.0)
REST_STAKE = 10.0
COMMISSION = 0.02


def tolerance_min(cp: float) -> float:
    """How late after a checkpoint a read may still stand for it: the feed's gap there, plus a minute."""
    from football.exchange_feed import cadence_seconds
    return cadence_seconds(cp - 1e-6) / 60.0 + 1.0


def _next_up(p: float) -> float:
    from trading.exchange import tick_up
    return tick_up(float(p) + 1e-6)


def _next_down(p: float) -> float:
    from trading.exchange import tick_down
    return tick_down(float(p) - 1e-6)


def spread_ticks(back: float, lay: float, cap: int = 100) -> int:
    n, p = 0, float(back)
    while p < float(lay) - 1e-9 and n < cap:
        p, n = _next_up(p), n + 1
    return n


def walk(levels: list[tuple[float, float]], stake: float) -> tuple[float | None, float]:
    """(volume-weighted price, filled) taking ``stake`` from the levels in order (best first)."""
    left, got, cost = stake, 0.0, 0.0
    for p, s in levels:
        if left <= 1e-9:
            break
        take = min(s, left)
        got, cost, left = got + take, cost + take * p, left - take
    return (cost / got if got > 0 else None), round(got, 2)


def _levels(ex: dict, key: str) -> list[tuple[float, float]]:
    return [(float(o["price"]), float(o["size"])) for o in (ex.get(key) or [])
            if o.get("price") is not None and o.get("size") is not None]


def _ladder(ex: dict) -> dict[float, float]:
    return {float(o["price"]): float(o["size"]) for o in (ex.get("tradedVolume") or [])
            if o.get("price") is not None and o.get("size") is not None}


class PaperBook:
    def __init__(self, checkpoints=CHECKPOINTS, rest_checkpoints=REST_CHECKPOINTS, stakes=STAKES):
        self.checkpoints, self.rest_checkpoints, self.stakes = tuple(checkpoints), tuple(rest_checkpoints), stakes
        self.meta: dict[str, dict] = {}            # market_id -> catalogue summary
        self.lines: list[dict] = []
        self.taken: set = set()                   # (market, selection, checkpoint)
        self.resting: dict = {}                   # (market, selection) -> its open resting lines
        self.by_market: dict = {}                 # market_id -> its lines
        self.done: set = set()                    # markets whose lines are all settled
        self.closes: dict = {}                    # (market, selection) -> close
        self.results: dict = {}                   # (market, selection) -> 1 winner, 0 loser
        self.ladders: dict = {}                   # (market, selection) -> traded ladder last read
        self.result_markets: set = set()

    def add_market(self, m: dict) -> None:
        ev, comp, desc = m.get("event") or {}, m.get("competition") or {}, m.get("description") or {}
        from football.exchange_feed import _utc, uk_date
        st = _utc(m.get("marketStartTime"))
        self.meta[m["marketId"]] = {
            "market_type": desc.get("marketType") or m.get("marketType") or "", "event": ev.get("name"),
            "country": ev.get("countryCode"), "competition": comp.get("name"), "start": st,
            "day": uk_date(st) if st else None,
            "runners": {int(r["selectionId"]): r.get("runnerName") for r in m.get("runners") or []}}

    # -- every read --------------------------------------------------------------------------------------------------

    def on_book(self, raw: dict, start: datetime, now: datetime) -> None:
        mid = raw.get("marketId")
        if mid not in self.meta:
            return
        mins = (start - now).total_seconds() / 60.0
        open_pre = raw.get("status") == "OPEN" and not raw.get("inplay")
        runners = raw.get("runners") or []
        for r in runners:
            sid = int(r["selectionId"])
            key = (mid, sid)
            if r.get("status") in ("WINNER", "LOSER"):
                self.results[key] = 1 if r["status"] == "WINNER" else 0
                self.result_markets.add(mid)
            ex = r.get("ex") or {}
            backs, lays = _levels(ex, "availableToBack"), _levels(ex, "availableToLay")
            lad = _ladder(ex)
            delta = {p: v - self.ladders.get(key, {}).get(p, 0.0) for p, v in lad.items()} if key in self.ladders else {}
            if lad:
                self.ladders[key] = lad
            if open_pre and backs and lays:
                self.closes[key] = {"close_back": backs[0][0], "close_lay": lays[0][0],
                                    "close_mid": (backs[0][0] + lays[0][0]) / 2, "close_utc": now.isoformat()}
            self._update_resting(key, backs, lays, delta, mins, open_pre, now)
        if open_pre and mins > 0:
            self._checkpoints(mid, runners, mins, now)
        if raw.get("status") == "CLOSED" or mid in self.result_markets:
            self._settle(mid)

    def _checkpoints(self, mid: str, runners: list[dict], mins: float, now: datetime) -> None:
        books = {int(r["selectionId"]): (_levels(r.get("ex") or {}, "availableToBack"),
                                         _levels(r.get("ex") or {}, "availableToLay")) for r in runners
                 if r.get("status", "ACTIVE") == "ACTIVE"}
        backs_best = [b[0][0] for b, _ in books.values() if b]
        over = sum(1 / p for p in backs_best) if len(backs_best) == len(books) and books else None
        for cp in self.checkpoints:
            if not (cp - tolerance_min(cp) <= mins <= cp):
                continue
            for sid, (backs, lays) in books.items():
                if (mid, sid, cp) in self.taken or not backs or not lays:
                    continue
                self.taken.add((mid, sid, cp))
                base = self._base(mid, sid, cp, mins, now)
                base.update({"best_back": backs[0][0], "best_back_size": backs[0][1], "best_lay": lays[0][0],
                             "best_lay_size": lays[0][1], "spread_ticks": spread_ticks(backs[0][0], lays[0][0]),
                             "book_over": round(over, 4) if over else None,
                             "depth_back_gbp": round(sum(s for _, s in backs), 2),
                             "depth_lay_gbp": round(sum(s for _, s in lays), 2)})
                for side, levels in (("back", backs), ("lay", lays)):
                    line = dict(base, kind="cross", side=side)
                    for s in self.stakes:
                        px, got = walk(levels, s)
                        line[f"px_{s:g}"], line[f"got_{s:g}"] = (round(px, 4) if px else None), got
                    self._add(line)
                if cp in self.rest_checkpoints:
                    for side in ("back", "lay"):
                        self._post(base, side, backs, lays)

    def _base(self, mid: str, sid: int, cp: float, mins: float, now: datetime) -> dict:
        m = self.meta[mid]
        return {"day": m["day"], "market_id": mid, "market_type": m["market_type"], "event": m["event"],
                "country": m["country"], "competition": m["competition"], "selection_id": sid,
                "runner": m["runners"].get(sid), "kickoff_utc": m["start"].isoformat() if m["start"] else None,
                "checkpoint_min": cp, "minutes_to_off": round(mins, 1), "time_utc": now.isoformat()}

    # -- resting orders ----------------------------------------------------------------------------------------------

    def _post(self, base: dict, side: str, backs, lays) -> None:
        b1, l1 = backs[0][0], lays[0][0]
        if side == "back":
            p = _next_up(b1)
            price, queue = (p, 0.0) if p < l1 - 1e-9 else (l1, lays[0][1])
        else:
            p = _next_down(l1)
            price, queue = (p, 0.0) if p > b1 + 1e-9 else (b1, backs[0][1])
        line = dict(base, kind="rest", side=side, post_price=price, queue_ahead=round(queue, 2), stake=REST_STAKE,
                    filled=0.0, fill_utc=None, fill_how=None)
        self._add(line)
        self.resting.setdefault((line["market_id"], line["selection_id"]), []).append(line)

    def _add(self, line: dict) -> None:
        self.lines.append(line)
        self.by_market.setdefault(line["market_id"], []).append(line)

    def _update_resting(self, key, backs, lays, delta, mins, open_pre, now) -> None:
        if key not in self.resting:
            return
        keep = []
        for o in self.resting[key]:
            if not open_pre or mins < REST_EXPIRE_MIN:
                o["expired_utc"] = now.isoformat()
                continue                          # cancelled: the off, a suspension, or the last checkpoint
            p, left = o["post_price"], o["stake"] - o["filled"]
            # a back at P rests among the offers to layers (Betfair's availableToLay); a lay among the offers to backers
            same = lays if o["side"] == "back" else backs
            shown = next((s for q, s in same if abs(q - p) < 1e-9), None)
            if shown is not None:
                o["queue_ahead"] = min(o["queue_ahead"], shown)
            elif same and ((o["side"] == "back" and p < same[0][0]) or (o["side"] == "lay" and p > same[0][0])):
                o["queue_ahead"] = 0.0             # inside the shown book and nothing there: nobody ahead
            through = (o["side"] == "back" and backs and backs[0][0] >= p - 1e-9) or \
                      (o["side"] == "lay" and lays and lays[0][0] <= p + 1e-9)
            if through:
                o["filled"], o["fill_how"] = o["stake"], "traded through"
            else:
                vol = max(0.0, delta.get(p, 0.0))
                ahead = o["queue_ahead"]
                fill = max(0.0, vol - ahead)
                o["queue_ahead"] = max(0.0, ahead - vol)
                if fill > 0:
                    o["filled"] = round(min(o["stake"], o["filled"] + fill), 2)
                    o["fill_how"] = "queue"
            if o["filled"] > 0 and o["fill_utc"] is None:
                o["fill_utc"] = now.isoformat()
            if o["filled"] + 1e-9 < o["stake"] and left > 0:
                keep.append(o)
        if keep:
            self.resting[key] = keep
        else:
            self.resting.pop(key, None)

    # -- settle ------------------------------------------------------------------------------------------------------

    def _settle(self, mid: str) -> None:
        if mid in self.done:
            return
        lines = self.by_market.get(mid, [])
        for ln in lines:
            key = (mid, ln["selection_id"])
            c = self.closes.get(key)
            if c and "close_mid" not in ln:
                ln.update(c)
                self._close_value(ln)
            if key in self.results and "won" not in ln:
                ln["won"] = self.results[key]
                self._pnl(ln)
        if lines and all("won" in ln for ln in lines):
            self.done.add(mid)

    @staticmethod
    def _unit(side: str, won: int, price: float) -> float:
        """GBP a GBP1 stake held to the result (a lay's stake is the backer's), after commission on a win."""
        g = (price - 1.0) if won else -1.0
        g = g if side == "back" else -g
        return g * (1 - COMMISSION) if g > 0 else g

    def _close_value(self, ln: dict) -> None:
        o = ln.get("post_price") if ln["kind"] == "rest" else ln.get("px_10") or ln.get("px_2")
        if not o:
            return
        if ln["side"] == "back":
            ln["green_close"] = round(o / ln["close_lay"] - 1.0, 5)
            ln["clv_mid"] = round(o / ln["close_mid"] - 1.0, 5)
        else:
            ln["green_close"] = round(1.0 - o / ln["close_back"], 5)
            ln["clv_mid"] = round(1.0 - o / ln["close_mid"], 5)

    def _pnl(self, ln: dict) -> None:
        if ln["kind"] == "rest":
            if ln["filled"] > 0:
                ln["pnl"] = round(ln["filled"] * self._unit(ln["side"], ln["won"], ln["post_price"]), 4)
                ln["stake"] = ln["filled"]
            return
        for s in self.stakes:
            px, got = ln.get(f"px_{s:g}"), ln.get(f"got_{s:g}") or 0.0
            if px and got:
                ln[f"pnl_{s:g}"] = round(got * self._unit(ln["side"], ln["won"], px), 4)
        if ln.get("pnl_10") is not None:
            ln["pnl"], ln["stake"] = ln["pnl_10"], ln["got_10"]

    # -- files -------------------------------------------------------------------------------------------------------

    def fields(self) -> list[str]:
        f = ["day", "market_id", "market_type", "event", "country", "competition", "selection_id", "runner",
             "kickoff_utc", "checkpoint_min", "minutes_to_off", "time_utc", "kind", "side", "best_back",
             "best_back_size", "best_lay", "best_lay_size", "spread_ticks", "book_over", "depth_back_gbp",
             "depth_lay_gbp"]
        for s in self.stakes:
            f += [f"px_{s:g}", f"got_{s:g}"]
        f += ["post_price", "queue_ahead", "filled", "fill_utc", "fill_how", "expired_utc", "close_back", "close_lay",
              "close_mid", "close_utc", "green_close", "clv_mid", "won", "stake", "pnl"]
        f += [f"pnl_{s:g}" for s in self.stakes]
        return f

    def files(self) -> list[tuple[str, str, bytes]]:
        """[(match day, file name, CSV bytes)], one paper_lines.csv a day."""
        by_day: dict[str, list] = {}
        for ln in self.lines:
            by_day.setdefault(ln["day"], []).append(ln)
        out = []
        for day, rows in by_day.items():
            if not day:
                continue
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=self.fields(), extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
            out.append((day, "paper_lines.csv", buf.getvalue().encode()))
        return out
