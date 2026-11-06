"""Replay of Betfair's Exchange Stream with queue-aware fills (read-only research; the owner, 5 Nov 2026: "yes" to the
stream replay backtester). The stream recorder (greyhound/stream_record.py) keeps every market change for the GB/IE
greyhound WIN and PLACE markets: here each market's book is rebuilt message by message and orders are simulated
against it as they would have rested in the queue.

The book (``Market``): the market definition (status, in play, the scheduled off, the runners and, once settled,
their SPs) and, for each runner, the full ladders (``atb``: the money offered to back, at each price; ``atl``: to lay),
the volume traded at each price (``trd``, cumulative), the last traded price and Betfair's projected SPs.

An order (``SimOrder``):
- placed at time t: what the book offers at or better than its price is taken at once (a LAY takes the ``atl`` offers
  at or under its price, cheapest first; a BACK the ``atb`` offers at or over it, dearest first);
- the rest rests at its price behind the money already there (a LAY rests among the ``atb`` offers at its price, a
  BACK among the ``atl``): ``queue_ahead`` is that money;
- each later change: the volume traded at its price (times ``vol_factor``: 1.0, or 0.5 if Betfair's traded volume
  counts both sides) first clears ``queue_ahead``, then fills the order; any volume traded beyond its price (under it
  for a LAY, over it for a BACK) means the market went through it, and fills it in full at its price; cancellations
  ahead are never assumed, but ``queue_ahead`` can never exceed the money still shown at the price;
- at ``close_secs`` before the off (or when the market turns in play), what is unmatched lapses and the matched part is
  closed at the best price on offer then (a LAY greened by a back at the best ``atb`` price: S x (1 - L / B); a BACK by
  a lay at the best ``atl``: S x (B / L - 1)), and beside it at the Betfair SP when the settled definition carries it.

Time: the stream's publish time (``pt``), else the recorder's receive time.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger(__name__)

PREFIX = "betfair_stream/greyhound"


@dataclass
class Runner:
    atb: dict = field(default_factory=dict)
    atl: dict = field(default_factory=dict)
    trd: dict = field(default_factory=dict)
    ltp: float | None = None
    tv: float | None = None
    spn: float | None = None
    spf: float | None = None

    def best_back(self):
        """The best price a backer can take now (the dearest money offered to back) and its size."""
        p = max(self.atb) if self.atb else None
        return (p, self.atb[p]) if p else (None, 0.0)

    def best_lay(self):
        p = min(self.atl) if self.atl else None
        return (p, self.atl[p]) if p else (None, 0.0)


@dataclass
class Market:
    id: str
    defn: dict = field(default_factory=dict)
    runners: dict = field(default_factory=dict)

    @property
    def off(self) -> datetime | None:
        t = self.defn.get("marketTime")
        try:
            return datetime.fromisoformat(str(t).replace("Z", "+00:00")) if t else None
        except ValueError:
            return None

    @property
    def inplay(self) -> bool:
        return bool(self.defn.get("inPlay"))

    def bsp(self, sid: int) -> float | None:
        for r in self.defn.get("runners", []) or []:
            if int(r.get("id", -1)) == sid and r.get("bsp"):
                try:
                    return float(r["bsp"])
                except (TypeError, ValueError):
                    return None
        return None

    def won(self, sid: int) -> int | None:
        """1 if the runner won, 0 if it lost (the settled definition's status), None before the result or if
        removed."""
        for r in self.defn.get("runners", []) or []:
            if int(r.get("id", -1)) == sid:
                st = str(r.get("status"))
                return 1 if st == "WINNER" else 0 if st in ("LOSER", "PLACED") else None
        return None


def result_pnl(side: str, price: float, stake: float, won: int | None, comm: float = 0.0) -> float | None:
    """The bet left to run to the result (target 2, the owner, 6 Nov 2026: "Win - winning the bet"): a back at B for S
    wins S (B - 1) or loses S; a lay at L for a backer's stake S loses S (L - 1) or wins S; commission on a gain (per
    bet: Betfair charges on the market's net, so this is a slight overstatement of the cost where bets net off)."""
    if won is None or not price or price <= 1 or not stake:
        return None
    if side == "BACK":
        v = stake * (price - 1) if won else -stake
    else:
        v = -stake * (price - 1) if won else stake
    return round(v * (1 - comm) if v > 0 else v, 4)


def _ladder(d: dict, pairs) -> None:
    for pr in pairs or []:
        if len(pr) < 2:
            continue
        p, v = float(pr[0]), float(pr[1])
        if v <= 0:
            d.pop(p, None)
        else:
            d[p] = v


def apply(markets: dict, mc: dict) -> tuple[str, set]:
    """One market change into the cache; returns the market and the runners changed."""
    mid = str(mc["id"])
    m = markets.get(mid)
    if m is None or mc.get("img"):
        old = m.defn if m is not None else {}
        m = markets[mid] = Market(mid, defn=old)
    if "marketDefinition" in mc:
        m.defn = mc["marketDefinition"]
    changed = set()
    for rc in mc.get("rc", []) or []:
        sid = int(rc["id"])
        r = m.runners.setdefault(sid, Runner())
        _ladder(r.atb, rc.get("atb"))
        _ladder(r.atl, rc.get("atl"))
        for pr in rc.get("trd") or []:                      # cumulative volume at each price: replaced, not added
            if len(pr) >= 2:
                if float(pr[1]) <= 0:
                    r.trd.pop(float(pr[0]), None)
                else:
                    r.trd[float(pr[0])] = float(pr[1])
        for k in ("ltp", "tv", "spn", "spf"):
            if k in rc and rc[k] is not None:
                try:
                    setattr(r, k, float(rc[k]))
                except (TypeError, ValueError):
                    pass
        changed.add(sid)
    return mid, changed


def events(lines):
    """(time, market change) for every ``mcm`` market change in the recorded lines ({"rt": ..., "m": ...})."""
    for ln in lines:
        try:
            row = json.loads(ln) if isinstance(ln, str) else ln
            msg = json.loads(row["m"])
        except (ValueError, KeyError, TypeError):
            continue
        if msg.get("op") != "mcm" or not msg.get("mc"):
            continue
        if msg.get("pt"):
            t = datetime.fromtimestamp(msg["pt"] / 1000.0, tz=timezone.utc)
        else:
            t = datetime.fromisoformat(row["rt"].replace("Z", "+00:00"))
        for mc in msg["mc"]:
            yield t, mc


def load_lines(store_get, listing, day: date) -> list[str]:
    """Every recorded line of the day, part by part in order (``listing(prefix)`` -> keys, ``store_get(key)`` ->
    bytes)."""
    keys = sorted(k for k in listing(f"{PREFIX}/{day:%Y-%m-%d}/") if k.endswith(".jsonl.gz"))
    out = []
    for k in keys:
        raw = store_get(k)
        if raw:
            out += gzip.decompress(raw).decode("utf-8", "replace").splitlines()
    return out


@dataclass
class SimOrder:
    market: str
    sid: int
    side: str                    # "LAY" or "BACK"
    price: float
    size: float
    t: datetime                  # when it goes in
    tag: str = ""
    queue_ahead: float = 0.0
    queue_at_place: float = 0.0
    matched: float = 0.0
    value: float = 0.0           # sum of stake x price matched
    taken: float = 0.0           # matched at once against the book
    placed: bool = False
    done: bool = False
    first_fill: datetime | None = None
    last_trd: float = 0.0
    last_through: float = 0.0
    via_through: bool = False    # filled because the market traded beyond the price (the dog moved through it)

    @property
    def remaining(self) -> float:
        return max(self.size - self.matched, 0.0)

    @property
    def avg(self) -> float | None:
        return self.value / self.matched if self.matched > 0 else None

    def _fill(self, size: float, price: float, t: datetime) -> None:
        size = min(size, self.remaining)
        if size <= 0:
            return
        self.matched += size
        self.value += size * price
        if self.first_fill is None:
            self.first_fill = t

    def _through(self, r: Runner) -> float:
        return sum(v for p, v in r.trd.items() if (p < self.price - 1e-9 if self.side == "LAY" else p > self.price + 1e-9))

    def place(self, r: Runner, t: datetime) -> None:
        """Take what the book offers at or better than the price, then rest behind the money at the price."""
        self.placed = True
        if self.side == "LAY":
            for p in sorted(x for x in r.atl if x <= self.price + 1e-9):
                self._fill(r.atl[p], p, t)
            same = r.atb.get(self.price, 0.0)
        else:
            for p in sorted((x for x in r.atb if x >= self.price - 1e-9), reverse=True):
                self._fill(r.atb[p], p, t)
            same = r.atl.get(self.price, 0.0)
        self.taken = self.matched
        self.queue_ahead = self.queue_at_place = same
        self.last_trd = r.trd.get(self.price, 0.0)
        self.last_through = self._through(r)

    def update(self, r: Runner, t: datetime, vol_factor: float = 1.0) -> None:
        if self.done or not self.placed or self.remaining <= 0:
            return
        now_trd = r.trd.get(self.price, 0.0)
        d = max(now_trd - self.last_trd, 0.0) * vol_factor
        self.last_trd = now_trd
        if d > 0:
            clear = min(self.queue_ahead, d)
            self.queue_ahead -= clear
            self._fill(d - clear, self.price, t)
        through = self._through(r)
        if through > self.last_through + 1e-9 and self.remaining > 0:   # traded beyond our price: we went first
            self.via_through = True
            self._fill(self.remaining, self.price, t)
        self.last_through = through
        shown = (r.atb if self.side == "LAY" else r.atl).get(self.price, 0.0)
        self.queue_ahead = min(self.queue_ahead, shown)


def close_pnl(o: SimOrder, r: Runner | None, sp: float | None, comm: float = 0.0) -> dict:
    """The matched part closed at the best price on offer (green) and, beside it, at the SP."""
    out = {"close_price": None, "pnl": None, "pnl_sp": None}
    if o.matched <= 0:
        return out
    L, S = o.avg, o.matched
    if r is not None:
        if o.side == "LAY":
            b, _ = r.best_back()
            if b:
                out["close_price"], out["pnl"] = b, S * (1 - L / b)
        else:
            b, _ = r.best_lay()
            if b:
                out["close_price"], out["pnl"] = b, S * (L / b - 1)
    if sp and sp > 1:
        out["pnl_sp"] = S * (1 - L / sp) if o.side == "LAY" else S * (L / sp - 1)
    for k in ("pnl", "pnl_sp"):
        if out[k] is not None:
            out[k] = round(out[k] * (1 - comm) if out[k] > 0 else out[k], 4)
    return out


def replay(lines, orders: list[SimOrder], close_secs: float = 5.0, vol_factor: float = 1.0,
           comm: float = 0.0) -> pd.DataFrame:
    """Run the orders through the recorded stream; one row an order with its fill and its close."""
    markets: dict = {}
    pending = sorted(orders, key=lambda o: o.t)
    by_market: dict = {}
    for o in pending:
        by_market.setdefault(o.market, []).append(o)
    rows = []

    def finish(o, m):
        o.done = True
        r = m.runners.get(o.sid) if m else None
        rows.append({"tag": o.tag, "market": o.market, "sid": o.sid, "side": o.side, "price": o.price,
                     "size": o.size, "placed": o.placed, "queue_at_place": round(o.queue_at_place, 2),
                     "taken": round(o.taken, 2), "matched": round(o.matched, 2),
                     "filled_share": round(o.matched / o.size, 4) if o.size else None,
                     "avg": round(o.avg, 3) if o.avg else None, "first_fill": o.first_fill,
                     "via_through": o.via_through,
                     **close_pnl(o, r, m.bsp(o.sid) if m else None, comm)})

    for t, mc in events(lines):
        mid = str(mc.get("id"))
        mine = by_market.get(mid)
        if mine and mid in markets:
            # an order due by now goes in against the book as it stood at its own time: the market's state before
            # this change (nothing changed on the market between the order's time and this message)
            m = markets[mid]
            for o in mine:
                if not o.done and not o.placed and t >= o.t and o.sid in m.runners and not m.inplay:
                    o.place(m.runners[o.sid], o.t)
        mid, changed = apply(markets, mc)
        if not mine:
            continue
        m = markets[mid]
        cutoff = (m.off - timedelta(seconds=close_secs)) if m.off else None
        for o in mine:
            if o.done:
                continue
            if (cutoff is not None and t >= cutoff) or m.inplay:
                finish(o, m)
                continue
            r = m.runners.get(o.sid)
            if r is None:
                continue
            if not o.placed:
                if t >= o.t:
                    o.place(r, t)
                continue
            if o.sid in changed:
                o.update(r, t, vol_factor)
    for o in pending:                                       # never reached its close in the recorded stream
        if not o.done:
            m = markets.get(o.market)
            finish(o, m)
            rows[-1]["note"] = "no close in the record" if o.placed else "never placed (no book)"
    # the SP: Betfair writes it into the market definition only once the market is settled, after the close above,
    # so it is read from the last definition the record holds
    for row in rows:
        m = markets.get(row["market"])
        row["sp"] = sp = m.bsp(row["sid"]) if m else None
        if row["pnl_sp"] is None and row["matched"] and row["avg"]:
            if sp and sp > 1:
                L, S = row["avg"], row["matched"]
                v = S * (1 - L / sp) if row["side"] == "LAY" else S * (L / sp - 1)
                row["pnl_sp"] = round(v * (1 - comm) if v > 0 else v, 4)
    # target 2: the matched part left to run to the result, read from the same settled definition
    for row in rows:
        m = markets.get(row["market"])
        row["won"] = m.won(row["sid"]) if m else None
        row["pnl_result"] = (result_pnl(row["side"], row["avg"], row["matched"], row["won"], comm)
                             if row["matched"] and row["avg"] else None)
    return pd.DataFrame(rows)


def ledger_orders(led: pd.DataFrame, tag: str) -> list[SimOrder]:
    """The early lay's lays (ledger rows 'lay') as simulated LAY orders at their time, price and stake."""
    lays = led[(led.action == "lay") & pd.to_numeric(led.price, errors="coerce").gt(1)]
    out = []
    for r in lays.itertuples(index=False):
        out.append(SimOrder(market=str(r.market_id), sid=int(float(r.selection_id)), side="LAY", price=float(r.price),
                            size=float(r.size), t=pd.Timestamp(r.time_utc).to_pydatetime(), tag=tag))
    return out


def summary(df: pd.DataFrame) -> dict:
    if not len(df):
        return {"orders": 0}
    placed = df[df.placed]
    m = placed[placed.matched > 0]
    return {"orders": len(df), "placed": len(placed), "matched": len(m),
            "filled_share": round(float(placed.matched.sum() / max(placed["size"].sum(), 1e-9)), 4),
            "taken_at_once": round(float(placed.taken.sum()), 2),
            "pnl_green": round(float(pd.to_numeric(m.pnl, errors="coerce").fillna(0).sum()), 2),
            "pnl_sp": round(float(pd.to_numeric(m.pnl_sp, errors="coerce").fillna(0).sum()), 2),
            "sp_known": int(pd.to_numeric(m.pnl_sp, errors="coerce").notna().sum()),
            "pnl_result": round(float(pd.to_numeric(m.get("pnl_result"), errors="coerce").fillna(0).sum()), 2)
            if "pnl_result" in m else None,
            "result_known": int(pd.to_numeric(m.pnl_result, errors="coerce").notna().sum()) if "pnl_result" in m else 0,
            "filled_through": int(m.via_through.sum()) if "via_through" in m else 0,
            "no_book": int((~df.placed).sum())}
