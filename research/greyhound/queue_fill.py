"""The early lay's paper fills with a place in the queue (read-only research): ``greyhound/early_lay.py replay`` counts
a resting lay as matched whenever a later book's last traded price reaches it, which takes no account of the money
offered at that price before ours. This module replays the same ledger lays with a queue-aware fill, so a paper day
can be set beside the optimistic one (``compare``).

The fill itself is ``greyhound/queue_fill.py fill`` (the profit simulator's, tested in tests/test_greyhound_queue_fill.py);
this module adds what the early lay's replay needs around it:

- **Betfair's ladder.** A lay at L is matched at once against the backers' offers (``available to lay``: lay1..lay3)
  priced at or under L, best (lowest) first, each at its own price. The rest rests among the layers' offers (``available
  to back``: back1..back3, best = highest) at L.
- **The queue ahead at L** (``lay_queue_ahead``): none when L is above back1 (L improves the best offer) or when no
  backer's price is on show; the size shown at L when L is one of the three levels; none when L lies inside the shown
  range without a level of its own, or below the last level of a side showing fewer than three (the book shows all it
  has); otherwise L is deeper than the book shows and the queue is unknown: assumed to be the largest size on show
  (``unknown="deepest"``, the default), or left unknown (``unknown="beyond_only"``: then only money traded beyond L
  fills it, greyhound/queue_fill.py's rule).
- **Fills after placement:** the money traded at L after the order was placed first clears the queue ahead, then fills
  ours, partly where it falls short; money traded strictly under L (beyond our price) fills the order in full, since
  every offer at L, ours too, was taken first. Cancellations ahead of us are not seen: the queue never shortens.
  Nothing fills after the green-up time (``green_seconds`` before the scheduled off).
- **The ladder's start.** The traded ladder is cumulative and recorded only from the live-key recorder's first read of
  the market (15 minutes before the off). An order placed before that read has no base to subtract: the market's first
  read is taken as the base (``ladder_base`` "first read after placement"), so the money traded between the placement
  and that read is never credited to the order (conservative).
- **The exit**, kept two ways: greened up at back1 of the last book at least ``green_seconds`` before the off, after the
  placement, as ``replay`` (P&L = matched x (1 - avg / B)); and greened at the SP (P&L = matched x (1 - avg / SP)): the
  BSP where the books or the settled reads hold it, else the last projected SP before the off (near, then far).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from greyhound import early_lay as el
from greyhound import queue_fill as qf

EPS = 1e-9
DEPTH = 3
SIZE_COLS = [f"{s}{i}{x}" for s in ("back", "lay") for i in (1, 2, 3) for x in ("", "_size")]
SP_COLS = ["sp_near", "sp_far", "sp_actual"]


# --------------------------------------------------------------------------------------------------------------------
# One order
# --------------------------------------------------------------------------------------------------------------------

def book_levels(row) -> dict:
    """A book row's levels: ``back`` (the layers' offers a backer takes, best = highest first) and ``lay`` (the
    backers' offers a layer takes, best = lowest first), as [(price, size)]; a missing size reads as 0."""
    def side(name):
        out = []
        for i in range(1, DEPTH + 1):
            p = pd.to_numeric(row.get(f"{name}{i}"), errors="coerce")
            if pd.notna(p) and p > 1:
                s = pd.to_numeric(row.get(f"{name}{i}_size"), errors="coerce")
                out.append((float(p), float(s) if pd.notna(s) else 0.0))
        return out
    return {"back": side("back"), "lay": side("lay")}


def lay_queue_ahead(price: float, backs: list, unknown: str = "deepest", depth: int = DEPTH) -> tuple:
    """(money ahead of a new lay at ``price``, assumed?). ``backs``: the layers' offers, best (highest) first. None
    (with ``unknown="beyond_only"``) when ``price`` is deeper than the book shows."""
    if not backs or price > backs[0][0] + EPS:
        return 0.0, False                                    # first in line
    for p, s in backs:
        if abs(p - price) < EPS:
            return float(s), False
    if price > backs[-1][0] + EPS or len(backs) < depth:
        return 0.0, False                                    # a price the book shows nothing at
    if unknown == "beyond_only":
        return None, True
    return float(max(s for _, s in backs)), True             # deeper than shown: the largest size on show


def _taken(price: float, size: float, lays: list) -> tuple[float, float]:
    """(stake taken at once, its value: stake x price) against the backers' offers at or under ``price``."""
    taken = value = 0.0
    for p, s in lays:
        if p > price + EPS or taken >= size - EPS:
            break
        x = min(float(s), size - taken)
        taken += x
        value += x * p
    return taken, value


def lay_fill(price: float, stake: float, placed_at, book: dict, traded: pd.DataFrame | None, until=None,
             ladder_from=None, unknown: str = "deepest", through_min: float = 0.0) -> dict:
    """One resting lay of ``stake`` at ``price`` placed at ``placed_at``, matched up to ``until``.

    ``book``: the runner's book when placed (``book_levels``). ``traded``: that runner's traded ladder (polled_utc,
    price, traded: cumulative). ``ladder_from``: when the ladder's record of the market began (its first read); an
    order placed before it is credited only with money traded after it. Returns matched (stake), taken (at once),
    avg_price, queue_ahead, queue_assumed, filled_share, first_fill_at, last_fill_at, full_at, ladder_base."""
    placed_at = pd.Timestamp(placed_at)
    placed_at = placed_at if placed_at.tzinfo else placed_at.tz_localize("UTC")
    backs, lays = list(book.get("back") or []), list(book.get("lay") or [])
    q, assumed = lay_queue_ahead(price, backs, unknown)
    rest = backs
    if q is not None and backs and price <= backs[0][0] + EPS and not any(abs(p - price) < EPS for p, _ in backs):
        rest = backs + [(price, q)]                          # greyhound.queue_fill reads the queue from the levels
    bk = {"back": rest, "lay": lays}
    base_at, base = placed_at, "at placement"
    have = traded is not None and len(traded) > 0
    if have and ladder_from is not None:
        lf = pd.Timestamp(ladder_from)
        lf = (lf if lf.tzinfo else lf.tz_localize("UTC")).floor("s")
        if lf > placed_at:
            base_at, base = lf, "first read after placement"
    if not have:
        base = "no traded ladder"
    f = qf.fill("lay", price, stake, base_at, bk, traded if have else None, until=until, through_min=through_min)
    taken, tval = _taken(price, stake, lays)
    taken = round(taken, 2)
    matched = round(min(f.matched, stake), 2)
    rested = max(matched - taken, 0.0)
    avg = (tval + rested * price) / matched if matched > EPS else None
    first = last = None
    if taken > EPS:
        first = last = placed_at
    if rested > EPS:
        if f.full_at is not None:
            last = pd.Timestamp(f.full_at)
        t = pd.to_datetime(traded.polled_utc, utc=True, format="ISO8601").dt.floor("s")
        u = pd.Timestamp(until) if until is not None else None
        u = (u if u is None or u.tzinfo else u.tz_localize("UTC"))
        times = sorted(set(t[(t > base_at) & ((t <= u) if u is not None else True)]))
        prev = taken
        for ti in times:
            m = qf.fill("lay", price, stake, base_at, bk, traded, until=ti, through_min=through_min).matched
            if m > prev + EPS:
                first = first if first is not None else ti
                last, prev = ti, m
            if m >= matched - EPS:
                break
    return {"matched": matched, "taken": taken, "avg_price": round(avg, 4) if avg else None, "queue_ahead": q,
            "queue_assumed": assumed, "filled_share": round(matched / stake, 4) if stake > 0 else 0.0,
            "first_fill_at": first, "last_fill_at": last,
            "full_at": pd.Timestamp(f.full_at) if f.full_at is not None and matched >= stake - EPS else None,
            "ladder_base": base}


def green_pnl(matched: float, avg_price: float, close_price: float | None) -> float | None:
    """A matched lay greened up by a back at ``close_price``: the same whatever wins, matched x (1 - avg / close)."""
    if not matched or not avg_price or not close_price or close_price <= 1:
        return None
    return round(matched * (1 - avg_price / close_price), 2)


# --------------------------------------------------------------------------------------------------------------------
# The ledger's lays, replayed
# --------------------------------------------------------------------------------------------------------------------

def _prep_books(books: pd.DataFrame) -> pd.DataFrame:
    b = books.copy()
    if "polled" not in b:
        b["polled"] = b.polled_utc
    b["polled"] = pd.to_datetime(b.polled, utc=True, errors="coerce")
    b["market_id"] = b.market_id.astype(str)
    b["selection_id"] = pd.to_numeric(b.selection_id, errors="coerce")
    for c in SIZE_COLS + SP_COLS + ["last_traded"]:
        b[c] = pd.to_numeric(b[c], errors="coerce") if c in b else np.nan
    return b.dropna(subset=["polled", "selection_id"]).sort_values("polled")


def _prep_ladder(traded: pd.DataFrame | None) -> pd.DataFrame | None:
    if traded is None or not len(traded):
        return None
    t = traded.copy()
    t["market_id"] = t.market_id.astype(str)
    t["selection_id"] = pd.to_numeric(t.selection_id, errors="coerce")
    t["price"] = pd.to_numeric(t.price, errors="coerce")
    t["traded"] = pd.to_numeric(t.traded, errors="coerce")
    t["_polled"] = pd.to_datetime(t.polled_utc, utc=True, format="ISO8601", errors="coerce")
    return t.dropna(subset=["_polled", "price", "traded"])


def sp_close(g: pd.DataFrame, bsp: float | None, off) -> tuple[float | None, str]:
    """The SP a matched lay is greened at: the BSP given or held in the books, else the last projected SP before the
    off (near, then far)."""
    if bsp is not None and pd.notna(bsp) and bsp > 1:
        return float(bsp), "bsp"
    a = g.sp_actual[g.sp_actual > 1]
    if len(a):
        return float(a.iloc[-1]), "bsp"
    pre = g[g.polled <= off]
    for c in ("sp_near", "sp_far"):
        v = pre[c][pre[c] > 1]
        if len(v):
            return float(v.iloc[-1]), c
    return None, ""


def replay_queue(rows: pd.DataFrame, books: pd.DataFrame, traded: pd.DataFrame | None, cfg: dict | None = None,
                 bsp: dict | None = None, unknown: str = "deepest", through_min: float = 0.0) -> pd.DataFrame:
    """``greyhound.early_lay.replay`` with queue-aware fills: the same rows (ledger lays: time_utc, seconds_to_off,
    market_id, selection_id, price, race_time, track, trap, f_bsp, mid) and the same columns out (``matched`` True
    where any part matched, ``matched_at`` the average price matched, ``pnl`` greened at back1), plus queue_ahead,
    filled_share and the rest of ``lay_fill``'s fields, ``matched_stake``, and the close at the SP (``sp``,
    ``sp_source``, ``pnl_sp``). ``books``: the runners' books with the three levels and sizes a side (``polled`` or
    ``polled_utc``); ``traded``: the live-key traded ladder; ``bsp``: {(market_id, selection_id): BSP}."""
    cfg = cfg or el.CONFIG
    b = _prep_books(books)
    lad = _prep_ladder(traded)
    lad_by = {k: g.drop(columns="_polled") for k, g in lad.groupby(["market_id", "selection_id"])} if lad is not None \
        else {}
    lad_from = lad.groupby("market_id")._polled.min().to_dict() if lad is not None else {}
    bsp = bsp or {}
    out = []
    for r in rows.itertuples(index=False):
        t = pd.Timestamp(r.time_utc)
        t = t if t.tzinfo else t.tz_localize("UTC")
        off = t + pd.Timedelta(seconds=float(r.seconds_to_off))
        green_t = off - pd.Timedelta(seconds=cfg["green_seconds"])
        price = float(r.price)
        stake = el.stake_for(price, cfg)
        mid, sid = str(r.market_id), float(r.selection_id)
        row = {"race_time": r.race_time, "track": r.track, "trap": r.trap, "lay_price": price, "f_bsp": r.f_bsp,
               "mid": r.mid, "stake": stake, "matched": None, "matched_at": None, "greened_at": None, "pnl": None,
               "note": "", "matched_stake": 0.0, "queue_ahead": None, "queue_assumed": None, "filled_share": 0.0,
               "taken": 0.0, "first_fill_at": None, "last_fill_at": None, "full_at": None, "ladder_base": None,
               "sp": None, "sp_source": "", "pnl_sp": None}
        g = b[(b.market_id == mid) & (b.selection_id == sid)]
        before = g[g.polled <= t]
        after = g[(g.polled > t) & (g.polled <= off)]
        if not len(before) and not len(after):
            out.append({**row, "note": "no books for this dog"})
            continue
        now = before.iloc[-1] if len(before) else after.iloc[0]
        f = lay_fill(price, stake, t, book_levels(now), lad_by.get((mid, sid)), until=green_t,
                     ladder_from=lad_from.get(mid), unknown=unknown, through_min=through_min)
        row.update({k: f[k] for k in ("queue_ahead", "queue_assumed", "filled_share", "taken", "first_fill_at",
                                      "last_fill_at", "full_at", "ladder_base")})
        row["matched_stake"] = f["matched"]
        sp, src = sp_close(g, bsp.get((mid, sid)), off)
        row.update({"sp": sp, "sp_source": src})
        if f["matched"] <= 0:
            out.append({**row, "matched": False, "note": "resting: the queue at our price never cleared"
                        if f["queue_ahead"] is not None else "resting: deeper than the book, never traded through"})
            continue
        how = "taken at once" if f["taken"] >= f["matched"] - EPS else (
            "partly taken, the rest through the queue" if f["taken"] > 0 else "rested, then through the queue")
        if f["matched"] < stake - EPS:
            how += f" (partial, {f['filled_share']:.0%})"
        row.update({"matched": True, "matched_at": f["avg_price"],
                    "pnl_sp": green_pnl(f["matched"], f["avg_price"], sp)})
        last = g[(g.polled <= green_t) & (g.polled > t) & (g.back1 > 1)]
        if not len(last):
            out.append({**row, "note": f"{how}; no book to green up in"})
            continue
        B = float(last.back1.iloc[-1])
        out.append({**row, "greened_at": B, "pnl": green_pnl(f["matched"], f["avg_price"], B), "note": how})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------------------------------
# Optimistic against queue-aware
# --------------------------------------------------------------------------------------------------------------------

def _totals(d: pd.DataFrame, name: str, matched_stake: pd.Series) -> dict:
    m = d.matched == True                                                # noqa: E712
    out = {"fill": name, "lays": int(len(d)), "no_books": int((d.note == "no books for this dog").sum()) if len(d) else 0,
           "matched": int(m.sum()), "matched_stake": round(float(matched_stake[m].sum()), 2),
           "greened": int(pd.to_numeric(d.pnl, errors="coerce").notna().sum()) if len(d) else 0,
           "pnl": round(float(pd.to_numeric(d.pnl, errors="coerce").fillna(0).sum()), 2) if len(d) else 0.0}
    if "pnl_sp" in d:
        out["pnl_sp"] = round(float(pd.to_numeric(d.pnl_sp, errors="coerce").fillna(0).sum()), 2)
    return out


def compare(rows: pd.DataFrame, books: pd.DataFrame, traded: pd.DataFrame | None, cfg: dict | None = None,
            bsp: dict | None = None, unknown: str = "deepest") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ledger's lays replayed both ways on the same books: (side by side, row by row; totals: lays, matched count,
    matched stake, P&L at the green-up, and the queue-aware fill's P&L at the SP)."""
    cfg = cfg or el.CONFIG
    b = _prep_books(books)
    opt = el.replay(rows, b[["market_id", "selection_id", "polled", "back1", "lay1", "last_traded"]], cfg)
    que = replay_queue(rows, b, traded, cfg, bsp=bsp, unknown=unknown)
    if not len(opt):
        return pd.DataFrame(), pd.DataFrame()
    keys = ["race_time", "track", "trap", "lay_price", "f_bsp", "mid", "stake"]
    side = pd.concat([opt[keys].reset_index(drop=True),
                      opt[["matched", "matched_at", "greened_at", "pnl", "note"]].add_prefix("opt_").reset_index(drop=True),
                      que.drop(columns=keys).add_prefix("q_").reset_index(drop=True)], axis=1)
    tot = pd.DataFrame([_totals(opt, "optimistic (early_lay.replay)", opt.stake),
                        _totals(que, f"queue-aware ({unknown})", que.matched_stake)])
    return side, tot


def load_day(day: date) -> dict:
    """A day's live-key record for ``compare``: the pre-off books with their levels, the traded ladder, the BSPs
    (betfair_recorder ``_read_day_csv``: the server's copy, else S3's)."""
    from betfair_recorder import _read_day_csv
    from greyhound import tradeout as to
    bk = _read_day_csv(day, "books_greyhound_live.csv")
    mk = _read_day_csv(day, "markets_greyhound_live.csv")
    lad = _read_day_csv(day, "traded_greyhound_live.csv")
    if bk is None or mk is None or not len(bk):
        return {}
    sp = bk[bk.source.astype(str).isin(["final", "bsp", "live"])].assign(
        sp_actual=lambda d: pd.to_numeric(d.sp_actual, errors="coerce"),
        selection_id=lambda d: pd.to_numeric(d.selection_id, errors="coerce"), market_id=lambda d: d.market_id.astype(str))
    sp = sp[sp.sp_actual > 1].groupby(["market_id", "selection_id"]).sp_actual.max().to_dict()
    return {"books": to.pre_off_books(bk, mk), "traded": lad, "bsp": sp}
