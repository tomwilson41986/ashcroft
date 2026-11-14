"""The live early lay against its replay, order by order (the owner, 14 Nov 2026: "may I build the live-against-replay
fill reconciliation - yes"). Read-only research: nothing here reaches Betfair or places an order.

Why: the live early lay (greyhound/early_lay.py: in each race's first traded book it rests LIMIT lays at 80% of the
forecast SP on the dogs the forecast says will drift, sized by liability, persistence LAPSE, greened up before the off,
orders under the ``el-`` reference) lost on every live day so far, while the replay of the same rule on recorded books
makes about +1.1% of the money at risk (research/greyhound/min_lay_price.py on greyhound/laytest.py's procedure).
The question here: does the replay's fill model credit fills, prices and green-ups the live market did not give, and by
how much? Registered before it ran: ``greyhound-fill-recon-1114`` (reports/research_ledger.jsonl).

What is replayed: EXACTLY the live orders (each ledger ``lay`` row: market, selection, the time it went in, its limit
price and its stake), never the rule's own choice of orders. Each through every fill model the day's records allow:

- ``traded_retimed``: the forward test's own fill on the forward test's own books (the market recorder's, re-timed:
  greyhound/tradeout.py ``pre_off_books(retimed=True)``): matched at once where the best lay price on offer is at or
  under the limit, else in full at the limit once a later book's last traded price reaches it (research/greyhound/
  limit_study.py ``fills`` 'traded'). The model the replayed lead rests on.
- ``through_retimed``: the same books, matched once a later book offers the price (best lay price at or under the
  limit: limit_study ``fills`` 'through').
- ``traded_live`` / ``through_live``: the same two rules on the live-key books (greyhound/book_record.py, 5 s near
  the off, not re-timed).
- ``queue_book``: queue-aware on the live-key books and the traded ladder (research/greyhound/queue_fill.py
  ``lay_fill``: what the book offers is taken, the rest waits behind the money shown at its price, filled by money
  traded at the price after that queue, or in full by money traded beyond it).
- ``queue_stream``: queue-aware on Betfair's Exchange Stream (research/greyhound/stream_replay.py ``replay``: the
  full ladder message by message, the queue at placement, volume x1.0), lapsed 5 s before the scheduled off as the
  live trader cancels.

Nothing fills after the green-up time (``CONFIG['green_seconds']`` before the scheduled off). A replayed fill is
greened up at the best back price of the last book from its fill to the green-up time (the stream: the best back price
at the close), else at the SP (noted). The live order's figures come from its ledger: the stake matched and its average
price (the close rows' ``lay_matched``, ``lay_price``), the green-up backs, and the P&L the trader locked (the ``green``
row); a lay closed at the SP or held is settled by the SP and the result where the books hold them. Betfair's settled
record (the ledger's ``reconcile.json``, listClearedOrders) is reported beside it per day.

Per order: the live matched share against each replay's, the average matched price, the time from placement to fill
(live: taken at once or rested; Betfair's fill time is not in the ledger), the green-up price, the P&L, the SP. Per
day and pooled, per model: orders, fill rates, the fills only the replay had (phantom) with their replay P&L, the
fills only live had (missed), and the P&L gap (replay less live, before commission) cut exactly into

  (i) phantom fills: the replay's P&L on stake live never matched (whole orders, and the excess of a partial);
  (ii) missed fills: less live's P&L on stake the replay never matched;
  (iii) price on common fills: the common stake at the replay's average price against live's, at live's green-up;
  (iv) green-up on common fills: the common stake at the replay's price, the replay's green-up against live's;

which sum to the gap (orders whose P&L is unknown on either side are counted and left out of both); by lay price band
and by minutes to the off at placement; and the adverse selection: mean log(SP / lay price) for the live-matched, the
live-unmatched and the replay-only (phantom) orders.

    python -m research.greyhound.fill_recon --from 2026-11-03 --to 2026-11-13 --out out/fill_recon
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FIRST_LIVE_DAY = date(2026, 11, 3)
LEDGER_PREFIX = "greyhound/early_lay/live"           # sources/greyhound/early_lay/live/<day>/ledger.csv in S3
EPS = 1e-9
BANDS = ((0.0, 3.0, "under 3"), (3.0, 5.0, "3-5"), (5.0, 8.0, "5-8"), (8.0, math.inf, "8 and over"))
MINUTES = ((0.0, 2.0, "under 2"), (2.0, 5.0, "2-5"), (5.0, 10.0, "5-10"), (10.0, 20.0, "10-20"),
           (20.0, math.inf, "20 and over"))
MODELS = ("traded_retimed", "through_retimed", "traded_live", "through_live", "queue_book", "queue_stream")
PRIMARY_MODEL = "traded_retimed"
#: the decision rule, fixed before the run (greyhound-fill-recon-1114): the replay is trusted for the early lay's
#: family only if, pooled over the live days, its P&L on the live orders is within GBP X of live's and its fill rate
#: (stake matched over stake asked) within Y percentage points, on at least MIN_COVERAGE of the live orders
X_GBP = 50.0
Y_POINTS = 5.0
MIN_COVERAGE = 0.9
BOOK_COLS = ["market_id", "selection_id", "polled", "back1", "lay1", "last_traded"]


def _num(v):
    return pd.to_numeric(v, errors="coerce")


def band(price: float, cuts=BANDS) -> str:
    for lo, hi, name in cuts:
        if lo <= price < hi:
            return name
    return "unknown"


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t if t.tzinfo else t.tz_localize("UTC")


# --------------------------------------------------------------------------------------------------------------------
# The live orders, from the ledger
# --------------------------------------------------------------------------------------------------------------------

def _live_pnl(close: pd.DataFrame, S: float, a: float, sp: float | None, won: int | None) -> tuple[float | None, str]:
    """A live lay's P&L from its close rows: the P&L the trader locked by the green-up; else (closed at the SP or held)
    settled by the SP and the result: the lay of ``S`` at ``a``, any green-up backs matched, and a back at the SP."""
    g = close[close.action == "green"]
    locked = _num(g.locked_pnl).dropna() if len(g) else pd.Series(dtype=float)
    if len(locked):
        return float(locked.sum()), "greened"
    act = str(close.action.iloc[-1]) if len(close) else ""
    if act not in ("sp_close", "hold"):
        return None, "matched, no close in the ledger"
    gb = close[close.action.isin(["green", "green_part"])]
    bs, bp = _num(gb.matched).fillna(0.0).to_numpy(), _num(gb.avg_price).fillna(0.0).to_numpy()
    s_sp = float(_num(close[close.action == "sp_close"]["size"]).fillna(0.0).sum())
    if won is None or (act == "sp_close" and s_sp > 0 and won == 1 and not (sp and sp > 1)):
        return None, f"{act}: result or SP not in the records"
    if won == 1:
        v = -S * (a - 1.0) + float((bs * (bp - 1.0)).sum()) + (s_sp * (sp - 1.0) if s_sp > 0 else 0.0)
    else:
        v = S - float(bs.sum()) - s_sp
    return v, act


def live_orders(led: pd.DataFrame, final: dict | None = None, offs: dict | None = None,
                green_seconds: float = 5.0) -> pd.DataFrame:
    """Each live lay (ledger ``lay`` rows with a price) with what live gave it: stake matched, average price, the
    green-up price, the P&L, how it was closed; its scheduled off and green-up time. ``final``: (market, selection) ->
    (SP, won); ``offs``: market -> scheduled off, where the ledger lacks ``seconds_to_off``."""
    final, offs = final or {}, offs or {}
    led = led.copy()
    for c in ("price", "size", "matched", "avg_price", "lay_matched", "lay_price", "locked_pnl", "seconds_to_off",
              "f_bsp", "mid"):
        led[c] = _num(led[c]) if c in led else np.nan
    for c in ("action", "track", "race_time", "trap", "status", "error"):
        if c not in led:
            led[c] = ""
    led["market_id"] = led.market_id.astype(str)
    led["selection_id"] = _num(led.selection_id)
    lays = led[(led.action == "lay") & (led.price > 1)].sort_values("time_utc")
    lays = lays.drop_duplicates(["market_id", "selection_id"], keep="first")
    closes = led[led.action.isin(["green", "green_part", "sp_close", "hold"])]
    by = {k: g for k, g in closes.groupby(["market_id", "selection_id"])}
    rows = []
    for r in lays.itertuples(index=False):
        mid, sid = str(r.market_id), float(r.selection_id)
        t = _utc(r.time_utc)
        if pd.notna(r.seconds_to_off):
            off = t + pd.Timedelta(seconds=float(r.seconds_to_off))
        elif mid in offs and pd.notna(offs[mid]):
            off = _utc(offs[mid])
        else:
            off = pd.NaT
        sp, won = final.get((mid, sid), (None, None))
        c = by.get((mid, sid), closes.iloc[:0])
        S = max(float(_num(c.lay_matched).max()) if len(c) and _num(c.lay_matched).notna().any() else 0.0,
                float(r.matched) if pd.notna(r.matched) else 0.0)
        a = float(_num(c.lay_price).dropna().iloc[-1]) if len(c) and _num(c.lay_price).notna().any() else (
            float(r.avg_price) if pd.notna(r.avg_price) and r.avg_price > 1 else float(r.price))
        gb = c[c.action.isin(["green", "green_part"])]
        bs = _num(gb.matched).fillna(0.0)
        green = float((_num(gb.avg_price).fillna(0.0) * bs).sum() / bs.sum()) if bs.sum() > 0 else None
        if S > EPS:
            pnl, how = _live_pnl(c, S, a, sp, won)
        else:
            pnl, how = 0.0, "unmatched"
        rows.append({"market_id": mid, "selection_id": sid, "track": r.track, "race_time": r.race_time,
                     "trap": r.trap, "placed_utc": t, "off_utc": off,
                     "green_utc": off - pd.Timedelta(seconds=green_seconds) if pd.notna(off) else pd.NaT,
                     "minutes_to_off": (off - t).total_seconds() / 60.0 if pd.notna(off) else np.nan,
                     "price": float(r.price), "size": float(r.size),
                     "f_bsp": r.f_bsp, "mid": r.mid, "status": r.status,
                     "live_matched": round(S, 2), "live_avg": round(a, 4) if S > EPS else None,
                     "live_taken_at_once": bool(pd.notna(r.matched) and r.matched > EPS),
                     "live_green": round(green, 4) if green else None, "live_pnl": pnl, "live_close": how,
                     "sp": sp, "won": won})
    out = pd.DataFrame(rows)
    if len(out):
        out["live_share"] = (out.live_matched / out["size"]).clip(upper=1.0)
        out["band"] = out.price.map(band)
        out["minutes_band"] = out.minutes_to_off.map(lambda m: band(m, MINUTES) if pd.notna(m) else "unknown")
    return out


# --------------------------------------------------------------------------------------------------------------------
# The replays
# --------------------------------------------------------------------------------------------------------------------

def _green(g: pd.DataFrame | None, tf, green_t, S: float, a: float, sp) -> tuple[float | None, float | None, str]:
    """(green-up price, P&L, note) of a replayed fill of ``S`` at ``a`` made at ``tf``: the best back price of the last
    book from the fill to the green-up time, else the SP."""
    if g is not None and len(g):
        last = g[(g.polled >= tf) & (g.polled <= green_t) & (g.back1 > 1)]
        if len(last):
            B = float(last.back1.iloc[-1])
            return B, S * (1.0 - a / B), ""
    if sp is not None and pd.notna(sp) and sp > 1:
        return float(sp), S * (1.0 - a / float(sp)), "no book to green up in: greened at the SP"
    return None, None, "no book to green up in and no SP"


def book_fill(o, g: pd.DataFrame | None, rule: str) -> dict:
    """One live order on a runner's pre-off books (sorted by ``polled``) under ``rule``: 'traded' (a later book's last
    traded price at or under the limit) or 'through' (a later book's best lay price at or under it); either way taken
    at once where the book at placement already offers the price. Matched in full (the fill rules of limit_study)."""
    t, green_t, price, S = o.placed_utc, o.green_utc, float(o.price), float(o["size"])
    if g is None or not len(g) or pd.isna(green_t):
        return {"covered": False, "note": "no books for this dog"}
    before = g[g.polled <= t]
    after = g[(g.polled > t) & (g.polled <= green_t)]
    if not len(before) and not len(after):
        return {"covered": False, "note": "no books for this dog before the green-up"}
    now = before.iloc[-1] if len(before) else after.iloc[0]
    lay1 = float(now.lay1) if pd.notna(now.lay1) and now.lay1 > 1 else None
    if lay1 is not None and lay1 <= price + EPS:
        tf, a, how = t, lay1, "taken at once"
    else:
        col = after.last_traded if rule == "traded" else after.lay1
        hit = after[(col > 1) & (col <= price + EPS)]
        if not len(hit):
            return {"covered": True, "matched": 0.0, "note": "resting: never reached"}
        tf, a, how = hit.polled.iloc[0], price, f"rested, then {rule}"
    B, pnl, note = _green(g, tf, green_t, S, a, o.sp)
    return {"covered": True, "matched": S, "avg": a, "fill_secs": (tf - t).total_seconds(), "green": B, "pnl": pnl,
            "note": "; ".join(x for x in (how, note) if x)}


def queue_book_fill(o, g: pd.DataFrame | None, ladder: pd.DataFrame | None, ladder_from) -> dict:
    """One live order through the queue-aware fill on the live-key books and traded ladder (queue_fill.lay_fill)."""
    from research.greyhound import queue_fill as rq
    if g is None or not len(g) or ladder is None or not len(ladder) or pd.isna(o.green_utc):
        return {"covered": False, "note": "no live-key books or traded ladder for this dog"}
    t, green_t, price, S = o.placed_utc, o.green_utc, float(o.price), float(o["size"])
    before = g[g.polled <= t]
    after = g[(g.polled > t) & (g.polled <= green_t)]
    if not len(before) and not len(after):
        return {"covered": False, "note": "no books for this dog before the green-up"}
    now = before.iloc[-1] if len(before) else after.iloc[0]
    f = rq.lay_fill(price, S, t, rq.book_levels(now), ladder, until=green_t, ladder_from=ladder_from)
    if f["matched"] <= EPS:
        return {"covered": True, "matched": 0.0, "queue_ahead": f["queue_ahead"], "note": "resting: queue never cleared"}
    tf = f["first_fill_at"] if f["first_fill_at"] is not None else t
    B, pnl, note = _green(g, _utc(tf), green_t, f["matched"], f["avg_price"], o.sp)
    return {"covered": True, "matched": f["matched"], "avg": f["avg_price"], "fill_secs": (_utc(tf) - t).total_seconds(),
            "green": B, "pnl": pnl, "queue_ahead": f["queue_ahead"], "note": note}


def stream_fills(orders: pd.DataFrame, lines: list | None, green_seconds: float = 5.0) -> dict:
    """Every live order through the stream replay (queue-aware, volume x1.0, lapsed ``green_seconds`` before the
    scheduled off): (market, selection) -> the model's fields."""
    from research.greyhound import stream_replay as sr
    if not lines or not len(orders):
        return {}
    sims = [sr.SimOrder(market=o.market_id, sid=int(o.selection_id), side="LAY", price=float(o.price),
                        size=float(o["size"]), t=o.placed_utc.to_pydatetime(), tag=f"{o.market_id}|{int(o.selection_id)}")
            for _, o in orders.iterrows()]
    df = sr.replay(lines, sims, close_secs=green_seconds, vol_factor=1.0, comm=0.0)
    out = {}
    for r in df.itertuples(index=False):
        mid, sid = r.tag.split("|")
        key = (mid, float(sid))
        if not r.placed:
            out[key] = {"covered": False, "note": "no stream book for this dog"}
            continue
        if not r.matched or r.matched <= EPS:
            out[key] = {"covered": True, "matched": 0.0, "queue_ahead": r.queue_at_place, "note": "resting: never filled",
                        "sp": r.sp, "won": r.won}
            continue
        o = orders[(orders.market_id == mid) & (orders.selection_id == float(sid))].iloc[0]
        B, pnl, note = (r.close_price, r.pnl, "") if r.close_price else (None, None, "")
        if pnl is None or pd.isna(pnl):
            sp = r.sp if r.sp else o.sp
            B, pnl, note = _green(None, None, None, float(r.matched), float(r.avg), sp)
        ff = r.first_fill
        out[key] = {"covered": True, "matched": float(r.matched), "avg": float(r.avg),
                    "fill_secs": (_utc(ff) - o.placed_utc).total_seconds() if ff is not None and pd.notna(ff) else None,
                    "green": B, "pnl": pnl, "queue_ahead": r.queue_at_place, "note": note, "sp": r.sp, "won": r.won}
    return out


def _prep(books: pd.DataFrame | None) -> dict:
    if books is None or not len(books):
        return {}
    b = books.copy()
    b["market_id"] = b.market_id.astype(str)
    b["selection_id"] = _num(b.selection_id)
    b["polled"] = pd.to_datetime(b.polled, utc=True)
    for c in ("back1", "lay1", "last_traded"):
        b[c] = _num(b[c]) if c in b else np.nan
    for i in (1, 2, 3):
        for s in ("back", "lay"):
            for x in ("", "_size"):
                if f"{s}{i}{x}" in b:
                    b[f"{s}{i}{x}"] = _num(b[f"{s}{i}{x}"])
    return {k: g.sort_values("polled") for k, g in b.groupby(["market_id", "selection_id"])}


def replay_day(orders: pd.DataFrame, data: dict, green_seconds: float = 5.0) -> pd.DataFrame:
    """The day's live orders with every model's fill beside them (columns ``<model>_<field>``)."""
    if not len(orders):
        return orders
    rec, live = _prep(data.get("recorder_books")), _prep(data.get("live_books"))
    lad = data.get("traded")
    lad_by, lad_from = {}, {}
    if lad is not None and len(lad):
        from research.greyhound import queue_fill as rq
        lp = rq._prep_ladder(lad)
        lad_by = {k: g.drop(columns="_polled") for k, g in lp.groupby(["market_id", "selection_id"])}
        lad_from = lp.groupby("market_id")._polled.min().to_dict()
    stream = stream_fills(orders, data.get("stream"), green_seconds)
    res = {m: [] for m in MODELS}
    for _, o in orders.iterrows():
        k = (o.market_id, float(o.selection_id))
        res["traded_retimed"].append(book_fill(o, rec.get(k), "traded"))
        res["through_retimed"].append(book_fill(o, rec.get(k), "through"))
        res["traded_live"].append(book_fill(o, live.get(k), "traded"))
        res["through_live"].append(book_fill(o, live.get(k), "through"))
        res["queue_book"].append(queue_book_fill(o, live.get(k), lad_by.get(k), lad_from.get(o.market_id)))
        res["queue_stream"].append(stream.get(k, {"covered": False, "note": "no stream record for the day"
                                                  if not data.get("stream") else "no stream book for this dog"}))
    out = orders.reset_index(drop=True).copy()
    # the SP and the result where the books lacked them: the stream's settled definition
    for i, s in enumerate(res["queue_stream"]):
        if (out.at[i, "sp"] is None or pd.isna(out.at[i, "sp"])) and s.get("sp"):
            out.at[i, "sp"] = s["sp"]
        if (out.at[i, "won"] is None or pd.isna(out.at[i, "won"])) and s.get("won") is not None:
            out.at[i, "won"] = s["won"]
    for m in MODELS:
        f = pd.DataFrame(res[m])
        for c in ("covered", "matched", "avg", "fill_secs", "green", "pnl", "queue_ahead", "note"):
            out[f"{m}_{c}"] = f[c].to_numpy() if c in f else None
        out[f"{m}_covered"] = out[f"{m}_covered"].fillna(False).astype(bool)
        out[f"{m}_matched"] = _num(out[f"{m}_matched"]).fillna(0.0)
        out[f"{m}_share"] = (out[f"{m}_matched"] / out["size"]).clip(upper=1.0)
    return out


# --------------------------------------------------------------------------------------------------------------------
# The reconciliation: fill rates, phantom and missed fills, the P&L gap cut into its parts
# --------------------------------------------------------------------------------------------------------------------

def decompose(d: pd.DataFrame, m: str) -> pd.DataFrame:
    """Per order (the rows ``m`` covers): live P&L ``L``, replay P&L ``R`` and the gap R - L cut into phantom, missed,
    price and green-up parts that sum to it exactly. A lay's P&L per unit of stake is u = 1 - a x k (a: its average
    price; k = 1 / green-up price, read off the P&L for live lays closed at the SP or held). On the common stake
    C = min(S_r, S_l): price C (a_l - a_r) k_l, green-up C a_r (k_l - k_r); the replay's excess stake at its own u is
    phantom, live's excess at its own u is missed."""
    x = d[d[f"{m}_covered"]].copy()
    S_l, S_r = x.live_matched.to_numpy(float), x[f"{m}_matched"].to_numpy(float)
    a_l = _num(x.live_avg).to_numpy(float)
    a_r = _num(x[f"{m}_avg"]).to_numpy(float)
    L = np.where(S_l > EPS, _num(x.live_pnl).to_numpy(float), 0.0)
    R = np.where(S_r > EPS, _num(x[f"{m}_pnl"]).to_numpy(float), 0.0)
    known = np.isfinite(L) & np.isfinite(R)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_l = np.where(S_l > EPS, L / S_l, 0.0)
        u_r = np.where(S_r > EPS, R / S_r, 0.0)
        k_l = np.where(S_l > EPS, (1.0 - u_l) / a_l, 0.0)
        k_r = np.where(S_r > EPS, (1.0 - u_r) / a_r, 0.0)
    C = np.minimum(S_l, S_r)
    both = (S_l > EPS) & (S_r > EPS)
    phantom = np.where(S_r > S_l, (S_r - C) * u_r, 0.0)
    missed = -np.where(S_l > S_r, (S_l - C) * u_l, 0.0)
    price = np.where(both, C * (a_l - a_r) * k_l, 0.0)
    green = np.where(both, C * a_r * (k_l - k_r), 0.0)
    x["L"], x["R"], x["pnl_known"] = np.where(known, L, np.nan), np.where(known, R, np.nan), known
    for n, v in (("phantom", phantom), ("missed", missed), ("price", price), ("green", green)):
        x[f"gap_{n}"] = np.where(known, v, np.nan)
    x["gap"] = x.R - x.L
    x["kind"] = np.select([both, (S_r > EPS) & ~(S_l > EPS), (S_l > EPS) & ~(S_r > EPS)],
                          ["both", "phantom", "missed"], "neither")
    return x


def _logret(sp, price) -> pd.Series:
    sp, price = _num(sp), _num(price)
    return np.log(sp.where(sp > 1) / price)


def summarise(d: pd.DataFrame, m: str) -> dict:
    """One model against live over the rows ``d``: coverage, fill rates, phantom and missed fills, the P&L gap and its
    parts, the adverse selection."""
    n = len(d)
    x = decompose(d, m) if n else d
    out = {"orders": int(n), "covered": int(len(x)), "coverage": round(len(x) / n, 3) if n else None}
    if not len(x):
        return out
    k = x[x.pnl_known]
    asked = float(x["size"].sum())
    live_rate = float(x.live_matched.clip(upper=x["size"]).sum() / asked) if asked else None
    rep_rate = float(x[f"{m}_matched"].clip(upper=x["size"]).sum() / asked) if asked else None
    ph, mi = x[x.kind == "phantom"], x[x.kind == "missed"]
    out.update({
        "stake_asked": round(asked, 2),
        "live_fill_rate%": round(100 * live_rate, 2) if live_rate is not None else None,
        "replay_fill_rate%": round(100 * rep_rate, 2) if rep_rate is not None else None,
        "fill_gap_points": round(100 * (rep_rate - live_rate), 2) if live_rate is not None else None,
        "live_matched_orders": int((x.live_matched > EPS).sum()),
        "replay_matched_orders": int((x[f"{m}_matched"] > EPS).sum()),
        "both_matched": int((x.kind == "both").sum()),
        "phantom_fills": int(len(ph)),
        "phantom_replay_pnl": round(float(_num(ph.R).sum()), 2),
        "missed_fills": int(len(mi)),
        "missed_live_pnl": round(float(_num(mi.L).sum()), 2),
        "pnl_unknown": int((~x.pnl_known).sum()),
        "live_pnl": round(float(k.L.sum()), 2), "replay_pnl": round(float(k.R.sum()), 2),
        "gap": round(float(k.gap.sum()), 2),
        "gap_parts": {p: round(float(k[f"gap_{p}"].sum()), 2) for p in ("phantom", "missed", "price", "green")},
        "mean_fill_secs": round(float(_num(x[f"{m}_fill_secs"]).mean()), 1)
        if _num(x[f"{m}_fill_secs"]).notna().any() else None,
    })
    both = x[x.kind == "both"]
    if len(both):
        out["common_avg_price_live"] = round(float(_num(both.live_avg).mean()), 3)
        out["common_avg_price_replay"] = round(float(_num(both[f"{m}_avg"]).mean()), 3)
        g = both[_num(both.live_green).notna() & _num(both[f"{m}_green"]).notna()]
        if len(g):
            out["common_green_live"] = round(float(_num(g.live_green).mean()), 3)
            out["common_green_replay"] = round(float(_num(g[f"{m}_green"]).mean()), 3)
    lm, lu = x[x.live_matched > EPS], x[x.live_matched <= EPS]
    out["adverse_selection"] = {
        "live_matched": _mean_n(_logret(lm.sp, lm.price)),
        "live_unmatched": _mean_n(_logret(lu.sp, lu.price)),
        "replay_only": _mean_n(_logret(ph.sp, ph.price)),
        "replay_matched": _mean_n(_logret(x[x[f"{m}_matched"] > EPS].sp, x[x[f"{m}_matched"] > EPS].price))}
    return out


def _mean_n(s: pd.Series) -> dict:
    s = s[np.isfinite(s)] if len(s) else s
    return {"mean": round(float(s.mean()), 4) if len(s) else None, "n": int(len(s))}


def by(d: pd.DataFrame, m: str, col: str, cuts) -> dict:
    return {name: summarise(d[d[col] == name], m) for _, _, name in cuts if (d[col] == name).any()}


def verdict(pooled: dict, x_gbp: float = X_GBP, y_points: float = Y_POINTS, min_cov: float = MIN_COVERAGE) -> str:
    """The registered reading for one model, pooled over the live days."""
    if not pooled.get("orders"):
        return "not judged: no live orders"
    if (pooled.get("coverage") or 0) < min_cov:
        return f"not judged: the model covers {pooled.get('covered')} of {pooled['orders']} live orders"
    ok = abs(pooled["gap"]) <= x_gbp and abs(pooled["fill_gap_points"]) <= y_points
    return "TRUSTED" if ok else "NOT TRUSTED"


# --------------------------------------------------------------------------------------------------------------------
# The records of a day, and the run
# --------------------------------------------------------------------------------------------------------------------

def _final(*books) -> dict:
    """(market, selection) -> (SP, won) from the recorders' settled reads (``source`` final)."""
    out = {}
    for b in books:
        if b is None or not len(b) or "source" not in b:
            continue
        f = b[b.source.astype(str) == "final"]
        for r in f.itertuples(index=False):
            k = (str(r.market_id), float(_num(pd.Series([r.selection_id])).iloc[0]))
            sp = _num(pd.Series([getattr(r, "sp_actual", None)])).iloc[0]
            st = str(getattr(r, "runner_status", ""))
            won = 1 if st == "WINNER" else 0 if st in ("LOSER", "PLACED") else None
            old = out.get(k, (None, None))
            out[k] = (float(sp) if pd.notna(sp) and sp > 1 else old[0], won if won is not None else old[1])
    return out


def load_day(store, day: date, s3=None) -> dict:
    """A live day's records, each None where missing: the live ledger and Betfair's settled record (S3 through
    ``store``), the live-key books and traded ladder, the market recorder's books re-timed, the stream lines of the
    ledger's markets (betfair_recorder's day files and the stream parts, S3 read-only)."""
    from betfair_recorder import _read_day_csv
    from greyhound import tradeout as to
    out: dict = {"day": f"{day}"}
    raw = store.get(f"{LEDGER_PREFIX}/{day:%Y-%m-%d}/ledger.csv")
    out["ledger"] = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}) if raw else None
    rec = store.get(f"{LEDGER_PREFIX}/{day:%Y-%m-%d}/reconcile.json")
    out["reconcile"] = json.loads(rec) if rec else None
    if out["ledger"] is None:
        return out
    mids = set(out["ledger"].market_id.dropna().astype(str))
    offs, finals = {}, []
    for name, retimed, key in (("books_greyhound_live.csv", False, "live_books"),
                               ("books_greyhound.csv", True, "recorder_books")):
        bk = _read_day_csv(day, name, s3=s3)
        mk = _read_day_csv(day, name.replace("books", "markets"), s3=s3)
        out[key] = None
        if bk is None or mk is None or not len(bk):
            continue
        bk = bk[bk.market_id.astype(str).isin(mids)]
        finals.append(bk)
        mk1 = mk.drop_duplicates("market_id")
        offs.update(dict(zip(mk1.market_id.astype(str), mk1.market_start_utc)))
        pre = to.pre_off_books(bk, mk, retimed=retimed)
        out[key] = pre if len(pre) else None
    lad = _read_day_csv(day, "traded_greyhound_live.csv", s3=s3)
    out["traded"] = lad[lad.market_id.astype(str).isin(mids)] if lad is not None and len(lad) else None
    out["final"], out["offs"] = _final(*finals), offs
    out["stream"] = stream_lines(day, mids, s3=s3)
    return out


def stream_lines(day: date, mids: set, s3=None) -> list | None:
    """The day's recorded stream lines that name one of ``mids`` (s3://$CAPTURE_BUCKET/betfair_stream/greyhound/<day>/),
    None when the day has no stream record."""
    import gzip
    from betfair_recorder import bucket, s3_client
    from research.greyhound import stream_replay as sr
    try:
        s3 = s3 or s3_client()
        keys, tok = [], None
        while True:
            kw = {"Bucket": bucket(), "Prefix": f"{sr.PREFIX}/{day:%Y-%m-%d}/", **({"ContinuationToken": tok} if tok else {})}
            r = s3.list_objects_v2(**kw)
            keys += [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".jsonl.gz")]
            if not r.get("IsTruncated"):
                break
            tok = r["NextContinuationToken"]
    except Exception as exc:
        log.warning("%s: the stream not listed (%s)", day, exc)
        return None
    if not keys or not mids:
        return None
    pat = re.compile("|".join(re.escape(m) for m in sorted(mids)))
    out = []
    for k in sorted(keys):
        raw = s3.get_object(Bucket=bucket(), Key=k)["Body"].read()
        out += [ln for ln in gzip.decompress(raw).decode("utf-8", "replace").splitlines() if pat.search(ln)]
    return out or None


def sources_of(data: dict) -> dict:
    """Which recorded books the day has for the ledger's markets."""
    def n(df):
        return int(df.market_id.astype(str).nunique()) if df is not None and len(df) else 0
    return {"live_key_books_markets": n(data.get("live_books")), "recorder_books_markets": n(data.get("recorder_books")),
            "traded_ladder_markets": n(data.get("traded")), "stream_lines": len(data.get("stream") or []),
            "settled_reads": len(data.get("final") or {})}


def reconcile_day(data: dict, green_seconds: float | None = None) -> tuple[dict, pd.DataFrame]:
    """One live day: (its report, its orders with every model beside them)."""
    from greyhound import early_lay as el
    gs = el.CONFIG["green_seconds"] if green_seconds is None else green_seconds
    rep = {"day": data.get("day")}
    led = data.get("ledger")
    if led is None or not len(led):
        rep["note"] = "no data found: no live ledger"
        return rep, pd.DataFrame()
    o = live_orders(led, data.get("final"), data.get("offs"), gs)
    rep["sources"] = sources_of(data)
    have = [k for k, v in (("live-key books", data.get("live_books")), ("recorder books re-timed",
                                                                         data.get("recorder_books")),
                           ("traded ladder", data.get("traded")), ("stream", data.get("stream")))
            if v is not None and len(v)]
    rep["books"] = ", ".join(have) if have else "none"
    if not have:
        rep["note"] = "no recorded books for the day: not replayed"
    if not len(o):
        rep["note"] = "no lays in the ledger"
        return rep, o
    d = replay_day(o, data, gs)
    d.insert(0, "day", data.get("day"))
    rep["orders"] = int(len(d))
    rep["live"] = {"stake_asked": round(float(d["size"].sum()), 2), "matched": round(float(d.live_matched.sum()), 2),
                   "matched_orders": int((d.live_matched > EPS).sum()),
                   "taken_at_once": int(d.live_taken_at_once.sum()),
                   "pnl_ledger": round(float(_num(d.live_pnl).sum()), 2),
                   "pnl_unknown": int(_num(d.live_pnl).isna().sum())}
    r = data.get("reconcile")
    if r:
        rep["betfair"] = {k: r.get(k) for k in ("settled", "markets_laid", "betfair_profit_gross", "betfair_profit_net",
                                                "ledger_locked", "difference")}
    rep["models"] = {m: summarise(d, m) for m in MODELS}
    return rep, d


def report(days: list[dict], orders: pd.DataFrame) -> dict:
    """Every day and the pooled reading, by model, by price band and by minutes to the off."""
    out = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "primary_model": PRIMARY_MODEL,
           "rule": {"x_gbp": X_GBP, "y_points": Y_POINTS, "min_coverage": MIN_COVERAGE}, "days": days,
           "days_without_ledger": [d["day"] for d in days if "no live ledger" in str(d.get("note", ""))],
           "days_without_books": [d["day"] for d in days if d.get("books") == "none"]}
    if not len(orders):
        out["note"] = "no data found: no live order on any day"
        return out
    out["pooled"], out["by_band"], out["by_minutes"], out["verdicts"] = {}, {}, {}, {}
    for m in MODELS:
        p = summarise(orders, m)
        out["pooled"][m] = p
        out["verdicts"][m] = verdict(p)
        out["by_band"][m] = by(orders, m, "band", BANDS)
        out["by_minutes"][m] = by(orders, m, "minutes_band", MINUTES)
    out["verdict"] = out["verdicts"][PRIMARY_MODEL]
    return out


def run(store, first: date, last: date, loader=None) -> tuple[dict, pd.DataFrame]:
    loader = loader or (lambda d: load_day(store, d))
    days, parts = [], []
    d = first
    while d <= last:
        try:
            rep, o = reconcile_day(loader(d))
        except Exception as exc:                                   # one bad day: noted, the rest still read
            rep, o = {"day": f"{d}", "note": f"failed: {str(exc)[:200]}"}, pd.DataFrame()
        days.append(rep)
        if len(o):
            parts.append(o)
        log.info("%s: %s", d, rep.get("note") or f"{rep.get('orders')} orders; books: {rep.get('books')}")
        d += timedelta(days=1)
    orders = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return report(days, orders), orders


def markdown(r: dict) -> str:
    L = ["## Greyhound early lay: live fills against the replay, order by order", "",
         f"Registered reading (greyhound-fill-recon-1114): a model is TRUSTED when, pooled over the live days, its P&L on "
         f"the live orders is within GBP{X_GBP:g} of live's and its fill rate within {Y_POINTS:g} points, on at least "
         f"{MIN_COVERAGE:.0%} of the live orders. The forward test's fill ({PRIMARY_MODEL}): **{r.get('verdict', 'not judged')}**. "
         "P&L before commission; the gap is the replay less live.", "", "### Days", "",
         "| Day | Orders | Books | Live matched | Live P&L (ledger) | Betfair gross | Note |", "|---|---|---|---|---|---|---|"]
    for d in r["days"]:
        lv, bf = d.get("live") or {}, d.get("betfair") or {}
        L.append(f"| {d['day']} | {d.get('orders', '')} | {d.get('books', '')} | {lv.get('matched', '')} | "
                 f"{lv.get('pnl_ledger', '')} | {bf.get('betfair_profit_gross', '')} | {d.get('note', '')} |")
    if "pooled" not in r:
        return "\n".join(L + ["", r.get("note", "")]) + "\n"
    cols = ["covered", "live_fill_rate%", "replay_fill_rate%", "fill_gap_points", "phantom_fills", "phantom_replay_pnl",
            "missed_fills", "missed_live_pnl", "live_pnl", "replay_pnl", "gap"]
    L += ["", "### Pooled, by model", "", "| Model | Verdict | " + " | ".join(cols) +
          " | phantom | missed | price | green-up |", "|---|---|" + "---|" * (len(cols) + 4)]
    for m, p in r["pooled"].items():
        gp = p.get("gap_parts") or {}
        L.append(f"| {m} | {r['verdicts'][m]} | " + " | ".join(str(p.get(c, "")) for c in cols) + " | " +
                 " | ".join(str(gp.get(k, "")) for k in ("phantom", "missed", "price", "green")) + " |")
    L += ["", "### Adverse selection: mean log(SP / lay price)", "",
          "| Model | Live matched | Live unmatched | Replay only (phantom) | Replay matched |", "|---|---|---|---|---|"]
    for m, p in r["pooled"].items():
        a = p.get("adverse_selection") or {}
        L.append(f"| {m} | " + " | ".join(f"{(a.get(k) or {}).get('mean')} (n {(a.get(k) or {}).get('n')})"
                                         for k in ("live_matched", "live_unmatched", "replay_only", "replay_matched")) + " |")
    for title, key in (("lay price band", "by_band"), ("minutes to the off at placement", "by_minutes")):
        L += ["", f"### By {title}", "", "| Model | Cut | Orders | Live fill% | Replay fill% | Phantom | Missed | "
              "Gap | phantom | missed | price | green-up |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, cuts in r[key].items():
            for name, p in cuts.items():
                gp = p.get("gap_parts") or {}
                L.append(f"| {m} | {name} | {p.get('covered', 0)} | {p.get('live_fill_rate%', '')} | "
                         f"{p.get('replay_fill_rate%', '')} | {p.get('phantom_fills', '')} | {p.get('missed_fills', '')} | "
                         f"{p.get('gap', '')} | " + " | ".join(str(gp.get(k, "")) for k in
                                                               ("phantom", "missed", "price", "green")) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="first", default=None)
    ap.add_argument("--to", dest="last", default=None)
    ap.add_argument("--out", default="out/fill_recon")
    a = ap.parse_args(argv)
    first = date.fromisoformat(a.first) if a.first else FIRST_LIVE_DAY
    last = date.fromisoformat(a.last) if a.last else datetime.now(timezone.utc).date() - timedelta(days=1)
    r, orders = run(Store(), first, last)
    r.update({"from": f"{first}", "to": f"{last}"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(r, indent=1, default=str))
    Path(a.out + ".md").write_text(markdown(r))
    if len(orders):
        orders.to_csv(a.out + "_orders.csv", index=False)
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
