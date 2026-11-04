"""The profit simulator on the live books (the owner, 3 Nov 2026: optimise the profit of each race; "Yes" to building
the simulator first). Every decision is judged by what it would have made, on the books the live trader reads.

- **The data:** a day's live-key record (greyhound/book_record.py): the books before the off (every 60 s from 15
  minutes out, every 5 s in the last 5), the money traded at each price (``traded_greyhound_live.csv``), and each
  race's BSP and winner from the settled reads.
- **Fills:** a resting order is taken where the book offers its price, then matched only as money traded at its price
  passes the queue ahead of it, or in full once money trades beyond it (greyhound/queue_fill.py).
- **The exit:** a matched lay is greened up ``green_seconds`` before the off at the best back prices of the last book
  then, down the best three (value-weighted), the result the same whatever wins. Where no book is left, it is closed
  at the SP and settled by the result. 2% commission on each race's net win.
- **Calibration before use:** ``replay`` takes the orders the live trader actually sent (the early lay's ledger) and
  simulates them, matched amount and P&L beside the real ones. A simulator that cannot reproduce the live results
  does not judge strategies.
- **Strategies:** ``early_lay`` runs the live rule (greyhound/early_lay.py ``decide``) at each runner's first live-key
  book with a trade, in variants, race by race.

    python -m research.greyhound.profit_sim --date 2026-11-03 --out out/greyhound_profit_sim
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound import queue_fill as qf

log = logging.getLogger(__name__)

COMMISSION = 0.02
GREEN_SECONDS = 5.0
DEPTH = 3


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------------------------------
# The day's data
# --------------------------------------------------------------------------------------------------------------------

def load_day(day: date) -> dict:
    """The live-key books before the off, the traded ladder, the offs, and each runner's BSP and result."""
    from betfair_recorder import _read_day_csv
    books = _read_day_csv(day, "books_greyhound_live.csv")
    markets = _read_day_csv(day, "markets_greyhound_live.csv")
    ladder = _read_day_csv(day, "traded_greyhound_live.csv")
    if books is None or markets is None:
        return {}
    return prepare(books, markets, ladder)


def prepare(books: pd.DataFrame, markets: pd.DataFrame, ladder: pd.DataFrame | None) -> dict:
    b = books.copy()
    b["market_id"] = b.market_id.astype(str)
    b["selection_id"] = pd.to_numeric(b.selection_id, errors="coerce")
    b["polled"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
    for c in ("back1", "back1_size", "back2", "back2_size", "back3", "back3_size", "lay1", "lay1_size", "lay2",
              "lay2_size", "lay3", "lay3_size", "last_traded", "sp_actual", "runner_total_matched",
              "market_total_matched"):
        if c in b:
            b[c] = pd.to_numeric(b[c], errors="coerce")
    offs = markets.drop_duplicates("market_id").set_index(markets.drop_duplicates("market_id").market_id.astype(str))
    off = pd.to_datetime(offs.market_start_utc, utc=True, errors="coerce")
    b["off"] = b.market_id.map(off)
    src = b.source.astype(str)
    final = b[(src == "final") & (b.status.astype(str) == "CLOSED")]      # read before it settled: not final
    res = final.groupby(["market_id", "selection_id"]).runner_status.last().rename("status").reset_index()
    sp = b[src.isin(["final", "bsp", "live"])].groupby(["market_id", "selection_id"]).sp_actual.max().rename("bsp")
    res = res.join(sp, on=["market_id", "selection_id"])
    res["won"] = (res.status.astype(str) == "WINNER").astype(int)
    pre = b[(src != "final") & (b.status.astype(str) == "OPEN") & (b.polled < b.off)]
    if "inplay" in pre:
        pre = pre[pre.inplay.astype(str).isin(["0", "0.0", "False", "false"])]
    lad = None
    if ladder is not None and len(ladder):
        lad = ladder.copy()
        lad["market_id"] = lad.market_id.astype(str)
        lad["selection_id"] = pd.to_numeric(lad.selection_id, errors="coerce")
        lad["price"] = pd.to_numeric(lad.price, errors="coerce")
        lad["traded"] = pd.to_numeric(lad.traded, errors="coerce")
    return {"pre": pre.sort_values("polled"), "result": res, "off": off, "ladder": lad}


def _book_at(pre: pd.DataFrame, mid: str, sid: float, t) -> dict | None:
    g = pre[(pre.market_id == mid) & (pre.selection_id == sid) & (pre.polled <= t)]
    if not len(g):
        return None
    r = g.iloc[-1]
    back = [(float(r[f"back{i}"]), float(r[f"back{i}_size"])) for i in (1, 2, 3)
            if pd.notna(r.get(f"back{i}")) and r.get(f"back{i}") > 1]
    lay = [(float(r[f"lay{i}"]), float(r[f"lay{i}_size"])) for i in (1, 2, 3)
           if pd.notna(r.get(f"lay{i}")) and r.get(f"lay{i}") > 1]
    return {"back": back, "lay": lay, "polled": r.polled}


# --------------------------------------------------------------------------------------------------------------------
# Fills, the exit, the P&L
# --------------------------------------------------------------------------------------------------------------------

def green_price(book: dict | None, value: float, side: str = "back") -> float | None:
    """The average price an order worth ``value`` (stake x price) gets down the best ``DEPTH`` prices it can take
    (``side`` back: the back prices; lay: the lay prices), value-weighted (each level gives its size at its price),
    the last level's price for whatever is left over."""
    if not book or not book.get(side):
        return None
    left, stake, got = value, 0.0, 0.0
    levels = book[side][:DEPTH]
    for p, s in levels:
        take = min(left, p * s)
        stake += take / p
        got += take
        left -= take
        if left <= 1e-9:
            break
    if left > 1e-9:                                  # more than the shown depth: the rest at the last price shown
        stake += left / levels[-1][0]
        got += left
    return got / stake if stake > 0 else None


def lay_pnl(stake: float, price: float, exit_price: float | None, bsp: float | None, won: int) -> tuple[float, str]:
    """A matched lay of ``stake`` at ``price``: greened up at ``exit_price`` (the same whatever wins), else closed at
    the SP and settled by the result, else held to the result."""
    if exit_price and exit_price > 1:
        return stake * (1 - price / exit_price), "green"
    if bsp and bsp > 1:
        return (stake * (bsp - price) if won else 0.0), "sp_close"
    return (-stake * (price - 1) if won else stake), "held"


def simulate_lay(day: dict, mid: str, sid: float, t_place, price: float, size: float,
                 green_seconds: float = GREEN_SECONDS) -> dict:
    """One lay order through the simulator: its fill (queue-aware) to the green-up moment, its exit and P&L."""
    pre, lad = day["pre"], day["ladder"]
    off = day["off"].get(mid)
    t_place = pd.Timestamp(t_place)
    out = {"market_id": mid, "selection_id": sid, "placed": t_place, "price": price, "size": size}
    if off is None or pd.isna(off):
        return {**out, "note": "no off time"}
    t_exit = off - pd.Timedelta(seconds=green_seconds)
    book = _book_at(pre, mid, sid, t_place)
    if book is None:
        return {**out, "note": "no book when placed"}
    tr = lad[(lad.market_id == mid) & (lad.selection_id == sid)] if lad is not None else None
    f = qf.fill("lay", price, size, t_place, book, tr, until=t_exit)
    matched = f.matched
    res = day["result"]
    rr = res[(res.market_id == mid) & (res.selection_id == sid)]
    bsp = float(rr.bsp.iloc[0]) if len(rr) and pd.notna(rr.bsp.iloc[0]) else None
    won = int(rr.won.iloc[0]) if len(rr) else 0
    out.update({"matched": matched, "taken": f.taken, "queue_ahead": f.queue_ahead, "bsp": bsp, "won": won,
                "ladder": tr is not None and len(tr) > 0})
    if matched <= 0:
        return {**out, "pnl": 0.0, "how": "unmatched"}
    exit_book = _book_at(pre, mid, sid, t_exit)
    B = green_price(exit_book, matched * price)
    pnl, how = lay_pnl(matched, price, B, bsp, won)
    return {**out, "green_at": B, "pnl": round(pnl, 2), "how": how}


def race_net(rows: pd.DataFrame) -> pd.DataFrame:
    """Each race's P&L after 2% commission on its net win."""
    g = rows.groupby("market_id").pnl.sum().rename("gross").reset_index()
    g["net"] = np.where(g.gross > 0, g.gross * (1 - COMMISSION), g.gross)
    return g


# --------------------------------------------------------------------------------------------------------------------
# Calibration: the live trader's own orders, simulated beside what happened
# --------------------------------------------------------------------------------------------------------------------

def replay(store, day: date, data: dict, since: str | None = None) -> pd.DataFrame:
    """The early lay's live orders of ``day`` through the simulator: for each, the matched amount and P&L simulated
    and those the ledger records (``lay_matched`` and ``locked_pnl`` of its close; the SP closes and holds settled
    by the result)."""
    raw = store.get(f"greyhound/early_lay/live/{day:%Y-%m-%d}/ledger.csv")
    if not raw:
        return pd.DataFrame()
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    led["selection_id"] = pd.to_numeric(led.selection_id, errors="coerce")
    lays = led[led.action == "lay"].copy()
    lays["t"] = pd.to_datetime(lays.time_utc, utc=True)
    if since:
        lays = lays[lays.t >= pd.Timestamp(since)]
    closes = led[led.action.isin(["green", "green_part", "sp_close", "hold"])]
    out = []
    for r in lays.itertuples(index=False):
        s = simulate_lay(data, r.market_id, r.selection_id, r.t, float(r.price), float(r.size))
        c = closes[(closes.market_id == r.market_id) & (closes.selection_id == r.selection_id)]
        real_m = float(pd.to_numeric(c.lay_matched, errors="coerce").max()) if len(c) else 0.0
        real_m = 0.0 if np.isnan(real_m) else real_m
        locked = pd.to_numeric(c.locked_pnl, errors="coerce").dropna()
        real_pnl, real_how = (float(locked.sum()), "green") if len(locked) else (None, c.action.iloc[-1] if len(c) else
                                                                                "unmatched")
        if real_pnl is None and real_m > 0 and s.get("bsp"):
            real_pnl = lay_pnl(real_m, float(r.price), None, s["bsp"], s["won"])[0]
        out.append({**s, "race_time": r.race_time, "track": r.track, "real_matched": round(real_m, 2),
                    "real_pnl": round(real_pnl, 2) if real_pnl is not None else (0.0 if real_m == 0 else None),
                    "real_how": real_how})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------------------------------------

def first_traded_books(pre: pd.DataFrame) -> pd.DataFrame:
    """Each runner's first live-key book with a trade (the early lay's mark on the live feed)."""
    t = pre[pre.last_traded > 1]
    return t.sort_values("polled").drop_duplicates(["market_id", "selection_id"])


def early_lay(store, day: date, data: dict, cfg: dict, fit: dict) -> pd.DataFrame:
    """The live early-lay rule at each race's first traded books (every runner priced from its own), its lays
    simulated to the green-up. The race is decided once its whole field has traded, as the live trader does."""
    from greyhound import early_lay as el
    raw = store.get(f"greyhound/live/predictions/{day:%Y-%m-%d}.csv")
    if not raw:
        return pd.DataFrame()
    preds = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    preds["selection_id"] = pd.to_numeric(preds.selection_id, errors="coerce")
    preds = preds.dropna(subset=["selection_id"]).drop_duplicates(["market_id", "selection_id"], keep="last")
    pre, res = data["pre"], data["result"]
    ft = first_traded_books(pre)
    out = []
    for mid, g in ft.groupby("market_id"):
        if mid not in set(res.market_id):
            continue                                            # unsettled: no result to score
        field = pre[pre.market_id == mid].selection_id.dropna().unique()
        p = preds[preds.market_id == mid]
        if not len(p) or set(field) - set(p.selection_id) or len(g) < len(field):
            continue                                            # the field not all traded, or not all priced
        t_dec = g.polled.max()                                  # the whole field has traded by then
        if (data["off"].get(mid) - t_dec).total_seconds() < cfg["green_seconds"] + cfg["min_rest_seconds"]:
            continue
        first = {int(r.selection_id): {"market_id": mid, "polled": r.polled, "back1": r.back1,
                                       "back1_size": r.back1_size, "lay1": r.lay1, "lay1_size": r.lay1_size,
                                       "last_traded": r.last_traded,
                                       "traded": float(r.runner_total_matched or 0.0),
                                       "market_traded": float(r.market_total_matched or 0.0)}
                 for r in g.itertuples(index=False)}
        hist = pre[pre.market_id == mid]
        prev = {}
        for sid, fb in first.items():
            h = hist[(hist.selection_id == sid) & (hist.polled <= fb["polled"] - pd.Timedelta(seconds=90))]
            prev[sid] = (h.back1.iloc[-1], h.lay1.iloc[-1]) if len(h) else None
        d = el.decide(el.first_rows(first, prev, p[p.selection_id.isin(list(first))]), fit, cfg)
        now = {int(sid): _book_at(pre, mid, sid, t_dec) for sid in first}
        for r in d.itertuples(index=False):
            if not r.lay:
                continue
            sid = int(r.selection_id)
            b = now.get(sid)
            if cfg.get("rest_only", True) and b and b["back"]:
                from model.race_book import market_now
                cur = float(market_now([b["back"][0][0]], [b["lay"][0][0] if b["lay"] else np.nan])[0])
                if float(r.lay_price) >= cur - 1e-9:
                    continue
            size = el.stake_for(float(r.lay_price), cfg)
            s = simulate_lay(data, mid, sid, t_dec, float(r.lay_price), size, cfg["green_seconds"])
            out.append({**s, "f_bsp": round(float(r.f_bsp), 3), "mid": round(float(r.mid), 3)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------------------------------
# Trading both sides of every dog up to the off (the owner, 3 Nov 2026: "it is allowed to execute trades on each side
# of the line on every dog right up to the off ... optimise towards our target")
# --------------------------------------------------------------------------------------------------------------------

def outcomes_of(fills: list[tuple[str, float, float]]) -> tuple[float, float]:
    """A dog's matched position, (side, price, stake) each: its P&L if it wins and if it loses."""
    win = sum(st * (p - 1) if sd == "back" else -st * (p - 1) for sd, p, st in fills)
    lose = sum(-st if sd == "back" else st for sd, p, st in fills)
    return win, lose


def hedge(fills, book: dict | None, bsp: float | None, won: int) -> tuple[float, str]:
    """The position greened up at the exit book: one order of stake ``(win - lose) / G`` (a lay when positive, a back
    when negative) at the price ``G`` it gets down the book makes both outcomes equal, ``lose + (win - lose) / G``.
    Its value ``|win - lose|`` does not depend on ``G``, so ``G`` is read down the book by value. No exit book: the
    same at the BSP (an approximation of the SP close), else settled by the result."""
    win, lose = outcomes_of(fills)
    if abs(win - lose) < 1e-9:
        return lose, "flat"
    side = "lay" if win > lose else "back"
    G = green_price(book, abs(win - lose), side)
    how = "green"
    if not G or G <= 1:
        if not bsp or bsp <= 1:
            return (win if won else lose), "held"
        G, how = bsp, "sp_close"
    return lose + (win - lose) / G, how


def fair_values(pre_m: pd.DataFrame, t, preds_m: pd.DataFrame, fit: dict, cfg: dict, closing: dict | None = None,
                off=None) -> pd.DataFrame | None:
    """The forecast SP of every dog from the books as they stood at ``t``, with the mid: the live trader's forecast
    (read at any book rather than only at the first traded one), or with ``closing`` the closing-price model's. None
    until every dog has traded."""
    from greyhound import early_lay as el
    now = pre_m[pre_m.polled <= t].groupby("selection_id").tail(1)
    if not len(now) or not (now.last_traded > 1).all() or set(now.selection_id) - set(preds_m.selection_id):
        return None
    first, prev = {}, {}
    for r in now.itertuples(index=False):
        sid = int(r.selection_id)
        first[sid] = {"market_id": r.market_id, "polled": r.polled, "back1": r.back1, "back1_size": r.back1_size,
                      "lay1": r.lay1, "lay1_size": r.lay1_size, "last_traded": r.last_traded,
                      "traded": float(r.runner_total_matched or 0.0), "market_traded": float(r.market_total_matched or 0.0)}
        h = pre_m[(pre_m.selection_id == r.selection_id) & (pre_m.polled <= t - pd.Timedelta(seconds=90))]
        prev[sid] = (h.back1.iloc[-1], h.lay1.iloc[-1]) if len(h) else None
    if closing is not None:
        return closing_values(now, prev, preds_m, closing, t, off)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = el.decide(el.first_rows(first, prev, preds_m[preds_m.selection_id.isin(list(first))]), fit,
                      dict(cfg, rest_only=False))
    return d[["selection_id", "f_bsp", "mid"]]


def closing_fit(store, day: date) -> dict | None:
    """The closing-price model (research/greyhound/bfsp_close.py ``gbm_pooled``, the most accurate of report section
    26): boosted trees on log(BSP) less log(mid), one model over every mark, fitted on every recorded day before
    ``day`` (the recorder's books, retimed). Inputs the training books never varied (the recorder's had no traded
    volume) are held at their training value, so a live book's volume cannot move the forecast where the fit never
    learnt it."""
    import lightgbm as lgb
    from greyhound import tradeout as to
    from research.greyhound import bfsp_close as bc
    panel = ls_load(store, to.FIRST_BOOK_DAY, day - timedelta(days=1))
    if not len(panel):
        return None
    p = bc.book_features(panel[panel["mark"].isin(bc.MARKS) & (panel.bsp > 1) & (panel["mid"] > 1)])
    if p.race_date.nunique() < 7:
        return None
    feats = [c for c in bc.GBM_FEATURES if c in p.columns]
    resid = np.log(p.bsp.to_numpy(float)) - p["log_mid"].to_numpy(float)
    m = lgb.train(bc.GBM_PARAMS, lgb.Dataset(p[feats].astype(float), resid), bc.GBM_ROUNDS)
    const = {c: float(p[c].dropna().iloc[0]) if p[c].notna().any() else np.nan
             for c in feats if p[c].nunique(dropna=True) <= 1}
    _log(f"closing model: {len(p)} runner-marks over {p.race_date.nunique()} days; held at training value: "
         f"{sorted(const)}")
    return {"model": m, "feats": feats, "book": bc.sp_book(p), "const": const}


def ls_load(store, first: date, last: date) -> pd.DataFrame:
    from research.greyhound import limit_study as ls
    return ls.load(store, first, last, source="retimed")


def closing_values(now: pd.DataFrame, prev: dict, preds_m: pd.DataFrame, closing: dict, t, off) -> pd.DataFrame:
    """The closing model's forecast SP of every dog from its book ``now`` (one row a dog), rescaled in the race to
    the SP book's usual total, with the mid."""
    from greyhound import tradeout as to
    from research.greyhound import bfsp_close as bc
    p = preds_m.drop_duplicates("selection_id").set_index("selection_id")
    d = now.copy()
    d["race_id"] = d.market_id.astype(str)
    d["mark"] = "live"
    d["p_model"] = d.selection_id.map(p.p_model).astype(float)
    d["p_model"] = d.p_model / d.p_model.sum()
    d["trap"] = d.selection_id.map(p.trap).astype(float) if "trap" in p else np.nan
    d["field"] = float(len(d))
    d["minutes_to_off"] = (off - t).total_seconds() / 60.0
    d["back1_prev"] = d.selection_id.map(lambda s: (prev.get(int(s)) or (np.nan, np.nan))[0])
    d["lay1_prev"] = d.selection_id.map(lambda s: (prev.get(int(s)) or (np.nan, np.nan))[1])
    d["bsp"] = np.nan
    d["won"] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = bc.book_features(to.add_features(d.reset_index(drop=True)))
        for c, v in closing["const"].items():
            x[c] = v
        raw = x["log_mid"].to_numpy(float) + closing["model"].predict(x[closing["feats"]].astype(float))
        lp = bc.rescale(raw, x, closing["book"])
    return pd.DataFrame({"selection_id": x.selection_id.astype(int).to_numpy(), "f_bsp": np.exp(lp),
                         "mid": x["mid"].to_numpy(float)})


def live_level(fv: pd.DataFrame) -> pd.DataFrame:
    """A forecast's shares within the race kept and its level taken from the race's own book now: the prices scaled
    so their chances sum to the mids' total (each race's own overround), not to a fixed SP book from past days."""
    p = 1.0 / fv.f_bsp.to_numpy(float)
    p = p / p.sum()
    book = float(np.sum(1.0 / fv["mid"].to_numpy(float)))
    return fv.assign(f_bsp=1.0 / (book * p))


#: The fair values the simulator can quote around: the live trader's forecast (cl), the closing-price model
#: (closing), each with its level from the race's live book (*_live), and the market mid itself (mid).
FAIRS = ("cl", "closing", "cl_live", "closing_live", "mid")


def fair_by(name: str, pre_m: pd.DataFrame, t, preds_m: pd.DataFrame, fit: dict, cfg: dict,
            closing: dict | None, off) -> pd.DataFrame | None:
    if name.startswith("closing"):
        fv = fair_values(pre_m, t, preds_m, fit, cfg, closing=closing, off=off)
    else:
        fv = fair_values(pre_m, t, preds_m, fit, cfg)
    if fv is None:
        return None
    if name == "mid":
        return fv.assign(f_bsp=fv["mid"])
    return live_level(fv) if name.endswith("_live") else fv


#: The live trader's forecast error, RMSE of log(BSP), by seconds to the off: its conditional logit at the first traded
#: book (taken here as 10 min out), 5, 3 and 1 min out (reports/greyhound_model.md section 26, read off the recorder's
#: books), interpolated between and held beyond. A quote at ``k`` of these from the forecast sits outside most of its
#: error; one well inside it is filled mostly when the market knows better (the fixed 2% and 5% margins, 3 Nov).
SIGMA_SECS = [60.0, 180.0, 300.0, 600.0]
SIGMA_RMSE = [0.146, 0.244, 0.414, 0.356]
#: the closing-price model's (trees pooled over marks, the same table)
SIGMA_RMSE_CLOSE = [0.144, 0.225, 0.370, 0.320]


def sigma_at(secs: float, fair: str = "cl") -> float:
    return float(np.interp(secs, SIGMA_SECS, SIGMA_RMSE_CLOSE if fair.startswith("closing") else SIGMA_RMSE))


def quote_prices(f: float, q: dict, secs: float) -> tuple[float, float]:
    """The lay and back quote around a forecast ``f``: a fixed share (``q['margin']``), or ``q['k']`` forecast errors
    at this distance from the off (log space: ``f x exp(-k sigma)`` and ``f x exp(k sigma)``)."""
    if q.get("k") is not None:
        w = q["k"] * sigma_at(secs, q.get("fair", "cl"))
        return f * float(np.exp(-w)), f * float(np.exp(w))
    return f * (1 - q["margin"]), f * (1 + q["margin"])


def quote_race(data: dict, mid: str, preds_m: pd.DataFrame, fit: dict, cfg: dict, q: dict,
               closing: dict | None = None) -> list[dict]:
    """One race traded both ways: at every live-key book from the moment the whole field has traded until
    ``q['stop_seconds']`` before the off, each dog's fair value (the forecast SP) quoted with a resting back at
    ``fair x (1 + margin)`` (rounded up the ladder; only above the mid) and a resting lay at ``fair x (1 - margin)``
    (rounded down; only below the mid). An order whose price no longer holds is cancelled (what it matched kept) and
    a new one placed, back of the queue. A dog stops taking new orders on a side once its worst outcome reaches
    ``q['max_loss']``. Every position is greened up ``green_seconds`` before the off."""
    from greyhound import early_lay as el
    from research.greyhound import limit_study as ls
    pre, lad = data["pre"], data["ladder"]
    off = data["off"].get(mid)
    pre_m = pre[pre.market_id == mid]
    t_exit = off - pd.Timedelta(seconds=cfg["green_seconds"])
    t_stop = off - pd.Timedelta(seconds=q.get("stop_seconds", cfg["green_seconds"] + 5))
    lad_m = {sid: g for sid, g in lad[lad.market_id == mid].groupby("selection_id")} if lad is not None else {}
    open_: dict = {}
    fills: dict = {}
    late: dict = {}
    orders = []

    def book_of(sid, t):
        return _book_at(pre_m, mid, sid, t)

    lat = pd.Timedelta(seconds=float(q.get("latency", 0.0)))

    def thr(o):
        tm = q.get("through_min", 0.0)
        return float(o["size"]) if tm == "size" else float(tm)

    def close(key, t1):
        # a cancel reaches the exchange ``latency`` after it is sent: the order can still fill until then
        o = open_.pop(key)
        f = qf.fill(o["side"], o["price"], o["size"], o["t0"], o["book"], lad_m.get(key[0]), until=t1 + lat,
                    through_min=thr(o))
        if f.matched > 0:
            fills.setdefault(key[0], []).append((o["side"], o["price"], f.matched))
            if o["t0"] >= off - pd.Timedelta(seconds=60):
                late[key[0]] = late.get(key[0], 0.0) + f.matched
        orders.append({**o, "t1": t1, "matched": f.matched})

    for t in sorted(pre_m.polled.unique()):
        t = pd.Timestamp(t)
        if t > t_stop:
            break
        fair = q.get("fair", "cl")
        fv = fair_values(pre_m, t, preds_m, fit, cfg) if fair == "cl" else \
            fair_by(fair, pre_m, t, preds_m, fit, cfg, closing, off)
        if fv is None:
            continue
        for r in fv.itertuples(index=False):
            sid = int(r.selection_id)
            f, m = float(r.f_bsp), float(r.mid)
            # what is matched so far (open orders read to now) sets the dog's worst outcome
            got = list(fills.get(sid, []))
            for (s2, side), o in open_.items():
                if s2 == sid:
                    ff = qf.fill(side, o["price"], o["size"], o["t0"], o["book"], lad_m.get(sid), until=t,
                                 through_min=thr(o))
                    if ff.matched > 0:
                        got.append((side, o["price"], ff.matched))
            win, lose = outcomes_of(got)
            lay_at, back_at = quote_prices(f, q, (off - t).total_seconds())
            for side in q["sides"]:
                if side == "lay":
                    target = float(ls.ladder([lay_at], up=False)[0])
                    ok = np.isfinite(target) and 1 < target < m - 1e-9 and target <= cfg["max_price"] \
                        and win > -q["max_loss"]                          # a lay loses if the dog wins
                else:
                    target = float(ls.ladder([back_at], up=True)[0])
                    ok = np.isfinite(target) and target > m + 1e-9 and target <= cfg["max_price"] \
                        and lose > -q["max_loss"]                         # a back loses if it loses
                key = (sid, side)
                cur = open_.get(key)
                if cur is not None and (not ok or abs(cur["price"] - target) > 1e-9):
                    close(key, t)
                    cur = None
                if ok and cur is None:
                    size = el.stake_for(target, cfg)
                    # placed on the book read at ``t``; it rests from ``t + latency`` (fills count from then)
                    open_[key] = {"market_id": mid, "selection_id": sid, "side": side, "price": target, "size": size,
                                  "t0": t + lat, "book": book_of(sid, t), "f_bsp": round(f, 3), "mid": round(m, 3)}
    for key in list(open_):
        close(key, t_exit)
    res = data["result"]
    out = []
    for sid, fl in fills.items():
        rr = res[(res.market_id == mid) & (res.selection_id == sid)]
        bsp = float(rr.bsp.iloc[0]) if len(rr) and pd.notna(rr.bsp.iloc[0]) else None
        won = int(rr.won.iloc[0]) if len(rr) else 0
        pnl, how = hedge(fl, book_of(sid, t_exit), bsp, won)
        out.append({"market_id": mid, "selection_id": sid, "orders": sum(o["selection_id"] == sid for o in orders),
                    "back_matched": round(sum(st for sd, p, st in fl if sd == "back"), 2),
                    "lay_matched": round(sum(st for sd, p, st in fl if sd == "lay"), 2),
                    "matched": round(sum(st for sd, p, st in fl), 2), "pnl": round(pnl, 2), "how": how,
                    "matched_last_60s": round(late.get(sid, 0.0), 2)})
    if not out:
        out.append({"market_id": mid, "selection_id": None, "orders": len(orders), "back_matched": 0.0,
                    "lay_matched": 0.0, "matched": 0.0, "pnl": 0.0, "how": "unmatched", "matched_last_60s": 0.0})
    return out


def two_sided(store, day: date, data: dict, cfg: dict, fit: dict, q: dict, closing: dict | None = None) -> pd.DataFrame:
    """Every settled race of the day traded both ways (``quote_race``)."""
    raw = store.get(f"greyhound/live/predictions/{day:%Y-%m-%d}.csv")
    if not raw:
        return pd.DataFrame()
    preds = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    preds["selection_id"] = pd.to_numeric(preds.selection_id, errors="coerce")
    preds = preds.dropna(subset=["selection_id"]).drop_duplicates(["market_id", "selection_id"], keep="last")
    settled = set(data["result"].market_id)
    out = []
    for mid in sorted(set(data["pre"].market_id) & settled & set(preds.market_id)):
        if pd.isna(data["off"].get(mid)):
            continue
        try:
            out += quote_race(data, mid, preds[preds.market_id == mid], fit, cfg, q, closing=closing)
        except Exception as exc:                                   # one race's fault never stops the day
            log.warning("two-sided %s: %s", mid, exc)
    return pd.DataFrame(out)


ACC_MARKS = {"10 min": 600.0, "5 min": 300.0, "3 min": 180.0, "1 min": 60.0, "15 s": 15.0}


def fair_accuracy(store, day: date, data: dict, fit: dict, cfg: dict, closing: dict | None) -> list[dict]:
    """Each fair value's error on the day's settled races, read off the live-key books at the poll nearest each mark
    (within 30 s, or 90 s for 10 min out where the books are a minute apart): RMSE of log(BSP) for the live
    trader's forecast, the closing model's and the market mid, the same runners for each."""
    raw = store.get(f"greyhound/live/predictions/{day:%Y-%m-%d}.csv")
    if not raw or closing is None:
        return []
    preds = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    preds["selection_id"] = pd.to_numeric(preds.selection_id, errors="coerce")
    preds = preds.dropna(subset=["selection_id"]).drop_duplicates(["market_id", "selection_id"], keep="last")
    res = data["result"].set_index(["market_id", "selection_id"]).bsp
    rows = []
    for mid in sorted(set(data["pre"].market_id) & set(data["result"].market_id) & set(preds.market_id)):
        off = data["off"].get(mid)
        if off is None or pd.isna(off):
            continue
        pre_m = data["pre"][data["pre"].market_id == mid]
        times = pd.Series(sorted(pre_m.polled.unique()))
        pm = preds[preds.market_id == mid]
        for name, secs in ACC_MARKS.items():
            if not len(times):
                continue
            target = off - pd.Timedelta(seconds=secs)
            i = (times - target).abs().idxmin()
            t = pd.Timestamp(times[i])
            if abs((t - target).total_seconds()) > (90 if secs >= 600 else 30):
                continue
            try:
                a = fair_values(pre_m, t, pm, fit, cfg)
                b = fair_values(pre_m, t, pm, fit, cfg, closing=closing, off=off)
            except Exception as exc:
                log.warning("accuracy %s %s: %s", mid, name, exc)
                continue
            if a is None or b is None:
                continue
            m = a.merge(b, on="selection_id", suffixes=("_cl", "_close"))
            m = m.assign(cl_live=live_level(a.rename(columns={"f_bsp": "f_bsp"})).set_index("selection_id").f_bsp
                         .reindex(m.selection_id).to_numpy(),
                         closing_live=live_level(b).set_index("selection_id").f_bsp.reindex(m.selection_id).to_numpy())
            for r in m.itertuples(index=False):
                y = res.get((mid, float(r.selection_id)))
                if y is None or not np.isfinite(y) or y <= 1:
                    continue
                rows.append({"mark": name, "market_id": mid, "y": np.log(y), "cl": np.log(r.f_bsp_cl),
                             "closing": np.log(r.f_bsp_close), "mid": np.log(r.mid_cl),
                             "cl_live": np.log(r.cl_live), "closing_live": np.log(r.closing_live)})
    if not rows:
        return []
    d = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    cols = ("mid", "cl", "closing", "cl_live", "closing_live")
    out = []
    for name in ACC_MARKS:
        g = d[d["mark"] == name]
        if not len(g):
            continue
        # within the race: each log price less its race's mean, so a level wrong for the whole race does not count
        rel = lambda c: g[c] - g.groupby("market_id")[c].transform("mean")
        yr = rel("y")
        out.append({"mark": name, "runners": int(len(g)), "races": int(g.market_id.nunique()),
                    **{c: round(float(np.sqrt(np.mean((g[c] - g.y) ** 2))), 4) for c in cols},
                    **{f"{c}_within": round(float(np.sqrt(np.mean((rel(c) - yr) ** 2))), 4)
                       for c in ("mid", "cl", "closing")}})
    return out


QUOTES = {
    "both_m02": {"sides": ("back", "lay"), "margin": 0.02, "max_loss": 60.0},
    "both_m05": {"sides": ("back", "lay"), "margin": 0.05, "max_loss": 60.0},
    "both_m10": {"sides": ("back", "lay"), "margin": 0.10, "max_loss": 60.0},
    "both_m20": {"sides": ("back", "lay"), "margin": 0.20, "max_loss": 60.0},
    "both_m30": {"sides": ("back", "lay"), "margin": 0.30, "max_loss": 60.0},
    "sigma_k05": {"sides": ("back", "lay"), "k": 0.5, "max_loss": 60.0},
    "sigma_k10": {"sides": ("back", "lay"), "k": 1.0, "max_loss": 60.0},
    "sigma_k15": {"sides": ("back", "lay"), "k": 1.5, "max_loss": 60.0},
    "close_m10": {"sides": ("back", "lay"), "margin": 0.10, "max_loss": 60.0, "fair": "closing"},
    "close_m20": {"sides": ("back", "lay"), "margin": 0.20, "max_loss": 60.0, "fair": "closing"},
    "close_m30": {"sides": ("back", "lay"), "margin": 0.30, "max_loss": 60.0, "fair": "closing"},
    "close_k10": {"sides": ("back", "lay"), "k": 1.0, "max_loss": 60.0, "fair": "closing"},
    "close_k15": {"sides": ("back", "lay"), "k": 1.5, "max_loss": 60.0, "fair": "closing"},
    "close_live_m20": {"sides": ("back", "lay"), "margin": 0.20, "max_loss": 60.0, "fair": "closing_live"},
    "close_live_k15": {"sides": ("back", "lay"), "k": 1.5, "max_loss": 60.0, "fair": "closing_live"},
    "cl_live_m20": {"sides": ("back", "lay"), "margin": 0.20, "max_loss": 60.0, "fair": "cl_live"},
    "cl_live_k15": {"sides": ("back", "lay"), "k": 1.5, "max_loss": 60.0, "fair": "cl_live"},
    "mid_m05": {"sides": ("back", "lay"), "margin": 0.05, "max_loss": 60.0, "fair": "mid"},
    # the same, harder on the fills: orders rest 1.5 s (or 3 s) after the book they were priced on; a trade beyond
    # our price fills us only once at least our own stake has traded there
    "mid_m05_lat": {"sides": ("back", "lay"), "margin": 0.05, "max_loss": 60.0, "fair": "mid", "latency": 1.5},
    "mid_m05_thru": {"sides": ("back", "lay"), "margin": 0.05, "max_loss": 60.0, "fair": "mid", "through_min": "size"},
    "mid_m03_hard": {"sides": ("back", "lay"), "margin": 0.03, "max_loss": 60.0, "fair": "mid", "latency": 1.5,
                     "through_min": "size"},
    "mid_m05_hard": {"sides": ("back", "lay"), "margin": 0.05, "max_loss": 60.0, "fair": "mid", "latency": 1.5,
                     "through_min": "size"},
    "mid_m05_hard3": {"sides": ("back", "lay"), "margin": 0.05, "max_loss": 60.0, "fair": "mid", "latency": 3.0,
                      "through_min": "size"},
    "mid_m07_hard": {"sides": ("back", "lay"), "margin": 0.07, "max_loss": 60.0, "fair": "mid", "latency": 1.5,
                     "through_min": "size"},
    "mid_m10_hard": {"sides": ("back", "lay"), "margin": 0.10, "max_loss": 60.0, "fair": "mid", "latency": 1.5,
                     "through_min": "size"},
    "mid_m10": {"sides": ("back", "lay"), "margin": 0.10, "max_loss": 60.0, "fair": "mid"},
    "mid_m15": {"sides": ("back", "lay"), "margin": 0.15, "max_loss": 60.0, "fair": "mid"},
    "mid_m20": {"sides": ("back", "lay"), "margin": 0.20, "max_loss": 60.0, "fair": "mid"},
    "lay_m20": {"sides": ("lay",), "margin": 0.20, "max_loss": 60.0},
    "back_m20": {"sides": ("back",), "margin": 0.20, "max_loss": 60.0},
}


def summarise_quotes(rows: pd.DataFrame) -> dict:
    if not len(rows):
        return {"races": 0}
    net = race_net(rows)
    return {"races": int(rows.market_id.nunique()), "orders": int(rows.orders.sum()),
            "dogs_matched": int((rows.matched > 0).sum()), "back_matched": round(float(rows.back_matched.sum()), 2),
            "lay_matched": round(float(rows.lay_matched.sum()), 2), "net": round(float(net.net.sum()), 2),
            "per_race": round(float(net.net.mean()), 3), "races_won": int((net.net > 0).sum()),
            "races_lost": int((net.net < 0).sum()), "worst_race": round(float(net.net.min()), 2),
            "best_race": round(float(net.net.max()), 2),
            # how much of the profit the five best races carry, and how much was matched in the last minute
            "best5_share": round(float(net.net.nlargest(5).sum() / net.net.sum()), 3) if net.net.sum() > 0 else None,
            "last_60s_share": round(float(rows.get("matched_last_60s", pd.Series(0.0)).sum() / rows.matched.sum()), 3)
            if rows.matched.sum() > 0 else None}


VARIANTS = {
    "live_rule": {},                                             # rest-only, 80% of the forecast, GBP20 a lay
    "no_guard": {"rest_only": False},
    "margin_10": {"margin": 0.10},
    "margin_30": {"margin": 0.30},
}


def summarise(rows: pd.DataFrame) -> dict:
    if not len(rows):
        return {"lays": 0}
    m = rows[rows.matched > 0]
    net = race_net(rows.fillna({"pnl": 0.0}))
    return {"lays": int(len(rows)), "matched_lays": int(len(m)), "races": int(rows.market_id.nunique()),
            "asked": round(float(rows["size"].sum()), 2), "matched": round(float(rows.matched.sum()), 2),
            "greened": int((m.how == "green").sum()) if len(m) else 0,
            "net": round(float(net.net.sum()), 2), "races_won": int((net.net > 0).sum()),
            "races_lost": int((net.net < 0).sum())}


def report_md(res: dict) -> str:
    L = [f"# The profit simulator, {res['day']} (live-key books)", "",
         "Queue-aware fills on the live-key books and traded ladder; greened up 5 s before the off down the best three "
         "back prices, else closed at the SP; 2% commission on each race's net win.", ""]
    c = res.get("calibration") or {}
    L += ["## 1. Calibration: the live trader's own orders, simulated against what happened", ""]
    if c.get("orders"):
        L += [f"{c['orders']} live lays since {c.get('since')}: matched simulated GBP{c['sim_matched']:.2f} against "
              f"GBP{c['real_matched']:.2f} live; matched or not, the same in {c['agree']} of {c['orders']}; P&L "
              f"simulated GBP{c['sim_pnl']:+.2f} against GBP{c['real_pnl']:+.2f} live (where both known).", ""]
        L += ["| Race | Track | Lay price | Size | Sim matched | Live matched | Sim P&L | Live P&L | Sim how | Live how "
              "| Ladder |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in c.get("rows", []):
            L.append(f"| {r.get('race_time')} | {r.get('track')} | {r.get('price')} | {r.get('size')} | "
                     f"{r.get('matched')} | {r.get('real_matched')} | {r.get('pnl')} | {r.get('real_pnl')} | "
                     f"{r.get('how')} | {r.get('real_how')} | {r.get('ladder')} |")
    else:
        L += ["(no live orders with live-key books to replay)"]
    L += ["", "## 2. Strategies on the day's settled races", "",
          "| Variant | Races | Lays | Matched lays | Asked | Matched | Greened | Net P&L | Races won / lost |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in (res.get("strategies") or {}).items():
        if s.get("lays"):
            L.append(f"| {k} | {s['races']} | {s['lays']} | {s['matched_lays']} | {s['asked']} | {s['matched']} | "
                     f"{s['greened']} | {s['net']:+.2f} | {s['races_won']} / {s['races_lost']} |")
        else:
            L.append(f"| {k} | 0 | 0 | | | | | | |")
    if res.get("fair_accuracy"):
        L += ["", "## 2b. The fair values on today's live-key books", "",
              "RMSE of log(BSP) on the day's settled races (lower is better), the same runners for each. Absolute: "
              "the market mid, the live trader's forecast (cl), the closing-price model (closing), and each with its "
              "level from the race's live book (*_live). Within the race: each log price less its race's mean, so "
              "only the dogs' order and spacing count (a level wrong for the whole race does not).", "",
              "| Before the off | Runners | Races | Mid | cl | closing | cl_live | closing_live | Mid within | cl within "
              "| closing within |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["fair_accuracy"]:
            L.append(f"| {r['mark']} | {r['runners']} | {r['races']} | {r['mid']} | {r['cl']} | {r['closing']} | "
                     f"{r.get('cl_live')} | {r.get('closing_live')} | {r.get('mid_within')} | {r.get('cl_within')} | "
                     f"{r.get('closing_within')} |")
    L += ["", "## 3. Both sides of every dog up to the off", "",
          "Each dog's forecast SP read at every live-key book (the live trader's forecast), a resting back above it and "
          "a resting lay below it (both_mNN: a fixed NN% either side; sigma_kNN: NN/10 of the forecast's error at that "
          "distance from the off, wide early and tighter near the off; close_*: the same around the closing-price "
          "model's forecast; *_live: the forecast's level from the race's live book; mid_*: around the market mid), never across the mid, requoted as it moves; every position greened up 5 s before "
          "the off. Our own orders do not move the simulated market, and two of our orders on one dog can count the "
          "same traded money: an upper bound on the fills.", "",
          "| Variant | Races | Orders | Dogs matched | Back matched | Lay matched | Net P&L | Per race | Races won / "
          "lost | Worst / best race | Best 5 races' share | Matched in the last 60 s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s_ in (res.get("quotes") or {}).items():
        if s_.get("races"):
            L.append(f"| {k} | {s_['races']} | {s_['orders']} | {s_['dogs_matched']} | {s_['back_matched']} | "
                     f"{s_['lay_matched']} | {s_['net']:+.2f} | {s_['per_race']:+.3f} | {s_['races_won']} / "
                     f"{s_['races_lost']} | {s_['worst_race']:+.2f} / {s_['best_race']:+.2f} | "
                     f"{s_.get('best5_share')} | {s_.get('last_60s_share')} |")
        else:
            L.append(f"| {k} | 0 | | | | | | | | |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    from greyhound import early_lay as el, laytest
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--date", default=None, help="UK racing day (default today)")
    ap.add_argument("--since", default=None, help="calibration: live orders from this UTC time (the ladder's start)")
    ap.add_argument("--out", default="out/greyhound_profit_sim")
    ap.add_argument("--only", default=None, help="comma-separated strategy and quote variants to run (default all)")
    a = ap.parse_args(argv)
    t0 = time.time()
    store = Store(root=a.store)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    data = load_day(day)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res = {"day": f"{day}"}
    if data:
        lad = data["ladder"]
        since = a.since or (str(lad.polled_utc.min()) if lad is not None and len(lad) else None)
        rp = replay(store, day, data, since=since)
        if len(rp):
            both = rp[rp.real_pnl.notna() & rp.pnl.notna()]
            res["calibration"] = {
                "since": since, "orders": int(len(rp)), "sim_matched": float(rp.matched.fillna(0).sum()),
                "real_matched": float(rp.real_matched.sum()),
                "agree": int(((rp.matched.fillna(0) > 0) == (rp.real_matched > 0)).sum()),
                "sim_pnl": float(both.pnl.sum()), "real_pnl": float(both.real_pnl.sum()),
                "rows": json.loads(rp.drop(columns=["placed"], errors="ignore").to_json(orient="records"))}
        fit = laytest.day_fit(store, day)
        res["strategies"] = {}
        if fit is not None:
            only = set(a.only.split(",")) if a.only else None
            for name, over in VARIANTS.items():
                if only and name not in only:
                    continue
                rows = early_lay(store, day, data, dict(el.CONFIG, **over), fit)
                res["strategies"][name] = summarise(rows)
                _log(f"{name}: {res['strategies'][name]}")
            res["quotes"] = {}
            closing = None
            if any(q.get("fair", "cl").startswith("closing") and (not only or n in only) for n, q in QUOTES.items()):
                closing = closing_fit(store, day)
                res["fair_accuracy"] = fair_accuracy(store, day, data, fit, dict(el.CONFIG), closing)
                _log(f"fair value accuracy: {res['fair_accuracy']}")
            for name, q in QUOTES.items():
                if only and name not in only:
                    continue
                if q.get("fair", "cl").startswith("closing") and closing is None:
                    continue
                rows = two_sided(store, day, data, dict(el.CONFIG), fit, q, closing=closing)
                res["quotes"][name] = summarise_quotes(rows)
                if len(rows):
                    rows.to_csv(Path(a.out).parent / f"{Path(a.out).name}_{name}.csv", index=False)
                _log(f"{name}: {res['quotes'][name]}")
    else:
        res["note"] = "no live-key data for the day"
    res["seconds"] = round(time.time() - t0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    md = report_md(res)
    out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
