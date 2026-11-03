"""The greyhound limit-order study: asking for a price instead of taking it (the owner, 3 Nov 2026: "build the
limit-order study ... take prices based on the behaviour of the greyhound market. I am open to any strategy").

Why (reports/greyhound_model.md section 20): our price helps call which runners will shorten to the SP, but a
greyhound book is thin until its last minutes, and the best back price on offer is usually shorter than the SP (a
median gap of 22% at the first traded book, 27% five minutes out, 9% one minute out). Taking it gives away more than
the forecast gains. So here each order is left in the book at a chosen price, matched only if the market comes to it
before the off, and otherwise lapses. A back matched at P is closed as the horse trader closes it: its winnings laid
at the SP (``greyhound.tradeout.positions``' settlement), so it pays ``P/SP - 1`` in expectation.

The orders, at each entry mark (first traded, 5 and 3 minutes out), for every runner of every race in the panel:

- ``ask``: a back at our forecast SP plus a margin (``MARGINS``), on Betfair's ladder; where the best back price
  already reaches it, it is taken there. ``bid``: a lay at the forecast SP less the margin, the mirror.
- ``join``: a back at the best lay price now (the head of the backers' queue) or a lay at the best back price now.
  No model: the market's own behaviour.
- ``ticks``: a back ``k`` ticks above the best back price now, or a lay ``k`` ticks below the best lay price.

Each with a gate: ``all`` runners, those our forecast says will ``shorten`` (forecast SP under the price now) or
``drift``. The forecast is the study's conditional logit on the book and our price (``research.greyhound.
tradeout_study``'s ``micro``), fitted walk-forward on the days before each day only.

A resting order is counted matched by one of two rules, as bounds: ``through`` (a later book, before the off, offers
the other side at or beyond our price: a back at P is matched once the best back price reaches P; certain, and late:
it misses touches between polls) and ``traded`` (a later last-traded price reaches P: money changed hands at our
price, though our place in the queue may not have come). The size is assumed matched in full, which flatters large
stakes; the stakes here are the horse trader's, to win GBP100.

Exits (the owner, 3 Nov 2026: "we want to trade out and take profit whenever we get the movement correct. I dont
care if the dog wins or not. We are trading the price move"): ``sp``, closed at the SP and settled by the result;
``last``, greened up at the market in the runner's last book before the off; ``tp<k>``, take profit: once the
entry is matched, a closing order ``k`` ticks better than the entry price (a lay at P backed at ``k`` ticks above P,
a back laid ``k`` ticks below) matched by the same fill rule in a later book, so the profit is locked as soon as the
move comes; where it never comes, greened up in the last book as ``last``. These need the books' path, not just
their extremes: the entry's fill time (the first later book that matches it), then the books after it.

What it reports: the market's behaviour (how each price moves from each mark to the SP, by price band and by the
market's order); every rule on the development days; the best of each family on the development days read once on
the holdout; and the adverse selection the passive orders meet (the CLV of the orders matched against those not).

    python -m research.greyhound.limit_study --source retimed --out out/greyhound_limit_study
    python -m research.greyhound.limit_study --pred-dir out/fastloop/pred/logbsp_fb_sd --out ...
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound import tradeout as to
from research.greyhound import tradeout_study as ts

log = logging.getLogger(__name__)

ORDER_MARKS = ("first_traded", "off_5", "off_3")
MARGINS = (0.0, 0.05, 0.10, 0.20)
TICKS = (1, 3, 5)
GATES = ("all", "shorten", "drift", "fav", "short", "long")
#: through / traded: matched once a later book offers, or a later trade is at, our price (no queue: the upper bounds);
#: strict (the owner, 3 Nov 2026: "count a lay as matched only when the money traded at our price exceeds what was
#: queued ahead of us"): matched only once a later trade is strictly beyond our price, which can happen only after
#: every order at our price, ours too, is matched; with no volume by price in the recorded books, the lower bound
FILLS = ("through", "traded", "strict")
TP_TICKS = (2, 5, 10)                  # take profit: green up once the price moves this many ticks our way
EXITS = ("sp", "last") + tuple(f"tp{k}" for k in TP_TICKS)
BOOK_PATH_COLS = ["market_id", "selection_id", "polled", "back1", "lay1", "last_traded"]
KEYS = ("mark", "order", "side", "param", "gate", "fill", "exit")
SHORT_PRICE = 3.0                      # 'short': the mid now under 3.0; 'long': 8.0 or more
LONG_PRICE = 8.0
BANDS = (1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, 1000.0)
MIN_TRAIN_DAYS = ts.MIN_TRAIN_DAYS
HOLDOUT_DAYS = ts.HOLDOUT_DAYS


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------------------------------
# Prices on Betfair's ladder
# --------------------------------------------------------------------------------------------------------------------

def ladder(x, up: bool) -> np.ndarray:
    """Each price moved onto Betfair's ladder (up: the smallest at or above; else the largest at or below)."""
    from trading.exchange import tick_down, tick_up
    f = tick_up if up else tick_down
    x = np.asarray(x, float)
    return np.array([f(v) if np.isfinite(v) and v > 1 else np.nan for v in x])


def ticks_from(x, k: int, up: bool) -> np.ndarray:
    """``k`` ticks above (``up``) or below each price on the ladder."""
    p = ladder(x, up)
    for _ in range(k):
        p = ladder(p + (1e-4 if up else -1e-4), up)
    return np.where(p > 1.0, p, np.nan)


# --------------------------------------------------------------------------------------------------------------------
# The data: the panel at each mark, and every runner's books after it
# --------------------------------------------------------------------------------------------------------------------

def future_extremes(pre: pd.DataFrame) -> pd.DataFrame:
    """For each pre-off book of a runner, the best prices from that book to the off: the highest best back price, the
    lowest best lay price, the highest and lowest last-traded prices (``fut_*``)."""
    b = pre[["market_id", "selection_id", "polled", "back1", "lay1", "last_traded"]].copy()
    b["last_traded"] = to._num(b.last_traded)
    b = b.sort_values(["market_id", "selection_id", "polled"], ascending=[True, True, False])
    g = b.groupby(["market_id", "selection_id"], sort=False)
    b["fut_back_max"] = g.back1.cummax()
    b["fut_lay_min"] = b.lay1.where(b.lay1 > 1).groupby([b.market_id, b.selection_id], sort=False).cummin()
    lt = b.last_traded.where(b.last_traded > 1)
    b["fut_lt_max"] = lt.groupby([b.market_id, b.selection_id], sort=False).cummax()
    b["fut_lt_min"] = lt.groupby([b.market_id, b.selection_id], sort=False).cummin()
    last = b.drop_duplicates(["market_id", "selection_id"])           # sorted newest first: each runner's last book
    b = b.merge(last[["market_id", "selection_id", "back1", "lay1"]].rename(
        columns={"back1": "last_back1", "lay1": "last_lay1"}), on=["market_id", "selection_id"], how="left")
    return b.drop(columns=["back1", "lay1", "last_traded"]).sort_values("polled").reset_index(drop=True)


def attach_future(panel: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    """Each panel row with the best prices of the books after its own (strictly later), up to the off."""
    p = panel.copy()
    p["_i"] = np.arange(len(p))
    p["polled"] = pd.to_datetime(p.polled, utc=True)
    f = fut.rename(columns={"polled": "fut_polled"})
    out = pd.merge_asof(p.sort_values("polled"), f, left_on="polled", right_on="fut_polled",
                        by=["market_id", "selection_id"], direction="forward", allow_exact_matches=False)
    return out.sort_values("_i").drop(columns="_i").reset_index(drop=True)


def load(store, first: date, last: date, source: str = "retimed", pred_dir: str | None = None,
         books_out: list | None = None) -> pd.DataFrame:
    """Every day's panel (``greyhound.tradeout.day_panel``, as the trade-out study builds it) with each row's later
    books attached; the pre-off books themselves (the path, for the take-profit exits) appended to ``books_out``."""
    from betfair_recorder import _read_day_csv
    frames, d = [], first
    while d <= last:
        try:
            if pred_dir:
                f = Path(pred_dir) / f"{d:%Y-%m-%d}.parquet"
                pred = pd.read_parquet(f) if f.exists() else None
            else:
                pred = store.get_parquet(f"{to.TRACK_PRED}/{d:%Y-%m-%d}.parquet")
            books = _read_day_csv(d, "books_greyhound.csv")
            markets = _read_day_csv(d, "markets_greyhound.csv")
            entry = books if source in ("recorder", "retimed") else _read_day_csv(d, to.ENTRY_BOOKS["live"])
            if pred is None or entry is None or books is None or markets is None:
                _log(f"{d}: no data")
            else:
                panel = to.day_panel(pred, entry, markets, d, closing_books=books, retimed=source == "retimed")
                if len(panel):
                    pre = to.pre_off_books(entry, markets, retimed=source == "retimed")
                    pre = pre[pre.market_id.isin(set(panel.market_id))]
                    frames.append(attach_future(panel, future_extremes(pre)))
                    if books_out is not None:
                        books_out.append(pre[BOOK_PATH_COLS].assign(lay1=pre.lay1.where(pre.lay1 > 1),
                                                                last_traded=to._num(pre.last_traded)))
                _log(f"{d}: {len(panel)} panel rows")
        except Exception as exc:                                    # one bad day: noted, the rest still read
            _log(f"{d}: failed ({exc})")
        d += timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------------------------------------
# The forecast of the SP, walk-forward
# --------------------------------------------------------------------------------------------------------------------

def forecast(panel: pd.DataFrame) -> pd.DataFrame:
    """Each scored day's rows with ``f_bsp``, the forecast SP from the conditional logit on the book and our price
    fitted on the days before that day only, at the same mark."""
    days = sorted(panel.race_date.astype(str).unique())
    out = []
    for i, day in enumerate(days):
        if i < MIN_TRAIN_DAYS:
            continue
        for mark, g in panel[panel["mark"].isin(ORDER_MARKS)].groupby("mark"):
            tr = g[g.race_date.astype(str) < day]
            te = g[g.race_date.astype(str) == day]
            if not len(tr) or not len(te):
                continue
            q, book = ts.predict(te, ts.fit_cl(tr, ts.CL_SETS["micro"]))
            out.append(te.assign(f_bsp=1.0 / (book * np.asarray(q, float))))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------------------------------------------------
# The orders and their fills
# --------------------------------------------------------------------------------------------------------------------

def order_prices(d: pd.DataFrame) -> list[tuple[str, str, float, np.ndarray]]:
    """Every order's limit price for each row: (order, side, param, price)."""
    back1, lay1 = d.back1.to_numpy(float), d.lay1.where(d.lay1 > d.back1).to_numpy(float)
    f = d.f_bsp.to_numpy(float)
    out = []
    for m in MARGINS:
        out.append(("ask", "back", m, ladder(f * (1 + m), up=True)))
        out.append(("bid", "lay", m, ladder(f * (1 - m), up=False)))
    out.append(("join", "back", 0.0, lay1))
    out.append(("join", "lay", 0.0, back1))
    out.append(("take", "back", 0.0, back1))                  # the price on offer now, taken at once
    out.append(("take", "lay", 0.0, lay1))
    for k in TICKS:
        out.append(("ticks", "back", float(k), ticks_from(back1, k, up=True)))
        out.append(("ticks", "lay", float(k), ticks_from(lay1, k, up=False)))
    return out


def fills(d: pd.DataFrame, side: str, price: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row's order at ``price``: matched or not, the price matched (a back at or above the best back price
    now is taken there; a resting order is matched at its own price), and whether it was taken at once."""
    back1, lay1 = d.back1.to_numpy(float), d.lay1.where(d.lay1 > 1).to_numpy(float)
    if side == "back":
        now = back1 >= price
        if rule == "strict":
            later = d.fut_lt_max.to_numpy(float) > price + 1e-9
        else:
            later = (d.fut_back_max if rule == "through" else d.fut_lt_max).to_numpy(float) >= price
        matched_at = np.where(now, back1, price)
    else:
        now = lay1 <= price
        if rule == "strict":
            later = d.fut_lt_min.to_numpy(float) < price - 1e-9
        else:
            later = (d.fut_lay_min if rule == "through" else d.fut_lt_min).to_numpy(float) <= price
        matched_at = np.where(now, lay1, price)
    ok = np.isfinite(price) & (price > 1.0) & (price <= to.MAX_PRICE)
    filled = ok & (now | later)                         # a NaN (no later book) compares False: not matched
    return filled, matched_at, ok & now


def exit_paths(e: pd.DataFrame, books: pd.DataFrame) -> np.ndarray:
    """For each entry order (rows of ``e``: market_id, selection_id, polled (the mark), side, fill, limit, filled,
    taken), the best price on the closing side in the books after the entry is matched: a back's lowest best lay
    price (``through``) or lowest traded price (``traded``), a lay's highest best back or traded price. The entry is
    matched at the mark if taken, else in the first later book that reaches its limit by the same rule. NaN where
    unmatched or no book follows the fill. Day by day, so the join of orders and books stays small."""
    best = np.full(len(e), np.nan)
    if books is None or not len(books) or not len(e):
        return best
    e = e.assign(_e=np.arange(len(e)), polled=pd.to_datetime(e.polled, utc=True))
    b = books.assign(polled=pd.to_datetime(books.polled, utc=True)).rename(columns={"polled": "polled_b"})
    b["market_id"] = b.market_id.astype(str)
    e["market_id"] = e.market_id.astype(str)
    for _, day in e[e.filled].groupby(e.race_date.astype(str)):
        j = day[["_e", "market_id", "selection_id", "polled", "side", "fill", "limit", "taken"]].merge(
            b[b.market_id.isin(set(day.market_id))], on=["market_id", "selection_id"])
        j = j[j.polled_b > j.polled]
        if not len(j):
            continue
        back, thr = (j.side == "back").to_numpy(), (j.fill == "through").to_numpy()
        lt = j.last_traded.where(j.last_traded > 1).to_numpy(float)
        entry = np.where(back, np.where(thr, j.back1, lt), np.where(thr, j.lay1, lt))
        lim = j.limit.to_numpy(float)
        strict = (j.fill == "strict").to_numpy()
        hit = np.where(back, np.where(strict, entry > lim + 1e-9, entry >= lim),
                       np.where(strict, entry < lim - 1e-9, entry <= lim))
        first = j.polled_b.where(hit).groupby(j._e).min()
        t_fill = j.polled.where(j.taken, j._e.map(first))
        k = j[(j.polled_b > t_fill).to_numpy()]
        if not len(k):
            continue
        kb, kt = (k.side == "back").to_numpy(), (k.fill == "through").to_numpy()
        klt = k.last_traded.where(k.last_traded > 1).to_numpy(float)
        close = np.where(kb, np.where(kt, k.lay1, klt), np.where(kt, k.back1, klt))
        signed = pd.Series(np.where(kb, -close, close), index=k.index).groupby(k._e.to_numpy()).max()
        sides = day.set_index("_e").side.reindex(signed.index)
        best[signed.index.to_numpy()] = np.where(sides == "back", -signed, signed)
    return best


def gate_flags(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Which runners each gate admits: all; our forecast says shorten (forecast SP under the mid now) or drift; the
    market's favourite now; short-priced (mid under SHORT_PRICE); long-priced (LONG_PRICE or more)."""
    mid = d["mid"].to_numpy(float)
    shorten = d.f_bsp.to_numpy(float) < mid
    return {"all": np.ones(len(d), bool), "shorten": shorten, "drift": ~shorten,
            "fav": d["rank"].to_numpy(float) == 1, "short": mid < SHORT_PRICE, "long": mid >= LONG_PRICE}


def settle_orders(o: pd.DataFrame) -> pd.DataFrame:
    """Each matched order's position, staked as the horse trader stakes (to win ``TO_WIN``) and closed by its exit:
    ``sp``, at the SP as the horse trader closes (a back's winnings laid at the SP; a lay closed by a back at the SP
    with its stake), settled by the result; ``last``, at the market in the runner's last book before the off (a back
    laid at the best lay price there, a lay backed at the best back price, both greened up: the same profit whatever
    the result), or at the SP where that book has no price; ``tp<k>``, a closing order ``k`` ticks better than the
    entry price, matched where a book after the fill reaches it (``exit_best``, from ``exit_paths``), the profit
    locked whatever the result; else as ``last``."""
    s = to.target_stake(o.price.to_numpy(float))
    m, bsp, won = o.price.to_numpy(float), o.bsp.to_numpy(float), o.won.to_numpy() == 1
    back = (o.side == "back").to_numpy()
    out_back = np.where(back, o.last_lay1.to_numpy(float), o.last_back1.to_numpy(float))
    at_market = (o["exit"] == "last").to_numpy() & np.isfinite(out_back) & (out_back > 1.0)
    ex = o["exit"].astype(str).to_numpy()
    tp = np.char.startswith(ex.astype(str), "tp")
    at_market = (tp | (ex == "last")) & np.isfinite(out_back) & (out_back > 1.0)
    sp_pnl = np.where(back, np.where(won, 0.0, -s + s * (m - 1.0) / (bsp - 1.0)), np.where(won, s * (bsp - m), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        mk_pnl = np.where(back, s * (m / out_back - 1.0), s * (1.0 - m / out_back))
        target = np.full(len(o), np.nan)
        for k in TP_TICKS:
            sel = ex == f"tp{k}"
            if sel.any():
                target[sel] = np.where(back[sel], ticks_from(m[sel], k, up=False), ticks_from(m[sel], k, up=True))
        eb = o["exit_best"].to_numpy(float) if "exit_best" in o else np.full(len(o), np.nan)
        took = tp & np.isfinite(target) & (target > 1.0) & np.where(back, eb <= target, eb >= target)
        tp_pnl = np.where(back, s * (m / target - 1.0), s * (1.0 - m / target))
    o = o.copy()
    o["stake"], o["target"] = s, s
    o["pnl"] = np.where(took, tp_pnl, np.where(at_market, mk_pnl, sp_pnl))
    o["took_profit"] = took
    o["risk"] = np.where(back, s, s * (m - 1.0))
    o["clv"] = np.where(back, m / bsp - 1.0, 1.0 - m / bsp)
    o["clv_value"] = o.clv * s
    o["held"] = np.where(back, np.where(won, s * (m - 1.0), -s), np.where(won, -s * (m - 1.0), s))
    o["closed_at_market"] = at_market | took
    return o


def orders(d: pd.DataFrame, books: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every order of every rule for the rows ``d`` (which carry ``f_bsp`` and the later books): its limit, matched or
    not by each fill rule, its gates as flags (``g_<gate>``), one row for each exit; settled by ``settle_orders``.
    The take-profit exits need the books' path (``books``, as ``load`` keeps them); without it, ``sp`` and ``last``
    only."""
    gates = gate_flags(d)
    cols = ["race_date", "market_id", "selection_id", "polled", "mark", "won", "bsp", "last_back1", "last_lay1"]
    base = d[[c for c in cols if c in d]].reset_index(drop=True)
    for g, f in gates.items():
        base[f"g_{g}"] = f
    rows = []
    for order, side, param, price in order_prices(d):
        ok = np.isfinite(price) & (price > 1.0) & (price <= to.MAX_PRICE)
        if not ok.any():
            continue
        for rule in FILLS:
            filled, matched_at, taken = fills(d, side, price, rule)
            rows.append(base[ok].assign(order=order, side=side, param=param, fill=rule, limit=price[ok],
                                        filled=filled[ok], price=matched_at[ok], taken=taken[ok]))
    if not rows:
        return pd.DataFrame()
    e = pd.concat(rows, ignore_index=True)
    use_path = books is not None and len(books) > 0 and {"selection_id", "polled"} <= set(e.columns)
    exits = EXITS if use_path else ("sp", "last")
    e["exit_best"] = exit_paths(e, books) if use_path else np.nan
    return settle_orders(pd.concat([e.assign(exit=ex) for ex in exits], ignore_index=True))


def rule_table(p: pd.DataFrame, keys=KEYS, days=None, n_boot: int = 2000) -> pd.DataFrame:
    """``greyhound.tradeout.rule_table``'s measures, grouped in one pass (that one scans every position once a rule,
    too slow for the thousand-odd rules here): each rule's trades, trades a day over every scored day, the return on
    the money at risk after commission on each market's net with its 90% interval (a bootstrap over every scored day,
    the days without a trade included), held to the result, the CLV of the prices against the SP."""
    from model.race_book import ratio_interval
    if p is None or not len(p):
        return pd.DataFrame()
    k = list(keys)
    all_days = sorted(set(days) if days is not None else set(p.race_date.astype(str)))
    m = to.settle(p, keys)
    m["race_date"] = m.race_date.astype(str)
    tot = m.groupby(k, dropna=False).agg(trades=("trades", "sum"), staked=("stake", "sum"), risk=("risk", "sum"),
                                         net=("net", "sum"), held_net=("held_net", "sum")).reset_index()
    tr = p.assign(_pos=(p.clv > 0).astype(float)).groupby(k, dropna=False).agg(
        clv_mean=("clv", "mean"), clv_pos=("_pos", "mean"), price_median=("price", "median")).reset_index()
    day = m.groupby(k + ["race_date"], dropna=False).agg(net=("net", "sum"), risk=("risk", "sum"))
    lo_hi, active = {}, {}
    for key, g in day.groupby(level=list(range(len(k))), dropna=False):
        g = g.droplevel(list(range(len(k)))).reindex(all_days, fill_value=0.0)
        lo, hi = ratio_interval(g.net.to_numpy(), g.risk.to_numpy(), n=n_boot)
        key = key if isinstance(key, tuple) else (key,)
        lo_hi[key] = [round(100 * lo, 2), round(100 * hi, 2)] if np.isfinite(lo) else None
        active[key] = int((g.risk > 0).sum())
    out = tot.merge(tr, on=k, how="left")
    keys_t = list(map(tuple, out[k].itertuples(index=False, name=None)))
    out["days"] = len(all_days)
    out["active_days"] = [active.get(t, 0) for t in keys_t]
    out["per_day"] = (out.trades / max(1, len(all_days))).round(1)
    out["roi%"] = (100 * out.net / out.risk).round(2)
    out["roi_90%"] = [lo_hi.get(t) for t in keys_t]
    out["held_roi%"] = (100 * out.held_net / out.risk).round(2)
    out["clv_mean%"] = (100 * out.pop("clv_mean")).round(2)
    out["clv_positive%"] = (100 * out.pop("clv_pos")).round(1)
    out["price_median"] = out.price_median.round(2)
    for c in ("staked", "risk", "net"):
        out[c] = out[c].round(2)
    return out.drop(columns=["held_net"])


def rule_summary(o: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """Each rule's orders placed, the share matched and taken at once, the CLV of those matched against the CLV the
    unmatched would have had at their limit (the adverse selection), and the matched positions' return settled by
    their exit with its 90% interval over every scored day (``greyhound.tradeout.rule_table``); one block a gate."""
    if not len(o):
        return pd.DataFrame()
    out = []
    for gate in GATES:
        col = f"g_{gate}"
        sub = o[o[col]] if col in o else (o if gate == "all" else o.iloc[0:0])
        if not len(sub):
            continue
        sub = sub.assign(gate=gate)
        placed = sub.groupby(list(KEYS)).agg(orders=("filled", "size"), filled_pct=("filled", "mean"),
                                             taken_pct=("taken", "mean"), _took=("took_profit", "sum"),
                                             _filled=("filled", "sum")).reset_index()
        placed["profit_taken%"] = (100 * placed.pop("_took") / placed.pop("_filled").replace(0, np.nan)).round(1)
        unm = sub[~sub.filled]
        unm = unm.assign(clv_if=np.where(unm.side == "back", unm.limit / unm.bsp - 1, 1 - unm.limit / unm.bsp))
        unm = unm.groupby(list(KEYS)).clv_if.mean().rename("clv_unmatched%").reset_index()
        f = sub[sub.filled]
        t = rule_table(f.assign(race_date=f.race_date.astype(str)), keys=KEYS, days=days) if len(f) else pd.DataFrame()
        blk = placed.merge(unm, on=list(KEYS), how="left")
        if len(t):
            blk = blk.merge(t.drop(columns=["fill%"], errors="ignore"), on=list(KEYS), how="left")
        out.append(blk)
    res = pd.concat(out, ignore_index=True)
    res["filled_pct"] = (100 * res.filled_pct).round(1)
    res["taken_pct"] = (100 * res.taken_pct).round(1)
    res["clv_unmatched%"] = (100 * res["clv_unmatched%"]).round(2)
    return res


ENTRY = [k for k in KEYS if k != "exit"]


def exit_compare(dev: pd.DataFrame, hold: pd.DataFrame | None) -> dict:
    """The exits set side by side on the same entries: for every entry rule (all keys but the exit) with a trade a day
    on the development days, each exit's return on the development days and the holdout, and for each exit against
    'last' (greened up at the last book) the median difference and the share of entry rules where it did better. The
    holdout here is read across every rule, so it describes; it does not select."""
    def wide(t):
        if t is None or not len(t) or "roi%" not in t:
            return pd.DataFrame()
        return t.pivot_table(index=ENTRY, columns="exit", values="roi%", aggfunc="first")
    dv = dev[dev.per_day >= 1] if len(dev) and "per_day" in dev else dev
    d, h = wide(dv), wide(hold)
    if not len(d):
        return {"rules": 0, "summary": [], "rows": []}
    h = h.reindex(d.index) if len(h) else pd.DataFrame(index=d.index, columns=d.columns)
    summary = []
    for ex in d.columns:
        if ex == "last" or "last" not in d.columns:
            continue
        row = {"exit": ex}
        for name, t in (("dev", d), ("holdout", h)):
            diff = (t[ex] - t["last"]).dropna() if ex in t and "last" in t else pd.Series(dtype=float)
            row[f"{name}_rules"] = int(len(diff))
            row[f"{name}_median_vs_last"] = round(float(diff.median()), 2) if len(diff) else None
            row[f"{name}_better%"] = round(100 * float((diff > 0).mean()), 1) if len(diff) else None
        summary.append(row)
    rows = d.add_prefix("dev_").join(h.add_prefix("hold_")).reset_index()
    return {"rules": int(len(d)), "summary": summary, "rows": _records(rows)}


def choose(dev: pd.DataFrame) -> pd.DataFrame:
    """The best rule of each family (mark, order, side, fill, exit) on the development days: the highest return among rules
    with a trade a day on average and positions on at least half the days; its param and gate fixed for the holdout."""
    if not len(dev) or "roi%" not in dev:
        return pd.DataFrame()
    d = dev[(dev.per_day >= 1) & (dev.active_days >= dev.days / 2)].dropna(subset=["roi%"])
    if not len(d):
        return pd.DataFrame()
    return d.sort_values("roi%", ascending=False).drop_duplicates(["mark", "order", "side", "fill", "exit"])


# --------------------------------------------------------------------------------------------------------------------
# The market's behaviour
# --------------------------------------------------------------------------------------------------------------------

def behaviour(panel: pd.DataFrame) -> dict:
    """How each runner's price moves from each mark to the SP: by price band (the mid now) and by the market's order
    (1 the favourite): the mean log(SP / mid), the share that drift (SP above the mid), the mean log(SP / best back)
    and log(best lay / SP), and the share whose best back price later reaches the SP before the off."""
    p = panel[panel["mark"].isin(ORDER_MARKS)].copy()
    p["band"] = pd.cut(p["mid"], BANDS, right=False).astype(str)
    p["order_now"] = p["rank"].clip(upper=6).astype(int).astype(str).replace({"6": "6+"})
    lay = p.lay1.where(p.lay1 > 1)
    p["x_mid"] = np.log(p.bsp / p["mid"])
    p["x_drift"] = (p.bsp > p["mid"]).astype(float)
    p["x_back"] = np.log(p.bsp / p.back1)
    p["x_lay"] = np.log(lay / p.bsp)
    p["x_reach"] = (p.fut_back_max >= p.bsp).astype(float)
    out = {}
    for by in ("band", "order_now"):
        t = p.groupby(["mark", by], observed=True).agg(
            runners=("x_mid", "size"), mean_log_sp_over_mid=("x_mid", "mean"), drift=("x_drift", "mean"),
            mean_log_sp_over_back=("x_back", "mean"), mean_log_lay_over_sp=("x_lay", "mean"),
            back_reaches_sp=("x_reach", "mean")).reset_index()
        for c in ("mean_log_sp_over_mid", "mean_log_sp_over_back", "mean_log_lay_over_sp"):
            t[c] = t[c].round(4)
        t["drift%"] = (100 * t.pop("drift")).round(1)
        t["back_reaches_sp%"] = (100 * t.pop("back_reaches_sp")).round(1)
        out[by] = json.loads(t.to_json(orient="records"))
    return out


# --------------------------------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------------------------------

def _tbl(rows: list[dict], cols: list[str]) -> list[str]:
    if not rows:
        return ["(none)"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    return out


RULE_COLS = ["mark", "order", "side", "param", "gate", "fill", "exit", "orders", "filled_pct", "taken_pct", "profit_taken%",
             "trades", "per_day",
             "roi%", "roi_90%", "clv_mean%", "clv_unmatched%", "held_roi%", "price_median"]


def report_md(res: dict) -> str:
    L = [f"# Greyhound limit-order study ({res['first_day']} to {res['last_day']})", "",
         f"Our price: {res['model']}. Books: {res['source']}. Panel: {res.get('panel_rows')} runner-marks; scored days "
         f"{res.get('scored_days')} (development {len(res.get('dev_days', []))}, holdout "
         f"{len(res.get('holdout_days', []))}).", "",
         "Orders rest in the book at their limit and lapse at the off unless matched. 'through': matched once a later "
         "book offers the other side at the limit (certain, late); 'traded': once a later traded price reaches it "
         "(the queue not modelled). clv_mean%: the matched orders' price against the SP; clv_unmatched%: what the "
         "unmatched would have had at their limit (the gap is the adverse selection). Returns settled by the result, "
         "the back's winnings laid at the SP, 2% commission on each market's net. exit 'sp': closed at the SP; "
         "'last': greened up at the market in the runner's last book before the off (lay to back, back to lay); "
         "'tp<k>': take profit, greened up as soon as a book after the fill offers k ticks better than the entry "
         "(profit_taken%: the share of matched orders that got there), else as 'last'. "
         "Gates: all, forecast to shorten / drift, the favourite now, short-priced (mid under 3), long (8 or more).",
         ""]
    L += ["## 1. The market's behaviour (all panel days)", "", "### By price band (the mid now)", ""]
    bc = ["mark", "band", "runners", "mean_log_sp_over_mid", "drift%", "mean_log_sp_over_back", "mean_log_lay_over_sp",
          "back_reaches_sp%"]
    L += _tbl(res["behaviour"]["band"], bc) + ["", "### By the market's order (1: the favourite now)", ""]
    L += _tbl(res["behaviour"]["order_now"], ["order_now" if c == "band" else c for c in bc]) + [""]
    L += ["## 2. Holdout, read once (each family's best rule on the development days)", ""]
    hc = res.get("holdout_count") or {}
    if hc:
        L += [f"{hc['rules']} rules read; {hc['lower_bound_above_zero']} with the 90% interval's lower bound above zero "
              f"(about {hc['expected_by_chance']} expected by chance alone at that level).", ""]
    L += _tbl(res["holdout"], RULE_COLS) + [""]
    xc = res.get("exit_compare") or {}
    if xc.get("summary"):
        L += ["## 2b. The exits on the same entries", "",
              f"{xc['rules']} entry rules with a trade a day on the development days; each exit's return against "
              "'last' (greened up at the last book) on those same entries. The holdout here is read across every "
              "rule: a description, not a selection.", ""]
        L += _tbl(xc["summary"], ["exit", "dev_rules", "dev_median_vs_last", "dev_better%", "holdout_rules",
                                  "holdout_median_vs_last", "holdout_better%"]) + [""]
        early = [r for r in xc["rows"] if r.get("mark") == "first_traded" and r.get("order") == "bid"
                 and r.get("side") == "lay" and r.get("fill") in ("traded", "strict")]
        if early:
            cols = list(ENTRY) + [f"{p}_{e}" for p in ("dev", "hold") for e in EXITS]
            L += ["The early lay's entries (first traded, lay at the forecast less the margin), by the traded fill and "
                  "by the strict one (a later trade beyond our price: the queue ahead certainly cleared):", ""]
            L += _tbl(early, [c for c in cols if c in early[0]]) + [""]
    L += ["## 3. The development days: the 40 best rules by return", ""]
    L += _tbl(res["dev_top"], RULE_COLS) + [""]
    L += ["## 4. Development days: each family's best rule", ""]
    L += _tbl(res["dev_chosen"], RULE_COLS) + [""]
    return "\n".join(L)


def _records(t: pd.DataFrame) -> list[dict]:
    if not len(t):
        return []
    return json.loads(t.to_json(orient="records"))


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--from", dest="first", default=str(to.FIRST_BOOK_DAY))
    ap.add_argument("--to", dest="last", default=None, help="default: yesterday (UK)")
    ap.add_argument("--source", choices=["recorder", "retimed", "live"], default="retimed")
    ap.add_argument("--pred-dir", default=None, help="our price from these day files instead of the 5 Oct model's")
    ap.add_argument("--out", default="out/greyhound_limit_study")
    a = ap.parse_args(argv)
    t0 = time.time()
    store = Store(root=a.store)
    last = date.fromisoformat(a.last) if a.last else uk_today() - timedelta(days=1)
    paths: list[pd.DataFrame] = []
    panel = load(store, date.fromisoformat(a.first), last, source=a.source, pred_dir=a.pred_dir, books_out=paths)
    books = pd.concat(paths, ignore_index=True) if paths else None
    res = {"first_day": a.first, "last_day": f"{last}", "source": a.source,
           "model": a.pred_dir or "the 5 Oct forward test's base model", "panel_rows": int(len(panel))}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(panel):
        days = sorted(panel.race_date.astype(str).unique())
        scored = days[MIN_TRAIN_DAYS:]
        hold = scored[-HOLDOUT_DAYS:] if len(scored) > HOLDOUT_DAYS + 3 else []
        dev = [d for d in scored if d not in hold]
        res.update({"scored_days": f"{scored[0]} to {scored[-1]}" if scored else None, "dev_days": dev,
                    "holdout_days": hold})
        res["behaviour"] = behaviour(panel)
        _log("forecasting the SP walk-forward")
        fc = forecast(panel)
        _log(f"{len(fc)} forecast rows; the orders")
        o = orders(fc, books) if len(fc) else pd.DataFrame(columns=["race_date"])
        o["race_date"] = o.race_date.astype(str)
        dev_t = rule_summary(o[o.race_date.isin(dev)], dev) if len(o) else pd.DataFrame()
        chosen = choose(dev_t)
        res["dev_top"] = _records(dev_t.sort_values("roi%", ascending=False).head(40)) if len(dev_t) else []
        res["dev_chosen"] = _records(chosen)
        res["dev_rules"] = _records(dev_t)
        ht_all = rule_summary(o[o.race_date.isin(hold)], hold) if hold and len(o) else None
        res["exit_compare"] = exit_compare(dev_t, ht_all)
        if hold and len(chosen):
            ho = o[o.race_date.isin(hold)]
            ht = rule_summary(ho, hold)
            ht = ht.merge(chosen[list(KEYS)], on=list(KEYS), how="inner")
            res["holdout"] = _records(ht)
            lo = ht["roi_90%"].map(lambda v: v[0] if isinstance(v, (list, tuple)) and len(v) else np.nan)
            res["holdout_count"] = {"rules": int(len(ht)), "lower_bound_above_zero": int((lo > 0).sum()),
                                    "expected_by_chance": round(0.05 * len(ht), 1)}
        else:
            res["holdout"] = []
    else:
        res.update({"note": "no panel", "behaviour": {"band": [], "order_now": []}, "holdout": [], "dev_top": [],
                    "dev_chosen": []})
    res["seconds"] = round(time.time() - t0)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    md = report_md(res)
    out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
