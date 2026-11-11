"""Where the greyhound edge is, and whether it can be traded (the owner, 11 Nov 2026: "run thorough investigation, QA,
cleaning. Exhaustive to identify how to make this a profitable edge. Or other profitable trading strategies. Be
bold, be visionary, but be hyper granular"). Read-only: S3 in, a report out; no Betfair call.

The day diagnostic of 11 Nov (reports/greyhound_model.md, section 42) found the early lay's forecast right about
which dogs drift (+24% from the lay price to the SP) and its fills wrong: matched where the dog did not drift. So
this study prices every entry at what the book **offers**, not at the mid or at a resting price:

- **taking**: a lay taken at the best price available to lay (``lay1``, the backers' offer) or a back taken at
  ``back1``, for no more than the money shown there, closed at the Betfair SP (an SP order always fills);
- at seven marks before the scheduled off (600, 300, 180, 120, 60, 30 and 15 seconds), on the live-key books
  (greyhound/book_record.py: three levels a side, every 60 s from 15 minutes out and every 5 s in the last 5);
- with no signal, with our price (``pred_bsp``), with Betfair's own projected SP (``sp_near``, ``sp_far``) and with
  both agreeing, each at five margins and in four price bands.

Every cell: bets, races, the mean return a pound staked net of commission, its interval by resampling days, the days
in profit, the money on offer. The cells are chosen on the first days (development) and read on the last (holdout),
with the false discovery rate over every cell tried. Nothing here is a pass: a rule it points to is registered and
run forward on paper before a penny is sent.

QA first: the books' coverage and age at each mark, the spread, late offs, missing SPs; where the SP ends against
the book (mid, back1, lay1) by mark, price and spread; and the live early lay's own orders against the book they were
sent into (crossing, inside the spread, joining the queue), with each class's fill rate and drift.

    python -m research.greyhound.edge_anatomy --from 2026-11-03 --out out/greyhound_edge
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.greyhound import day_diagnostic as dd

COMM = 0.02
MARKS = (600, 300, 180, 120, 60, 30, 15)
MAX_AGE = 75.0                     # a book this many seconds older than its mark is not that mark's book
THRESHOLDS = (0.0, 0.05, 0.10, 0.20, 0.30)
BANDS = ((1.01, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 1001.0))
STAKE = 10.0                       # a taker's stake is the money on offer, at most this (GBP)
BOOT = 400
MIN_BETS = 100
num = dd.num


def _net(g: np.ndarray) -> np.ndarray:
    return np.where(g > 0, g * (1 - COMM), g)


def day_marks(books: pd.DataFrame, markets: pd.DataFrame, close: pd.DataFrame, preds: pd.DataFrame | None,
              day: date) -> tuple[pd.DataFrame, dict]:
    """One day's WIN runners at each mark: the book offered there, the projected SP, the SP, the result, our price."""
    mk = markets.assign(market_id=markets.market_id.astype(str))
    if "market_type" in mk:
        mk = mk[mk.market_type.astype(str) == "WIN"]
    start = pd.to_datetime(mk.drop_duplicates("market_id").set_index("market_id").market_start_utc, utc=True,
                           errors="coerce")
    venue = mk.drop_duplicates("market_id").set_index("market_id").venue if "venue" in mk else None
    b = books[books.market_id.astype(str).isin(start.index)].copy()
    b["market_id"] = b.market_id.astype(str)
    b["selection_id"] = num(b.selection_id)
    b["polled"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
    b["secs"] = (b.market_id.map(start) - b.polled).dt.total_seconds()
    inplay = b.inplay.astype(str).str.lower().isin(["true", "1"]) if "inplay" in b else False
    went = b[inplay].groupby("market_id").polled.min()
    pre = b[(~inplay) & (b.status.astype(str) == "OPEN") & (b.source.astype(str) != "final")]
    for c in ("back1", "back1_size", "back2", "back2_size", "lay1", "lay1_size", "lay2", "lay2_size", "last_traded",
              "runner_total_matched", "market_total_matched", "sp_near", "sp_far"):
        pre[c] = num(pre[c]) if c in pre else np.nan
    qa = {"day": f"{day}", "win_markets": int(len(start)), "pre_off_rows": int(len(pre)),
          "markets_with_books": int(pre.market_id.nunique()),
          "late_off_median_s": round(float((went - start.reindex(went.index)).dt.total_seconds().median()), 1)
          if len(went) else None,
          "late_off_p90_s": round(float((went - start.reindex(went.index)).dt.total_seconds().quantile(0.9)), 1)
          if len(went) else None}
    out = []
    for m in MARKS:
        w = pre[(pre.secs >= m) & (pre.secs <= m + MAX_AGE)]
        last = w.sort_values("secs").groupby(["market_id", "selection_id"], sort=False).head(1)
        out.append(last.assign(mark=m, age=last.secs - m))
    d = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not len(d):
        return d, qa
    d = d.merge(close.assign(market_id=close.market_id.astype(str)), on=["market_id", "selection_id"], how="left")
    if preds is not None and len(preds):
        p = preds.assign(market_id=preds.market_id.astype(str), selection_id=num(preds.selection_id))
        keep = [c for c in ("market_id", "selection_id", "pred_bsp", "p_model", "track", "race_time", "trap",
                            "model_objective") if c in p]
        d = d.merge(p[keep].drop_duplicates(["market_id", "selection_id"]), on=["market_id", "selection_id"],
                    how="left")
    for c in ("pred_bsp", "p_model", "track", "trap"):
        if c not in d:
            d[c] = np.nan
    if venue is not None:
        d["track"] = d.track.where(d.track.notna(), d.market_id.map(venue))
    d["day"] = f"{day}"
    d["actual_secs"] = d.secs + d.market_id.map((went - start.reindex(went.index)).dt.total_seconds()).fillna(0.0)
    d["mid"] = np.sqrt(d.back1 * d.lay1)
    d["spread"] = np.log(d.lay1 / d.back1)
    qa["runners_at_60s"] = int((d.mark == 60).sum())
    qa["with_sp_at_60s"] = int(((d.mark == 60) & d.bsp.notna()).sum())
    qa["priced_by_us_at_60s"] = int(((d.mark == 60) & d.pred_bsp.notna()).sum())
    return d, qa


def load(store, first: date, last: date, s3=None) -> tuple[pd.DataFrame, list[dict]]:
    from betfair_recorder import _read_day_csv
    from greyhound import fav_back as fb
    parts, qas = [], []
    day = first
    while day <= last:
        books = _read_day_csv(day, "books_greyhound_live.csv", s3=s3)
        markets = _read_day_csv(day, "markets_greyhound_live.csv", s3=s3)
        if books is None or markets is None or not len(books):
            qas.append({"day": f"{day}", "note": "no live-key books"})
        else:
            close = fb.closing(day, s3=s3)
            preds = dd._csv(store, f"greyhound/live/predictions/{day:%Y-%m-%d}.csv")
            d, qa = day_marks(books, markets, close, preds, day)
            qa["model"] = str(preds.model_objective.iloc[0]) if preds is not None and "model_objective" in preds \
                and len(preds) else None
            qas.append(qa)
            if len(d):
                parts.append(d)
            print(json.dumps(qa), flush=True)
        day += timedelta(days=1)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), qas


# ---- the entries -----------------------------------------------------------------------------------------------------

def entries(d: pd.DataFrame) -> pd.DataFrame:
    """Each runner at each mark as a taken lay and a taken back, closed at the SP: the return a pound staked net of
    commission, the stake the book allows, and the signals (log margins) that would have chosen it."""
    ok = d.bsp.notna() & (d.bsp > 1) & (d.back1 > 1) & (d.lay1 > 1) & (d.lay1 >= d.back1)
    d = d[ok]
    f, near, far = num(d.pred_bsp), d.sp_near.where(d.sp_near > 1), d.sp_far.where(d.sp_far > 1)
    base = d[["day", "market_id", "selection_id", "mark", "track", "trap", "spread", "bsp", "won"]]
    lay = base.assign(side="lay", price=d.lay1, size=d.lay1_size, g=1 - d.lay1 / d.bsp,
                      s_model=np.log(f / d.lay1), s_near=np.log(near / d.lay1), s_far=np.log(far / d.lay1))
    back = base.assign(side="back", price=d.back1, size=d.back1_size, g=d.back1 / d.bsp - 1,
                       s_model=np.log(d.back1 / f), s_near=np.log(d.back1 / near), s_far=np.log(d.back1 / far))
    e = pd.concat([lay, back], ignore_index=True)
    e["net"] = _net(e.g.to_numpy())
    e["stake"] = np.minimum(num(e["size"]).fillna(0.0), STAKE)
    e["s_both"] = np.minimum(e.s_model, e.s_near)
    e["band"] = pd.cut(e.price, [b[0] for b in BANDS] + [BANDS[-1][1]], right=False,
                       labels=[f"{a:g}-{b:g}" if b < 1000 else f"{a:g}+" for a, b in BANDS]).astype(str)
    return e


def boot(day_sum: np.ndarray, day_n: np.ndarray, rng) -> tuple[float, float, float]:
    """The interval of sum / n over resampled days, and the share of resamples at or under 0."""
    k = len(day_sum)
    if k < 2 or day_n.sum() == 0:
        return np.nan, np.nan, np.nan
    ix = rng.integers(0, k, size=(BOOT, k))
    m = day_sum[ix].sum(axis=1) / np.maximum(day_n[ix].sum(axis=1), 1e-9)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95)), float((m <= 0).mean())


def cell(x: pd.DataFrame, days: list[str], rng) -> dict:
    """A rule's bets: the return a pound staked (stake-weighted: the money the book offered, capped), by day."""
    if not len(x):
        return {"bets": 0}
    w = x.stake.to_numpy()
    g = x.groupby("day")
    ds = (x.net * x.stake).groupby(x.day).sum().reindex(days).fillna(0.0).to_numpy()
    dn = x.stake.groupby(x.day).sum().reindex(days).fillna(0.0).to_numpy()
    lo, hi, p = boot(ds, dn, rng)
    return {"bets": int(len(x)), "races": int(x.market_id.nunique()), "days": int(g.ngroups),
            "ret%": round(100 * float((x.net * w).sum() / max(w.sum(), 1e-9)), 2),
            "lo%": round(100 * lo, 2) if np.isfinite(lo) else None,
            "hi%": round(100 * hi, 2) if np.isfinite(hi) else None, "p": round(p, 4) if np.isfinite(p) else None,
            "days_up": int((ds > 0).sum()), "staked": round(float(w.sum()), 0),
            "pnl": round(float((x.net * w).sum()), 2), "ret_flat%": round(100 * float(x.net.mean()), 2),
            "mean_size": round(float(num(x["size"]).mean()), 1)}


def grid(e: pd.DataFrame, days: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    e = e[e.day.isin(days) & (e.stake > 0)]
    for side in ("lay", "back"):
        for mark in MARKS:
            base = e[(e.side == side) & (e.mark == mark)]
            for band in ["all"] + sorted(base.band.unique()):
                xb = base if band == "all" else base[base.band == band]
                rows.append({"side": side, "mark": mark, "band": band, "signal": "none", "thr": None,
                             **cell(xb, days, rng)})
                for sig in ("s_model", "s_near", "s_far", "s_both"):
                    for thr in THRESHOLDS:
                        rows.append({"side": side, "mark": mark, "band": band, "signal": sig[2:], "thr": thr,
                                     **cell(xb[xb[sig] >= thr], days, rng)})
    return pd.DataFrame(rows)


def bh(p: pd.Series, q: float = 0.10) -> pd.Series:
    ok = p.notna()
    out = pd.Series(False, index=p.index)
    s = p[ok].sort_values()
    if not len(s):
        return out
    k = np.arange(1, len(s) + 1)
    passed = s.to_numpy() <= q * k / len(s)
    if passed.any():
        out[s.index[: np.flatnonzero(passed).max() + 1]] = True
    return out


# ---- QA and where the SP ends ----------------------------------------------------------------------------------------

def sp_against_book(d: pd.DataFrame) -> pd.DataFrame:
    x = d[d.bsp.notna() & (d.back1 > 1) & (d.lay1 >= d.back1)].copy()
    x["band"] = pd.cut(x.mid, [1.01, 3, 6, 12, 1001], right=False, labels=["<3", "3-6", "6-12", "12+"])
    x["spread_band"] = pd.cut(x.spread, [-0.001, 0.03, 0.07, 0.15, 9], labels=["<3%", "3-7%", "7-15%", "15%+"])
    g = x.groupby(["mark", "band", "spread_band"], observed=True)
    t = pd.DataFrame({"runners": g.size(), "sp_vs_mid": g.apply(lambda q: np.log(q.bsp / q.mid).mean()),
                      "sp_vs_back1": g.apply(lambda q: np.log(q.bsp / q.back1).mean()),
                      "sp_vs_lay1": g.apply(lambda q: np.log(q.bsp / q.lay1).mean()),
                      "sp_above_lay1%": g.apply(lambda q: 100 * (q.bsp > q.lay1).mean()),
                      "sp_below_back1%": g.apply(lambda q: 100 * (q.bsp < q.back1).mean()),
                      "near_vs_sp_rmse": g.apply(lambda q: np.sqrt((np.log(q.sp_near.where(q.sp_near > 1) / q.bsp) ** 2)
                                                                   .mean())),
                      "mid_vs_sp_rmse": g.apply(lambda q: np.sqrt((np.log(q.mid / q.bsp) ** 2).mean())),
                      "lay1_size": g.lay1_size.median(), "back1_size": g.back1_size.median()}).reset_index()
    return t


def book_qa(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("mark")
    return pd.DataFrame({"runners": g.size(), "markets": g.market_id.nunique(), "age_median_s": g.age.median(),
                         "both_sides%": g.apply(lambda q: 100 * ((q.back1 > 1) & (q.lay1 > 1)).mean()),
                         "spread_median%": g.apply(lambda q: 100 * q.spread.median()),
                         "spread_p90%": g.apply(lambda q: 100 * q.spread.quantile(0.9)),
                         "crossed%": g.apply(lambda q: 100 * (q.lay1 < q.back1).mean()),
                         "has_near%": g.apply(lambda q: 100 * (q.sp_near > 1).mean()),
                         "has_sp%": g.apply(lambda q: 100 * q.bsp.notna().mean()),
                         "priced_by_us%": g.apply(lambda q: 100 * q.pred_bsp.notna().mean()),
                         "matched_median": g.market_total_matched.median(),
                         "lay1_size_median": g.lay1_size.median(), "back1_size_median": g.back1_size.median(),
                         "model_rmse": g.apply(lambda q: np.sqrt((np.log(num(q.pred_bsp) / q.bsp) ** 2).mean())),
                         "mid_rmse": g.apply(lambda q: np.sqrt((np.log(q.mid / q.bsp) ** 2).mean())),
                         "near_rmse": g.apply(lambda q: np.sqrt((np.log(q.sp_near.where(q.sp_near > 1) / q.bsp) ** 2)
                                                                .mean()))}).reset_index()


def live_orders(store, days: list[date], s3=None) -> pd.DataFrame:
    """Every lay the live early lay sent, against the book it was sent into (the last read at or before it, at most
    15 s old): crossing (at or above lay1), inside the spread, at back1, or behind it."""
    from betfair_recorder import _read_day_csv
    from greyhound import early_lay as el, fav_back as fb
    out = []
    for day in days:
        led = dd._csv(store, f"{el.PREFIX}/live/{day:%Y-%m-%d}/ledger.csv")
        if led is None or not (led.action == "lay").any():
            continue
        lays = dd.lay_frame(led, fb.closing(day, s3=s3))
        books = _read_day_csv(day, "books_greyhound_live.csv", s3=s3)
        if books is None or not len(lays):
            continue
        b = books.assign(market_id=books.market_id.astype(str), selection_id=num(books.selection_id),
                         polled=pd.to_datetime(books.polled_utc, utc=True, errors="coerce"))
        b = b[b.market_id.isin(set(lays.market_id))].dropna(subset=["polled"]).sort_values("polled")
        lays = lays.assign(placed=pd.to_datetime(lays.time_utc, utc=True, errors="coerce")).dropna(subset=["placed"])
        j = pd.merge_asof(lays.sort_values("placed"), b[["polled", "market_id", "selection_id", "back1", "back1_size",
                                                         "lay1", "lay1_size", "sp_near"]],
                          left_on="placed", right_on="polled", by=["market_id", "selection_id"],
                          tolerance=pd.Timedelta(seconds=15))
        for c in ("back1", "lay1", "back1_size", "lay1_size", "sp_near"):
            j[c] = num(j[c])
        j["where"] = np.select([j.back1.isna() | j.lay1.isna(), j.price >= j.lay1, j.price > j.back1,
                                np.isclose(j.price, j.back1)],
                               ["no book within 15 s", "crossing (at or above lay1)", "inside the spread",
                                "at back1 (joins the queue)"], "below back1 (behind the queue)")
        j["take_now"] = 1 - j.lay1 / j.bsp                       # had it been taken at lay1 instead
        j["day"] = f"{day}"
        out.append(j)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def live_orders_table(j: pd.DataFrame) -> pd.DataFrame:
    g = j.groupby("where")
    m = j[j.is_matched].groupby("where")
    return pd.DataFrame({"lays": g.size(), "matched": g.is_matched.sum(), "fill%": 100 * g.is_matched.mean(),
                         "liab_matched": g.liab_matched.sum(), "locked": g.locked.sum(),
                         "drift_all": g.drift_log.mean(), "drift_matched": m.drift_log.mean(),
                         "log(lay1/price)": g.apply(lambda q: np.log(q.lay1 / q.price).mean()),
                         "log(sp/lay1)": g.apply(lambda q: np.log(q.bsp / q.lay1).mean()),
                         "taken_at_lay1_ret%": g.apply(lambda q: 100 * _net(q.take_now.dropna().to_numpy()).mean()
                                                       if q.take_now.notna().any() else np.nan),
                         "lay1_size": g.lay1_size.median()}).reset_index()


# ---- the report ------------------------------------------------------------------------------------------------------

def run(store, first: date, last: date, out: Path, holdout_days: int = 3, s3=None) -> dict:
    d, qas = load(store, first, last, s3=s3)
    res = {"from": f"{first}", "to": f"{last}", "read_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "days": qas}
    L = [f"# racing² - greyhound edge anatomy, {first} to {last} (read {res['read_utc']})", "",
         "Read-only, from the live-key books in S3. A taken lay or back is priced at what the book offered, staked "
         f"at the money on offer (at most GBP{STAKE:g}) and closed at the Betfair SP, net of {COMM:.0%} commission. "
         "Nothing here is a pass: see the end.", "", "## 1. The days read", ""] + dd._md(pd.DataFrame(qas))
    out.parent.mkdir(parents=True, exist_ok=True)
    if not len(d):
        out.with_suffix(".md").write_text("\n".join(L + ["", "no data found"]) + "\n")
        out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
        return res
    L += ["", "## 2. The books at each mark (QA)", "",
          "age: seconds the book is older than its mark; rmse: of log(price / SP) for our price, the mid and "
          "Betfair's projected SP (near).", ""] + dd._md(book_qa(d), 3)
    L += ["", "## 3. Where the SP ends against the book", "",
          "log(SP / price), by mark, the mid's price band and the spread. sp_vs_lay1 above 0: the SP ended beyond "
          "the price a lay could be taken at (a taken lay, closed at the SP, wins before commission).", ""]
    spb = sp_against_book(d)
    L += dd._md(spb[spb.runners >= 150], 3)
    e = entries(d)
    days = sorted(e.day.unique())
    hold = days[-holdout_days:] if len(days) > holdout_days + 1 else []
    dev = [x for x in days if x not in hold]
    res["development_days"], res["holdout_days"] = dev, hold
    g_dev = grid(e, dev)
    g_all = grid(e, days, seed=1)
    k = ["side", "mark", "band", "signal", "thr"]
    g_dev["thr"], g_all["thr"] = g_dev.thr.astype(object), g_all.thr.astype(object)
    tried = g_dev[g_dev.bets >= MIN_BETS].copy()
    tried["fdr"] = bh(tried.p)
    res["cells_tried"], res["cells_with_100_bets"] = int(len(g_dev)), int(len(tried))
    res["expected_by_chance_at_5%"] = round(0.05 * len(tried), 1)
    cand = tried[(tried["lo%"] > 0)].sort_values("ret%", ascending=False)
    res["development_cells_with_interval_above_0"] = int(len(cand))
    res["of_them_passing_fdr_10%"] = int(cand.fdr.sum())
    L += ["", "## 4. Taking at the offered price, closed at the SP: every cell", "",
          f"Development days {dev[0] if dev else '-'} to {dev[-1] if dev else '-'} ({len(dev)}); holdout "
          f"{', '.join(hold) if hold else 'none (too few days)'}. Cells tried {len(g_dev)}, with {MIN_BETS} bets or "
          f"more {len(tried)}; with a 90% interval above 0 in development {len(cand)} (about "
          f"{res['expected_by_chance_at_5%']} expected by chance), {int(cand.fdr.sum())} within a 10% false "
          "discovery rate.", "", "**No signal: the whole book taken, by mark (all days)**", ""]
    none = g_all[(g_all.signal == "none")]
    L += dd._md(none[none.band == "all"].drop(columns=["signal", "thr", "band"]), 2)
    L += ["", "**No signal, by price band (all days)**", ""] + dd._md(
        none[none.band != "all"].drop(columns=["signal", "thr"]), 2)
    if hold:
        g_hold = grid(e, hold, seed=2)
        g_hold["thr"] = g_hold.thr.astype(object)
        j = cand.merge(g_hold, on=k, how="left", suffixes=("_dev", "_hold"))
        cols = k + ["bets_dev", "ret%_dev", "lo%_dev", "hi%_dev", "days_up_dev", "fdr", "bets_hold", "ret%_hold",
                    "lo%_hold", "hi%_hold", "days_up_hold", "pnl_hold", "staked_hold", "mean_size_hold"]
        L += ["", "**The development's candidates, read on the holdout** (best 60 by development return)", ""] + \
            dd._md(j[cols].head(60), 2)
        res["candidates_positive_on_holdout"] = int((j["ret%_hold"] > 0).sum())
        res["candidates_with_holdout_interval_above_0"] = int((j["lo%_hold"] > 0).sum())
        j.to_csv(out.parent / f"{out.name}_candidates.csv", index=False)
    else:
        L += ["", "**The best cells (all days; no holdout: describes, does not select)**", ""] + dd._md(
            g_all[g_all.bets >= MIN_BETS].sort_values("ret%", ascending=False).head(60), 2)
    for sig in ("model", "near", "both"):
        x = g_all[(g_all.signal == sig) & (g_all.band == "all")]
        L += ["", f"**Signal: {sig}, all prices, by mark and margin (all days)**", ""] + dd._md(
            x.drop(columns=["signal", "band"]), 2)
    g_all.to_csv(out.parent / f"{out.name}_grid_all.csv", index=False)
    g_dev.to_csv(out.parent / f"{out.name}_grid_dev.csv", index=False)
    try:
        lo = live_orders(store, [date.fromisoformat(x) for x in days], s3=s3)
        if len(lo):
            L += ["", "## 5. The live early lay's own orders against the book they were sent into", "",
                  "drift: log(SP / our lay price); log(sp/lay1): what taking at the offered price would have left; "
                  "taken_at_lay1_ret%: the return a pound of lay stake had the lay been taken there, net.", ""] + \
                dd._md(live_orders_table(lo), 3)
            L += ["", "By day:", ""] + dd._md(lo.groupby("day").apply(lambda q: pd.Series({
                "lays": len(q), "matched": int(q.is_matched.sum()), "locked": q.locked.sum(),
                "drift_matched": q[q.is_matched].drift_log.mean(), "drift_unmatched": q[~q.is_matched].drift_log.mean(),
                "log(sp/lay1)_unmatched": np.log(q[~q.is_matched].bsp / q[~q.is_matched].lay1).mean()})).reset_index(), 3)
            lo.to_csv(out.parent / f"{out.name}_live_orders.csv", index=False)
            res["live_orders"] = int(len(lo))
        else:
            L += ["", "## 5. The live early lay's own orders", "", "no data found"]
    except Exception as exc:
        L += ["", f"## 5. The live orders: failed ({str(exc)[:200]})"]
    L += ["", "## 6. How to read this", "",
          "A cell is a description of the days read. A rule is traded only after it is registered in the ledger and "
          "run forward on paper on days it has not seen, at the book's own sizes."]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main(argv=None) -> int:
    from zoneinfo import ZoneInfo
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="first", default="2026-11-03")
    ap.add_argument("--to", dest="last")
    ap.add_argument("--holdout-days", type=int, default=3)
    ap.add_argument("--out", default="out/greyhound_edge")
    a = ap.parse_args(argv)
    last = date.fromisoformat(a.last) if a.last else datetime.now(ZoneInfo("Europe/London")).date()
    res = run(Store(), date.fromisoformat(a.first), last, Path(a.out), holdout_days=a.holdout_days)
    print(json.dumps({k: v for k, v in res.items() if k != "days"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
