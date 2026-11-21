"""The take and close's SP forecast refitted on the live-key books (the owner, 18 Nov 2026: "refit the forecast on the
live books"; ledger greyhound-take-close-refit-1118, registered before the run).

On 18 Nov the live take and close backed dogs whose price was 8%+ over the forecast SP; after the stale-book fix
(PR 419) the SP still came in 14% over the forecast on the dogs it picked (ledger greyhound-take-close-fresh-paper-1118
amendment). The forecast in use (``greyhound.laytest.day_fit``) is fitted on the recorder's greyhound books - the
delayed key, a minute apart, re-timed - at each runner's first traded book, and applied live to the live key's book at
the poll where every runner has traded. This study fits the same conditional logit (``tradeout_study.fit_cl``, the
micro columns) on the live key's own books (greyhound/book_record.py, ``books_greyhound_live.csv``) at that same poll,
and scores both, walk-forward by day, on the live books:

- ``current``: the production fit, on the recorder's re-timed books at each runner's first traded book, every day
  before the scored day;
- ``refit``: the same model on the live-key books at the race-traded poll (the first poll at which every runner in
  the race has traded: the book the fixed live trader decides on), every live-key day before the scored day.

Our price, both arms, every day: the live recipe's published out-of-sample prices (``PRICES``, the first prefix holding
the day) where they exist, else the day's served predictions (``greyhound/live/predictions``) with the result from
the settled books. The SP and result are the recorder's settled books.

For each arm on each scored day: the cross-entropy of the SP book's shares, the bias of the forecast (mean
log(BSP / forecast)) over every runner and over the runners the rule takes, and the rule itself as the live trader
trades it (``greyhound.take_close.plan``: the best three backs while each is 8%+ over the forecast SP, up to GBP100,
closed by laying the winnings at the SP, 2% on each market's positive net). Intervals: 2,000 bootstrap draws over
days. Deflated Sharpe of the refit's per-market returns counting 240 trials.

    python -m research.greyhound.take_close_refit --out out/study
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-take-close-refit-1118"
PRICES = ("greyhound/live/oos_prices_stack_mlsitd_mem", "greyhound/live/oos_prices_stack_mlsitd",
          "greyhound/live/oos_prices_stack_mlsi_mem", "greyhound/live/oos_prices_stack_mlsi")
FIRST_LIVE_DAY = date(2026, 11, 3)
N_TRIALS = 240
MIN_TRAIN_DAYS = 7


# --------------------------------------------------------------------------------------------------------------------
# The data
# --------------------------------------------------------------------------------------------------------------------

def _winners(books: pd.DataFrame) -> pd.DataFrame:
    fin = books[books.source.astype(str) == "final"] if "source" in books else books.iloc[0:0]
    w = fin.assign(market_id=fin.market_id.astype(str), selection_id=pd.to_numeric(fin.selection_id, errors="coerce"))
    return w[w.runner_status.astype(str) == "WINNER"][["market_id", "selection_id"]].drop_duplicates()


def day_prices(store, d: date, books: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str]:
    """Our price for day ``d``: the first published out-of-sample prefix holding it, else the served predictions with
    the field and the result filled in from the settled books."""
    for prefix in PRICES:
        f = store.get_parquet(f"{prefix}/{d:%Y-%m-%d}.parquet")
        if f is not None and len(f):
            return f, prefix
    raw = store.get(f"greyhound/live/predictions/{d:%Y-%m-%d}.csv")
    if not raw or books is None:
        return None, "no prices"
    p = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    if not {"race_id", "market_id", "selection_id", "p_model"} <= set(p.columns):
        return None, "served predictions without market ids"
    p["selection_id"] = pd.to_numeric(p.selection_id, errors="coerce")
    p = p.dropna(subset=["selection_id", "market_id"]).drop_duplicates(["market_id", "selection_id"], keep="last")
    w = _winners(books)
    p = p.merge(w.assign(won=1), on=["market_id", "selection_id"], how="left")
    p["won"] = p.won.fillna(0).astype(int)
    p = p[p.market_id.astype(str).isin(set(w.market_id))]
    p["field"] = p.groupby("race_id").selection_id.transform("size")
    p["race_date"] = f"{d:%Y-%m-%d}"
    for c in ("race_time", "track", "trap"):
        if c not in p:
            p[c] = ""
    return p, "served predictions"


def race_traded(pre: pd.DataFrame) -> pd.DataFrame:
    """Each runner's book at its race's race-traded poll (the first poll at which every runner in the race has a
    trade: the latest of the runners' first traded polls), with its book 90 seconds or more before (``*_prev``)."""
    from greyhound import tradeout as to
    pre = pre.sort_values("polled")
    traded = pre[pre.last_traded > 1]
    if not len(traded):
        return pd.DataFrame()
    first = traded.groupby(["market_id", "selection_id"]).polled.min()
    t_star = first.groupby(level=0).max().rename("t_star")
    x = pre.merge(t_star, left_on="market_id", right_index=True)
    x = x[x.polled <= x.t_star].sort_values("polled").drop_duplicates(["market_id", "selection_id"], keep="last")
    x = x[x.last_traded > 1].assign(mark="race_traded")
    keep = ["market_id", "selection_id", "mark", "polled", "minutes_to_off"] + to.BOOK_COLS
    e = x[keep + [c for c in to.SP_COLS if c in x]]
    prev = pre[["market_id", "selection_id", "polled", "back1", "lay1"]].rename(
        columns={"polled": "polled_prev", "back1": "back1_prev", "lay1": "lay1_prev"}).sort_values("polled_prev")
    e = e.assign(_t=e.polled - pd.Timedelta(seconds=90)).sort_values("_t")
    return pd.merge_asof(e, prev, left_on="_t", right_on="polled_prev", by=["market_id", "selection_id"],
                         direction="backward").drop(columns="_t")


def panel_rows(pred: pd.DataFrame, entry: pd.DataFrame, closing: pd.DataFrame, markets: pd.DataFrame, d: date,
               live: bool) -> pd.DataFrame:
    """One day's rows: the live-key books at the race-traded poll (``live``), else the recorder's re-timed books at
    each runner's first traded book (the production fit's rows)."""
    from greyhound import tradeout as to
    if not live:
        x = to.day_panel(pred, entry, markets, d, closing_books=closing, retimed=True)
        return x[x["mark"] == "first_traded"] if len(x) else x
    p = pred.copy()
    p["market_id"] = p.market_id.astype(str)
    p["selection_id"] = to._num(p.selection_id)
    p = p[["race_id", "race_date", "race_time", "track", "trap", "field", "won", "p_model", "market_id",
           "selection_id"]]
    pre = to.pre_off_books(entry, markets, retimed=False)
    if not len(pre):
        return pd.DataFrame()
    e = race_traded(pre)
    if not len(e):
        return pd.DataFrame()
    bsp = to.closing_prices(closing, markets, d)
    x = p.merge(e, on=["market_id", "selection_id"], how="inner").merge(bsp, on=["market_id", "selection_id"],
                                                                       how="inner")
    x = x[(x.back1 > 1) & (x.bsp > 1)]
    whole = x.groupby("race_id").selection_id.transform("size") == x.field
    return to.add_features(x[whole].reset_index(drop=True))


def load(store, first: date, last: date) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    from betfair_recorder import _read_day_csv
    from greyhound import tradeout as to
    rec_rows, live_rows, notes, d = [], [], [], first
    while d <= last:
        try:
            books = _read_day_csv(d, "books_greyhound.csv")
            markets = _read_day_csv(d, "markets_greyhound.csv")
            pred, src = day_prices(store, d, books)
            if pred is None or books is None or markets is None:
                notes.append({"day": f"{d}", "note": f"no data ({src})"})
            else:
                r = panel_rows(pred, books, books, markets, d, live=False)
                lv = None
                if d >= FIRST_LIVE_DAY:
                    lb = _read_day_csv(d, to.ENTRY_BOOKS["live"])
                    lm = _read_day_csv(d, "markets_greyhound_live.csv")
                    if lb is not None:
                        lv = panel_rows(pred, lb, books, lm if lm is not None else markets, d, live=True)
                if len(r):
                    rec_rows.append(r)
                if lv is not None and len(lv):
                    live_rows.append(lv)
                notes.append({"day": f"{d}", "prices": src, "recorder_rows": int(len(r)),
                              "live_rows": int(len(lv)) if lv is not None else 0})
        except Exception as exc:
            notes.append({"day": f"{d}", "note": f"failed ({exc})"[:300]})
        log.info("%s", notes[-1])
        d += timedelta(days=1)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return cat(rec_rows), cat(live_rows), notes


# --------------------------------------------------------------------------------------------------------------------
# The scoring
# --------------------------------------------------------------------------------------------------------------------

def takes(te: pd.DataFrame, f_bsp: np.ndarray, comm: float, cfg: dict | None = None) -> pd.DataFrame:
    """The live rule on these rows: ``greyhound.take_close.plan`` on each runner's three best backs, closed at the SP;
    each market's net after commission on its positive net."""
    from greyhound.take_close import CONFIG, plan
    rows = []
    for (_, r), fb in zip(te.iterrows(), f_bsp):
        lv = [(float(r[f"back{i}"]), float(r[f"back{i}_size"])) for i in (1, 2, 3)
              if np.isfinite(r.get(f"back{i}", np.nan)) and np.isfinite(r.get(f"back{i}_size", np.nan))]
        pl = plan(lv, float(fb), cfg or CONFIG)
        if pl is None:
            continue
        s, p, b = pl["stake"], pl["avg"], float(r.bsp)
        pnl = 0.0 if r.won == 1 else -s + s * (p - 1.0) / (b - 1.0)
        rows.append({"race_date": str(r.race_date), "market_id": r.market_id, "selection_id": r.selection_id,
                     "stake": s, "price": p, "bsp": b, "f_bsp": float(fb), "won": int(r.won), "pnl": pnl,
                     "clv": p / b - 1.0})
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    m = t.groupby(["race_date", "market_id"], as_index=False).agg(pnl=("pnl", "sum"), stake=("stake", "sum"))
    m["net"] = m.pnl - comm * m.pnl.clip(lower=0)
    return t.merge(m[["market_id", "net"]].rename(columns={"net": "market_net"}), on="market_id", how="left")


def score(rec: pd.DataFrame, live: pd.DataFrame, comm: float = 0.02) -> dict:
    from model.race_book import ratio_interval
    from research.greyhound import tradeout_study as ts
    from research.overfit import dsr_check
    from research.greyhound.trial_count import greyhound_trials
    n_trials = greyhound_trials(N_TRIALS)          # every greyhound rule in the ledger, never fewer than this study's own
    from greyhound import tradeout as to
    from greyhound.early_lay import vol_in_fit
    live_days = sorted(live.race_date.astype(str).unique()) if len(live) else []
    out = {"live_days": len(live_days), "scored_days": [], "arms": {}}
    per = {"current": [], "refit": []}
    acc = {"current": [], "refit": []}
    bias = {"current": [], "refit": []}
    for i, day in enumerate(live_days):
        if i < MIN_TRAIN_DAYS:
            continue
        te = live[live.race_date.astype(str) == day]
        tr_live = live[live.race_date.astype(str) < day]
        tr_rec = rec[rec.race_date.astype(str) < day] if len(rec) else rec
        if not len(te) or not len(tr_live) or not len(tr_rec):
            continue
        out["scored_days"].append(day)
        for arm, tr in (("current", tr_rec), ("refit", tr_live)):
            f = ts.fit_cl(tr, ts.CL_SETS["micro"])
            # as the live trader forecasts (take_close.forecast): a fit that never saw volume reads none
            x = te if vol_in_fit(f) else to.add_features(te.assign(runner_total_matched=np.nan))
            q, book = ts.predict(x, f)
            fb = 1.0 / (book * np.asarray(q, float))
            acc[arm].append(ts.accuracy(x, np.asarray(q, float), book))
            bias[arm].append(pd.DataFrame({"race_date": day, "lerr": np.log(x.bsp.to_numpy(float) / fb)}))
            t = takes(x, fb, comm)
            if len(t):
                per[arm].append(t)
    days = out["scored_days"]
    for arm in ("current", "refit"):
        a = acc[arm]
        res = {"rows": int(sum(r["rows"] for r in a)), "races": int(sum(r["races"] for r in a))}
        if a:
            n = sum(r["races"] for r in a)
            res["xent"] = round(sum(r["xent"] * r["races"] for r in a) / n, 5)
            res["xent_market"] = round(sum(r["xent_market"] * r["races"] for r in a) / n, 5)
        b = pd.concat(bias[arm], ignore_index=True) if bias[arm] else pd.DataFrame(columns=["lerr"])
        res["bias_all_%"] = round(100 * float(b.lerr.mean()), 2) if len(b) else None
        t = pd.concat(per[arm], ignore_index=True) if per[arm] else pd.DataFrame()
        if len(t):
            t["clv_value"] = t.clv * t.stake
            d_ = t.groupby("race_date").agg(stake=("stake", "sum"), clv_value=("clv_value", "sum"),
                                            pnl=("pnl", "sum")).reindex(days, fill_value=0.0)
            mk = t.drop_duplicates("market_id")
            dn = mk.groupby("race_date").market_net.sum().reindex(days, fill_value=0.0)
            lo, hi = ratio_interval(d_.clv_value.to_numpy(), d_.stake.to_numpy(), n=2000)
            nlo, nhi = ratio_interval(dn.to_numpy(), d_.stake.to_numpy(), n=2000)
            mret = (mk.set_index("market_id").market_net /
                    t.groupby("market_id").stake.sum()).to_numpy(float)
            res.update({
                "backs": int(len(t)), "markets": int(t.market_id.nunique()), "staked": round(float(t.stake.sum()), 2),
                "drifted": int((t.bsp > t.price).sum()), "shortened": int((t.bsp < t.price).sum()),
                "clv_by_stake_%": round(100 * float(t.clv_value.sum() / t.stake.sum()), 2),
                "clv_90%": [round(100 * lo, 2), round(100 * hi, 2)] if np.isfinite(lo) else None,
                "bias_taken_%": round(100 * float(np.log(t.bsp / t.f_bsp).mean()), 2),
                "net": round(float(mk.market_net.sum()), 2),
                "roi_%": round(100 * float(mk.market_net.sum() / t.stake.sum()), 2),
                "roi_90%": [round(100 * nlo, 2), round(100 * nhi, 2)] if np.isfinite(nlo) else None,
                "dsr": dsr_check(mret, n_trials=n_trials)})
        else:
            res["backs"] = 0
        out["arms"][arm] = res
    r = out["arms"].get("refit", {})
    ci = r.get("clv_90%")
    out["reading"] = {
        "clv_interval_above_zero": bool(ci and ci[0] > 0),
        "taken_unbiased": bool(r.get("bias_taken_%") is not None and abs(r["bias_taken_%"]) <= 3.0),
        "net_positive": bool((r.get("net") or 0) > 0),
        "dsr_passed": bool((r.get("dsr") or {}).get("passed")),
        "min_backs_met": bool((r.get("backs") or 0) >= 100)}
    out["reading"]["pass"] = all(out["reading"].values())
    return out


def main(argv=None) -> int:
    from sources.common import Store
    from greyhound import tradeout as to
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="first", default=str(to.FIRST_BOOK_DAY))
    ap.add_argument("--to", dest="last", default=str(date(2026, 11, 17)))
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/take_close_refit")
    a = ap.parse_args(argv)
    store = Store(root=a.store) if a.store else Store()
    rec, live, notes = load(store, date.fromisoformat(a.first), date.fromisoformat(a.last))
    res = {"tag": TAG, "notes": notes, "recorder_rows": int(len(rec)), "live_rows": int(len(live))}
    res.update(score(rec, live) if len(live) and len(rec) else {"note": "no data found: no live or recorder rows"})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "notes"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
