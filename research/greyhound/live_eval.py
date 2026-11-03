"""Every strategy on the live books (the owner, 3 Nov 2026: "now we have live data. We evaluate all of our possible
strategies, on how they would have done on live books").

The study so far (research/greyhound/limit_study.py) scored every order on the recorder's books: polled a minute
apart, about 90 seconds stale. The first live trades showed the trader sees other books, so the study is run again
here on the books the live trader reads: the live-key record (greyhound/book_record.py, ``books_greyhound_live.csv``,
every 60 s from 15 minutes out and every 5 s in the last 5), with the BSP and the result from its settled reads.

- Our price: the live model's (``greyhound/live/predictions/<day>.csv``), the one the live trader uses.
- The forecast of the SP at each mark: the conditional logit (``micro``) fitted on every recorded day before each
  scored day, on the recorder's books, as the live trader's forecast is fitted, and once enough live-key days exist,
  on the live-key days before it too (``--fit-on live``).
- Every order of the study: ask / bid at the forecast +- a margin, join, ticks, take; each gate; each fill (through,
  traded, strict); each exit (sp, last, take profit), settled as the study settles them.

A day scored alone describes; it does not select. With several days, the rules' intervals are over days.

    python -m research.greyhound.live_eval --from 2026-11-03 --to 2026-11-03 --out out/greyhound_live_eval
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

from greyhound import tradeout as to
from research.greyhound import limit_study as ls, tradeout_study as ts

log = logging.getLogger(__name__)


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


def live_panel(store, day: date) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """One day's runners at each entry mark on the live-key books, with our live price, the BSP and the result from
    the live-key settled reads, the later books attached; and the pre-off books themselves (the path)."""
    from betfair_recorder import _read_day_csv
    books = _read_day_csv(day, to.ENTRY_BOOKS["live"])
    markets = _read_day_csv(day, "markets_greyhound_live.csv")
    raw = store.get(f"greyhound/live/predictions/{day:%Y-%m-%d}.csv")
    if books is None or markets is None or not raw:
        _log(f"{day}: no live-key books or no prices")
        return pd.DataFrame(), None
    pred = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    pred["selection_id"] = pd.to_numeric(pred.selection_id, errors="coerce")
    fin = books[books.source.astype(str) == "final"].copy()
    fin["selection_id"] = pd.to_numeric(fin.selection_id, errors="coerce")
    win = fin[fin.runner_status.astype(str) == "WINNER"][["market_id", "selection_id"]].drop_duplicates()
    pred = pred.merge(win.assign(won=1), on=["market_id", "selection_id"], how="left")
    pred["won"] = pred.won.fillna(0).astype(int)
    pred = pred[pred.market_id.astype(str).isin(set(fin.market_id.astype(str)))]   # settled races only
    panel = to.day_panel(pred, books, markets, day, closing_books=books, retimed=False)
    if not len(panel):
        _log(f"{day}: no live-key panel rows")
        return pd.DataFrame(), None
    pre = to.pre_off_books(books, markets, retimed=False)
    pre = pre[pre.market_id.isin(set(panel.market_id))]
    path = pre[ls.BOOK_PATH_COLS].assign(lay1=pre.lay1.where(pre.lay1 > 1), last_traded=to._num(pre.last_traded))
    _log(f"{day}: {panel.race_id.nunique()} races, {len(panel)} runner-marks on the live-key books")
    return ls.attach_future(panel, ls.future_extremes(pre)), path


def fits(store, first_scored: date, fit_from: date) -> dict:
    """The forecast at each mark, fitted on the recorder's books of every day before the first scored day."""
    hist = ls.load(store, fit_from, first_scored - timedelta(days=1), source="retimed")
    out = {}
    for mark in ls.ORDER_MARKS:
        g = hist[hist["mark"] == mark] if len(hist) else hist
        if len(g) and g.race_date.nunique() >= ls.MIN_TRAIN_DAYS:
            out[mark] = ts.fit_cl(g, ts.CL_SETS["micro"])
            _log(f"forecast at {mark}: fitted on {g.race_date.nunique()} recorded days, {len(g)} rows")
    return out


def forecast(panel: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = []
    for mark, g in panel[panel["mark"].isin(list(f))].groupby("mark"):
        q, book = ts.predict(g, f[mark])
        out.append(g.assign(f_bsp=1.0 / (book * np.asarray(q, float))))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report_md(res: dict) -> str:
    L = [f"# Every strategy on the live books ({res['first_day']} to {res['last_day']})", "",
         f"Books: the live-key record (greyhound/book_record.py), {res.get('races')} settled races, "
         f"{res.get('rows')} runner-marks. Our price: the live model's. Forecast: fitted on {res.get('fit')}. "
         f"Days scored: {res.get('days')}. One day describes; it does not select. Fills: through (a later book offers "
         "our price), traded (a later trade at it), strict (a later trade beyond it: the queue ahead certainly "
         "cleared). Exits: sp, last (greened up in the last book before the off), tp<k>.", ""]
    L += ["## 1. The early lay (first traded, lay the forecast drifters at 80% of the forecast SP)", ""]
    L += ls._tbl(res["early_lay"], ls.RULE_COLS) + [""]
    L += ["## 2. The 40 best rules by return (a trade a day or more)", ""]
    L += ls._tbl(res["top"], ls.RULE_COLS) + [""]
    L += ["## 3. By family: each family's best and worst rule, and how many made money", ""]
    L += ls._tbl(res["families"], ["mark", "order", "side", "fill", "exit", "rules", "positive", "best_roi%",
                                   "worst_roi%", "best_rule"]) + [""]
    return "\n".join(L)


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--from", dest="first", default="2026-11-03", help="the first live-key day (3 Nov 2026)")
    ap.add_argument("--to", dest="last", default=None, help="default: today (UK), its settled races")
    ap.add_argument("--fit-from", default=str(to.FIRST_BOOK_DAY))
    ap.add_argument("--out", default="out/greyhound_live_eval")
    a = ap.parse_args(argv)
    t0 = time.time()
    store = Store(root=a.store)
    first = date.fromisoformat(a.first)
    last = date.fromisoformat(a.last) if a.last else uk_today()
    panels, paths, d = [], [], first
    while d <= last:
        p, path = live_panel(store, d)
        if len(p):
            panels.append(p)
            paths.append(path)
        d += timedelta(days=1)
    res = {"first_day": f"{first}", "last_day": f"{last}", "early_lay": [], "top": [], "families": []}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if panels:
        panel = pd.concat(panels, ignore_index=True)
        books = pd.concat(paths, ignore_index=True)
        f = fits(store, first, date.fromisoformat(a.fit_from))
        res["fit"] = f"the recorder's books, {a.fit_from} to {first - timedelta(days=1)}"
        fc = forecast(panel, f)
        days = sorted(fc.race_date.astype(str).unique()) if len(fc) else []
        res.update({"races": int(panel.race_id.nunique()), "rows": int(len(panel)), "days": len(days)})
        o = ls.orders(fc, books) if len(fc) else pd.DataFrame()
        if len(o):
            o["race_date"] = o.race_date.astype(str)
            t = ls.rule_summary(o, days)
            res["rules"] = ls._records(t)
            el = t[(t["mark"] == "first_traded") & (t["order"] == "bid") & (t.side == "lay") & (t.param == 0.2)
                   & (t.gate == "drift")]
            res["early_lay"] = ls._records(el)
            ok = t[t.per_day >= 1].dropna(subset=["roi%"]) if "per_day" in t else t
            res["top"] = ls._records(ok.sort_values("roi%", ascending=False).head(40))
            fam = []
            for key, g in ok.groupby(["mark", "order", "side", "fill", "exit"]):
                b = g.sort_values("roi%", ascending=False)
                fam.append(dict(zip(["mark", "order", "side", "fill", "exit"], key),
                                rules=int(len(g)), positive=int((g["roi%"] > 0).sum()),
                                **{"best_roi%": float(b["roi%"].iloc[0]), "worst_roi%": float(b["roi%"].iloc[-1]),
                                   "best_rule": f"param {b.param.iloc[0]}, gate {b.gate.iloc[0]}"}))
            res["families"] = sorted(fam, key=lambda r: -r["best_roi%"])
    res["seconds"] = round(time.time() - t0)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    md = report_md(res)
    out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
