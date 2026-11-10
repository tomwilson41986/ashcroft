"""The back study (the owner, 10 Nov 2026: "I want to back. Either back and trade out at BFSP, or back just to win.
Research strategies").

Every back rule the recorded books allow, read on the development days and on the holdout alike (nothing chosen on
the holdout), with the no-model controls beside them:

- **When:** the first traded book, 5 and 3 minutes before the off (research/greyhound/limit_study.py's marks).
- **The order:** ``take`` the best back price now; ``join`` the backers' queue at the best lay price; ``ticks``: rest
  1, 3, 5 ticks above the best back price; ``ask``: rest at our forecast SP plus 0, 5, 10, 20% (the forecast: the
  study's conditional logit on the book and the model's price, walk-forward by day).
- **Which dogs (gates):** every dog (``all``); those the forecast says will shorten (``shorten``: forecast SP under the
  mid now; ``shorten10``, ``shorten20``: at least 10% or 20% under); the market's favourite (``fav``); the model's
  favourite (``model_fav``); where the model's win chance is worth 10% over the best back price now (``value10``);
  short (mid under 3) and long (8 or more) prices; the favourite the forecast says will shorten (``fav_shorten``).
- **How it ends (the owner's two):** **trade out at the SP** (``sp``: the back's winnings laid at the SP, the profit
  ``P/SP - 1`` a unit whatever wins), greened in the last book before the off (``last``) or at a take-profit
  (``tp2/5/10``) for reference; and **back to win** (``hold``: settled by the result).
- **Filled:** ``traded`` (a later trade at our price) and ``strict`` (a later trade beyond it: the queue ahead
  certainly cleared, the lower bound).

Each rule's return on the money at risk after 2% commission on each market's net, with a 90% interval from a bootstrap
over days; a pass: development above zero, holdout's lower bound above zero, at least ``MIN_HOLD_TRADES`` holdout
trades. Every rule is counted, with the false discovery rate (Benjamini-Hochberg on one-sided bootstrap p-values) and
the number expected by chance; the controls (``take``/``join``/``ticks`` on ``all``) say what the market does on its
own.

    python -m research.greyhound.back_study --pred-dir s3:greyhound/live/oos_prices_stack_all3 --out out/back_study
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

from research.greyhound import limit_study as ls

log = logging.getLogger(__name__)

GATES = ("all", "shorten", "shorten10", "shorten20", "fav", "model_fav", "value10", "short", "long", "fav_shorten")
FILLS = ("traded", "strict")
EXITS = ("sp", "last", "tp2", "tp5", "tp10")
KEYS = ("mark", "order", "param", "gate", "fill")
MIN_HOLD_TRADES = 50
N_BOOT = 2000


def gate_flags(d: pd.DataFrame) -> pd.DataFrame:
    """Each row's gates (the rows of the forecast panel: one runner at one mark)."""
    mid = d["mid"].to_numpy(float)
    f = d.f_bsp.to_numpy(float)
    back1 = d.back1.to_numpy(float)
    p = d.p_model.to_numpy(float)
    rank_model = d.groupby(["market_id", "mark"]).p_model.rank(ascending=False, method="first").to_numpy()
    fav = d["rank"].to_numpy(float) == 1
    g = {"all": np.ones(len(d), bool), "shorten": f < mid, "shorten10": f < 0.9 * mid, "shorten20": f < 0.8 * mid,
         "fav": fav, "model_fav": rank_model == 1, "value10": p * (1 + (back1 - 1) * 0.98) - 1 > 0.10,
         "short": mid < ls.SHORT_PRICE, "long": mid >= ls.LONG_PRICE}
    g["fav_shorten"] = fav & g["shorten"]
    out = d[["market_id", "selection_id", "mark"]].copy()
    for k, v in g.items():
        out[f"G_{k}"] = np.asarray(v, bool)
    return out.drop_duplicates(["market_id", "selection_id", "mark"])


def back_orders(fc: pd.DataFrame, books: pd.DataFrame | None) -> pd.DataFrame:
    """The limit study's orders for the back side only, with this study's gates."""
    o = ls.orders(fc, books)
    o = o[(o.side == "back") & o.fill.isin(FILLS) & o["exit"].isin(EXITS)].copy()
    o = o.merge(gate_flags(fc), on=["market_id", "selection_id", "mark"], how="left")
    o["race_date"] = o.race_date.astype(str)
    return o


def _interval(net: np.ndarray, risk: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """The 90% interval of sum(net)/sum(risk), days resampled, and the one-sided p-value of a return at or below 0."""
    if not len(net) or risk.sum() <= 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(net), size=(N_BOOT, len(net)))
    r = net[idx].sum(1) / np.maximum(risk[idx].sum(1), 1e-12)
    return float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95)), float((r <= 0).mean())


def table(o: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """Each rule (mark, order, param, gate, fill) with each ending: trades, a day, the return on the money at risk and
    its interval, for each exit, and held to the result (the back to win)."""
    from greyhound import tradeout as to
    if not len(o):
        return pd.DataFrame()
    rows = []
    for gate in GATES:
        sub = o[o[f"G_{gate}"].fillna(False).astype(bool) & o.filled].assign(gate=gate)
        if not len(sub):
            continue
        for ex in EXITS:
            e = sub[sub["exit"] == ex]
            if not len(e):
                continue
            m = to.settle(e, KEYS)
            m["race_date"] = m.race_date.astype(str)
            for key, g in m.groupby(list(KEYS), dropna=False):
                day = g.groupby("race_date").agg(net=("net", "sum"), held=("held_net", "sum"), risk=("risk", "sum"))
                day = day.reindex(days, fill_value=0.0)
                lo, hi, pv = _interval(day.net.to_numpy(), day.risk.to_numpy())
                row = {**dict(zip(KEYS, key)), "exit": ex, "trades": int(g.trades.sum()),
                       "per_day": round(g.trades.sum() / max(1, len(days)), 1),
                       "roi%": round(100 * g.net.sum() / g.risk.sum(), 2) if g.risk.sum() else None,
                       "roi_90%": [round(100 * lo, 2), round(100 * hi, 2)], "p": round(pv, 4)}
                if ex == "sp":                      # held to the result: the same entries, settled by the winner
                    hlo, hhi, hpv = _interval(day.held.to_numpy(), day.risk.to_numpy())
                    rows.append({**dict(zip(KEYS, key)), "exit": "hold", "trades": row["trades"],
                                 "per_day": row["per_day"],
                                 "roi%": round(100 * g.held_net.sum() / g.risk.sum(), 2) if g.risk.sum() else None,
                                 "roi_90%": [round(100 * hlo, 2), round(100 * hhi, 2)], "p": round(hpv, 4)})
                rows.append(row)
    return pd.DataFrame(rows)


def bh(p: pd.Series, q: float = 0.10) -> pd.Series:
    """Benjamini-Hochberg: True where the rule is a discovery at false discovery rate ``q``."""
    s = p.fillna(1.0).sort_values()
    n = len(s)
    ok = s.to_numpy() <= q * np.arange(1, n + 1) / max(n, 1)
    k = np.flatnonzero(ok).max() + 1 if ok.any() else 0
    out = pd.Series(False, index=p.index)
    out.loc[s.index[:k]] = True
    return out


def combine(dev: pd.DataFrame, hold: pd.DataFrame) -> pd.DataFrame:
    k = list(KEYS) + ["exit"]
    j = dev.merge(hold, on=k, how="outer", suffixes=("_dev", "_hold"))
    lo = j["roi_90%_hold"].map(lambda v: v[0] if isinstance(v, list) else np.nan)
    j["passes"] = (j["roi%_dev"] > 0) & (lo > 0) & (j["trades_hold"] >= MIN_HOLD_TRADES)
    j["control"] = j.order.isin(["take", "join", "ticks"]) & (j.gate == "all")
    j["bh_discovery"] = False
    tried = ~j.control
    j.loc[tried, "bh_discovery"] = bh(j.loc[tried, "p_hold"]) & (j.loc[tried, "roi%_dev"] > 0)
    return j


def markdown(res: dict, j: pd.DataFrame) -> str:
    def cell(v):
        return f"{v[0]} to {v[1]}" if isinstance(v, list) else ""
    lines = [f"## The back study ({res['first_day']} to {res['last_day']})", "",
             f"Our price: {res['model']}. Scored days {res.get('scored_days')} (development {len(res.get('dev_days', []))}"
             f", holdout {len(res.get('holdout_days', []))}). Commission 2%. Returns on the money at risk.", "",
             f"**{res['rules_tried']} rules, {res['passes']} pass** (development above zero, holdout's 90% lower bound "
             f"above zero, {MIN_HOLD_TRADES}+ holdout trades; about {res['expected_by_chance']} by chance); "
             f"**{res['bh_discoveries']}** survive a 10% false discovery rate.", "",
             "### The market on its own (no model: every runner)", "",
             "| Mark | Order | Param | Fill | Exit | Dev trades | Dev return | Holdout trades | Holdout return (90%) |",
             "|---|---|---|---|---|---|---|---|---|"]
    c = j[j.control].sort_values(["exit", "mark", "order", "param", "fill"])
    for _, r in c.iterrows():
        if r["exit"] not in ("sp", "hold"):
            continue
        lines.append(f"| {r['mark']} | {r['order']} | {r['param']} | {r['fill']} | {r['exit']} | {r.get('trades_dev')} | "
                     f"{r.get('roi%_dev')}% | {r.get('trades_hold')} | {r.get('roi%_hold')}% "
                     f"({cell(r.get('roi_90%_hold'))}) |")
    for ex, title in (("sp", "Back and trade out at the SP"), ("hold", "Back to win"), ("last", "Back and green up in the last book")):
        lines += ["", f"### {title}: the passes, and the 15 best holdout rules", "",
                  "| Mark | Order | Param | Gate | Fill | Dev trades | Dev return (90%) | Holdout trades | "
                  "Holdout return (90%) | Pass | FDR 10% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        t = j[(j["exit"] == ex) & ~j.control]
        lo = t["roi_90%_hold"].map(lambda v: v[0] if isinstance(v, list) else -1e9)
        t = t.assign(_lo=lo).sort_values("_lo", ascending=False)
        t = pd.concat([t[t.passes], t[~t.passes].head(max(0, 15 - int(t.passes.sum())))])
        for _, r in t.iterrows():
            lines.append(f"| {r['mark']} | {r['order']} | {r['param']} | {r['gate']} | {r['fill']} | {r.get('trades_dev')} | "
                         f"{r.get('roi%_dev')}% ({cell(r.get('roi_90%_dev'))}) | {r.get('trades_hold')} | "
                         f"{r.get('roi%_hold')}% ({cell(r.get('roi_90%_hold'))}) | {'yes' if r['passes'] else ''} | "
                         f"{'yes' if r['bh_discovery'] else ''} |")
    return "\n".join(lines) + "\n"


def run(store, first: date, last: date, pred_dir: str | None, out: Path, source: str = "retimed") -> dict:
    t0 = time.time()
    paths: list[pd.DataFrame] = []
    panel = ls.load(store, first, last, source=source, pred_dir=pred_dir, books_out=paths)
    books = pd.concat(paths, ignore_index=True) if paths else None
    res = {"first_day": f"{first}", "last_day": f"{last}", "model": pred_dir or "the 5 Oct model",
           "panel_rows": int(len(panel))}
    if not len(panel):
        res["note"] = "no panel"
        return res
    days = sorted(panel.race_date.astype(str).unique())
    scored = days[ls.MIN_TRAIN_DAYS:]
    hold = scored[-ls.HOLDOUT_DAYS:] if len(scored) > ls.HOLDOUT_DAYS + 3 else []
    dev = [d for d in scored if d not in hold]
    res.update({"scored_days": f"{scored[0]} to {scored[-1]}" if scored else None, "dev_days": dev,
                "holdout_days": hold})
    fc = ls.forecast(panel)
    o = back_orders(fc, books) if len(fc) else pd.DataFrame()
    td, th = table(o[o.race_date.isin(dev)], dev), table(o[o.race_date.isin(hold)], hold)
    j = combine(td, th) if len(td) and len(th) else pd.DataFrame()
    tried = j[~j.control] if len(j) else j
    res.update({"rules_tried": int(len(tried)), "passes": int(tried.passes.sum()) if len(j) else 0,
                "bh_discoveries": int(tried.bh_discovery.sum()) if len(j) else 0,
                "expected_by_chance": round(0.05 * len(tried), 1), "seconds": round(time.time() - t0)})
    out.parent.mkdir(parents=True, exist_ok=True)
    res["rules"] = json.loads(j.to_json(orient="records")) if len(j) else []
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    out.with_suffix(".md").write_text(markdown(res, j) if len(j) else "no rules\n")
    return {k: v for k, v in res.items() if k != "rules"}


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from greyhound import tradeout as to
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--from", dest="first", default=str(to.FIRST_BOOK_DAY))
    ap.add_argument("--to", dest="last", default=None)
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--source", choices=["recorder", "retimed", "live"], default="retimed")
    ap.add_argument("--out", default="out/back_study")
    a = ap.parse_args(argv)
    last = date.fromisoformat(a.last) if a.last else uk_today() - timedelta(days=1)
    print(json.dumps(run(Store(root=a.store), date.fromisoformat(a.first), last, a.pred_dir, Path(a.out), a.source),
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
