"""Run the open-close drift models on a panel and write the figures and the report.

    python -m research.open_close.run --panel out/panel.parquet --out out/open_close_<name>.json \\
        --report reports/open_close/model_<name>.md [--name <name>] [--holdout-months 3 | --holdout-from 2026-04]
        [--split-by market_type] [--refit-every 1] [--min-train-months 12] [--commission 0.02 --commission-alt 0.05]

The last three months of the panel (or the months from ``--holdout-from``) are the holdout: nothing is chosen on
them; they are scored once, at the end, with the thresholds chosen on the development months, and labelled as such.
The development months are scored walk-forward (drift.walk_forward); the leak test (drift.shuffle_test) refits on
permuted targets. ``run_panel`` is the same for a frame in memory (the CI query imports it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import COMMISSION, COMMISSION_ALT, THRESHOLDS, accuracy_table, by_group, sweep
from .drift import default_models, shuffle_test, walk_forward
from .features import MODEL_COL, prepare

GROUP_KEYS = ("band", "market_type", "country", "sport")


def _log(msg: str) -> None:
    print(msg, flush=True)


def split_months(months: list[str], holdout_months: int, holdout_from: str | None) -> tuple[list[str], list[str]]:
    if holdout_from:
        dev = [m for m in months if m < holdout_from]
        hold = [m for m in months if m >= holdout_from]
    else:
        dev, hold = months[:-holdout_months] if holdout_months else months, months[-holdout_months:] if holdout_months else []
    return dev, hold


def choose_threshold(sw: pd.DataFrame, min_bets_per_day: float = 1.0) -> float:
    """The development threshold a model takes to the holdout: the best lower 90% bound of the ROI a tick worse at
    each end, among thresholds that still trade (at least ``min_bets_per_day`` a day); the smallest when none pays."""
    ok = sw[(sw.get("bets", 0) > 0) & (sw.get("bets_per_day", 0) >= min_bets_per_day)]
    if ok.empty or "roi_spread_lo90" not in ok.columns:
        return float(THRESHOLDS[0])
    return float(ok.loc[ok["roi_spread_lo90"].astype(float).idxmax(), "threshold"])


def score_window(scored: pd.DataFrame, pred_cols: list[str], thresholds: dict | None, commission: float,
                 commission_alt: float, n_boot: int) -> dict:
    """Accuracy and the hedged trade for the scored rows. ``thresholds``: a model -> threshold map fixed in advance
    (the holdout), else every threshold is swept and one is chosen (the development window)."""
    out = {"rows": int(len(scored)), "markets": int(scored["market_id"].nunique()),
           "days": int(scored["day"].nunique()), "months": [str(scored["month"].min()), str(scored["month"].max())],
           "accuracy": accuracy_table(scored, pred_cols).to_dict(orient="index"), "models": {}}
    for c in pred_cols:
        name = c.removeprefix("pred_")
        if name == "no_change":
            continue
        sw = sweep(scored, c, commission=commission, commission_alt=commission_alt, n_boot=n_boot)
        t = thresholds[name] if thresholds else choose_threshold(sw)
        entry = {"sweep": sw.to_dict(orient="records"), "threshold": t, "by": {}}
        for key in GROUP_KEYS:
            if scored[key].nunique() > 1 or key == "band":
                entry["by"][key] = by_group(scored, c, t, key, commission=commission).to_dict(orient="index")
                entry["by"][f"{key}_spread"] = by_group(scored, c, t, key, commission=commission,
                                                        spread=True).to_dict(orient="index")
        entry["by"]["month"] = by_group(scored, c, t, "month", commission=commission, n_boot=0).to_dict(orient="index")
        out["models"][name] = entry
    return out


def run_panel(panel: pd.DataFrame, name: str, holdout_months: int = 3, holdout_from: str | None = None,
              min_train_months: int = 12, refit_every: int = 1, commission: float = COMMISSION,
              commission_alt: float = COMMISSION_ALT, n_jobs: int = 4, n_boot: int = 2000, shuffle: bool = True,
              read_holdout: bool = True, log=_log) -> dict:
    """The whole evaluation of one panel: development walk-forward, the leak test, then the holdout once."""
    t0 = time.time()
    d = prepare(panel)
    with_model = MODEL_COL in d.columns and d[MODEL_COL].gt(1).any()
    months = sorted(d["month"].unique())
    dev_months, hold_months = split_months(months, holdout_months, holdout_from)
    res = {"name": name, "rows": int(len(d)), "markets": int(d["market_id"].nunique()),
           "months": [months[0], months[-1]] if months else [], "with_model": bool(with_model),
           "model_rows": int(d[MODEL_COL].gt(1).sum()) if with_model else 0,
           "development_months": [dev_months[0], dev_months[-1]] if dev_months else [],
           "holdout_months": [hold_months[0], hold_months[-1]] if hold_months else [],
           "settings": {"min_train_months": min_train_months, "refit_every": refit_every, "commission": commission,
                        "commission_alt": commission_alt, "thresholds": list(THRESHOLDS), "n_boot": n_boot},
           "by_market_rows": d.groupby(["sport", "country", "market_type"]).size().reset_index()
           .rename(columns={0: "rows"}).to_dict(orient="records")}
    log(f"[{name}] {len(d):,} rows, {res['markets']:,} markets, {months[0] if months else '-'} to "
        f"{months[-1] if months else '-'}; development to {dev_months[-1] if dev_months else '-'}, holdout "
        f"{res['holdout_months']}; with model: {with_model}")
    dev = d[d["month"].isin(dev_months)]
    models = default_models(with_model=with_model, n_jobs=n_jobs)
    scored = walk_forward(dev, models, min_train_months=min_train_months, refit_every=refit_every, log=log)
    pred_cols = [c for c in scored.columns if c.startswith("pred_")]
    if scored.empty:
        res["error"] = "too few months to score"
        return res
    res["development"] = score_window(scored, pred_cols, None, commission, commission_alt, n_boot)
    res["development"]["label"] = "development: walk-forward by month, thresholds chosen here"
    if with_model:
        sub = scored[scored[MODEL_COL].gt(1)]
        if len(sub) > 100:
            res["development"]["model_rows_only"] = {"rows": int(len(sub)), "months": [str(sub["month"].min()), str(sub["month"].max())],
                                                     "accuracy": accuracy_table(sub, pred_cols).to_dict(orient="index")}
    # the leak test: the same walk-forward on permuted training targets
    if shuffle:
        every = max(refit_every, 6)
        sh = shuffle_test(dev, default_models(with_model=with_model, n_jobs=n_jobs), min_train_months=min_train_months,
                          refit_every=every, log=None)
        if not sh.empty:
            res["shuffle_test"] = {"refit_every": every, "accuracy": accuracy_table(
                sh, [c for c in sh.columns if c.startswith("pred_")]).to_dict(orient="index")}
    # the holdout, read once, with the development thresholds
    if read_holdout and hold_months:
        thresholds = {k: v["threshold"] for k, v in res["development"]["models"].items()}
        fitted = [m.fit(dev) for m in default_models(with_model=with_model, n_jobs=n_jobs)]
        hold = d[d["month"].isin(hold_months)].copy()
        hold["fit_month"] = dev_months[-1]
        for m in fitted:
            hold[f"pred_{m.name}"] = m.predict(hold)
        res["holdout"] = score_window(hold, pred_cols, thresholds, commission, commission_alt, n_boot)
        res["holdout"]["label"] = ("HOLDOUT: the last months, read once, models fitted on the development months, "
                                   "thresholds fixed there")
        imp = next((m for m in fitted if getattr(m, "name", "") == "gbm"), None)
        if imp is not None:
            res["feature_importance_gbm"] = dict(sorted(imp.importance().items(), key=lambda kv: -kv[1]))
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    try:
        return f"{100 * float(x):+.2f}%" if np.isfinite(float(x)) else "-"
    except (TypeError, ValueError):
        return "-"


def _f(x, nd=4) -> str:
    try:
        return f"{float(x):.{nd}f}" if np.isfinite(float(x)) else "-"
    except (TypeError, ValueError):
        return "-"


def accuracy_md(acc: dict) -> str:
    lines = ["| model | n | RMSE | RMSE no-change | R2 vs no-change | R2 | direction hit | share RMSE | share RMSE no-change |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, a in acc.items():
        lines.append(f"| {name} | {a.get('n', 0):,} | {_f(a.get('rmse'))} | {_f(a.get('rmse_no_change'))} | "
                     f"{_f(a.get('r2_vs_no_change'))} | {_f(a.get('r2'))} | {_pct(a.get('direction_hit'))} | "
                     f"{_f(a.get('rmse_share'), 5)} | {_f(a.get('rmse_share_no_change'), 5)} |")
    return "\n".join(lines)


def sweep_md(sw: list[dict]) -> str:
    lines = ["| threshold | bets | bets/day | ROI 2% (90%) | ROI 5% | worst dd (GBP per GBP1 bets) | right way | "
             "ROI 2%, a tick worse each end (90%) | ROI 5%, spread | worst dd, spread |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sw:
        if not r.get("bets"):
            lines.append(f"| {r['threshold']:.2f} | 0 | - | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {r['threshold']:.2f} | {r['bets']:,} | {r['bets_per_day']:.1f} | {_pct(r['roi'])} "
                     f"({_pct(r['roi_lo90'])} to {_pct(r['roi_hi90'])}) | {_pct(r.get('roi_alt_commission'))} | "
                     f"{_f(r['worst_drawdown'], 1)} | {_pct(r['right_way'])} | {_pct(r.get('roi_spread'))} "
                     f"({_pct(r.get('roi_spread_lo90'))} to {_pct(r.get('roi_spread_hi90'))}) | "
                     f"{_pct(r.get('roi_spread_alt_commission'))} | {_f(r.get('worst_drawdown_spread'), 1)} |")
    return "\n".join(lines)


def group_md(by: dict, by_spread: dict | None) -> str:
    lines = ["| group | bets | bets/day | ROI 2% (90%) | right way | ROI, a tick worse each end (90%) |",
             "|---|---|---|---|---|---|"]
    for k, r in by.items():
        s = (by_spread or {}).get(k, {})
        lines.append(f"| {k} | {int(r.get('bets', 0)):,} | {_f(r.get('bets_per_day'), 1)} | {_pct(r.get('roi'))} "
                     f"({_pct(r.get('roi_lo90'))} to {_pct(r.get('roi_hi90'))}) | {_pct(r.get('right_way'))} | "
                     f"{_pct(s.get('roi'))} ({_pct(s.get('roi_lo90'))} to {_pct(s.get('roi_hi90'))}) |")
    return "\n".join(lines)


def window_md(w: dict, title: str) -> list[str]:
    out = [f"### {title}", "", f"{w.get('label', '')}", "",
           f"{w['rows']:,} selections in {w['markets']:,} markets over {w['days']:,} days, {w['months'][0]} to {w['months'][1]}.",
           "", "Accuracy of log(close/open):", "", accuracy_md(w["accuracy"]), ""]
    if "model_rows_only" in w:
        m = w["model_rows_only"]
        out += [f"The rows that carry the model's forecast only ({m['rows']:,}, {m['months'][0]} to {m['months'][1]}):", "",
                accuracy_md(m["accuracy"]), ""]
    for name, e in w["models"].items():
        out += [f"Hedged trade, {name} (threshold on |forecast log move|; GBP1 a selection; chosen threshold {e['threshold']:.2f}):",
                "", sweep_md(e["sweep"]), ""]
        for key in ("band", "market_type", "country", "sport"):
            if key in e["by"]:
                out += [f"{name} at {e['threshold']:.2f} by {key}:", "", group_md(e["by"][key], e["by"].get(f"{key}_spread")), ""]
    return out


def report_md(res: dict) -> str:
    s = res["settings"]
    out = [f"# Open-close drift model: {res['name']}", "",
           f"Panel: {res['rows']:,} selections, {res['markets']:,} markets, {res['months'][0]} to {res['months'][1]}"
           f"{'; the model forecast on ' + format(res['model_rows'], ',') + ' rows' if res['with_model'] else '; market-only'}.",
           f"Walk-forward by month, first {s['min_train_months']} months fit only, refit every {s['refit_every']} month(s); "
           f"commission {s['commission']:.0%} on each market's net (sensitivity {s['commission_alt']:.0%}); "
           f"{s['n_boot']} day-bootstrap draws; a tick worse at each end on Betfair's ladder (trading/exchange.py).",
           "", "Rows by market:", "", "| sport | country | market_type | rows |", "|---|---|---|---|"]
    out += [f"| {r['sport']} | {r['country']} | {r['market_type']} | {r['rows']:,} |" for r in res["by_market_rows"]]
    out += [""]
    if "error" in res:
        return "\n".join(out + [f"Not scored: {res['error']}", ""])
    out += window_md(res["development"], "Development window")
    if "shuffle_test" in res:
        out += ["### Leak test: training targets permuted", "",
                f"The same walk-forward with the training targets shuffled (refit every {res['shuffle_test']['refit_every']} "
                "months). Every model should fall to the baseline or under it:", "", accuracy_md(res["shuffle_test"]["accuracy"]), ""]
    if "holdout" in res:
        out += window_md(res["holdout"], "Holdout (read once)")
    if "feature_importance_gbm" in res:
        top = list(res["feature_importance_gbm"].items())[:12]
        out += ["### Gradient boosting: feature gain (fit on the development months)", "", "| feature | gain |", "|---|---|"]
        out += [f"| {k} | {v:,.0f} |" for k, v in top]
        out += [""]
    out += [f"Run time {res.get('seconds', '-')} s.", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_panel(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p, low_memory=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", required=True, help="the open-close panel, parquet or csv")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", required=True, help="the figures, JSON")
    ap.add_argument("--report", default=None, help="the report, markdown")
    ap.add_argument("--holdout-months", type=int, default=3)
    ap.add_argument("--holdout-from", default=None, help="YYYY-MM: the holdout starts here instead")
    ap.add_argument("--min-train-months", type=int, default=12)
    ap.add_argument("--refit-every", type=int, default=1)
    ap.add_argument("--commission", type=float, default=COMMISSION)
    ap.add_argument("--commission-alt", type=float, default=COMMISSION_ALT)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--split-by", default=None, help="a column: run each value as its own dataset")
    ap.add_argument("--no-shuffle", action="store_true", help="skip the leak test")
    ap.add_argument("--no-holdout", action="store_true", help="do not read the holdout")
    a = ap.parse_args(argv)
    panel = read_panel(a.panel)
    name = a.name or Path(a.panel).stem
    kw = dict(holdout_months=a.holdout_months, holdout_from=a.holdout_from, min_train_months=a.min_train_months,
              refit_every=a.refit_every, commission=a.commission, commission_alt=a.commission_alt, n_jobs=a.n_jobs,
              n_boot=a.n_boot, shuffle=not a.no_shuffle, read_holdout=not a.no_holdout)
    if a.split_by:
        results = [run_panel(part, f"{name}:{a.split_by}={val}", **kw) for val, part in panel.groupby(a.split_by)]
    else:
        results = [run_panel(panel, name, **kw)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1, default=str) + "\n")
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text("\n\n".join(report_md(r) for r in results))
    _log(f"wrote {a.out}" + (f" and {a.report}" if a.report else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
