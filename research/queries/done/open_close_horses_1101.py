"""Open-to-close drift on UK and Irish win markets, market-only and with the model (the owner's ask, 1 Nov 2026).

The panel (research/open_close/panels.horses_panel): every runner in betfair_prices with a morning price (MORNINGWAP,
the volume-weighted average of the morning's trades: the open) and a BSP (the close), UK win and IE win from 2021,
with the morning volume, the pre-play WAP and the result; the model's out-of-sample forecast (data/oos_predictions.csv,
predicted_bfsp, 31 May 2024 to 21 Mar 2026) joined through race_results as ``model_price`` for the "with model" arm.
No predictions table or feature cache holds an earlier forecast, so the model arm covers those months only, and the
market-only arm the whole window.

The models (research/open_close/drift): no change, OLS by price band, LightGBM on the opening book, a conditional
logit of the closing shares (a coherent book), fitted walk-forward by month; scored by RMSE and R2 against no
change, the direction hit rate, and the hedged trade (GBP1 at the open, closed at the BSP, greened; 2% commission on
each market's net, 5% as the sensitivity; a tick worse at each end). Development 2021-01 to 2026-03 (the first 12
months fit only); holdout April 2026 on, read once with the development thresholds (research/open_close/run).

Also, where the account reaches S3: the greyhounds from Betfair's Historic Data (first traded price as the open, the
last pre-play price as the close, Jan-Sep 2026, GB WIN); and the research agent's pre-registered market-only cell on
the lay side (reports/open_close/markets.md 6.1) on betfair_live_marks (the recorder's books from 1 Oct 2026): a lay
at the best lay at the morning clock marks on runners priced 21-51 and 51+, closed by a back at the BSP, reported
apart. OPEN_CLOSE_SPORTS (default "horse,greyhound,lay") or ``--sport`` chooses; OPEN_CLOSE_REFIT the refit step.
Read-only; the output is the job log (tables, then a compact JSON block between the === JSON === lines).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from model.race_book import ratio_interval  # noqa: E402
from research.open_close.panels import greyhound_panel, horses_panel  # noqa: E402
from research.open_close.run import report_md, run_panel  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
DB = os.environ.get("HORSE_DB") or "horse_racing.db"
FORECASTS = Path(os.environ.get("OOS_PREDICTIONS") or "data/oos_predictions.csv")
FROM = "2021-01-01"
HOLDOUT_FROM = "2026-04"                      # the repository's locked holdout
COMMISSION = 0.02
N_BOOT = int(os.environ.get("OPEN_CLOSE_NBOOT", "2000"))     # day-bootstrap draws
N_JOBS = int(os.environ.get("OPEN_CLOSE_NJOBS", "4"))
LAY_BANDS = [(21.0, 51.0, "21-51"), (51.0, 1001.0, "51+")]
CLOCK = ["clock_08:00", "clock_09:00", "clock_10:00", "clock_11:00", "clock_12:00"]
PRE_REGISTERED = {"band": "21-51", "marks": ("clock_09:00", "clock_10:00", "clock_11:00"), "min_size": 20.0}

PRICES_SQL = f"""
    SELECT b.country, b.market_type, b.race_date, b.race_time, b.event_id, b.event_name, b.selection_id,
           b.selection_name, b.win_lose, b.bsp, b.ppwap, b.morningwap, b.morning_vol, b.pp_vol, b.race_results_id,
           r.track, r.race_time AS rr_time, r.horse_name
    FROM betfair_prices b LEFT JOIN race_results r ON r.id = b.race_results_id
    WHERE LOWER(b.market_type) = 'win' AND b.race_date >= '{FROM}' AND b.country IN ('uk', 'ire')
"""
MARKS_SQL = """
    SELECT m.race_date, m.market_id, m.selection_id, m.mark, m.minutes_to_off, m.back1, m.back1_size, m.lay1,
           m.lay1_size, m.bsp, k.country, k.market_type, k.venue, k.race_results_id
    FROM betfair_live_marks m
    JOIN (SELECT market_id, selection_id, country, market_type, venue, race_results_id
          FROM betfair_live_markets) k
      ON k.market_id = m.market_id AND k.selection_id = m.selection_id
    WHERE k.market_type = 'WIN' AND (m.mark LIKE 'clock_%' OR m.mark = 'final')
"""
# the final mark carries no BSP before 7 Oct 2026 (the inventory, A2): the price files' BSP fills it through race_results
BSP_FILL_SQL = """
    SELECT race_results_id, bsp FROM betfair_prices
    WHERE LOWER(market_type) = 'win' AND race_date >= '2026-10-01' AND bsp > 1 AND race_results_id IS NOT NULL
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def compact(res: dict) -> dict:
    """The run's figures without the per-month and spread-duplicate breakdowns (the job log's JSON block)."""
    out = {k: v for k, v in res.items() if k not in ("development", "holdout", "shuffle_test", "feature_importance_gbm")}
    for w in ("development", "holdout"):
        if w in res:
            win = res[w]
            out[w] = {"label": win.get("label"), "rows": win["rows"], "markets": win["markets"], "days": win["days"],
                      "months": win["months"], "accuracy": win["accuracy"], "models": {}}
            if "model_rows_only" in win:
                out[w]["model_rows_only"] = win["model_rows_only"]
            for name, e in win["models"].items():
                keep = ["threshold", "bets", "bets_per_day", "roi", "roi_lo90", "roi_hi90", "roi_alt_commission",
                        "worst_drawdown", "right_way", "roi_spread", "roi_spread_lo90", "roi_spread_hi90",
                        "roi_spread_alt_commission", "worst_drawdown_spread"]
                out[w]["models"][name] = {"threshold": e["threshold"],
                                          "sweep": [{k: r.get(k) for k in keep if k in r} for r in e["sweep"]],
                                          "by_band": e["by"].get("band"), "by_band_spread": e["by"].get("band_spread"),
                                          "by_month": e["by"].get("month")}
    if "shuffle_test" in res:
        out["shuffle_test"] = {"refit_every": res["shuffle_test"]["refit_every"],
                               "r2_vs_no_change": {k: v.get("r2_vs_no_change") for k, v in
                                                   res["shuffle_test"]["accuracy"].items()}}
    if "feature_importance_gbm" in res:
        out["feature_importance_gbm_top"] = dict(list(res["feature_importance_gbm"].items())[:10])
    return out


# ---------------------------------------------------------------------------
# the pre-registered lay cell (markets.md 6.1), on the recorder's books
# ---------------------------------------------------------------------------

def lay_trades(marks: pd.DataFrame) -> pd.DataFrame:
    """One row a runner-mark at the morning clock marks: a lay of GBP1 at the best lay there (taking the offer: the
    spread is paid at the open; the close at the BSP is the SP, no spread), closed by a back at the BSP, greened:
    profit 1 - lay / BSP a pound of stake, liability lay - 1. The BSP is the runner's final mark's."""
    m = marks.copy()
    for c in ("lay1", "lay1_size", "back1", "bsp", "minutes_to_off"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    bsp = m[m["mark"].eq("final") & (m["bsp"] > 1)].drop_duplicates(["market_id", "selection_id"]) \
        .set_index(["market_id", "selection_id"])["bsp"]
    t = m[m["mark"].isin(CLOCK) & (m["lay1"] > 1) & (m["minutes_to_off"] > 0)].copy()
    t["bsp_final"] = bsp.reindex(pd.MultiIndex.from_arrays([t["market_id"], t["selection_id"]])).to_numpy()
    t = t[t["bsp_final"] > 1].copy()
    t["band"] = pd.cut(t["lay1"], [b[0] for b in LAY_BANDS] + [LAY_BANDS[-1][1]], labels=[b[2] for b in LAY_BANDS],
                       right=False).astype(str)
    t = t[t["band"].isin([b[2] for b in LAY_BANDS])].copy()
    t["profit"] = 1.0 - t["lay1"] / t["bsp_final"]
    t["liability"] = t["lay1"] - 1.0
    t["stake"] = 1.0
    t["month"] = t["race_date"].astype(str).str[:7]
    t["day"] = t["race_date"].astype(str)
    t["first_mark"] = t.groupby(["market_id", "selection_id"])["mark"].transform("min") == t["mark"]
    return t.reset_index(drop=True)


def lay_summary(t: pd.DataFrame, commission: float = COMMISSION, n_boot: int = 2000) -> dict:
    """Bets, liability, the liability-weighted return after commission on each market's net, its 90% interval by
    day, the return a pound of stake, the share closed shorter than the lay (the lay lost)."""
    if t.empty:
        return {"bets": 0}
    mk = t.groupby(["day", "market_id"]).agg(profit=("profit", "sum"), liability=("liability", "sum"),
                                              stake=("stake", "sum")).reset_index()
    mk["net"] = mk["profit"] - commission * mk["profit"].clip(lower=0)
    day = mk.groupby("day").agg(net=("net", "sum"), liability=("liability", "sum"), stake=("stake", "sum"))
    lo, hi = ratio_interval(day["net"].to_numpy(), day["liability"].to_numpy(), n=n_boot)
    return {"bets": int(len(t)), "days": int(len(day)), "liability": float(day["liability"].sum()),
            "return_on_liability": float(day["net"].sum() / day["liability"].sum()),
            "lo90": lo, "hi90": hi, "return_per_stake": float(day["net"].sum() / day["stake"].sum()),
            "closed_shorter": float((t["bsp_final"] < t["lay1"]).mean()),
            "median_size_offered": float(t["lay1_size"].median())}


def lay_cells(t: pd.DataFrame) -> pd.DataFrame:
    """Every band x mark x size cell, each runner-mark a bet, and the pre-registered cell apart."""
    rows = []
    for band in [b[2] for b in LAY_BANDS]:
        for mark in CLOCK + ["09:00-11:00"]:
            marks = PRE_REGISTERED["marks"] if mark == "09:00-11:00" else (mark,)
            for size in (0.0, PRE_REGISTERED["min_size"]):
                part = t[t["band"].eq(band) & t["mark"].isin(marks) & (t["lay1_size"] >= size)]
                s = lay_summary(part, n_boot=500)
                pre = (band == PRE_REGISTERED["band"] and mark == "09:00-11:00" and size == PRE_REGISTERED["min_size"])
                rows.append({"band": band, "mark": mark, "min_size": size, "pre_registered": pre, **s})
    return pd.DataFrame(rows)


def run_lay_cell(conn) -> dict:
    marks = pd.read_sql_query(MARKS_SQL, conn)
    log(f"\n== the lay side, market-only (markets.md 6.1): {len(marks):,} runner-marks read from betfair_live_marks "
        f"(WIN), {marks['race_date'].min()} to {marks['race_date'].max()}")
    fin = marks["mark"].eq("final")
    had = int((fin & (pd.to_numeric(marks["bsp"], errors="coerce") > 1)).sum())
    fill = pd.read_sql_query(BSP_FILL_SQL, conn).drop_duplicates("race_results_id").set_index("race_results_id")["bsp"]
    need = fin & ~(pd.to_numeric(marks["bsp"], errors="coerce") > 1)
    marks.loc[need, "bsp"] = marks.loc[need, "race_results_id"].map(fill).to_numpy()
    log(f"   final marks with a BSP: {had:,} from the record, "
        f"{int((fin & (pd.to_numeric(marks['bsp'], errors='coerce') > 1)).sum()) - had:,} more from the price files")
    t = lay_trades(marks)
    log(f"   {len(t):,} runner-marks at the clock marks priced 21+ at the best lay with a final BSP, "
        f"{t['market_id'].nunique():,} markets, {t['day'].nunique()} days; countries "
        f"{t['country'].value_counts().to_dict()}")
    cells = lay_cells(t)
    show = ["band", "mark", "min_size", "bets", "days", "liability", "return_on_liability", "lo90", "hi90",
            "return_per_stake", "closed_shorter", "median_size_offered"]
    log("\n   every cell (each runner-mark a GBP1 lay at the best lay, closed by a back at the BSP, 2% on each market's "
        "net; return on liability):")
    log("   " + cells[show].round(4).to_string(index=False).replace("\n", "\n   "))
    pre = t[t["band"].eq(PRE_REGISTERED["band"]) & t["mark"].isin(PRE_REGISTERED["marks"])
            & (t["lay1_size"] >= PRE_REGISTERED["min_size"])]
    out = {"pre_registered": {"cell": {**PRE_REGISTERED, "marks": list(PRE_REGISTERED["marks"])},
                              "all_runner_marks": lay_summary(pre),
                              "first_mark_a_runner": lay_summary(pre[pre["first_mark"]]),
                              "by_month": {m: lay_summary(p, n_boot=500) for m, p in pre.groupby("month")},
                              "by_country": {c: lay_summary(p, n_boot=500) for c, p in pre.groupby("country")}},
           "cells": cells.to_dict(orient="records")}
    s = out["pre_registered"]["all_runner_marks"]
    log(f"\n   PRE-REGISTERED (21-51 at the best lay, 09:00-11:00 marks, GBP20+ offered): {s.get('bets', 0):,} runner-marks, "
        f"liability GBP{s.get('liability', 0):,.0f}, return on liability {100 * s.get('return_on_liability', np.nan):+.2f}% "
        f"(90% by day {100 * s.get('lo90', np.nan):+.2f} to {100 * s.get('hi90', np.nan):+.2f}), "
        f"closed shorter than the lay on {100 * s.get('closed_shorter', np.nan):.1f}%")
    s = out["pre_registered"]["first_mark_a_runner"]
    log(f"   the same, one lay a runner at its first mark in the window: {s.get('bets', 0):,} bets, return on liability "
        f"{100 * s.get('return_on_liability', np.nan):+.2f}% ({100 * s.get('lo90', np.nan):+.2f} to {100 * s.get('hi90', np.nan):+.2f})")
    log("   by month: " + json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                                      for k, v in out["pre_registered"]["by_month"].items()}))
    return out


# ---------------------------------------------------------------------------
# the panels and the runs
# ---------------------------------------------------------------------------

def load_forecasts() -> pd.DataFrame | None:
    if not FORECASTS.exists():
        log(f"no forecasts at {FORECASTS}: market-only")
        return None
    f = pd.read_csv(FORECASTS, dtype={"race_time": str}, usecols=["race_date", "race_time", "track", "horse_name",
                                                                   "predicted_bfsp"])
    log(f"forecasts: {len(f):,} rows, {f['race_date'].min()} to {f['race_date'].max()} ({FORECASTS})")
    return f


def run_horses(conn, refit_every: int, budget_minutes: float) -> tuple[list[dict], str]:
    t0 = time.time()
    prices = pd.read_sql_query(PRICES_SQL, conn)
    log(f"betfair_prices win rows from {FROM}: {len(prices):,}; by country {prices['country'].value_counts().to_dict()}; "
        f"with a result row {prices['race_results_id'].notna().mean():.1%}")
    panel = horses_panel(prices, load_forecasts())
    results, reports = [], []
    for country in ("uk", "ire"):
        part = panel[panel["country"].eq(country)]
        elapsed = (time.time() - t0) / 60.0
        step = refit_every if elapsed < budget_minutes * 0.6 else max(refit_every, 3)
        if step != refit_every:
            log(f"[{country}] {elapsed:.0f} min gone: refitting every {step} months to keep the budget")
        res = run_panel(part, f"horse_{country}_win", holdout_from=HOLDOUT_FROM, min_train_months=12,
                        refit_every=step, commission=COMMISSION, commission_alt=0.05, n_jobs=N_JOBS, n_boot=N_BOOT, log=log)
        results.append(res)
        reports.append(report_md(res))
        log(f"[{country}] done in {res.get('seconds')} s")
    return results, "\n\n".join(reports)


def run_greyhounds() -> tuple[list[dict], str]:
    os.environ.setdefault("CAPTURE_BUCKET", os.environ.get("PRED_BUCKET") or "ashcroft")
    os.environ.setdefault("CAPTURE_REGION", os.environ.get("PRED_REGION") or "eu-west-2")
    from sources.common import Store
    store = Store()
    keys = [k for k in store.listing("betfair_historic/") if "markets_greyhound_racing_" in k and k.endswith(".parquet")]
    log(f"\n== greyhounds: Betfair Historic Data tables {sorted(keys)}")
    if not keys:
        return [], ""
    bf = pd.concat([store.get_parquet(k) for k in sorted(keys)], ignore_index=True)
    results, reports = [], []
    # the close at the BSP first (the hedged lay-at-first, back-at-BSP and back-at-first, lay-at-BSP the coordinator
    # asked for, as the horses are scored), then at the last pre-play traded price (the data agent's reading)
    for close in ("bsp", "ltp_close"):
        panel = greyhound_panel(bf, close=close)
        panel = panel[panel["country"].eq("gb")]
        log(f"   GB WIN runners with a first price and a {close}: "
            f"{int(((panel['open_price'] > 1) & (panel['close_price'] > 1)).sum()):,} of {len(panel):,}; "
            f"{panel['off_time_utc'].min()} to {panel['off_time_utc'].max()} (no volume at the first trade in the BASIC files)")
        res = run_panel(panel, f"greyhound_gb_win_first_to_{close}", holdout_months=3, min_train_months=2,
                        refit_every=1, commission=COMMISSION, commission_alt=0.05, n_jobs=N_JOBS, n_boot=N_BOOT, log=log)
        results.append(res)
        reports.append(report_md(res))
    return results, "\n\n".join(reports)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    sports = os.environ.get("OPEN_CLOSE_SPORTS", "horse,greyhound,lay")
    if "--sport" in argv:
        sports = argv[argv.index("--sport") + 1]
    sports = [s.strip() for s in sports.split(",") if s.strip()]
    refit_every = int(os.environ.get("OPEN_CLOSE_REFIT", "1"))
    budget = float(os.environ.get("OPEN_CLOSE_MAX_MINUTES", "40"))
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out: dict = {"sports": sports, "results": [], "lay_cell": None, "errors": {}}
    reports = []
    for sport in sports:
        t0 = time.time()
        try:
            if sport == "horse":
                res, md = run_horses(conn, refit_every, budget)
            elif sport == "greyhound":
                res, md = run_greyhounds()
            elif sport == "lay":
                out["lay_cell"] = run_lay_cell(conn)
                continue
            else:
                raise ValueError(f"unknown sport {sport}")
            out["results"] += res
            reports.append(md)
        except Exception as exc:                                 # one part failing must not lose the others
            import traceback
            traceback.print_exc()
            out["errors"][sport] = f"{type(exc).__name__}: {exc}"[:500]
        log(f"[{sport}] {(time.time() - t0) / 60:.1f} min")
    print("\n\n".join(reports))
    print("\n=== JSON ===")
    print(json.dumps({**out, "results": [compact(r) for r in out["results"]]}, default=str))
    print("=== JSON ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
