"""Run the registered strategies of the 100-strategy harness and print the reading (conventions K1-K10, 24 Nov 2026).

    python -m research.strat100.run                         every data set, every strategy
    python -m research.strat100.run --dataset b_live        one data set (comma-separated for more)
    python -m research.strat100.run --only S001,S017        some strategies (their benchmarks are added)
    python -m research.strat100.run --list                  the registry, no data read

Each data set is loaded once by its builder's loader (data_a.load_a_model, data_a.load_a_market, data_b.load_b_live:
a dict of frames with 'days', the ISO race days the data set holds). Every strategy runs inside try/except: a failure
is recorded against the strategy and the run goes on. Writes <out>/strat100_results.json and <out>/strat100_daily.csv
after each data set. Research only: nothing here places a bet, and the locked holdout is refused (core.HoldoutError).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.strat100 import core

PKG = "research.strat100"
DATASETS = {      # data set: (strategies module, loader module, loader)
    "a_model": ("strategies_a_model", "data_a", "load_a_model"),
    "a_market": ("strategies_a_market", "data_a", "load_a_market"),
    "b_live": ("strategies_b_live", "data_b", "load_b_live"),
}
TABLE = ["id", "family", "dataset", "side", "market", "select_days", "select_bets", "select_stake", "select_pnl",
         "select_ret", "select_lo", "select_hi", "confirm_days", "confirm_bets", "confirm_stake", "confirm_pnl",
         "confirm_ret", "confirm_lo", "confirm_hi", "confirm_ret_2pct", "confirm_ret_5pct", "all_bets", "all_pnl",
         "all_ret", "all_lo", "all_hi", "dsr_window", "dsr_sharpe", "sr_star", "dsr", "n_trials", "verdict", "fails",
         "paired", "pair_confirm_mean", "pair_confirm_lo", "pair_confirm_hi", "pair_select_mean", "pair_dsr",
         "pair_verdict", "flags", "voids", "unsettled", "seconds", "name"]


def say(*a):
    print(*a, flush=True)


@contextlib.contextmanager
def time_limit(seconds: float):
    """Raise TimeoutError in the block after ``seconds`` (where the platform has SIGALRM; else no limit)."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(*_):
        raise TimeoutError(f"over {seconds:.0f} s")
    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(max(1, seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def discover() -> tuple[list, dict]:
    """Every Strategy the three builders registered (STRATEGIES in each module), and what could not be imported."""
    found, problems, seen = [], {}, set()
    for ds, (mod, _, _) in DATASETS.items():
        try:
            items = list(importlib.import_module(f"{PKG}.{mod}").STRATEGIES)
        except Exception as exc:  # noqa: BLE001 -- one builder's module must not stop the others
            problems[mod] = f"{type(exc).__name__}: {exc}"
            continue
        for s in items:
            if not isinstance(s, core.Strategy) or s.dataset != ds or s.id in seen:
                problems[f"{mod}:{getattr(s, 'id', s)}"] = "not a Strategy, wrong dataset for its module, or id used twice"
                continue
            seen.add(s.id)
            found.append(s)
    return found, problems


def load(dataset: str):
    """The data set's frames from its builder's loader, the holdout refused, and its race days."""
    _, mod, fn = DATASETS[dataset]
    frames = getattr(importlib.import_module(f"{PKG}.{mod}"), fn)()
    tables = {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame)} if isinstance(frames, dict) else {}
    if isinstance(frames, pd.DataFrame):
        tables = {dataset: frames}
    if dataset != "b_live":
        for k, v in tables.items():
            core.assert_no_holdout(v, what=f"{dataset}[{k}]")
    days = frames.get("days") if isinstance(frames, dict) else None
    if days is None:
        days = sorted({d for v in tables.values() if "race_date" in v.columns
                       for d in v["race_date"].astype(str).str[:10].unique()})
    days = sorted({str(d)[:10] for d in days})
    if dataset != "b_live" and days and days[-1] >= core.HOLDOUT_FROM:
        raise core.HoldoutError(f"{dataset}: its days run to {days[-1]}")
    return frames, days, tables


def _shape(tables: dict) -> dict:
    return {k: (v.shape, tuple(v.columns)) for k, v in tables.items()}


def failed(s: core.Strategy, msg: str) -> dict:
    return {"id": s.id, "name": s.name, "family": s.family, "dataset": s.dataset, "side": s.side, "market": s.market,
            "paired": s.paired, "blind": s.blind, "benchmark": s.benchmark, "tested_before": s.tested_before,
            "note": s.note, "error": msg, "verdict": "error", "fails": "", "pair_verdict": None}


def run_one(s: core.Strategy, frames, days: list, tables: dict, draws: int, seconds: float) -> dict:
    t0, before = time.time(), _shape(tables)
    try:
        with time_limit(seconds):
            out = s.build(frames)
            bets, own_days = out if isinstance(out, tuple) else (out, None)
            r = core.score(s, bets, days if own_days is None else own_days, draws=draws)
    except Exception as exc:  # noqa: BLE001 -- recorded against the strategy; the run goes on
        r = failed(s, f"{type(exc).__name__}: {exc}")
        r["traceback"] = traceback.format_exc(limit=6)
    if _shape(tables) != before:
        r["note"] = (r.get("note", "") + " CHANGED THE SHARED FRAMES: every later strategy of this data set is "
                     "suspect").strip()
    r["seconds"] = round(time.time() - t0, 1)
    return r


def finish(readings: list, n_trials: int, draws: int) -> None:
    """One data set's readings deflated by its own strategies' Sharpes (K10 c), the pairs read, benchmarks marked."""
    good = [r for r in readings if not r.get("error")]
    sharpes = [r["dsr_sharpe"] for r in good if r["dsr_sharpe"] is not None]
    for r in good:
        core.deflate(r, n_trials, sharpes)
    by_id = {r["id"]: r for r in readings}
    pairs = [(r, by_id.get(r["paired"])) for r in good if r.get("paired")]
    ok = [(r, b) for r, b in pairs if b is not None and not b.get("error")]
    pair_sharpes = [x for x in (core.pair_sharpe(r, b) for r, b in ok) if np.isfinite(x)]
    for r, b in ok:
        core.pair(r, b, n_trials, pair_sharpes, draws)
    for r, b in pairs:
        if b is None or b.get("error"):
            r["pair_verdict"] = "not read: benchmark missing or failed"
    for r in good:
        if r.get("benchmark"):
            r["verdict"] = f"benchmark ({r['verdict']})"


def write(out: Path, readings: list, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in readings]
    (out / "strat100_results.json").write_text(json.dumps({**meta, "strategies": rows}, indent=1, default=str))
    daily = []
    for r in readings:
        if "_daily" in r:
            d = r["_daily"].assign(id=r["id"], dataset=r["dataset"])
            if "_pair" in r:
                d = d.merge(r["_pair"][["race_date", "diff"]].rename(columns={"diff": "pnl_less_benchmark"}),
                            on="race_date", how="left")
            daily.append(d)
    cols = ["id", "dataset", "race_date", "window", "bets", "stake", "pnl", "pnl_less_benchmark"]
    frame = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame(columns=cols)
    frame.reindex(columns=cols).round(4).to_csv(out / "strat100_daily.csv", index=False)


def table(readings: list) -> pd.DataFrame:
    t = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in readings])
    t = t.reindex(columns=TABLE + [c for c in ("error",) if c in t.columns])
    return t.sort_values("confirm_ret", ascending=False, na_position="last", kind="stable")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated strategy ids")
    ap.add_argument("--dataset", default="", help="comma-separated data sets: " + ", ".join(DATASETS))
    ap.add_argument("--max-minutes", type=float, default=330.0, help="the whole run's guard: nothing new starts after it")
    ap.add_argument("--strategy-minutes", type=float, default=30.0, help="one strategy's guard")
    ap.add_argument("--draws", type=int, default=core.DRAWS)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--list", action="store_true", help="print the registry and stop")
    a = ap.parse_args(argv)
    t0 = time.time()
    pd.set_option("display.width", 400, "display.max_columns", None, "display.max_colwidth", 70)

    registered, problems = discover()
    for k, v in problems.items():
        say(f"REGISTRY PROBLEM {k}: {v}")
    if a.list:
        cols = ("id", "dataset", "family", "side", "market", "paired", "blind", "name", "tested_before")
        if registered:
            say(pd.DataFrame([{k: getattr(s, k) for k in cols} for s in registered]).to_string(index=False))
        say(f"{len(registered)} strategies registered")
        return 0
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    sets = {x.strip() for x in a.dataset.split(",") if x.strip()}
    if sets - set(DATASETS):
        ap.error(f"--dataset: {sorted(sets - set(DATASETS))} not one of {list(DATASETS)}")
    chosen = [s for s in registered if (not sets or s.dataset in sets) and (not only or s.id in only)]
    need = {s.paired for s in chosen if s.paired} - {s.id for s in chosen}
    chosen += [s for s in registered if s.id in need]                      # a paired strategy brings its benchmark
    trials = core.n_trials_for(len(registered))
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **trials,
            "strategies_run": len(chosen), "registry_problems": problems, "draws": a.draws, "seed": core.SEED,
            "commission": core.COMMISSION, "windows": core.WINDOWS, "holdout_from": core.HOLDOUT_FROM,
            "partial": bool(only or sets),
            "note": "research only: a PASS is a candidate for a registered forward paper test; no strategy here "
                    "places a bet or changes the live trader"}
    say(f"strat100: {len(registered)} registered, {len(chosen)} to run; ledger rules {trials['ledger_rules']}, "
        f"n_trials {trials['n_trials']}; draws {a.draws}, seed {core.SEED}")
    if meta["partial"]:
        say("PARTIAL RUN: the deflated Sharpe's spread is taken from the strategies run, not the registered reading")

    readings: list[dict] = []

    def left() -> float:
        return a.max_minutes * 60 - (time.time() - t0)

    for ds in DATASETS:
        todo = [s for s in chosen if s.dataset == ds]
        if not todo:
            continue
        mine: list[dict] = []
        try:
            if left() <= 0:
                raise TimeoutError("not run: the --max-minutes guard was reached before this data set")
            with time_limit(left()):
                frames, days, tables = load(ds)
            say(f"\n== {ds}: {len(days)} race days {days[0] if days else '-'} .. {days[-1] if days else '-'}; "
                + ", ".join(f"{k} {len(v):,}" for k, v in tables.items()) + f"; {len(todo)} strategies")
        except Exception as exc:  # noqa: BLE001 -- the data set's strategies carry the error; the others still run
            say(f"\n== {ds}: NOT LOADED ({type(exc).__name__}: {exc})")
            mine = [failed(s, f"data: {type(exc).__name__}: {exc}") for s in todo]
            todo = []
        for s in todo:
            if left() <= 0:
                mine.append(failed(s, "not run: the --max-minutes guard was reached"))
                continue
            r = run_one(s, frames, days, tables, a.draws, min(a.strategy_minutes * 60, left()))
            say(f"  {s.id:10} {r['seconds']:7.1f}s  confirm bets {r.get('confirm_bets', '-')}, "
                f"ret {r.get('confirm_ret', '-')}" + (f"  ERROR {r['error']}" if r.get("error") else ""))
            mine.append(r)
        finish(mine, trials["n_trials"], a.draws)
        readings += mine
        frames = tables = None                                             # the next data set needs the memory
        write(a.out, readings, meta)

    if not readings:
        say("no strategy to run")
        write(a.out, readings, meta)
        return 0
    t = table(readings)
    say("\n== the reading, by CONFIRM return (per GBP staked for backs, per GBP of liability for lays; 90% interval "
        "by race day; a PASS is a candidate for a forward paper test only)")
    say(t.to_string(index=False))
    errs = [r for r in readings if r.get("error")]
    for r in errs:
        say(f"ERROR {r['id']}: {r['error']}")
    counts = t["verdict"].value_counts().to_dict()
    say(f"\ntrials: {len(registered)} registered, {len(readings)} run, {len(errs)} errors; ledger rules "
        f"{trials['ledger_rules']}; n_trials used for the deflated Sharpe {trials['n_trials']}")
    say("verdicts: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    say(f"wrote {a.out / 'strat100_results.json'} and {a.out / 'strat100_daily.csv'} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
