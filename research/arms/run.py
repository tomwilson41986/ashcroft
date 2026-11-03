"""Run the strategy arms over the recorded days, a day a process, and print the tables (read-only).

    python -m research.arms.run --days 2026-10-30,2026-10-31 --arms all --jobs 2 --out out/arms
    python -m research.arms.run --days 2026-10-12..2026-11-02 --arms baseline,floor30,lays --jobs 4 --max-minutes 240

Arms (the groups share a day's loaded books):
  win books:   baseline floor30 floor50 floor30_11 bar1 bar2 stop0900 topups0 topups2 lays lays10 book book_lays
               book0800_lays exits
  place books: place3 place6
  the evening before: evening
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.arms import core  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

WIN_ARMS = {
    "baseline": ("rule", {}, {}),
    "floor30": ("rule", {"floor_long": (21.0, 30.0)}, {}),
    "floor50": ("rule", {"floor_long": (21.0, 50.0)}, {}),
    "floor30_11": ("rule", {"floor_long": (11.0, 30.0)}, {}),
    "bar1": ("rule", {"bar_slope": 0.01}, {}),
    "bar2": ("rule", {"bar_slope": 0.02}, {}),
    "stop0900": ("rule", {"stop_after": "09:00"}, {}),
    "topups0": ("rule", {}, {"max_topups": 0}),
    "topups2": ("rule", {}, {"max_topups": 2}),
    "lays": ("lay", {"lay_bar": 0.03, "max_lay_price": 20.0}, {}),
    "lays10": ("lay", {"lay_bar": 0.03, "max_lay_price": 10.0}, {}),
    "book": ("book", {"lays": False}, {}),
    "book_lays": ("book", {"lays": True}, {}),
    "book0800_lays": ("book", {"lays": True, "book_from": "08:00"}, {}),
}
PLACE_ARMS = {"place3": {"clv_bar": 0.03}, "place6": {"clv_bar": 0.06}}
ALL = list(WIN_ARMS) + ["exits"] + list(PLACE_ARMS) + ["evening"]


def parse_days(s: str) -> list[date]:
    if ".." in s:
        a, b = s.split("..")
        a, b = date.fromisoformat(a), date.fromisoformat(b)
        return [a + timedelta(n) for n in range((b - a).days + 1)]
    return [date.fromisoformat(x) for x in s.split(",") if x]


def _session_for(kind: str):
    from research.arms import sessions
    return {"rule": sessions.ArmSession, "lay": sessions.LaySession, "book": sessions.RaceBookSession}[kind]


def run_win_arms(D, arms: list[str], out: Path, draws: int) -> tuple[list[dict], pd.DataFrame]:
    from research.arms import sessions
    from research.arms.exits import exits_day
    rows, exit_rows = [], pd.DataFrame()
    t0 = core.session_start(D, "win")
    for name in arms:
        if name == "exits":
            exit_rows = exits_day(D)
            continue
        kind, policy, over = WIN_ARMS[name]
        cfg = core.make_cfg(clv_draws=draws, **over)
        core.set_plan_cfg(core.make_cfg(clv_draws=draws), reset=False)
        pol = {**policy, "_D": D}
        tic = time.time()
        led, s, x = core.run_session(D, _session_for(kind), cfg, pol, t0)
        if kind == "rule":
            r = core.score(led, D, name, s.state.settled_pnl, s.state.turnover >= cfg.limits.max_daily_turnover - 2)
            b = led[(led.get("event") == "back") & (pd.to_numeric(led.get("matched"), errors="coerce") > 0)] \
                if len(led) else led
            r["last_back_uk"] = (pd.to_datetime(b["ts"], utc=True).max().tz_convert("Europe/London").strftime("%H:%M")
                                 if len(b) else None)
            led.to_csv(out / f"ledger_{D.day}_{name}.csv", index=False)
        elif kind == "lay":
            r = core.score_side(s.lay_positions_frame(), D, name, side="lay")
            r["fills"] = int(sum(1 for z in s.lay_rows if z["event"] == "lay" and z["matched"] > 0))
            pd.DataFrame(s.lay_rows).to_csv(out / f"lays_{D.day}_{name}.csv", index=False)
        else:
            r = sessions.score_book(s.book_positions_frame(), D, name)
            pd.DataFrame(s.book_rows).to_csv(out / f"book_{D.day}_{name}.csv", index=False)
        r["seconds"] = round(time.time() - tic)
        rows.append(r)
        print({k: v for k, v in r.items() if k not in ("day",)}, flush=True)
    return rows, exit_rows


def run_place_arms(day: date, arms: list[str], out: Path, draws: int, db: str) -> list[dict]:
    from research.arms import place, sessions
    rows = []
    model_path = place.fit_place_closing(db, day, out / "place_models")
    if model_path is None:
        return [{"day": str(day), "arm": a, "note": "no place closing model (panel too thin)"} for a in arms]

    def preds_fn(markets, places):
        return place.place_prices(core.predictions(day), markets, places)

    D = core.load_day(day, "place", preds_fn=preds_fn)
    if D is None:
        return [{"day": str(day), "arm": a, "note": "no place day (file or books missing)"} for a in arms]
    print("  place prices: ", D.preds[["p_place", "places"]].describe().loc[["mean", "min", "max"]].round(3).to_dict()
          if "p_place" in D.preds else "none")
    t0 = core.session_start(D, "win")
    for name in arms:
        over = PLACE_ARMS[name]
        cfg = core.make_cfg(clv_draws=draws, closing_model_novol=model_path, closing_model=model_path, **over)
        core.set_plan_cfg(core.make_cfg(clv_draws=draws, closing_model_novol=model_path, closing_model=model_path,
                                        **over), reset=True)
        tic = time.time()
        led, s, x = core.run_session(D, sessions.ArmSession, cfg, {"_D": D}, t0)
        r = core.score(led, D, name, s.state.settled_pnl, s.state.turnover >= cfg.limits.max_daily_turnover - 2)
        r["seconds"] = round(time.time() - tic)
        led.to_csv(out / f"ledger_{D.day}_{name}.csv", index=False)
        rows.append(r)
        print({k: v for k, v in r.items() if k != "day"}, flush=True)
    core.set_plan_cfg(core.make_cfg(clv_draws=draws), reset=True)
    return rows


def run_evening(day: date, out: Path, draws: int) -> tuple[list[dict], dict]:
    from research.arms import sessions
    D = core.load_day(day, "evening")
    if D is None:
        return [{"day": str(day), "arm": "evening", "note": "no evening books"}], {}
    sizes = core.log_sizes(D, "evening")
    cfg = core.make_cfg(clv_draws=draws)
    core.set_plan_cfg(core.make_cfg(clv_draws=draws), reset=True)
    t0 = core.session_start(D, "evening")
    t_end = (datetime(day.year, day.month, day.day, 21, 45, tzinfo=core.UK) - timedelta(days=1)).astimezone(timezone.utc)
    tic = time.time()
    led, s, x = core.run_session(D, sessions.ArmSession, cfg, {"_D": D}, t0, t_end=t_end,
                                 stale=timedelta(minutes=15))
    r = core.score(led, D, "evening", s.state.settled_pnl, s.state.turnover >= cfg.limits.max_daily_turnover - 2)
    r["seconds"] = round(time.time() - tic)
    skips = led[led.get("event") == "skip"] if len(led) and "event" in led else pd.DataFrame()
    r["skips"] = int(len(skips))
    led.to_csv(out / f"ledger_{D.day}_evening.csv", index=False)
    print({k: v for k, v in r.items() if k != "day"}, flush=True)
    return [r], sizes


def day_job(day_iso: str, arms: list[str], out: str, draws: int, db: str) -> dict:
    day = date.fromisoformat(day_iso)
    out_p = core.ensure_dir(out)
    res = {"day": day_iso, "rows": [], "exits": None, "evening_sizes": {}, "errors": []}
    win = [a for a in arms if a in WIN_ARMS or a == "exits"]
    plc = [a for a in arms if a in PLACE_ARMS]
    eve = "evening" in arms
    try:
        if win:
            D = core.load_day(day, "win")
            if D is None:
                res["rows"] += [{"day": day_iso, "arm": a, "note": "no price file yet"} for a in win]
            else:
                rows, ex = run_win_arms(D, win, out_p, draws)
                res["rows"] += rows
                if ex is not None and len(ex):
                    res["exits"] = ex.to_dict("records")
                res["win_sizes"] = core.log_sizes(D, "win")
                del D
    except Exception:
        res["errors"].append(f"win arms: {traceback.format_exc()}")
    try:
        if plc:
            res["rows"] += run_place_arms(day, plc, out_p, draws, db)
    except Exception:
        res["errors"].append(f"place arms: {traceback.format_exc()}")
    try:
        if eve:
            rows, sizes = run_evening(day, out_p, draws)
            res["rows"] += rows
            res["evening_sizes"] = sizes
    except Exception:
        res["errors"].append(f"evening: {traceback.format_exc()}")
    return res


def report(rows: pd.DataFrame, exits: pd.DataFrame, sizes: dict, out: Path) -> None:
    from research.arms.exits import summarise_exits
    rows.to_csv(out / "arms_by_day.csv", index=False)
    done = rows[rows.get("note").isna()] if "note" in rows else rows
    if "note" in rows and rows["note"].notna().any():
        print("\n== days skipped\n" + rows[rows["note"].notna()][["day", "arm", "note"]].to_string(index=False))
    rule = done[done.arm.isin([a for a, (k, _, _) in WIN_ARMS.items() if k == "rule"] + list(PLACE_ARMS) + ["evening"])]
    if len(rule):
        print("\n== the rule's arms (backs closed at the BSP): the days together")
        t = rule.groupby("arm", sort=False).agg(days=("day", "nunique"), horses=("horses", "sum"),
                                                staked=("staked", "sum"), clv_x_stake=("clv_x_stake", "sum"),
                                                clv_x_stake_reduced=("clv_x_stake_reduced", "sum"),
                                                result=("result", "sum"), result_reduced=("result_reduced", "sum"),
                                                first_staked=("first_staked", "sum"),
                                                topup_staked=("topup_staked", "sum"),
                                                limit_days=("day_limit", "sum"))
        t["clv"] = t.clv_x_stake / t.staked
        t["clv_reduced"] = t.clv_x_stake_reduced / t.staked
        t["a_day_reduced"] = t.result_reduced / t.days
        print(t.round(4).to_string())
        print("\n== result after the cuts, by day")
        print(rule.pivot_table(index="arm", columns="day", values="result_reduced", sort=False).round(0).to_string())
        print("\n== CLV at the fill, by day")
        print(rule.pivot_table(index="arm", columns="day", values="clv", sort=False).round(3).to_string())
    lays = done[done.arm.isin([a for a, (k, _, _) in WIN_ARMS.items() if k == "lay"])]
    if len(lays):
        print("\n== the lay side (closed by a back at the BSP): the days together")
        t = lays.groupby("arm", sort=False).agg(days=("day", "nunique"), horses=("horses", "sum"),
                                                staked=("staked", "sum"), liability=("liability", "sum"),
                                                clv_x_stake=("clv_x_stake", "sum"),
                                                clv_x_stake_reduced=("clv_x_stake_reduced", "sum"),
                                                result=("result", "sum"), result_reduced=("result_reduced", "sum"))
        t["clv_on_stake"] = t.clv_x_stake / t.staked
        t["clv_on_liability"] = t.clv_x_stake / t.liability
        t["a_day_reduced"] = t.result_reduced / t.days
        print(t.round(4).to_string())
        print(lays.pivot_table(index="arm", columns="day", values="result_reduced", sort=False).round(0).to_string())
    books = done[done.arm.isin([a for a, (k, _, _) in WIN_ARMS.items() if k == "book"])]
    if len(books):
        print("\n== the race books (closed at the BSP; held to the result for information): the days together")
        t = books.groupby("arm", sort=False).agg(days=("day", "nunique"), races=("races", "sum"),
                                                 positions=("positions", "sum"), backs=("backs", "sum"),
                                                 lays=("lays", "sum"), exposure=("exposure", "sum"),
                                                 clv_x_stake=("clv_x_stake", "sum"), result=("result", "sum"),
                                                 result_held=("result_held", "sum"))
        t["clv_on_exposure"] = t.clv_x_stake / t.exposure
        t["a_day"] = t.result / t.days
        print(t.round(4).to_string())
        print(books.pivot_table(index="arm", columns="day", values="result", sort=False).round(0).to_string())
    if exits is not None and len(exits):
        exits.to_csv(out / "exits_by_fill.csv", index=False)
        print("\n== exits before the off, on the trader's own fills (profit after 2%; vs_sp: against the lay at the BSP)")
        print(summarise_exits(exits).to_string())
        fired = exits[exits.fired & exits.rule.str.startswith(("tp", "sl"))]
        if len(fired):
            print("\n   when a rule fired: minutes to the off (median) and the BSP's side of it (what the SP hedge "
                  "would have made on those fills)")
            f = fired.assign(sp_profit=lambda d: d.matched * (d.price / d.bsp - 1)).groupby("rule").agg(
                fills=("matched", "size"), minutes_to_off=("minutes_to_off", "median"),
                exit_profit=("profit", "sum"), sp_profit=("sp_profit", "sum"))
            print(f.round(1).to_string())
    for kind, key in (("win", "win_sizes"), ("evening", "evening_sizes")):
        s = sizes.get(key) or {}
        if s:
            print(f"\n== the {kind} books by UK hour (median over days: polls, markets, runners two-sided, spread, "
                  f"best-back size, matched)")
            print(pd.DataFrame(s).T.round(3).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default="out/arms")
    ap.add_argument("--draws", type=int, default=20000, help="closing draws a race (the live config's 20000)")
    ap.add_argument("--max-minutes", type=float, default=280.0)
    ap.add_argument("--db", default=str(ROOT / "horse_racing.db"))
    a = ap.parse_args(argv)
    days = parse_days(a.days)
    arms = ALL if a.arms == "all" else [x for x in a.arms.split(",") if x]
    bad = [x for x in arms if x not in ALL]
    if bad:
        ap.error(f"unknown arms {bad}; known: {ALL}")
    out = core.ensure_dir(a.out)
    print(f"arms {arms} on {len(days)} days ({days[0]}..{days[-1]}), {a.jobs} processes, {a.draws} draws, "
          f"at most {a.max_minutes:.0f} minutes", flush=True)
    tic = time.time()
    deadline = tic + a.max_minutes * 60
    rows, exits, sizes = [], [], {"win_sizes": [], "evening_sizes": []}
    results = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(day_job, d.isoformat(), arms, str(out), a.draws, a.db): d for d in days}
        try:
            for f in as_completed(futs, timeout=max(60.0, deadline - time.time())):
                d = futs[f]
                try:
                    r = f.result()
                except Exception as exc:                      # a day that died
                    r = {"day": d.isoformat(), "rows": [], "exits": None, "errors": [repr(exc)]}
                results[d] = r
                for e in r.get("errors") or []:
                    print(f"!! {d}: {e}", flush=True)
                rows += r.get("rows") or []
                if r.get("exits"):
                    exits += r["exits"]
                for key in ("win_sizes", "evening_sizes"):
                    if r.get(key):
                        sizes[key].append(pd.DataFrame(r[key]).T)
                print(f"-- {d} done ({(time.time() - tic) / 60:.0f} min in)", flush=True)
        except TimeoutError:
            print(f"!! the time budget ran out after {(time.time() - tic) / 60:.0f} minutes: "
                  f"{len(days) - len(results)} days not finished", flush=True)
            for f in futs:
                f.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
    med = {}
    for key in ("win_sizes", "evening_sizes"):
        if sizes[key]:
            allv = pd.concat(sizes[key])
            med[key] = allv.groupby(level=0).median().to_dict("index")
    (out / "sizes.json").write_text(json.dumps(med, indent=1, default=str))
    rows_df = pd.DataFrame(rows)
    report(rows_df, pd.DataFrame(exits), med, out)
    print(f"\ndone in {(time.time() - tic) / 60:.0f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
