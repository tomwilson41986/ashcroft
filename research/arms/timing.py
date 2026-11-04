"""When to bet (the owner, 4 Nov 2026: "Evaluate when is the best time to bet; both on prices and market depth /
liquidity. Night before? At 4am? ... balance available bet sizes vs prices. Model the optimal timing for profit").

Read-only research on the recorded books, through the live trader's own code (research/arms/core.py): the owner's
rule replayed from different start times, with the GBP4,000 day limit and without it, every back closed at the BSP,
2% on the winnings, and scored after the non-runners' reduction factors (the honest figure; live has run at 0.80 of
the replay). What the books hold decides what can be taken: each level the replay takes stays taken, so the price
and the size are judged together, not apart.

The arms (each a start time, the session stopping 15 minutes before each off as live):
  live         06:15 UK, the GBP4,000 limit (the rule as it trades)
  from0515     05:15 UK: the trader's books before 06:15 exist only on GMT days (26 Oct on; in BST the session's
               05:17 UTC start is 06:17 UK), so it is read on those days
  evening      17:00 UK the evening before (the evening record, ten-minute snapshots), then the morning; one day limit
               across both; books up to 15 minutes old are read (the evening's snapshots are ten minutes apart)
  live_s15     06:15 with the same 15-minute staleness: the evening arm's own baseline
  from0700, from0800  later starts (the real trader's own fills are already out of these books: the later arms are
               read low by that footprint)
  uncapped     06:15 with no day limit and no funds limit: every back the rule wants, so the money and the value at
               each hour, and the result at any day limit (the capped run is the same session cut at the limit)
  uncapped_evening, uncapped_0515   the same from the evening before and from 05:15
  uncapped_evening_noguard   the evening without the live guard of GBP100 matched on the runner (the evening's
               runners have little matched: is it the guard or the price that keeps the evening out)
  win1000      uncapped, GBP1,000 to win a horse (GBP500 a bet at most): would bigger stakes find the size, and when

Nothing at 04:00 or overnight can be measured: no book is recorded between the evening record (to about 22:00 UK)
and the trader's start (05:17 UTC). The report says so and the overnight record (books_overnight, from the nightly
prices job) fills it; the evening arm reads it once it is there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.arms import core  # noqa: E402

BIG = 1e7
ARMS = {
    #  name:              (kind, trade_from, stale minutes, overrides, balance)
    "live": ("win", "06:15", 6, {}, core.BALANCE),
    "from0515": ("win", "05:15", 10, {}, core.BALANCE),
    "evening": ("evening_morning", "06:15", 15, {}, core.BALANCE),
    "live_s15": ("win", "06:15", 15, {}, core.BALANCE),
    "from0700": ("win", "07:00", 6, {}, core.BALANCE),
    "from0800": ("win", "08:00", 6, {}, core.BALANCE),
    "uncapped": ("win", "06:15", 6, {"limits.max_daily_turnover": BIG}, BIG),
    "uncapped_evening": ("evening_morning", "06:15", 15, {"limits.max_daily_turnover": BIG}, BIG),
    "uncapped_evening_noguard": ("evening_morning", "06:15", 15, {"limits.max_daily_turnover": BIG,
                                                                  "limits.min_runner_matched": 0.0}, BIG),
    "uncapped_0515": ("win", "05:15", 10, {"limits.max_daily_turnover": BIG}, BIG),
    "win1000": ("win", "06:15", 6, {"limits.max_daily_turnover": BIG, "target": 1000.0,
                                    "limits.max_stake": 500.0}, BIG),
}
LIMITS = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]


def _start(D, kind: str, trade_from: str) -> datetime:
    d = D.day
    if kind == "evening_morning":
        return (datetime(d.year, d.month, d.day, 17, 0, tzinfo=core.UK) - timedelta(days=1)).astimezone(timezone.utc)
    h, m = (int(x) for x in trade_from.split(":"))
    return (datetime(d.year, d.month, d.day, h, m, tzinfo=core.UK) - timedelta(minutes=2)).astimezone(timezone.utc)


def fills(led: pd.DataFrame, D, arm: str) -> pd.DataFrame:
    """Every back matched, with its BSP and result, the price after the reductions of the runners taken out after
    it, and what it locked in at the BSP (before commission): stake x (price / BSP - 1), and the same at the cut
    price when the horse won."""
    if led is None or led.empty or "event" not in led:
        return pd.DataFrame()
    led = led.reindex(columns=list(dict.fromkeys([*led.columns, "result", "bsp", "matched", "avg_price", "ts"])))
    st = led[(led.event == "settle") & (led.result != "REMOVED")].copy()
    st["bsp"] = pd.to_numeric(st["bsp"], errors="coerce")
    st = st[np.isfinite(st.bsp) & (st.bsp > 1)].drop_duplicates(["market_id", "selection_id"])
    b = led[(led.event == "back") & (pd.to_numeric(led.matched, errors="coerce") > 0)].copy()
    if b.empty or st.empty:
        return pd.DataFrame()
    b["matched"] = pd.to_numeric(b.matched, errors="coerce")
    b["avg_price"] = pd.to_numeric(b.avg_price, errors="coerce")
    b["k"] = b.groupby(["market_id", "selection_id"]).cumcount()
    b = b.drop(columns=[c for c in ("bsp", "result") if c in b]).merge(
        st[["market_id", "selection_id", "bsp", "result"]], on=["market_id", "selection_id"])
    t = pd.to_datetime(b.ts, utc=True)
    starts = pd.Series({m.market_id: m.start for m in D.markets})
    rem = D.removed
    fac = []
    for r, tt in zip(b.itertuples(), t):
        x = rem[(rem.market_id == r.market_id) & (rem.selection_id != r.selection_id) & (rem.rt > tt)] \
            if len(rem) else rem
        fac.append(float(np.prod(1 - x.af.to_numpy() / 100.0)) if len(x) else 1.0)
    b["p_red"] = np.maximum(1.01, b.avg_price.to_numpy() * np.array(fac))
    b["locked"] = b.matched * (b.avg_price / b.bsp - 1)
    won = b.result.eq("WINNER")
    b["locked_red"] = b.locked - np.where(won, b.matched * (b.avg_price - b.p_red), 0.0)
    b["t"] = t.to_numpy()
    b["uk"] = t.dt.tz_convert("Europe/London")
    off = pd.to_datetime(b.market_id.map(starts), utc=True)
    b["mins_to_off"] = (off - t).dt.total_seconds() / 60.0
    b["evening"] = b["uk"].dt.date < D.day
    b["day"], b["arm"] = str(D.day), arm
    return b[["day", "arm", "t", "uk", "evening", "mins_to_off", "market_id", "selection_id", "k", "matched",
              "avg_price", "p_red", "bsp", "result", "locked", "locked_red"]]


def day_job(day_iso: str, arms: list[str], out: str, draws: int) -> dict:
    from research.arms import sessions
    day = date.fromisoformat(day_iso)
    res = {"day": day_iso, "rows": [], "fills": [], "sizes": {}, "errors": []}
    loaded = {}
    for name in arms:
        kind, frm, stale, over, balance = ARMS[name]
        try:
            if kind not in loaded:
                loaded[kind] = core.load_day(day, kind)
                if loaded[kind] is not None:
                    res["sizes"][kind] = core.log_sizes(loaded[kind], kind)
            D = loaded[kind]
            if D is None:
                res["rows"].append({"day": day_iso, "arm": name, "note": "no books or no price file"})
                continue
            cfg = core.make_cfg(clv_draws=draws, trade_from=frm, **over)
            core.set_plan_cfg(core.make_cfg(clv_draws=draws), reset=False)
            tic = time.time()
            led, s, x = core.run_session(D, sessions.ArmSession, cfg, {"_D": D}, _start(D, kind, frm),
                                         balance=balance, stale=timedelta(minutes=stale))
            limit = cfg.limits.max_daily_turnover
            r = core.score(led, D, name, s.state.settled_pnl, s.state.turnover >= limit - 2)
            f = fills(led, D, name)
            if len(f):
                r["first_uk"] = f.uk.min().strftime("%m-%d %H:%M")
                r["evening_staked"] = round(float(f.loc[f.evening, "matched"].sum()), 2)
                res["fills"].append(f.assign(uk=f.uk.astype(str), t=f.t.astype(str)).to_dict("records"))
            r["seconds"] = round(time.time() - tic)
            res["rows"].append(r)
            print({k: v for k, v in r.items() if k != "day"}, flush=True)
        except Exception:
            res["errors"].append(f"{name}: {traceback.format_exc()}")
    return res


def _median_clock(ts: pd.Series):
    """The median UK clock time of a set of moments, the evening before read as before midnight."""
    if ts.empty:
        return None
    m = [(x.hour * 60 + x.minute) - (1440 if x.date() < date.fromisoformat(d) else 0) for d, x in ts.items()]
    v = int(np.median(m))
    return ("eve " if v < 0 else "") + f"{(v % 1440) // 60:02d}:{v % 60:02d}"


def _hour_label(r) -> str:
    return ("eve " if r.evening else "") + r.uk.strftime("%H")


def report(rows: pd.DataFrame, f: pd.DataFrame, sizes: dict, out: Path) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if rows.empty:
        print("nothing replayed")
        return
    ok = rows[rows.get("note").isna()] if "note" in rows else rows
    skip = rows[rows["note"].notna()] if "note" in rows else pd.DataFrame()
    if len(skip):
        print("\n== days skipped\n" + skip[["day", "arm", "note"]].to_string(index=False))
    gmt = ok["day"] >= "2026-10-26"

    def together(sub):
        g = sub.groupby("arm").agg(days=("day", "nunique"), horses=("horses", "sum"), staked=("staked", "sum"),
                                   clv_x_stake=("clv_x_stake", "sum"),
                                   clv_x_stake_reduced=("clv_x_stake_reduced", "sum"),
                                   result=("result", "sum"), result_reduced=("result_reduced", "sum"),
                                   limit_days=("day_limit", "sum"))
        g["clv"] = g.clv_x_stake / g.staked
        g["clv_reduced"] = g.clv_x_stake_reduced / g.staked
        g["a_day_reduced"] = g.result_reduced / g.days
        g["a_day_at_0.80"] = 0.8 * g.a_day_reduced
        return g.reindex([a for a in ARMS if a in g.index]).round(4)

    print("\n== the arms, every day replayed (result after 2%, then after the non-runners' cuts; scale 1.0)")
    print(together(ok).to_string())
    print("\n== GMT days only (26 Oct on: the books before 06:15 exist)")
    print(together(ok[gmt]).to_string())
    days_eve = set(ok.loc[(ok.arm == "evening") & (ok.get("evening_staked", 0) > 0), "day"])
    print(f"\n== result after the cuts by day (the evening arm staked in the evening on {len(days_eve)} days)")
    print(ok.pivot_table(index="arm", columns="day", values="result_reduced").reindex(
        [a for a in ARMS if a in set(ok.arm)]).round(0).to_string())
    if f.empty:
        return
    f = f.copy()
    f["uk"] = pd.to_datetime(f.uk, utc=True).dt.tz_convert("Europe/London")
    f["evening"] = f.evening.astype(bool)
    f["hour"] = [_hour_label(r) for r in f.itertuples()]
    ndays = f.groupby("arm")["day"].nunique()

    def by(col, arm):
        g = f[f.arm == arm].groupby(col, sort=True).agg(fills=("matched", "size"), staked=("matched", "sum"),
                                                        locked=("locked", "sum"), locked_red=("locked_red", "sum"),
                                                        horses=("selection_id", "nunique"))
        g["clv"] = g.locked / g.staked
        g["clv_reduced"] = g.locked_red / g.staked
        g["staked_a_day"] = g.staked / ndays.get(arm, 1)
        g["locked_red_a_day"] = g.locked_red / ndays.get(arm, 1)
        return g.round(4)

    for arm in ("uncapped_evening", "uncapped", "uncapped_0515", "win1000", "live", "evening"):
        if arm in set(f.arm):
            print(f"\n== {arm}: the backs by UK hour entered (locked = stake x (price / BSP - 1) before commission; "
                  f"reduced = after the cuts; a day = over {ndays.get(arm)} days)")
            print(by("hour", arm).to_string())
    bins = [-1, 15, 30, 60, 120, 240, 480, 720, 1080, 1e9]
    labels = ["15", "15-30", "30-60", "1-2h", "2-4h", "4-8h", "8-12h", "12-18h", "18h+"]
    f["to_off"] = pd.cut(f.mins_to_off, bins=bins, labels=labels)
    for arm in ("uncapped_evening", "uncapped"):
        if arm in set(f.arm):
            print(f"\n== {arm}: the backs by time to the off")
            print(by("to_off", arm).to_string())
    # the result at every day limit: the uncapped session cut where its turnover reaches the limit (the capped
    # session is the same session until then: the same reads, the same plans, the same fills)
    for arm in ("uncapped", "uncapped_evening", "uncapped_0515", "win1000"):
        g = f[f.arm == arm].sort_values(["day", "t"])
        if g.empty:
            continue
        g = g.assign(cum=g.groupby("day").matched.cumsum())
        rows_l = []
        for L in LIMITS:
            h = g[g.cum - g.matched < L].copy()
            h["scale"] = np.where(h.cum > L, (L - (h.cum - h.matched)) / h.matched, 1.0)
            lr = (h.locked_red * h.scale).sum()
            st = (h.matched * h.scale).sum()
            lr_net = lr - 0.02 * max(lr, 0)
            rows_l.append({"day_limit": L, "staked_a_day": st / ndays[arm], "clv_reduced": lr / st if st else np.nan,
                           "locked_red_a_day": lr / ndays[arm], "after_2pc_a_day_approx": lr_net / ndays[arm],
                           "at_0.80": 0.8 * lr_net / ndays[arm],
                           "days_reaching_it": int((g.groupby("day").cum.max() >= L).sum()),
                           "reached_by_uk_median": _median_clock(g[g.cum >= L].groupby("day").uk.min())})
        print(f"\n== {arm}: the result at each day limit (the session cut at the limit; locked after the cuts, "
              f"2% on the day's net as an approximation)")
        print(pd.DataFrame(rows_l).round(3).to_string(index=False))
    for kind, s in sizes.items():
        if s:
            print(f"\n== the {kind} books by UK hour (median over days: polls, markets, runners, two-sided, spread, "
                  f"best-back size, market matched)")
            print(pd.DataFrame(s).T.round(3).to_string())
    f.drop(columns=["uk"]).to_csv(out / "fills.csv.gz", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True, help="a..b or a list")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="out/timing")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--max-minutes", type=float, default=290.0)
    a = ap.parse_args(argv)
    if ".." in a.days:
        x, y = (date.fromisoformat(v) for v in a.days.split(".."))
        days = [x + timedelta(n) for n in range((y - x).days + 1)]
    else:
        days = [date.fromisoformat(v) for v in a.days.split(",") if v]
    arms = list(ARMS) if a.arms == "all" else [v for v in a.arms.split(",") if v]
    bad = [v for v in arms if v not in ARMS]
    if bad:
        ap.error(f"unknown arms {bad}; known {list(ARMS)}")
    out = core.ensure_dir(a.out)
    print(f"timing arms {arms} on {len(days)} days ({days[0]}..{days[-1]}), {a.jobs} processes", flush=True)
    print("no books are recorded between the evening record (to about 22:00 UK) and the trader's start (05:17 UTC):"
          " 04:00 and the night cannot be read until the overnight record (books_overnight) is kept", flush=True)
    tic = time.time()
    deadline = tic + a.max_minutes * 60
    rows, fl, sizes = [], [], {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(day_job, d.isoformat(), arms, str(out), a.draws): d for d in days}
        try:
            for fut in as_completed(futs, timeout=max(60.0, deadline - time.time())):
                d = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    r = {"day": d.isoformat(), "rows": [], "fills": [], "sizes": {}, "errors": [repr(exc)]}
                for e in r["errors"]:
                    print(f"!! {d}: {e}", flush=True)
                rows += r["rows"]
                for part in r["fills"]:
                    fl += part
                for kind, s in r["sizes"].items():
                    if s:
                        sizes.setdefault(kind, []).append(pd.DataFrame(s).T)
                print(f"-- {d} done ({(time.time() - tic) / 60:.0f} min in)", flush=True)
        except TimeoutError:
            print(f"!! the time budget ran out after {(time.time() - tic) / 60:.0f} minutes", flush=True)
            for fut in futs:
                fut.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
    med = {k: pd.concat(v).groupby(level=0).median().to_dict("index") for k, v in sizes.items()}
    (out / "sizes.json").write_text(json.dumps(med, indent=1, default=str))
    report(pd.DataFrame(rows), pd.DataFrame(fl), med, out)
    print(f"\ndone in {(time.time() - tic) / 60:.0f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
