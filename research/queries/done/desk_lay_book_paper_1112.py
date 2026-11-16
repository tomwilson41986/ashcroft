"""The lay book's 30-day forward paper test (ledger `desk-lay-book-paper-1112`, registered before its first day).

The owner, 12 Nov 2026: "start the 30-day forward paper test of the lay book, with no money at risk". No order is
placed and no Betfair call is made: each forward day is replayed from the live trader's own recorded books
(s3://.../betfair_live/<day>/, research/arms/core.py load_day) through the desk (desk/session.py DeskSession with
desk/algos/lay_book.py LayBook), with the strategy's registered params and limits
(desk/strategies/horse_racing/horse-lay-book.json), unchanged. A day's result is research/arms/core.py score_side
result_reduced: each lay closed by a back at the BSP of Betfair's price file, after the non-runners' reduction factors
and 2% on each market's net.

The window: forward days only, from 2026-11-13 (the first day after registration) to yesterday. A day with no recorded
books or no price file yet is "no data found" and waits; it is never guessed.

Blind until complete: while fewer than 30 forward days have data, the query prints only the count and stops (no
replay, no P&L). Then it replays the first 30 forward days with data (later days only while the fills are under 300;
under 300 it prints only the fill count). Once both are met it reads the pinned gate once: a positive sum, the 90% day-bootstrap interval of the mean above zero and a
deflated Sharpe >= 0.95 with every horse-racing trial in the ledger counted (desk/trials.py count_trials).

A pass makes the lay book a candidate for pilot, which is the owner's decision alone (after the lock-down and the
DESK_* envelope in docs/agent_desk.md); a fail returns it to research. The fills are the replay's taker fills on
3-level books in 1-minute steps; real fills are checked only at pilot.
"""
import os
import sys
import time
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from desk import registry, trials  # noqa: E402
from desk.algos.lay_book import LayBook  # noqa: E402
from desk.session import DeskSession  # noqa: E402
from research.arms import core  # noqa: E402
from research.overfit import dsr_check  # noqa: E402

FIRST = date(2026, 11, 13)
LAST = date.today() - timedelta(days=1)
DRAWS = 20000                                   # research/arms/run.py's default: the live config's draws
pd.set_option("display.width", 220)

reg = {s.id: s for s in registry.load()[0]}["horse-lay-book"]
GATE = reg.gate
MIN_DAYS, MIN_BETS = int(GATE["min_days"]), int(GATE["min_bets"])
print(f"strategy {reg.id}: stage {reg.stage}, prefix {reg.order_ref_prefix}, params {reg.params}")
print(f"gate (pinned): {GATE} sha256 {reg.gate_sha256[:12]}; window {FIRST} .. {LAST}")

# 1. which forward days have data (blind: nothing but the count)
days, missing = [], []
day = FIRST
while day <= LAST:
    try:
        D = core.load_day(day, "win")
    except Exception as exc:                    # a day that cannot be read is reported, never guessed
        print(f"  {day}: not read ({str(exc)[:160]})", flush=True)
        D = None
    (days if D is not None else missing).append(day)
    day += timedelta(days=1)
print(f"\n== forward days with data: {len(days)} of {len(days) + len(missing)} asked; no data found: "
      f"{', '.join(str(d) for d in missing) or 'none'}")
if len(days) < MIN_DAYS:
    print(f"NOT YET READ: {len(days)} forward days with data, under {MIN_DAYS}; blind until then (no replay, no P&L)")
    sys.exit(0)

# 2. replay the first MIN_DAYS forward days with data, and later days only while the fills are under MIN_BETS (the
# window is fixed by the data, never by the day the query is run)
out, reasons, t_all = [], Counter(), time.time()
for day in days:
    if len(out) >= MIN_DAYS and sum(o["fills"] for o in out) >= MIN_BETS:
        break
    tic = time.time()
    D = core.load_day(day, "win")
    t0 = core.session_start(D, "win")
    core.set_plan_cfg(core.make_cfg(clv_draws=DRAWS), reset=True)
    _, desk, _ = core.run_session(D, DeskSession, core.make_cfg(clv_draws=DRAWS),
                                  {"algos": [LayBook(**reg.params)], "registry": [reg], "envelope": None, "_D": D}, t0)
    d = core.score_side(desk.positions_frame(reg.id, "LAY"), D, "desk", side="lay")
    intents = [r for r in desk.desk_rows if r["event"] == "intent"]
    for r in intents:
        if r.get("reason"):
            reasons[("refused: " if r["status"] == "REFUSED" else "cut: ") + str(r["reason"])] += 1
    fills = sum(1 for r in desk.desk_rows if r.get("event") in ("lay", "intent") and (r.get("matched") or 0) > 0)
    out.append({"day": str(day), "fills": fills, "refused": sum(1 for r in intents if r["status"] == "REFUSED"),
                "horses": d["horses"], "staked": d["staked"], "liability": d["liability"], "clv": d["clv"],
                "result_reduced": d["result_reduced"], "seconds": round(time.time() - tic)})
    print(f"  {day}: replayed", flush=True)   # blind: the day's figures are printed only with the reading
res = pd.DataFrame(out)
n_fills = int(res.fills.sum())
print(f"\n== {len(res)} forward days replayed, {n_fills} fills; {round((time.time() - t_all) / 60)} minutes")
if n_fills < MIN_BETS:
    print(f"NOT YET READ: {n_fills} fills, under {MIN_BETS}; blind until then")
    sys.exit(0)

# 3. the reading, once
print("\n== the reading: the lay book on paper, forward days, under its registered limits")
print(res.to_string(index=False))
print("the kernel's refusals and cuts, by reason (every day together):")
for k, n in reasons.most_common():
    print(f"  {n:6d}  {k}")
if not reasons:
    print("  none")
lines, _ = trials.parse(trials.LEDGER.read_text())
n_trials = trials.count_trials(lines, "horse_racing")
x = res["result_reduced"].astype(float).to_numpy()
rng = np.random.default_rng(0)
boot = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
lo, hi = np.percentile(boot, [5, 95])
chk = dsr_check(x, n_trials)
print(f"days {len(x)}; fills {n_fills}; staked GBP{res.staked.sum():,.2f} on liability GBP{res.liability.sum():,.2f}; "
      f"result GBP{x.sum():+,.2f} (mean GBP{x.mean():+.2f} a day, 90% {lo:+.2f} to {hi:+.2f}), up on "
      f"{int((x > 0).sum())} of {len(x)} days; deflated Sharpe {chk['dsr']} with {n_trials} trials "
      f"({chk['reason'] or 'computed'})")
if x.sum() > 0 and lo > 0 and chk["passed"]:
    print("verdict: PASSES on paper - a candidate for pilot, the owner's decision (lock-down and envelope first)")
else:
    print("verdict: does NOT pass on paper - the lay book returns to research")
