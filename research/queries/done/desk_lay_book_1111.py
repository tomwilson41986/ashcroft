"""The desk's first strategy on the recorded books: the lay book through the desk session and the risk kernel
(read-only; registered before the run: ledger `desk-lay-book-1111`; docs/agent_desk.md phase 1).

Two things, on every recorded win day from 12 Oct 2026 to the last with Betfair's price file:

1. The machinery check. The lay arm (research/arms/sessions.py LaySession, as research/arms/run.py runs it: the
   live config, 20,000 closing draws) and the desk's port of it (desk/algos/lay_book.py through desk/session.py,
   the kernel in paper mode with the registry's limits) on the same day: the same fills and the same score, or the
   desk is not doing what the arm did. Reported day by day; any difference is a failure of the port, not a result.
2. The registered reading of the desk strategy (written before any figure of this run is seen):
   the daily `result_reduced` of research/arms/core.py score_side (closed by a back at the BSP, after the
   non-runners' reduction factors and 2% on each market's net), over the days replayed. It passes only with a
   positive sum, the 90% day-bootstrap interval of the mean daily result above zero, and a deflated Sharpe of the
   daily results >= 0.95 with every horse-racing trial in the research ledger counted (desk/trials.py
   count_trials). Fewer than 30 days (research/overfit.py MIN_OBS): "not yet read", whatever the figures.
   The days to 1 Nov are the 21 on which the arm was found (TRADING.md:1787-1790), so they are not independent of
   the idea: the days from 3 Nov are reported apart, descriptive only.
   A pass moves the strategy to paper (forward days, its registry gate), never to money; nothing here touches the
   live trader, its config or Betfair.
"""
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from desk import registry, trials  # noqa: E402
from desk.algos.lay_book import LayBook  # noqa: E402
from desk.session import DeskSession  # noqa: E402
from research.arms import core, sessions  # noqa: E402
from research.overfit import MIN_OBS, dsr_check  # noqa: E402

FIRST = date(2026, 10, 12)
LAST = date.today() - timedelta(days=1)
DRAWS = 20000                                   # research/arms/run.py's default: the live config's draws
FOUND_TO = date(2026, 11, 1)                    # the arm's 21 days (TRADING.md:1787-1790)
pd.set_option("display.width", 220)

reg = {s.id: s for s in registry.load()[0]}["horse-lay-book"]
print(f"strategy {reg.id}: stage {reg.stage}, prefix {reg.order_ref_prefix}, params {reg.params}")
print(f"registry gate (forward, paper): {reg.gate} sha256 {reg.gate_sha256[:12]}")


def fills(rows):
    return [(r["market_id"], r["selection_id"], round(float(r["matched"]), 2), round(float(r["avg_price"]), 4))
            for r in rows if r.get("event") in ("lay", "intent") and (r.get("matched") or 0) > 0]


out, t_all = [], time.time()
day = FIRST
while day <= LAST:
    tic = time.time()
    try:
        D = core.load_day(day, "win")
    except Exception as exc:                    # a day that cannot be read is reported, never guessed
        print(f"  {day}: not read ({str(exc)[:160]})", flush=True)
        D = None
    if D is None:
        out.append({"day": str(day), "note": "no data found"})
        day += timedelta(days=1)
        continue
    t0 = core.session_start(D, "win")
    core.set_plan_cfg(core.make_cfg(clv_draws=DRAWS), reset=True)
    _, arm, _ = core.run_session(D, sessions.LaySession, core.make_cfg(clv_draws=DRAWS),
                                 {"lay_bar": 0.03, "max_lay_price": 20.0, "_D": D}, t0)
    core.set_plan_cfg(core.make_cfg(clv_draws=DRAWS), reset=True)
    _, desk, _ = core.run_session(D, DeskSession, core.make_cfg(clv_draws=DRAWS),
                                  {"algos": [LayBook(**reg.params)], "registry": [reg], "envelope": None, "_D": D}, t0)
    a = core.score_side(arm.lay_positions_frame(), D, "arm", side="lay")
    d = core.score_side(desk.positions_frame(reg.id, "LAY"), D, "desk", side="lay")
    fa, fd = fills(arm.lay_rows), fills(desk.desk_rows)
    refused = sum(1 for r in desk.desk_rows if r["event"] == "intent" and r["status"] == "REFUSED")
    row = {"day": str(day), "note": None, "same_fills": fa == fd, "arm_fills": len(fa), "desk_fills": len(fd),
           "refused": refused, "horses": d["horses"], "staked": d["staked"], "liability": d["liability"],
           "clv": d["clv"], "result": d["result"], "result_reduced": d["result_reduced"],
           "arm_result_reduced": a["result_reduced"], "seconds": round(time.time() - tic)}
    out.append(row)
    print({k: v for k, v in row.items() if k != "note"}, flush=True)
    day += timedelta(days=1)

res = pd.DataFrame(out)
done = res[res["note"].isna()].copy() if "note" in res else res
print(f"\n== {len(res)} days asked ({FIRST} .. {LAST}), {len(done)} replayed, "
      f"{len(res) - len(done)} with no data found; {round((time.time() - t_all) / 60)} minutes")
if done.empty:
    print("no data found: nothing to read")
    sys.exit(0)

print("\n== 1. the machinery check: the desk's port against the lay arm, day by day")
bad = done[~done["same_fills"].astype(bool) | (done["result_reduced"] != done["arm_result_reduced"])]
print(done[["day", "arm_fills", "desk_fills", "same_fills", "refused", "arm_result_reduced", "result_reduced"]]
      .to_string(index=False))
print(f"days where the desk differs from the arm: {len(bad)}"
      + (" - the port is not the arm; the reading below is not valid" if len(bad) else " - the port is the arm"))

print("\n== 2. the registered reading: the lay book's daily result at the BSP close, after the cuts and 2%")
lines, _ = trials.parse(trials.LEDGER.read_text())
n_trials = trials.count_trials(lines, "horse_racing")
x = done["result_reduced"].astype(float).to_numpy()
rng = np.random.default_rng(0)
boot = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
lo, hi = np.percentile(boot, [5, 95])
chk = dsr_check(x, n_trials)
print(f"days {len(x)}; staked GBP{done.staked.sum():,.2f} on liability GBP{done.liability.sum():,.2f}; "
      f"result GBP{x.sum():+,.2f} (mean GBP{x.mean():+.2f} a day, 90% {lo:+.2f} to {hi:+.2f}), up on "
      f"{int((x > 0).sum())} of {len(x)} days; deflated Sharpe {chk['dsr']} with {n_trials} trials "
      f"({chk['reason'] or 'computed'})")
if len(bad):
    verdict = "NOT READ: the port differs from the arm"
elif len(x) < MIN_OBS:
    verdict = f"NOT YET READ: {len(x)} days, under {MIN_OBS}"
elif x.sum() > 0 and lo > 0 and chk["passed"]:
    verdict = "PASSES: a candidate for paper (forward days under the registry gate), not money"
else:
    verdict = "does NOT pass"
print(f"verdict: {verdict}")

later = done[pd.to_datetime(done["day"]).dt.date > FOUND_TO]
if len(later):
    y = later["result_reduced"].astype(float)
    print(f"\ndescriptive only, the days after the arm was found ({later.day.min()} .. {later.day.max()}): "
          f"{len(y)} days, GBP{y.sum():+,.2f}, up on {int((y > 0).sum())}")
