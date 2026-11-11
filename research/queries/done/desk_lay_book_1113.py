"""The lay book through the desk, a third time: desk-lay-book-1112's machinery check failed on 15 of 29 days, mostly
by a penny (ledger `desk-lay-book-1112`, run 42988481613), so this is its rerun after the second fix (read-only;
registered before the run: ledger `desk-lay-book-1113`).

The cause: desk/risk.py _floor_pence floored size * 100 in binary floating point, so some exact stakes lost a penny
(1.13 * 100 = 112.999...: 1.13 became 1.12), and a penny less matched moved later top-ups. The fix rounds before the
floor and sends an order no limit cut as asked (tests/test_desk.py test_an_exact_stake_passes_whole). The first fix
(desk-lay-book-1111): each lay sized as it is sent, after the minute's earlier fills (desk/strategy.py revise).

The design and the reading are 1112's, unchanged:
1. The machinery check: the lay arm and the desk's port with kernel limits that cannot bind (PARITY_LIMITS). Any day
   whose fills or result_reduced differ makes the run not read.
2. The registered reading: the desk with the registry's limits (desk/strategies/horse_racing/horse-lay-book.json),
   the daily result_reduced of research/arms/core.py score_side (closed by a back at the BSP, after the reduction
   factors and 2% on each market's net). It passes only with a positive sum, the 90% day-bootstrap interval of the mean
   above zero and a deflated Sharpe >= 0.95 with every horse-racing trial in the ledger counted (desk/trials.py
   count_trials); under 30 days, not yet read. Every refusal and cut by the kernel is counted by reason.
3. Not blind: 1111's and 1112's figures were seen before this registration. A pass moves the strategy to paper under
   its registry gate (forward days), never to money; nothing here touches the live trader.
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
from research.arms import core, sessions  # noqa: E402
from research.overfit import MIN_OBS, dsr_check  # noqa: E402

FIRST = date(2026, 10, 12)
LAST = date.today() - timedelta(days=1)
DRAWS = 20000                                   # research/arms/run.py's default: the live config's draws
FOUND_TO = date(2026, 11, 1)                    # the arm's 21 days (TRADING.md:1787-1790)
pd.set_option("display.width", 220)

reg = {s.id: s for s in registry.load()[0]}["horse-lay-book"]
PARITY_LIMITS = {"max_order": 1e9, "max_day_turnover": 1e12, "max_day_loss": 1e12, "max_market_exposure": 1e12,
                 "max_orders_per_minute": 1e9, "price_collar": 1e9, "max_book_age_s": 1e9, "min_stake": 0.01}
parity = registry._strategy({**reg.raw, "limits": {**reg.raw["limits"], **PARITY_LIMITS}})
reasons = Counter()
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
    _, port, _ = core.run_session(D, DeskSession, core.make_cfg(clv_draws=DRAWS),
                                  {"algos": [LayBook(**parity.params)], "registry": [parity], "envelope": None,
                                   "_D": D}, t0)
    core.set_plan_cfg(core.make_cfg(clv_draws=DRAWS), reset=True)
    _, desk, _ = core.run_session(D, DeskSession, core.make_cfg(clv_draws=DRAWS),
                                  {"algos": [LayBook(**reg.params)], "registry": [reg], "envelope": None, "_D": D}, t0)
    a = core.score_side(arm.lay_positions_frame(), D, "arm", side="lay")
    pp = core.score_side(port.positions_frame(reg.id, "LAY"), D, "port", side="lay")
    d = core.score_side(desk.positions_frame(reg.id, "LAY"), D, "desk", side="lay")
    fa, fp, fd = fills(arm.lay_rows), fills(port.desk_rows), fills(desk.desk_rows)
    intents = [r for r in desk.desk_rows if r["event"] == "intent"]
    refused = sum(1 for r in intents if r["status"] == "REFUSED")
    for r in intents:
        if r.get("reason"):
            reasons[("refused: " if r["status"] == "REFUSED" else "cut: ") + str(r["reason"])] += 1
    row = {"day": str(day), "note": None, "same_fills": fa == fp, "arm_fills": len(fa), "port_fills": len(fp),
           "port_result_reduced": pp["result_reduced"], "desk_fills": len(fd), "refused": refused,
           "horses": d["horses"], "staked": d["staked"], "liability": d["liability"], "clv": d["clv"],
           "result": d["result"], "result_reduced": d["result_reduced"],
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

print("\n== 1. the machinery check: the desk's port (limits that cannot bind) against the lay arm, day by day")
bad = done[~done["same_fills"].astype(bool) | (done["port_result_reduced"] != done["arm_result_reduced"])]
print(done[["day", "arm_fills", "port_fills", "same_fills", "arm_result_reduced", "port_result_reduced"]]
      .to_string(index=False))
print(f"days where the desk differs from the arm: {len(bad)}"
      + (" - the port is not the arm; the reading below is not valid" if len(bad) else " - the port is the arm"))

print("\n== 2. the registered reading: the lay book under its registered limits, a day")
print(done[["day", "desk_fills", "refused", "horses", "staked", "liability", "clv", "result_reduced"]]
      .to_string(index=False))
print("the kernel's refusals and cuts under the registered limits, by reason (every day together):")
for k, n in reasons.most_common():
    print(f"  {n:6d}  {k}")
if not reasons:
    print("  none")
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
