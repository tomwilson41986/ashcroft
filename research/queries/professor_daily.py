"""The Professor's daily read (standing query; dispatch research-query.yml on this branch): read-only.

1. The record: every tips file kept in reports/professor_dryrun_*/ (the published rule's tips.json and the
   short-priced shadow rule's tips_shadow_short.json) settled against its day's ledger in S3 (professor/settle.py),
   at 1 point level at the advised price and at the Betfair SP after 2%; by day and in total, published and shadow.
2. Today's positions: the live ledger's horses with their race facts (professor/candidates.py), written to
   out/professor_candidates_<day>.csv (the run's artifact), so the card can be priced at the bookmakers from a host
   Racing Post answers (it refuses GitHub's runners): python -m professor --card --candidates-file <that file>.
"""
import glob
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, ".")
from professor import settle as st  # noqa: E402
from professor.candidates import build, frame  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

print("== 1. the record")
frames = {"published": [], "shadow": []}
for f in sorted(glob.glob("reports/professor_dryrun_*/tips*.json")):
    settled, summ = st.settle_file(f)
    if settled.empty:
        continue
    kind = "shadow" if "shadow" in f else "published"
    frames[kind].append(settled)
    print(f"\n{f}\n{st.report(settled, summ)}")
for kind, fr in frames.items():
    if not fr:
        continue
    allt = pd.concat(fr, ignore_index=True)
    allt = allt[~allt["withheld"].astype(bool)]
    run = allt[allt["staked"] > 0]
    by = run.groupby("day").agg(tips=("runner", "size"), winners=("result", lambda r: int((r == "WINNER").sum())),
                                pts_advised=("pts_advised", "sum"), pts_bsp=("pts_bsp", "sum"), mean_clv=("clv", "mean"))
    print(f"\n== the record, {kind}: {len(run)} settled tips, {int(run['result'].eq('WINNER').sum())} winners, "
          f"{run['pts_advised'].sum():+.2f} pts at the advised price, {run['pts_bsp'].sum():+.2f} at the BSP, "
          f"mean CLV {run['clv'].mean():+.1%}; pending {int(allt['result'].eq('PENDING').sum())}")
    print(by.round(3).to_string())

day = datetime.now(ZoneInfo("Europe/London")).date()
print(f"\n== 2. today's positions, {day}")
cands = build(day)
os.makedirs("out", exist_ok=True)
if cands:
    df = frame(cands)
    path = f"out/professor_candidates_{day}.csv"
    df.to_csv(path, index=False)
    print(f"{len(cands)} positions in {df.groupby(['venue', 'off_uk']).ngroups} races written to {path}; first backed "
          f"by hour {df['first_uk'].str.slice(0, 2).value_counts().sort_index().to_dict()}")
else:
    print("no positions in the ledger yet")
