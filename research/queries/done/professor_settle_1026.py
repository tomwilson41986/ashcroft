"""The Professor's dry-run tips settled from the trader's ledger (the owner asked, 26 Oct: how did today's tips do?).
Read-only: every tips file kept in reports/professor_dryrun_*/ (the published rule's tips.json and the short-priced
shadow rule's tips_shadow_short.json) against its day's ledger in S3 (the settle rows' result and Betfair SP), at
1 point level at the advised price and at the BSP after 2% (professor/settle.py)."""
import glob
import sys

import pandas as pd

sys.path.insert(0, ".")
from professor import settle as st  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
frames = {"published": [], "shadow": []}
for f in sorted(glob.glob("reports/professor_dryrun_*/tips*.json")):
    settled, summ = st.settle_file(f)
    if settled.empty:
        continue
    kind = "shadow" if "shadow" in f else "published"
    settled["file"] = f
    frames[kind].append(settled)
    print(f"\n== {f}\n{st.report(settled, summ)}")
for kind, fr in frames.items():
    if not fr:
        continue
    allt = pd.concat(fr, ignore_index=True)
    summ = st.summary(allt)
    print(f"\n== the record, {kind} ({allt['day'].nunique()} days)")
    print(st.report(allt, summ).splitlines()[-2] if kind == "published" else st.report(allt, summ).splitlines()[-1])
    by = allt[allt["staked"] > 0].groupby("day").agg(tips=("runner", "size"), winners=("result", lambda r: (r == "WINNER").sum()),
                                                     pts_advised=("pts_advised", "sum"), pts_bsp=("pts_bsp", "sum"),
                                                     mean_clv=("clv", "mean"))
    print(by.round(3).to_string())
