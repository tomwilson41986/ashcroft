"""The owner's day report for 4 Nov, sixth run, with the CLV and the risks (the owner: "check performance of the live
model today. What was CLV? Any risks?"; read-only). The day report as standard (research/queries/done/day_report.py),
then from its frames: the CLV against the BSP on the horses run, at the fill and after the non-runners' cuts, by UK
hour, against the last 7 days at the fill; the risks: the money still to run, the biggest races, the backs not yet
laid at the SP, the ledger's errors and holds, the trader's last line."""
import io
import os
import runpy
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

os.environ["REPORT_DAY"] = "2026-11-04"
sys.argv = ["day_report.py"]
g = runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "day_report.py"), run_name="__main__")
live, led, m, K, ledger, DAY = g["live"], g["led"], g["m"], g["K"], g["ledger"], g["DAY"]

print("\n== the CLV against the BSP (horses run, our non-runners out)")
r = live[live.bsp_now > 1].copy()
r["clv_fill"] = r.avg_price / r.bsp_now - 1
r["clv_cut"] = r.P_red / r.bsp_now - 1
w = r.matched.sum()
print(f"   {r.groupby(K).ngroups} horses, GBP{w:,.2f} staked: at the fill {(r.clv_fill * r.matched).sum() / w:+.2%}, "
      f"after the cuts {(r.clv_cut * r.matched).sum() / w:+.2%} (GBP{(r.clv_cut * r.matched).sum():+,.2f} locked)")
r["hour"] = r.t.dt.tz_convert("Europe/London").dt.hour
h = r.groupby("hour").apply(lambda x: pd.Series({
    "staked": x.matched.sum(), "clv_fill": (x.clv_fill * x.matched).sum() / x.matched.sum(),
    "clv_cut": (x.clv_cut * x.matched).sum() / x.matched.sum()}))
print(h.round(4).to_string())

rows = []
for i in range(1, 8):
    d = (date.fromisoformat(DAY) - timedelta(days=i)).isoformat()
    p = ledger(d)
    if p is None:
        continue
    b = p[(p.event == "back") & (p.matched > 0)].copy()
    st = p[p.event == "settle"].drop_duplicates(K, keep="last")[K + ["bsp"]].rename(columns={"bsp": "bsp_s"})
    b = b.merge(st, on=K, how="left")
    b = b[b.bsp_s > 1]
    if b.empty:
        continue
    rows.append({"day": d, "staked": round(b.matched.sum(), 2),
                 "clv_fill": round((b.matched * (b.avg_price / b.bsp_s - 1)).sum() / b.matched.sum(), 4)})
print("\n   the last 7 days at the fill (the ledger's settled BSPs)")
print(pd.DataFrame(rows).to_string(index=False))

print("\n== the risks")
left = m[~m.settled]
print(f"   still to run: {len(left)} races, GBP{left.staked.sum():,.2f} staked; mark at the mid GBP{left.mark_mid.sum():+,.2f}, "
      f"Cash Out GBP{left.cash_out.sum():+,.2f}")
print(left.sort_values("staked", ascending=False).head(5)[["off_uk", "venue", "horses", "staked", "mark_mid",
                                                            "cash_out", "forecast"]].round(2).to_string(index=False))
bk = led[(led.event == "back") & (led.matched > 0)].copy()
hed = pd.to_numeric(bk.get("hedged"), errors="coerce") if "hedged" in bk else pd.Series(np.nan, index=bk.index)
lays = led[led.event.astype(str).str.contains("hedge|lay", case=False, na=False)]
print(f"   backs matched {len(bk)}; SP lays in the ledger {len(lays)} ({lays.event.value_counts().to_dict()})")
errs = led[led.get("error").notna() & (led.get("error").astype(str) != "")] if "error" in led else led.iloc[:0]
print(f"   ledger rows with an error or skip reason: {len(errs)}; by event: {errs.event.value_counts().head(8).to_dict()}")
print("   the commonest reasons:")
print(errs.error.astype(str).str.slice(0, 90).value_counts().head(8).to_string())
print(f"   events today: {led.event.value_counts().to_dict()}")
print(f"   the trader's last line: {led.ts.max()}  (books read to {g['bk'].polled_utc.max()})")
