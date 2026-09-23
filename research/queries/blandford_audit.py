"""Is the Blandford/Timeform feed's "pre-race" rating really pre-race, and how far does it reach?

Three tests, read-only against horse_racing.db:

1. Coverage by month: linked rows, and the share with a pre-race master rating
   and a performance rating.
2. The update test. If m_t (pre_race_master_rating) is set BEFORE race t, the
   next race's rating m_{t+1} moves with how far the horse ran above or below
   it: corr(m_{t+1} - m_t, p_t - m_t) clearly positive. If m_t already knows
   race t's result, m_{t+1} - m_t no longer tracks that surprise.
3. The debut test. A rating set before a debut cannot know the debut run, so
   among debut runs corr(m_t, p_t) should be weak and coverage thin. A rating
   written after the race tracks the debut performance closely.
"""
import sqlite3

import numpy as np
import pandas as pd

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
b = pd.read_sql_query("""SELECT meeting_date, horse_code, race_results_id, pre_race_master_rating AS m,
                                pre_race_adjusted_rating AS a, performance_rating AS p, position, race_type
                         FROM blandford_results""", conn)
print(f"blandford_results rows: {len(b):,}; linked to race_results: {b.race_results_id.notna().mean():.1%}")
b["month"] = b.meeting_date.str[:7]
for c in ("m", "a", "p"):
    b[c] = pd.to_numeric(b[c], errors="coerce")
    b.loc[(b[c] <= 0) | (b[c] >= 999), c] = np.nan
cov = b.groupby("month").agg(rows=("horse_code", "size"), linked=("race_results_id", lambda s: s.notna().mean()),
                             has_m=("m", lambda s: s.notna().mean()), has_p=("p", lambda s: s.notna().mean()))
rr = pd.read_sql_query("SELECT substr(race_date,1,7) AS month, COUNT(*) AS rr_rows FROM race_results "
                       "WHERE race_date >= '2021-01-01' GROUP BY 1", conn).set_index("month")
cov = cov.join(rr, how="outer")
cov["share_of_results_linked"] = cov["rows"] * cov["linked"] / cov["rr_rows"]
pd.set_option("display.width", 200)
print("\n1. Coverage by month")
print(cov.to_string(float_format=lambda x: f"{x:.3f}"))

b = b.dropna(subset=["horse_code"]).sort_values(["horse_code", "meeting_date"])
g = b.groupby("horse_code", sort=False)
b["m_next"] = g["m"].shift(-1)
b["run_no"] = g.cumcount()
ok = b[["m", "p", "m_next"]].notna().all(axis=1)
x = b[ok]
print("\n2. The update test (all runs with m_t, p_t and m_{t+1}):", f"{len(x):,} runs")
print(f"   corr(m_t, p_t)                     {np.corrcoef(x.m, x.p)[0, 1]:+.3f}")
print(f"   corr(m_(t+1), p_t)                 {np.corrcoef(x.m_next, x.p)[0, 1]:+.3f}")
print(f"   corr(m_(t+1) - m_t, p_t - m_t)     {np.corrcoef(x.m_next - x.m, x.p - x.m)[0, 1]:+.3f}")
print(f"   share with m_(t+1) == m_t          {np.mean(x.m_next == x.m):.1%}")
for yr, xx in x.groupby(x.meeting_date.str[:4]):
    print(f"     {yr}: n {len(xx):7,}  corr(update, surprise) {np.corrcoef(xx.m_next - xx.m, xx.p - xx.m)[0, 1]:+.3f}"
          f"  corr(m_t, p_t) {np.corrcoef(xx.m, xx.p)[0, 1]:+.3f}")

d = b[b.run_no == 0]
print(f"\n3. The debut test: {len(d):,} first appearances in the feed")
print(f"   share with a pre-race master rating  {d.m.notna().mean():.1%}   (all runs: {b.m.notna().mean():.1%})")
dd = d.dropna(subset=["m", "p"])
if len(dd) > 50:
    print(f"   corr(m_t, p_t) on debut runs          {np.corrcoef(dd.m, dd.p)[0, 1]:+.3f}  (n {len(dd):,})")
    print(f"   mean |m_t - p_t| debut / later runs   {np.mean(np.abs(dd.m - dd.p)):.2f} / "
          f"{np.mean(np.abs(x.m - x.p)):.2f}")
    print("   first appearances are debuts only for horses whose first race is inside the feed's window;"
          " restricted to 2023+ meetings:")
    d3 = dd[dd.meeting_date >= "2023-01-01"]
    if len(d3) > 50:
        print(f"   2023+: n {len(d3):,}  corr(m_t, p_t) {np.corrcoef(d3.m, d3.p)[0, 1]:+.3f}")
