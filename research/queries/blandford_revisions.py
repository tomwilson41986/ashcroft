"""Are Timeform-feed ratings revised after the fact?

Look-ahead enters a backfilled rating database when raters revise a run's
rating once they see how the race "worked out" -- later runs by the same
horses, including, in a backtest, the very race being predicted. The test:

1. When was each period loaded? loaded_at against meeting_date, by month.
2. For dates whose rows were first loaded within a few days of the racing,
   fetch the same dates from the Blandford API NOW and compare runner by
   runner: performance rating, timefigure, pre-race master rating. Any
   systematic change is a revision, and a revision is information the
   historical backtest saw and a live model would not.
"""
import sqlite3
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import blandford_sync as bs  # noqa: E402

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
b = pd.read_sql_query("""SELECT meeting_date, course_bf, race_number, horse_code, performance_rating AS p,
                                timefigure AS tfig, pre_race_master_rating AS m, loaded_at
                         FROM blandford_results""", conn)
b["lag_days"] = (pd.to_datetime(b.loaded_at) - pd.to_datetime(b.meeting_date)).dt.days
b["month"] = b.meeting_date.str[:7]
print("1. When each month's rows were loaded")
lo = b.groupby("month").agg(first_load=("loaded_at", "min"), last_load=("loaded_at", "max"),
                            median_lag_days=("lag_days", "median"))
print(lo.to_string())

early = b[b.lag_days.between(0, 4)]
days = sorted(early.meeting_date.unique())
print(f"\n2. {len(days)} meeting dates have rows loaded within 4 days of racing")
if not days:
    sys.exit(0)
pick = [days[i] for i in np.linspace(0, len(days) - 1, min(8, len(days))).astype(int)]
sess = bs._session()
rows = []
for d in pick:
    dd = date.fromisoformat(d)
    try:
        r = sess.get(f"{bs.BASE_URL}/api/APIData_Table2?startDate={d}&endDate={d}", timeout=300)
        data = r.json().get("data", []) if r.status_code == 200 else []
    except Exception as exc:                        # noqa: BLE001
        print(f"   {d}: fetch failed ({exc})")
        continue
    now = bs.parse_rows(data)
    if now.empty:
        print(f"   {d}: API returned nothing (HTTP {r.status_code})")
        continue
    now = now.rename(columns={"performance_rating": "p_now", "timefigure": "tfig_now", "pre_race_master_rating": "m_now"})
    then = early[early.meeting_date == d]
    j = then.merge(now[["meeting_date", "course_bf", "race_number", "horse_code", "p_now", "tfig_now", "m_now"]],
                   on=["meeting_date", "course_bf", "race_number", "horse_code"], how="inner")
    for col, c_now in (("p", "p_now"), ("tfig", "tfig_now"), ("m", "m_now")):
        a, bb = pd.to_numeric(j[col], errors="coerce"), pd.to_numeric(j[c_now], errors="coerce")
        both = a.notna() & bb.notna()
        diff = (bb - a)[both]
        rows.append({"date": d, "field": col, "n": int(both.sum()), "changed": float((diff != 0).mean()) if both.any() else np.nan,
                     "mean_change": float(diff.mean()) if both.any() else np.nan,
                     "mean_abs_change": float(diff.abs().mean()) if both.any() else np.nan,
                     "filled_since": int((a.isna() & bb.notna()).sum()), "blanked_since": int((a.notna() & bb.isna()).sum())})
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
