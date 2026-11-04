"""Is our GBGB copy up to date (the owner, 4 Nov)? The raw day files of the last 14 days in S3: meetings, races and runs
in each, whether the day reached GBGB's 500-race list cap (meetings may be missing), and the built table's last day
(read-only)."""
import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.getcwd())
from sources.common import Store, gunzip               # noqa: E402

store = Store()
today = date(2026, 11, 4)
print("day        | meetings | races | runs | capped | weights")
for back in range(14, -1, -1):
    d = today - timedelta(days=back)
    raw = store.get(f"gbgb/raw/{d.year}/{d:%Y-%m-%d}.json.gz")
    if not raw:
        print(f"{d} | no data found")
        continue
    day = json.loads(gunzip(raw))
    ms = day.get("meetings") or []
    races = runs = weighed = 0
    for m in ms:
        for mm in (m if isinstance(m, list) else [m]):
            for r in mm.get("races", []):
                races += 1
                for t in r.get("traps", []):
                    runs += 1
                    weighed += bool(t.get("resultDogWeight"))
    print(f"{d} | {len(ms)} | {races} | {runs} | {day.get('capped')} | {weighed}")
t = store.get_parquet(f"gbgb/runs_{today.year}.parquet")
if t is not None and len(t):
    col = "race_date" if "race_date" in t else t.columns[0]
    print("\nbuilt table gbgb/runs_2026.parquet:", len(t), "rows, last day", str(t[col].max()))
else:
    print("\nbuilt table: no data found")
