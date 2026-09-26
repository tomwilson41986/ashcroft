"""Which count is sire_runners in the training matrix? The parity query found it larger than the live path's.

matrix_live_parity.py (run 36156133812) compared the cached matrix with the 06:00
path on 28 and 25 March 2026. Of the 615 served features, only sire_runners and
damsire_runners differed with a perfect card: the matrix's value was never
smaller, 7% larger at the median and up to 1,600 runs larger. Both paths build
it with the same lag-safe helper (_lag_race_runs, earlier days only), so one of
them is being handed different rows.

For every runner of those days in the matrix, the count straight from the
database of the sire's (and damsire's) runs from 2021-01-01:
  before  on days before the row's day, what the feature should be
  total   up to the end of the data, what a lookahead would give
and the matrix's own value beside both, and the live path's from the parity
artifact's parquet when it is present. Read-only.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from model import feature_cache  # noqa: E402

DAYS = ["2026-03-28", "2026-03-25"]
START = "2021-01-01"

key = feature_cache.cache_key("horse_racing.db", START)
path = Path(".feature_cache") / f"features_{key}.parquet"
if not path.exists():
    print(f"no cached matrix for key {key}: nothing to check")
    sys.exit(0)
names = set(pq.read_schema(path).names)
want = [c for c in ["race_date", "track", "race_time", "horse_name", "stallion", "dam_stallion", "sire_runners",
                    "damsire_runners"] if c in names]
print("columns present:", want)
mat, info = feature_cache.load(".feature_cache", key, columns=want)
print("matrix built_at", info.get("built_at"), "feature_code_hash", info.get("feature_code_hash"),
      "rows", info.get("rows"))
sub = mat.loc[mat.race_date.isin(pd.to_datetime(DAYS)), want].copy()
del mat

conn = sqlite3.connect("horse_racing.db")
db = pd.read_sql_query("SELECT race_date, stallion, dam_stallion FROM race_results WHERE race_date >= ?", conn,
                       params=[START])
conn.close()
db["race_date"] = pd.to_datetime(db["race_date"])
end = db["race_date"].max()
print(f"database rows from {START}: {len(db):,}; last day {end.date()}")
for col, feat in (("stallion", "sire_runners"), ("dam_stallion", "damsire_runners")):
    if col not in sub.columns:
        print(f"{col} not in the matrix: skipped")
        continue
    k = db[col].fillna("unknown").str.strip().str.lower()
    for day in DAYS:
        d = sub[sub.race_date == pd.Timestamp(day)].copy()
        d["_k"] = d[col].fillna("unknown").str.strip().str.lower()
        before = db[db.race_date < pd.Timestamp(day)].assign(_k=k).groupby("_k").size()
        total = db.assign(_k=k).groupby("_k").size()
        d["before"] = d["_k"].map(before).fillna(0)
        d["total"] = d["_k"].map(total).fillna(0)
        d["same_day_and_before"] = d["_k"].map(db[db.race_date <= pd.Timestamp(day)].assign(_k=k)
                                               .groupby("_k").size()).fillna(0)
        m = d[feat].to_numpy(float)
        for ref in ("before", "same_day_and_before", "total"):
            r = d[ref].to_numpy(float)
            print(f"{day} {feat}: matrix == {ref:20s} on {np.mean(m == r):.3f} of {len(d)} runners; "
                  f"mean matrix - {ref} {np.mean(m - r):+.1f}")
        print(d[["horse_name", col, feat, "before", "same_day_and_before", "total"]].head(6).to_string())
