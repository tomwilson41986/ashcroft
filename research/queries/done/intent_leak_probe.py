"""Leak probe on real data: does anything of a day's own results reach that day's intent features?

Iteration 21 found the intent block (model/intent_features.py) sharpening the forecast of
BSP far more than any block before it (-0.0080 in mean absolute log error, paired). The
synthetic day-flip test says no result or price of a day can reach that day's features.
This asks the same of the real database, where duplicates, odd values and gaps live:

1. Build the block on 2023-01 .. 2026-03 (development data; the holdout is not read).
2. For 12 days drawn from 2025-26Q1, ONE AT A TIME, (a) shuffle every race's finishing
   order and BSPs, and separately (b) blank them, as the 06:00 card has them; rebuild;
   compare that day's features bit for bit. Any difference is a leak.
3. As a control, the day AFTER each probed day must change somewhere -- otherwise the
   probe could not have seen a leak even if there were one.
"""
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.intent_features import add_intent_features  # noqa: E402

t0 = time.time()
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, race_name, surface_type, race_class, horse_name, horse_sex,
           headgear, trainer, jockey_name, jockeys_claim, career_runs, placing_numerical, bfsp
    FROM race_results WHERE race_date >= '2023-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
print(f"{len(d):,} rows ({time.time() - t0:.0f}s)")

base, cols = add_intent_features(d.copy())
rng = np.random.default_rng(24)
days = sorted(rng.choice(sorted(d.race_date[d.race_date >= "2025-01-01"].unique()), 12, replace=False))
key = ["raceid", "horse_name"]
print(f"base built ({time.time() - t0:.0f}s); probing {len(days)} days one at a time")


def same(a, b):
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return (x == y) | (np.isnan(x) & np.isnan(y))


# ONE day changed at a time: changing several at once lets an earlier probed day's results
# reach a later probed day through history, which is legitimate and would read as a leak.
for label, change in (("(a) the day's finishing orders and BSPs shuffled", "shuffle"),
                      ("(b) the day's results and BSPs blanked (the 06:00 card)", "blank")):
    bad, rows, moved_next = {}, 0, 0
    for day in days:
        alt = d.copy()
        on = alt.race_date == day
        if change == "shuffle":
            for _, idx in alt[on].groupby("raceid").groups.items():
                idx = list(idx)
                alt.loc[idx, "placing_numerical"] = rng.permutation(alt.loc[idx, "placing_numerical"].to_numpy())
                alt.loc[idx, "bfsp"] = rng.permutation(alt.loc[idx, "bfsp"].to_numpy())
        else:
            alt.loc[on, ["placing_numerical", "bfsp"]] = np.nan
        out, _ = add_intent_features(alt)
        a = base[base.race_date == day].set_index(key)[cols].sort_index()
        b = out[out.race_date == day].set_index(key)[cols].sort_index()
        rows += len(a)
        ok = same(a, b)
        for j, c in enumerate(cols):
            if not ok[:, j].all():
                bad[c] = bad.get(c, 0) + int((~ok[:, j]).sum())
        later = base.race_date[base.race_date > day].min()
        a2 = base[base.race_date == later].set_index(key)[cols].sort_index()
        b2 = out[out.race_date == later].set_index(key)[cols].sort_index()
        moved_next += int((~same(a2, b2)).sum())
    print(f"\n{label}: {rows:,} runner-days on {len(days)} days, {len(cols)} features")
    print(f"  the changed day: {'NO feature moved -- no leak' if not bad else 'LEAK in ' + str(bad)}")
    print(f"  control, the next racing day: {moved_next:,} feature values moved "
          f"({'the probe can see a leak' if moved_next else 'WARNING: the probe is blind'})")
print(f"\ndone in {time.time() - t0:.0f}s")
