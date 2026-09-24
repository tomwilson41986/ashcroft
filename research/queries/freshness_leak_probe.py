"""Leak probe on real data: does anything of a day's own results reach that day's freshness features?

Iteration 26 found the freshness block (model/freshness_features.py) sharpening the BSP
forecast (-0.0049) AND the win probability against the market (Brier skill +0.0010,
resolved) -- the first block the 06:00 card can carry to do the second. A gain against
the market is where a leak would show, so this asks of the real database what the
synthetic day-flip test asks of a toy one:

1. Build the block on 2023-01 .. 2026-03 (development data; the holdout is not read).
2. For 12 days drawn from 2025-26Q1, ONE AT A TIME, (a) shuffle every race's finishing
   order and BSPs, and (b) blank them as the 06:00 card has them; rebuild; compare that
   day's features bit for bit. Any difference is a leak.
3. Control: the next racing day must move somewhere, or the probe could not see a leak.

And one question of the input itself. Today's gap is HRB's days_since_lr. If that field
were anything but the days SINCE the previous run -- days to the NEXT run, say -- it would
carry the future. So: how often does it equal the gap back to the previous run held here,
how often the gap forward to the next one?
"""
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.freshness_features import add_freshness_features  # noqa: E402

t0 = time.time()
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, surface_type, horse_name, trainer, career_runs,
           days_since_lr, placing_numerical, bfsp
    FROM race_results WHERE race_date >= '2023-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
print(f"{len(d):,} rows ({time.time() - t0:.0f}s)")

# --- what HRB's days_since_lr is -------------------------------------------------------------
s = d.assign(dt=pd.to_datetime(d.race_date), dsl=pd.to_numeric(d.days_since_lr, errors="coerce"))
s = s.sort_values(["horse_name", "dt"]).drop_duplicates(["horse_name", "dt"])
g = s.groupby("horse_name")["dt"]
s["back"] = (s.dt - g.shift(1)).dt.days
s["fwd"] = (g.shift(-1) - s.dt).dt.days
both = s.dsl.notna() & s.back.notna()
print(f"\ndays_since_lr present on {s.dsl.notna().mean():.1%} of runs; "
      f"values <= 0: {(s.dsl <= 0).sum():,}")
print(f"  where a previous run is held ({both.sum():,} runs):")
print(f"    equal to the gap BACK to it      {(s.dsl[both] == s.back[both]).mean():.1%}")
print(f"    shorter (a run not held here)    {(s.dsl[both] < s.back[both]).mean():.1%}")
print(f"    longer (should not happen)       {(s.dsl[both] > s.back[both]).mean():.1%}")
fw = s.dsl.notna() & s.fwd.notna()
print(f"  equal to the gap FORWARD to the next run: {(s.dsl[fw] == s.fwd[fw]).mean():.1%} "
      f"(chance level; a future-looking field would be near 100%)")
first = s.back.isna()
print(f"  first run held: career_runs==0 on {(pd.to_numeric(s.career_runs[first], errors='coerce') == 0).mean():.1%}; "
      f"days_since_lr missing on {s.dsl[first & (pd.to_numeric(s.career_runs, errors='coerce') == 0)].isna().mean():.1%} "
      f"of those debuts")

# --- the day-flip probe ---------------------------------------------------------------------
base, cols = add_freshness_features(d.copy())
rng = np.random.default_rng(26)
days = sorted(rng.choice(sorted(d.race_date[d.race_date >= "2025-01-01"].unique()), 12, replace=False))
key = ["raceid", "horse_name"]
print(f"\nbase built ({time.time() - t0:.0f}s); probing {len(days)} days one at a time")


def same(a, b):
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return (x == y) | (np.isnan(x) & np.isnan(y))


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
        out, _ = add_freshness_features(alt)
        a = base[base.race_date == day].set_index(key)[cols].sort_index()
        b = out[out.race_date == day].set_index(key)[cols].sort_index()
        rows += len(a)
        ok = same(a, b)
        for j, c in enumerate(cols):
            if not ok[:, j].all():
                bad[c] = bad.get(c, 0) + int((~ok[:, j]).sum())
        # the control: the next fortnight reads the day's wins and places
        end = str((pd.Timestamp(str(day)) + pd.Timedelta(days=14)).date())
        a2 = base[(base.race_date > day) & (base.race_date <= end)].set_index(key)[cols].sort_index()
        b2 = out[(out.race_date > day) & (out.race_date <= end)].set_index(key)[cols].sort_index()
        moved_next += int((~same(a2, b2)).sum())
    print(f"\n{label}: {rows:,} runner-days on {len(days)} days, {len(cols)} features")
    print(f"  the changed day: {'NO feature moved -- no leak' if not bad else 'LEAK in ' + str(bad)}")
    print(f"  control, the next fortnight: {moved_next:,} feature values moved "
          f"({'the probe can see a leak' if moved_next else 'WARNING: the probe is blind'})")
print(f"\ndone in {time.time() - t0:.0f}s")
