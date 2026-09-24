"""Real-data probe of the form-window and shape-form blocks, before any model sees them.

model/form_windows.py puts every per-run measure over career, last run, last 3, last 5
and the last 3/5/10 weighted by recency; model/shape_form.py reads each past run against
the pace and draw it met and measures the market's own misses by draw and position.
Synthetic tests pin the mechanics; this asks the real database (development data,
2024-07 .. 2026-03; the holdout is not read):

1. Leaks. For 6 days drawn from 2025-26Q1, ONE AT A TIME, (a) shuffle every race's
   finishing order, margins and BSPs, and (b) blank them with the comments, as the
   06:00 card has them; rebuild both blocks; compare that day's features bit for bit.
   The next fortnight is the control: it must move, or the probe is blind.
2. Coverage: how often each feature is known on the priced runs of 2025-07 .. 2026-03.
3. What the price already knows: over the same runs, the mean of won less the BSP's
   normalised chance (A - E) by quintile of the features that claim to see past the
   price (the pace-and-draw credit, the adjusted form, the market-miss cells, the
   windows of A - E). A flat profile says the market prices it; a slope is a lead for
   the model, not a result -- the walk-forward iterations decide.

fw_rsr_* are empty here: the speed ratings come from the engine's standard times,
which this probe does not build. Every other feature is built as the engine builds it.
"""
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.draw_curve import add_draw_curve  # noqa: E402
from model.form_windows import add_form_windows  # noqa: E402
from model.race_shape import add_race_shape_features  # noqa: E402
from model.shape_form import add_shape_form  # noqa: E402

t0 = time.time()
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, surface_type, horse_name, number_of_runners,
           placing_numerical, total_dst_bt, dist_furlongs, official_rating, bfsp, comment, stall
    FROM race_results WHERE race_date >= '2024-07-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
d["race_date"] = pd.to_datetime(d["race_date"])
print(f"{len(d):,} rows ({time.time() - t0:.0f}s)")


def build(frame):
    f = frame.copy()
    f, wcols = add_form_windows(f)
    f, _ = add_race_shape_features(f)
    f, _ = add_draw_curve(f)
    f, scols = add_shape_form(f)
    return f, wcols + scols


base, cols = build(d)
print(f"built {len(cols)} features ({time.time() - t0:.0f}s)")

# --- 1. the day-flip probe ------------------------------------------------------------------
rng = np.random.default_rng(29)
pool = sorted(d.race_date[d.race_date >= "2025-01-01"].unique())
days = sorted(rng.choice(pool, 6, replace=False))
key = ["raceid", "horse_name"]
RESULT = ["placing_numerical", "total_dst_bt", "bfsp"]


def same(a, b):
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    return (x == y) | (np.isnan(x) & np.isnan(y))


for label, change in (("(a) the day's finishing orders, margins and BSPs shuffled", "shuffle"),
                      ("(b) the day's results, margins, BSPs and comments blanked (the 06:00 card)", "blank")):
    bad, rows, moved_next = {}, 0, 0
    for day in days:
        alt = d.copy()
        on = alt.race_date == day
        if change == "shuffle":
            for _, idx in alt[on].groupby("raceid").groups.items():
                idx = list(idx)
                for c in RESULT:
                    alt.loc[idx, c] = rng.permutation(alt.loc[idx, c].to_numpy())
        else:
            alt.loc[on, RESULT + ["comment"]] = np.nan
        out, _ = build(alt)
        a = base[base.race_date == day].set_index(key)[cols].sort_index()
        b = out[out.race_date == day].set_index(key)[cols].sort_index()
        rows += len(a)
        ok = same(a, b)
        for j, c in enumerate(cols):
            if not ok[:, j].all():
                bad[c] = bad.get(c, 0) + int((~ok[:, j]).sum())
        end = pd.Timestamp(day) + pd.Timedelta(days=14)
        later = (base.race_date > day) & (base.race_date <= end)
        a2 = base[later].set_index(key)[cols].sort_index()
        b2 = out[(out.race_date > day) & (out.race_date <= end)].set_index(key)[cols].sort_index()
        moved_next += int((~same(a2, b2)).sum())
        print(f"  {pd.Timestamp(day).date()} {change}: done ({time.time() - t0:.0f}s)")
    print(f"\n{label}: {rows:,} runner-days on {len(days)} days, {len(cols)} features")
    print(f"  the changed day: {'NO feature moved -- no leak' if not bad else 'LEAK in ' + str(bad)}")
    print(f"  control, the next fortnight: {moved_next:,} feature values moved "
          f"({'the probe can see a leak' if moved_next else 'WARNING: the probe is blind'})")

# --- 2. coverage on the priced runs --------------------------------------------------------
dev = base[(base.race_date >= "2025-07-01") & (pd.to_numeric(base.bfsp, errors="coerce") > 1)].copy()
cover = dev[cols].notna().mean().sort_values()
print(f"\ncoverage on {len(dev):,} priced runs, 2025-07 .. 2026-03 (lowest first):")
print(cover.head(15).round(3).to_string())
print(f"  median over the {len(cols)} features: {cover.median():.3f}")

# --- 3. A - E by quintile ------------------------------------------------------------------
p = 1.0 / pd.to_numeric(dev.bfsp, errors="coerce")
dev["pn"] = p / p.groupby(dev.raceid).transform("sum")
pos = pd.to_numeric(dev.placing_numerical, errors="coerce")
dev["ae"] = (pos == 1).astype(float) - dev["pn"]
dev = dev[pos.notna()]
print("\nmean A - E (won less the BSP's normalised chance) by quintile, x100; SE in brackets")
probe = ["sf_bias_l1", "sf_bias_w5", "sf_adj_l1", "sf_adj_w5", "sf_draw_ae", "sf_pos_ae",
         "sf_speed_inside", "sf_speed_near", "fw_ae_w5", "fw_ae_car", "fw_nres_w5", "fw_perf_w5_vs_or",
         "fw_mkt_w5", "fw_lbsc_w3"]
for c in probe:
    x = dev[c]
    ok = x.notna()
    if ok.sum() < 1000 or x[ok].nunique() < 5:
        print(f"  {c:18s} too few values ({ok.sum():,})")
        continue
    q = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
    g = dev.loc[ok, "ae"].groupby(q.to_numpy())
    m, se = g.mean() * 100, g.std() / np.sqrt(g.count()) * 100
    cells = "  ".join(f"{m[i]:+.2f}({se[i]:.2f})" for i in range(5))
    print(f"  {c:18s} {cells}   n={ok.sum():,}")
print(f"\ndone in {time.time() - t0:.0f}s")
