"""Is every metric calculated on what it claims? A real-data audit of the inputs and of every feature.

The owner's ask (25 Sep): test every metric calculation. Unit tests check the
arithmetic on hand-built races; this checks the real data the arithmetic runs
on, and every built feature's health.

Part A, the raw fields the metrics rest on (race_results, 2021 on, development
window only):
  1. number_of_runners against the rows each race has and its highest placing:
     NFP divides by (N - 1), so a declared field that counts non-runners makes
     every horse's NFP too high;
  2. comptime_numeric: one value per race (the winning time) or one per runner;
  3. total_dst_bt: how often it parses to lengths, winners and non-winners;
  4. placing_numerical: finishers, non-finishers, placings beyond the field;
  5. stall_positioning and going_description: values and coverage.
Part B, every feature of the cached matrix (when the research-query job
restored it): coverage, constant columns, exact duplicates of another column,
and the rank correlation with log BSP, the target; the served features first.
"""
import hashlib
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
t0 = time.time()
UNTIL = "2026-04-01"

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query(f"""
    SELECT race_date, race_time, track, horse_name, number_of_runners, placing_numerical, place,
           total_dst_bt, comptime_numeric, dist_furlongs, stall, stall_positioning, going_description, race_type, bfsp
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '{UNTIL}'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
print(f"{len(d):,} rows, {d.raceid.nunique():,} races, {d.race_date.min()}..{d.race_date.max()}")

# ---- A1: the field size NFP divides by -------------------------------------
g = d.groupby("raceid")
pos = pd.to_numeric(d.placing_numerical, errors="coerce")
r = pd.DataFrame({
    "declared": pd.to_numeric(g.number_of_runners.first(), errors="coerce"),
    "rows": g.size(),
    "max_place": pos.groupby(d.raceid).max(),
    "finishers": pos.notna().groupby(d.raceid).sum(),
})
r["gap"] = r.declared - r.rows
print("\nA1. number_of_runners less the rows the race has (share of races):")
print(r.gap.value_counts(normalize=True).sort_index().head(12).round(4).to_string())
print(f"   races where declared > rows: {(r.gap > 0).mean():.2%}; < rows: {(r.gap < 0).mean():.2%}")
print(f"   races where the highest placing exceeds number_of_runners: {(r.max_place > r.declared).mean():.2%}")
print(f"   races where the highest placing exceeds the rows: {(r.max_place > r.rows).mean():.2%}")
bad = r[r.gap != 0]
if len(bad):
    print("   where they differ, the last-placed horse's NFP under each field size:")
    last = (bad.declared - bad.max_place) / (bad.declared - 1)
    print(f"     with number_of_runners: median {last.median():.3f} (should be 0 for the last finisher)")

# ---- A2: the race time --------------------------------------------------------
ct = pd.to_numeric(d.comptime_numeric, errors="coerce")
nuniq = ct.groupby(d.raceid).nunique()
have = ct.notna().groupby(d.raceid).any()
print(f"\nA2. comptime_numeric: present in {have.mean():.1%} of races; one distinct value in "
      f"{(nuniq[have] == 1).mean():.1%} of those, more than one in {(nuniq[have] > 1).mean():.1%}")
print("   seconds per furlong, quantiles:",
      (ct / pd.to_numeric(d.dist_furlongs, errors="coerce")).quantile([0.01, 0.1, 0.5, 0.9, 0.99]).round(2).to_dict())

# ---- A3: lengths beaten --------------------------------------------------------
from model.perf_figures import parse_beaten_lengths  # noqa: E402
lb = d.total_dst_bt.map(parse_beaten_lengths)
won = pos == 1
print(f"\nA3. total_dst_bt parses for {lb[~won & pos.notna()].notna().mean():.1%} of placed non-winners, "
      f"{lb[won].notna().mean():.1%} of winners (a winner's is read as 0)")
print("   lengths beaten of non-winners, quantiles:", lb[~won & pos.notna()].quantile([0.5, 0.9, 0.99]).round(1).to_dict())
print("   unparsed values, most common:", d.total_dst_bt[lb.isna() & pos.notna() & ~won].value_counts().head(8).to_dict())

# ---- A4: placings ---------------------------------------------------------------
print(f"\nA4. placing_numerical: {pos.notna().mean():.1%} of rows have one; the rest (non-finishers) "
      f"read place as: {d.place[pos.isna()].value_counts().head(10).to_dict()}")

# ---- A5: stall placement and going ------------------------------------------------
sp = d.stall_positioning.fillna("").astype(str).str.strip().str.lower()
flat = ~d.race_type.fillna("").str.lower().str.contains("hurdle|chase|bumper|nh flat")
print(f"\nA5. stall_positioning on flat runs: {(sp[flat] != '').mean():.1%} filled; values "
      f"{sp[flat].value_counts().head(10).to_dict()}")
print("   by year:", sp[flat].ne("").groupby(d.race_date.str[:4][flat]).mean().round(3).to_dict())
print("   going, most common:", d.going_description.value_counts().head(12).to_dict())

# ---- B: every feature of the cached matrix ---------------------------------------------
from model import feature_cache  # noqa: E402
from model.bfsp_features import ALL_FEATURE_COLS, FORM_WINDOW_FEATURES, SHAPE_FORM_FEATURES  # noqa: E402

key = feature_cache.cache_key("horse_racing.db", "2021-01-01")
hit = feature_cache.load(".feature_cache", key)
if hit is None:
    print("\nB. no cached matrix for this code and data; part B skipped")
else:
    m = hit[0]
    m = m[pd.to_datetime(m.race_date) < UNTIL]
    feats = [c for c in list(ALL_FEATURE_COLS) + list(FORM_WINDOW_FEATURES) + list(SHAPE_FORM_FEATURES) if c in m.columns]
    y = np.log(pd.to_numeric(m.bfsp, errors="coerce"))
    rows = []
    hashes = {}
    for c in feats:
        v = pd.to_numeric(m[c], errors="coerce")
        h = hashlib.sha256(pd.util.hash_pandas_object(v.fillna(-9.99e99), index=False).to_numpy().tobytes()).hexdigest()
        hashes.setdefault(h, []).append(c)
        ok = v.notna() & y.notna()
        rho = v[ok].rank().corr(y[ok].rank()) if ok.sum() > 1000 and v[ok].nunique() > 1 else np.nan
        rows.append({"feature": c, "served": c in ALL_FEATURE_COLS, "coverage": v.notna().mean(),
                     "unique": v.nunique(), "rho_logbsp": rho})
    t = pd.DataFrame(rows)
    dups = [g for g in hashes.values() if len(g) > 1]
    print(f"\nB. {len(feats)} features on {len(m):,} rows")
    print(f"   constant (one value or none): {t[t.unique <= 1].feature.tolist()}")
    print(f"   under 5% coverage: {t[t.coverage < 0.05].feature.tolist()}")
    print(f"   exact duplicate groups ({len(dups)}):")
    for g in dups:
        print("     ", g)
    print("   weakest rank correlation with log BSP (|rho| < 0.01), served:")
    print("     ", t[t.served & (t.rho_logbsp.abs() < 0.01)].feature.tolist())
    print("\n   every feature (coverage, distinct values, rank correlation with log BSP):")
    print(t.sort_values("rho_logbsp", key=lambda s: -s.abs()).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\n{time.time() - t0:.0f}s")
