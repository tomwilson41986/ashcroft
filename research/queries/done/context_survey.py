"""Survey for a 'context' block: season, day, race conditions and the horse's history in them.

Development data only (race_date before 2026-04-01; the holdout is not read). The
feature inventory lists these proposals as never built: J1 month/season, J2 day of
week, J3 horse seasonality, G2 handicap vs non-handicap form, G4 age restriction, K1
major-race experience, L2 class x speed, L4 weight per rating point.

1. The fields: race_restrictions_age, major, race_class, race_type/race_name (handicap,
   black type), weekday and month -- values and coverage.
2. Does the price take each context as a whole? Wins against the race-normalised BSP
   probability (A/E) and the mean of won - p, 2023 on, by month, weekday, age
   restriction, black type, handicap, the horse's age against the restriction, and a
   first run of the season.
3. Does a trainer's record against the price in a calendar quarter persist (2021-23 ->
   2024-26Q1)? And a horse's own form by quarter (J3), and in handicaps against other
   races (G2): does the difference seen before predict the difference now?
"""
import sqlite3
import time

import numpy as np
import pandas as pd

t0 = time.time()
pd.set_option("display.width", 220)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, race_name, race_class, race_restrictions_age, major,
           number_of_runners, horse_name, horse_age, trainer, official_rating, pounds, days_since_lr,
           placing_numerical, bfsp
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
print(f"{len(d):,} rows, {d.raceid.nunique():,} races ({time.time() - t0:.0f}s)")

# --- 1. fields ------------------------------------------------------------------------------
for c in ("race_restrictions_age", "major", "race_class"):
    vc = d[c].fillna("<null>").astype(str).str.strip().replace("", "<empty>").value_counts()
    print(f"\n== {c}: {vc.size} distinct; top 25")
    print(vc.head(25).to_string())
dt = pd.to_datetime(d.race_date)
d["month"] = dt.dt.month
d["weekday"] = dt.dt.day_name().str[:3]
text = (d.race_type.fillna("") + " " + d.race_name.fillna("")).str.lower()
d["hcap"] = text.str.contains(r"handicap|nursery|h'cap|hcap")
d["jumps"] = text.str.contains(r"hurdle|chase|nh flat|bumper|national hunt")
bt = (d.major.fillna("").astype(str) + " " + d.race_name.fillna("")).str.lower()
d["black_type"] = bt.str.contains(r"group|grade [123]|\bg[123]\b|listed|\(l\)")
print(f"\nhandicaps {d.hcap.mean():.3f}; jumps {d.jumps.mean():.3f}; black type {d.black_type.mean():.3f}")
print("rows by weekday:", d.weekday.value_counts().to_dict())

# --- the market's own chance -------------------------------------------------------------------
bsp = pd.to_numeric(d.bfsp, errors="coerce")
inv = 1 / bsp.where(bsp > 1)
full = inv.groupby(d.raceid).transform("count") == d.groupby("raceid").horse_name.transform("size")
d["p"] = (inv / inv.groupby(d.raceid).transform("sum")).where(full)
d["won"] = pd.to_numeric(d.placing_numerical, errors="coerce").eq(1).astype(float)
d["r"] = d.won - d.p
age = pd.to_numeric(d.horse_age, errors="coerce")
ra = d.race_restrictions_age.fillna("").astype(str).str.lower()
lo = pd.to_numeric(ra.str.extract(r"(\d+)")[0], errors="coerce")
# (run 28 matched "o$" here, which every "..yo" value ends in, so its age_band table was all "open";
# age_vs_min, which answers G4, was unaffected. Fixed below, not re-run.)
d["age_band"] = np.select([ra.str.contains(r"\+|up|and over"), ra.str.contains(r"\d")], ["open", "exact"], "none")
d["age_vs_min"] = (age - lo).clip(-1, 4)
dsl = pd.to_numeric(d.days_since_lr, errors="coerce")
d["season_opener"] = dsl.ge(90) & d.month.isin([3, 4, 5]) & ~d.jumps

s = d[d.p.notna() & (d.race_date >= "2023-01-01")]


def ae_table(col):
    g = s.groupby(col, observed=True)
    t = pd.DataFrame({"runners": g.size(), "A/E": g.won.sum() / g.p.sum(), "won-p": g.r.mean()})
    t["t"] = t["won-p"] / (g.r.std() / np.sqrt(g.size()))
    return t.round(4)


print(f"\n== 2. does the price take each context as a whole? (2023 on, {len(s):,} runners)")
for col in ("month", "weekday", "age_band", "age_vs_min", "black_type", "hcap", "season_opener"):
    print(f"\nby {col}")
    print(ae_table(col).to_string())

# --- 3. persistence -----------------------------------------------------------------------------
e = d[d.p.notna()].copy()
e["half"] = np.where(e.race_date < "2024-01-01", 0, 1)
e["quarter"] = (e.month - 1) // 3
e["tr"] = e.trainer.fillna("").str.lower().str.strip()
print("\n== 3. trainer x quarter against the price: 2021-23 -> 2024-26Q1 (cells with 50+ runners in both)")
c = e.groupby(["tr", "quarter", "half"]).r.agg(["mean", "size"]).unstack()
c = c[(c[("size", 0)] >= 50) & (c[("size", 1)] >= 50)]
# the trainer's seasonal SHAPE: its quarter residual less its all-year residual, each half
allyr = e.groupby(["tr", "half"]).r.mean().unstack()
shape0 = c[("mean", 0)] - allyr.reindex(c.index.get_level_values(0))[0].to_numpy()
shape1 = c[("mean", 1)] - allyr.reindex(c.index.get_level_values(0))[1].to_numpy()
print(f"  {len(c)} cells: corr of the quarter residual {np.corrcoef(c[('mean', 0)], c[('mean', 1)])[0, 1]:.3f}; "
      f"of the seasonal shape (quarter less all-year) {np.corrcoef(shape0, shape1)[0, 1]:.3f}")

n = pd.to_numeric(d.number_of_runners, errors="coerce")
pos = pd.to_numeric(d.placing_numerical, errors="coerce")
d["nfp_c"] = ((n - pos) / (n - 1)).where(n > 1)
d["nfp_c"] = d.nfp_c - d.groupby("raceid").nfp_c.transform("mean")
h = d[d.nfp_c.notna()].copy()
h["quarter"] = (h.month - 1) // 3
h["yr"] = h.race_date.str[:4].astype(int)
print("\n== 3b. the horse's own seasonality (J3): same quarter in earlier years, less its overall form then")
prev = h[h.yr <= 2023]; now = h[h.yr >= 2024]
overall = prev.groupby("horse_name").nfp_c.agg(["mean", "size"])
byq = prev.groupby(["horse_name", "quarter"]).nfp_c.agg(["mean", "size"])
byq = byq[byq["size"] >= 3]
x = now.join(overall, on="horse_name").join(byq, on=["horse_name", "quarter"], rsuffix="_q").dropna(subset=["mean", "mean_q"])
x = x[x["size"] >= 6]
sig = x["mean_q"] - x["mean"]
out = x.nfp_c - x["mean"]
print(f"  {len(x):,} runs 2024+ by horses with 6+ runs and 3+ in the quarter before: "
      f"corr(seasonal shape then, form less usual now) {np.corrcoef(sig, out)[0, 1]:+.4f}")
print("\n== 3c. handicap against other form (G2): the gap seen before, and the gap now")
hc = prev.assign(hcap=d.loc[prev.index, "hcap"]).groupby(["horse_name", "hcap"]).nfp_c.agg(["mean", "size"]).unstack()
hc = hc[(hc[("size", True)] >= 3) & (hc[("size", False)] >= 3)]
gap_then = (hc[("mean", True)] - hc[("mean", False)]).rename("gap")
y = now.assign(hcap=d.loc[now.index, "hcap"]).join(gap_then, on="horse_name").join(overall, on="horse_name").dropna(subset=["gap", "mean"])
sgn = np.where(y.hcap, 1.0, -1.0)
print(f"  {len(y):,} runs 2024+: corr(gap then x side today, form less usual now) "
      f"{np.corrcoef(y.gap * sgn, y.nfp_c - y['mean'])[0, 1]:+.4f}")
print(f"\ndone in {time.time() - t0:.0f}s")
