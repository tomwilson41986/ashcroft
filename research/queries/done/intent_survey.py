"""Survey for an 'intent' block: the connections' choices before the off, against the price.

Development data only (race_date before 2026-04-01; the holdout is not read).

1. The fields the angles need: horse_sex, headgear, race_class, race_type, trainer,
   jockeys_claim and career_runs -- values and coverage.
2. What career_runs counts: against the runs the database holds for horses first seen
   well after it starts (whose whole career it should hold). 0 = runs before this one.
3. How often each angle occurs (2023 on), and whether the market prices it as a whole:
   wins against the race-normalised BSP probability (A/E), and the mean of won - p.
4. Does a trainer's record with an angle, against the price, persist? For trainer x
   angle cells with enough runners in both halves (2021-23 and 2024-26Q1), the
   correlation of won - p between the halves, and the second half's residual for the
   trainers whose first half was best and worst. No persistence, no block.
"""
import sqlite3
import time

import numpy as np
import pandas as pd

t0 = time.time()
pd.set_option("display.width", 220)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, surface_type, race_class, race_name, number_of_runners,
           horse_name, horse_sex, headgear, trainer, jockey_name, jockeys_claim, career_runs, days_since_lr,
           placing_numerical, bfsp
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
print(f"{len(d):,} rows, {d.raceid.nunique():,} races ({time.time() - t0:.0f}s)")

# --- 1. fields ------------------------------------------------------------------------------
for c in ("horse_sex", "headgear", "race_class", "jockeys_claim"):
    vc = d[c].fillna("<null>").astype(str).str.strip().replace("", "<empty>").value_counts()
    print(f"\n== {c}: {vc.size} distinct; top 30")
    print(vc.head(30).to_string())
cr = pd.to_numeric(d.career_runs, errors="coerce")
print(f"\n== career_runs: coverage {cr.notna().mean():.3f}; quantiles "
      f"{cr.quantile([0, .1, .5, .9, 1]).round(1).tolist()}")
print(f"== trainer coverage {d.trainer.fillna('').str.strip().ne('').mean():.3f}; "
      f"bfsp coverage {pd.to_numeric(d.bfsp, errors='coerce').gt(1).mean():.3f}")

# --- the horse's previous run (an earlier day) -------------------------------------------------
d["day"] = pd.to_datetime(d.race_date).values.astype("datetime64[D]").astype(np.int64)
t = d.race_time.astype(str).str.strip().str.rstrip(".").str.replace(".", ":", regex=False)
hm = t.str.extract(r"^(\d{1,2})(?::(\d{1,2}))?")
h = pd.to_numeric(hm[0], errors="coerce")
d["mins"] = (h.where((h >= 11) | h.isna(), h + 12) * 60 + pd.to_numeric(hm[1], errors="coerce").fillna(0)).fillna(0)
d = d.sort_values(["horse_name", "day", "mins"]).reset_index(drop=True)
g = d.groupby("horse_name", sort=False)
d["seen"] = g.cumcount()
first_day = g["day"].transform("min")
same_day_prev = g["day"].shift(1).eq(d.day)
print(f"\nhorse rows sharing a day with the horse's previous row: {int(same_day_prev.sum())}")
for c in ("trainer", "horse_sex", "headgear", "race_class", "day", "jockeys_claim"):
    d[f"prev_{c}"] = g[c].shift(1)

# --- 2. what career_runs counts --------------------------------------------------------------
late = d[first_day >= pd.Timestamp("2022-01-01").to_datetime64().astype("datetime64[D]").astype(np.int64)]
diff = (pd.to_numeric(late.career_runs, errors="coerce") - late.seen).dropna()
print(f"\n== career_runs less the runs held, horses first seen from 2022 ({len(diff):,} rows)")
print(diff.clip(-3, 6).value_counts().sort_index().to_string())
fs = d[d.seen == 0]
print("career_runs on each horse's first row held, by year first seen")
print(fs.assign(y=fs.race_date.str[:4], cr=pd.to_numeric(fs.career_runs, errors="coerce").clip(upper=5))
      .groupby("y").cr.value_counts(normalize=True).unstack().round(3).to_string())

# --- 3. angles -------------------------------------------------------------------------------
rt = (d.race_type.fillna("") + " " + d.race_name.fillna("")).str.lower()
d["hcap"] = rt.str.contains("handicap|nursery")
d["prior_hcaps"] = d.hcap.astype(int).groupby(d.horse_name).cumsum() - d.hcap.astype(int)
code = np.select([rt.str.contains("chase"), rt.str.contains("hurdle"),
                  rt.str.contains("nh flat|bumper|national hunt flat")], ["chase", "hurdle", "nhflat"], "flat")
code = np.where((code == "flat") & d.surface_type.fillna("").str.lower().str.contains(
    r"\baw\b|all[- ]weather|polytrack|tapeta|fibresand|standard|sand"), "aw", code)
d["code"] = code
d["prior_in_code"] = d.groupby(["horse_name", "code"]).cumcount()
sex = d.horse_sex.fillna("").astype(str).str.strip().str.lower().str[:1]
psex = d.prev_horse_sex.fillna("").astype(str).str.strip().str.lower().str[:1]
hg = d.headgear.fillna("").astype(str).str.strip().str.lower()
d["has_hg"] = hg.ne("")
d["prior_hg"] = d.has_hg.astype(int).groupby(d.horse_name).cumsum() - d.has_hg.astype(int)
cls = pd.to_numeric(d.race_class.astype(str).str.extract(r"(\d)")[0], errors="coerce")
pcls = pd.to_numeric(d.prev_race_class.astype(str).str.extract(r"(\d)")[0], errors="coerce")
cr = pd.to_numeric(d.career_runs, errors="coerce")
first_cr = cr.where(d.seen.eq(0)).groupby(d.horse_name).transform("max")
complete = first_cr.le(1) | (first_day >= first_day.min() + 400)
tr = d.trainer.fillna("").str.strip().str.lower()
ptr = d.prev_trainer.fillna("").astype(str).str.strip().str.lower()
has_prev = d.seen > 0
gap = d.day - d.prev_day
d["a_new_yard"] = has_prev & ptr.ne("") & tr.ne("") & tr.ne(ptr)
d["a_gelded"] = has_prev & sex.eq("g") & psex.isin(["c", "h", "r"])
d["a_hcap_debut"] = has_prev & complete & d.hcap & d.prior_hcaps.eq(0)
d["a_ft_headgear"] = has_prev & (complete | d.seen.ge(5)) & d.has_hg & d.prior_hg.eq(0)
d["a_layoff"] = has_prev & gap.ge(90)
d["a_class_drop"] = has_prev & (cls - pcls).ge(1)
d["a_code_switch"] = has_prev & (complete | d.seen.ge(5)) & d.prior_in_code.eq(0)
d["a_debut"] = (d.seen == 0) & (pd.to_numeric(d.career_runs, errors="coerce").fillna(0) <= 0) & \
               (d.day >= first_day.min() + 30)
ANGLES = [c for c in d.columns if c.startswith("a_")]

bsp = pd.to_numeric(d.bfsp, errors="coerce")
inv = (1 / bsp.where(bsp > 1))
d["p"] = inv / inv.groupby(d.raceid).transform("sum")
d["won"] = pd.to_numeric(d.placing_numerical, errors="coerce").eq(1).astype(float)
d["r"] = d.won - d.p
ok = d.p.notna() & inv.groupby(d.raceid).transform("count").eq(d.groupby("raceid").horse_name.transform("size"))
s = d[ok & (d.race_date >= "2023-01-01")]
print(f"\n== angles, 2023 on ({len(s):,} runners in fully priced races)")
rows = []
for a in ANGLES:
    x = s[s[a]]
    se = x.r.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan
    rows.append({"angle": a[2:], "runners": len(x), "share": len(x) / len(s), "win": x.won.mean(),
                 "p_bsp": x.p.mean(), "A/E": x.won.sum() / x.p.sum(), "won-p": x.r.mean(), "t": x.r.mean() / se})
print(pd.DataFrame(rows).set_index("angle").round(4).to_string())

# --- 4. persistence of trainer x angle residuals ------------------------------------------------
print("\n== trainer x angle: does the residual against BSP persist? (2021-23 -> 2024-26Q1)")
e = d[ok].copy()
e["half"] = np.where(e.race_date < "2024-01-01", 0, 1)
e["tr"] = tr[ok]
out = []
for a in ANGLES + ["all"]:
    x = e if a == "all" else e[e[a]]
    c = x.groupby(["tr", "half"]).r.agg(["mean", "size"]).unstack().reindex(
        columns=pd.MultiIndex.from_product([["mean", "size"], [0, 1]]))
    c[[("size", 0), ("size", 1)]] = c[[("size", 0), ("size", 1)]].fillna(0)
    for m in (20, 50):
        cc = c[(c[("size", 0)] >= m) & (c[("size", 1)] >= m)]
        if len(cc) < 10:
            out.append({"angle": a.replace("a_", ""), "min_n": m, "cells": len(cc)})
            continue
        m0, m1 = cc[("mean", 0)], cc[("mean", 1)]
        q = pd.qcut(m0.rank(method="first"), 5, labels=False)
        w1 = cc[("size", 1)]
        out.append({"angle": a.replace("a_", ""), "min_n": m, "cells": len(cc), "corr": np.corrcoef(m0, m1)[0, 1],
                    "top5th_later": np.average(m1[q == 4], weights=w1[q == 4]),
                    "bottom5th_later": np.average(m1[q == 0], weights=w1[q == 0])})
print(pd.DataFrame(out).set_index(["angle", "min_n"]).round(4).to_string())
print(f"\ndone in {time.time() - t0:.0f}s")
