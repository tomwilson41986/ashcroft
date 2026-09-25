"""Survey for the draw and position rebuild: what the comments, margins and stalls hold.

Development data only (race_date before 2026-04-01; the holdout is not read).

1. Comments: coverage by year and race type; how often the first-phrase parser in
   model/race_shape.py finds a position; the openings of the comments it cannot read.
2. The parser in production (model/pace_metrics.parse_run_style) against it: how often
   production calls a horse a leader (>= 5.8) that the first phrase does not, and how
   often production falls back to midfield (3.0) on a comment with no position.
3. Margins: total_dst_bt formats and parse rate, winners' values, monotone in placing.
4. Stalls: coverage by race type, stall_positioning values, field sizes, and how many
   races each course x distance (x field-size band) cell holds.
5. Style persistence: last run's class against this run's; projected (EWM of earlier runs)
   against actual.
6. Outcomes by actual and by projected early position, and the lone-leader effect:
   normalised finishing position and lengths beaten (in lbs at the trip) by position,
   by field size, by the number of projected leaders in the race.
"""
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.pace_metrics import parse_run_style  # noqa: E402
from model.perf_figures import lbs_per_length, parse_beaten_lengths  # noqa: E402
from model.race_shape import STYLE_CLASSES, parse_styles, style_class  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 200)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, surface_type, race_class, number_of_runners,
           horse_name, stall, stall_positioning, placing_numerical, total_dst_bt, distbt,
           dist_furlongs, going_description, comment, bfsp
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
print(f"{len(d):,} rows, {d.race_date.min()} to {d.race_date.max()} ({time.time() - t0:.0f}s)")
d["year"] = d.race_date.str[:4]
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d["n"] = pd.to_numeric(d.number_of_runners, errors="coerce")
d["pos"] = pd.to_numeric(d.placing_numerical, errors="coerce")
d["stall"] = pd.to_numeric(d.stall, errors="coerce")
rt = d.race_type.fillna("").str.lower()
d["code"] = np.select([rt.str.contains("chase"), rt.str.contains("hurdle"),
                       rt.str.contains("nh flat|bumper|national hunt flat")],
                      ["chase", "hurdle", "nhflat"], default="flat")
d.loc[(d.code == "flat") & d.surface_type.fillna("").str.lower().str.contains("aw|all|poly|tapeta|fibre|sand"),
      "code"] = "aw"

print("\n== race_type values (top 25)")
print(d.race_type.value_counts().head(25).to_string())
print("\n== surface_type values")
print(d.surface_type.value_counts(dropna=False).head(10).to_string())
print("\n== rows by code")
print(d.code.value_counts().to_string())

# --- 1. comments ------------------------------------------------------------------------
has = d.comment.fillna("").str.strip().str.len() > 0
print("\n== 1. comment present, by year and code")
print(pd.crosstab(d.year, d.code, values=has, aggfunc="mean").round(3).to_string())
t1 = time.time()
d["sty"] = parse_styles(d.comment)
d["cls"] = style_class(d["sty"])
print(f"\nparsed in {time.time() - t1:.0f}s; position found on {d['sty'].notna()[has].mean():.3f} of comments")
print(pd.crosstab(d.code, d.cls.map(dict(enumerate(STYLE_CLASSES))).fillna("none"), normalize="index").round(3).to_string())
print("\nstyle score distribution:")
print(d["sty"].value_counts(dropna=False).sort_index().to_string())
unread = d.loc[has & d["sty"].isna(), "comment"].str.lower().str.replace(r"[^a-z ]", " ", regex=True)
opener = unread.str.split().str[:3].str.join(" ")
print(f"\ncomments with no position found: {len(unread):,}; commonest openings:")
print(opener.value_counts().head(40).to_string())
print("\nexamples:")
for c in d.loc[has & d["sty"].isna(), "comment"].sample(min(25, int((has & d['sty'].isna()).sum())),
                                                          random_state=1):
    print("  ", c)

# --- 2. the production parser ----------------------------------------------------------
sample = d[has].sample(min(200_000, int(has.sum())), random_state=2)
t2 = time.time()
old = sample.comment.map(lambda c: parse_run_style(c)["early_pos"])
print(f"\n== 2. production parser on {len(sample):,} comments ({time.time() - t2:.0f}s)")
lead_old = old >= 5.8
print(f"production calls a leader: {lead_old.mean():.3f}; first phrase calls a leader: {(sample.cls == 0).mean():.3f}")
mis = lead_old & (sample.cls != 0)
print(f"production leader, first phrase not: {mis.mean():.3f} of comments "
      f"({mis.sum() / max(lead_old.sum(), 1):.3f} of its leaders)")
print(pd.crosstab(sample.cls.map(dict(enumerate(STYLE_CLASSES))).fillna("none")[mis], "n").to_string())
for c in sample.loc[mis, "comment"].head(15):
    print("  ", c)
dflt = (old == 3.0) & sample["sty"].isna()
print(f"production 3.0 (midfield) on a comment with no position: {dflt.mean():.3f}")
agree = pd.crosstab(style_class(old), sample.cls, normalize="all").round(3)
print("class agreement (rows production, columns first phrase):")
print(agree.to_string())

# --- 3. margins -------------------------------------------------------------------------
d["lb"] = d.total_dst_bt.map(parse_beaten_lengths)
print("\n== 3. total_dst_bt")
print(f"parsed: {d.lb.notna().mean():.3f}; winners parsed {d.lb[d.pos == 1].notna().mean():.3f} "
      f"(values {d.lb[d.pos == 1].round(2).value_counts().head(5).to_dict()})")
bad = d.loc[d.total_dst_bt.notna() & d.lb.isna(), "total_dst_bt"].astype(str)
print("unparsed formats:", bad.value_counts().head(20).to_dict())
q = d.loc[d.pos > 1, "lb"].quantile([.1, .25, .5, .75, .9, .99]).round(2).to_dict()
print("lengths behind the winner (placed 2nd+):", q)
s = d[d.pos.notna() & d.lb.notna()].sort_values(["race", "pos"])
mono = s.groupby("race")["lb"].apply(lambda x: bool((np.diff(x.to_numpy()) >= -1e-9).all()))
print(f"cumulative (non-decreasing in placing) in {mono.mean():.3f} of races")
d["lbs"] = (d.lb.where(d.pos > 1, 0.0) * lbs_per_length(d.dist_furlongs.fillna(8))).clip(upper=20)
print("lbs behind the winner at the trip (capped 20):",
      d.loc[d.pos > 1, "lbs"].quantile([.25, .5, .75, .9]).round(2).to_dict())

# --- 4. stalls --------------------------------------------------------------------------
print("\n== 4. stalls")
print(pd.crosstab(d.code, d.stall.gt(0), normalize="index").round(3).to_string())
print("stall_positioning (flat/aw):")
print(d.loc[d.code.isin(["flat", "aw"]), "stall_positioning"].value_counts(dropna=False).head(25).to_string())
f = d[d.code.isin(["flat", "aw"]) & d.stall.gt(0)].copy()
races = f.drop_duplicates("race")
print("field sizes (flat/aw races):", races.n.describe().round(1).to_dict())
races = races.assign(dist=races.dist_furlongs.round(0),
                     band=pd.cut(races.n, [0, 7, 11, 15, 99], labels=["2-7", "8-11", "12-15", "16+"]))
cd = races.groupby(["track", "dist"]).size()
print(f"course x distance cells: {len(cd):,}; races per cell quantiles "
      f"{cd.quantile([.1, .25, .5, .75, .9]).to_dict()}; cells with >= 50 races: {(cd >= 50).sum()}")
cdb = races.groupby(["track", "dist", "band"], observed=True).size()
print(f"course x distance x field band cells: {len(cdb):,}; >= 30 races: {(cdb >= 30).sum()}; "
      f"median {cdb.median():.0f}")
sp = races.groupby(["track", "dist"])["stall_positioning"].nunique()
print(f"course x distance cells with more than one stall_positioning value: {(sp > 1).mean():.3f}")

# --- 5. style persistence and the projection -------------------------------------------
from model.race_shape import add_race_shape, add_run_outcomes, add_run_styles, add_style_projection  # noqa: E402

print("\n== 5. style persistence (horses' consecutive parsed runs)")
t5 = time.time()
d["raceid"] = d["race"]
d = add_run_outcomes(d)
d = add_run_styles(d)
d = add_style_projection(d)
d = add_race_shape(d)
print(f"projection and shape built in {time.time() - t5:.0f}s")
names = dict(enumerate(STYLE_CLASSES))
d = d.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
d["prev_cls"] = d.groupby("horse_name", sort=False)["rs_class"].shift(1)
m = d.prev_cls.notna() & d.rs_class.notna()
print("P(this run's class | last run's class):")
print(pd.crosstab(d.prev_cls[m].map(names), d.rs_class[m].map(names), normalize="index").round(3).to_string())
P = d[["p_lead", "p_prom", "p_mid", "p_rear"]].to_numpy()
d["pred_cls"] = P.argmax(axis=1).astype(float)
ok = d.rs_epf.notna() & (d["style_n_eff"] > 0)
print(f"corr(projected epf, actual epf), horses with history: {np.corrcoef(d.pred_epf[ok], d.rs_epf[ok])[0, 1]:.3f} "
      f"on {ok.sum():,} runs")
m = d.rs_class.notna()
print("P(actual class | projected class), all runs incl. debutants:")
print(pd.crosstab(d.pred_cls[m].map(names), d.rs_class[m].map(names), normalize="index").round(3).to_string())
print("mean projected P by actual class (calibration):")
print(d[m].groupby(d.rs_class[m].map(names))[["p_lead", "p_prom", "p_mid", "p_rear"]].mean().round(3).to_string())
act = pd.get_dummies(d.rs_class[m].map(names)).reindex(columns=list(STYLE_CLASSES), fill_value=0).to_numpy(float)
brier = ((P[m.to_numpy()] - act) ** 2).sum(axis=1).mean()
base = ((act.mean(axis=0) - act) ** 2).sum(axis=1).mean()
print(f"Brier score of the class projection {brier:.4f} vs base rates {base:.4f} (skill {1 - brier / base:.3f})")

# --- 6. outcomes by position ------------------------------------------------------------
print("\n== 6. outcomes by early position (2022 onwards, runners with a result)")
d["won"] = (d.pos == 1).astype(float)
e = d[(d.race_date >= "2022-01-01") & d.pos.notna()].copy()
e["band"] = pd.cut(e.n, [0, 7, 11, 15, 99], labels=["2-7", "8-11", "12-15", "16+"])
e["shape"] = e.shape_bin.map({0.0: "no natural leader", 1.0: "one natural leader", 2.0: "contested lead"})


def tab(frame, by, label):
    t = frame.groupby(by, observed=True).agg(runs=("rs_nfp_c", "size"), nfp_c=("rs_nfp_c", "mean"),
                                             lbs_c=("rs_lbs_c", "mean"), win=("won", "mean"), wins=("won", "sum"))
    exp = frame.groupby(by, observed=True)["n"].apply(lambda x: (1 / x).sum())
    t["win_vs_random"] = t.wins / exp
    print(f"\n{label}")
    print(t.drop(columns="wins").round(3).to_string())


tab(e[e.rs_class.notna()].assign(pos=e.rs_class.map(names)), ["code", "pos"], "by ACTUAL early class")
tab(e[e.rs_class.notna()].assign(pos=e.rs_class.map(names)), ["band", "pos"], "by ACTUAL class and field size")
tab(e.assign(pos=e.pred_cls.map(names)), ["code", "pos"], "by PROJECTED class")
tab(e.assign(pos=e.pred_cls.map(names)), ["shape", "pos"], "by PROJECTED class within projected shape")
tab(e[e.rs_class.notna()].assign(pos=e.rs_class.map(names)), ["shape", "pos"], "by ACTUAL class within projected shape")
r = e.drop_duplicates("race")
print("\nraces by projected shape:", r["shape"].value_counts().to_dict())
print("lead1 quantiles:", r.shape_lead1.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
print("lead2 quantiles:", r.shape_lead2.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
print("expected leaders quantiles:", r.shape_exp_leaders.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
lead_n = e.groupby("race")["rs_class"].apply(lambda x: int((x == 0).sum()))
r2 = r.set_index("race").join(lead_n.rename("actual_leaders"))
print("actual leaders per race (comments), by projected shape:")
print(pd.crosstab(r2["shape"], r2.actual_leaders.clip(upper=3), normalize="index").round(3).to_string())
print(f"\ndone in {time.time() - t0:.0f}s")
