"""Which way does a horse's last-run NFP point? (read-only check of nfp_formula_check's rank correlation, 6 Oct)

The rank correlation in done/nfp_formula_check.py came out with better past finishes beside worse current ones, for
both formulas. This reads the same windows on the same sample and prints: one horse's runs (dates, places, NFP and
the window read before each run), the win rate and the mean finishing place by fifth of the last-run NFP, the share
of the field each fifth finishes in front of, and the sign of the conditional logit's coefficient.
"""
import os, sqlite3, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from residual_screen import Part, fit_clogit  # noqa: E402

con = sqlite3.connect(os.environ.get("DB_PATH") or "horse_racing.db")
d = pd.read_sql_query("SELECT race_date, race_time, track, horse_name, number_of_runners, placing_numerical, place, bfsp "
                      "FROM race_results WHERE race_date >= '2021-01-01'", con)
con.close()
print("race_date examples:", d.race_date.drop_duplicates().head(5).tolist(), "| place examples:",
      d[["place", "placing_numerical"]].drop_duplicates().head(8).values.tolist())
for c in ("number_of_runners", "placing_numerical", "bfsp"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["date"] = pd.to_datetime(d.race_date, errors="coerce")
print("unparsed dates:", int(d.date.isna().sum()))
d["race"] = d.race_date.astype(str) + "|" + d.track.astype(str) + "|" + d.race_time.astype(str)
N = d.number_of_runners
F = d.placing_numerical.where((d.placing_numerical >= 1) & (d.placing_numerical <= N))
d["nfp"] = (N - F) / (N - 1)
d = d.sort_values(["horse_name", "date", "race_time"], kind="stable")
d["nfp_l1"] = d.groupby("horse_name", sort=False).nfp.shift(1)
d["runs_in_race"] = d.groupby("race").horse_name.transform("size")
h = d[d.horse_name == d.horse_name[d.groupby("horse_name").horse_name.transform("size") >= 8].iloc[0]]
print(h[["race_date", "track", "number_of_runners", "runs_in_race", "placing_numerical", "nfp", "nfp_l1"]].head(10).to_string())
e = d[(d.date >= "2022-01-01") & d.nfp_l1.notna() & d.placing_numerical.notna()].copy()
e["fifth"] = pd.qcut(e.nfp_l1.rank(method="first"), 5, labels=["worst", "2", "3", "4", "best"])
e["won"] = e.placing_numerical == 1
e["beaten_share"] = (e.number_of_runners - e.placing_numerical) / (e.number_of_runners - 1)
print(e.groupby("fifth", observed=True).agg(runs=("won", "size"), win_rate=("won", "mean"),
                                            mean_place=("placing_numerical", "mean"),
                                            share_beaten=("beaten_share", "mean"),
                                            nfp_l1=("nfp_l1", "mean")).round(4).to_string())
print("runners in the table vs number_of_runners (share equal):", float((d.runs_in_race == d.number_of_runners).mean()))
w = e.groupby("race").agg(n=("won", "size"), wins=("won", "sum"))
s = e[e.race.isin(w.index[(w.wins == 1) & (w.n >= 3)])].sort_values(["date", "race"]).reset_index(drop=True)
codes = pd.factorize(s.race)[0]
p = Part(np.arange(len(s)), codes, s.won.to_numpy(float), np.zeros((len(s), 0)))
b = fit_clogit(((s.nfp_l1 - s.nfp_l1.mean()) / s.nfp_l1.std()).to_numpy()[:, None], p.y, *p.blk)
print("conditional logit on the winner, z of last-run NFP: coefficient", np.round(b, 4))
