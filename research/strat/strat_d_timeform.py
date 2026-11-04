"""Strategy study D (4 Nov 2026, read-only): Timeform ratings from the Railway backend (APIData_Table2), GB + IRE.

The owner: "you should use timeform on railway too". The Timeform record carries what the HRB database does not:
a performance rating for every run (performanceRating), Timeform's own PRE-RACE ratings for the race
(preRaceAdjustedRating, preRaceMasterRating), a time figure, and a stable horse id (horseCode). So the improver
question can be asked of the ratings themselves, not only of finishing positions.

Every signal on a run reads the horse's EARLIER runs (by horseCode) plus Timeform's pre-race figures, which are
published before the off. Chosen on 2014-2021, reported on 2022 to Nov 2026, against the Betfair SP on the same
Timeform row (betfairWinSP), 5% commission, staked level and to win GBP100 (strat_lib).

    python research/strat/strat_d_timeform.py <dir of monthly parquet from fetch_timeform.py>
"""

import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)

files = sorted(glob.glob(f"{sys.argv[1]}/*.parquet"))
tf = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print("rows in", len(tf), "files", len(files))
num = ["raceNumber", "numberOfRunners", "raceClass", "horseCode", "horseAge", "positionOfficial", "Win",
       "betfairWinSP", "ispDecimal", "performanceRating", "preRaceAdjustedRating", "preRaceMasterRating",
       "timefigure", "distanceBeaten", "distance", "officialRating", "prizeFund"]
for c in num:
    tf[c] = pd.to_numeric(tf[c], errors="coerce")
tf["date"] = pd.to_datetime(tf["meetingDate"].str[:10], errors="coerce")
tf["race_date"] = tf["date"].dt.strftime("%Y-%m-%d")
tf["year"] = tf["date"].dt.year
tf["rk"] = tf["race_date"] + "|" + tf["courseName"].astype(str) + "|" + tf["raceNumber"].astype(str)
dup = tf.duplicated(["rk", "horseCode"])
print("duplicate rows dropped", int(dup.sum()))
tf = tf[~dup & tf["horseCode"].notna() & tf["date"].notna()].copy()
tf["bsp"] = tf["betfairWinSP"]
tf["won"] = (tf["positionOfficial"] == 1).astype(float)
tf["won"] = tf["Win"].fillna(tf["won"]).clip(0, 1)
# the Railway table holds Flat racing only (raceType is "Flat" on every row): turf or all-weather
tf = tf[tf["raceType"].fillna("Flat") == "Flat"].copy()
tf["code"] = np.where(tf["raceSurfaceName"].fillna("").str.contains("Weather", case=False), "aw", "flat")
tf["hcap"] = tf["raceTitle"].fillna("").str.contains("handicap", case=False)
print("by code", tf["code"].value_counts().to_dict())
print("BSP held", round(float((tf["bsp"] > 1).mean()), 3), "| TFR held", round(float(tf["performanceRating"].notna().mean()), 3),
      "| pre-race adjusted held", round(float(tf["preRaceAdjustedRating"].notna().mean()), 3),
      "| time figure held", round(float((tf["timefigure"] > 0).mean()), 3))
print("dates", tf["race_date"].min(), tf["race_date"].max())

tf = tf.sort_values(["horseCode", "date", "scheduledTimeOfRaceLocal"]).reset_index(drop=True)
g = tf.groupby("horseCode", sort=False)
tfr = tf["performanceRating"].where(tf["performanceRating"] > 0)
tf["tfr"] = tfr
for i in (1, 2, 3):
    tf[f"tfr{i}"] = g["tfr"].shift(i)
tf["prev_date"] = g["date"].shift(1)
tf["prev_fin"] = g["positionOfficial"].shift(1)
tf["prev_tfig"] = g["timefigure"].shift(1)
tf["prev_master"] = g["preRaceMasterRating"].shift(1)
tf["runs_before"] = g.cumcount()
tf["best_before"] = g["tfr"].transform(lambda s: s.shift(1).cummax())
tf["best_before2"] = g["tfr"].transform(lambda s: s.shift(2).cummax())  # best before the LAST run
tf["dslr"] = (tf["date"] - tf["prev_date"]).dt.days
# today's field: Timeform's pre-race adjusted rating, its rank, and the gap to the next best
tf["adj_rank"] = tf.groupby("rk")["preRaceAdjustedRating"].rank(ascending=False, method="min")
tf["bsp_rank"] = tf.groupby("rk")["bsp"].rank(method="min")
second = tf.groupby("rk")["preRaceAdjustedRating"].transform(lambda s: s.nlargest(2).iloc[-1] if s.notna().sum() > 1 else np.nan)
tf["adj_gap"] = tf["preRaceAdjustedRating"] - second
d = tf[(tf["bsp"] > 1) & tf["positionOfficial"].notna()].copy()
print("priced runs", len(d))

# ---- 1. Timeform's top-rated against the market -----------------------------------------------------------------
top = d[d["adj_rank"] == 1].copy()
top["mkt"] = pd.cut(top["bsp_rank"], [0, 1, 2, 3, 99], labels=["BSP fav", "2nd fav", "3rd fav", "4th+"])
L.show("Timeform top-rated (pre-race adjusted) by its place in the BSP market", L.tt(top, ["code", "mkt"], 200))
top["gapb"] = pd.cut(top["adj_gap"], [-1, 0, 2, 5, 10, 99], labels=["tie", "1-2", "3-5", "6-10", "11+"])
L.show("Timeform top-rated, NOT the BSP favourite, by its margin over the next best", L.tt(top[top["bsp_rank"] > 1], ["gapb"], 200))

# ---- 2. improvers on the ratings ----------------------------------------------------------------------------------
d["pb_lto"] = d["tfr1"] > d["best_before2"]                      # last run was a career best
d["pb_margin"] = d["tfr1"] - d["best_before2"]
d["up2"] = (d["tfr1"] > d["tfr2"]) & (d["tfr2"] > d["tfr3"])     # rating rose at each of the last two runs
d["slope"] = (d["tfr1"] - d["tfr3"]) / 2
d["above_master"] = d["tfr1"] - d["prev_master"]                # ran above Timeform's own master rating last time
d["lightly"] = d["runs_before"] <= 6
d["fast_time"] = (d["prev_tfig"] > 0) & (d["prev_tfig"] - d["tfr1"] >= 5)  # time figure well above the run's rating
rows = []
for code in ("flat", "aw"):
    c = d["code"] == code
    rows += [
        L.rule(f"{code}: career-best TFR last time (by 5+)", d, c & (d["pb_margin"] >= 5)),
        L.rule(f"{code}: career-best TFR last time, lightly raced (<=6 runs)", d, c & d["pb_lto"] & d["lightly"]),
        L.rule(f"{code}: TFR up at each of last two runs", d, c & d["up2"]),
        L.rule(f"{code}: TFR up twice, by 10+ over the two", d, c & d["up2"] & (d["slope"] >= 5)),
        L.rule(f"{code}: ran 5+ above its master rating last time", d, c & (d["above_master"] >= 5)),
        L.rule(f"{code}: time figure 5+ above its TFR last time", d, c & d["fast_time"]),
        L.rule(f"{code}: TFR up twice AND back within 21 days", d, c & d["up2"] & (d["dslr"] <= 21)),
        L.rule(f"{code}: career-best LTO AND Timeform top-rated today, not BSP fav", d,
               c & d["pb_lto"] & (d["adj_rank"] == 1) & (d["bsp_rank"] > 1)),
    ]
print("\n=== TIMEFORM IMPROVERS (train 2014-21 | t_ test 2022-26) ===")
L.rules_table(rows)

# ---- 3. the rating trajectory against the market, finer --------------------------------------------------------------
d["slope_b"] = pd.cut(d["slope"], [-99, -10, -5, -2, 2, 5, 10, 99], labels=["<-10", "-10..-5", "-5..-2", "flat", "2..5", "5..10", "10+"])
L.show("A/E by the TFR slope over the last three runs", L.tt(d, ["code", "slope_b"], 300))

# ---- 4. days since the last run on the Timeform record (same horse id, every country held) ----------------------------
bins = [0, 3, 4, 6, 7, 10, 14, 21, 35, 60, 120, 365, 5000]
labels = ["1-3", "4", "5-6", "7", "8-10", "11-14", "15-21", "22-35", "36-60", "61-120", "121-365", "366+"]
d["dslr_b"] = pd.cut(d["dslr"], bins, labels=labels)
L.show("Timeform record: DSLR by code", L.tt(d, ["code", "dslr_b"], 200))
q = d[d["dslr"] < 5]
L.show("DSLR<5 by code and whether it ran to a career best last time", L.tt(q, ["code", "pb_lto"], 80))
d.to_parquet(sys.argv[1] + "/../tf_features.parquet", index=False)
