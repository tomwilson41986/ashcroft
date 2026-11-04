"""Strategy study A (4 Nov 2026, read-only): days since the last run, improving horses, and other systems.

The owner: "Days Since last run, e.g. <5 on flat and All Weather. Also check jumps. ... Improving horses.
Identifying these early and then backing them on runs up the curve. Research any other systems that seem to fit on
our databases. We want to use variable stakes and bet to win a set amount."

Every signal reads runs on earlier days only. Chosen on 2010-2021, reported on 2022 to Nov 2026 (strat_lib).
Scored against the Betfair SP and the morning price (2018+, where GBP50+ was matched in the morning).
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)

df = L.load()
print("rows", len(df), "dates", df.race_date.min(), df.race_date.max())
print(df.code.value_counts().to_string())

# ---- the horse's earlier runs -------------------------------------------------------------------------------
g = df.groupby("horse_name", sort=False)
for k, src in (("prev_fin", "fin"), ("prev_won", "won"), ("prev_bsp", "bsp"), ("prev_rank", "bsp_rank"),
               ("prev_or", "official_rating"), ("prev_code", "code"), ("prev_track", "track"),
               ("prev_class", "race_class"), ("prev_hg", "headgear"), ("prev_hcap", "hcap"), ("prev_date", "date"),
               ("prev_field", "field"), ("prev_dist", "dist_furlongs")):
    df[k] = g[src].shift(1)
for k, src in (("prev2_fin", "fin"), ("prev2_rank", "bsp_rank"), ("prev2_or", "official_rating")):
    df[k] = g[src].shift(2)
df["prev3_fin"] = g["fin"].shift(3)
df["runs_held"] = g.cumcount()
df["hcap_runs_before"] = g["hcap"].cumsum() - df["hcap"].astype(int)
gap = (df["date"] - df["prev_date"]).dt.days
df["dslr"] = df["days_since_lr"].where(df["days_since_lr"] > 0, gap)
# a horse's earlier wins at today's track (earlier days only)
df["_tw"] = df["won"]
trk = df.groupby(["horse_name", "track"], sort=False)["_tw"].cumsum() - df["_tw"]
df["course_winner"] = trk > 0
d = L.priced(df).copy()
print("priced rows", len(d))
L.show("CONTROL: every runner by code (win and place A/E should sit near 1.00)", L.tt(d, ["code"], 100))

# ============================================================================================================
# 1. DAYS SINCE THE LAST RUN, by code
bins = [0, 3, 4, 6, 7, 10, 14, 21, 35, 60, 120, 365, 5000]
labels = ["1-3", "4", "5-6", "7", "8-10", "11-14", "15-21", "22-35", "36-60", "61-120", "121-365", "366+"]
d["dslr_b"] = pd.cut(d["dslr"], bins, labels=labels)
L.show("DSLR by code (train 2010-21 | t_ test 2022-26)", L.tt(d, ["code", "dslr_b"], 200))

q = d[d["dslr"] < 5]
L.show("DSLR<5: by code and whether it won last time", L.tt(q, ["code", "prev_won"], 100))
L.show("DSLR<5: by code and handicap", L.tt(q, ["code", "hcap"], 100))
d["band"] = pd.cut(d["bsp"], [1, 3, 6, 11, 21, 1001], labels=["<3", "3-6", "6-11", "11-21", "21+"])
L.show("DSLR<5: by code and BSP band", L.tt(q.assign(band=d.loc[q.index, "band"]), ["code", "band"], 100))
d["same_code"] = d["prev_code"] == d["code"]
L.show("DSLR<5: same code as last time?", L.tt(d[d["dslr"] < 5], ["code", "same_code"], 100))
L.show("DSLR<=7 after a win (penalty-type), handicaps, by code",
       L.tt(d[(d["dslr"] <= 7) & (d["prev_won"] == 1) & d["hcap"]], ["code"], 50))

# ============================================================================================================
# 2. IMPROVING HORSES
d["beat_mkt"] = (d["prev_rank"] - d["prev_fin"]) >= 3
d["beat_mkt2"] = ((d["prev_rank"] - d["prev_fin"]) >= 2) & ((d["prev2_rank"] - d["prev2_fin"]) >= 2)
d["or_up2"] = (d["official_rating"] > d["prev_or"]) & (d["prev_or"] > d["prev2_or"])
d["fin_improving"] = (d["prev3_fin"] > d["prev2_fin"]) & (d["prev2_fin"] > d["prev_fin"])
d["lightly"] = d["career_runs"].fillna(d["runs_held"]) <= 6
d["hcap_debut"] = d["hcap"] & (d["hcap_runs_before"] == 0) & (d["runs_held"] >= 2)
rows = []
for code in ("flat", "aw", "hurdle", "chase"):
    c = d["code"] == code
    rows += [
        L.rule(f"{code}: beat its market rank by 3+ places last time, placed", d, c & d["beat_mkt"] & (d["prev_fin"] <= 3)),
        L.rule(f"{code}: beat market rank by 2+ in each of last two", d, c & d["beat_mkt2"]),
        L.rule(f"{code}: OR up at each of last two runs (handicap)", d, c & d["or_up2"] & d["hcap"]),
        L.rule(f"{code}: finishing position better at each of last three", d, c & d["fin_improving"]),
        L.rule(f"{code}: lightly raced (<=6 runs), won last time", d, c & d["lightly"] & (d["prev_won"] == 1)),
        L.rule(f"{code}: handicap debut after 2+ runs", d, c & d["hcap_debut"]),
        L.rule(f"{code}: beat market rank last time AND back within 14 days", d,
               c & d["beat_mkt"] & (d["prev_fin"] <= 3) & (d["dslr"] <= 14)),
        L.rule(f"{code}: OR up twice AND back within 14 days", d, c & d["or_up2"] & d["hcap"] & (d["dslr"] <= 14)),
    ]
print("\n=== IMPROVERS (train | t_ test) ===")
L.rules_table(rows)
ALL = list(rows)

# ============================================================================================================
# 3. OTHER SYSTEMS
hg = d["headgear"].fillna("").astype(str).str.strip()
phg = d["prev_hg"].fillna("").astype(str).str.strip()
print("\nheadgear values:", hg.value_counts().head(15).to_dict())
d["first_hg"] = (hg != "") & (phg == "") & d["prev_date"].notna()
cls = pd.to_numeric(d["race_class"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
pcls = pd.to_numeric(d["prev_class"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
d["class_drop"] = cls > pcls
d["class_drop2"] = (cls - pcls) >= 2

# the trainer's last 14 days, earlier days only
td = df[df["bsp"] > 1].assign(p=lambda x: 1 / x["bsp"]).groupby(["trainer", "date"])[["won", "p"]].sum()
td["runs"] = df[df["bsp"] > 1].groupby(["trainer", "date"]).size()
td = td.reset_index().sort_values(["trainer", "date"])
roll = (td.set_index("date").groupby("trainer")[["won", "p", "runs"]]
          .rolling("14D", closed="left").sum().reset_index())
roll.columns = ["trainer", "date", "t14_w", "t14_e", "t14_n"]
d = d.merge(roll, on=["trainer", "date"], how="left")
d["t14_ae"] = (d["t14_w"] + 2) / (d["t14_e"] + 2)
rows = [
    L.rule("first-time headgear (any code)", d, d["first_hg"]),
    L.rule("first-time headgear, flat/AW", d, d["first_hg"] & d["code"].isin(["flat", "aw"])),
    L.rule("first-time headgear, jumps", d, d["first_hg"] & d["code"].isin(["hurdle", "chase"])),
    L.rule("beaten favourite last time", d, (d["prev_rank"] == 1) & (d["prev_fin"] > 1)),
    L.rule("beaten fav LTO, back within 14 days", d, (d["prev_rank"] == 1) & (d["prev_fin"] > 1) & (d["dslr"] <= 14)),
    L.rule("class drop (GB classes)", d, d["class_drop"]),
    L.rule("class drop 2+ classes", d, d["class_drop2"]),
    L.rule("course winner", d, d["course_winner"]),
    L.rule("trainer hot: 14-day A/E>1.5 on 10+ runs", d, (d["t14_n"] >= 10) & (d["t14_ae"] > 1.5)),
    L.rule("trainer cold: 14-day 0 wins on 15+ runs", d, (d["t14_n"] >= 15) & (d["t14_w"] == 0)),
    L.rule("trainer hot + horse back <=14 days", d, (d["t14_n"] >= 10) & (d["t14_ae"] > 1.5) & (d["dslr"] <= 14)),
    L.rule("first run in a handicap, trainer hot", d, d["hcap_debut"] & (d["t14_n"] >= 10) & (d["t14_ae"] > 1.3)),
]
print("\n=== OTHER SYSTEMS (train | t_ test) ===")
L.rules_table(rows)
ALL += rows
L.show("trainer 14-day A/E bands", L.tt(d[d["t14_n"] >= 10].assign(
    b=pd.cut(d["t14_ae"], [0, .6, .8, 1, 1.2, 1.5, 2, 9])), ["b"], 500))

# ============================================================================================================
# 4. MORE SYSTEMS, written for the place market as much as the win (the owner, 4 Nov: back both)
dd = d["dist_furlongs"] - d["prev_dist"]
jd = df[df["bsp"] > 1].assign(p=lambda x: 1 / x["bsp"]).groupby(["jockey_name", "date"])[["won", "p"]].sum()
jd["runs"] = df[df["bsp"] > 1].groupby(["jockey_name", "date"]).size()
jr = (jd.reset_index().sort_values(["jockey_name", "date"]).set_index("date").groupby("jockey_name")[["won", "p", "runs"]]
        .rolling("14D", closed="left").sum().reset_index())
jr.columns = ["jockey_name", "date", "j14_w", "j14_e", "j14_n"]
d = d.merge(jr, on=["jockey_name", "date"], how="left")
d["j14_ae"] = (d["j14_w"] + 2) / (d["j14_e"] + 2)
claim = d["jockeys_claim"].fillna(0)
jumps = d["code"].isin(["hurdle", "chase"])
fl = d["code"].isin(["flat", "aw"])
rows = []
for code in ("flat", "aw", "hurdle", "chase"):
    c = d["code"] == code
    rows += [
        L.rule(f"{code}: DSLR 1-4", d, c & (d["dslr"] <= 4)),
        L.rule(f"{code}: DSLR 1-4, placed last time", d, c & (d["dslr"] <= 4) & (d["prev_fin"] <= 3)),
        L.rule(f"{code}: DSLR 5-7", d, c & d["dslr"].between(5, 7)),
        L.rule(f"{code}: placed last time, handicap, back within 14 days", d,
               c & (d["prev_fin"] <= 3) & d["hcap"] & (d["dslr"] <= 14)),
        L.rule(f"{code}: course winner, placed last time", d, c & d["course_winner"] & (d["prev_fin"] <= 3)),
        L.rule(f"{code}: BSP favourite", d, c & (d["bsp_rank"] == 1)),
        L.rule(f"{code}: BSP under 2.0", d, c & (d["bsp"] < 2)),
        L.rule(f"{code}: jockey hot (14d A/E>1.5, 10+ rides)", d, c & (d["j14_n"] >= 10) & (d["j14_ae"] > 1.5)),
        L.rule(f"{code}: trainer AND jockey hot", d, c & (d["j14_n"] >= 10) & (d["j14_ae"] > 1.3)
               & (d["t14_n"] >= 10) & (d["t14_ae"] > 1.3)),
    ]
rows += [
    L.rule("jumps: up 4f+ in trip", d, jumps & (dd >= 4)),
    L.rule("jumps: down 4f+ in trip", d, jumps & (dd <= -4)),
    L.rule("flat/AW: up 2f+ in trip", d, fl & (dd >= 2)),
    L.rule("flat/AW: down 2f+ in trip", d, fl & (dd <= -2)),
    L.rule("flat/AW handicap: 7lb claimer", d, fl & d["hcap"] & (claim == 7)),
    L.rule("jumps handicap: 7lb+ conditional/amateur", d, jumps & d["hcap"] & (claim >= 7)),
    L.rule("chase: first run over fences after hurdling", d, (d["code"] == "chase") & (d["prev_code"] == "hurdle")),
    L.rule("hurdle: back to hurdles from chasing", d, (d["code"] == "hurdle") & (d["prev_code"] == "chase")),
    L.rule("AW: from turf flat", d, (d["code"] == "aw") & (d["prev_code"] == "flat")),
    L.rule("flat: from AW", d, (d["code"] == "flat") & (d["prev_code"] == "aw")),
]
print("\n=== MORE SYSTEMS (train | t_ test) ===")
L.rules_table(rows)
ALL += rows

print(f"\n=== THE SCREEN: {len(ALL)} rules through the owner's gate (PASS, then train-only, by test WAX) ===")
print(L.gate.__doc__)
t = pd.DataFrame(ALL)
t.insert(1, "gate", [L.gate(r) for r in ALL])
t = t[t["gate"] != "-"].sort_values(["gate", "t_WAX"], ascending=[True, False])
with pd.option_context("display.max_colwidth", 90):
    print(L._fmt(t).to_string(index=False) if len(t) else "(no rule passed TRAIN)")
print(f"\n=== THE SCREEN AT THE MORNING PRICES (2018+): {len(ALL)} rules ===")
print(L.gate_am.__doc__)
t = pd.DataFrame(ALL)
t.insert(1, "gate_am", [L.gate_am(r) for r in ALL])
t = t[t["gate_am"] != "-"].sort_values(["gate_am", "t_A/E_am"], ascending=[True, False])
with pd.option_context("display.max_colwidth", 90):
    print(L._fmt(t).to_string(index=False) if len(t) else "(no rule passed TRAIN)")
pd.DataFrame(ALL).assign(gate=[L.gate(r) for r in ALL], gate_am=[L.gate_am(r) for r in ALL]).to_csv(
    "out/strat_screen_rules.csv", index=False)
