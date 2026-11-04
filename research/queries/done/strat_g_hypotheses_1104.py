"""Strategy discovery G (4 Nov 2026, read-only): hypotheses written down before the data was read.

The owner: "run discovery to find other strategies. create hypotheses and test them". Each hypothesis below carries
the reason it might beat the market. Each is scored win and place, at BSP and at the morning price, through the
gates fixed on 4 Nov (strat_lib.gate, gate_am). With ~35 hypotheses x codes tested, a pass at the TRAIN bar
(p < 0.001) is expected by chance about once in a thousand tests per tail; the TEST bar is independent data.

H1  Hard luck last time (comment: hampered, no clear run, short of room, slowly away, stumbled, carried wide):
    the result understated the horse, the form book is read at face value.
H2  Finished strongly last time ("ran on", "kept on", "never nearer", "staying on") and up in trip today.
H3  Jockey upgrade: today's jockey has a prior strike rate 5+ points above the last run's jockey.
H4  First run for a new trainer.
H5  Well handicapped: running off an official rating below the one it last won off (handicaps).
H6  Falling mark: OR down 5+ over its last two runs (handicaps): the handicapper relents, the market sees poor form.
H7  Weight drop: 7lb+ less than last time in a handicap.
H8  Fell, unseated or brought down last time (jumps): no form shown, not the horse's fault.
H9  Pulled up last time (jumps): the market may forgive too little or too much.
H10 Course and distance winner; H11 winner on today's going; H12 2+ wins at the course.
H13 First run of the season after 120+ days, jumps, from a trainer whose earlier first-time-out runners beat the BSP.
H14 Trainer at the course: the trainer's earlier runners at this track beat the BSP (shrunk A/E > 1.2, 30+ runs).
H15 Trainer-jockey pair: their earlier rides together beat the BSP (shrunk A/E > 1.2, 20+ rides).
H16 Morning favourite / morning price bands (a price known before the off): the favourite-longshot bias at 08:00.
H17 Morning money: the horse with the most matched in the morning in its race.
H18 Top-rated on OR in a handicap (top weight) and bottom of the handicap.
H19 Three-year-olds against older horses in handicaps, Jul-Oct (weight for age against improvement).
H20 Small fields (<= 6) and big fields (16+).
H21 Beaten under 1 length last time, in a handicap, same OR or lower today.
H22 Headgear off (worn last time, not today).
H23 Fillies and mares in mixed-sex races (flat); mares in mixed jumps races.
H24 Evening racing (off at 17:30 or later) and Sunday cards.
H25 Lightly raced: second career run (the debut taught it), flat/AW.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 280); pd.set_option("display.max_columns", 50)
K = 15.0

df = L.load()
h = df.groupby("horse_name", sort=False)
for k, src in (("prev_date", "date"), ("prev_fin", "fin"), ("prev_won", "won"), ("prev_or", "official_rating"),
               ("prev_jockey", "jockey_name"), ("prev_trainer", "trainer"), ("prev_comment", "comment"),
               ("prev_dist", "dist_furlongs"), ("prev_pounds", "pounds"), ("prev_hg", "headgear"),
               ("prev_place", "place"), ("prev_dstbt", "total_dst_bt"), ("prev_hcap", "hcap"), ("prev_code", "code")):
    df[k] = h[src].shift(1)
df["prev2_or"] = h["official_rating"].shift(2)
df["runs_held"] = h.cumcount()
df["dslr"] = df["days_since_lr"].where(df["days_since_lr"] > 0, (df["date"] - df["prev_date"]).dt.days)
# the OR it last won off, on earlier runs
wor = df["official_rating"].where(df["won"] == 1)
df["last_win_or"] = wor.groupby(df["horse_name"]).shift(1).groupby(df["horse_name"]).ffill()
# earlier wins: at the course, at the course and distance, on the going
df["_w"] = df["won"]
df["dist_b"] = (df["dist_furlongs"] * 2).round() / 2
for nm, keys in (("crs_w", ["horse_name", "track"]), ("cd_w", ["horse_name", "track", "dist_b"]),
                 ("going_w", ["horse_name", "going_description"])):
    df[nm] = df.groupby(keys, sort=False)["_w"].cumsum() - df["_w"]
# jockey quality: shrunk prior strike rate (earlier days)
df["_r"] = 1.0
pj = L.prior_cumsum(df, "jockey_name", ["_r", "_w"])
df["jq"] = (pj["_w"] + 2) / (pj["_r"] + 20)
df["prev_jq"] = df.groupby("horse_name", sort=False)["jq"].shift(1)
# prior A/E by trainer at the track, by trainer-jockey pair, trainer first-time-out (jumps, 120+ days)
df["p"] = np.where(df["bsp"] > 1, 1 / df["bsp"], 0.0)
df["has"] = (df["bsp"] > 1).astype(float)
df["pw"] = df["won"] * df["has"]
df["k_tt"] = df["trainer"].astype(str) + "|" + df["track"].astype(str)
df["k_tj"] = df["trainer"].astype(str) + "|" + df["jockey_name"].astype(str)
for k, nm in (("k_tt", "tt"), ("k_tj", "tj")):
    pr = L.prior_cumsum(df, k, ["has", "pw", "p"])
    df[nm + "_n"], df[nm + "_ae"] = pr["has"], (pr["pw"] + K) / (pr["p"] + K)
df["fto"] = (df["dslr"] >= 120) & df["code"].isin(["hurdle", "chase"])
df["k_tf"] = df["trainer"].astype(str) + "|fto"
ff = df[df["fto"]]
pr = L.prior_cumsum(ff, "k_tf", ["has", "pw", "p"])
df["tf_n"] = pr["has"].reindex(df.index)
df["tf_ae"] = ((pr["pw"] + K) / (pr["p"] + K)).reindex(df.index)

d = L.priced(df).copy()
am = d["morningwap"].where(d["morning_vol"] >= L.MIN_AM_VOL)
d["am"] = am
d["am_rank"] = d.groupby("rk")["am"].rank(method="min")
d["vol_rank"] = d.groupby("rk")["morning_vol"].rank(method="min", ascending=False)
d["or_rank"] = d.groupby("rk")["official_rating"].rank(method="min", ascending=False)
d["or_rank_low"] = d.groupby("rk")["official_rating"].rank(method="min", ascending=True)
pc = d["prev_comment"].fillna("").str.lower()
hour = pd.to_numeric(d["race_time"].astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce")
hour = np.where(hour < 11, hour + 12, hour)  # cards print 1.30 for 13:30
d["hour"] = hour
dow = d["date"].dt.dayofweek
sex = d["horse_sex"].fillna("").str.lower()
fem = sex.str.contains("filly|mare")
mixed = fem.astype(float).groupby(d["rk"]).transform("mean")
d["mixed_sex"] = mixed.between(0.01, 0.99)
fl = d["code"].isin(["flat", "aw"])
jumps = d["code"].isin(["hurdle", "chase"])
main = d["code"].isin(["flat", "aw", "hurdle", "chase"])
H = []


def add(name, mask, codes=("flat", "aw", "hurdle", "chase")):
    H.append(L.rule(name + " [all codes]", d, mask & main))
    for c in codes:
        m = mask & (d["code"] == c)
        if m.sum() >= 1500:
            H.append(L.rule(f"{name} [{c}]", d, m))


add("H1 hard luck last time", pc.str.contains(r"hampered|no clear run|not clear run|denied a clear|short of room|"
                                             r"slowly away|missed the break|stumbled|carried wide|badly hmpd|squeezed"))
add("H2 finished strongly LTO, up 1f+ in trip", pc.str.contains(r"ran on|kept on|never nearer|staying on|stayed on")
    & (d["dist_furlongs"] - d["prev_dist"] >= 1))
add("H3 jockey upgrade (prior strike rate +5 pts)", (d["jockey_name"] != d["prev_jockey"]) & (d["jq"] - d["prev_jq"] >= 0.05))
add("H4 first run for a new trainer", d["prev_trainer"].notna() & (d["trainer"] != d["prev_trainer"]))
add("H5 handicap, OR below its last winning OR", d["hcap"] & (d["official_rating"] < d["last_win_or"]))
add("H5b handicap, OR 5+ below its last winning OR", d["hcap"] & (d["official_rating"] <= d["last_win_or"] - 5))
add("H6 handicap, OR down 5+ over last two runs", d["hcap"] & (d["official_rating"] <= d["prev2_or"] - 5))
add("H7 handicap, 7lb+ less weight than LTO", d["hcap"] & d["prev_hcap"].fillna(False).astype(bool)
    & (d["pounds"] - d["prev_pounds"] <= -7))
add("H8 fell/unseated/brought down LTO", d["prev_place"].fillna("").str.upper().isin(["F", "UR", "BD", "SU"]),
    codes=("hurdle", "chase"))
add("H9 pulled up LTO", d["prev_place"].fillna("").str.upper().eq("PU"), codes=("hurdle", "chase"))
add("H10 course and distance winner", d["cd_w"] > 0)
add("H11 winner on today's going", d["going_w"] > 0)
add("H12 2+ wins at the course", d["crs_w"] >= 2)
add("H13 jumps FTO 120+ days, trainer FTO A/E>1.2 (30+)", d["fto"] & (d["tf_n"] >= 30) & (d["tf_ae"] > 1.2),
    codes=("hurdle", "chase"))
add("H14 trainer at the course A/E>1.2 (30+ runs)", (d["tt_n"] >= 30) & (d["tt_ae"] > 1.2))
add("H15 trainer-jockey pair A/E>1.2 (20+ rides)", (d["tj_n"] >= 20) & (d["tj_ae"] > 1.2))
add("H16 morning favourite", d["am_rank"] == 1)
for lo, hi in ((1, 2.5), (2.5, 4), (4, 7), (7, 12), (12, 21), (21, 51), (51, 1001)):
    add(f"H16b morning price {lo}-{hi}", (d["am"] >= lo) & (d["am"] < hi))
add("H17 most matched in the morning in its race", (d["vol_rank"] == 1) & (d["morning_vol"] >= 200))
add("H18 handicap top-rated on OR", d["hcap"] & (d["or_rank"] == 1))
add("H18b handicap bottom of the weights on OR", d["hcap"] & (d["or_rank_low"] == 1))
add("H19 3yo in an all-aged handicap, Jul-Oct", d["hcap"] & (d["horse_age"] == 3) & d["date"].dt.month.between(7, 10)
    & (d.groupby("rk")["horse_age"].transform("max") >= 4), codes=("flat", "aw"))
add("H20 small field (<=6)", d["field"] <= 6)
add("H20b big field (16+)", d["field"] >= 16)
add("H21 handicap, beaten <1L LTO, OR same or lower", d["hcap"] & (d["prev_dstbt"] > 0) & (d["prev_dstbt"] < 1)
    & (d["official_rating"] <= d["prev_or"]))
add("H22 headgear off", (d["prev_hg"].fillna("").str.strip() != "") & (d["headgear"].fillna("").str.strip() == ""))
add("H23 female in a mixed-sex race", fem & d["mixed_sex"])
add("H24 evening racing (17:30 or later)", d["hour"] >= 17.5)
add("H24b Sunday", dow == 6)
add("H25 second career run", (d["runs_held"] == 1) & (d["career_runs"].fillna(1) <= 2), codes=("flat", "aw"))

print(L.gate.__doc__); print(L.gate_am.__doc__)
t = L.rules_table(H)
t.to_csv("out/strat_hypotheses_rules.csv", index=False)
print("\n=== PASSES ===")
print(L._fmt(t[(t["gate"] == "PASS") | (t["gate_am"] == "PASS")]).to_string(index=False))
print("\n=== TRAIN ONLY (failed TEST) ===")
print(t[(t["gate"] == "train only") | (t["gate_am"] == "train only")]["rule"].to_string(index=False))
