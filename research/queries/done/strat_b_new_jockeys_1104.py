"""Strategy study B (4 Nov 2026, read-only): new jockeys ahead of the market.

The owner: "Which 'new' jockeys seem to be ahead of the market - profile based on historic jockey curves and which
ones are fitting."

A NEW jockey has a first ride held here on or after 2012-01-01 (the history starts in 2010, so the start of the
career is seen). Every figure on a ride is that jockey's record on EARLIER DAYS only.

  1. The curve: A/E against the BSP by career ride number, by allowance. Does the market misprice the apprentice
     at some stage of the curve?
  2. Persistence: does an early A/E carry into the next block of rides? If not, there is nothing to bet.
  3. The elite profile: what the jockeys who went on to 2,000+ rides looked like at ride 200, against the rest.
  4. The rule, chosen on 2010-21 and reported on 2022-26: back a new jockey's rides while his shrunk prior A/E
     clears a bar.
  5. Who fits now: jockeys whose first ride is 2023 or later, with their record and the rule's verdict today.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
K = 15.0  # shrinkage of an A/E: (W + K) / (E + K)

df = L.load()
df = df[df["jockey_name"].fillna("").str.strip() != ""].copy()
df["p"] = np.where(df["bsp"] > 1, 1 / df["bsp"], np.nan)
df["has"] = (df["bsp"] > 1).astype(float)
df["pw"] = df["won"] * df["has"]
df["pp"] = df["p"].fillna(0)
df["hasp"] = (df["pbsp"] > 1).astype(float)
df["plw"] = df["placed"] * df["hasp"]
df["plp"] = np.where(df["pbsp"] > 1, 1 / df["pbsp"], 0.0)
pri = L.prior_cumsum(df, "jockey_name", ["has", "pw", "pp"])
df["j_rides"], df["j_w"], df["j_e"] = pri["has"], pri["pw"], pri["pp"]
df["j_ae"] = (df["j_w"] + K) / (df["j_e"] + K)
first = df.groupby("jockey_name")["date"].transform("min")
df["j_first"] = first
df["new"] = first >= "2012-01-01"
df["career_days"] = (df["date"] - first).dt.days
df["claim"] = df["jockeys_claim"].fillna(0).astype(int)
d = L.priced(df).copy()
n = d[d["new"]].copy()
print("rides", len(d), "new-jockey rides", len(n), "new jockeys", n["jockey_name"].nunique())

# ---- 1. the curve -------------------------------------------------------------------------------------------
rb = [-1, 49, 99, 199, 399, 799, 1599, 3199, 10**6]
rl = ["0-49", "50-99", "100-199", "200-399", "400-799", "800-1599", "1600-3199", "3200+"]
n["ride_b"] = pd.cut(n["j_rides"], rb, labels=rl)
L.show("NEW jockeys: A/E by career ride number", L.tt(n, ["ride_b"], 200))
L.show("NEW jockeys: by ride number and today's allowance", L.tt(n, ["ride_b", "claim"], 300))
L.show("NEW jockeys: by ride number and code", L.tt(n, ["ride_b", "code"], 300))
L.show("ALL jockeys: by allowance", L.tt(d, ["claim"], 300))

# ---- 2. persistence: rides 1-200 against rides 201-500 -------------------------------------------------------
n2 = n.copy()
n2["idx"] = n2.groupby("jockey_name").cumcount()
a = n2[n2["idx"] < 200].groupby("jockey_name").agg(w1=("won", "sum"), e1=("p", "sum"), n1=("won", "size"),
                                                  bsp1=("bsp", "median"))
b = n2[(n2["idx"] >= 200) & (n2["idx"] < 500)].groupby("jockey_name").agg(w2=("won", "sum"), e2=("p", "sum"),
                                                                         n2=("won", "size"))
pb = a.join(b, how="inner")
pb = pb[(pb["n1"] == 200) & (pb["n2"] == 300)]
pb["ae1"], pb["ae2"] = pb["w1"] / pb["e1"], pb["w2"] / pb["e2"]
print(f"\n=== Persistence: {len(pb)} new jockeys with 500+ rides ===")
print("spearman A/E(1-200) vs A/E(201-500):", round(pb["ae1"].corr(pb["ae2"], method="spearman"), 3))
pb["q"] = pd.qcut(pb["ae1"], 5, labels=False, duplicates="drop")
print(pb.groupby("q").agg(jockeys=("ae1", "size"), ae1=("ae1", "median"),
                          ae2_pooled=("w2", "sum")).assign(
    ae2_pooled=lambda x: (pb.groupby("q")["w2"].sum() / pb.groupby("q")["e2"].sum()).round(3)).round(3).to_string())

# ---- 3. the elite profile at ride 200 -----------------------------------------------------------------------
tot = n.groupby("jockey_name").size()
cut = n2[n2["idx"] < 200].groupby("jockey_name").agg(
    days_to_200=("career_days", "max"), sr=("won", "mean"), w=("won", "sum"), e=("p", "sum"),
    med_bsp=("bsp", "median"), claim0=("claim", "first"), jumps=("code", lambda s: s.isin(["hurdle", "chase", "nhflat"]).mean()))
cut = cut[n2[n2["idx"] < 200].groupby("jockey_name").size() == 200]
cut["ae"] = cut["w"] / cut["e"]
done = n.groupby("jockey_name")["j_first"].first() <= "2019-12-31"  # time enough to reach 2,000
cut = cut[done.reindex(cut.index).fillna(False)]
cut["elite"] = tot.reindex(cut.index) >= 2000
print(f"\n=== Profile at ride 200 (first ride 2012-19, n={len(cut)}): went on to 2,000+ rides vs not ===")
print(cut.groupby("elite")[["days_to_200", "sr", "ae", "med_bsp", "jumps"]].median().round(3).to_string())
print(cut.groupby("elite").size().to_string())
for col in ("days_to_200", "ae", "sr", "med_bsp"):
    cut[col + "_q"] = pd.qcut(cut[col], 4, labels=False, duplicates="drop")
    print(f"share going on to 2,000+ by quartile of {col}:",
          cut.groupby(col + "_q")["elite"].mean().round(3).to_dict())

# ---- 4. the rule ----------------------------------------------------------------------------------------------
rows = []
for lo, hi in ((50, 400), (100, 800), (200, 1500)):
    for bar in (1.1, 1.2, 1.3):
        m = n["j_rides"].between(lo, hi) & (n["j_ae"] > bar)
        rows.append(L.rule(f"new jockey, prior rides {lo}-{hi}, shrunk prior A/E>{bar}", n, m))
        rows.append(L.rule(f"  ... and riding at BSP<=10", n, m & (n["bsp"] <= 10)))
# ride volume: fast starters (rides in the last 60 days on earlier days)
jd = df.groupby(["jockey_name", "date"]).size().rename("r").reset_index().sort_values(["jockey_name", "date"])
r60 = jd.set_index("date").groupby("jockey_name")["r"].rolling("60D", closed="left").sum().reset_index()
r60.columns = ["jockey_name", "date", "rides60"]
n = n.merge(r60, on=["jockey_name", "date"], how="left")
for bar in (1.1, 1.2):
    m = n["j_rides"].between(100, 800) & (n["j_ae"] > bar) & (n["rides60"] >= 60)
    rows.append(L.rule(f"new jockey 100-800 rides, A/E>{bar}, busy (60+ rides in 60 days)", n, m))
    m = n["j_rides"].between(100, 800) & (n["j_ae"] > bar) & n["claim"].isin([3, 5, 7])
    rows.append(L.rule(f"new jockey 100-800 rides, A/E>{bar}, still claiming", n, m))
print("\n=== RULES on new jockeys (train | t_ test) ===")
L.rules_table(rows)

# ---- 5. who fits now ------------------------------------------------------------------------------------------
last = df["date"].max()
now = df[df["j_first"] >= "2023-01-01"].groupby("jockey_name").agg(
    first=("date", "min"), last=("date", "max"), rides=("has", "sum"), wins=("pw", "sum"), exp=("pp", "sum"), places=("plw", "sum"), pexp=("plp", "sum"),
    claim_now=("claim", "last"), med_bsp=("bsp", "median"),
    jumps=("code", lambda s: s.isin(["hurdle", "chase", "nhflat"]).mean()))
recent = df[df["date"] > last - pd.Timedelta(days=60)].groupby("jockey_name").size()
now["rides_60d"] = recent.reindex(now.index).fillna(0).astype(int)
now = now[now["rides"] >= 75]
now["ae"] = now["wins"] / now["exp"]
now["ae_shrunk"] = (now["wins"] + K) / (now["exp"] + K)
now["WAX"] = now["wins"] - now["exp"]
now["ae_place"] = now["places"] / now["pexp"]
now["sr%"] = 100 * now["wins"] / now["rides"]
# level-stakes P&L at BSP across his career
pl = df[df["j_first"] >= "2023-01-01"].assign(
    pnl=lambda x: np.where(x["won"] == 1, (x["bsp"] - 1) * (1 - L.COMMISSION), -1.0) * x["has"])
now["pl_bsp"] = pl.groupby("jockey_name")["pnl"].sum().reindex(now.index)
now["fits"] = (now["ae_shrunk"] > 1.2) & (now["ae_place"] > 1) & (now["rides"] >= 100)
print(f"\n=== New jockeys since 2023 with 75+ rides (as at {last.date()}), by shrunk A/E ===")
print(now.sort_values("ae_shrunk", ascending=False).round(3).head(40).to_string())
now.sort_values("ae_shrunk", ascending=False).round(4).to_csv("out/strat_new_jockeys_now.csv")
