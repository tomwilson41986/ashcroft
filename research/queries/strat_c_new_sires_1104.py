"""Strategy study C (4 Nov 2026, read-only): new sires showing elite early indicators.

The owner: "Which new sires are showing elite early indicators and how can we bet ahead of the curve on them."

A NEW sire has his first runner held here on or after 2012-01-01 (the history starts in 2010, so his first crop is
seen). Every figure on a run is the sire's progeny record on EARLIER DAYS only.

  1. The curve: A/E of a new sire's progeny by how many progeny runs he has had. Is the market slow to learn?
  2. Persistence: does a sire's progeny A/E over his first 150 runs carry into his next 600?
  3. Early indicators: which first-150-run measure (A/E, winners to runners, two-year-old strike rate, the market's
     own regard = median BSP) best predicts the later A/E and winners to runners.
  4. The rule, chosen on 2010-21 and reported on 2022-26: back a new sire's progeny while his shrunk prior A/E clears
     a bar; by age, debutants, code.
  5. Who fits now: sires whose first runner is 2023 or later.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
K = 15.0


def norm(s: pd.Series) -> pd.Series:
    return s.fillna("").str.replace(r"\s*\([A-Z]{2,3}\)\s*$", "", regex=True).str.strip().str.lower()


df = L.load()
df["sire"] = norm(df["stallion"])
df = df[df["sire"] != ""].copy()
df["p"] = np.where(df["bsp"] > 1, 1 / df["bsp"], np.nan)
df["has"] = (df["bsp"] > 1).astype(float)
df["pw"] = df["won"] * df["has"]
df["pp"] = df["p"].fillna(0)
hg = df.groupby("horse_name", sort=False)
df["debut"] = (hg.cumcount() == 0) & (df["race_date"] >= "2011-01-01")  # first run held, after the first year
df["first_run_flag"] = (hg.cumcount() == 0).astype(float)
pri = L.prior_cumsum(df, "sire", ["has", "pw", "pp", "first_run_flag"])
df["s_runs"], df["s_w"], df["s_e"], df["s_runners"] = pri["has"], pri["pw"], pri["pp"], pri["first_run_flag"]
df["s_ae"] = (df["s_w"] + K) / (df["s_e"] + K)
# individual winners on earlier days: a horse counts once, from the day after its first win
fw = df[df["won"] == 1].groupby("horse_name")["date"].min().rename("first_win")
df = df.merge(fw, on="horse_name", how="left")
df["_fwflag"] = (df["date"] == df["first_win"]).astype(float) * (df["won"] == 1)
df["_fwflag"] = df["_fwflag"].where(~df.duplicated(["horse_name", "date"]), 0)
df["s_winners"] = L.prior_cumsum(df, "sire", ["_fwflag"])["_fwflag"]
df["s_wr"] = (df["s_winners"] + 2) / (df["s_runners"] + 8)
first = df.groupby("sire")["date"].transform("min")
df["s_first"] = first
df["new"] = first >= "2012-01-01"
age = pd.to_numeric(df["horse_age"], errors="coerce")
df["age"] = age
d = L.priced(df).copy()
n = d[d["new"]].copy()
print("runs", len(d), "new-sire progeny runs", len(n), "new sires", n["sire"].nunique())

# ---- 1. the curve ---------------------------------------------------------------------------------------------
rb = [-1, 24, 99, 249, 499, 999, 1999, 10**7]
rl = ["0-24", "25-99", "100-249", "250-499", "500-999", "1000-1999", "2000+"]
n["sb"] = pd.cut(n["s_runs"], rb, labels=rl)
L.show("NEW sires: progeny A/E by the sire's progeny runs so far", L.tt(n, ["sb"], 200))
L.show("NEW sires: by progeny runs so far, flat/AW two-year-olds only",
       L.tt(n[(n["age"] == 2) & n["code"].isin(["flat", "aw"])], ["sb"], 200))
L.show("NEW sires: debutants only, by code", L.tt(n[n["debut"]], ["sb", "code"], 150))
L.show("ALL sires: debutants by code", L.tt(d[d["debut"]], ["code"], 200))

# ---- 2/3. persistence and early indicators --------------------------------------------------------------------
n["idx"] = n.groupby("sire").cumcount()
n["wname"] = n["horse_name"].where(n["won"] == 1)
n["won2"] = n["won"].where(n["age"] == 2)
early = n[n["idx"] < 150].groupby("sire").agg(
    w1=("won", "sum"), e1=("p", "sum"), runs1=("won", "size"), med_bsp1=("bsp", "median"),
    runners1=("horse_name", "nunique"), winners1=("wname", "nunique"),
    two_sr=("won2", "mean"),
    first_date=("date", "min"))
later = n[(n["idx"] >= 150) & (n["idx"] < 750)].groupby("sire").agg(
    w2=("won", "sum"), e2=("p", "sum"), runs2=("won", "size"),
    runners2=("horse_name", "nunique"), winners2=("wname", "nunique"))
ps = early.join(later, how="inner")
ps = ps[(ps["runs1"] == 150) & (ps["runs2"] == 600)]
ps["ae1"], ps["ae2"] = ps["w1"] / ps["e1"], ps["w2"] / ps["e2"]
ps["wr1"], ps["wr2"] = ps["winners1"] / ps["runners1"], ps["winners2"] / ps["runners2"]
print(f"\n=== Persistence and early indicators: {len(ps)} new sires with 750+ progeny runs ===")
for x in ("ae1", "wr1", "two_sr", "med_bsp1"):
    print(f"spearman {x:9s} -> ae2 {ps[x].corr(ps['ae2'], method='spearman'):+.3f}   "
          f"-> wr2 {ps[x].corr(ps['wr2'], method='spearman'):+.3f}")
ps["q"] = pd.qcut(ps["ae1"], 5, labels=False, duplicates="drop")
print("by quintile of early A/E: later pooled A/E",
      (ps.groupby("q")["w2"].sum() / ps.groupby("q")["e2"].sum()).round(3).to_dict())
ps["qm"] = pd.qcut(ps["med_bsp1"], 5, labels=False, duplicates="drop")
print("by quintile of early median BSP (market regard): later pooled A/E",
      (ps.groupby("qm")["w2"].sum() / ps.groupby("qm")["e2"].sum()).round(3).to_dict())
# the combination: good early A/E that the market still prices long
ps["sleeper"] = (ps["ae1"] > ps["ae1"].median()) & (ps["med_bsp1"] > ps["med_bsp1"].median())
print("early A/E above median AND priced longer than median: later pooled A/E",
      (ps.groupby("sleeper")["w2"].sum() / ps.groupby("sleeper")["e2"].sum()).round(3).to_dict())

# ---- 4. the rule ----------------------------------------------------------------------------------------------
rows = []
for lo, hi in ((30, 300), (50, 600), (100, 1200)):
    for bar in (1.1, 1.2, 1.3):
        m = n["s_runs"].between(lo, hi) & (n["s_ae"] > bar)
        rows.append(L.rule(f"new sire, prior progeny runs {lo}-{hi}, shrunk A/E>{bar}", n, m))
m = n["s_runs"].between(50, 600) & (n["s_ae"] > 1.2)
rows += [
    L.rule("  ... two-year-olds only", n, m & (n["age"] == 2)),
    L.rule("  ... debutants only", n, m & n["debut"]),
    L.rule("  ... flat/AW only", n, m & n["code"].isin(["flat", "aw"])),
    L.rule("  ... jumps only", n, m & n["code"].isin(["hurdle", "chase", "nhflat"])),
    L.rule("  ... priced 5.0-21 at BSP", n, m & n["bsp"].between(5, 21)),
    L.rule("new sire, 50-600 runs, winners/runners > 0.30 (shrunk)", n, n["s_runs"].between(50, 600) & (n["s_wr"] > 0.30)),
    L.rule("new sire, 50-600 runs, A/E>1.2 AND w/r > 0.30", n,
           n["s_runs"].between(50, 600) & (n["s_ae"] > 1.2) & (n["s_wr"] > 0.30)),
    L.rule("new sire, first 0-24 progeny runs (unknown), two-year-old debutants", n,
           (n["s_runs"] <= 24) & n["debut"] & (n["age"] == 2)),
]
print("\n=== RULES on new sires' progeny (train | t_ test) ===")
L.rules_table(rows)

# ---- 5. who fits now ------------------------------------------------------------------------------------------
last = df["date"].max()
cur = df[df["s_first"] >= "2023-01-01"].assign(wname=lambda x: x["horse_name"].where(x["won"] == 1))
now = cur.groupby("sire").agg(first=("date", "min"), runs=("has", "sum"), wins=("pw", "sum"), exp=("pp", "sum"),
                              runners=("horse_name", "nunique"),
                              winners=("wname", "nunique"),
                              two_runs=("age", lambda s: int((s == 2).sum())),
                              med_bsp=("bsp", "median"))
now = now[now["runs"] >= 40]
now["ae"] = now["wins"] / now["exp"]
now["ae_shrunk"] = (now["wins"] + K) / (now["exp"] + K)
now["w/r"] = now["winners"] / now["runners"]
pl = cur.assign(pnl=lambda x: np.where(x["won"] == 1, (x["bsp"] - 1) * (1 - L.COMMISSION), -1.0) * x["has"])
now["pl_bsp"] = pl.groupby("sire")["pnl"].sum().reindex(now.index)
now["fits"] = (now["ae_shrunk"] > 1.2) & (now["runs"] >= 50)
print(f"\n=== New sires (first runner 2023+) with 40+ progeny runs (as at {last.date()}), by shrunk A/E ===")
print(now.sort_values("ae_shrunk", ascending=False).round(3).head(40).to_string())
now.sort_values("ae_shrunk", ascending=False).to_csv("out/strat_new_sires_now.csv")
