"""Strategy discovery H (4 Nov 2026, read-only): a data-driven scan for underbet profiles.

For each dimension (course, trainer, jockey, sire, damsire, month, class, field size, distance, going, age, hour,
weekday, course x distance x draw third, trainer x code, jockey x code, sire x code), every level is scored on TRAIN
(2010-21): win A/E against BSP, place A/E against the place BSP, and the chi-square p of each.

  SELECTED = levels with 300+ runners, win A/E > 1 with p < 0.001, and place A/E > 1 (the owner's bar per level).
  The selected levels are pooled into one portfolio and scored on TEST (2022 to Nov 2026), win and place, with the
  same gate. The OVERBET levels (win A/E < 1, p < 0.001) are pooled too: a list to avoid.
  PERSISTENCE: across levels with 300+ runners in both periods, the Spearman correlation of TRAIN A/E with TEST A/E.
  Near 0 = the market corrects; a profile edge would show as a clearly positive figure.

The same is done against the MORNING price, choosing on 2018-21. A level is a pre-race fact in every dimension.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2 as C2

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)

df = L.load()
d = L.priced(df).copy()
d = d[d["code"].isin(["flat", "aw", "hurdle", "chase"])]
d["dist_b"] = pd.cut(d["dist_furlongs"], [0, 6.5, 8.5, 11, 14, 18, 22, 40], labels=["5-6f", "7-8f", "9-11f", "12-14f", "15-18f", "19-22f", "23f+"]).astype(str)
d["field_b"] = pd.cut(d["field"], [0, 5, 7, 10, 13, 16, 40], labels=["<=5", "6-7", "8-10", "11-13", "14-16", "17+"]).astype(str)
d["month"] = d["date"].dt.month
d["dow"] = d["date"].dt.dayofweek
hr = pd.to_numeric(d["race_time"].astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce")
d["hour"] = np.where(hr < 11, hr + 12, hr)
fl = d["code"].isin(["flat", "aw"]) & (d["field"] >= 8) & (d["stall"] > 0)
srank = d["stall"].where(fl).groupby(d["rk"]).rank(pct=True)
d["draw3"] = pd.cut(srank, [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"]).astype(str)
d["k_draw"] = np.where(fl, d["track"].astype(str) + "|" + d["dist_b"] + "|" + d["draw3"], None)
d["k_trainer_code"] = d["trainer"].astype(str) + "|" + d["code"]
d["k_jockey_code"] = d["jockey_name"].astype(str) + "|" + d["code"]
d["sire"] = d["stallion"].fillna("").str.replace(r"\s*\([A-Z]{2,3}\)\s*$", "", regex=True)
d["k_sire_code"] = d["sire"] + "|" + d["code"]
d["p"] = 1 / d["bsp"]
d["pp"] = np.where(d["pbsp"] > 1, 1 / d["pbsp"], np.nan)
d["plw"] = np.where(d["pbsp"] > 1, d["placed"], np.nan)
d["am"] = d["morningwap"].where(d["morning_vol"] >= L.MIN_AM_VOL)
d["pam"] = 1 / d["am"]
d["wam"] = np.where(d["am"] > 1, d["won"], np.nan)
dims = ["track", "trainer", "jockey_name", "sire", "dam_stallion", "month", "race_class", "field_b", "dist_b",
        "going_description", "horse_age", "hour", "dow", "k_draw", "k_trainer_code", "k_jockey_code", "k_sire_code"]


def agg(x: pd.DataFrame, key: str, am: bool = False) -> pd.DataFrame:
    if am:
        x = x[x["am"] > 1]
        g = x.groupby(key)
        a = pd.DataFrame({"n": g.size(), "W": g["won"].sum(), "E": g["pam"].sum()})
    else:
        g = x.groupby(key)
        a = pd.DataFrame({"n": g.size(), "W": g["won"].sum(), "E": g["p"].sum(),
                          "P": g["plw"].sum(), "Ep": g["pp"].sum(), "np": g["pp"].count()})
    a["ae"] = a["W"] / a["E"]
    x2 = (a["W"] - a["E"]) ** 2 / a["E"] + (a["W"] - a["E"]) ** 2 / (a["n"] - a["E"])
    a["p"] = C2.sf(x2, 1)
    if not am:
        a["ae_pl"] = a["P"] / a["Ep"]
    return a


def run(tag: str, train: pd.DataFrame, test: pd.DataFrame, am: bool) -> list[dict]:
    rows = []
    for k in dims:
        a = agg(train[train[k].notna()], k, am)
        b = agg(test[test[k].notna()], k, am)
        both = a[a["n"] >= 300].join(b[b["n"] >= 300], rsuffix="_t", how="inner")
        rho = both["ae"].corr(both["ae_t"], method="spearman") if len(both) >= 10 else np.nan
        sel = a[(a["n"] >= 300) & (a["ae"] > 1) & (a["p"] < 0.001)]
        if not am:
            sel = sel[sel["ae_pl"] > 1]
        bad = a[(a["n"] >= 300) & (a["ae"] < 1) & (a["p"] < 0.001)]
        r = {"scan": tag, "dimension": k, "levels_300+": int((a["n"] >= 300).sum()), "persistence_rho": rho,
             "selected": len(sel), "overbet": len(bad)}
        for nm, lv in (("sel", sel.index), ("bad", bad.index)):
            tt = test[test[k].isin(lv)]
            s = L.score(tt)
            key = "A/E_am" if am else "A/E"
            r[f"{nm}_test_n"] = s.get("n_am" if am else "n", 0)
            r[f"{nm}_test_A/E"] = s.get(key, np.nan)
            r[f"{nm}_test_p"] = s.get("p_am" if am else "p", np.nan)
            r[f"{nm}_test_A/E_pl"] = s.get("A/E_pl_am" if am else "A/E_pl", np.nan)
            r[f"{nm}_test_roi_both%"] = s.get("roi_both_am%" if am else "roi_both%", np.nan)
        r["sel_examples"] = ", ".join(map(str, sel.sort_values("ae", ascending=False).index[:6]))
        rows.append(r)
        if len(sel):
            top = sel.sort_values("p").head(15).join(b, rsuffix="_t")
            cols = ["n", "ae", "p"] + (["ae_pl"] if not am else []) + ["n_t", "ae_t", "p_t"] + (["ae_pl_t"] if not am else [])
            print(f"\n--- {tag} | {k}: selected on TRAIN, as they read on TEST (top 15 by train p) ---")
            print(top[cols].round(4).to_string())
    return rows


a, b = L.split(d)
R = run("BSP (train 2010-21 -> test 2022-26)", a, b, am=False)
R += run("MORNING (train 2018-21 -> test 2022-26)", a[a["race_date"] >= "2018-01-01"], b, am=True)
t = pd.DataFrame(R)
print("\n=== THE SCAN: per dimension, the persistence of A/E and the TEST score of the TRAIN-selected portfolio ===")
with pd.option_context("display.max_colwidth", 70):
    print(t.round(4).to_string(index=False))
t.to_csv("out/strat_profile_scan.csv", index=False)
