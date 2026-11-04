"""Strategy discovery I (4 Nov 2026, read-only): predict the morning-to-off steam, back the predicted steamers early.

What studies A-H found: the BSP is efficient against everything held before the race, and the edges sit in the
MORNING market -- horses whose price shortens by the off (the quick return; Ghiani; New Bay). So:

H26  A model fitted on 2018-21, from facts known at 08:00 only, predicts y = log(morning price / BSP) (positive =
     the horse steamed). On 2022-26, the horses it rates top, backed at the morning price, win and place, beat the
     morning prices (A/E_am > 1) by more than commission.
H27  "Gamble yards": a trainer whose earlier runners steamed (mean y on earlier days, shrunk) sends out runners that
     steam again, and backing them early pays.

Features (all known at 08:00): code, handicap, class, field size, age, OR and its rank in the race, the jockey's
allowance, days since the last run, last run's finish / market rank / BSP / steam, runs held, trainer and jockey
change, headgear on/off, the morning price, its rank in the race, the morning volume and its share of the race, the
race's morning overround, month, hour, and the trainer's and jockey's earlier record: strike rate, A/E against BSP and
mean steam (all from earlier days). LightGBM; nothing from the result or the BSP of today's race enters.
"""

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
K = 15.0

df = L.load()
df = df[df["code"].isin(["flat", "aw", "hurdle", "chase"])].copy()
df["am"] = df["morningwap"].where((df["morning_vol"] >= L.MIN_AM_VOL) & (df["morningwap"] > 1))
df["steam"] = np.log(df["am"] / df["bsp"]).where((df["bsp"] > 1) & df["am"].notna())
h = df.groupby("horse_name", sort=False)
for k, src in (("prev_date", "date"), ("prev_fin", "fin"), ("prev_rank", "bsp_rank"), ("prev_bsp", "bsp"),
               ("prev_steam", "steam"), ("prev_trainer", "trainer"), ("prev_jockey", "jockey_name"), ("prev_hg", "headgear")):
    df[k] = h[src].shift(1)
df["runs_held"] = h.cumcount()
df["dslr"] = df["days_since_lr"].where(df["days_since_lr"] > 0, (df["date"] - df["prev_date"]).dt.days)
df["p"] = np.where(df["bsp"] > 1, 1 / df["bsp"], 0.0)
df["has"] = (df["bsp"] > 1).astype(float)
df["pw"] = df["won"] * df["has"]
df["s_has"] = df["steam"].notna().astype(float)
df["s_sum"] = df["steam"].fillna(0)
for key, nm in (("trainer", "tr"), ("jockey_name", "jk")):
    pr = L.prior_cumsum(df, key, ["has", "pw", "p", "s_has", "s_sum"])
    df[f"{nm}_n"] = pr["has"]
    df[f"{nm}_sr"] = (pr["pw"] + 2) / (pr["has"] + 20)
    df[f"{nm}_ae"] = (pr["pw"] + K) / (pr["p"] + K)
    df[f"{nm}_steam"] = pr["s_sum"] / (pr["s_has"] + 30)   # shrunk to 0 (no steam) on a small record
    df[f"{nm}_steam_n"] = pr["s_has"]
d = L.priced(df).copy()
d = d[d["race_date"] >= "2018-01-01"]
d["am_rank"] = d.groupby("rk")["am"].rank(method="min")
d["vol_share"] = d["morning_vol"] / d.groupby("rk")["morning_vol"].transform("sum")
d["am_over"] = (1 / d["am"]).groupby(d["rk"]).transform("sum")
d["or_rank"] = d.groupby("rk")["official_rating"].rank(method="min", ascending=False)
d["cls"] = pd.to_numeric(d["race_class"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
hr = pd.to_numeric(d["race_time"].astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce")
d["hour"] = np.where(hr < 11, hr + 12, hr)
d["month"] = d["date"].dt.month
d["code_i"] = d["code"].map({"flat": 0, "aw": 1, "hurdle": 2, "chase": 3})
d["trainer_change"] = (d["prev_trainer"].notna() & (d["trainer"] != d["prev_trainer"])).astype(int)
d["jockey_change"] = (d["prev_jockey"].notna() & (d["jockey_name"] != d["prev_jockey"])).astype(int)
hg, phg = d["headgear"].fillna("").str.strip(), d["prev_hg"].fillna("").str.strip()
d["hg_on"] = ((hg != "") & (phg == "")).astype(int)
d["hg_off"] = ((hg == "") & (phg != "")).astype(int)
d["log_am"] = np.log(d["am"])
d["log_vol"] = np.log1p(d["morning_vol"])
F = ["code_i", "hcap", "cls", "field", "horse_age", "official_rating", "or_rank", "jockeys_claim", "dslr", "prev_fin",
     "prev_rank", "prev_bsp", "prev_steam", "runs_held", "trainer_change", "jockey_change", "hg_on", "hg_off", "log_am",
     "am_rank", "log_vol", "vol_share", "am_over", "month", "hour", "tr_n", "tr_sr", "tr_ae", "tr_steam", "jk_n",
     "jk_sr", "jk_ae", "jk_steam"]
d["hcap"] = d["hcap"].astype(int)
m = d[d["steam"].notna()].copy()
tr, te = L.split(m)
print(f"rows with a morning price: train 2018-21 {len(tr)}, test 2022-26 {len(te)}")
model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=400,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
# fit on 2018-20, check on 2021, then refit on 2018-21 for the test
early, late = tr[tr["race_date"] < "2021-01-01"], tr[tr["race_date"] >= "2021-01-01"]
model.fit(early[F], early["steam"])
r_late = np.corrcoef(model.predict(late[F]), late["steam"])[0, 1]
model.fit(tr[F], tr["steam"])
te = te.assign(pred=model.predict(te[F]))
tr = tr.assign(pred=model.predict(tr[F]))  # in-sample, for the thresholds only
print(f"correlation of predicted with actual steam: 2021 (fit 2018-20) {r_late:.3f} | test 2022-26 "
      f"{np.corrcoef(te['pred'], te['steam'])[0, 1]:.3f}")
imp = pd.Series(model.booster_.feature_importance("gain"), index=F).sort_values(ascending=False)
print("feature gain (top 15):", (imp / imp.sum()).round(3).head(15).to_dict())

te["dec"] = pd.qcut(te["pred"], 10, labels=False, duplicates="drop")
rows = []
for k_, g in te.groupby("dec"):
    s = L.score(g)
    rows.append({"decile (0 = least steam)": k_, "n": s["n"], "pred": g["pred"].mean(), "actual_steam": g["steam"].mean(),
                 "A/E_am": s.get("A/E_am"), "p_am": s.get("p_am"), "A/E_pl_am": s.get("A/E_pl_am"),
                 "roi_am%": s.get("roi_am%"), "roi_both_am%": s.get("roi_both_am%"), "tw_both_am_roi%": s.get("tw_both_am_roi%"),
                 "A/E (bsp)": s.get("A/E"), "roi_both% (bsp)": s.get("roi_both%")})
L.show("TEST 2022-26 by decile of predicted steam (backed at the morning price unless marked bsp)", pd.DataFrame(rows))

# rules: thresholds fixed from the TRAIN distribution of predictions
d2 = pd.concat([tr, te])
R = []
for q in (0.90, 0.95, 0.98):
    thr = tr["pred"].quantile(q)
    R.append(L.rule(f"H26 predicted steam in the top {100 * (1 - q):.0f}% (train threshold {thr:.3f})", d2, d2["pred"] >= thr))
    for c in ("flat", "aw", "hurdle", "chase"):
        R.append(L.rule(f"   ... {c}", d2, (d2["pred"] >= thr) & (d2["code"] == c)))
R.append(L.rule("H27 gamble yard: trainer mean steam > +0.05 on 100+ earlier runners", d2,
                (d2["tr_steam"] > 0.05) & (d2["tr_steam_n"] >= 100)))
R.append(L.rule("H27b gamble yard > +0.08", d2, (d2["tr_steam"] > 0.08) & (d2["tr_steam_n"] >= 100)))
R.append(L.rule("H27c horse steamed 15%+ last time", d2, d2["prev_steam"] > np.log(1.15)))
print("\nNB: TRAIN columns here are in-sample for the model (fitted on 2018-21); only TEST is out of sample.")
print(L.gate_am.__doc__)
t = L.rules_table(R)
t.to_csv("out/strat_steamers_rules.csv", index=False)
