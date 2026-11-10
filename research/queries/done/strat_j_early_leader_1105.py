"""Strategy J: back the projected early leader (the owner, 5 Nov 2026: "How about if we just backed our predicted front
runners / leader based on historic run style / early position + sectionals. Try and model early leader. And see how
our projected early leader would do if we bet it.").

1. The early position of every past run, from its comment (model/race_shape.py: the comment's first positional phrase;
   led = 'made all', 'led', 'soon led', a lead taken in the first furlong or two; lead class also counts 'disputed').
2. The leader model: LightGBM on 'led' from pre-race facts only -- the horse's projected run style (race_shape, from its
   EARLIER runs), its last run's style, how often it has led, the field's projected styles (race shape), the draw, the
   trip, the code, the jockey's and trainer's earlier lead rates, the horse's earlier sectional early position
   (data/external/sectionals.csv.gz, Flat/AW Jun 2024 - Jun 2025 and from Apr 2026) and the race's ratings.
   Fitted twice, so every scored race is out of sample: on 2010-15 predicting 2016-21 (the rules' TRAIN), and on
   2010-21 predicting 2022 on (TEST). The projected leader is each race's highest predicted chance.
3. How well it finds the leader (top-1 hit rate against the plain run-style projection), and what backing the projected
   leader returns: win and place A/E against BSP with chi-square, at the morning price, by code, trip, field, confidence
   and race shape; the owner's gates (strat_lib.gate, gate_am). And the prize: what the ACTUAL leaders (hindsight) did.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402
from model import race_shape as RS  # noqa: E402
from model.blocks.sectionals import norm_name  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 250)
df = L.load()
df = df[df["code"].isin(["flat", "aw", "hurdle", "chase"])].copy()
print(f"rows {len(df):,} ({time.time()-t0:.0f}s); dates {df.race_date.min()} .. {df.race_date.max()}")

# ---- 1. run styles and the projection (lag-safe) ------------------------------------------------------------------
df = RS.add_run_styles(df)
df = RS.add_style_projection(df)
df = RS.add_race_shape(df)
print(f"styles {time.time()-t0:.0f}s; comment parsed on {df.rs_style.notna().mean():.3f} of runs")
df["led"] = np.where(df["rs_style"].isna(), np.nan, (df["rs_style"] >= 5.8).astype(float))
df["front"] = np.where(df["rs_class"].isna(), np.nan, (df["rs_class"] == 0).astype(float))
print("share of runs that led:", round(float(df.led.mean()), 4), "| led or disputed:", round(float(df.front.mean()), 4))
nl = df.groupby("rk")["led"].sum()
print("races by number of runners coded 'led':", nl.clip(upper=3).value_counts(normalize=True).round(3).to_dict())

# ---- 2. features --------------------------------------------------------------------------------------------------
df = df.sort_values(["date", "race_time", "track"], kind="stable").reset_index(drop=True)
h = df.groupby("horse_name", sort=False)
df["prev_style"] = h["rs_style"].shift(1)
df["prev_led"] = h["led"].shift(1)
df["prev2_led"] = h["led"].shift(2)
df["prev_date"] = h["date"].shift(1)
df["known"] = df["led"].notna().astype(float)
df["led0"] = df["led"].fillna(0.0)
pr = L.prior_cumsum(df, "horse_name", ["known", "led0"])
df["h_runs_known"], df["h_led_n"] = pr["known"], pr["led0"]
df["h_led_rate"] = (pr["led0"] + 0.5) / (pr["known"] + 5)
for key, nm in (("jockey_name", "jk"), ("trainer", "tr")):
    p = L.prior_cumsum(df, key, ["known", "led0"])
    df[f"{nm}_led_rate"] = (p["led0"] + 2) / (p["known"] + 30)
    df[f"{nm}_n"] = p["known"]
# sectionals: the horse's earlier early position (share of the field, 0 in front)
try:
    sc = pd.read_csv("data/external/sectionals.csv.gz", usecols=["day", "name_norm", "early_pct"])
    sc = sc.dropna(subset=["early_pct"]).groupby(["day", "name_norm"], as_index=False)["early_pct"].mean()
    df["name_norm"] = df["horse_name"].map(norm_name)
    df = df.merge(sc.rename(columns={"day": "race_date"}), on=["race_date", "name_norm"], how="left")
    df["sc_has"] = df["early_pct"].notna().astype(float)
    df["sc_sum"] = df["early_pct"].fillna(0.0)
    p = L.prior_cumsum(df, "horse_name", ["sc_has", "sc_sum"])
    df["sc_early_mean"] = (p["sc_sum"] / p["sc_has"]).where(p["sc_has"] > 0)
    df["sc_runs"] = p["sc_has"]
    print(f"sectionals: {df.sc_has.sum():,.0f} runs with an early position; "
          f"{df.sc_early_mean.notna().mean():.3f} of rows have an earlier one")
except Exception as exc:
    print("no sectionals:", exc)
    df["sc_early_mean"], df["sc_runs"] = np.nan, 0.0
df["dslr"] = (df["date"] - df["prev_date"]).dt.days
df["draw_rel"] = (df["stall"] / df["field"]).where((df["code"].isin(["flat", "aw"])) & (df["stall"] > 0))
df["or_rank"] = df.groupby("rk")["official_rating"].rank(ascending=False, method="min") / df["field"]
df["code_i"] = df["code"].map({"flat": 0, "aw": 1, "hurdle": 2, "chase": 3})
df["hcap_i"] = df["hcap"].astype(int)
g = df.groupby("rk")
df["h_led_rate_rank"] = g["h_led_rate"].rank(ascending=False, method="min")
df["prev_style_max_other"] = (g["prev_style"].transform("max"))
df["field_led_rate_sum"] = g["h_led_rate"].transform("sum")
FEATS = ["p_lead", "p_prom", "p_mid", "p_rear", "pred_epf", "style_n_eff", "lead_share", "lead_rank", "rel_epf",
         "shape_exp_leaders", "shape_lead1", "shape_lead2", "shape_lead_clarity", "shape_front_share",
         "prev_style", "prev_led", "prev2_led", "h_runs_known", "h_led_n", "h_led_rate", "h_led_rate_rank",
         "field_led_rate_sum", "jk_led_rate", "jk_n", "tr_led_rate", "tr_n", "sc_early_mean", "sc_runs",
         "dslr", "draw_rel", "stall", "field", "dist_furlongs", "code_i", "hcap_i", "or_rank", "official_rating",
         "horse_age", "jockeys_claim"]
print(f"features built {time.time()-t0:.0f}s")

# ---- 3. the leader model, out of sample -------------------------------------------------------------------------
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

df["pred"] = np.nan


def fit(train_mask):
    x = df[train_mask & df["led"].notna()]
    if len(x) > 1_500_000:
        x = x.sample(1_500_000, random_state=0)
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    m.fit(x[FEATS], x["led"])
    return m


for fit_to, score_from, score_to in (("2015-12-31", "2016-01-01", "2021-12-31"), ("2021-12-31", "2022-01-01", "2099")):
    m = fit((df.race_date >= "2010-01-01") & (df.race_date <= fit_to))
    sc_mask = (df.race_date >= score_from) & (df.race_date <= score_to)
    df.loc[sc_mask, "pred"] = m.predict_proba(df.loc[sc_mask, FEATS])[:, 1]
    print(f"fitted to {fit_to}, scored {score_from}..: {time.time()-t0:.0f}s")
imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
print("feature gain (last fit):", (imp / imp.sum()).round(3).head(15).to_dict())

s = df[df.pred.notna()].copy()
g = s.groupby("rk")
s["pred_share"] = s["pred"] / g["pred"].transform("sum")
s["pred_rank"] = g["pred"].rank(ascending=False, method="first")
s["pred2"] = s["pred"].where(s["pred_rank"] == 2).groupby(s["rk"]).transform("max").fillna(0)
s["pred_gap"] = g["pred"].transform("max") - s["pred2"]
s["base_rank"] = g["p_lead"].rank(ascending=False, method="first")
races_known = s[s.led.notna()].groupby("rk")["led"].sum()
k = s[s.rk.isin(races_known[races_known >= 1].index)]
for nm, col in (("leader model", "pred_rank"), ("run-style projection alone", "base_rank")):
    for per, (a, b) in (("2016-21", ("2016", "2021-12-31")), ("2022-26", ("2022", "2099"))):
        kk = k[(k.race_date >= a) & (k.race_date <= b)]
        top = kk[kk[col] == 1]
        print(f"{nm:28s} {per}: projected leader actually led in {top.led.mean():.3f} of {len(top):,} races "
              f"(led or disputed {top.front.mean():.3f})")
tst = s[(s.race_date >= "2022") & s.led.notna()]
print("AUC 2022-26: leader model", round(roc_auc_score(tst.led, tst.pred), 4), "| p_lead alone",
      round(roc_auc_score(tst.led, tst.p_lead), 4))
cal = tst.assign(b=pd.qcut(tst.pred, 10, duplicates="drop")).groupby("b", observed=True).agg(pred=("pred", "mean"),
                                                                                              led=("led", "mean"), n=("led", "size"))
print("calibration 2022-26 (decile: predicted vs actual lead rate):\n", cal.round(3).to_string())

# ---- 4. betting the projected leader ------------------------------------------------------------------------------
s = L.priced(s)
s["year"] = s["date"].dt.year
top = s["pred_rank"] == 1
d_band = pd.cut(s["dist_furlongs"], [0, 6.5, 9.5, 14, 99], labels=["<=6f", "7-9f", "10-14f", "15f+"])
f_band = pd.cut(s["field"], [0, 7, 11, 15, 99], labels=["2-7", "8-11", "12-15", "16+"])
rows = []
add = lambda nm, m: rows.append(L.rule(nm, s, m))  # noqa: E731
add("projected leader (model), all codes", top)
add("projected leader (run-style projection alone)", s["base_rank"] == 1)
add("HINDSIGHT actual leaders (led), all codes", s["led"] == 1)
for c in ("flat", "aw", "hurdle", "chase"):
    add(f"projected leader, {c}", top & (s.code == c))
for b in ["<=6f", "7-9f", "10-14f", "15f+"]:
    add(f"projected leader, trip {b}", top & (d_band == b))
for b in ["2-7", "8-11", "12-15", "16+"]:
    add(f"projected leader, field {b}", top & (f_band == b))
for lo in (0.3, 0.4, 0.5, 0.6):
    add(f"projected leader, predicted chance >= {lo}", top & (s.pred >= lo))
add("projected leader, clear (gap to 2nd >= 0.25)", top & (s.pred_gap >= 0.25))
add("projected leader, lone speed (2nd's chance < 0.15)", top & (s.pred2 < 0.15))
add("projected leader, contested (2nd's chance >= 0.35)", top & (s.pred2 >= 0.35))
add("projected leader, flat/aw sprints <=6f, chance >= 0.4", top & s.code.isin(["flat", "aw"]) & (d_band == "<=6f") & (s.pred >= 0.4))
add("projected leader, flat/aw, low draw (inside third)", top & s.code.isin(["flat", "aw"]) & (s.draw_rel <= 0.34))
add("projected leader, handicaps", top & s.hcap)
add("projected leader, non-handicaps", top & ~s.hcap)
add("HINDSIGHT projected leader, BSP >= 4", top & (s.bsp >= 4))
add("projected leader, morning price 3-11", top & s.morningwap.between(3, 11))
add("projected leader, not the market's favourite (morning rank > 1)",
    top & (s.groupby("rk")["morningwap"].rank(method="min") > 1))
t = L.rules_table(rows)
t.to_csv("out/strat_j_early_leader_rules.csv", index=False)

print("\n=== the projected leader by year (TEST and TRAIN): win A/E vs BSP / place A/E, actual lead rate ===")
for y, gg in s[top].groupby("year"):
    sc_ = L.score(gg)
    print(y, sc_.get("n"), round(sc_.get("A/E", np.nan), 3), round(sc_.get("A/E_pl", np.nan), 3),
          "A/E_am", round(sc_.get("A/E_am", np.nan), 3) if "A/E_am" in sc_ else "-",
          "led", round(float(gg.led.mean()), 3))
L.show("projected leader, win A/E by predicted chance band (TRAIN vs TEST)",
       L.tt(s[top].assign(band=pd.cut(s.loc[top, "pred"], [0, .2, .3, .4, .5, .6, .7, 1])), ["band"], 200))
L.show("ALL runners by predicted chance of leading (does the market price the lead?)",
       L.tt(s.assign(band=pd.cut(s.pred, [0, .05, .1, .2, .3, .4, .5, .6, 1])), ["band"], 500))
print(f"\ndone {time.time()-t0:.0f}s")
