"""Strategy J, round 2 (the owner, 5 Nov 2026: "try and improve the modelling here. recency weighting, run style last
time out, decayed over 1 run, 3 runs, 5 runs, 10 runs, career. Normalised by field size.").

Strategy J (strat_j_early_leader_1105.py) found the leader with AUC 0.795 and its pick returned win A/E 1.006 at BSP.
Here every run's early position is put on one scale normalised by field size -- rs_epf, 0 = led, 1 = last
(model/race_shape.py: from the comment's first positional phrase, or the place it names over the field) -- and each
horse's EARLIER runs are summarised five ways for three readings (early position, led, led or disputed):

    l1                last time out
    hl1 hl3 hl5 hl10  recency-weighted means, a run k runs back weighing 2^(-k / half-life), over every earlier run
    w3 w5 w10         plain means of the last 3, 5, 10 runs
    car               career mean
    + the count of earlier runs with a parsed position, and the runs since it last led

and each is also read against today's field: its rank in the race over the field size (0 = the most forward), and its
gap to the race's best. Three leader models, fitted the same way (2010-15 -> 2016-21, 2010-21 -> 2022 on):

    A  strategy J's features
    B  A + the recency set
    C  the recency set with its race-relative readings, draw, trip, code, field (no race_shape projection)

Each is scored on finding the leader (AUC, log loss, top-1 hit rate) and on backing its pick (win and place A/E against
BSP and the morning price, the owner's gates). Ties at the top of a race are broken at random (the table stores a
race's runners in finishing order: strat_j_tie_check_1105.py).
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
df = RS.add_run_styles(df)
df = RS.add_style_projection(df)
df = RS.add_race_shape(df)
df["led"] = np.where(df["rs_style"].isna(), np.nan, (df["rs_style"] >= 5.8).astype(float))
df["front"] = np.where(df["rs_class"].isna(), np.nan, (df["rs_class"] == 0).astype(float))
df["epf"] = df["rs_epf"]
df = df.sort_values(["date", "race_time", "track"], kind="stable").reset_index(drop=True)
df["day"] = RS.day_index(df)
print(f"rows {len(df):,}; styles {time.time()-t0:.0f}s")

# ---- strategy J's features ----------------------------------------------------------------------------------------
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
sc = pd.read_csv("data/external/sectionals.csv.gz", usecols=["day", "name_norm", "early_pct"])
sc = sc.dropna(subset=["early_pct"]).groupby(["day", "name_norm"], as_index=False)["early_pct"].mean()
df["name_norm"] = df["horse_name"].map(norm_name)
df = df.merge(sc.rename(columns={"day": "race_date"}), on=["race_date", "name_norm"], how="left")
df["sc_has"] = df["early_pct"].notna().astype(float)
df["sc_sum"] = df["early_pct"].fillna(0.0)
p = L.prior_cumsum(df, "horse_name", ["sc_has", "sc_sum"])
df["sc_early_mean"] = (p["sc_sum"] / p["sc_has"]).where(p["sc_has"] > 0)
df["sc_runs"] = p["sc_has"]
df["dslr"] = (df["date"] - df["prev_date"]).dt.days
df["draw_rel"] = (df["stall"] / df["field"]).where((df["code"].isin(["flat", "aw"])) & (df["stall"] > 0))
df["or_rank"] = df.groupby("rk")["official_rating"].rank(ascending=False, method="min") / df["field"]
df["code_i"] = df["code"].map({"flat": 0, "aw": 1, "hurdle": 2, "chase": 3})
df["hcap_i"] = df["hcap"].astype(int)
g = df.groupby("rk")
df["h_led_rate_rank"] = g["h_led_rate"].rank(ascending=False, method="min")
df["field_led_rate_sum"] = g["h_led_rate"].transform("sum")
FEATS_J = ["p_lead", "p_prom", "p_mid", "p_rear", "pred_epf", "style_n_eff", "lead_share", "lead_rank", "rel_epf",
           "shape_exp_leaders", "shape_lead1", "shape_lead2", "shape_lead_clarity", "shape_front_share",
           "prev_style", "prev_led", "prev2_led", "h_runs_known", "h_led_n", "h_led_rate", "h_led_rate_rank",
           "field_led_rate_sum", "jk_led_rate", "jk_n", "tr_led_rate", "tr_n", "sc_early_mean", "sc_runs",
           "dslr", "draw_rel", "stall", "field", "dist_furlongs", "code_i", "hcap_i", "or_rank", "official_rating",
           "horse_age", "jockeys_claim"]

# ---- the recency set ----------------------------------------------------------------------------------------------
REC = []
vals = {"epf": df["epf"].to_numpy(float), "led": df["led"].to_numpy(float), "front": df["front"].to_numpy(float)}
for hl in (1, 3, 5, 10, np.inf):
    sums, cnt, runs = RS.horse_decayed_prior(df["horse_name"].astype(str), df["day"].to_numpy(), vals, hl)
    tag = "car" if np.isinf(hl) else f"hl{hl}"
    for v in vals:
        df[f"{v}_{tag}"] = np.where(cnt[v] > 0, sums[v] / np.where(cnt[v] > 0, cnt[v], 1), np.nan)
        REC.append(f"{v}_{tag}")
    if np.isinf(hl):
        df["runs_parsed"] = cnt["epf"]
        df["runs_all"] = runs
h = df.groupby("horse_name", sort=False)
for v in vals:
    prev = h[v].shift(1)
    df[f"{v}_l1"] = prev
    REC.append(f"{v}_l1")
    for w in (3, 5, 10):
        df[f"{v}_w{w}"] = prev.groupby(df["horse_name"], sort=False).rolling(w, min_periods=1).mean() \
            .reset_index(level=0, drop=True)
        REC.append(f"{v}_w{w}")
# runs since it last led (earlier runs only)
run_no = df.groupby("horse_name", sort=False).cumcount().astype(float)
last = pd.Series(np.where(df["led"].to_numpy() == 1, run_no, np.nan), index=df.index)
last = last.groupby(df["horse_name"], sort=False).shift(1).groupby(df["horse_name"], sort=False).ffill()
df["runs_since_led"] = run_no - last
REC += ["runs_parsed", "runs_all", "runs_since_led"]
# against today's field (normalised by field size): rank over the field (0 = most forward), gap to the race's best
REL = []
g = df.groupby("rk")
for c in ["epf_l1", "epf_hl1", "epf_hl3", "epf_hl5", "epf_hl10", "epf_car", "epf_w3", "epf_w5", "epf_w10"]:
    df[f"{c}_rk"] = (g[c].rank(method="min") - 1) / (df["field"] - 1).clip(lower=1)
    df[f"{c}_gap"] = df[c] - g[c].transform("min")
    REL += [f"{c}_rk", f"{c}_gap"]
for c in ["led_hl3", "led_hl5", "led_car", "front_hl3", "led_w5"]:
    df[f"{c}_rk"] = (g[c].rank(ascending=False, method="min") - 1) / (df["field"] - 1).clip(lower=1)
    df[f"{c}_gap"] = g[c].transform("max") - df[c]
    df[f"{c}_field_sum"] = g[c].transform("sum")
    REL += [f"{c}_rk", f"{c}_gap", f"{c}_field_sum"]
print(f"recency set {len(REC)} + race-relative {len(REL)} features, {time.time()-t0:.0f}s")
print("coverage of the recency set (share of rows):", df[["epf_l1", "epf_hl3", "epf_car", "led_w5"]].notna().mean().round(3).to_dict())

BASE = ["draw_rel", "stall", "field", "dist_furlongs", "code_i", "hcap_i", "or_rank", "dslr", "jk_led_rate",
        "tr_led_rate"]
VARIANTS = {"A_strategy_J": FEATS_J,
            "B_J_plus_recency": FEATS_J + REC + REL,
            "C_recency_only": REC + REL + BASE}

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

rng = np.random.default_rng(0)
df["_u"] = rng.random(len(df))                   # a random tie-break (rows within a race are in finishing order)
WINDOWS = (("2015-12-31", "2016-01-01", "2021-12-31"), ("2021-12-31", "2022-01-01", "2099"))
results, picks = [], {}
for name, feats in VARIANTS.items():
    df[f"pred_{name}"] = np.nan
    for fit_to, a, b in WINDOWS:
        x = df[(df.race_date >= "2010-01-01") & (df.race_date <= fit_to) & df["led"].notna()]
        if len(x) > 1_500_000:
            x = x.sample(1_500_000, random_state=0)
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        m.fit(x[feats], x["led"])
        msk = (df.race_date >= a) & (df.race_date <= b)
        df.loc[msk, f"pred_{name}"] = m.predict_proba(df.loc[msk, feats])[:, 1]
    imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print(f"\n{name}: {len(feats)} features, fitted {time.time()-t0:.0f}s; top gain:",
          (imp / imp.sum()).round(3).head(12).to_dict())
    col = f"pred_{name}"
    s = df[df[col].notna()]
    rk = s.assign(_k=s[col] + s["_u"] * 1e-9).groupby("rk")["_k"].rank(ascending=False, method="first")
    df.loc[s.index, f"rank_{name}"] = rk
    for per, (lo, hi) in (("2016-21", ("2016", "2021-12-31")), ("2022-26", ("2022", "2099"))):
        q = s[(s.race_date >= lo) & (s.race_date <= hi)]
        k = q[q.led.notna()]
        races = k.groupby("rk")["led"].sum()
        top = k[(df.loc[k.index, f"rank_{name}"] == 1) & k.rk.isin(races[races >= 1].index)]
        results.append({"model": name, "period": per, "AUC": roc_auc_score(k.led, k[col]),
                        "logloss": log_loss(k.led, k[col].clip(1e-6, 1 - 1e-6)),
                        "top1_led": top.led.mean(), "top1_led_or_disputed": top.front.mean(), "races": len(top)})
R = pd.DataFrame(results)
print("\n=== finding the leader ===\n", R.round(4).to_string(index=False))

# ---- betting each model's pick ------------------------------------------------------------------------------------
s = L.priced(df[df["pred_A_strategy_J"].notna()]).copy()
s["year"] = s["date"].dt.year
rows = []
for name in VARIANTS:
    top = s[f"rank_{name}"] == 1
    p = s[f"pred_{name}"]
    rows.append(L.rule(f"{name}: projected leader", s, top))
    rows.append(L.rule(f"{name}: projected leader, chance >= 0.5", s, top & (p >= 0.5)))
    rows.append(L.rule(f"{name}: projected leader, flat/aw", s, top & s.code.isin(["flat", "aw"])))
    rows.append(L.rule(f"{name}: projected leader, jumps", s, top & s.code.isin(["hurdle", "chase"])))
    p2 = p.where(s[f"rank_{name}"] == 2).groupby(s["rk"]).transform("max").fillna(0)
    rows.append(L.rule(f"{name}: projected leader, lone (2nd's chance < 0.15)", s, top & (p2 < 0.15)))
rows.append(L.rule("HINDSIGHT actual leaders", s, s["led"] == 1))
t = L.rules_table(rows)
t.to_csv("out/strat_j2_leader_recency_rules.csv", index=False)
best = R[R.period == "2022-26"].sort_values("AUC").iloc[-1]["model"]
L.show(f"{best}: every runner by predicted chance of leading, TRAIN vs TEST",
       L.tt(s.assign(band=pd.cut(s[f"pred_{best}"], [0, .05, .1, .2, .3, .4, .5, .6, 1])), ["band"], 500))
print(f"\ndone {time.time()-t0:.0f}s")
