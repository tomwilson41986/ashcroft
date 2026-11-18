"""Phase 0 baseline_v0 (the owner's model brief, 18 Nov 2026; horse racing only; targets: the BFSP, and normalised
beaten lengths adjusted for field size and class). Read-only.

baseline_v0 is the served recipe ("st": the served features and blocks, huber, 12,000 rounds) as the research loop
fitted it walk-forward in iteration 126 (run 43895762307, artifact research-iter126-stack-collateral-164,
oos_st.csv): six 3-month folds, 2024-12-28 to 2026-03-31, every prediction out of sample (each fold's model trained on
races before it, purged 30 days). The locked holdout (>= 2026-04-01) is not read.

Scored here, every interval resampled by race day (1,000 draws):
  1. the BFSP target: mean absolute log error of the predicted price against the BFSP, overall and by code, price
     band, field size and quarter
  2. win probability against the market (the BSP, normalised in the race): log loss and Brier per runner, the
     calibration by decile of our probability (ECE), and the Benter test: a conditional logit on [log p_model,
     log p_market] fitted walk-forward by fold (each fold scored by a fit on the folds before it), its weights and its
     log-likelihood against the market alone
  3. the normalised beaten lengths: within each race, Spearman's correlation of our price order with race-centred
     pounds beaten (lbsc, model/form_windows.run_measures; the rule of 18 Nov: the most informative outcome first),
     with lbsc over the field-size spread s(n) (model/blocks/nbl_rating.py; class is constant within a race, so the
     class adjustment does not move a within-race reading) and with the normalised finishing position; the BSP's
     order read the same way beside it
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

sys.path.insert(0, os.getcwd())   # the repo's own modules (model/), run from the root

RUN, ART, FILE = 43895762307, "research-iter126-stack-collateral-164", "oos_st.csv"
OUT = "out/phase0_baseline_v0.json"
B = 1000
rng = np.random.default_rng(20261118)
rep: dict = {"source": f"run {RUN} artifact {ART} {FILE}"}


def say(*a):
    print(*a, flush=True)


def fetch():
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts?per_page=100", headers=h,
                     timeout=60)
    arts = [a for a in r.json().get("artifacts", []) if a["name"] == ART]
    if not arts:
        raise SystemExit(f"artifact {ART} not found ({r.status_code})")
    z = requests.get(arts[0]["archive_download_url"], headers=h, timeout=600)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        return pd.read_csv(zf.open(FILE))


def boot(day_codes: np.ndarray, num: np.ndarray, den: np.ndarray | None = None, level=0.95):
    """Ratio of sums (or mean) with a race-day bootstrap interval."""
    den = np.ones_like(num) if den is None else den
    days, inv = np.unique(day_codes, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(days))
    sd = np.bincount(inv, weights=den, minlength=len(days))
    est = sn.sum() / sd.sum()
    idx = rng.integers(0, len(days), size=(B, len(days)))
    bs = sn[idx].sum(axis=1) / sd[idx].sum(axis=1)
    a = (1 - level) / 2
    return round(float(est), 5), round(float(np.quantile(bs, a)), 5), round(float(np.quantile(bs, 1 - a)), 5)


# ------------------------------------------------------------------ data
oos = fetch()
say(f"== {rep['source']}: {len(oos):,} rows, folds {sorted(oos.fold_idx.unique())}, "
    f"{oos.race_date.min()} to {oos.race_date.max()}")
oos["race_date"] = pd.to_datetime(oos.race_date).dt.strftime("%Y-%m-%d")
assert oos.race_date.max() < "2026-04-01", "the locked holdout must not be read"

con = sqlite3.connect("horse_racing.db")
cols = [r[1] for r in con.execute('PRAGMA table_info("race_results")')]
want = [c for c in ("raceid", "race_date", "race_time", "track", "horse_name", "placing_numerical", "number_of_runners",
                    "LB", "total_dst_bt", "dist_furlongs", "prize_money", "race_type", "surface_type", "official_rating",
                    "RSR", "bfsp") if c in cols]
res = pd.read_sql_query(f"SELECT {', '.join(want)} FROM race_results WHERE race_date >= ? AND race_date < ?", con,
                        params=(oos.race_date.min(), "2026-04-01"))
con.close()
res["race_date"] = pd.to_datetime(res.race_date).dt.strftime("%Y-%m-%d")
from model.blocks.nbl_rating import _s_of, run_parts, spread_by_field  # noqa: E402
from model.form_windows import run_measures  # noqa: E402

m = run_measures(res)
res["lbsc"], res["nfp_out"] = m["lbsc"], m["nfp"]
# s(n) from the races before 2023 is the block's definition; this window starts in Dec 2024, so the spread is read
# from this window's own races (descriptive only: a per-race constant, it does not move a within-race correlation)
p = run_parts(res)
p_early = np.ones(len(res), dtype=bool)
res["rel_s"] = p["rel"] / _s_of(p["n"], spread_by_field(p["rel"], p["n"], p["race"], p_early))
k = ["race_date", "race_time", "track", "horse_name"]
before = len(oos)
d = oos.merge(res[k + ["lbsc", "nfp_out", "rel_s"]].drop_duplicates(k), on=k, how="left")
rep["join"] = {"oos_rows": before, "joined_rows": len(d), "lbsc_found": int(d.lbsc.notna().sum())}
say("join", rep["join"])

d["race"] = pd.factorize(d.race_date + "|" + d.race_time.astype(str) + "|" + d.track)[0]
d["day"] = pd.factorize(d.race_date)[0]
ok = (d.bfsp > 1) & (d.predicted_bfsp > 1)
rep["rows_scored_bfsp"] = int(ok.sum())
e = d[ok].copy()
e["err"] = np.abs(np.log(e.predicted_bfsp) - np.log(e.bfsp))

# ------------------------------------------------------------------ 1. the BFSP target
say("\n== 1. BFSP target: mean absolute log error (95% CI by race day)")
tab = {"overall": boot(e.day.to_numpy(), e.err.to_numpy())}
e["code"] = np.where(e.race_code.astype(str).str.contains("Hurdle|Chase|NH|National Hunt", case=False), "jumps",
                     np.where(e.surface_type.astype(str).str.contains("Polytrack|Tapeta|Fibresand|All", case=False),
                              "aw", "flat turf"))
e["band"] = pd.cut(e.bfsp, [1, 3, 6, 12, 25, 50, 1e9], labels=["<3", "3-6", "6-12", "12-25", "25-50", "50+"])
e["fs"] = pd.cut(e.number_of_runners, [0, 7, 11, 15, 99], labels=["2-7", "8-11", "12-15", "16+"])
e["qtr"] = pd.to_datetime(e.race_date).dt.to_period("Q").astype(str)
for g in ("code", "band", "fs", "qtr"):
    for v, s in e.groupby(g, observed=True):
        tab[f"{g}={v}"] = boot(s.day.to_numpy(), s.err.to_numpy()) + (len(s),)
for kk, v in tab.items():
    say(f"  {kk:20s} {v}")
rep["bfsp_log_mae"] = tab

# ------------------------------------------------------------------ 2. win probability against the market
w = d[ok & d.placing_numerical.notna()].copy()
w = w[w.groupby("race").won.transform("sum") == 1]          # races with one winner among the rows held
q_m = 1 / w.bfsp
w["p_mkt"] = q_m / q_m.groupby(w.race).transform("sum")
q = 1 / w.predicted_bfsp
w["p_mod"] = q / q.groupby(w.race).transform("sum")
y = w.won.astype(float).to_numpy()
say(f"\n== 2. win probability: {w.race.nunique():,} races, {len(w):,} runners with one winner held")
for name in ("p_mod", "p_mkt"):
    pr = np.clip(w[name].to_numpy(), 1e-9, 1 - 1e-9)
    ll = -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    br = (pr - y) ** 2
    rep[f"{name}_logloss"] = boot(w.day.to_numpy(), ll)
    rep[f"{name}_brier"] = boot(w.day.to_numpy(), br)
    say(f"  {name}: log loss {rep[f'{name}_logloss']}, Brier {rep[f'{name}_brier']}")
pm, pk = np.clip(w.p_mod.to_numpy(), 1e-9, 1 - 1e-9), np.clip(w.p_mkt.to_numpy(), 1e-9, 1 - 1e-9)
diff = -(y * np.log(pm) + (1 - y) * np.log(1 - pm)) + (y * np.log(pk) + (1 - y) * np.log(1 - pk))
rep["logloss_model_minus_market"] = boot(w.day.to_numpy(), diff)
say(f"  log loss, model minus market: {rep['logloss_model_minus_market']} (positive: the market is better)")
w["dec"] = pd.qcut(w.p_mod, 10, labels=False, duplicates="drop")
cal = w.groupby("dec").agg(n=("won", "size"), p=("p_mod", "mean"), obs=("won", "mean"), mkt=("p_mkt", "mean"))
rep["ece_model"] = round(float((cal.n * (cal.p - cal.obs).abs()).sum() / cal.n.sum()), 5)
say("  calibration by decile of p_model\n" + cal.round(4).to_string() + f"\n  ECE {rep['ece_model']}")
rep["calibration"] = cal.round(5).reset_index().to_dict(orient="records")


def clogit_nll(beta, X, race_idx, yv):
    s = X @ beta
    mx = pd.Series(s).groupby(race_idx).transform("max").to_numpy()
    ex = np.exp(s - mx)
    den = np.bincount(race_idx, weights=ex)
    lse = np.log(den) + pd.Series(mx).groupby(race_idx).first().to_numpy()
    return -(s[yv == 1].sum() - lse.sum())


def fit(X, race_idx, yv):
    return minimize(clogit_nll, np.ones(X.shape[1]), args=(X, race_idx, yv), method="BFGS").x


w["lm"], w["lk"] = np.log(w.p_mod), np.log(w.p_mkt)
folds = sorted(w.fold_idx.unique())
per_race = []
coefs = []
for f in folds[1:]:
    tr, te = w[w.fold_idx < f], w[w.fold_idx == f]
    rtr = pd.factorize(tr.race)[0]
    b2 = fit(tr[["lm", "lk"]].to_numpy(), rtr, tr.won.to_numpy())
    b1 = fit(tr[["lk"]].to_numpy(), rtr, tr.won.to_numpy())
    coefs.append({"fold": int(f), "beta_model": round(float(b2[0]), 4), "beta_market": round(float(b2[1]), 4),
                  "beta_market_alone": round(float(b1[0]), 4)})
    for (race, day), g in te.groupby(["race", "day"]):
        s2 = g[["lm", "lk"]].to_numpy() @ b2
        s1 = g[["lk"]].to_numpy() @ b1
        yy = g.won.to_numpy() == 1
        l2 = s2[yy][0] - np.log(np.exp(s2 - s2.max()).sum()) - s2.max()
        l1 = s1[yy][0] - np.log(np.exp(s1 - s1.max()).sum()) - s1.max()
        per_race.append((day, l2 - l1))
pr_arr = np.array(per_race)
rep["benter"] = {"weights_by_fold": coefs,
                 "loglik_gain_per_race_x1000": tuple(round(x * 1000, 3) for x in
                                                     boot(pr_arr[:, 0], pr_arr[:, 1])),
                 "races_scored": len(pr_arr), "note": "fold 0 has no earlier fold to fit on, so it is not scored"}
say("\n  Benter (walk-forward by fold):")
for c in coefs:
    say("   ", c)
say(f"  log-likelihood gain over the market alone, per race x1000: {rep['benter']['loglik_gain_per_race_x1000']} "
    f"({len(pr_arr):,} races)")

# ------------------------------------------------------------------ 3. normalised beaten lengths
say("\n== 3. normalised beaten lengths: Spearman within race, race mean (95% CI by race day)")
nb = d[ok].copy()
nb["s_mod"], nb["s_mkt"] = -np.log(nb.predicted_bfsp), -np.log(nb.bfsp)
rows = []
for out in ("lbsc", "rel_s", "nfp_out"):
    sub = nb[nb[out].notna()]
    sub = sub[sub.groupby("race")[out].transform("size") >= 4]
    rk = sub.groupby("race")[[out, "s_mod", "s_mkt"]].rank()
    rk["race"], rk["day"] = sub.race, sub.day
    cm = rk.groupby("race").apply(lambda g: pd.Series({
        "mod": g[out].corr(g.s_mod), "mkt": g[out].corr(g.s_mkt), "day": g.day.iloc[0]}), include_groups=False)
    cm = cm.dropna() if len(cm) else cm
    if len(cm) == 0 or "mod" not in cm.columns:
        rows.append({"outcome": out, "races": 0, "note": "no data found"})
        say(f"  {out:8s} no data found")
        continue
    r_mod = boot(cm.day.to_numpy(), cm["mod"].to_numpy())
    r_mkt = boot(cm.day.to_numpy(), cm["mkt"].to_numpy())
    r_dif = boot(cm.day.to_numpy(), (cm["mod"] - cm["mkt"]).to_numpy())
    rows.append({"outcome": out, "races": len(cm), "model": r_mod, "market_bsp": r_mkt, "model_minus_market": r_dif})
    say(f"  {out:8s} races {len(cm):,}: model {r_mod}, BSP {r_mkt}, model - BSP {r_dif}")
rep["nbl"] = rows

os.makedirs("out", exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1, default=str)
say(f"\nwritten {OUT}")
