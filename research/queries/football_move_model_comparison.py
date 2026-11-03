"""QA experiment (development seasons only): OLS vs LightGBM vs a stacked blend for the early-to-close home move."""
import json, sys
import numpy as np, pandas as pd, lightgbm as lgb
from football import feature_eval as fe
d = fe.prepare(pd.read_parquet(sys.argv[1]))
m = d[d["same_book"] & (d["open_1x2_src"] == "PS")].copy()
m["yh"] = np.log(m["qc_h"] / m["qo_h"])
m["lq_fav"] = np.log(m["qo_h"] / m["qo_a"]); m["q_draw"] = m["qo_d"]
m["over_open"] = 1/m["open_h"] + 1/m["open_d"] + 1/m["open_a"]
m["league"] = pd.Categorical(m["fd_code"].astype(str))
team = fe.FAMILIES["all"]; mkt = ["lq_fav", "q_draw", "over_open", "qo_o"]
rng = np.random.default_rng(0)
def gbm(): return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
rows = []
for s in range(2016, 2025):
    tr, te = m[(m.season >= 2013) & (m.season < s)], m[m.season == s]
    if te.empty: continue
    X, Xt = tr[team].to_numpy(float), te[team].to_numpy(float)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0); sd[sd == 0] = 1
    Z = np.c_[np.ones(len(tr)), np.nan_to_num((X-mu)/sd)]; Zt = np.c_[np.ones(len(te)), np.nan_to_num((Xt-mu)/sd)]
    y = tr.yh.to_numpy()
    p_ols = Zt @ np.linalg.lstsq(Z, y, rcond=None)[0]
    g1 = gbm().fit(tr[team], y); p_gbm = g1.predict(te[team])
    g2 = gbm().fit(tr[team + mkt + ["league"]], y); p_gbm_mkt = g2.predict(te[team + mkt + ["league"]])
    gp = gbm().fit(tr[team + mkt + ["league"]], rng.permutation(y)); p_perm = gp.predict(te[team + mkt + ["league"]])
    rows.append(te.assign(p_ols=p_ols, p_gbm=p_gbm, p_gbm_mkt=p_gbm_mkt, p_perm=p_perm, ybar=y.mean()))
t = pd.concat(rows)
t["p_stack"] = (t.p_ols + t.p_gbm_mkt) / 2
out = {"rows": int(len(t)), "seasons": sorted(int(x) for x in t.season.unique())}
sse = lambda p: float(((t.yh - p) ** 2).sum()); base = sse(t.ybar)
for c in ("p_ols", "p_gbm", "p_gbm_mkt", "p_stack", "p_perm"):
    dec = t[c].abs() >= t[c].abs().quantile(0.9)
    caught = (np.sign(t.loc[dec, c]) * t.loc[dec, "yh"])
    out[c] = {"r2_vs_mean": 1 - sse(t[c]) / base, "top_decile_caught": float(caught.mean()),
              "top_decile_right_way": float((caught > 0).mean())}
print(json.dumps(out, indent=1))
