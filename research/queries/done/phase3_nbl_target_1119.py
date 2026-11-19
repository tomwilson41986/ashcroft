"""Phase 3, step 1 of the owner's model brief (19 Nov 2026: "carry on with Phase 3"; 18 Nov: "a new model target, which
is normalised lengths beaten ... field size and class (prize money) adjusted"; iteration 129's step two). A screen,
read-only: does a model trained on the normalised beaten-lengths figure bring the served price forecast closer to the
BFSP?

1. The target, per run: the figure of model/blocks/nbl_rating.py, race-centred pounds beaten over the field-size
   spread s(n) plus GAMMA x class (log prize less the code's median), GAMMA chosen there on races before 2023.
   Finishers with a margin only (a non-finisher has no figure).
2. The model: LightGBM regression on the served model's 1,181 features (the cached matrix, the 24 served blocks
   built as training builds them, race dates from 2021-01-01; the locked holdout from 2026-04-01 dropped before
   anything is computed), refitted at the start of each of baseline_v0's folds on races before it less 30 days
   (fold 5, four days, is priced by fold 4's model). Every runner of a fold is priced, finisher or not.
3. The screen, on baseline_v0's out-of-sample rows (iteration 126, run 43895762307, oos_st.csv, 2024-12-28 to
   2026-03-31): the blend log price = log(st's price) - b x (the runner's predicted figure less its race's mean),
   the book renormalised, b chosen on the folds before each fold (from a grid), folds 1-5 scored. Read: the paired
   mean absolute log error against the BFSP, blend minus st (90% interval by race day), by code; and the order
   agreement with race-centred pounds beaten of the figure model alone, st and the BSP (research/harness.py).

Pass (the owner's reading for a price-model change): the blend's paired error at least 0.0007 below st, the 90%
interval excluding zero; then the figure model is built as a block for the research loop. One trial.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.getcwd())
from model import blocks, feature_cache  # noqa: E402
from model.blocks import nbl_rating as nb  # noqa: E402
from research import harness as h  # noqa: E402

RUN, ART, FILE = 43895762307, "research-iter126-stack-collateral-164", "oos_st.csv"
FROM, HOLDOUT, PURGE = "2021-01-01", "2026-04-01", 30
KEY = ["race_date", "race_time", "track", "horse_name"]
GRID = np.round(np.arange(0.0, 0.401, 0.02), 3)
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 500,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 5, "lambda_l2": 1.0, "seed": 42,
          "deterministic": True, "force_col_wise": True, "verbose": -1}
ROUNDS = 800
OUT = "out/phase3_nbl_target.json"
rep: dict = {}


def say(*a):
    print(*a, flush=True)


def fetch_oos():
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY", "tomwilson41986/ashcroft")
    hd = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts?per_page=100", headers=hd,
                     timeout=60)
    art = [a for a in r.json().get("artifacts", []) if a["name"] == ART]
    z = requests.get(art[0]["archive_download_url"], headers=hd, timeout=600)
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        return pd.read_csv(zf.open(FILE))


def norm_keys(df):
    out = df.copy()
    out["race_date"] = pd.to_datetime(out["race_date"]).dt.strftime("%Y-%m-%d")
    for c in ("race_time", "track", "horse_name"):
        out[c] = out[c].astype(str).str.strip()
    return out


# ---- the matrix
meta = json.load(open("data/models/bfsp_model_meta.json"))
served, served_blocks = list(meta["feature_cols"]), list(meta["drop_in_blocks"])
mat = None
for start in ("2018-01-01", "2021-01-01"):
    hit = feature_cache.load(".feature_cache", feature_cache.cache_key("horse_racing.db", start))
    if hit is not None:
        mat = hit[0]
        say(f"matrix from {start}: {len(mat):,} rows")
        break
if mat is None:
    raise SystemExit("no cached feature matrix: no data found")
d = pd.to_datetime(mat["race_date"])
mat = mat.loc[d < pd.Timestamp(HOLDOUT)].reset_index(drop=True)
say(f"after the locked holdout's rows are dropped: {len(mat):,}")
mat, _ = blocks.attach_as_trained(mat, served_blocks)

# ---- the target (nbl_rating's figure), on the whole frame so the race means and spreads are the block's
gamma = nb.build(mat[[c for c in nb.READS if c in mat.columns]].copy()).attrs.get("nb_gamma", nb.GAMMA_DEFAULT)
p = nb.run_parts(mat)
s = nb._s_of(p["n"], nb.spread_by_field(p["rel"], p["n"], p["race"], p["early"]))
med = {j: np.nanmedian(p["prize"][p["early"] & (p["jumps"] == j)]) for j in (False, True)}
cls = np.nan_to_num(p["prize"] - np.where(p["jumps"], med[True], med[False]), nan=0.0)
y = p["rel"] / s + gamma * cls
rep["gamma"] = float(gamma)
say(f"target: GAMMA {gamma}; rows with a figure {np.isfinite(y).sum():,} of {len(y):,}")

feats = [c for c in served if c in mat.columns]
use = (pd.to_datetime(mat["race_date"]) >= pd.Timestamp(FROM)).to_numpy()
idx = np.flatnonzero(use)
X = np.empty((len(idx), len(feats)), dtype=np.float32)                # one array, filled a column at a time
for j, c in enumerate(feats):
    X[:, j] = pd.to_numeric(mat[c].iloc[idx], errors="coerce").to_numpy(dtype=np.float32)
dates = pd.to_datetime(mat["race_date"].iloc[idx]).to_numpy()
keys = norm_keys(mat[KEY].iloc[idx]).reset_index(drop=True)
y = y[idx]
del mat
say(f"features: {len(feats)} of {len(served)} served")

# ---- the out-of-sample rows and their folds
oos = norm_keys(fetch_oos())
starts = oos.groupby("fold_idx")["race_date"].min().sort_index()
say("fold starts: " + json.dumps(starts.to_dict()))
pos = keys.reset_index().merge(oos[KEY + ["fold_idx"]], on=KEY, how="inner")
say(f"out-of-sample rows found in the matrix: {len(pos):,} of {len(oos):,}")

import lightgbm as lgb  # noqa: E402

pred = pd.Series(np.nan, index=oos.index)
oos_idx = oos.reset_index().merge(pos[KEY + ["index"]].rename(columns={"index": "row"}), on=KEY, how="left")
gain = None
model = None
for f, start in starts.items():
    if f < 5 or model is None:
        cut = pd.Timestamp(start) - pd.Timedelta(days=PURGE)
        tr = (dates >= np.datetime64(FROM)) & (dates < np.datetime64(cut)) & np.isfinite(y)
        model = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], free_raw_data=True), ROUNDS)
        say(f"fold {f}: trained on {tr.sum():,} runs before {cut.date()}")
        g = pd.Series(model.feature_importance("gain"), index=feats)
        gain = g if gain is None else gain + g
    sel = oos_idx[(oos_idx["fold_idx"] == f) & oos_idx["row"].notna()]
    pred.loc[sel["index"].to_numpy()] = model.predict(X[sel["row"].astype(int).to_numpy()])
oos["nbl_pred"] = pred
say(f"priced: {oos['nbl_pred'].notna().sum():,} of {len(oos):,}")
rep["top_gain"] = (gain / gain.sum()).sort_values(ascending=False).head(20).round(4).to_dict()
say("top features of the figure model (gain share): " + json.dumps(rep["top_gain"]))

# ---- the screen
w = oos[(oos["bfsp"] > 1) & (oos["predicted_bfsp"] > 1) & oos["nbl_pred"].notna()].copy()
w["race"] = h.race_ids(w)
w["z"] = w["nbl_pred"] - w.groupby("race")["nbl_pred"].transform("mean")
w["lst"] = np.log(w["predicted_bfsp"])
w["lb"] = np.log(w["bfsp"])


def blend_err(frame, b):
    lp = frame["lst"] - b * frame["z"]
    q = h.normalise(np.exp(lp).to_numpy(), frame["race"].to_numpy())
    return np.abs(np.log(1.0 / q) - frame["lb"].to_numpy())


chosen = {}
parts = []
for f in sorted(w["fold_idx"].unique()):
    if f == 0:
        continue
    past = w[w["fold_idx"] < f]
    b = min(GRID, key=lambda g: blend_err(past, g).mean())
    chosen[int(f)] = float(b)
    cur = w[w["fold_idx"] == f].copy()
    cur["err_blend"] = blend_err(cur, b)
    cur["err_st"] = np.abs(cur["lst"] - cur["lb"])
    parts.append(cur)
s = pd.concat(parts)
s["day"] = h.day_ids(s)
s["code"] = h.code_of(s)
diff = (s["err_blend"] - s["err_st"]).to_numpy()
rep["b_by_fold"] = chosen
rep["paired"] = {"runners": len(s), "st": float(s["err_st"].mean()), "blend": float(s["err_blend"].mean()),
                 "difference_90": h.day_bootstrap(s["day"].to_numpy(), diff, level=0.90)}
for c, g in s.groupby("code"):
    rep["paired"][f"difference_90_{c}"] = h.day_bootstrap(g["day"].to_numpy(), (g["err_blend"] - g["err_st"]).to_numpy(),
                                                          level=0.90)
say("\n== the blend against st (folds 1-5), mean absolute log error against the BFSP")
say(json.dumps(rep["paired"], indent=1))
say("b chosen on the earlier folds: " + json.dumps(chosen))

# order agreement with race-centred pounds beaten (the outcome first, the rule of 18 Nov)
import sqlite3  # noqa: E402
from model.form_windows import run_measures  # noqa: E402
con = sqlite3.connect("horse_racing.db")
cols = ["race_date", "race_time", "track", "horse_name", "placing_numerical", "number_of_runners", "total_dst_bt",
        "dist_furlongs", "official_rating", "bfsp"]
res = norm_keys(pd.read_sql_query(f"SELECT {', '.join(cols)} FROM race_results WHERE race_date >= ? AND race_date < ?",
                                  con, params=(oos["race_date"].min(), HOLDOUT)))
con.close()
res["lbsc"] = run_measures(res)["lbsc"]
o = oos.merge(res[KEY + ["lbsc"]].drop_duplicates(KEY), on=KEY, how="left")
o = o[o["nbl_pred"].notna()]
fig = h.order_agreement(o.assign(predicted_bfsp=np.exp(-o["nbl_pred"])), "lbsc")
st = h.order_agreement(o, "lbsc")
rep["order_lbsc"] = {"figure_model": fig.get("model"), "st": st.get("model"), "bsp": st.get("market"),
                     "figure_minus_bsp": fig.get("model_minus_market"), "races": fig.get("races")}
say("\n== order agreement with race-centred pounds beaten (Spearman within race, 95% by race day)")
say(json.dumps(rep["order_lbsc"], indent=1))
os.makedirs("out", exist_ok=True)
json.dump(rep, open(OUT, "w"), indent=1, default=str)
say(f"written {OUT}")
