"""Phase 2, step 1 of the owner's model brief (19 Nov 2026: "carry on with Phase 2"): the descriptive audit of every
served feature, horse racing only. Read-only; no model is fitted.

For each of the served model's 1,181 features (data/models/bfsp_model_meta.json feature_cols; the three members share
them), on the cached feature matrix with the served blocks built as training builds them, race dates 2021-01-01 to
2026-03-31 only (the locked holdout from 2026-04-01 is dropped before anything is computed):

    family        the block that builds it (block:<name>), else the smallest list in model/bfsp_features.py that holds it
    missing       the share of rows with no value, overall and by year
    psi_<year>    the population stability index of the year against 2021 (deciles of 2021's values, missing its
                  own bin); above 0.25 is the usual mark of a shifted distribution
    gain_share    the served main model's split gain share (data/models/bfsp_feature_importance.csv; that model is
                  trained through 2026-11-12, so this is descriptive, not an out-of-sample importance)

Writes out/phase2_feature_audit.csv (one row a feature) and prints the family summary and the flags: features
empty in a year, features missing on more than 90% of rows, features with PSI above 0.25 in any year, features with
no gain at all. The out-of-sample test of each family (drop it, refit, score against the BFSP) is the next,
registered step.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from model import bfsp_features, blocks, feature_cache  # noqa: E402

FROM, HOLDOUT = "2021-01-01", "2026-04-01"
OUT = "out/phase2_feature_audit.csv"
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)


def say(*a):
    print(*a, flush=True)


meta = json.load(open("data/models/bfsp_model_meta.json"))
served = list(meta["feature_cols"])
served_blocks = list(meta["drop_in_blocks"])
say(f"served: {len(served)} features, {len(served_blocks)} blocks")

# ---- the family of each feature
family = {}
for name in served_blocks:
    for c in blocks.load(name).FEATURES:
        family.setdefault(c, f"block:{name}")
lists = [(k, v) for k, v in vars(bfsp_features).items()
         if k.endswith("_FEATURES") and isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v)]
for k, v in sorted(lists, key=lambda kv: len(kv[1])):          # the most specific list wins over a catch-all
    for c in v:
        family.setdefault(c, k.replace("_FEATURES", "").lower())
fam = pd.Series({c: family.get(c, "other") for c in served})
say(f"families: {fam.nunique()}; features with no family: {(fam == 'other').sum()}")

# ---- the matrix, development window only, with the served blocks built as training builds them
mat = None
for start in ("2018-01-01", "2021-01-01"):
    key = feature_cache.cache_key("horse_racing.db", start)
    hit = feature_cache.load(".feature_cache", key)
    if hit is not None:
        mat = hit[0]
        say(f"matrix: cache key {key} (from {start}), {len(mat):,} rows, {mat.shape[1]} columns")
        break
if mat is None:
    raise SystemExit("no cached feature matrix for this code and data: no data found")
d = pd.to_datetime(mat["race_date"])
before = len(mat)
mat = mat.loc[d < pd.Timestamp(HOLDOUT)].reset_index(drop=True)
say(f"rows dropped at the locked holdout (>= {HOLDOUT}): {before - len(mat):,}; kept {len(mat):,}")
mat, added = blocks.attach_as_trained(mat, served_blocks)
say(f"blocks built: {len(added)} features")
d = pd.to_datetime(mat["race_date"])
keep = (d >= pd.Timestamp(FROM)).to_numpy()
year = d[keep].dt.year.to_numpy()
absent = [c for c in served if c not in mat.columns]
say(f"served features not in the frame: {len(absent)} {absent[:10]}")
X = mat.loc[keep, [c for c in served if c in mat.columns]]
del mat
say(f"audit rows: {len(X):,}, {pd.Series(year).value_counts().sort_index().to_dict()}")

# ---- missingness, PSI by year, gain
years = sorted(set(year))
ref_year = years[0]


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    r_ok, c_ok = np.isfinite(ref), np.isfinite(cur)
    edges = np.unique(np.quantile(ref[r_ok], np.linspace(0, 1, 11))) if r_ok.any() else np.array([])
    def dist(v, ok):
        out = [np.mean(~ok)]
        if len(edges) > 1:
            b = np.clip(np.searchsorted(edges, v[ok], side="right") - 1, 0, len(edges) - 2)
            out += list(np.bincount(b, minlength=len(edges) - 1) / max(len(v), 1))
        return np.array(out) + 1e-4
    p, q = dist(ref, r_ok), dist(cur, c_ok)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


gain = pd.read_csv("data/models/bfsp_feature_importance.csv").set_index("feature")["importance"]
gain_share = (gain / gain.sum()).reindex(served).fillna(0.0)
rows = []
for c in X.columns:
    v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(v)
    row = {"feature": c, "family": fam[c], "missing": 1 - ok.mean(), "gain_share": gain_share[c]}
    ref = v[year == ref_year]
    for y in years:
        cur = v[year == y]
        row[f"missing_{y}"] = 1 - np.isfinite(cur).mean() if len(cur) else np.nan
        if y != ref_year:
            row[f"psi_{y}"] = psi(ref, cur)
    rows.append(row)
a = pd.DataFrame(rows)
psi_cols = [c for c in a.columns if c.startswith("psi_")]
a["psi_max"] = a[psi_cols].max(axis=1)
miss_cols = [f"missing_{y}" for y in years]
a["empty_some_year"] = (a[miss_cols] >= 0.999).any(axis=1)
os.makedirs("out", exist_ok=True)
a.to_csv(OUT, index=False)
say(f"\nwritten {OUT}: {len(a)} features")

# ---- the family summary
g = a.groupby("family").agg(features=("feature", "size"), gain_share=("gain_share", "sum"),
                            missing_mean=("missing", "mean"), psi_max_median=("psi_max", "median"),
                            psi_over_025=("psi_max", lambda s: int((s > 0.25).sum())),
                            no_gain=("gain_share", lambda s: int((s <= 0).sum())))
g = g.sort_values("gain_share", ascending=False)
say("\n== families by the served model's gain share\n" + g.round(4).to_string())

# ---- the flags
say(f"\n== flags (of {len(a)})")
say(f"   no gain at all in the served model: {int((a.gain_share <= 0).sum())}")
say(f"   missing on more than 90% of rows: {int((a.missing > 0.9).sum())}")
say(f"   empty in at least one year: {int(a.empty_some_year.sum())}")
say(f"   PSI above 0.25 in at least one year (against {ref_year}): {int((a.psi_max > 0.25).sum())}")
for title, sel in (("empty in a year", a.empty_some_year), ("PSI over 0.25", a.psi_max > 0.25),
                   ("missing over 90%", a.missing > 0.9)):
    sub = a.loc[sel].sort_values("gain_share", ascending=False)
    say(f"\n-- {title}: {len(sub)} (the 25 with most gain)")
    if len(sub):
        say(sub[["feature", "family", "gain_share", "missing", "psi_max"] + miss_cols].head(25).round(4)
            .to_string(index=False))
say("\n-- the 30 features with most gain")
say(a.sort_values("gain_share", ascending=False)[["feature", "family", "gain_share", "missing", "psi_max"]]
    .head(30).round(4).to_string(index=False))
