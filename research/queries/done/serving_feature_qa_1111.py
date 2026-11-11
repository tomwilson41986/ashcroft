"""QA of the served model on real cards, feature by feature (the owner, 11 Nov: "a full review and QA of our model";
read-only).

The stall error was found by comparing the card's inputs with the results table's. This is the same test one level
down, on every served feature: for two recent days the served model's features are built twice by the live code
(prepare_and_predict), with the history before the day:

  served   the real 06:00 card (S3 racecards/<day>_*.csv after declared_runners), as the 04:00 job builds it from
           12 Nov (CARD_AS_TABLE=stall)
  perfect  the day's race_results rows with every outcome blanked and no fill: the same code with the card the model
           was trained on (the final field, the final going, the table's own fields)

On the runners both hold: each feature's share of runners that differ, the mean absolute difference in units of the
feature's own spread, and that times the feature's share of the main member's gain (a rough ranking of where the
served price can be hurt); the top features, and the same summed by feature family (prefix). Then the prices: the
error against the BSP served and perfect, each race centred, paired by race.
"""
import io
import os
import sqlite3
import sys
import time
from datetime import date

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 200)
DAYS = ["2026-11-07", "2026-11-09"]
OUTCOME_COLS = ["place", "distbt", "odds", "fav", "comptime", "comptime_numeric", "total_dst_bt", "placing_numerical",
                "bfsp", "bfsp_place", "comment", "horse_prizewin"]
T0 = time.time()
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"


def key(x):
    return (x["track"].map(normalise_track) + "|" + x["race_time"].map(race_time_to_24h).astype(str) + "|"
            + x["horse_name"].map(normalise_horse))


from daily_predictions import declared_runners  # noqa: E402
from model import card_enrich  # noqa: E402
from predict_bfsp_today import load_bfsp_model, load_historical, prepare_and_predict  # noqa: E402

model, feature_cols, vocab = load_bfsp_model("data/models")
hist = load_historical("horse_racing.db", start_date="2018-01-01")
hist["race_date"] = pd.to_datetime(hist["race_date"])
gain = pd.read_csv("data/models/bfsp_feature_importance.csv").set_index("feature").importance
gain = gain / gain.sum()
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
frames = {}
for dd in DAYS:
    td = date.fromisoformat(dd)
    before = hist[hist.race_date.dt.date < td].copy()
    truth = hist[hist.race_date.dt.date == td].copy()
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{dd}_").get("Contents", []))
    card = declared_runners(pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=keys[0])["Body"].read())), quiet=True)
    os.environ["CARD_AS_TABLE"] = "stall"
    t = time.time()
    a = prepare_and_predict(before.copy(), card.copy(), model, feature_cols, td, vocab)
    print(f"{dd} served ({keys[0]}): {len(a)} runners in {(time.time() - t) / 60:.1f} min", flush=True)
    perfect = truth.copy()
    for c in OUTCOME_COLS:
        if c in perfect.columns:
            perfect[c] = np.nan
    real = card_enrich.enrich_card
    card_enrich.enrich_card = lambda c, h, as_table=None: c
    t = time.time()
    try:
        b = prepare_and_predict(before.copy(), perfect, model, feature_cols, td, vocab)
    finally:
        card_enrich.enrich_card = real
    print(f"{dd} perfect: {len(b)} runners in {(time.time() - t) / 60:.1f} min", flush=True)
    a, b = a.assign(_k=dd + "|" + key(a)), b.assign(_k=dd + "|" + key(b))
    frames[dd] = (a.drop_duplicates("_k").set_index("_k"), b.drop_duplicates("_k").set_index("_k"))
os.environ.pop("CARD_AS_TABLE", None)

A = pd.concat([f[0] for f in frames.values()])
B = pd.concat([f[1] for f in frames.values()])
both = A.index.intersection(B.index)
print(f"\nrunners: served {len(A)}, perfect {len(B)}, both {len(both)}")
A, B = A.loc[both], B.loc[both]
rows = []
for c in feature_cols:
    x = pd.to_numeric(A[c], errors="coerce").to_numpy(float)
    y = pd.to_numeric(B[c], errors="coerce").to_numpy(float)
    nan_x, nan_y = np.isnan(x), np.isnan(y)
    same = (nan_x & nan_y) | (np.abs(x - y) <= 1e-9 + 1e-6 * np.abs(y))
    sd = np.nanstd(y)
    d = np.abs(np.where(nan_x | nan_y, np.nan, x - y))
    rows.append({"feature": c, "differ": 1 - same.mean(), "missing_served_only": (nan_x & ~nan_y).mean(),
                 "missing_perfect_only": (~nan_x & nan_y).mean(),
                 "mean_abs_sd": float(np.nanmean(d) / sd) if sd > 0 and np.isfinite(np.nanmean(d)) else np.nan,
                 "gain": float(gain.get(c, 0.0))})
f = pd.DataFrame(rows)
f["impact"] = f.gain * (f.mean_abs_sd.fillna(0) + f.missing_served_only + f.missing_perfect_only)
print(f"\nfeatures {len(f)}: identical on every runner {int((f.differ == 0).sum())}; differ on over half the runners "
      f"{int((f.differ > 0.5).sum())}; missing only when served (over 5% of runners) {int((f.missing_served_only > 0.05).sum())}")
print("\ntop 45 by impact (gain x difference):")
print(f.sort_values("impact", ascending=False).head(45).round(4).to_string(index=False))
print("\nfeatures missing only when served, over 5% of runners:")
print(f[f.missing_served_only > 0.05].sort_values("missing_served_only", ascending=False).head(40).round(4).to_string(index=False))


def family(c):
    for p in ("rr_", "rw_", "mr_", "fw_", "fv_", "tf_", "kal_", "hca_", "it_", "fr_", "cw_", "dm_", "h2h_", "fl_",
              "bk_", "ex_", "tv_", "sec_", "td_", "draw", "pace", "trainer", "jockey", "sire", "dam"):
        if c.startswith(p):
            return p
    return "r*" if c.startswith("r") and len(c) > 1 and c[1].isupper() else "other"


f["family"] = f.feature.map(family)
g = f.groupby("family").agg(n=("feature", "size"), gain=("gain", "sum"), differ=("differ", "mean"),
                            mean_abs_sd=("mean_abs_sd", "mean"), impact=("impact", "sum")).sort_values("impact", ascending=False)
print("\nby family:")
print(g.round(4).to_string())

# the prices
bsp = pd.to_numeric(hist.loc[hist.race_date.dt.strftime("%Y-%m-%d").isin(DAYS), "bfsp"], errors="coerce")
tr = hist.loc[bsp.index]
tr = tr.assign(_k=tr.race_date.dt.strftime("%Y-%m-%d") + "|" + key(tr), bsp=bsp).drop_duplicates("_k").set_index("_k")
r = pd.DataFrame({"served": A.predicted_bfsp, "perfect": B.predicted_bfsp}).join(tr[["bsp"]], how="inner")
r = r[(r.bsp > 1) & r.served.notna() & r.perfect.notna()]
r["race"] = r.index.str.rsplit("|", n=1).str[0]
lb = np.log(r.bsp)
for c in ("served", "perfect"):
    e = np.log(r[c]) - lb
    r[f"e_{c}"] = (e - e.groupby(r.race).transform("mean")).abs()
per = (r.e_served - r.e_perfect).groupby(r.race).mean()
print(f"\nprices against the BSP (each race centred): {len(r)} runners, {len(per)} races; served {r.e_served.mean():.4f}, "
      f"perfect {r.e_perfect.mean():.4f}, gap {per.mean():+.4f} (se {per.std() / np.sqrt(len(per)):.4f})")
mv = np.log(r.served / r.perfect).abs()
print(f"served against perfect price: median |log ratio| {mv.median():.4f}, 90th pct {mv.quantile(0.9):.4f}, max {mv.max():.3f}")
print(f"\ndone in {(time.time() - T0) / 60:.0f} min")
