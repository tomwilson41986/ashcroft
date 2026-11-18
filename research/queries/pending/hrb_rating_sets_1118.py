"""The owner's HorseRaceBase rating sets against our served model and the Betfair SP (ledger hrb-rating-sets-1118,
registered before the download; read-only).

Raw downloads: s3 sources/hrb_ratings_raw/ratings_<set>_<day>.csv (hrb-ratings.yml), names from sets.json there.
Live days 6 Oct - 17 Nov 2026: the morning predictions the trader used (s3 predictions/<day>.csv, the model served
that day), joined to the result and BSP in race_results by date, track, time and horse. Per set (model/rating_eval):
coverage, standalone concordance and softmax log-loss beside the model's and the BSP's, and the walk-forward stacked
log-loss gain over [ln p_model, ln p_BSP] with its interval at 1 - 0.10/k (Bonferroni over the k sets read), then
the knockoff screen across all sets. The download's result columns (placing, odds) are never read as a rating
(hrb_ratings.parse_csv). Secondary, descriptive: each set's race z-score against the move from the 08:00 book's mid
to the BSP (betfair_live_marks), as a rank correlation and beyond the model's own view (the model's price over the
mid).
"""
import io
import json
import os
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import hrb_ratings as hr  # noqa: E402
from betfair_prices import db_time_to_24h, normalise_horse, normalise_track  # noqa: E402
from model.rating_eval import evaluate_rating_sets, knockoff_screen  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
PREFIX = "sources/hrb_ratings_raw"
D0, D1 = date(2026, 10, 6), date(2026, 11, 17)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)


def obj(key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


keys, tok = [], None
while True:
    r = s3.list_objects_v2(Bucket=bucket, Prefix=PREFIX + "/", **({"ContinuationToken": tok} if tok else {}))
    keys += [o["Key"] for o in r.get("Contents", [])]
    if not r.get("IsTruncated"):
        break
    tok = r["NextContinuationToken"]
names = {}
if f"{PREFIX}/sets.json" in keys:
    names = {int(k): v for k, v in json.loads(obj(f"{PREFIX}/sets.json")).items()}
hr.RATING_SETS.update(names)
files = [k for k in keys if k.rsplit("/", 1)[-1].startswith("ratings_")]
print(f"raw files: {len(files)}; sets named on the account: {len(names)} {names}")

parts, bad = [], []
for k in files:
    fn = k.rsplit("/", 1)[-1]
    sid, day = fn[len("ratings_"):-4].split("_", 1)
    if not (D0.isoformat() <= day <= D1.isoformat()):
        continue
    try:
        parts.append(hr.parse_csv(obj(k).decode("utf-8", "replace"), int(sid), date.fromisoformat(day)))
    except Exception as e:  # noqa: BLE001
        bad.append((fn, str(e)[:60]))
R = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
print(f"parsed rows {len(R):,}; files not parsed {len(bad)} {bad[:5]}")
if R.empty:
    raise SystemExit
cov = R.groupby(["set_id", "set_name"]).agg(days=("race_date", "nunique"), rows=("rating", "size"),
                                             nonnull=("rating", lambda s: s.notna().mean()))
print(cov.to_string())

# ---- the morning predictions and the result
preds = []
d = D0
while d <= D1:
    try:
        p = pd.read_csv(io.BytesIO(obj(f"predictions/{d:%Y-%m-%d}.csv")), dtype={"race_time": str})
        p["race_date"] = d.isoformat()
        preds.append(p)
    except Exception:  # noqa: BLE001
        pass
    d += timedelta(days=1)
P = pd.concat(preds, ignore_index=True)
P["_k"] = (P.race_date + "|" + P.venue.map(normalise_track) + "|" + P.race_time.map(db_time_to_24h) + "|"
           + P.runner_name.map(normalise_horse))
res = pd.read_sql_query("""SELECT race_date, track, race_time, horse_name, bfsp, placing_numerical FROM race_results
                           WHERE race_date BETWEEN ? AND ?""", conn, params=(D0.isoformat(), D1.isoformat()))
res["_k"] = (res.race_date.astype(str) + "|" + res.track.map(normalise_track) + "|" + res.race_time.map(db_time_to_24h)
             + "|" + res.horse_name.map(normalise_horse))
X = P.merge(res.drop_duplicates("_k")[["_k", "bfsp", "placing_numerical"]], on="_k", how="inner")
X["raceid"] = X._k.str.rsplit("|", n=1).str[0]
X["bfsp"] = pd.to_numeric(X.bfsp, errors="coerce")
X = X[(X.bfsp > 1) & X.predicted_win_prob.notna()].copy()
X["won"] = (pd.to_numeric(X.placing_numerical, errors="coerce") == 1).astype(int)
X = X[X.groupby("raceid").won.transform("sum") == 1]
X["p_model"] = X.predicted_win_prob / X.groupby("raceid").predicted_win_prob.transform("sum")
X["p_market"] = (1 / X.bfsp) / X.groupby("raceid").bfsp.transform(lambda s: (1 / s).sum())
print(f"\npredictions joined to a result and BSP: {len(X):,} runners, {X.raceid.nunique():,} races, "
      f"{X.race_date.nunique()} days (of {len(preds)} prediction files)")

R["_k"] = (R.race_date.astype(str) + "|" + R.track.map(normalise_track) + "|" + R.race_time_24.astype(str) + "|"
           + R.horse_norm)
R["col"] = "hrb_" + R.set_name.map(hr.slug)
W = R.pivot_table(index="_k", columns="col", values="rating", aggfunc="first").reset_index()
cols = [c for c in W.columns if c.startswith("hrb_")]
X = X.merge(W, on="_k", how="left")
k = len(cols)
level = 1 - 0.10 / max(k, 1)
print(f"rating sets read: k = {k}; the stacked gain's interval at {level:.4f} (Bonferroni)")
print("coverage on the joined runners:", X[cols].notna().mean().round(3).to_dict())

ev = evaluate_rating_sets(X, cols, level=level)
ev["leak_flag"] = ev.get("softmax_logloss") < ev.get("market_logloss_same_rows")
ev["passes_interval"] = ev.get("stack_gain_lo90") > 0
print("\n=== per set (stack_gain_lo90/hi90 at the Bonferroni level above) ===")
show = [c for c in ["rating", "coverage", "n", "races", "concordance", "softmax_logloss", "model_logloss_same_rows",
                    "market_logloss_same_rows", "stack_gain_logloss", "stack_gain_lo90", "stack_gain_hi90",
                    "stack_gain_bss", "leak_flag", "passes_interval"] if c in ev.columns]
print(ev[show].round(5).to_string(index=False))
if k > 1:
    print("\n=== knockoff screen across all sets (race z-scores) with ln p_model and ln p_BSP always in ===")
    print(knockoff_screen(X, cols).round(4).to_string(index=False))

# ---- secondary: the move from the 08:00 mid to the BSP
mk = pd.read_sql_query("""SELECT m.race_date, m.market_id, m.selection_id, m.back1, m.lay1, m.bsp, k.venue, k.runner_name,
                                 k.market_start_utc FROM betfair_live_marks m JOIN betfair_live_markets k
                          ON k.market_id = m.market_id AND k.selection_id = m.selection_id
                          WHERE m.mark = '08:00' AND m.race_date BETWEEN ? AND ? AND k.market_type = 'WIN'""", conn,
                       params=(D0.isoformat(), D1.isoformat()))
if mk.empty:
    print("\nno 08:00 marks: the secondary reading is not made")
else:
    from betfair_prices import utc_iso_to_uk_hhmm
    mk["mid"] = np.sqrt(pd.to_numeric(mk.back1, errors="coerce") * pd.to_numeric(mk.lay1, errors="coerce"))
    mk["_k"] = (mk.race_date + "|" + mk.venue.map(normalise_track) + "|" + mk.market_start_utc.map(utc_iso_to_uk_hhmm)
                + "|" + mk.runner_name.map(normalise_horse))
    Y = X.merge(mk.drop_duplicates("_k")[["_k", "mid"]], on="_k", how="inner")
    Y = Y[(Y.mid > 1) & (Y.bfsp > 1)].copy()
    Y["move"] = np.log(Y.mid / Y.bfsp)                    # above zero: shortened from 08:00 to the off
    Y["move"] -= Y.groupby("raceid").move.transform("mean")
    Y["model_view"] = np.log(Y.p_model * Y.mid)           # the model's price over the mid, in logs
    Y["model_view"] -= Y.groupby("raceid").model_view.transform("mean")
    print(f"\n=== secondary: the 08:00 mid to the BSP, {len(Y):,} runners in {Y.raceid.nunique():,} races ===")
    rows = []
    for c in cols:
        z = Y[c]
        z = (z - Y.groupby("raceid")[c].transform("mean")) / Y.groupby("raceid")[c].transform("std")
        ok = z.notna() & np.isfinite(z)
        if ok.sum() < 300:
            continue
        a = Y.loc[ok]
        A = np.column_stack([np.ones(ok.sum()), a.model_view.values, z[ok].values])
        beta, *_ = np.linalg.lstsq(A, a.move.values, rcond=None)
        res_ = a.move.values - A @ beta
        cl = pd.Series(A[:, 2] * res_).groupby(a.raceid.values).sum()
        XtX = np.linalg.inv(A.T @ A)
        se = float(np.sqrt(XtX[2, 2] ** 2 * (cl ** 2).sum()))
        rows.append({"rating": c, "n": int(ok.sum()), "rank_corr_move": float(z[ok].rank().corr(a.move.rank())),
                     "slope_per_sd_beyond_model": float(beta[2]), "se": se, "t": float(beta[2] / se) if se else np.nan})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
