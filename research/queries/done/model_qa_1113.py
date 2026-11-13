"""Model QA, 13 Nov 2026 (the owner: "Do a full QA and review. Are there issues with our model? Results have dropped
in recent days. Is our model worse for National Hunt than Flat racing?"; read-only, descriptive, no rule tested).

1. The served model's 06:00 price against the BSP, by code (flat turf, all-weather, hurdle, chase, bumper) and by
   period since 6 Oct: error of log(predicted / BSP), bias, rank correlation, and beside it the same error for
   Betfair's own morning weighted average price (how hard that code's prices are to call at all).
2. The model's value at the morning price by code: runners whose morning price is 10% or more over ours, backed at
   that price and closed at the BSP.
3. The live trader by code and by day: money matched, CLV at the settled price against the BSP, and the settled
   profit before commission, from its own ledgers.

Every number is from horse_racing.db (betfair_prices joined to race_results) and the S3 files predictions/<day>.csv
and trading/live/<day>/ledger.csv. Days whose Betfair price file is not loaded are named and left out of 1 and 2.
"""
import io
import os
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import botocore
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402
from model.race_shape import race_code  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
FROM, TO = date(2026, 10, 6), date(2026, 11, 13)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name", "market_id")}

frames, d = [], FROM
while d <= TO:
    try:
        raw = s3.get_object(Bucket=bucket, Key=f"predictions/{d.isoformat()}.csv")["Body"].read()
        x = pd.read_csv(io.BytesIO(raw), dtype=text)
        x["file_day"] = d.isoformat()
        frames.append(x)
    except botocore.exceptions.ClientError:
        print(f"no 06:00 file for {d}")
    d += timedelta(days=1)
pred = pd.concat(frames, ignore_index=True)
for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if a not in pred.columns and b in pred.columns:
        pred[a] = pred[b]
pred["race_date"] = pred["race_date"].fillna(pred["file_day"])
pred["predicted_bfsp"] = pd.to_numeric(pred["predicted_bfsp"], errors="coerce")
pred["selection_id"] = pd.to_numeric(pred.get("selection_id"), errors="coerce")

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
bf = pd.read_sql_query(f"""SELECT r.race_date, r.track, r.race_time, r.horse_name, r.placing_numerical, r.race_type,
                                  r.surface_type, b.morningwap, b.bsp
                           FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
                           WHERE LOWER(b.market_type) = 'win' AND r.race_date >= '{FROM}' AND r.race_date <= '{TO}'""",
                       conn)
rr = pd.read_sql_query(f"""SELECT race_date, track, race_time, race_type, surface_type FROM race_results
                           WHERE race_date >= '{FROM}' AND race_date <= '{TO}'""", conn)
print("results rows by day (the last 6):", rr.groupby(rr.race_date.astype(str).str.slice(0, 10)).size().tail(6).to_dict())
print("Betfair price rows by day (the last 6):", bf.groupby(bf.race_date.astype(str).str.slice(0, 10)).size().tail(6).to_dict())


def rkey(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str))


def key(x):
    return rkey(x) + "|" + x["horse_name"].map(normalise_horse)


def period(day):
    return np.select([day < "2026-10-20", day < "2026-11-03", day < "2026-11-10"],
                     ["a: 6-19 Oct", "b: 20 Oct-2 Nov", "c: 3-9 Nov"], "d: 10 Nov on")


CODE2 = {"flat": "Flat turf", "aw": "All-weather", "hurdle": "Hurdle", "chase": "Chase", "nhflat": "Bumper"}
bf["code"] = race_code(bf).map(CODE2)
rr["code"] = race_code(rr).map(CODE2)
pred["_k"], bf["_k"] = key(pred), key(bf)
pred["_r"], rr["_r"] = rkey(pred), rkey(rr)
race_code_of = rr.drop_duplicates("_r").set_index("_r")["code"]

m = pred.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]].merge(bf.drop_duplicates("_k"), on="_k")
m = m[(m.bsp > 1) & (m.predicted_bfsp > 1)].copy()
m["day"] = m.race_date.astype(str).str.slice(0, 10)
m["period"] = period(m.day)
m["jumps"] = np.where(m.code.isin(["Hurdle", "Chase", "Bumper"]), "National Hunt", "Flat and all-weather")
m["err"] = np.log(m.predicted_bfsp / m.bsp)
m["merr"] = np.where(m.morningwap > 1, np.log(m.morningwap / m.bsp), np.nan)
missing = sorted(set(pred.file_day) - set(m.day))
print(f"\njoined: {len(m):,} runners, {m._k.str.rsplit('|', n=1).str[0].nunique():,} races, {m.day.min()} .. {m.day.max()}; "
      f"06:00 files with no Betfair price file loaded yet: {', '.join(missing) or 'none'}")


def score(g):
    w = g[g.merr.notna()]
    return pd.Series({"runners": len(g), "races": g._k.str.rsplit("|", n=1).str[0].nunique(),
                      "model_mae": g.err.abs().mean(), "model_bias": g.err.mean(),
                      "rank_corr": g.predicted_bfsp.corr(g.bsp, method="spearman"),
                      "morning_mae": w.merr.abs().mean(), "model_mae_same_rows": w.err.abs().mean(),
                      "median_bsp": g.bsp.median(), "mean_field": g.groupby(g._k.str.rsplit("|", n=1).str[0]).size().mean()})


print("\n== 1a. the model's 06:00 price against the BSP, National Hunt against the Flat (6 Oct on)")
print(m.groupby("jumps").apply(score).round(4).to_string())
print("\n== 1b. by code")
print(m.groupby("code").apply(score).round(4).to_string())
print("\n== 1c. by period (all codes)")
print(m.groupby("period").apply(score).round(4).to_string())
print("\n== 1d. by period and National Hunt / Flat")
print(m.groupby(["period", "jumps"]).apply(score).round(4).to_string())
print("\n== 1e. by price band and National Hunt / Flat (the same price compared like with like)")
m["band"] = pd.cut(m.bsp, [1, 4, 8, 16, 40, 1001], labels=["under 4", "4-8", "8-16", "16-40", "40+"])
print(m.groupby(["band", "jumps"], observed=True).apply(score)[["runners", "model_mae", "model_bias", "morning_mae",
                                                               "model_mae_same_rows"]].round(4).to_string())
print("\n== 1f. the share of each code in the day's runners, by period")
print((m.groupby(["period", "code"]).size().unstack(fill_value=0).pipe(lambda t: t.div(t.sum(axis=1), axis=0))).round(3).to_string())
print("\n== 1g. the last 14 days, all codes")
print(m[m.day >= "2026-10-30"].groupby("day").apply(score)[["runners", "model_mae", "model_bias", "rank_corr",
                                                           "morning_mae"]].round(4).to_string())

print("\n== 2. value at the morning price: the morning price 10% or more over ours, backed there, closed at the BSP")
v = m[(m.morningwap > 1) & (m.morningwap >= 1.10 * m.predicted_bfsp)].copy()
v["clv"] = v.morningwap / v.bsp - 1
v["to_win"] = 1 / (v.morningwap - 1)


def val(g):
    return pd.Series({"bets": len(g), "clv_per_unit": g.clv.mean(),
                      "clv_staked_to_win": (g.clv * g.to_win).sum() / g.to_win.sum() if len(g) else np.nan})


print(v.groupby("jumps").apply(val).round(4).to_string())
print(v.groupby("code").apply(val).round(4).to_string())
print(v.groupby(["period", "jumps"]).apply(val).round(4).to_string())

print("\n== 3. the live trader, from its own ledgers")
K = ["market_id", "selection_id"]
# a horse runs once a day, so the day and its name give the race (the 06:00 file's Betfair ids are not always filled)
pred["_h"] = pred["horse_name"].map(normalise_horse)
pm = pred.drop_duplicates(["file_day", "_h"], keep="last")[["file_day", "_h", "_r"]]
print("06:00 file columns:", list(pred.columns)[:20], "; selection_id filled on", int(pred.selection_id.notna().sum()), "of", len(pred))
rows = []
d = date(2026, 10, 10)
while d <= TO:
    day = d.isoformat()
    d += timedelta(days=1)
    try:
        raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    except botocore.exceptions.ClientError:
        continue
    L = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ["matched", "avg_price", "bsp", "clv", "selection_id", "pnl", "commission"]:
        if c in L.columns:
            L[c] = pd.to_numeric(L[c], errors="coerce")
    b = L[(L.event == "back") & (L.matched > 0)].copy()
    if b.empty:
        continue
    b["sxp"] = b.matched * b.avg_price
    hm = b.groupby(K).agg(staked=("matched", "sum"), sxp=("sxp", "sum"), fills=("matched", "size")).reset_index()
    hm["matched_price"] = hm.sxp / hm.staked
    st = L[L.event == "settle"].drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price", "pnl"]] \
        .rename(columns={"avg_price": "settled_price"})
    h = hm.merge(st, on=K, how="left")
    h["day"] = day
    h["commission_day"] = L.loc[L.event == "commission", "commission"].sum() if "commission" in L.columns else np.nan
    names = L.dropna(subset=["runner"]).drop_duplicates(K, keep="last")[K + ["runner", "venue", "off"]]
    h = h.merge(names, on=K, how="left")
    h["_h"] = h["runner"].astype(str).map(normalise_horse)
    h = h.merge(pm[pm.file_day == day][["_h", "_r"]], on="_h", how="left")
    rows.append(h)
T = pd.concat(rows, ignore_index=True)
print("ledger sample:", T[["day", "runner", "venue", "off", "_h", "_r"]].head(3).to_dict("records"))
print(f"backed horses with a race from the 06:00 file: {T._r.notna().sum():,} of {len(T):,}; of those with a result row: {T._r.isin(race_code_of.index).sum():,}")
T["code"] = T._r.map(race_code_of).fillna("not matched to a result")
T["jumps"] = np.select([T.code.isin(["Hurdle", "Chase", "Bumper"]), T.code.isin(["Flat turf", "All-weather"])],
                       ["National Hunt", "Flat and all-weather"], "not matched to a result")
T["period"] = period(T.day)
ok = T.result.isin(["WINNER", "LOSER"]) & (T.bsp > 1) & (T.settled_price > 1)
T["clv"] = np.where(ok, T.settled_price / T.bsp - 1, np.nan)
T["locked"] = T.staked * T.clv


def live(g):
    s = g[g.clv.notna()]
    return pd.Series({"horses": len(g), "staked": g.staked.sum(), "clv": s.locked.sum() / s.staked.sum() if len(s) else np.nan,
                      "pnl_before_commission": g.pnl.sum(), "pnl_per_100_staked": 100 * g.pnl.sum() / g.staked.sum(),
                      "winners": (g.result == "WINNER").sum(), "median_price": g.matched_price.median()})


print(f"horses backed {len(T):,} on {T.day.nunique()} days, {T.day.min()} .. {T.day.max()}; "
      f"not matched to a result's code: {(T.jumps == 'not matched to a result').sum():,}")
print("\n== 3a. National Hunt against the Flat, all days")
print(T.groupby("jumps").apply(live).round(4).to_string())
print("\n== 3b. by code, all days")
print(T.groupby("code").apply(live).round(4).to_string())
print("\n== 3c. by period")
print(T.groupby("period").apply(live).round(4).to_string())
print("\n== 3d. by period and National Hunt / Flat")
print(T.groupby(["period", "jumps"]).apply(live).round(4).to_string())
print("\n== 3e. by day: the day's stake, CLV, profit before commission, commission, and the share staked on National Hunt")
dd = T.groupby("day").apply(live)[["horses", "staked", "clv", "pnl_before_commission"]]
dd["commission"] = T.groupby("day").commission_day.first()
dd["nh_share_of_stake"] = T[T.jumps == "National Hunt"].groupby("day").staked.sum().reindex(dd.index).fillna(0) / dd.staked
print(dd.round(4).to_string())
print("\n== 3f. by price band and National Hunt / Flat")
T["band"] = pd.cut(T.matched_price, [1, 4, 8, 16, 40, 1001], labels=["under 4", "4-8", "8-16", "16-40", "40+"])
print(T[T.jumps != "not matched to a result"].groupby(["band", "jumps"], observed=True).apply(live).round(4).to_string())
