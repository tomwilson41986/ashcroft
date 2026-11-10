"""Which card input costs the 06:00 forecast its 0.008? (the BFSP plan of 10 Nov, step 3 follow-up; read-only)

card_vs_results_inputs_1110 found the 06:00 card unlike the results table the model was trained on in: weight (22% of
runners, by the claim: the card 135 where the table has 130), stall (48%: the card numbers jumps runners, the table has
none), race class (27% blank on the card), going (27%, changed in the day) and field size (non-runners). The live path
already strips the claim from the jockey's name (model/card_enrich.py), so the jockey key matches.

A: how far each explains itself: the weight difference against the claim printed with the jockey, the stall
   difference by code, the blank class by country and race type.
B: the served model re-prices the same runners (those that ran) from the results table's rows, then again with one
   input at a time taken from the 06:00 card instead (weight, stall, class, going), on the three latest days. The
   paired change in error says what each costs.
"""
import os
import re
import sqlite3
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

FROM, TO = date(2026, 10, 24), date(2026, 11, 8)
REPRICE_DAYS = ["2026-11-08", "2026-11-07", "2026-11-06"]
SWAPS = {"weight": ["pounds"], "stall": ["stall"], "class": ["race_class"], "going": ["going_description"]}
BUDGET_MIN = 290
T0 = time.time()
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
os.makedirs("live_cards", exist_ok=True)
import boto3  # noqa: E402

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
d = FROM
while d <= TO:
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{d.isoformat()}_")
                  .get("Contents", []))
    if keys:
        s3.download_file(bucket, keys[0], f"live_cards/{d.isoformat()}.csv")
    d += timedelta(days=1)
frames = []
for f in sorted(os.listdir("live_cards")):
    x = pd.read_csv(os.path.join("live_cards", f), dtype=str)
    x["race_date"] = x.get("race_date", pd.Series(f[:10], index=x.index)).fillna(f[:10])
    frames.append(x)
card = pd.concat(frames, ignore_index=True)

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
res = pd.read_sql_query(f"SELECT * FROM race_results WHERE race_date >= '{FROM}' AND race_date <= '{TO}'", conn)
res["bfsp"] = pd.to_numeric(res["bfsp"], errors="coerce")
res = res[res.bfsp > 1].copy()


def key(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str) + "|" + x["horse_name"].map(normalise_horse))


card["_k"], res["_k"] = key(card), key(res)
card = card.drop_duplicates("_k", keep="last").set_index("_k")
res = res.drop_duplicates("_k", keep="last")
m = res.join(card[["pounds", "stall", "race_class", "going_description", "jockey_name"]], on="_k", rsuffix="_card")
m = m[m.jockey_name_card.notna()].copy()
print(f"{len(m):,} runners on both the card and the results table")

# --- A ---------------------------------------------------------------------------------------------------------------
from model.race_shape import race_code  # noqa: E402

m["code"] = race_code(m)
claim = pd.to_numeric(m.jockey_name_card.astype(str).str.extract(r"\((\d{1,2})\)\s*$", expand=False), errors="coerce")
wd = pd.to_numeric(m.pounds_card, errors="coerce") - pd.to_numeric(m.pounds, errors="coerce")
print("\n== weight: card less table, against the claim printed with the card's jockey")
t = pd.DataFrame({"diff": wd, "claim": claim.fillna(0)})
print(t.groupby("claim")["diff"].describe()[["count", "mean", "min", "max"]].round(2).to_string())
print(f"weight differs and equals the claim: {((wd != 0) & (wd == claim)).mean():.1%} of runners; "
      f"differs otherwise: {((wd != 0) & (wd != claim.fillna(0))).mean():.1%}")
st_card = pd.to_numeric(m.stall_card, errors="coerce")
st_res = pd.to_numeric(m.stall, errors="coerce")
print("\n== stall by code: share of runners with a card stall, share with a table stall (> 0), share differing")
print(pd.DataFrame({"code": m.code, "card": st_card.gt(0), "table": st_res.gt(0),
                    "differ": st_card.fillna(0).ne(st_res.fillna(0))}).groupby("code").mean().round(3).to_string())
blank = m.race_class_card.isna() | m.race_class_card.astype(str).str.strip().isin(["", "nan"])
print("\n== race class blank on the card, by the table's class")
print(pd.crosstab(m.race_class.fillna("(blank)"), blank).to_string())

# --- B ---------------------------------------------------------------------------------------------------------------
from predict_bfsp_today import load_bfsp_model, load_historical, prepare_and_predict  # noqa: E402

model, feature_cols, vocab = load_bfsp_model("data/models")
hist = load_historical("horse_racing.db", start_date="2018-01-01")
hist["race_date"] = pd.to_datetime(hist["race_date"])
hist["_k"] = key(hist)
out = []
for dd in REPRICE_DAYS:
    td = date.fromisoformat(dd)
    before = hist[hist.race_date.dt.date < td].drop(columns="_k").copy()
    day = hist[(hist.race_date.dt.date == td)]
    day = day[(pd.to_numeric(day.bfsp, errors="coerce") > 1) & day._k.isin(card.index)].copy()
    for arm in ["table"] + list(SWAPS):
        if (time.time() - T0) / 60 > BUDGET_MIN:
            print(f"  time budget reached before {dd} {arm}")
            break
        rows = day.copy()
        for c in SWAPS.get(arm, []):
            rows[c] = rows._k.map(card[c]).to_numpy()
        t = time.time()
        p = prepare_and_predict(before, rows.drop(columns="_k"), model, feature_cols, td, vocab)
        p = p.assign(_k=key(p), arm=arm)[["_k", "arm", "predicted_bfsp"]]
        out.append(p)
        print(f"  {dd} {arm}: {len(p):,} runners in {(time.time() - t) / 60:.1f} min", flush=True)

if out:
    r = pd.concat(out).pivot_table(index="_k", columns="arm", values="predicted_bfsp", aggfunc="last")
    r = r.join(res.set_index("_k")["bfsp"]).dropna()
    r["race"] = r.index.str.rsplit("|", n=1).str[0]
    pb = (1 / r.bfsp) / (1 / r.bfsp).groupby(r.race).transform("sum")
    err = {}
    for arm in ["table"] + list(SWAPS):
        if arm in r:
            pn = (1 / r[arm]) / (1 / r[arm]).groupby(r.race).transform("sum")
            err[arm] = np.abs(np.log(pn / pb))
    e = pd.DataFrame(err)
    print(f"\n== B: {len(e):,} runners in {r.race.nunique():,} races, re-priced from the table and with one card input")
    for arm in e.columns:
        dlt = (e[arm] - e["table"]).groupby(r.race).mean()
        print(f"  {arm:8s} error {e[arm].mean():.4f}   change against the table {dlt.mean():+.4f} "
              f"(se {dlt.std() / np.sqrt(len(dlt)):.4f}, {len(dlt)} races)")
print(f"\ndone in {(time.time() - T0) / 60:.0f} min")
