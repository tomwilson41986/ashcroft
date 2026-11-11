"""Does reading the 06:00 card as the results table records it help the forecast? (the BFSP plan, 11 Nov; read-only)

card-inputs-swap-1110 found two card fields the model reads differently live from training: the stall (the card scrape
fills it from the saddlecloth number over jumps, where the table has 0; +0.0103, se 0.0056) and the weight (the card's
is before the jockey's claim, the table's after; +0.0025, se 0.0024). model/card_enrich.py card_as_table puts both as
the table has them, behind CARD_AS_TABLE (off live).

Here the served model re-prices recent mornings from the card the 04:00 UTC run fetched (S3 racecards/, after
declared_runners), once as served (flag off) and once with the card read as the table (flag on), each day both arms
before the next day, newest first while time allows. Each priced against the BSP over the runners that ran, each race
renormalised; the change paired by race, overall, over jumps and over the Flat.
"""
import io
import os
import sqlite3
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

DAYS = ["2026-11-10", "2026-11-09", "2026-11-08", "2026-11-07", "2026-11-06", "2026-11-05", "2026-11-04"]
BUDGET_MIN = 295
T0 = time.time()
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
import boto3  # noqa: E402

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")


def key(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str) + "|" + x["horse_name"].map(normalise_horse))


from daily_predictions import declared_runners  # noqa: E402
from model.race_shape import race_code  # noqa: E402
from predict_bfsp_today import load_bfsp_model, load_historical, prepare_and_predict  # noqa: E402

model, feature_cols, vocab = load_bfsp_model("data/models")
hist = load_historical("horse_racing.db", start_date="2018-01-01")
hist["race_date"] = pd.to_datetime(hist["race_date"])
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
res = pd.read_sql_query(f"SELECT race_date, track, race_time, horse_name, bfsp, race_type, surface_type FROM race_results "
                        f"WHERE race_date >= '{min(DAYS)}' AND race_date <= '{max(DAYS)}'", conn)
res["bfsp"] = pd.to_numeric(res["bfsp"], errors="coerce")
res = res[res.bfsp > 1].copy()
res["_k"] = key(res)
res["code"] = race_code(res)
res = res.drop_duplicates("_k").set_index("_k")

out = []
for dd in DAYS:
    if (time.time() - T0) / 60 > BUDGET_MIN - 55:
        print(f"  time budget: stopped before {dd}")
        break
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{dd}_").get("Contents", []))
    if not keys:
        print(f"  {dd}: no card")
        continue
    card = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=keys[0])["Body"].read()))
    card = declared_runners(card, quiet=True)
    td = date.fromisoformat(dd)
    before = hist[hist.race_date.dt.date < td].copy()
    for arm, flag in (("served", "0"), ("as_table", "1")):
        os.environ["CARD_AS_TABLE"] = flag
        t = time.time()
        p = prepare_and_predict(before, card.copy(), model, feature_cols, td, vocab)
        p = p.assign(_k=key(p), arm=arm, day=dd)[["_k", "arm", "day", "predicted_bfsp"]]
        out.append(p)
        print(f"  {dd} {arm} ({keys[0]}): {len(p):,} runners in {(time.time() - t) / 60:.1f} min", flush=True)
os.environ.pop("CARD_AS_TABLE", None)

r = pd.concat(out).pivot_table(index="_k", columns="arm", values="predicted_bfsp", aggfunc="last").dropna()
r = r.join(res[["bfsp", "code"]], how="inner")
r["race"] = r.index.str.rsplit("|", n=1).str[0]
pb = (1 / r.bfsp) / (1 / r.bfsp).groupby(r.race).transform("sum")
for arm in ("served", "as_table"):
    pn = (1 / r[arm]) / (1 / r[arm]).groupby(r.race).transform("sum")
    r[f"err_{arm}"] = np.abs(np.log(pn / pb))
r["d"] = r.err_as_table - r.err_served
r["jumps"] = r.code.isin(["hurdle", "chase", "nhflat"])
print(f"\n{len(r):,} runners that ran, in {r.race.nunique():,} races, {r.index.str.slice(0, 10).nunique()} days")
for name, g in (("all", r), ("jumps", r[r.jumps]), ("flat and aw", r[~r.jumps])):
    per = g.groupby("race").d.mean()
    print(f"  {name:12s} served {g.err_served.mean():.4f}  as table {g.err_as_table.mean():.4f}  change {per.mean():+.4f} "
          f"(se {per.std() / np.sqrt(max(len(per), 1)):.4f}, {len(per)} races)")
print("\nby day (change paired by race):")
r["day"] = r.index.str.slice(0, 10)
print(r.groupby("day").apply(lambda g: g.groupby("race").d.mean().mean()).round(4).to_string())
print(f"\ndone in {(time.time() - T0) / 60:.0f} min")
