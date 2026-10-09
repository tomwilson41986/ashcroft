"""Has the prediction model got worse? Its 06:00 prices against Betfair, day by day (the owner, 9 Oct: "what has
happened to our model, it was performing so well"; read-only).

The live losses of 7-8 Oct came with changes to the trading rule (the volume model on the live key's feed from 6 Oct
15:25 UK; no day limit and no stop from 7 Oct). The prediction model's files have not changed since 30 Sep. This
scores the model alone, apart from the trader, on every runner of every 06:00 file from 1 Sep:

- its price against the BSP: the error of log(predicted / BSP), the rank correlation, the slope of log BSP on log
  predicted (1 = calibrated in scale);
- its value at the morning price: the runners whose morning weighted average price (Betfair's price files) is at least
  10% / 20% over our price, backed at that price and closed at the BSP, CLV per unit and staked to win;
- by day, and by period: before the gated three (to 28 Sep), the live days 30 Sep - 5 Oct, and 6-8 Oct.

Days whose price file is not loaded yet (8 Oct until tonight) are named and left out.
"""
import os
import sqlite3
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

FROM, TO = date(2026, 9, 1), date(2026, 10, 9)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
os.makedirs("live_q", exist_ok=True)
import boto3  # noqa: E402
import botocore  # noqa: E402

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
d = FROM
while d <= TO:
    try:
        s3.download_file(bucket, f"predictions/{d.isoformat()}.csv", f"live_q/{d.isoformat()}.csv")
    except botocore.exceptions.ClientError:
        print(f"  no 06:00 file for {d}")
    d += timedelta(days=1)

text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
frames = []
for f in sorted(os.listdir("live_q")):
    x = pd.read_csv(os.path.join("live_q", f), dtype=text)
    x["file_day"] = f[:10]
    frames.append(x)
pred = pd.concat(frames, ignore_index=True)
for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if a not in pred.columns and b in pred.columns:
        pred[a] = pred[b]
pred["race_date"] = pred["race_date"].fillna(pred["file_day"])
pred["predicted_bfsp"] = pd.to_numeric(pred["predicted_bfsp"], errors="coerce")

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
bf = pd.read_sql_query(f"""SELECT r.race_date, r.track, r.race_time, r.horse_name, r.placing_numerical,
                                  b.morningwap, b.morning_vol, b.bsp
                           FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
                           WHERE LOWER(b.market_type) = 'win' AND r.race_date >= '{FROM}' AND r.race_date <= '{TO}'""",
                       conn)
print("Betfair price rows by day (the last days):")
print(bf.groupby(bf.race_date.astype(str).str.slice(0, 10)).size().tail(12).to_string())


def key(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str) + "|" + x["horse_name"].map(normalise_horse))


pred["_k"], bf["_k"] = key(pred), key(bf)
m = pred.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]].merge(bf.drop_duplicates("_k"), on="_k")
m = m[(m.bsp > 1) & (m.predicted_bfsp > 1)].copy()
m["day"] = m.race_date.astype(str).str.slice(0, 10)
m["race"] = m.day + "|" + m.track + "|" + m.race_time.astype(str)
m["err"] = np.log(m.predicted_bfsp / m.bsp)
m["lp"], m["lb"] = np.log(m.predicted_bfsp), np.log(m.bsp)
m["period"] = np.select([m.day < "2026-09-29", m.day <= "2026-10-05"], ["a: to 28 Sep", "b: 29 Sep - 5 Oct"],
                        "c: 6 Oct on")
print(f"\njoined: {len(m):,} runners, {m.race.nunique():,} races, {m.day.min()} .. {m.day.max()}")


def price_stats(g):
    slope = np.polyfit(g.lp, g.lb, 1)[0] if len(g) > 10 else np.nan
    return pd.Series({"runners": len(g), "races": g.race.nunique(), "mae_log": g.err.abs().mean(),
                      "bias_log": g.err.mean(), "spearman": g.lp.corr(g.lb, method="spearman"), "slope": slope})


def value_stats(g, edge):
    v = g[(g.morningwap > 1) & (np.log(g.morningwap / g.predicted_bfsp) >= edge) & (g.morning_vol >= 100)]
    if v.empty:
        return pd.Series({f"n{int(edge*100)}": 0, f"clv{int(edge*100)}": np.nan, f"towin_clv{int(edge*100)}": np.nan})
    clv = v.morningwap / v.bsp - 1
    w = 1.0 / (v.morningwap - 1)                                     # staked to win one unit
    return pd.Series({f"n{int(edge*100)}": len(v), f"clv{int(edge*100)}": clv.mean(),
                      f"towin_clv{int(edge*100)}": float((clv * w).sum() / w.sum())})


pd.set_option("display.width", 250)
for by in ("period", "day"):
    out = m.groupby(by).apply(price_stats)
    for e in (0.1, 0.2):
        out = out.join(m.groupby(by).apply(value_stats, edge=e))
    print(f"\n== the model's 06:00 price against Betfair, by {by}")
    print(out.round(4).to_string())

days = sorted(set(pred.file_day) - set(m.day))
print("\n06:00 files with no Betfair price file loaded yet:", ", ".join(days) or "none")
