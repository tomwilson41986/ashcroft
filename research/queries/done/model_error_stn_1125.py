"""gate_stn (served from 22 Nov) and gate_st (its shadow) against the BSP, and the stack blend before them (ledger
model-error-stn-1125; read-only, descriptive). The served 06:00 price (S3 predictions/<day>.csv) and the shadow's
(predictions_shadow/<day>.csv) joined to the live record's WIN markets and the final mark's BSP as
done/model_vs_market_1124.py does. Error = mean |log(price / BSP)|; the paired difference shadow - served on the same
runners with a 90% interval by resampling days. By day 15-25 Nov (15-21 Nov the stack blend, 22-25 Nov gate_stn)."""
import io
import os
import sqlite3
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h, utc_iso_to_uk_hhmm  # noqa: E402

pd.set_option("display.width", 220)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
D0, D1 = date(2026, 11, 15), date(2026, 11, 25)
rng = np.random.default_rng(20261125)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)


def load(prefix):
    text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
    out, d = [], D0
    while d <= D1:
        try:
            raw = s3.get_object(Bucket=bucket, Key=f"{prefix}/{d.isoformat()}.csv")["Body"].read()
        except Exception:  # noqa: BLE001
            raw = None
        if raw is not None:
            x = pd.read_csv(io.BytesIO(raw), dtype=text, low_memory=False)
            x["file_day"] = d.isoformat()
            out.append(x)
        d += timedelta(days=1)
    if not out:
        return pd.DataFrame(columns=["_k", "p"])
    p = pd.concat(out, ignore_index=True)
    for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
        if a not in p.columns and b in p.columns:
            p[a] = p[b]
    p["race_date"] = p["race_date"].fillna(p["file_day"]).astype(str).str.slice(0, 10)
    p["p"] = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    p["_k"] = (p.race_date + "|" + p.track.map(normalise_track) + "|" + p.race_time.map(race_time_to_24h).astype(str)
               + "|" + p.horse_name.map(normalise_horse))
    print(f"{prefix}: {p.file_day.nunique()} day files, {len(p):,} rows")
    return p.drop_duplicates("_k", keep="last")[["_k", "p"]]


mk = pd.read_sql_query(f"""SELECT race_date, market_id, selection_id, venue, market_start_utc, runner_name, race_type
                           FROM betfair_live_markets WHERE race_date >= '{D0}' AND race_date <= '{D1}'
                           AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
mk = mk.drop_duplicates(["market_id", "selection_id"])
mk["_k"] = (mk.race_date + "|" + mk.venue.map(normalise_track) + "|" + mk.market_start_utc.map(utc_iso_to_uk_hhmm).astype(str)
            + "|" + mk.runner_name.map(normalise_horse))
mk = mk.merge(load("predictions").rename(columns={"p": "served"}), on="_k", how="left")
mk = mk.merge(load("predictions_shadow").rename(columns={"p": "shadow"}), on="_k", how="left")
fin = pd.read_sql_query(f"""SELECT market_id, selection_id, bsp FROM betfair_live_marks WHERE race_date >= '{D0}'
                            AND race_date <= '{D1}' AND mark = 'final'""", conn)
fin["selection_id"] = pd.to_numeric(fin.selection_id, errors="coerce")
mk = mk.merge(fin.drop_duplicates(["market_id", "selection_id"]), on=["market_id", "selection_id"], how="left")
ok = mk[(mk.bsp > 1) & (mk.served > 1)].copy()
ok["e_served"] = np.abs(np.log(ok.served / ok.bsp))
print(f"\nrunners with a served price and a BSP: {len(ok):,}; days {sorted(ok.race_date.unique())}")
t = ok.groupby("race_date").agg(runners=("e_served", "size"), served=("e_served", "mean"),
                                bias=("served", lambda s: float(np.mean(np.log(s / ok.loc[s.index, "bsp"])))))
print("\n== the served price against the BSP by day (15-21 Nov the stack blend; from 22 Nov gate_stn)")
print(t.round(4).to_string())
for name, lo, hi in (("stack 15-21 Nov", "2026-11-15", "2026-11-21"), ("gate_stn 22-25 Nov", "2026-11-22", "2026-11-25")):
    x = ok[(ok.race_date >= lo) & (ok.race_date <= hi)]
    print(f"   {name}: {len(x):,} runners, error {x.e_served.mean():.4f}")
pr = ok[ok.shadow > 1].copy()
if len(pr):
    pr["e_shadow"] = np.abs(np.log(pr.shadow / pr.bsp))
    pr["d"] = pr.e_shadow - pr.e_served
    g = pr.groupby("race_date").agg(n=("d", "size"), d_sum=("d", "sum"), served=("e_served", "mean"),
                                     shadow=("e_shadow", "mean"))
    num, den = g.d_sum.to_numpy(), g.n.to_numpy(float)
    est = num.sum() / den.sum()
    if len(g) >= 3:
        idx = rng.integers(0, len(g), size=(5000, len(g)))
        r = num[idx].sum(1) / den[idx].sum(1)
        lo, hi = np.quantile(r, [0.05, 0.95])
    else:
        lo = hi = np.nan
    print("\n== gate_st (shadow) against gate_stn (served), paired on the same runners")
    print(g.round(4).to_string())
    print(f"   paired: {len(pr):,} runners over {len(g)} days; served {pr.e_served.mean():.4f}, shadow "
          f"{pr.e_shadow.mean():.4f}; shadow - served {est:+.4f} (90% by day {lo:+.4f} to {hi:+.4f})")
else:
    print("no shadow prices joined")
