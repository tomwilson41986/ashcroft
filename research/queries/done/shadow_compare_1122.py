"""gate_stn (served from 22 Nov) against gate_st (its shadow) on 22 Nov's card, runner by runner (ledger shadow-1122).

Copied from shadow_check_1121.py: the same join; here the two are different models, so the spread between them is
the reading, not a pass. Error against the BSP once Betfair's price file for the day is loaded (the next day).

The first reading (shadow-m1-1121), read-only:

data/models/shadow.json named the served stack blend (commit a467f45) for 21 Nov: predict-shadow.yml priced the
morning's kept card with it. Both files priced the same card with the same models on the same code path, so the
shadow must reproduce the trader's file: s3 predictions_shadow/2026-11-21.csv against predictions/2026-11-21.csv,
runner by runner. The pass, fixed before the run: every runner of the trader's file in the shadow's, and the
predicted BFSP identical or within 1e-6 in log on every one.
"""
import io
import os

import boto3
import numpy as np
import pandas as pd

DAY = "2026-11-22"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"


def get(key):
    try:
        return pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()), dtype={"race_time": str})
    except Exception as e:  # noqa: BLE001
        print(f"{key}: no data found ({type(e).__name__})")
        return None


live, shadow = get(f"predictions/{DAY}.csv"), get(f"predictions_shadow/{DAY}.csv")
if live is None or shadow is None:
    raise SystemExit("no data found")
print(f"live {len(live):,} rows, columns {list(live.columns)[:14]}")
print(f"shadow {len(shadow):,} rows, columns {list(shadow.columns)[:14]}")
# the trader's file is pipeline/predict.py's (date, venue, runner_name); the shadow's is predict_bfsp_today's
# (race_date, track, horse_name): join on the time and the runner's name, both trimmed and lower-cased (the first
# reading of this query joined on race_time alone, every runner of a race paired with every other: 21 Nov 10:05)
norm = lambda s: s.fillna("").astype(str).str.strip().str.lower()
live["k_name"], shadow["k_name"] = norm(live["runner_name"]), norm(shadow["horse_name"])
live["k_time"], shadow["k_time"] = norm(live["race_time"]), norm(shadow["race_time"])
key = ["k_time", "k_name"]
for name, f in (("live", live), ("shadow", shadow)):
    dup = int(f.duplicated(key).sum())
    print(f"{name}: {dup} duplicated keys")
    if dup:
        raise SystemExit(f"{name}: the key does not name one runner")
price = next(c for c in ("predicted_bfsp", "bfsp_pred", "predicted_price") if c in live.columns)
print(f"joined on {key}, the price column {price}")
m = live.merge(shadow, on=key, how="outer", suffixes=("_live", "_shadow"), indicator=True)
print(m["_merge"].value_counts().to_dict())
b = m[m["_merge"] == "both"]
d = np.abs(np.log(b[f"{price}_shadow"].astype(float) / b[f"{price}_live"].astype(float)))
print(f"paired {len(b):,}: identical {int((d == 0).sum()):,}, within 1e-6 {int((d <= 1e-6).sum()):,}, largest |log ratio| "
      f"{d.max():.3g}, mean {d.mean():.3g}")
worst = b.assign(d=d).sort_values("d", ascending=False).head(10)
print(worst[key + [f"{price}_live", f"{price}_shadow", "d"]].to_string(index=False))
ok = (m["_merge"] == "left_only").sum() == 0 and len(b) > 0 and (d <= 1e-6).all()
print(f"\nverdict: {'PASS' if ok else 'FAIL'} (every runner of the trader's file in the shadow's, the price within 1e-6 in log)")

print("\nthe spread between the two blends (log of shadow/live), by the live blend's rank in the race:")
b = b.assign(lr=np.log(b[f"{price}_shadow"].astype(float) / b[f"{price}_live"].astype(float)))
b["rank"] = b.groupby("k_time")[f"{price}_live"].rank(method="first")
b["band"] = pd.cut(b["rank"], [0, 1, 2, 3, 7, 99], labels=["1", "2", "3", "4-7", "8+"])
print(b.groupby("band", observed=True)["lr"].agg(["count", "mean", lambda x: x.abs().mean()]).round(4).to_string())
for m in ("db_modified", "live_modified"):
    if m in shadow.columns:
        print(m, shadow[m].iloc[0])
