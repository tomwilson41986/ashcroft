"""The shadow price's end-to-end check (ledger shadow-m1-1121), read-only.

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

DAY = "2026-11-21"
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
key = [c for c in ("race_date", "race_time", "track", "horse_name") if c in live.columns and c in shadow.columns]
if "horse_name" not in key:
    key = [c for c in ("race_time", "track", "horse") if c in live.columns and c in shadow.columns]
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
