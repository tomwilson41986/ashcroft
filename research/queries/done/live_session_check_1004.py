"""4 Oct.s live session so far (a Sunday; started by hand at 08:4x UTC after the 06:50 schedule did not fire), from the ledger it keeps in S3
after every order (read-only), with the predictions file the session reads."""

from __future__ import annotations

import io
import os

import boto3
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
DAY = "2026-10-04"

for prefix in (f"trading/live/{DAY}/", f"betfair_live/{DAY}/"):
    page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for o in page.get("Contents", []) or []:
        print(f"  {o['Key']}: {o['Size']:,} bytes, {o['LastModified']:%H:%M:%S} UTC")

obj = s3.get_object(Bucket=bucket, Key=f"trading/live/{DAY}/ledger.csv")
d = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype={"market_id": str}, low_memory=False)
d["ts"] = pd.to_datetime(d["ts"], utc=True)
uk = lambda t: t.tz_convert("Europe/London").strftime("%H:%M")  # noqa: E731
print(f"\nledger: {len(d)} rows, {uk(d.ts.min())} to {uk(d.ts.max())} UK; {d.event.value_counts().to_dict()}")
b = d[d.event == "back"].copy()
b["matched"] = pd.to_numeric(b["matched"], errors="coerce").fillna(0.0)
m = b[b.matched > 0]
print(f"backs sent {len(b)}, matched {len(m)} on {m[['market_id', 'selection_id']].drop_duplicates().shape[0]} horses in "
      f"{m.market_id.nunique()} races, GBP{m.matched.sum():,.2f}; first matched {uk(m.ts.min()) if len(m) else '-'} UK")
print("back statuses:", b["status"].value_counts(dropna=False).to_dict())
print("back errors:", b["error"].value_counts(dropna=False).head(10).to_dict())
lays = d[d.event == "trade_out"]
print(f"SP lays {len(lays)}: statuses {lays['status'].value_counts(dropna=False).to_dict()}, "
      f"errors {lays['error'].value_counts(dropna=False).head(5).to_dict()}")
sk = d[d.event == "skip"]
print("skips by reason:", sk["reason"].astype(str).str.slice(0, 60).value_counts().head(12).to_dict())
if len(m):
    cols = ["ts", "venue", "off", "minutes_to_off", "runner", "best_back", "target", "asked", "matched", "avg_price"]
    print("\nfirst matched backs:")
    print(m[cols].head(8).assign(ts=lambda x: x.ts.map(uk)).to_string(index=False))
    stake_by_horse = m.groupby(["market_id", "selection_id"]).matched.sum()
    print(f"\nlargest stake on a horse GBP{stake_by_horse.max():.2f} (limit GBP300); minutes to off at entry: "
          f"min {m.minutes_to_off.min():.0f}, median {m.minutes_to_off.median():.0f}")

# the morning's predictions file, the one the session reads (the 06:00 run)
for key in (f"predictions/{DAY}.csv",):
    try:
        h = s3.head_object(Bucket=bucket, Key=key)
        p = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()))
        print(f"\n{key}: {h['ContentLength']:,} bytes, written {h['LastModified']:%H:%M} UTC; {len(p)} runners in "
              f"{p.groupby(['track', 'race_time']).ngroups if {'track', 'race_time'} <= set(p.columns) else '?'} races")
    except Exception as exc:
        print(f"\n{key}: {type(exc).__name__}")
