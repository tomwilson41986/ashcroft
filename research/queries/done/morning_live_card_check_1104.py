"""What the 06:00 card and record carry for the morning live models (4 Nov 2026): the columns of the latest
racecards/<day>_*.csv and predictions/<day>.csv, the frozen steamer model in S3, and the live ledger's columns."""
import io, json, os
import boto3, pandas as pd

b = os.environ.get("PRED_BUCKET", "ashcroft")
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION", "eu-west-2"))
day = "2026-11-04"
def keys(p):
    out, tok = [], None
    while True:
        kw = {"Bucket": b, "Prefix": p}
        if tok: kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [(o["Key"], o["LastModified"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"): break
        tok = r["NextContinuationToken"]
    return out
cards = keys(f"racecards/{day}")
print("racecards:", cards)
if cards:
    k = sorted(cards, key=lambda x: x[1])[-1][0]
    c = pd.read_csv(io.BytesIO(s3.get_object(Bucket=b, Key=k)["Body"].read()), dtype=str)
    print(k, c.shape); print(list(c.columns))
    for col in ("race_code", "jockeys_claim", "race_type", "race_name", "surface_type", "days_since_lr", "headgear",
                "race_class", "number_of_runners", "official_rating"):
        if col in c: print(col, c[col].value_counts(dropna=False).head(8).to_dict())
        else: print(col, "MISSING")
    print(c.head(3).T.to_string())
p = pd.read_csv(io.BytesIO(s3.get_object(Bucket=b, Key=f"predictions/{day}.csv")["Body"].read()), dtype=str)
print("predictions", p.shape, list(p.columns))
for k in keys("paper/morning_edges/model/"): print("model:", k)
try: print(json.loads(s3.get_object(Bucket=b, Key="paper/morning_edges/model/meta.json")["Body"].read()))
except Exception as e: print("no meta", e)
try:
    l = pd.read_csv(io.BytesIO(s3.get_object(Bucket=b, Key=f"trading/live/{day}/ledger.csv")["Body"].read()), dtype=str)
    print("live ledger", l.shape, list(l.columns)); print(l.event.value_counts().to_dict())
    print(l[l.event == "back"].head(3).T.to_string() if "event" in l else "")
except Exception as e: print("no live ledger", e)
print("time of first live back:", l[l.event=="back"].ts.min() if "event" in l else None)
