"""Is the greyhound stream recorder (greyhound/stream_record.py) working? Read-only: its status file and the parts
uploaded to S3 today, and the first lines of the first part (the connection, the authentication reply, a market
change)."""
import gzip
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from betfair_recorder import PREFIX, bucket, s3_client   # noqa: E402

DAY = "2026-11-05"
s3 = s3_client()
for key in (f"{PREFIX}/{DAY}/status_greyhound_stream.json", f"{PREFIX}/{DAY}/status_greyhound_live.json"):
    try:
        print(key, s3.get_object(Bucket=bucket(), Key=key)["Body"].read().decode()[:1500])
    except Exception as exc:
        print(key, "no data found:", str(exc)[:200])
parts = s3.list_objects_v2(Bucket=bucket(), Prefix=f"betfair_stream/greyhound/{DAY}/").get("Contents", [])
print("parts:", [(p["Key"].rsplit("/", 1)[-1], p["Size"], str(p["LastModified"])) for p in parts] or "no data found")
if parts:
    body = gzip.decompress(s3.get_object(Bucket=bucket(), Key=parts[0]["Key"])["Body"].read()).decode().splitlines()
    print("lines in the first part:", len(body))
    for ln in body[:3]:
        r = json.loads(ln)
        print(r["rt"], r["m"][:300])
    ops = {}
    for ln in body:
        m = json.loads(json.loads(ln)["m"])
        k = m.get("op") + ("/" + m["ct"] if m.get("ct") else "")
        ops[k] = ops.get(k, 0) + 1
    print("message kinds in the first part:", ops)
