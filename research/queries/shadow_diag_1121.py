"""Why the shadow check (shadow-m1-1121) read FAIL: 153 of 459 runners identical, the rest up to 0.17 apart in log.

A diagnosis, not a new test: when each file was written (S3 LastModified, the object's versions), the live file's
predicted_at, the morning cards kept, and per race whether the two differ by one ratio (a race re-normalised: a
non-runner's probability shared out) or by runner (different features or models).
"""
import io
import os

import boto3
import numpy as np
import pandas as pd

DAY = "2026-11-21"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
B = os.environ.get("PRED_BUCKET") or "ashcroft"

for k in (f"predictions/{DAY}.csv", f"predictions_shadow/{DAY}.csv"):
    h = s3.head_object(Bucket=B, Key=k)
    print(f"{k}: LastModified {h['LastModified']}, {h['ContentLength']:,} bytes")
    try:
        vs = s3.list_object_versions(Bucket=B, Prefix=k).get("Versions", [])
        print("  versions:", [(str(v["LastModified"]), v["Size"]) for v in vs][:10])
    except Exception as e:  # noqa: BLE001
        print("  versions: not readable", type(e).__name__)
cards = s3.list_objects_v2(Bucket=B, Prefix=f"racecards/{DAY}_").get("Contents", [])
print("cards kept:", [(c["Key"], str(c["LastModified"]), c["Size"]) for c in cards])
for pre in ("predictions_now/", f"predictions/{DAY}", "non_runners/"):
    got = s3.list_objects_v2(Bucket=B, Prefix=pre, MaxKeys=1000).get("Contents", [])
    got = [g for g in got if DAY in g["Key"]]
    print(f"{pre}: {[(g['Key'], str(g['LastModified'])) for g in got][:10]}")


def get(k):
    return pd.read_csv(io.BytesIO(s3.get_object(Bucket=B, Key=k)["Body"].read()), dtype={"race_time": str})


live, sh = get(f"predictions/{DAY}.csv"), get(f"predictions_shadow/{DAY}.csv")
print("live predicted_at:", live["predicted_at"].value_counts().head(5).to_dict() if "predicted_at" in live else "none")
norm = lambda s: s.fillna("").astype(str).str.strip().str.lower()
for f, n in ((live, "runner_name"), (sh, "horse_name")):
    f["k_time"], f["k_name"] = norm(f["race_time"]), norm(f[n])
m = live.merge(sh, on=["k_time", "k_name"], suffixes=("_l", "_s"))
m["d"] = np.log(m["predicted_bfsp_s"] / m["predicted_bfsp_l"])
m["track"] = m["track"] if "track" in m else m["venue"]
rows = []
for (t, tr), g in m.groupby(["k_time", "track"]):
    rows.append({"time": t, "track": tr, "n": len(g), "same": int((g["d"].abs() <= 1e-6).sum()),
                 "d_min": round(g["d"].min(), 4), "d_max": round(g["d"].max(), 4),
                 "distinct_d": g["d"].round(5).nunique(),
                 "sum_p_live": round((1 / g["predicted_bfsp_l"]).sum(), 3),
                 "sum_p_shadow": round((1 / g["predicted_bfsp_s"]).sum(), 3)})
r = pd.DataFrame(rows).sort_values(["track", "time"])
print(r.to_string(index=False))
print("\nraces all identical:", int((r["same"] == r["n"]).sum()), "of", len(r))
for c in ("predicted_win_prob",):
    if f"{c}_l" in m and f"{c}_s" in m:
        print(c, "max abs diff", float((m[f"{c}_l"] - m[f"{c}_s"]).abs().max()))
