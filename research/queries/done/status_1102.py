"""The paper models' standing, 2 Nov 2026 (the owner's ask: keep UK/IE safe, paper-trial the rest; read-only).

Prints: the greyhound forward test's running summary (sources/greyhound/track/summary.json), the football paper
trader's ledger (sources/football/live/paper/), the greyhound live/paper ledgers (sources/greyhound/live/), and the
first evening racing census (betfair_live/<day>/census_racing_*.json)."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def s3():
    import boto3
    return boto3.client("s3", region_name=os.getenv("PRED_REGION", "eu-west-2"))


B = os.getenv("PRED_BUCKET", "ashcroft")


def keys(c, prefix, limit=2000):
    out, tok = [], None
    while True:
        kw = {"Bucket": B, "Prefix": prefix, "MaxKeys": 1000}
        if tok:
            kw["ContinuationToken"] = tok
        r = c.list_objects_v2(**kw)
        out += [(o["Key"], o["Size"], o["LastModified"].isoformat()[:16]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated") or len(out) >= limit:
            return out
        tok = r["NextContinuationToken"]


def show(c, key, n=4000):
    try:
        body = c.get_object(Bucket=B, Key=key)["Body"].read()
        print(f"--- {key} ({len(body)} B)")
        print(body[:n].decode("utf-8", "replace"))
    except Exception as exc:
        print(f"--- {key}: {type(exc).__name__} {str(exc)[:200]}")


def main():
    c = s3()
    print("== 1. greyhound forward test")
    ks = keys(c, "sources/greyhound/track/")
    print(f"{len(ks)} objects; days files: {sum('/days/' in k for k, _, _ in ks)}")
    for k, sz, lm in ks:
        if k.endswith("summary.json") or k.endswith("meta.json"):
            show(c, k, 6000)
    print("\n== 2. football paper trader")
    ks = keys(c, "sources/football/live/")
    days = sorted({k.split("/")[4] for k, _, _ in ks if len(k.split("/")) > 5})
    print(f"{len(ks)} objects; modes {sorted({k.split('/')[3] for k, _, _ in ks if len(k.split('/')) > 4})}; days {days[:3]} .. {days[-3:]} ({len(days)})")
    summ = [k for k, _, _ in ks if "summary" in k or "settle" in k]
    for k in summ[-3:]:
        show(c, k, 3000)
    if not summ:
        for k, sz, lm in ks[-6:]:
            print(k, sz, lm)
    print("\n== 3. greyhound trader ledgers")
    ks = keys(c, "sources/greyhound/live/")
    modes = sorted({k.split("/")[3] for k, _, _ in ks if len(k.split("/")) > 4})
    print(f"{len(ks)} objects; modes {modes}")
    for m in modes:
        d = sorted({k.split("/")[4] for k, _, _ in ks if k.split("/")[3] == m and len(k.split("/")) > 5})
        print(m, len(d), d[:2], d[-2:])
    summ = [k for k, _, _ in ks if "summary" in k or "settle" in k]
    for k in summ[-2:]:
        show(c, k, 3000)
    print("\n== 4. the racing census")
    for day in ("2026-11-01", "2026-11-02"):
        for k, sz, lm in keys(c, f"betfair_live/{day}/"):
            if "census_racing" in k:
                body = json.loads(c.get_object(Bucket=B, Key=k)["Body"].read())
                print(f"--- {k} read {body.get('read_utc')} window {body.get('hours')} h")
                for r in body.get("countries", []):
                    print(f"  {r.get('country'):4s} markets {r.get('markets')} win {r.get('win_markets')} "
                          f"matched {r.get('win_matched_total')} median {r.get('win_matched_median')} "
                          f"venues {(r.get('venues') or [])[:6]} {r.get('error', '')}")


if __name__ == "__main__":
    main()
