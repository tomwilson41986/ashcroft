"""Every Betfair racing census held, 1-9 Nov 2026 (read-only): which countries Betfair lists racing in, their win
markets, venues and the money matched by the reading (sources/betfair_census.py; before 3 Nov the live-record.yml
evening step of PR #179). The owner's question: do Hong Kong and Japan trade on Betfair at all?"""
import json
import os

import boto3

B = os.getenv("PRED_BUCKET", "ashcroft")


def main():
    c = boto3.client("s3", region_name=os.getenv("PRED_REGION", "eu-west-2"))
    found = 0
    for d in range(1, 10):
        prefix = f"betfair_live/2026-11-{d:02d}/"
        tok, keys = None, []
        while True:
            kw = {"Bucket": B, "Prefix": prefix, "MaxKeys": 1000}
            if tok:
                kw["ContinuationToken"] = tok
            r = c.list_objects_v2(**kw)
            keys += [o["Key"] for o in r.get("Contents", []) if "census_racing" in o["Key"]]
            if not r.get("IsTruncated"):
                break
            tok = r["NextContinuationToken"]
        for k in sorted(keys):
            found += 1
            body = json.loads(c.get_object(Bucket=B, Key=k)["Body"].read())
            print(f"--- {k}: read {body.get('read_utc')}, window {body.get('hours')} h, "
                  f"{len(body.get('countries', []))} countries")
            for r in body.get("countries", []):
                print(f"  {str(r.get('country')):4s} markets {r.get('markets')} win {r.get('win_markets')} "
                      f"matched {r.get('win_matched_total')} median {r.get('win_matched_median')} "
                      f"first off {r.get('first_off_utc')} venues {(r.get('venues') or [])[:8]} {r.get('error', '')}")
    print(f"\ncensus files found: {found}")
    if not found:
        print("no data found: no census_racing file under betfair_live/2026-11-01..09/")


if __name__ == "__main__":
    main()
