"""What the Tote record of 6 Oct 2026 holds (read-only; the evening check-in of 6 Oct).

tote_recorder.py ran inside the trader's job from 12:55 UTC (the restarted session) and again from 14:25 UTC (the
session restarted for the greyhound trader), both appending to the UK server's day files, copied to
s3://$PRED_BUCKET/tote/live/2026-10-06/. For each file (pools, detail, results): how many answers, their statuses,
when they start and stop; the races and marks the win and place answers cover, and the exotic answers; then the
shape of a declared-dividend answer (race-card/race-results), which the comparison with Betfair has to read, and
one race's answer in full. The answers carry no key (the recorder redacts it from every line).
"""

from __future__ import annotations

import collections
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import boto3
import pandas as pd

sys.path.insert(0, os.getcwd())
from tote_recorder import read_lines, runner_rows  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 120)
DAY = os.environ.get("TOTE_DAY") or "2026-10-06"
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
prefix = f"tote/live/{DAY}/"

listed = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
print(f"s3://{bucket}/{prefix}: " + ", ".join(f"{o['Key'].rsplit('/', 1)[-1]} ({o['Size']:,} bytes, "
                                               f"{o['LastModified']:%H:%M} UTC)" for o in listed))
tmp = Path(tempfile.mkdtemp())
records: dict[str, list[dict]] = {}
for o in listed:
    name = o["Key"].rsplit("/", 1)[-1]
    path = tmp / name
    s3.download_file(bucket, o["Key"], str(path))
    records[name.split(".")[0]] = read_lines(path)


def shape(x, depth=0, max_depth=5):
    """A JSON value's structure: dict keys with their shapes, a list's length and its first item's shape."""
    if depth >= max_depth:
        return type(x).__name__
    if isinstance(x, dict):
        return {k: shape(v, depth + 1, max_depth) for k, v in list(x.items())[:40]}
    if isinstance(x, list):
        return [f"list of {len(x)}", shape(x[0], depth + 1, max_depth)] if x else "empty list"
    return type(x).__name__


for kind, recs in records.items():
    st = collections.Counter(str(r.get("status")) for r in recs)
    times = sorted(str(r.get("polled_utc")) for r in recs if r.get("polled_utc"))
    print(f"\n== {kind}: {len(recs):,} answers, status {dict(st)}, "
          f"{times[0][:19] if times else '-'} to {times[-1][:19] if times else '-'} UTC")

det = records.get("detail", [])
if det:
    d = pd.DataFrame([{k: r.get(k) for k in ("race_id", "pool", "mark", "status", "polled_utc")} for r in det])
    print(f"\n== detail by pool and mark (answers): {d.race_id.nunique()} races")
    print(d.pivot_table(index="mark", columns="pool", values="race_id", aggfunc="count").fillna(0).astype(int)
          .to_string())
    rows = runner_rows(det)
    if len(rows):
        print(f"\nwin and place runner rows: {len(rows):,}; races {rows.race_id.nunique()}; "
              f"base dividend known on {rows.base.notna().mean():.1%}, shown on {rows.shown.notna().mean():.1%}")
        win = rows[(rows.pool == "WIN") & ~rows.scratched & (rows.base > 1)]
        book = win.groupby(["race_id", "mark"]).base.apply(lambda s: float((1 / s).sum())).rename("book")
        tot = win.groupby(["race_id", "mark"]).pool_total.max()
        print("the win pools' book (sum of 1/base dividend) and total by mark, median over races:")
        print(pd.concat([book, tot], axis=1).groupby(level="mark").median().round(3).to_string())
    exo = [r for r in det if r.get("pool") not in ("WIN", "PLACE")]
    if exo:
        print(f"\nexotic answers: {collections.Counter(r.get('pool') for r in exo)}; the first's shape:")
        print(json.dumps(shape(exo[0].get("answer")), indent=1)[:3000])

res = records.get("results", [])
if res:
    ok = [r for r in res if r.get("answer") is not None]
    print(f"\n== results: {len(ok)} answers with a body of {len(res)}; marks {collections.Counter(str(r.get('mark')) for r in res)}")
    if ok:
        last = ok[-1]
        print("the last answer's shape (race ids asked:", last.get("race_id"), ")")
        print(json.dumps(shape(last["answer"]), indent=1)[:6000])
        ans = last["answer"]
        items = ans if isinstance(ans, list) else next((v for v in (ans or {}).values() if isinstance(v, list)), [])
        if items:
            print("\none race's answer in full (first 8,000 characters):")
            print(json.dumps(items[0], indent=1, default=str)[:8000])
            print(f"\n{len(items)} races in that answer; their keys: {sorted(set().union(*(set(i) for i in items if isinstance(i, dict))))}")
