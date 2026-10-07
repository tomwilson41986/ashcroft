"""The first whole day on the live application key (7 Oct; read-only).

Today's session started at 07:06 UTC (run 37585355891, "The trader reads the live key's feed", GBP2,577.13 free, the
GBP4,000 limit) and the owner restarted it at 08:00 UTC on PR #122's code (run 37590881649: no day limit, with the
funds the owner added this morning). What S3 holds so far:

- the ledger (copied whenever it grows): backs by session, the backs' reasons ("closing CLV" alone is the model fitted
  with volume, ", feed without volume" the delayed feed's), matched, killed (fill-or-kill) and refused, and every
  error by kind (INVALID_APP_KEY, INSUFFICIENT_FUNDS, the session), against 1-6 Oct up to the same UK time;
- the trader's books (books_trader): each runner's matched money, reported on the live key's feed (0 on the delayed
  key's), and the places paid (number_of_winners, PR #120) on every book of the day record;
- the day's files under betfair_live/2026-10-07/ and tote/live/2026-10-07/, and when each last changed.
"""

from __future__ import annotations

import gzip
import io
import os
from collections import Counter

import boto3
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
DAY = "2026-10-07"
RESTART = pd.Timestamp(f"{DAY} 08:00", tz="UTC")
NOW = pd.Timestamp.now(tz="UTC")
pd.set_option("display.width", 220)


def get(key: str) -> bytes | None:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        print(f"  {key}: {type(exc).__name__}")
        return None


def csv(key: str) -> pd.DataFrame:
    raw = get(key)
    if raw is None:
        return pd.DataFrame()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


for prefix in (f"trading/live/{DAY}/", f"betfair_live/{DAY}/", f"tote/live/{DAY}/"):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
    print(prefix, [(o["Key"].split("/")[-1], o["Size"], o["LastModified"].strftime("%H:%M")) for o in objs] or
          "nothing")


def morning(day: str, upto_uk: str) -> dict:
    """A day's ledger up to the same UK time of day: backs, matched, killed, refused, the reasons."""
    led = csv(f"trading/live/{day}/ledger.csv")
    if led.empty:
        return {"day": day, "rows": 0}
    t = pd.to_datetime(led["ts"], utc=True, errors="coerce")
    uk = t.dt.tz_convert("Europe/London")
    cut = pd.Timestamp(f"{day} {upto_uk}", tz="Europe/London")
    d = led[uk <= cut]
    backs = d[d["event"].astype(str).eq("back")]
    m = pd.to_numeric(backs["matched"], errors="coerce").fillna(0)
    status = backs["status"].astype(str)
    err = backs["error"].astype(str)
    return {"day": day, "rows": len(d), "backs_sent": len(backs), "matched_backs": int((m > 0).sum()),
            "matched": round(float(m.sum()), 2), "killed": int(err.str.contains("EXPIRED|FILL_OR_KILL|LAPSED").sum()),
            "refused": int((status.eq("FAILURE") & ~err.str.contains("EXPIRED")).sum()),
            "reasons": dict(Counter(backs.loc[m > 0, "reason"].astype(str))),
            "errors": dict(Counter(d.loc[d["status"].astype(str).eq("FAILURE"), "error"].astype(str).str[:60]))}


upto = NOW.tz_convert("Europe/London").strftime("%H:%M")
print(f"\n== the mornings up to {upto} UK")
rows = [morning(d, upto) for d in ("2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05",
                                    "2026-10-06", DAY)]
for r in rows:
    print(r)

led = csv(f"trading/live/{DAY}/ledger.csv")
if not led.empty:
    t = pd.to_datetime(led["ts"], utc=True, errors="coerce")
    for name, part in (("first session (07:06-08:00 UTC)", led[t < RESTART]), ("since the 08:00 UTC restart", led[t >= RESTART])):
        backs = part[part["event"].astype(str).eq("back")]
        m = pd.to_numeric(backs["matched"], errors="coerce").fillna(0)
        print(f"\n{name}: {len(part)} rows, events {dict(Counter(part['event'].astype(str)))}")
        print(f"  backs {len(backs)}, matched {int((m > 0).sum())} for GBP{m.sum():.2f}; reasons "
              f"{dict(Counter(backs['reason'].astype(str)))}")
        bad = part[part["status"].astype(str).eq("FAILURE")]
        print("  failures:", dict(Counter(bad["error"].astype(str).str[:80])) or "none")
        skips = part[part["event"].astype(str).eq("skip")]
        print("  skips:", dict(Counter(skips["error"].astype(str).str[:60]).most_common(8)))

print("\n== the books")
bt = csv(f"betfair_live/{DAY}/books_trader.csv.gz")
if not bt.empty:
    tm = pd.to_numeric(bt.get("runner_total_matched"), errors="coerce")
    pt = pd.to_datetime(bt["polled_utc"], utc=True, errors="coerce")
    print(f"books_trader: {len(bt):,} rows, {bt['market_id'].nunique()} markets, {pt.min()} to {pt.max()}")
    for name, sel in (("before 08:00", pt < RESTART), ("after 08:00", pt >= RESTART)):
        x = tm[sel]
        print(f"  {name}: runners with matched money reported {(x > 0).mean():.1%} of {len(x):,}; median "
              f"GBP{x[x > 0].median() if (x > 0).any() else 0:.0f}")
    if "number_of_winners" in bt.columns:
        print("  number_of_winners:", dict(Counter(bt["number_of_winners"].astype(str))))
    else:
        print("  number_of_winners: not a column")
for name in ("books", "books_other_place"):
    b = csv(f"betfair_live/{DAY}/{name}.csv.gz")
    if b.empty:
        continue
    pt = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
    print(f"{name}: {len(b):,} rows, {b['market_id'].nunique()} markets, last poll {pt.max()}; number_of_winners "
          f"{dict(Counter(b['number_of_winners'].astype(str))) if 'number_of_winners' in b.columns else 'not a column'}")
    if "source" in b.columns:
        print(f"  sources {dict(Counter(b['source'].astype(str)))}")
mk = csv(f"betfair_live/{DAY}/markets_other_place.csv.gz")
if not mk.empty:
    print("markets_other_place each_way_divisor:",
          dict(Counter(mk["each_way_divisor"].astype(str))) if "each_way_divisor" in mk.columns else "not a column")
