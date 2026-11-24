"""The data map for the 24 Nov audit (read-only): every table of horse_racing.db with its columns, rows and date
range, the null share of each column of the price and live tables, and the S3 record's prefixes by day count.
Descriptive; no test."""
import os
import sqlite3

import boto3
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 500)
con = sqlite3.connect("horse_racing.db")
tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
print("tables:", len(tabs))
DATE_COLS = ("race_date", "date", "day", "event_date", "market_time", "off_utc", "placed_date", "ts", "time_utc")
for t in tabs:
    cols = [(r[1], r[2]) for r in con.execute(f'pragma table_info("{t}")')]
    try:
        n = con.execute(f'select count(*) from "{t}"').fetchone()[0]
    except Exception as e:  # noqa: BLE001
        n = f"error {e}"
    rng = ""
    for c in DATE_COLS:
        if c in [x[0] for x in cols]:
            try:
                lo, hi = con.execute(f'select min("{c}"), max("{c}") from "{t}"').fetchone()
                rng = f"{c} {lo} .. {hi}"
            except Exception:  # noqa: BLE001
                pass
            break
    print(f"\n== {t}: {n} rows; {rng}")
    print("   " + ", ".join(f"{c}:{ty}" for c, ty in cols))

DETAIL = [t for t in tabs if t.startswith(("betfair_", "live_", "tote", "predictions", "morning", "card", "racecard"))]
for t in DETAIL:
    try:
        n = con.execute(f'select count(*) from "{t}"').fetchone()[0]
        if not n:
            continue
        df = pd.read_sql(f'select * from "{t}" order by rowid desc limit 200000', con)
        print(f"\n== {t}: null share and a sample (the last {len(df):,} rows)")
        print((df.isna().mean().round(3)).to_string())
        print(df.head(3).T.to_string())
        for c in ("market_type", "mark", "mode", "event", "status", "country", "source_file"):
            if c in df.columns and df[c].nunique() < 60:
                print(f"   {c}: {df[c].value_counts(dropna=False).to_dict()}")
    except Exception as e:  # noqa: BLE001
        print(f"   {t}: {e}")

for t, c in (("race_results", "race_date"), ("betfair_prices", "race_date"), ("betfair_live_marks", "race_date"),
             ("betfair_live_markets", "race_date"), ("live_orders", "race_date")):
    try:
        print(f"\n== {t} rows by year-month")
        print(pd.read_sql(f'select substr("{c}",1,7) m, count(*) n from "{t}" group by 1 order by 1', con)
              .set_index("m")["n"].tail(40).to_dict())
    except Exception as e:  # noqa: BLE001
        print(f"   {t}: {e}")

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"


def prefixes(prefix):
    out, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += [p["Prefix"] for p in r.get("CommonPrefixes", [])]
        out += [o["Key"] for o in r.get("Contents", [])]
        token = r.get("NextContinuationToken")
        if not token:
            return out


print(f"\n== s3://{bucket}/ top level")
for p in prefixes(""):
    sub = prefixes(p) if p.endswith("/") else []
    print(f"   {p}: {len(sub)} entries; first {sub[:2]}, last {sub[-2:]}")
    if p in ("sources/", "trading/", "morning/", "tote/", "research/"):
        for q in sub:
            if q.endswith("/"):
                s2 = prefixes(q)
                print(f"      {q}: {len(s2)} entries; first {s2[:1]}, last {s2[-1:]}")
try:
    ks = prefixes("betfair_live/2026-11-23/")
    print("\n== betfair_live/2026-11-23/:", ks)
except Exception as e:  # noqa: BLE001
    print(e)
