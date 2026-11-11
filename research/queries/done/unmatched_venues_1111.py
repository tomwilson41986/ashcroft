"""QA of every course's name across the sources (the owner, 11 Nov, after Bangor: Betfair "Bangor-on-Dee", the card
"Bangor", never matched, never traded). Read-only.

A. The live record since 1 Oct 2026: every Betfair WIN venue, the share of its runners matched to a race_results row,
   and whether the trader ever placed an order there.
B. Names side by side, normalised as the trader does (betfair_prices.normalise_track, with the aliases): Betfair's
   live venues, the cards' tracks (S3 racecards, every day held), and race_results' GB/IE tracks from 2018. Any name on
   one side with no partner on another, with its nearest candidates.
C. Betfair's price files 2018 on (the abbreviated menu hints, BF_COURSE_MAP): by course, the share of runners matched
   to race_results; and every race_results GB/IE track by year with the share of its runners carrying a Betfair
   price-file row: a course lost in the hint map shows as a low share.
"""
import difflib
import io
import os
import sqlite3
import sys

import boto3
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import TRACK_ALIASES, normalise_track  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
print("aliases in force:", TRACK_ALIASES)

# ---------------------------------------------------------------- A
m = pd.read_sql_query("""SELECT venue, country, race_date, market_id, race_results_id FROM betfair_live_markets
                         WHERE race_date >= '2026-10-01' AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN'""", conn)
last_res = pd.read_sql_query("SELECT MAX(race_date) d FROM race_results", conn).d[0]
m = m[m.race_date <= last_res]
try:
    lo = pd.read_sql_query("SELECT DISTINCT market_id FROM live_orders", conn)
    m["traded"] = m.market_id.astype(str).isin(set(lo.market_id.astype(str)))
except Exception as e:  # noqa: BLE001
    print("live_orders:", e)
    m["traded"] = False
g = m.groupby(["country", "venue"]).agg(days=("race_date", "nunique"), markets=("market_id", "nunique"),
                                        runners=("market_id", "size"),
                                        matched=("race_results_id", lambda x: x.notna().mean()),
                                        markets_traded=("traded", lambda x: m.loc[x.index[x], "market_id"].nunique())).round(3)
print(f"\nA. live record: {len(m):,} WIN runners, {m.race_date.min()} to {m.race_date.max()} (results to {last_res}); "
      f"{len(g)} venues")
print(g.sort_values("matched").to_string())
print("\n   venues matched under 90% or never traded:")
print(g[(g.matched < 0.9) | (g.markets_traded == 0)].to_string())

# ---------------------------------------------------------------- B
bf = set(m.venue.dropna().unique())
rr = pd.read_sql_query("""SELECT track, COUNT(*) n, MIN(race_date) first, MAX(race_date) last FROM race_results
                          WHERE race_date >= '2018-01-01' GROUP BY track""", conn)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
card = {}
tok = None
keys = []
while True:
    r = s3.list_objects_v2(Bucket=bucket, Prefix="racecards/", **({"ContinuationToken": tok} if tok else {}))
    keys += [o["Key"] for o in r.get("Contents", [])]
    if not r.get("IsTruncated"):
        break
    tok = r["NextContinuationToken"]
first_of_day = {}
for k in sorted(keys):
    first_of_day.setdefault(k.split("/")[-1][:10], k)
for d, k in first_of_day.items():
    try:
        c = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=k)["Body"].read()), usecols=lambda x: x in ("track", "venue"))
        for t in c.iloc[:, 0].dropna().unique():
            card.setdefault(t, []).append(d)
    except Exception as e:  # noqa: BLE001
        print("  card", k, type(e).__name__)
print(f"\nB. names: {len(bf)} Betfair live venues; {len(card)} card tracks over {len(first_of_day)} days "
      f"({min(first_of_day)} to {max(first_of_day)}); {len(rr)} race_results tracks from 2018")
nb = {normalise_track(v): v for v in bf}
nc = {normalise_track(v): v for v in card}
nr = {normalise_track(v): v for v in rr.track}


def near(k, pool):
    c = difflib.get_close_matches(k, list(pool), n=2, cutoff=0.6)
    c += [p for p in pool if (p.startswith(k) or k.startswith(p)) and p not in c]
    return [pool[x] for x in c]


def gaps(a, b, la, lb):
    miss = sorted(k for k in a if k not in b)
    print(f"\n   {la} names with no {lb} partner: {len(miss)}")
    for k in miss:
        print(f"     {a[k]!r} ({k}) -> nearest {lb}: {near(k, b)}")
    return miss


gaps(nb, nc, "Betfair live", "card")
gaps(nc, nb, "card", "Betfair live")
gaps(nb, nr, "Betfair live", "race_results")
gaps(nc, nr, "card", "race_results")
print("\n   race_results tracks (2018 on, run since 2025) not seen on Betfair live since 1 Oct (not checked by A; the "
      "nearest Betfair name if any):")
recent = rr[rr["last"] >= "2025-01-01"]
for t in sorted(recent.track):
    k = normalise_track(t)
    if k not in nb:
        row = recent[recent.track == t].iloc[0]
        print(f"     {t!r}: {row.n:,} runners, last {row['last']}; nearest Betfair: {near(k, nb)}")

# ---------------------------------------------------------------- C
try:
    bp = pd.read_sql_query("""SELECT track_bf, country, substr(race_date, 1, 4) y, COUNT(*) n,
                                     SUM(race_results_id IS NOT NULL) matched
                              FROM betfair_prices WHERE market_type = 'win' OR market_type = 'WIN'
                              GROUP BY track_bf, country, y""", conn)
    t = bp.groupby(["country", "track_bf"]).agg(n=("n", "sum"), matched=("matched", "sum"))
    t["share"] = (t.matched / t.n).round(3)
    print(f"\nC. Betfair price files: {int(t.n.sum()):,} WIN runners; courses matched under 85% to race_results:")
    print(t[t.share < 0.85].sort_values("share").to_string())
    cov = pd.read_sql_query("""SELECT r.track, substr(r.race_date, 1, 4) y, COUNT(*) n,
                                      SUM(EXISTS (SELECT 1 FROM betfair_prices b WHERE b.race_results_id = r.id)) priced
                               FROM race_results r WHERE r.race_date >= '2024-01-01' AND r.race_date <= '2026-10-31'
                               GROUP BY r.track, y""", conn)
    c2 = cov.groupby("track").agg(n=("n", "sum"), priced=("priced", "sum"))
    c2["share"] = (c2.priced / c2.n).round(3)
    print("\n   race_results tracks 2024 to Oct 2026 with under 85% of runners carrying a Betfair price-file row:")
    print(c2[c2.share < 0.85].sort_values("share").to_string())
    print(f"\n   all tracks: median share {c2.share.median():.3f}; {len(c2)} tracks")
except Exception as e:  # noqa: BLE001
    print("C failed:", type(e).__name__, e)
