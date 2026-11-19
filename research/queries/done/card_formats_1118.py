"""Formats for the Betfair card fallback (the owner, 18 Nov: "build the fallback"; read-only).

1. Betfair venue -> the database's track name, from betfair_live_markets rows linked to race_results (race_results_id).
2. The HRB card as saved (s3 racecards/<day>_HHMM.csv, the last of 17 Nov): columns, and a sample of race_time,
   track, race_distance, going_description, race_class, race_name, horse_name, jockey_name, odds.
3. The recorder's catalogue for the same day (s3 betfair_live/<day>/markets.csv): market names, race_type values,
   venues, a sample of runner rows, and how many runner names match the HRB card's (exact, and without the country
   suffix).
4. The going recorded in race_results by track and code over the last 60 days: how often a track's going is
   all-weather ('Standard'/'Slow') for flat and for jumps (the fallback's going estimate rests on this).
"""
import gzip
import io
import os
import re
import sqlite3

import boto3
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 40)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
DAY = "2026-11-17"


def obj(key):
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return gzip.decompress(raw) if key.endswith(".gz") else raw


print("== 1. Betfair venue -> race_results.track")
m = pd.read_sql_query("""SELECT DISTINCT m.venue, m.country, r.track FROM betfair_live_markets m
                         JOIN race_results r ON r.id = m.race_results_id
                         WHERE m.market_type = 'WIN' AND m.race_results_id IS NOT NULL""", conn)
m["same"] = m.venue.str.lower().str.replace(r"[^a-z0-9]", "", regex=True) == \
    m.track.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
print(f"   {len(m)} venue/track pairs; differing after lower-case alphanumeric: {int((~m.same).sum())}")
print(m[~m.same].sort_values("venue").to_string(index=False))
dup = m.groupby("venue").track.nunique()
print("   venues with more than one track name:", dup[dup > 1].to_dict())
unlinked = pd.read_sql_query("""SELECT venue, country, COUNT(DISTINCT market_id) AS markets FROM betfair_live_markets
                                WHERE market_type = 'WIN' AND country IN ('GB', 'IE') AND race_results_id IS NULL
                                GROUP BY venue, country ORDER BY markets DESC""", conn)
print("   venues with WIN markets never linked to a result:")
print(unlinked.head(30).to_string(index=False))

print("\n== 2. the HRB card")
keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{DAY}").get("Contents", []))
print("   files:", keys)
card = pd.read_csv(io.BytesIO(obj(keys[-1])), dtype=str)
print("   columns:", list(card.columns))
cols = [c for c in ("race_time", "track", "race_distance", "going_description", "race_class", "prize_money",
                    "race_name", "horse_name", "jockey_name", "trainer", "pounds", "odds", "stall",
                    "official_rating", "headgear", "days_since_lr", "card_stallion", "card_dam", "card_sex")
        if c in card.columns]
print(card[cols].drop_duplicates(["track", "race_time"]).head(12).to_string(index=False))
print(card[cols].head(8).to_string(index=False))

print("\n== 3. the recorder's catalogue")
mkeys = [o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"betfair_live/{DAY}/").get("Contents", [])]
print("   files:", [k for k in mkeys if "market" in k])
mk = pd.read_csv(io.BytesIO(obj(next(k for k in mkeys if k.rsplit("/", 1)[-1].startswith("markets")))), dtype=str)
print("   columns:", list(mk.columns))
w = mk[(mk.market_type == "WIN") & mk.country.isin(["GB", "IE"])]
print(f"   WIN GB/IE: {w.market_id.nunique()} markets, {len(w.drop_duplicates(['market_id', 'selection_id']))} runners")
print("   race_type:", w.drop_duplicates("market_id").race_type.value_counts(dropna=False).to_dict())
print("   market names:", sorted(w.market_name.dropna().unique())[:60])
print(w.drop_duplicates(["market_id", "selection_id"]).head(8).to_string(index=False))


def norm(s, keep_suffix=True):
    s = "" if pd.isna(s) else str(s)
    if not keep_suffix:
        s = re.sub(r"\s*\((?:[A-Z]{2,3})\)\s*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


bf_names = set(w.runner_name.map(norm))
bf_base = set(w.runner_name.map(lambda s: norm(s, False)))
if "horse_name" in card:
    hn = card.horse_name.dropna()
    print(f"   HRB card horses {len(hn)}: in Betfair's catalogue exact {hn.map(norm).isin(bf_names).sum()}, "
          f"without the suffix {hn.map(lambda s: norm(s, False)).isin(bf_base).sum()}")
    hist = pd.read_sql_query("SELECT DISTINCT horse_name FROM race_results WHERE race_date >= '2024-01-01'", conn)
    hk, hb = set(hist.horse_name.map(norm)), set(hist.horse_name.map(lambda s: norm(s, False)))
    r = w.drop_duplicates(["market_id", "selection_id"]).runner_name
    print(f"   Betfair runners {len(r)}: in history since 2024 exact {r.map(norm).isin(hk).sum()}, "
          f"without the suffix {r.map(lambda s: norm(s, False)).isin(hb).sum()}")

print("\n== 4. going by track and code, last 60 days of race_results")
g = pd.read_sql_query("""SELECT track, race_type, going_description, race_date FROM race_results
                         WHERE race_date >= date('now', '-60 days')""", conn).drop_duplicates()
g["jumps"] = g.race_type.fillna("").str.contains("Hurdle|Chase|NH Flat|Hunters", case=False)
g["aw"] = g.going_description.fillna("").str.contains("standard|slow", case=False, regex=True)
print(g.groupby(["track", "jumps"]).agg(days=("race_date", "nunique"), aw_share=("aw", "mean"),
                                        last=("race_date", "max")).round(2).to_string())
