"""Second run, after the fixes of the first (names: suffixes, joint licences, nicknames; headgear joined as HRB writes it; the trip snapped to the course's; removals from the night's books only). The Betfair card fallback against the HRB card, field by field (the owner, 18 Nov: "build the fallback"; read-only).

For each day 10-17 Nov: the card card_betfair.fetch_card builds from the UK server's recorded catalogue (S3
betfair_live/<day>/), against the morning's HRB card (S3 racecards/<day>_HHMM.csv, the first of the day). Races and
runners on each, and for the runners on both (matched by race time, track and the horse's name without its country
suffix) how often each field agrees. Also the values Betfair's WEARING takes and the HRB headgear codes, so the
mapping can be checked.

Pass for the card (set before the run): races and runners on Betfair's card at least 95% of HRB's; on the runners on
both, track, time, distance, age and weight agree on at least 98%; jockey and trainer on at least 90%.
"""
import gzip
import io
import os
import re
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import card_betfair as cb  # noqa: E402
from model.card_enrich import parse_distance_furlongs  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
DAYS = [date(2026, 11, 10) + timedelta(days=i) for i in range(8)]


def norm(s):
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\s*\((?:[A-Z]{2,3})\)\s*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def hrb_card(day):
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{day:%Y-%m-%d}_")
                  .get("Contents", []))
    if not keys:
        return None, None
    raw = s3.get_object(Bucket=bucket, Key=keys[0])["Body"].read()
    return pd.read_csv(io.BytesIO(raw), dtype={"race_time": str}), keys[0]


rows, wear, gear = [], [], []
for day in DAYS:
    bf = cb.fetch_card(day, db_path="horse_racing.db", s3=s3, bucket=bucket)
    hr, key = hrb_card(day)
    if bf.empty or hr is None:
        print(f"{day}: Betfair card {len(bf)} runners; HRB card {'missing' if hr is None else len(hr)}")
        continue
    raw = cb._s3_csv(s3, bucket, f"betfair_live/{day:%Y-%m-%d}/markets_overnight.csv.gz")
    if raw is not None and "wearing" in raw:
        wear.append(raw.wearing.dropna())
    if "headgear" in hr:
        gear.append(hr.headgear.dropna())
    for d in (bf, hr):
        d["k"] = d.track.map(norm) + "|" + d.race_time.astype(str).str.strip() + "|" + d.horse_name.map(norm)
        d["race"] = d.track.map(norm) + "|" + d.race_time.astype(str).str.strip()
    both = bf.merge(hr, on="k", suffixes=("_bf", "_hr"))
    r = {"day": str(day), "hrb_file": key.rsplit("/", 1)[-1], "races_hrb": hr.race.nunique(),
         "races_bf": bf.race.nunique(), "races_both": len(set(bf.race) & set(hr.race)),
         "runners_hrb": len(hr), "runners_bf": len(bf), "runners_both": len(both)}

    def agree(col, f=lambda s: s.astype(str).str.strip().str.lower()):
        a, b = f(both[f"{col}_bf"]), f(both[f"{col}_hr"])
        return float((a == b).mean()) if len(both) else np.nan

    num = lambda s: pd.to_numeric(s, errors="coerce").round(1)  # noqa: E731
    r["dist"] = float((both.race_distance_bf.map(parse_distance_furlongs).round()
                       == both.race_distance_hr.map(parse_distance_furlongs).round()).mean())
    r["age"] = agree("horse_age", num)
    r["pounds"] = agree("pounds", num)
    r["or"] = agree("official_rating", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).round())
    r["days"] = agree("days_since_lr", num)
    r["jockey"] = agree("jockey_name", lambda s: s.astype(str).str.replace(r"\(\d+\)\s*$", "", regex=True)
                        .str.replace(r"[^A-Za-z]", "", regex=True).str.lower())
    r["trainer"] = agree("trainer", lambda s: s.astype(str).str.replace(r"[^A-Za-z]", "", regex=True).str.lower())
    r["headgear"] = agree("headgear", lambda s: s.fillna("").astype(str).str.lower().str.split().map(
        lambda x: " ".join(sorted(x))))
    r["going"] = agree("going_description")
    r["going_bf_filled"] = float(both.going_description_bf.notna().mean())
    lo = np.log(pd.to_numeric(both.odds_bf, errors="coerce") / pd.to_numeric(both.odds_hr, errors="coerce"))
    r["odds_mad_log"] = float(lo.abs().median())
    rows.append(r)
    if day == DAYS[-1]:
        cols = ["horse_name_bf", "jockey_name_bf", "jockey_name_hr", "trainer_bf", "trainer_hr", "headgear_bf",
                "headgear_hr", "going_description_bf", "going_description_hr", "race_distance_bf", "race_distance_hr"]
        miss = both[both.jockey_name_bf.str.replace(r"[^A-Za-z]", "", regex=True).str.lower()
                    != both.jockey_name_hr.astype(str).str.replace(r"\(\d+\)\s*$", "", regex=True)
                    .str.replace(r"[^A-Za-z]", "", regex=True).str.lower()]
        print(f"\n{day}: jockeys that differ ({len(miss)} of {len(both)}):")
        print(miss[cols[:3]].head(25).to_string(index=False))
        tmiss = both[both.trainer_bf.str.replace(r"[^A-Za-z]", "", regex=True).str.lower()
                     != both.trainer_hr.astype(str).str.replace(r"[^A-Za-z]", "", regex=True).str.lower()]
        print(f"\n{day}: trainers that differ ({len(tmiss)} of {len(both)}):")
        print(tmiss[["horse_name_bf", "trainer_bf", "trainer_hr"]].head(25).to_string(index=False))
        print(f"\n{day}: a sample of the other fields")
        print(both[cols[5:]].drop_duplicates().head(20).to_string(index=False))
        only_hr = hr[~hr.race.isin(bf.race)].drop_duplicates("race")[["race_time", "track"]]
        only_bf = bf[~bf.race.isin(hr.race)].drop_duplicates("race")[["race_time", "track"]]
        print(f"\n{day}: races only on HRB's card: {only_hr.to_dict('records')}; only on Betfair's: "
              f"{only_bf.to_dict('records')}")

t = pd.DataFrame(rows)
print("\n== the Betfair card against the HRB card, by day (agreement shares on the runners on both)")
print(t.round(3).to_string(index=False))
if not t.empty:
    s = t[["races_hrb", "races_bf", "races_both", "runners_hrb", "runners_bf", "runners_both"]].sum()
    print(f"\n   all days: races {s.races_both}/{s.races_hrb} ({s.races_both / s.races_hrb:.1%}); runners "
          f"{s.runners_both}/{s.runners_hrb} ({s.runners_both / s.runners_hrb:.1%})")
    print("   mean agreement:", t[["dist", "age", "pounds", "or", "days", "jockey", "trainer", "headgear", "going",
                                 "going_bf_filled", "odds_mad_log"]].mean().round(3).to_dict())
if wear:
    print("\nBetfair WEARING values:", pd.concat(wear).value_counts().head(30).to_dict())
if gear:
    print("HRB headgear values:", pd.concat(gear).value_counts().head(30).to_dict())
