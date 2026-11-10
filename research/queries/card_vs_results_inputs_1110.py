"""Which inputs differ between the 06:00 card and the results table? (the BFSP plan of 10 Nov, step 3 follow-up; read-only)

morning_card_gap_1110 found the 06:00 file about 0.008 worse than the same model re-priced on the results table's rows,
in clean races as much as in races with non-runners. Same model, same history: only the day's own rows differ. This
compares, runner by runner, every column the 06:00 card carries (S3 racecards/<day>_<fetched_at>.csv, the earliest of
the day: the card as the 04:00 UTC run fetched it) that the results table also has (going, jockey, weight, rating,
field size, days since the last run, headgear, stall, class, trip ...), 24 Oct - 8 Nov: how often each differs, and the
06:00 error in races where it differs against races where it does not. (The first version read predictions/, which
holds prices only.)
"""
import os
import sqlite3
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

FROM, TO = date(2026, 10, 24), date(2026, 11, 8)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
for d_ in ("live_q", "live_cards"):
    os.makedirs(d_, exist_ok=True)
import boto3  # noqa: E402
import botocore  # noqa: E402

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
d = FROM
while d <= TO:
    try:
        s3.download_file(bucket, f"predictions/{d.isoformat()}.csv", f"live_q/{d.isoformat()}.csv")
    except botocore.exceptions.ClientError:
        print(f"  no 06:00 file for {d}")
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{d.isoformat()}_")
                  .get("Contents", []))
    if keys:
        s3.download_file(bucket, keys[0], f"live_cards/{d.isoformat()}.csv")
        print(f"  {d}: card {keys[0]} ({len(keys)} fetched that day)")
    else:
        print(f"  no card for {d}")
    d += timedelta(days=1)


def read_dir(path, text=None):
    frames = []
    for f in sorted(os.listdir(path)):
        if f[:10] < FROM.isoformat() or f[:10] > TO.isoformat():
            continue
        x = pd.read_csv(os.path.join(path, f), dtype=text if text else str)
        x["file_day"] = f[:10]
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
        if a not in out.columns and b in out.columns:
            out[a] = out[b]
    if "race_date" not in out.columns:
        out["race_date"] = out["file_day"]
    out["race_date"] = out["race_date"].fillna(out["file_day"])
    return out


pred = read_dir("live_q")
card = read_dir("live_cards")
print("racecard columns:", ", ".join(card.columns))

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
res = pd.read_sql_query(f"SELECT * FROM race_results WHERE race_date >= '{FROM}' AND race_date <= '{TO}'", conn)
res["bfsp"] = pd.to_numeric(res["bfsp"], errors="coerce")
res = res[res.bfsp > 1]


def key(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str) + "|" + x["horse_name"].map(normalise_horse))


card["_k"], res["_k"], pred["_k"] = key(card), key(res), key(pred)
card = card.drop_duplicates("_k", keep="last")
res = res.drop_duplicates("_k", keep="last")
pred = pred.drop_duplicates("_k", keep="last")[["_k", "predicted_bfsp"]]
skip = {"race_date", "race_time", "track", "horse_name", "_k", "file_day", "predicted_bfsp", "bfsp",
        "predicted_win_prob_norm", "model_rank", "actual_bfsp", "bfsp_diff_pct", "value_edge", "placing_numerical",
        "odds", "id"}
cols = [c for c in card.columns if c in res.columns and c not in skip]
print("columns compared:", ", ".join(cols))
m = card.drop(columns=[c for c in ("predicted_bfsp", "bfsp") if c in card.columns]).merge(pred, on="_k")
m = m.merge(res[["_k", "bfsp"] + cols], on="_k", suffixes=("_card", "_res"))
m["race"] = m._k.str.rsplit("|", n=1).str[0]
p = pd.to_numeric(m["predicted_bfsp"], errors="coerce")
m["pn"] = (1 / p) / (1 / p).groupby(m.race).transform("sum")
m["pb"] = (1 / m.bfsp) / (1 / m.bfsp).groupby(m.race).transform("sum")
m["err"] = np.abs(np.log(m.pn / m.pb))
print(f"\n{len(m):,} runners matched in {m.race.nunique():,} races; mean 06:00 error {m.err.mean():.4f}")


def differs(a, b):
    na, nb = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    num = na.notna() & nb.notna()
    out = pd.Series(False, index=a.index)
    out[num] = (na[num] - nb[num]).abs() > 1e-6
    sa = a.fillna("").astype(str).str.strip().str.lower()
    sb = b.fillna("").astype(str).str.strip().str.lower()
    out[~num] = sa[~num] != sb[~num]
    return out


rows = []
for c in cols:
    dff = differs(m[f"{c}_card"], m[f"{c}_res"])
    race_any = dff.groupby(m.race).transform("any")
    ex = m.loc[dff, [f"{c}_card", f"{c}_res"]].head(3).astype(str).agg(" -> ".join, axis=1).tolist()
    rows.append({"column": c, "runners_differ": float(dff.mean()), "races_any": float(race_any.mean()),
                 "err_races_differ": m.err[race_any].mean() if race_any.any() else np.nan,
                 "err_races_same": m.err[~race_any].mean() if (~race_any).any() else np.nan,
                 "examples": "; ".join(ex)[:110]})
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 110)
if rows:
    out = pd.DataFrame(rows).sort_values("runners_differ", ascending=False)
    print("\n== each column, 06:00 card against the results table (the same runners)")
    print(out.round(4).to_string(index=False))
else:
    print("no column in common")
missing = [c for c in ("going_description", "jockey_name", "pounds", "official_rating", "number_of_runners",
                       "days_since_lr", "median_or", "prize_money", "race_class", "dist_furlongs")
           if c not in card.columns]
print("\nresults-table inputs the card lacks:", ", ".join(missing) or "none")
