"""Which inputs differ between the 06:00 card and the results table? (the BFSP plan of 10 Nov, step 3 follow-up; read-only)

morning_card_gap_1110 found the 06:00 file about 0.008 worse than the same model re-priced on the results table's rows,
in clean races as much as in races with non-runners. Same model, same history: only the day's own rows differ. This
compares, runner by runner, every column the 06:00 file carries that the results table also has (going, jockey, weight,
rating, field size, days since the last run, headgear, stall, class, trip ...), 24 Oct - 8 Nov: how often each differs,
and the 06:00 error in races where it differs against races where it does not.
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
os.makedirs("live_q", exist_ok=True)
import boto3  # noqa: E402
import botocore  # noqa: E402

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
d = FROM
while d <= TO:
    try:
        s3.download_file(bucket, f"predictions/{d.isoformat()}.csv", f"live_q/{d.isoformat()}.csv")
    except botocore.exceptions.ClientError:
        print(f"  no 06:00 file for {d}")
    d += timedelta(days=1)

frames = []
for f in sorted(os.listdir("live_q")):
    if f[:10] < FROM.isoformat() or f[:10] > TO.isoformat():
        continue
    x = pd.read_csv(os.path.join("live_q", f), dtype=str)
    x["file_day"] = f[:10]
    frames.append(x)
card = pd.concat(frames, ignore_index=True)
for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if a not in card.columns and b in card.columns:
        card[a] = card[b]
card["race_date"] = card["race_date"].fillna(card["file_day"])
print("06:00 file columns:", ", ".join(card.columns))

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
res = pd.read_sql_query(f"SELECT * FROM race_results WHERE race_date >= '{FROM}' AND race_date <= '{TO}'", conn)
res["bfsp"] = pd.to_numeric(res["bfsp"], errors="coerce")
res = res[res.bfsp > 1]


def key(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str) + "|" + x["horse_name"].map(normalise_horse))


card["_k"], res["_k"] = key(card), key(res)
card = card.drop_duplicates("_k", keep="last")
res = res.drop_duplicates("_k", keep="last")
skip = {"race_date", "race_time", "track", "horse_name", "_k", "file_day", "predicted_bfsp",
        "predicted_win_prob_norm", "model_rank", "actual_bfsp", "bfsp_diff_pct", "value_edge", "placing_numerical",
        "odds"}
cols = [c for c in card.columns if c in res.columns and c not in skip]
m = card.merge(res[["_k", "bfsp"] + cols], on="_k", suffixes=("_card", "_res"))
m["race"] = m._k.str.rsplit("|", n=1).str[0]
pred = pd.to_numeric(m["predicted_bfsp"], errors="coerce")
m["pn"] = (1 / pred) / (1 / pred).groupby(m.race).transform("sum")
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
    e_in = m.err[race_any].mean() if race_any.any() else np.nan
    e_out = m.err[~race_any].mean() if (~race_any).any() else np.nan
    ex = m.loc[dff, [f"{c}_card", f"{c}_res"]].head(3).astype(str).agg(" -> ".join, axis=1).tolist()
    rows.append({"column": c, "runners_differ": f"{dff.mean():.1%}", "races_any": f"{race_any.mean():.1%}",
                 "err_races_differ": e_in, "err_races_same": e_out, "examples": "; ".join(ex)[:120]})
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
out = pd.DataFrame(rows).sort_values("runners_differ", ascending=False)
print("\n== each column, 06:00 card against the results table (the same runners)")
print(out.round(4).to_string(index=False))
