"""How much does the 06:00 forecast lose to the horses withdrawn after it? (the BFSP plan of 10 Nov, step 3; read-only)

Training builds every within-race feature on the horses that ran. The 06:00 file builds them on the card, which still
holds the later non-runners, and the update for non-runners only renormalises each race's book. One day (22 Oct) read
0.412 on the 06:00 file against 0.369 re-priced on the final field: more than any feature block is worth.

A: every 06:00 file since the served model (24 Oct) against the BSP, races with a later non-runner against clean ones,
   the 06:00 price renormalised over the runners that ran (as the trader reads it), by the share of the 06:00 book the
   non-runners held.
B: the same days re-priced by the served model on the final field (the runners with a BSP, as training sees a race),
   newest first while time allows. Difference in differences: (06:00 - re-priced) in races with non-runners, less the
   same in clean races (which carries every other card-to-results difference: going, jockeys, the sectionals file).
"""
import os
import sqlite3
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse, normalise_track, race_time_to_24h  # noqa: E402

FROM, TO = date(2026, 10, 24), date(2026, 11, 9)
BUDGET_MIN = 230
T0 = time.time()

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

text = {c: str for c in ("date", "race_date", "venue", "track", "race_time", "runner_name", "horse_name")}
frames = []
for f in sorted(os.listdir("live_q")):
    x = pd.read_csv(os.path.join("live_q", f), dtype=text)
    x["file_day"] = f[:10]
    frames.append(x)
card = pd.concat(frames, ignore_index=True)
for a, b in (("race_date", "date"), ("track", "venue"), ("horse_name", "runner_name")):
    if a not in card.columns and b in card.columns:
        card[a] = card[b]
card["race_date"] = card["race_date"].fillna(card["file_day"])
card["predicted_bfsp"] = pd.to_numeric(card["predicted_bfsp"], errors="coerce")


def rkey(x):
    return (x["race_date"].astype(str).str.slice(0, 10) + "|" + x["track"].map(normalise_track) + "|"
            + x["race_time"].map(race_time_to_24h).astype(str))


def hkey(x):
    return rkey(x) + "|" + x["horse_name"].map(normalise_horse)


conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
res = pd.read_sql_query(f"""SELECT race_date, track, race_time, horse_name, bfsp, race_type, surface_type
                            FROM race_results WHERE race_date >= '{FROM}' AND race_date <= '{TO}'""", conn)
res["bfsp"] = pd.to_numeric(res["bfsp"], errors="coerce")
ran = res[res.bfsp > 1].copy()
ran["_r"], ran["_k"] = rkey(ran), hkey(ran)
card["_r"], card["_k"] = rkey(card), hkey(card)
card = card.drop_duplicates("_k", keep="last")
card = card[card._r.isin(set(ran._r))]                     # races with a result
card["ran"] = card._k.isin(set(ran._k))
print(f"06:00 files {card.file_day.min()} .. {card.file_day.max()}: {len(card):,} card runners in "
      f"{card._r.nunique():,} races with a result; {int((~card.ran).sum()):,} did not run")
unmatched = ran[~ran._k.isin(set(card._k)) & ran._r.isin(set(card._r))]
print(f"runners with a BSP missing from the 06:00 file in the same races (name matching, reserves): {len(unmatched):,}")

card["p06"] = 1.0 / card.predicted_bfsp
book = card.groupby("_r").p06.transform("sum")
card["nr_p"] = np.where(card.ran, 0.0, card.p06 / book)
race = card.groupby("_r").agg(nr=("ran", lambda s: int((~s).sum())), nr_share=("nr_p", "sum"),
                              size=("ran", "size")).reset_index()
code = ran.drop_duplicates("_r").set_index("_r")
from model.race_shape import race_code  # noqa: E402
code["code"] = race_code(code)
race["code"] = race._r.map(code["code"])


def score(df, col):
    """Mean |log(model / market)| over the runners that ran, each race renormalised on both sides."""
    x = df[df.ran].copy()
    x["pm"] = (1 / x[col]) / x.groupby("_r")[col].transform(lambda s: (1 / s).sum())
    x["pb"] = (1 / x.bsp) / x.groupby("_r").bsp.transform(lambda s: (1 / s).sum())
    x["err"] = np.abs(np.log(x.pm / x.pb))
    return x


m = card.merge(ran[["_k", "bfsp"]].rename(columns={"bfsp": "bsp"}), on="_k", how="left")
a = score(m, "predicted_bfsp").merge(race, on="_r")
a["band"] = pd.cut(a.nr_share, [-0.001, 0.0, 0.05, 0.15, 1.0], labels=["clean", "<5%", "5-15%", ">15%"])
pd.set_option("display.width", 220)
print("\n== A: the 06:00 price (renormalised over the runners that ran) against the BSP, by the book the non-runners held")
print(a.groupby("band", observed=True).agg(runners=("err", "size"), races=("_r", "nunique"),
                                            mae_log=("err", "mean")).round(4).to_string())
print("\nby code, clean against any non-runner:")
a["has_nr"] = np.where(a.nr > 0, "non-runner", "clean")
print(a.pivot_table(index="code", columns="has_nr", values="err", aggfunc=["mean", "size"]).round(4).to_string())

# --- B: re-priced on the final field -----------------------------------------------------------------------------
print("\n== B: re-priced by the served model on the final field (newest first)")
from predict_bfsp_today import load_bfsp_model, load_historical, prepare_and_predict  # noqa: E402

model, feature_cols, vocab = load_bfsp_model("data/models")
hist = load_historical("horse_racing.db", start_date="2018-01-01")
hist["race_date"] = pd.to_datetime(hist["race_date"])
days = sorted(set(card.file_day), reverse=True)
rep = []
for dd in days:
    if (time.time() - T0) / 60 > BUDGET_MIN:
        print(f"  time budget reached; stopped before {dd}")
        break
    t = time.time()
    td = date.fromisoformat(dd)
    before = hist[hist.race_date.dt.date < td].copy()
    runners = hist[(hist.race_date.dt.date == td)].copy()
    runners = runners[pd.to_numeric(runners["bfsp"], errors="coerce") > 1]
    if runners.empty:
        continue
    try:
        p = prepare_and_predict(before, runners, model, feature_cols, td, vocab)
    except Exception as exc:  # noqa: BLE001
        print(f"  {dd}: failed: {exc!r}")
        continue
    p = p[["race_date", "track", "race_time", "horse_name", "predicted_bfsp"]].copy()
    p["_k"] = hkey(p)
    rep.append(p[["_k", "predicted_bfsp"]].rename(columns={"predicted_bfsp": "repriced"}))
    print(f"  {dd}: {len(p):,} runners re-priced in {(time.time() - t) / 60:.1f} min", flush=True)

if rep:
    r = pd.concat(rep, ignore_index=True).drop_duplicates("_k")
    b = m.merge(r, on="_k", how="inner")
    b = b[b._r.isin(set(b[b.ran]._r))]
    # the same runners on both sides: the races where every runner that ran was re-priced
    full = b[b.ran].groupby("_r").repriced.apply(lambda s: s.notna().all())
    b = b[b._r.isin(full[full].index)]
    s06 = score(b, "predicted_bfsp").merge(race, on="_r")
    srp = score(b, "repriced")[["_k", "err"]].rename(columns={"err": "err_rep"})
    s = s06.merge(srp, on="_k")
    s["has_nr"] = np.where(s.nr > 0, "non-runner", "clean")
    s["band"] = pd.cut(s.nr_share, [-0.001, 0.0, 0.05, 0.15, 1.0], labels=["clean", "<5%", "5-15%", ">15%"])
    s["gap"] = s.err - s.err_rep
    print(f"\n{s.file_day.nunique()} days, {len(s):,} runners in {s._r.nunique():,} races")
    print(s.groupby("band", observed=True).agg(runners=("err", "size"), races=("_r", "nunique"),
                                                mae_06=("err", "mean"), mae_repriced=("err_rep", "mean"),
                                                gap=("gap", "mean")).round(4).to_string())
    per_race = s.groupby(["_r", "has_nr"]).gap.mean().reset_index()
    g = per_race.groupby("has_nr").gap
    print("\nper race mean gap (06:00 less re-priced):")
    print(g.agg(["size", "mean", "std"]).round(4).to_string())
    if {"clean", "non-runner"} <= set(per_race.has_nr):
        c, n = per_race[per_race.has_nr == "clean"].gap, per_race[per_race.has_nr == "non-runner"].gap
        did = n.mean() - c.mean()
        se = np.sqrt(n.var() / len(n) + c.var() / len(c))
        print(f"\ndifference in differences (non-runner races less clean): {did:+.4f} (se {se:.4f})")
        share = (s.has_nr == "non-runner").mean()
        print(f"share of runners in races with a non-runner: {share:.1%}; the whole-day cost "
              f"{did * share:+.4f} on the mean error")
print(f"\ndone in {(time.time() - T0) / 60:.0f} min")
