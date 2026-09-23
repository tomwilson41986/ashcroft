"""Does the filled live card give the model what the results table gives it? (QA C1)

card_enrich_check.py scores the filled fields one at a time. This scores what the
model serves. On two development days the deployed model prices every runner
through prepare_and_predict -- the function the 06:00 job calls -- four ways,
each with the same history before the day:

  table    the day's result rows: every field, as the backtest sees them
  blind    the same rows with the day's outcomes blanked (place, beaten distances,
           times, SP and the favourite flag, BSP, the comment, prize won). A
           feature that moves between table and blind reads the day's own result:
           a backtest leak, which would also explain an edge the live record lacks
  filled   the rows cut down to what the HTML card carries, then enrich_card: the
           live path with the fix
  raw      the same card without the fill, prize money as the card prints it: the
           live path before the fix

filled against blind is what the fix leaves open; raw against blind is what it
closed. Reported: the served price's gap, the race's top pick, and every feature
that differs, weighted by the model's gain. The model was trained through 17 Sep,
so its error against BSP on these days is in-sample for every variant alike.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import model.card_enrich as card_enrich  # noqa: E402
from predict_bfsp_today import load_bfsp_model, load_historical, prepare_and_predict  # noqa: E402

CARD_COLS = ["race_date", "race_time", "track", "going_description", "race_class", "race_distance", "prize_money",
             "race_name", "stall", "horse_name", "horse_age", "pounds", "jockey_name", "trainer", "official_rating",
             "headgear", "days_since_lr", "number_of_runners"]
OUTCOME_COLS = ["place", "distbt", "odds", "fav", "comptime", "comptime_numeric", "total_dst_bt", "placing_numerical",
                "bfsp", "bfsp_place", "comment", "horse_prizewin"]
KEY = ["track", "race_time", "horse_name"]
RACE = ["track", "race_time"]

model, feature_cols, vocab = load_bfsp_model("data/models")
gain = pd.read_csv("data/models/bfsp_feature_importance.csv").set_index("feature")["importance"]
gain = gain / gain.sum()
hist = load_historical("horse_racing.db", start_date="2020-01-01")
hist = hist[hist.race_date < "2026-04-01"].reset_index(drop=True)          # development data only
print(f"model: {len(feature_cols)} features; history {len(hist):,} rows to {hist.race_date.max().date()}")

march = hist[hist.race_date >= "2026-03-01"]
n_races = march.groupby(march.race_date.dt.date).apply(lambda g: g.groupby(RACE).ngroups)
busy = n_races[n_races >= 12]
days = [max(d for d in busy.index if d.weekday() == 5), max(d for d in busy.index if d.weekday() == 2)]
print("days:", ", ".join(f"{d} ({n_races[d]} races)" for d in days))


def serve(day, runners, fill=True):
    """prepare_and_predict as the 06:00 job calls it: history before the day, the day's runners appended."""
    real = card_enrich.enrich_card
    if not fill:
        card_enrich.enrich_card = lambda card, history: card
    try:
        before = hist[hist.race_date < pd.Timestamp(day)]         # a fresh frame: the call may add columns to it
        out = prepare_and_predict(before, runners.copy(), model, feature_cols, day, vocab)
    finally:
        card_enrich.enrich_card = real
    keep = KEY + ["predicted_bfsp"] + [f for f in feature_cols if f in out.columns]
    out = out[keep].copy()
    gc.collect()
    return out


results = {}
for day in days:
    t0 = time.time()
    truth = hist[hist.race_date == pd.Timestamp(day)].copy()
    blind = truth.copy()
    for c in OUTCOME_COLS:
        if c in blind.columns:
            blind[c] = np.nan
    card = truth[[c for c in CARD_COLS if c in truth.columns]].copy()
    card["official_rating"] = pd.to_numeric(card.official_rating, errors="coerce").replace(0, np.nan)
    raw = card.copy()
    raw["prize_money"] = pd.to_numeric(raw.prize_money, errors="coerce").map(
        lambda v: f"{v:,.0f}" if pd.notna(v) else "")
    out = {"table": serve(day, truth), "blind": serve(day, blind), "filled": serve(day, card),
           "raw": serve(day, raw, fill=False)}
    bsp = truth[KEY + ["bfsp"]].copy()
    results[day] = (out, bsp)
    os.makedirs("out/live_parity", exist_ok=True)                   # kept in the artifact for re-analysis
    for name, frame in {**out, "bsp": bsp}.items():
        frame.to_csv(f"out/live_parity/{day}_{name}.csv.gz", index=False)
    print(f"{day}: {len(truth):,} runners served four ways in {time.time() - t0:.0f}s", flush=True)


def compare(ref, var):
    m = ref.merge(var, on=KEY, suffixes=("_r", "_v"))
    m = m[m.predicted_bfsp_r.notna() & m.predicted_bfsp_v.notna()]
    lr, lv = np.log(m.predicted_bfsp_r), np.log(m.predicted_bfsp_v)
    d = (lv - lr).abs()
    top_r = m.loc[m.groupby(RACE).predicted_bfsp_r.idxmin(), RACE + ["horse_name"]].set_index(RACE)
    top_v = m.loc[m.groupby(RACE).predicted_bfsp_v.idxmin(), RACE + ["horse_name"]].set_index(RACE)
    top = (top_r.horse_name == top_v.horse_name.reindex(top_r.index)).mean()
    diff = []
    for f in feature_cols:
        if f + "_r" not in m.columns or f + "_v" not in m.columns:
            continue
        x, y = pd.to_numeric(m[f + "_r"], errors="coerce"), pd.to_numeric(m[f + "_v"], errors="coerce")
        same = (x.isna() & y.isna()) | ((x - y).abs() <= 1e-9 + 1e-6 * x.abs())
        diff.append((f, 1 - same.mean(), gain.get(f, 0.0)))
    diff = pd.DataFrame(diff, columns=["feature", "share_differs", "gain"])
    return m, d, top, diff


print("\nServed price and top pick, each variant against its reference")
print(f"{'pair':18s} {'runners':>8s} {'|dlog| mean':>11s} {'median':>7s} {'p90':>6s} {'corr':>6s} {'top pick':>9s}"
      f" {'features differ':>16s} {'gain-weighted':>14s}")
pairs = [("table", "blind"), ("blind", "filled"), ("blind", "raw"), ("table", "filled")]
diffs = {}
for ref, var in pairs:
    ms, ds, tops, dfs = [], [], [], []
    for day, (out, _) in results.items():
        m, d, top, diff = compare(out[ref], out[var])
        ms.append(m)
        ds.append(d)
        tops.append(top)
        dfs.append(diff)
    d = pd.concat(ds)
    m = pd.concat(ms)
    diff = pd.concat(dfs).groupby("feature", as_index=False).agg(share_differs=("share_differs", "mean"),
                                                                 gain=("gain", "first"))
    diffs[(ref, var)] = diff
    corr = np.corrcoef(np.log(m.predicted_bfsp_r), np.log(m.predicted_bfsp_v))[0, 1]
    print(f"{ref + ' -> ' + var:18s} {len(m):8,d} {d.mean():11.4f} {d.median():7.4f} {d.quantile(.9):6.3f} "
          f"{corr:6.3f} {np.mean(tops):9.3f} {(diff.share_differs > 0).sum():7d} of {len(diff):3d} "
          f"{(diff.share_differs * diff.gain).sum():14.3f}")

print("\nError against BSP, |log price - log BSP| (in-sample for the model on every variant)")
for name in ("table", "blind", "filled", "raw"):
    errs = []
    for day, (out, bsp) in results.items():
        m = out[name].merge(bsp, on=KEY)
        b = pd.to_numeric(m.bfsp, errors="coerce")
        ok = b > 1
        errs.append((np.log(m.predicted_bfsp[ok]) - np.log(b[ok])).abs())
    e = pd.concat(errs)
    print(f"  {name:7s} {e.mean():.4f} on {len(e):,} runners")

for (ref, var), diff in diffs.items():
    diff = diff[diff.share_differs > 0].assign(weighted=lambda x: x.share_differs * x.gain)
    print(f"\nFeatures that differ, {ref} -> {var}: {len(diff)} (top 30 by gain x share)")
    for r in diff.sort_values("weighted", ascending=False).head(30).itertuples():
        print(f"  {r.feature:36s} differs {r.share_differs:6.3f}  gain {r.gain:.4f}")
