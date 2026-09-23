"""Does the card fill reproduce the results table? (QA C1)

For 40 race days in the development window: take the day's result rows, keep only
what the HTML card carries (the race header, stall, horse, age, weight, jockey,
trainer, OR with unrated blank, headgear, days since last run), run
model.card_enrich.enrich_card with the history before that day, and score every
filled field against the value the results table holds.
"""
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.card_enrich import CARD_FILLED, enrich_card  # noqa: E402

CARD_COLS = ["race_date", "race_time", "track", "going_description", "race_class", "race_distance", "prize_money",
             "race_name", "stall", "horse_name", "horse_age", "pounds", "jockey_name", "trainer", "official_rating",
             "headgear", "days_since_lr", "number_of_runners"]
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("SELECT * FROM race_results WHERE race_date < '2026-04-01'", conn)
d = d.sort_values(["race_date", "race_time"], kind="stable").reset_index(drop=True)
days = sorted(d.loc[d.race_date >= "2025-10-01", "race_date"].unique())
rng = np.random.default_rng(0)
days = sorted(rng.choice(days, size=min(40, len(days)), replace=False))
print(f"{len(days)} days from {days[0]} to {days[-1]}")

rows = []
t0 = time.time()
for day in days:
    truth = d[d.race_date == day]
    card = truth[CARD_COLS].copy()
    card["official_rating"] = pd.to_numeric(card.official_rating, errors="coerce").replace(0, np.nan)   # blank on the card
    out = enrich_card(card, d[d.race_date < day])
    out.index = truth.index
    rows.append((truth, out))
print(f"filled in {time.time() - t0:.0f}s")

truth = pd.concat([t for t, _ in rows])
fill = pd.concat([o for _, o in rows])
print(f"\n{len(truth):,} runners in {truth.groupby(['race_date', 'track', 'race_time']).ngroups:,} races\n")
print(f"{'field':18s} {'truth known':>11s} {'filled':>7s} {'agree':>7s}   note")
for col in CARD_FILLED:
    if col not in truth.columns:
        continue
    t, f = truth[col], fill[col]
    num = col in ("dist_furlongs", "career_runs", "jockeys_claim", "max_or_in_race", "median_or", "official_rating",
                  "prize_money")
    if num:
        tn, fn = pd.to_numeric(t, errors="coerce"), pd.to_numeric(f, errors="coerce")
        known = tn.notna()
        both = known & fn.notna()
        agree = np.isclose(tn[both], fn[both], atol=0.051).mean() if both.any() else np.nan
        note = f"mean |diff| {np.nanmean(np.abs(tn[both] - fn[both])):.3f}" if both.any() else ""
        if col == "median_or":
            note += f"; truth null {(~known).mean():.3f}, fill null {fn.isna().mean():.3f}, " \
                    f"null together {((~known) == fn.isna()).mean():.3f}"
    else:
        ts, fs = t.astype("string"), f.astype("string")
        known = t.notna() & ts.str.strip().ne("")
        both = known & f.notna()
        agree = (ts[both] == fs[both]).mean() if both.any() else np.nan
        miss = (ts[both] != fs[both])
        note = ""
        if miss.any():
            top = pd.DataFrame({"t": ts[both][miss], "f": fs[both][miss]}).value_counts().head(3)
            note = "; ".join(f"{a} -> {b} ({n})" for (a, b), n in top.items())
    print(f"{col:18s} {known.mean():11.3f} {both.sum() / max(known.sum(), 1):7.3f} {agree:7.3f}   {note}")

# what the card path loses for debutants: sex and pedigree have no earlier row to come from
deb = pd.to_numeric(truth.career_runs, errors="coerce") == 0
print(f"\ndebutants {deb.mean():.3f} of runners; their sex filled {fill.loc[deb, 'horse_sex'].notna().mean():.3f}, "
      f"sire filled {fill.loc[deb, 'stallion'].notna().mean():.3f}")
