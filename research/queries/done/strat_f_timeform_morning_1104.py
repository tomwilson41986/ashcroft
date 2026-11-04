"""Strategy study F (4 Nov 2026, read-only): the Timeform improver signals at the Betfair MORNING price.

Study D (research/strat/strat_d_timeform.py, run on the Railway Timeform table, Flat/AW only) found one Timeform
signal that passes the owner's gate at the BSP: a horse that ran a career-best Timeform performance rating, by 5lb or
more, last time (Flat: train 2014-21 A/E 1.09, p 2e-12; test 2022-26 A/E 1.04, p 0.02; place positive in both).
At the BSP it is thin after commission. Study E found the quick return's edge is far larger at the morning price,
because the market reprices those horses by the off. Does the improver behave the same way?

The flags come from research/strat/data/timeform_flags_2014_2026.csv.gz (study D's output: every Timeform run
carrying one of the flags, keyed by date and horse name). They are joined here to race_results on the date and the
normalised horse name (upper case, country suffix and punctuation dropped); a name that appears twice on one day
is dropped rather than guessed. Morning prices are held from 2018, so TRAIN here is 2018-21 and TEST 2022-26.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)


def norm(s: pd.Series) -> pd.Series:
    return (s.fillna("").str.upper().str.replace(r"\s*\([A-Z]{2,4}\)\s*$", "", regex=True)
            .str.replace(r"[^A-Z0-9]", "", regex=True))


df = L.load()
g = df.groupby("horse_name", sort=False)
df["prev_date"] = g["date"].shift(1)
df["prev_won"] = g["won"].shift(1)
df["dslr"] = df["days_since_lr"].where(df["days_since_lr"] > 0, (df["date"] - df["prev_date"]).dt.days)
df["hn"] = norm(df["horse_name"])
fl = pd.read_csv("research/strat/data/timeform_flags_2014_2026.csv.gz")
fl["hn"] = norm(fl["horse"])
fl = fl.drop_duplicates(["race_date", "hn"], keep=False)
flags = [c for c in fl if c.startswith("tf_")]
dup = df.duplicated(["race_date", "hn"], keep=False)
m = df[~dup].merge(fl[["race_date", "hn"] + flags], on=["race_date", "hn"], how="left")
for c in flags:
    m[c] = m[c].fillna(0).astype(int)
fa = m[m["code"].isin(["flat", "aw"]) & (m["race_date"] >= "2014-01-01")]
print("flag rows", len(fl), "| matched to race_results", int((fa[flags].sum(axis=1) > 0).sum()),
      f"({100 * (fa[flags].sum(axis=1) > 0).sum() / len(fl):.1f}%)", "| per flag matched", fa[flags].sum().to_dict())

d = L.priced(fa).copy()
d = d[d["race_date"] >= "2018-01-01"]  # the morning prices; TRAIN 2018-21, TEST 2022-26
q = d["dslr"].between(1, 4)
rows = []
for code in ("flat", "aw", None):
    c = (d["code"] == code) if code else d["code"].isin(["flat", "aw"])
    nm = code or "flat+AW"
    rows += [
        L.rule(f"{nm}: career-best TFR last time by 5+", d, c & (d["tf_pb5"] == 1)),
        L.rule(f"{nm}: ran 5+ above its Timeform master rating last time", d, c & (d["tf_above_master5"] == 1)),
        L.rule(f"{nm}: TFR up twice, by 10+", d, c & (d["tf_up2_10"] == 1)),
        L.rule(f"{nm}: Timeform top-rated (pre-race adjusted)", d, c & (d["tf_top_adj"] == 1)),
        L.rule(f"{nm}: TFR flat over last three (exposed)", d, c & (d["tf_flat_traj"] == 1)),
        L.rule(f"{nm}: career-best TFR LTO by 5+ AND back within 1-4 days", d, c & (d["tf_pb5"] == 1) & q),
        L.rule(f"{nm}: career-best TFR LTO by 5+, did not win LTO", d, c & (d["tf_pb5"] == 1) & (d["prev_won"] == 0)),
    ]
print("\n=== TIMEFORM SIGNALS AT BSP AND AT THE MORNING PRICE (train 2018-21 | t_ test 2022-26) ===")
print(L.gate.__doc__); print(L.gate_am.__doc__)
t = L.rules_table(rows)
t.to_csv("out/strat_timeform_morning_rules.csv", index=False)
