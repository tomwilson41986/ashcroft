"""Strategy J follow-up: the plain run-style projection's pick (highest p_lead) passed at BSP (TEST A/E 1.06) where the
fuller leader model's pick did not (1.01). p_lead is the same number for every horse with no parsed history (the
population prior), so a race can have a tie at the top, and rank(method='first') then takes the first row -- and the
table's rows within a race may be in finishing order. That would pick the winner. Checked: the share of races with a tie
at the top, the pick's A/E with ties broken at random, and with tied races left out; and the row order itself."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research/strat")
import strat_lib as L
from model import race_shape as RS

df = L.load()
df = df[df["code"].isin(["flat", "aw", "hurdle", "chase"])].copy()
df = RS.add_run_styles(df); df = RS.add_style_projection(df)
df = df.sort_values(["date", "race_time", "track"], kind="stable").reset_index(drop=True)
df = df[df.race_date >= "2016-01-01"]
g = df.groupby("rk")
pos = g.cumcount()
print("within a race, does row order follow the finish? corr(row position, placing):",
      round(float(pd.Series(pos.to_numpy()).corr(df["placing_numerical"].reset_index(drop=True))), 3),
      "| the first row won:", round(float(df[pos == 0]["won"].mean()), 3), "vs every row", round(float(df.won.mean()), 3))
mx = g["p_lead"].transform("max")
ntop = (df["p_lead"] == mx).groupby(df["rk"]).transform("sum")
print("races with a tie at the top of p_lead:", round(float((ntop[df.p_lead == mx] > 1).groupby(df.rk[df.p_lead == mx]).first().mean()), 3))
s = L.priced(df).copy()
rng = np.random.default_rng(0)
s["u"] = rng.random(len(s))
gg = s.groupby("rk")
s["first"] = gg["p_lead"].rank(ascending=False, method="first") == 1
s["rand"] = s.sort_values("u").groupby("rk")["p_lead"].rank(ascending=False, method="first").reindex(s.index) == 1
mx = gg["p_lead"].transform("max")
s["untied"] = (s.p_lead == mx) & ((s.p_lead == mx).groupby(s.rk).transform("sum") == 1)
rows = [L.rule("p_lead pick, ties by row order (as run)", s, s["first"]),
        L.rule("p_lead pick, ties broken at random", s, s["rand"]),
        L.rule("p_lead pick, races with no tie at the top", s, s["untied"])]
L.rules_table(rows)
