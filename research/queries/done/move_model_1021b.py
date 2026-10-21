"""Round 2 of the model prototype (21 Oct; done/move_model_1021.py run 39595362090, read-only). Round 1 found the
richer move models predict the size of the move better on average (correlation with the CLV 0.36 LightGBM, 0.35 ridge,
0.29 served) and the catalogue corrects the 06:00 price (|log error| 0.396 -> 0.377), but at equal volume the served
closing model still picked the best horses. So, on the same walk-forward days:

- the served model with a veto: back only where the served model clears 3% and LightGBM (or ridge) does not expect a
  loss (prediction >= 0);
- the served model and ridge averaged;
- the served closing model fed the corrected 06:00 price (ridge correction of round 1, walk-forward) in place of the
  model's own: does a better price into the same blend pick better horses?
Each at the bar (the rule) and at equal volume."""
import os
import runpy
import sys

import numpy as np
import pandas as pd

sys.argv = ["move_model_1021.py"]
g = runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "move_model_1021.py"), run_name="__main__")
S, T, rb, served, BAR = g["S"], g["T"], g["rb"], g["served"], g["BAR"]

corr = T[["market_id", "selection_id", "ridge", "gbm"]].rename(columns={"ridge": "corr_ridge", "gbm": "corr_gbm"})
S = S.merge(corr, on=["market_id", "selection_id"], how="left")
S["pb_corr"] = S.predicted_bfsp * np.exp(S.corr_ridge.fillna(0.0))
parts = []
for d, x in S.groupby("day"):
    x = x.copy()
    gg = rb.race_codes(x.snap)
    inp = rb.closing_inputs(x.assign(race=x.snap, morningwap=x.now, morning_vol=0.0, predicted_bfsp=x.pb_corr),
                            race_col="race")
    x["p_served_corr"] = x.back1 * served.book * served.predict(inp, gg) - 1
    parts.append(x)
S = pd.concat(parts, ignore_index=True)
S["guard_corr"] = S.back1 >= S.pb_corr - 1e-9
S["p_veto_gbm"] = np.where(S.p_gbm >= 0, S.p_served, -1.0)
S["p_veto_ridge"] = np.where(S.p_ridge >= 0, S.p_served, -1.0)
S["p_avg"] = (S.p_served + S.p_ridge) / 2
ARMS = {"served (now)": ("p_served", "guard"), "served, LightGBM veto": ("p_veto_gbm", "guard"),
        "served, ridge veto": ("p_veto_ridge", "guard"), "served and ridge averaged": ("p_avg", "guard"),
        "served on the corrected 06:00 price": ("p_served_corr", "guard_corr")}


def rule(col, guard):
    sel = S[(S[col] >= BAR) & S[guard] & (S.back1 <= 200)].sort_values("polled_utc").drop_duplicates(
        ["day", "market_id", "selection_id"])
    w = np.clip(400 / (sel.back1 - 1), 15, 200)
    return sel, w


rows, served_n = {}, None
for name, (col, guard) in ARMS.items():
    sel, w = rule(col, guard)
    if served_n is None:
        served_n = sel.groupby("day").size()
    by_day = sel.assign(w=w).groupby("day").apply(lambda x: (x.z * x.w).sum() / x.w.sum())
    picks = []
    for d, x in S[S[guard] & (S.back1 <= 200)].groupby("day"):
        picks.append(x.sort_values(col, ascending=False).drop_duplicates(["market_id", "selection_id"])
                     .head(int(served_n.get(d, 0))))
    eq = pd.concat(picks)
    we = np.clip(400 / (eq.back1 - 1), 15, 200)
    rows[name] = {"rule_horses": len(sel), "rule_staked": float(w.sum()), "rule_clv": float((sel.z * w).sum() / w.sum()),
                  "rule_clv_x_stake": float((sel.z * w).sum()), "days_ahead_of_served": None,
                  "equal_volume_clv": float((eq.z * we).sum() / we.sum()), "equal_volume_clv_x_stake": float((eq.z * we).sum()),
                  "corr_with_clv": float(S[col].where(S[col] > -1).corr(S.z))}
    rows[name]["_by_day"] = by_day
base = rows["served (now)"]["_by_day"]
for name in rows:
    rows[name]["days_ahead_of_served"] = int((rows[name]["_by_day"] > base.reindex(rows[name]["_by_day"].index)).sum())
print("\n== round 2: the served model with a veto, averaged, and fed the corrected 06:00 price (14-20 Oct)")
print(pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "_by_day"} for k, v in rows.items()}).T.round(4).to_string())
print("\n   the rule's CLV by day")
print(pd.DataFrame({k: v["_by_day"] for k, v in rows.items()}).round(4).to_string())
