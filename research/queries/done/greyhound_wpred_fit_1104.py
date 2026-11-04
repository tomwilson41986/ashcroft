"""How well the predicted weigh-in fits (the owner, 4 Nov: "model it and fit against the data we know"; read-only).
Batch 9's R2 lines swung from 0.97 to -2 by year, the sign of outlying weights. Here, on the fast loop's cache: the raw
weights' range and how many runs carry an impossible one (outside 15-50 kg), then, each year, the forecast's median and
mean absolute error against the day's weight beside the naive forecast (the last run's weight), on plausible weights
only, and the share of runs the forecast gets within 0.5 kg."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from sources.common import Store                       # noqa: E402
from greyhound import fastloop as fl, blocks           # noqa: E402

df, meta = fl.load_cache(Store())
print("cache", meta.get("hash"), len(df), "rows, last day", meta.get("last_day"))
w = pd.to_numeric(df.get("weight_kg"), errors="coerce")
print("weight_kg: present", int(w.notna().sum()), "missing", int(w.isna().sum()))
print("weight_kg quantiles:", w.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(2).to_dict())
bad = w.notna() & ~w.between(15, 50)
print("impossible weights (outside 15-50 kg):", int(bad.sum()), "e.g.", sorted(w[bad].round(1).unique())[:15])
blocks.attach(df, "wpred")
d = pd.DataFrame({"year": pd.to_datetime(df.t).dt.year, "w": w, "pred": df.wp_pred_weight.astype(float),
                  "prev": df.wp_prev_weight.astype(float)})
ok = d.w.between(15, 50) & d.prev.between(15, 50) & d.pred.notna()
print("\nrows scored:", int(ok.sum()), "of", len(d), "(dropped: no forecast, or an impossible weight today or last run)")
print("\nyear | runs | MAE forecast | MAE last weight | median AE forecast | median AE last weight | within 0.5 kg: forecast / last")
for yr, g in d[ok].groupby("year"):
    ef, en = (g.pred - g.w).abs(), (g.prev - g.w).abs()
    print(f"{yr} | {len(g)} | {ef.mean():.3f} | {en.mean():.3f} | {ef.median():.3f} | {en.median():.3f} | "
          f"{(ef <= 0.5).mean():.1%} / {(en <= 0.5).mean():.1%}")
