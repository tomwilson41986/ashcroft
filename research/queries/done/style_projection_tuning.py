"""How far back, and how hard toward the population, should a horse's run style be projected?

Development data only (before 2026-04-01). The projection (model/race_shape.add_style_projection)
is rebuilt for a grid of recency half-lives (in the horse's runs), shrinkage strengths toward
the population, and with or without a separate level for the horse's runs in today's code,
and scored on 2023-01 onwards against the class and early position the run's comment gave:

  brier       four-class Brier score (lead / prominent / mid / rear), lower is better
  skill       1 - brier / brier of the base rates
  lead_ll     log loss of p_lead for "led or disputed", lower is better
  epf_corr    correlation of the projected early position with the parsed one

Also: parser coverage after the vocabulary was widened, and what is still unread.
"""
import itertools
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.race_shape import add_run_styles, add_style_projection, race_code  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 200)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, surface_type, number_of_runners, horse_name,
           dist_furlongs, comment
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
d = add_run_styles(d)
has = d.comment.fillna("").str.strip().str.len() > 0
print(f"{len(d):,} rows; position read from {d.rs_style.notna()[has].mean():.4f} of comments "
      f"({time.time() - t0:.0f}s)")
print("class shares:", d.rs_class.value_counts(normalize=True).sort_index().round(3).to_dict())
unread = d.loc[has & d.rs_style.isna(), "comment"].str.lower().str.replace(r"[^a-z ]", " ", regex=True)
print(f"still unread: {len(unread):,}; commonest openings:")
print(unread.str.split().str[:3].str.join(" ").value_counts().head(25).to_string())

score = (d.race_date >= "2023-01-01") & d.rs_class.notna()
act = np.zeros((int(score.sum()), 4))
act[np.arange(act.shape[0]), d.loc[score, "rs_class"].astype(int).to_numpy()] = 1
base = ((act.mean(axis=0) - act) ** 2).sum(axis=1).mean()
y_lead = act[:, 0]
print(f"\nscored runs: {act.shape[0]:,}; base-rate brier {base:.4f}")

rows = []
for hl, pr, cp in itertools.product((2.0, 4.0, 8.0, 16.0), (1.0, 2.0, 4.0), (None, 2.0)):
    t = time.time()
    p = add_style_projection(d[["race_date", "race_time", "track", "race_type", "surface_type", "number_of_runners",
                                "horse_name", "dist_furlongs", "raceid", "rs_style", "rs_class", "rs_epf"]].copy(),
                             halflife_runs=hl, prior_runs=pr, code_prior_runs=cp)
    P = p.loc[score, ["p_lead", "p_prom", "p_mid", "p_rear"]].to_numpy()
    brier = ((P - act) ** 2).sum(axis=1).mean()
    pl = np.clip(P[:, 0], 1e-6, 1 - 1e-6)
    ll = -(y_lead * np.log(pl) + (1 - y_lead) * np.log(1 - pl)).mean()
    ok = score & p.rs_epf.notna()
    corr = np.corrcoef(p.loc[ok, "pred_epf"], p.loc[ok, "rs_epf"])[0, 1]
    rows.append(dict(halflife_runs=hl, prior_runs=pr, code_level=cp if cp else "off", brier=brier,
                     skill=1 - brier / base, lead_ll=ll, epf_corr=corr, secs=time.time() - t))
res = pd.DataFrame(rows).sort_values("brier")
print("\nprojection grid, best first")
print(res.round(4).to_string(index=False))

best = res.iloc[0]
p = add_style_projection(d.copy(), halflife_runs=best.halflife_runs, prior_runs=best.prior_runs,
                         code_prior_runs=None if best.code_level == "off" else float(best.code_level))
e = p[score]
e = e.assign(code=race_code(e))
print("\nbest setting, calibration of p_lead (deciles): mean p_lead vs share that led")
q = pd.qcut(e.p_lead.rank(method="first"), 10, labels=False)
print(pd.DataFrame({"p_lead": e.p_lead, "led": (e.rs_class == 0).astype(float)}).groupby(q).mean().round(3).T.to_string())
print("\nbest setting, skill by code:")
for c, g in e.groupby("code"):
    a = np.zeros((len(g), 4))
    a[np.arange(len(g)), g.rs_class.astype(int).to_numpy()] = 1
    Pg = g[["p_lead", "p_prom", "p_mid", "p_rear"]].to_numpy()
    b0 = ((a.mean(axis=0) - a) ** 2).sum(axis=1).mean()
    print(f"  {c:7s} runs {len(g):>7,}  skill {1 - ((Pg - a) ** 2).sum(axis=1).mean() / b0:.3f}")
print(f"\ndone in {time.time() - t0:.0f}s")
