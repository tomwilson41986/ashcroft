"""How hard should the draw and position cells pool, and how fast should they forget?

Development data only (before 2026-04-01). Run 18's comparison found the draw edge about
twice as extreme as the outcomes it predicts (calibration slope ~0.5) while the course x
trip cells held up out of sample (corr 0.81): the finest cells are too noisy, so the
pooling is too weak. This scores a grid on races it had not seen:

  fit        slope of the race-centred outcome on the race-centred edge, 2023-2024
  r2_oos     out-of-sample within-race R2 x1000 on 2025-26Q1, using the fitted slope
  slope_oos  the same slope refitted on 2025-26Q1 (1 = calibrated)

Draw: shrinkage multiplier on every level x half-life x a going split at the finest cell;
flat and all-weather races from stalls, outcome pounds beaten (rs_lbs_c). The draw edge by
running style is scored alongside. Position value: shrinkage multiplier x half-life, all
races, outcome pounds beaten.
"""
import itertools
import sqlite3
import sys
import time

import pandas as pd

sys.path.insert(0, ".")
from model.draw_curve import DRAW_K, add_draw_curve  # noqa: E402
from model.draw_metrics import classify_going  # noqa: E402
from model.race_shape import (PV_K, add_position_value, add_race_shape, add_run_outcomes, add_run_styles,  # noqa: E402
                              add_style_projection, race_code)

t0 = time.time()
pd.set_option("display.width", 200)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, surface_type, number_of_runners, horse_name, stall,
           placing_numerical, total_dst_bt, dist_furlongs, going_description, comment
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
d = add_run_outcomes(d)
d = add_run_styles(d)
d = add_style_projection(d)
d = add_race_shape(d)
d["_going"] = d.going_description.map(classify_going)
print(f"{len(d):,} rows prepared ({time.time() - t0:.0f}s)")

fit = (d.race_date >= "2023-01-01") & (d.race_date < "2025-01-01")
test = d.race_date >= "2025-01-01"


def centred(frame, col):
    x = frame[col].astype(float)
    return (x - x.groupby(frame.raceid).transform("mean")).to_numpy()


def score(frame, feat, y="rs_lbs_c"):
    """(fit slope, oos R2 x1000 with the fitted slope, oos slope)."""
    out = []
    for m in (fit, test):
        f = frame[m & frame[feat].notna() & frame[y].notna()]
        out.append((centred(f, feat), centred(f, y)))
    (xf, yf), (xt, yt) = out
    b = (xf * yf).sum() / (xf * xf).sum()
    r2 = 1000 * (1 - ((yt - b * xt) ** 2).sum() / (yt * yt).sum())
    return b, r2, (xt * yt).sum() / (xt * xt).sum()


flat = race_code(d).isin(["flat", "aw"]).to_numpy()
rows = []
for mult, hl, going in itertools.product((1, 2, 4, 8, 16), (365.0, 730.0, 1460.0), (False, True)):
    t = time.time()
    f, _ = add_draw_curve(d.copy(), halflife_days=hl, ks=tuple(k * mult for k in DRAW_K),
                          extra_key="_going" if going else None)
    f = f[flat & f.dc_edge_lbs.notna().to_numpy()]
    b, r2, bo = score(f, "dc_edge_lbs")
    sb, sr2, sbo = score(f, "dc_edge_style_lbs")
    rows.append(dict(k_mult=mult, halflife=hl, going=going, fit_slope=b, r2_oos=r2, slope_oos=bo,
                     style_r2_oos=sr2, style_slope_oos=sbo, secs=time.time() - t))
res = pd.DataFrame(rows).sort_values("r2_oos", ascending=False)
print("\nDRAW, flat/aw from stalls, pounds beaten (best first)")
print(res.round(3).to_string(index=False))

rows = []
for mult, hl in itertools.product((0.5, 1, 2, 4), (730.0, 1095.0, 2190.0)):
    t = time.time()
    f = add_position_value(d.copy(), halflife_days=hl, ks=tuple(k * mult for k in PV_K))
    r = dict(k_mult=mult, halflife=hl)
    for feat in ("pv_exp_lbs", "pv_act_lbs"):
        b, r2, bo = score(f, feat)
        r.update({f"{feat}_r2_oos": r2, f"{feat}_slope_oos": bo})
    r["secs"] = time.time() - t
    rows.append(r)
res = pd.DataFrame(rows).sort_values("pv_exp_lbs_r2_oos", ascending=False)
print("\nPOSITION VALUE, all races, pounds beaten (best first)")
print(res.round(3).to_string(index=False))
print(f"\ndone in {time.time() - t0:.0f}s")
