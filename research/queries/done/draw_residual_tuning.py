"""Draw cells, second round: measured on rating residuals, and how hard the style split pools.

Development data only (before 2026-04-01). With the pooling from run 19 (eight times the
first guess), this tries the cells on outcomes less their rating-implied part
(model/draw_curve.rating_residual) against raw outcomes, and the shrinkage of the draw-by-
running-style cell toward the draw cell. Scored as run 19: slope fitted 2023-24, within-race
R2 x1000 and slope out of sample on 2025-26Q1, flat and all-weather races from stalls.
"""
import itertools
import sqlite3
import sys
import time

import pandas as pd

sys.path.insert(0, ".")
import model.draw_curve as dc  # noqa: E402
from model.race_shape import add_race_shape, add_run_outcomes, add_run_styles, add_style_projection, race_code  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 200)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, surface_type, number_of_runners, horse_name, stall,
           placing_numerical, total_dst_bt, dist_furlongs, official_rating, comment
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
d = add_race_shape(add_style_projection(add_run_styles(add_run_outcomes(d))))
flat = race_code(d).isin(["flat", "aw"]).to_numpy()
print(f"{len(d):,} rows prepared ({time.time() - t0:.0f}s); DRAW_K {dc.DRAW_K}")


def centred(frame, col):
    x = frame[col].astype(float)
    return (x - x.groupby(frame.raceid).transform("mean")).to_numpy()


def score(frame, feat, y):
    """Slope fitted on 2023-24; (R2 x1000 with it, refitted slope) on 2025-26Q1."""
    fit = (frame.race_date >= "2023-01-01") & (frame.race_date < "2025-01-01")
    test = frame.race_date >= "2025-01-01"
    out = []
    for m in (fit, test):
        f = frame[m & frame[feat].notna() & frame[y].notna()]
        out.append((centred(f, feat), centred(f, y)))
    (xf, yf), (xt, yt) = out
    b = (xf * yf).sum() / (xf * xf).sum()
    return 1000 * (1 - ((yt - b * xt) ** 2).sum() / (yt * yt).sum()), (xt * yt).sum() / (xt * xt).sum()


rows = []
for resid, style_k in itertools.product((False, True), (40.0, 160.0, 640.0, 2560.0)):
    dc.DRAW_STYLE_K = style_k
    t = time.time()
    f, _ = dc.add_draw_curve(d.copy(), residualise=resid)
    f = f[flat & f.dc_edge_lbs.notna().to_numpy()]
    r = dict(residualise=resid, style_k=style_k)
    for feat, y in (("dc_edge_lbs", "rs_lbs_c"), ("dc_edge_nfp", "rs_nfp_c"), ("dc_edge_style_lbs", "rs_lbs_c")):
        r2, so = score(f, feat, y)
        r[f"{feat}_r2"] = r2
        r[f"{feat}_slope"] = so
    r["secs"] = time.time() - t
    rows.append(r)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"\ndone in {time.time() - t0:.0f}s")
