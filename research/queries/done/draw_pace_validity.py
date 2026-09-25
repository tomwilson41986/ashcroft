"""Do the rebuilt draw and position metrics predict the result better than the ones the model serves?

Second run (run 21): the blocks as tuned in runs 18-20 -- projection prior 4 runs with no
per-code level, draw and position cells pooled 8x and 3x harder, the draw-by-style cell
pooled at 160.

Development data only: 2021-01-01 to 2026-03-31 (the holdout is not read). Every metric is
built over the whole span with earlier-days-only statistics, and scored on 2023-01 onwards
(two years of burn-in for the cells).

Outcomes, centred within each race: finishing position (rs_nfp_c), pounds beaten less than
the race average (rs_lbs_c) and the win (won - 1/n).

1. Each feature alone: within-race demeaned, slope through the origin, race-clustered t,
   share of within-race variance explained.
2. Each SET jointly, out of sample: least squares on the within-race demeaned features,
   fitted on 2023-2024, scored on 2025-2026Q1. Old set, new set, both.
3. Calibration: the new edges binned, against what happened.
4. The benefit of expected position within the expected shape, 2023 onwards.
5. Draw cells: does a course and trip's predicted low-minus-high edge hold up in races it
   had not seen? Plus whether stall placement or going should split the cells.
"""
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.draw_curve import add_draw_curve  # noqa: E402
from model.draw_metrics import DrawMetricsEngine, classify_going  # noqa: E402
from model.pace_metrics import PaceMetricsEngine  # noqa: E402
from model.race_shape import STYLE_CLASSES, add_race_shape_features  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 220)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, surface_type, number_of_runners, horse_name, stall,
           stall_positioning, placing_numerical, total_dst_bt, dist_furlongs, going_description, comment,
           official_rating, jockey_name, trainer, track_direction, rail_move
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
print(f"{len(d):,} rows, {d.raceid.nunique():,} races ({time.time() - t0:.0f}s)")

# --- the new blocks ---------------------------------------------------------------------
t1 = time.time()
d, shape_cols = add_race_shape_features(d)
d, draw_cols = add_draw_curve(d)
d["_going"] = d.going_description.map(classify_going)
alt = {}
for name, key in (("stall_positioning", "stall_positioning"), ("going", "_going")):
    a, _ = add_draw_curve(d[["raceid", "race_date", "track", "race_type", "surface_type", "number_of_runners",
                             "horse_name", "stall", "dist_furlongs", "rs_nfp_c", "rs_lbs_c", key]].copy(),
                          extra_key=key)
    alt[name] = a["dc_edge_lbs"].to_numpy()
print(f"new blocks in {time.time() - t1:.0f}s")

# --- the served ones, on the same rows ----------------------------------------------------
t2 = time.time()
n = pd.to_numeric(d.number_of_runners, errors="coerce")
pos = pd.to_numeric(d.placing_numerical, errors="coerce")
d["NFP"] = (n - pos) / (n - 1).where(n > 1)
d["stall_num_raw"] = pd.to_numeric(d.stall, errors="coerce")
d["draw_relative"] = (d.stall_num_raw - 1) / (n - 1).replace(0, np.nan)
d["draw_quartile"] = pd.to_numeric(pd.cut(d.draw_relative, [0, .25, .5, .75, 1.0], labels=[1, 2, 3, 4],
                                          include_lowest=True), errors="coerce")
old_in = d[["raceid", "race_date", "race_time", "track", "horse_name", "number_of_runners", "placing_numerical",
            "comment", "NFP", "jockey_name", "trainer", "dist_furlongs", "stall", "stall_num_raw", "draw_relative",
            "draw_quartile", "official_rating", "going_description", "track_direction", "rail_move"]].copy()
old_in["_row"] = np.arange(len(old_in))
for c in ("jockey_name", "trainer", "comment", "going_description", "track_direction", "rail_move"):
    old_in[c] = old_in[c].fillna("").astype(str)
old = PaceMetricsEngine().calculate(old_in)
old = DrawMetricsEngine().calculate(old)
old = old.sort_values("_row").reset_index(drop=True)
OLD_DRAW = ["draw_bias_ev", "draw_bias_ev_adj", "draw_bias_ev_stall", "draw_bias_ev_course", "td_draw_bias",
            "tdg_draw_bias", "draw_bias_alignment", "going_draw_alignment", "draw_advantage_field_adj",
            "draw_advantage_composite", "draw_pct_adj", "draw_relative"]
OLD_PACE = ["pred_epf_norm", "predicted_lead_prob", "pace_suit", "pace_advantage", "track_style_fit",
            "lead_prob_edge", "pace_position_delta", "horse_epf_norm_mean", "horse_career_early_pos",
            "lone_front_runner", "lead_competition", "pace_mismatch", "front_sustainability"]
for c in OLD_DRAW + OLD_PACE:
    d[f"old_{c}"] = old[c].to_numpy() if c in old.columns else np.nan
print(f"served engines in {time.time() - t2:.0f}s")

NEW_DRAW = ["dc_edge_nfp", "dc_edge_lbs", "dc_edge_rel_lbs", "dc_draw_pct", "dc_edge_style_lbs",
            "dc_edge_style_rel_lbs"]
NEW_PACE = ["p_lead", "p_prom", "p_mid", "pred_epf", "lead_share", "lead_rank", "rel_epf",
            "pv_exp_nfp", "pv_exp_lbs", "pv_act_nfp", "pv_act_lbs"]
OLD_DRAW = [f"old_{c}" for c in OLD_DRAW]
OLD_PACE = [f"old_{c}" for c in OLD_PACE]

d["won_c"] = (pos == 1).astype(float) - 1 / n
d.loc[pos.isna(), "won_c"] = np.nan
e = d[(d.race_date >= "2023-01-01") & pos.notna()].copy()
flat = e.race_type.fillna("").str.lower().str.contains("hurdle|chase|bumper|nh flat") == False  # noqa: E712
print(f"scored: {len(e):,} runners from 2023, {flat.mean():.2f} of them on the flat")
OUT = {"nfp": "rs_nfp_c", "lbs": "rs_lbs_c", "win": "won_c"}


def demean(frame, col):
    x = frame[col].astype(float)
    return x - x.groupby(frame.raceid).transform("mean")


def single(frame, cols, label):
    rows = []
    for c in cols:
        r = {"feature": c}
        for nm, yc in OUT.items():
            ok = frame[c].notna() & frame[yc].notna()
            f = frame[ok]
            x, y = demean(f, c).to_numpy(), demean(f, yc).to_numpy()
            sxx = (x * x).sum()
            if sxx <= 0:
                r[f"{nm}_t"], r[f"{nm}_r2x1000"] = np.nan, np.nan
                continue
            b = (x * y).sum() / sxx
            s = pd.Series(x * (y - b * x)).groupby(f.raceid.to_numpy()).sum()
            r[f"{nm}_t"] = b / np.sqrt((s ** 2).sum() / sxx ** 2)
            r[f"{nm}_r2x1000"] = 1000 * b * b * sxx / (y * y).sum()
        r["coverage"] = frame[c].notna().mean()
        rows.append(r)
    print(f"\n{label}")
    print(pd.DataFrame(rows).set_index("feature").round(2).to_string())


def joint(frame, sets, label):
    fit = frame.race_date < "2025-01-01"
    rows = []
    for nm, yc in OUT.items():
        f = frame[frame[yc].notna()]
        y = demean(f, yc).to_numpy()
        r = {"outcome": nm}
        for sname, cols in sets.items():
            X = np.column_stack([demean(f, c).fillna(0.0).to_numpy() for c in cols])
            m = (f.race_date < "2025-01-01").to_numpy()
            beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
            pr = X[~m] @ beta
            r[sname] = 1000 * (1 - ((y[~m] - pr) ** 2).sum() / (y[~m] ** 2).sum())
        rows.append(r)
    print(f"\n{label} (out-of-sample within-race R2 x1000: fit 2023-24, score 2025-26Q1; "
          f"{int((~fit).sum()):,} scored runners)")
    print(pd.DataFrame(rows).set_index("outcome").round(3).to_string())


print("\n== 0. run-style projection against the early position the comment gave (2023 on)")
ok = e.rs_epf.notna()
lead = (e.rs_class == 0).astype(float)
for nm, col in (("new pred_epf", "pred_epf"), ("served pred_epf_norm", "old_pred_epf_norm")):
    m = ok & e[col].notna()
    print(f"  {nm:22s} corr with parsed early position {np.corrcoef(e.loc[m, col], e.loc[m, 'rs_epf'])[0, 1]:.3f}"
          f" on {int(m.sum()):,} runs")
for nm, col in (("new p_lead", "p_lead"), ("served predicted_lead_prob", "old_predicted_lead_prob")):
    m = e.rs_class.notna() & e[col].notna()
    pl = e.loc[m, col].clip(1e-6, 1 - 1e-6)
    ll = -(lead[m] * np.log(pl) + (1 - lead[m]) * np.log(1 - pl)).mean()
    print(f"  {nm:26s} log loss for leading {ll:.4f} (base rate {-(lead[m].mean() * np.log(lead[m].mean()) + (1 - lead[m].mean()) * np.log(1 - lead[m].mean())):.4f})")

ef = e[flat & e.dc_draw_pct.notna()]
print("\n== 1. single features, flat and all-weather races from stalls")
single(ef, NEW_DRAW, "NEW draw")
single(ef, OLD_DRAW, "SERVED draw")
print("\n== 1b. single features, every race with a result")
single(e, NEW_PACE, "NEW position")
single(e, OLD_PACE, "SERVED position")

print("\n== 2. sets, jointly and out of sample")
joint(ef, {"new": NEW_DRAW, "served": OLD_DRAW, "both": NEW_DRAW + OLD_DRAW}, "draw, flat/aw from stalls")
joint(e, {"new": NEW_PACE, "served": OLD_PACE, "both": NEW_PACE + OLD_PACE}, "position, all races")
joint(e[flat], {"new": NEW_PACE, "served": OLD_PACE, "both": NEW_PACE + OLD_PACE}, "position, flat/aw")
joint(e[~flat], {"new": NEW_PACE, "served": OLD_PACE, "both": NEW_PACE + OLD_PACE}, "position, jumps")

print("\n== 3. calibration (edges demeaned within race, deciles; realised outcome demeaned too)")
for feat, yc in (("dc_edge_lbs", "rs_lbs_c"), ("dc_edge_nfp", "rs_nfp_c")):
    f = ef[ef[feat].notna() & ef[yc].notna()]
    x, y = demean(f, feat), demean(f, yc)
    q = pd.qcut(x.rank(method="first"), 10, labels=False)
    t = pd.DataFrame({"pred": x, "real": y}).groupby(q).mean()
    print(f"\n{feat}: predicted vs realised by decile")
    print(t.round(3).T.to_string())
for feat, yc in (("pv_exp_lbs", "rs_lbs_c"), ("pv_act_lbs", "rs_lbs_c"), ("pv_exp_nfp", "rs_nfp_c")):
    f = e[e[feat].notna() & e[yc].notna()]
    x, y = demean(f, feat), demean(f, yc)
    q = pd.qcut(x.rank(method="first"), 10, labels=False)
    t = pd.DataFrame({"pred": x, "real": y}).groupby(q).mean()
    print(f"\n{feat}: predicted vs realised by decile")
    print(t.round(3).T.to_string())

print("\n== 4. what the expected position was worth, by expected shape (2023 onwards)")
e["exp_pos"] = e[["p_lead", "p_prom", "p_mid", "p_rear"]].to_numpy().argmax(axis=1)
e["exp_pos"] = e.exp_pos.map(dict(enumerate(STYLE_CLASSES)))
e["shape"] = e.shape_bin.map({0.0: "0 no natural leader", 1.0: "1 one natural leader", 2.0: "2 contested lead"})
e["code"] = np.where(flat, "flat/aw", "jumps")
t = e.groupby(["code", "shape", "exp_pos"]).agg(runs=("rs_lbs_c", "size"), lbs=("rs_lbs_c", "mean"),
                                                 nfp=("rs_nfp_c", "mean"), win=("won_c", "mean"))
print(t.round(3).to_string())
print("\nraces by shape:", e.drop_duplicates("raceid").groupby(["code", "shape"]).size().to_dict())

print("\n== 5. draw cells out of sample: predicted and realised low-minus-high, by course and trip")
ef = ef.assign(low=ef.dc_draw_pct <= 0.25, high=ef.dc_draw_pct >= 0.75, dist=ef.dist_furlongs.round(0))
late = ef[ef.race_date >= "2025-01-01"]
keys = ["track", "dist"]
lo, hi = late[late.low].groupby(keys), late[late.high].groupby(keys)
cell = pd.DataFrame({
    "races": late.groupby(keys)["raceid"].nunique(),
    "pred": lo["dc_edge_lbs"].mean() - hi["dc_edge_lbs"].mean(),
    "real": lo["rs_lbs_c"].mean() - hi["rs_lbs_c"].mean(),
}).dropna()
big = cell[cell.races >= 40]
print(f"{len(big)} course x trip cells with 40+ races in 2025-26Q1: corr(predicted, realised) "
      f"{np.corrcoef(big.pred, big.real)[0, 1]:.3f}, weighted slope "
      f"{np.polyfit(big.pred, big.real, 1, w=np.sqrt(big.races))[0]:.2f}")
print(big.reindex(big.pred.abs().sort_values(ascending=False).index).head(15).round(2).to_string())
f = ef[ef.rs_lbs_c.notna()].copy()
for name, v in alt.items():
    f[f"alt_{name}"] = pd.Series(v, index=d.index).loc[f.index]
joint(f, {"course x trip x field": ["dc_edge_lbs"], "+ stall placement": ["alt_stall_positioning"],
          "+ going": ["alt_going"]}, "draw cells split further")
print(f"\ndone in {time.time() - t0:.0f}s")
