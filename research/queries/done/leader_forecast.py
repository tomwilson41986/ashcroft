"""Can a trained model forecast who leads, and do the leaders it finds beat their price?

research/queries/done/pace_draw_bias.py (25 Sep): horses that ACTUALLY led beat
their BSP by 1.3-1.46x, but our projection of who leads (model/race_shape.py,
a recency-weighted average of each horse's past positions) has an AUC of 0.711
and the runners it picks return what the market expects. So the pace edge, if
there is one to have, is in forecasting the lead better than the market does.

A walk-forward LightGBM of "led" (the in-running comment says it led or
disputed) on pre-race inputs only, each from EARLIER DAYS:
    the horse's past positions: led / close up / early position over the last
      1, 3, 5, 10 runs and its career, and how often it started slowly
    the horse's record of leading at this course; the jockey's and trainer's
      share of leaders (decayed over a year)
    the draw (share of the field), field size, code, trip, going, stall placement
    the rivals for the lead: its rank in the field on recent leading, the best
      rival's rate, how many rivals lead often
    first-time headgear; the projection's own p_lead
Folds: fit on every race before T, score the next six months, T from Jan 2023
to Jul 2025 (development window to 31 Mar 2026 only).

Reported: AUC and log loss against the projection; how often the model's
likeliest leader led; the field's expected leaders against those who led; and
the test that matters -- A/E against the BSP of the runners the model thinks
will lead, overall and where it disagrees most with the projection.
"""
import re
import sqlite3
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
pd.set_option("display.width", 220)
t0 = time.time()
WARM, FROM, UNTIL = "2019-01-01", "2021-01-01", "2026-04-01"

from model.blocks.draw_v2 import going_group  # noqa: E402
from model.freshness_features import _Days  # noqa: E402
from model.race_shape import _codes, add_race_shape_features, asof_decayed_mean, day_index, race_code, race_key  # noqa: E402
from model.shrinkage import shrunk_mean  # noqa: E402

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query(f"""
    SELECT race_date, race_time, track, horse_name, jockey_name, trainer, number_of_runners, placing_numerical,
           total_dst_bt, dist_furlongs, race_type, surface_type, comment, stall, stall_positioning,
           going_description, headgear, bfsp
    FROM race_results WHERE race_date >= '{WARM}' AND race_date < '{UNTIL}'""", conn)
d["race_date"] = pd.to_datetime(d["race_date"])
d = d.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)
d, _ = add_race_shape_features(d)
print(f"{len(d):,} rows; race shape built in {time.time() - t0:.0f}s")

day = day_index(d)
race = pd.factorize(race_key(d))[0]
n = pd.to_numeric(d["number_of_runners"], errors="coerce").to_numpy(dtype=float)
rc = d["rs_class"].to_numpy(dtype=float)
led = np.where(np.isfinite(rc), (rc == 0).astype(float), np.nan)
front = np.where(np.isfinite(rc), (rc <= 1).astype(float), np.nan)
epf = d["rs_epf"].to_numpy(dtype=float)
slow = d["comment"].fillna("").str.lower().str.contains(
    r"slowly away|missed (?:the )?break|dwelt|started slowly|slow start|awkward(?:ly)? (?:start|away|leaving)"
    r"|reared (?:as|leaving|start)|lost ground (?:at|leaving) (?:the )?start|anticipated start", regex=True
).to_numpy(dtype=float)
slow = np.where(d["comment"].notna().to_numpy(), slow, np.nan)


def key(s):
    s = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(s.isin(["", "nan", "none"]).to_numpy(), -1, _codes(s))


horse, jockey, trainer, track = key(d["horse_name"]), key(d["jockey_name"]), key(d["trainer"]), key(d["track"])
H = _Days(horse, day)


def windows(v, name):
    pb = H.per_block(v)
    blocks = np.arange(len(H.u))
    cols = {}
    lag = {j: np.where(H.valid & (H.pos >= j), pb[np.clip(blocks - j, 0, None)], np.nan) for j in range(1, 11)}
    cols[f"{name}_l1"] = lag[1]
    for k in (3, 5, 10):
        m = np.column_stack([lag[j] for j in range(1, k + 1)])
        with np.errstate(invalid="ignore"):
            cols[f"{name}_m{k}"] = np.nanmean(np.where(np.isfinite(m), m, np.nan), axis=1)
    f = np.isfinite(pb)
    cs = pd.Series(np.where(f, pb, 0.0)).groupby(H.key).cumsum().to_numpy()
    cn = pd.Series(f.astype(float)).groupby(H.key).cumsum().to_numpy()
    prev = np.clip(blocks - 1, 0, None)
    has = H.valid & (H.pos >= 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols[f"{name}_car"] = np.where(has & (cn[prev] > 0), cs[prev] / cn[prev], np.nan)
    cols[f"{name}_n"] = np.where(has, cn[prev], 0.0)
    return {c: H.to_rows(v_) for c, v_ in cols.items()}


import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=RuntimeWarning)
X = {}
for v, nm in ((led, "led"), (front, "front"), (epf, "epf"), (slow, "slow")):
    X.update(windows(v, nm))
X.pop("front_n"); X.pop("epf_n"); X.pop("slow_n")

# connections and the course, from earlier days
for who, nm, hl in ((jockey, "jk_led", 365.0), (trainer, "tr_led", 365.0)):
    m_, n_ = asof_decayed_mean(who, day, led, who, day, halflife_days=hl)
    X[nm] = shrunk_mean(np.nan_to_num(m_) * n_, n_, 0.125, 50)
ht = np.where((horse >= 0) & (track >= 0), pd.factorize(pd.Series(horse).astype(str) + "|" + pd.Series(track).astype(str))[0], -1)
m_, n_ = asof_decayed_mean(ht, day, led, ht, day)
X["course_led"] = shrunk_mean(np.nan_to_num(m_) * n_, n_, np.nan_to_num(X["led_car"], nan=0.125), 3)
tt = np.where(track >= 0, track, -1)
m_, n_ = asof_decayed_mean(tt, day, led, tt, day, halflife_days=730.0)
X["track_led_rate"] = m_

# the race: draw, conditions, rivals for the lead
stall = pd.to_numeric(d["stall"], errors="coerce").to_numpy(dtype=float)
X["draw_rel"] = np.where((stall > 0) & (n > 1), (stall - 1) / (n - 1), np.nan)
X["field"] = n
X["dist"] = pd.to_numeric(d["dist_furlongs"], errors="coerce").to_numpy(dtype=float)
code = race_code(d)
X["code"] = pd.factorize(code)[0].astype(float)
X["going"] = pd.factorize(going_group(d))[0].astype(float)
X["stalls_pos"] = pd.factorize(d["stall_positioning"].fillna("").astype(str).str.lower())[0].astype(float)
X["track_id"] = track.astype(float)
hg = d["headgear"].fillna("").astype(str).str.strip()
hg_prev = H.to_rows(np.where(H.valid & (H.pos >= 1), np.r_[np.nan, H.per_block((hg != "").astype(float).to_numpy())[:-1]], np.nan))
X["first_headgear"] = np.where((hg != "").to_numpy() & (hg_prev == 0), 1.0, np.where(np.isfinite(hg_prev), 0.0, np.nan))
X["p_lead_proj"] = d["p_lead"].to_numpy(dtype=float)
X["pred_epf_proj"] = d["pred_epf"].to_numpy(dtype=float)
lr = pd.Series(np.nan_to_num(X["led_m5"], nan=-1.0))
g = lr.groupby(race)
X["led_rank"] = g.rank(ascending=False, method="min").to_numpy()
top = g.transform("max").to_numpy()
second = lr.groupby(race).transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else np.nan).to_numpy()
mine = lr.to_numpy()
X["best_rival_led"] = np.where(mine >= top, second, top)
X["rivals_leading"] = (lr >= 0.4).groupby(race).transform("sum").to_numpy() - (mine >= 0.4)
X["led_share"] = np.where(mine > 0, mine / lr.clip(lower=0).groupby(race).transform("sum").to_numpy(), 0.0)
F = pd.DataFrame(X)
feats = list(F.columns)
print(f"{len(feats)} features in {time.time() - t0:.0f}s")

# the market's chance, for the A/E test
bsp = pd.to_numeric(d["bfsp"], errors="coerce")
p = pd.Series(np.where(bsp > 1, 1.0 / bsp, np.nan))
p_mkt = (p / p.groupby(race).transform("sum")).to_numpy()
pos = pd.to_numeric(d["placing_numerical"], errors="coerce").to_numpy(dtype=float)
won = (pos == 1).astype(float)

ok = np.isfinite(led) & (d["race_date"] >= FROM).to_numpy()
dates = d["race_date"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=7, deterministic=True,
              force_col_wise=True)
cats = ["code", "going", "stalls_pos", "track_id"]
pred = np.full(len(d), np.nan)
for T in pd.date_range("2023-01-01", "2025-07-01", freq="6MS"):
    tr = ok & (dates < T).to_numpy()
    te = ok & (dates >= T).to_numpy() & (dates < T + pd.DateOffset(months=6)).to_numpy()
    if te.sum() == 0:
        continue
    ds = lgb.Dataset(F[tr], label=led[tr], categorical_feature=cats, free_raw_data=True)
    b = lgb.train(PARAMS, ds, num_boost_round=400)
    pred[te] = b.predict(F[te])
    print(f"  fold {T.date()}: trained on {tr.sum():,}, scored {te.sum():,} ({time.time() - t0:.0f}s)")


def auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    pos_ = y == 1
    return (r[pos_].sum() - pos_.sum() * (pos_.sum() + 1) / 2) / (pos_.sum() * (~pos_).sum())


te = ok & np.isfinite(pred) & np.isfinite(X["p_lead_proj"])
y = led[te]
print(f"\n## Who leads, {te.sum():,} runners scored {dates[te].min().date()}..{dates[te].max().date()} (base rate {y.mean():.3f})")
for nm, s in (("projection p_lead", X["p_lead_proj"][te]), ("trained model", pred[te])):
    s_ = np.clip(s, 1e-4, 1 - 1e-4)
    ll = -np.mean(y * np.log(s_) + (1 - y) * np.log(1 - s_))
    print(f"   {nm:18s} AUC {auc(y, s):.3f}  log loss {ll:.4f}")
imp = pd.Series(b.feature_importance("gain"), index=feats).sort_values(ascending=False)
print("   the last fold's inputs by gain (top 15):", ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(15).items()))

sub = pd.DataFrame({"race": race[te], "led": y, "pred": pred[te], "proj": X["p_lead_proj"][te],
                    "won": won[te], "p_mkt": p_mkt[te]})
for nm in ("pred", "proj"):
    top1 = sub.loc[sub.groupby("race")[nm].idxmax()]
    print(f"   {'model' if nm == 'pred' else 'projection'}'s likeliest leader led in {top1.led.mean():.1%} of {len(top1):,} races")
rr = sub.groupby("race").agg(exp=("pred", "sum"), real=("led", "sum"), expp=("proj", "sum"))
print(f"   expected leaders against those who led: model {rr.exp.corr(rr.real):.3f}, projection {rr.expp.corr(rr.real):.3f}")


def ae(frame):
    f = frame[frame.p_mkt.notna()]
    e = f.p_mkt.sum()
    return pd.Series({"runners": len(f), "win%": 100 * f.won.mean(), "led%": 100 * f.led.mean(),
                      "A/E": f.won.sum() / e, "z": (f.won.sum() - e) / np.sqrt((f.p_mkt * (1 - f.p_mkt)).sum())})


print("\n## The test that matters: runners by the model's chance of leading, against the price")
sub["band"] = pd.cut(sub.pred, [0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 1.0])
print(sub.groupby("band", observed=True).apply(ae).round(3).to_string())
print("\n   where the model and the projection disagree most (model minus projection, deciles):")
sub["gap"] = sub.pred - sub.proj
sub["gap_band"] = pd.qcut(sub.gap, 10, duplicates="drop")
print(sub.groupby("gap_band", observed=True).apply(ae).round(3).to_string())
top1 = sub.loc[sub.groupby("race")["pred"].idxmax()]
print("\n   the model's likeliest leader in each race:", ae(top1).round(3).to_dict())
print("   ... when it is not the projection's:", ae(top1[top1.index.isin(sub.loc[sub.groupby('race')['proj'].idxmax()].index) == False]).round(3).to_dict())
print(f"\n{time.time() - t0:.0f}s")
