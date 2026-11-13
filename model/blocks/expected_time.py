"""Expected time: a forecast of the speed figure each horse will run today, and the finishing order it implies.

The time figure block (model/blocks/time_figure.py) rates each past run from the clock and hands the trees windows
of it. Nothing forecasts the figure a horse will run TODAY, at this trip, on this ground, at this course, for this
yard and rider, in this race set-up (the owner, 12 Nov 2026: "predicting what the expected time for each horse in the
race will be"). Here a first-stage model does: a gradient-boosted regression of a run's time figure (pounds; a pound
is a fixed share of a length at the trip, so the figure is the time on a scale that travels across courses) on what
was known before the run:

    past performances       the horse's figures: last, last 3, best of 5, career, decayed over a year; runs; days off
    times on conditions     its figures at today's trip band, going group, course and code (mean and runs)
    times by pedigree       the sire's and the damsire's runners' figures, and the sire's at today's trip and going
    times by connections    the trainer's, the jockey's and the pairing's runners' figures; the trainer's on this code
    the race set-up         trip, going, code, runners, the stall as a share of the field, weight, age, the mark,
                            and the pace projection (model/blocks/pace_v2.py: early position, its rank, the race's
                            heat, a contested lead, a closer's set-up)

The model is refitted every FOLD_DAYS on the runs of the TRAIN_DAYS before the fold only, and forecasts the fold's
runs, so no run's forecast has seen its own day or any later one; every input is from earlier days.

    et_pred      the forecast figure (pounds)
    et_gap       et_pred less the best rival's
    et_z         et_pred standardised within the race
    et_rank      its rank in the race (1 the fastest)
    et_lengths   the forecast distance behind the race's fastest, in lengths at the trip
    et_win       the forecast's share of the race: a softmax of et_pred over the field at WIN_SCALE pounds
    et_uplift    et_pred less the horse's own decayed mean figure: what today's conditions, connections and set-up add
    et_inputs    how many of the horse's own figure inputs are known (0 for a debutant)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import time_figure as _tf
from model.blocks.cond_form import GOING, classify_going
from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.perf_figures import lbs_per_length
from model.race_shape import DIST_BANDS, asof_decayed_mean, bands, day_index, race_code, race_key

FEATURES = ["et_pred", "et_gap", "et_z", "et_rank", "et_lengths", "et_win", "et_uplift", "et_inputs"]
POST_RACE: set[str] = set()
PACE = ["p2_pred", "p2_pred_rank", "p2_race_heat", "p2_contested_lead", "p2_closer_setup"]
READS = sorted(set(_tf.READS) | {"trainer", "jockey_name", "stallion", "dam_stallion", "number_of_runners", "stall",
                                 "horse_age", "official_rating"} | set(PACE))
AFTER = ("pace_v2",)
FOLD_DAYS = 365
TRAIN_DAYS = 4 * 365
MIN_TRAIN = 20000
N_TREES = 250
MIN_CHILD = 200
WIN_SCALE = 8.0
TEST_REACH = {"FOLD_DAYS": 10, "MIN_TRAIN": 60, "N_TREES": 20, "MIN_CHILD": 5}


def _key(*parts) -> np.ndarray:
    """One integer key from several code arrays; -1 where any part is missing."""
    bad = np.zeros(len(parts[0]), dtype=bool)
    out = np.zeros(len(parts[0]), dtype=np.int64)
    for p in parts:
        p = np.asarray(p, dtype=np.int64)
        bad |= p < 0
        out = out * (int(p.max()) + 2) + np.where(p < 0, 0, p)
    return np.where(bad, -1, pd.factorize(out, sort=True)[0]).astype(np.int64)


def _name_codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The first-stage model's inputs (all from earlier days or the card), the run's own figure, and the day."""
    fig = _tf.figures(df)["fig"]
    day = day_index(df)
    horse = _name_codes(df["horse_name"])
    trainer, jockey = _name_codes(_col(df, "trainer")), _name_codes(_col(df, "jockey_name"))
    sire, damsire = _name_codes(_col(df, "stallion")), _name_codes(_col(df, "dam_stallion"))
    course = _name_codes(_col(df, "track"))
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    tb = pd.Series(bands(dist, DIST_BANDS)).fillna(-1).to_numpy().astype(np.int64)
    gg = pd.to_numeric(_col(df, "going_description").map(classify_going).map(GOING), errors="coerce").fillna(-1) \
        .to_numpy().astype(np.int64)
    code = pd.factorize(race_code(df), sort=True)[0].astype(np.int64)

    X: dict[str, np.ndarray] = {}

    def hist(tag, key, half):
        m, n = asof_decayed_mean(key, day, fig, key, day, halflife_days=half)
        X[f"{tag}_fig"], X[f"{tag}_n"] = m, n

    hist("h", horse, 365.0)
    hist("h_trip", _key(horse, tb), 730.0)
    hist("h_going", _key(horse, gg), 730.0)
    hist("h_course", _key(horse, course), 730.0)
    hist("h_code", _key(horse, code), 730.0)
    hist("sire", sire, 1095.0)
    hist("sire_trip", _key(sire, tb), 1095.0)
    hist("sire_going", _key(sire, gg), 1095.0)
    hist("damsire", damsire, 1095.0)
    hist("tr", trainer, 365.0)
    hist("tr_code", _key(trainer, code), 365.0)
    hist("jk", jockey, 365.0)
    hist("tr_jk", _key(trainer, jockey), 730.0)

    H = _Days(horse, day)
    lad = _ladder(H, H.per_block(fig))
    for w in ("l1", "m3", "car"):
        X[f"h_{w}"] = H.to_rows(lad[w])
    X["h_runs"] = H.to_rows(lad["_cnt"])
    X["h_best5"] = _tf.build(df)["tf_best5"].to_numpy(float) if "tf_best5" not in df.columns \
        else pd.to_numeric(df["tf_best5"], errors="coerce").to_numpy(float)
    prev = np.clip(np.arange(len(H.u)) - 1, 0, None)
    X["h_days_off"] = H.to_rows(np.where(H.valid & (H.pos >= 1), H.day - H.day[prev], np.nan))

    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    stall = pd.to_numeric(_col(df, "stall"), errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        X["stall_pct"] = np.where((stall > 0) & (n > 1), (stall - 1) / (n - 1), np.nan)
        X["log_trip"] = np.where(dist > 0, np.log(np.where(dist > 0, dist, 1.0)), np.nan)
    X["runners"], X["trip_band"], X["going"], X["code"] = n, tb.astype(float), gg.astype(float), code.astype(float)
    for c, name in (("pounds", "weight"), ("horse_age", "age"), ("official_rating", "mark")):
        X[name] = pd.to_numeric(_col(df, c), errors="coerce").to_numpy(float)
    for c in PACE:
        X[c] = pd.to_numeric(_col(df, c), errors="coerce").to_numpy(float)
    return pd.DataFrame(X, index=df.index), np.asarray(fig, dtype=float), day


def forecast(X: pd.DataFrame, y: np.ndarray, day: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Walk-forward: each fold of FOLD_DAYS is forecast by a model fitted on the TRAIN_DAYS before it. The rows are
    fitted in `order` (day, race, horse), so the frame's own row order cannot move a forecast."""
    import lightgbm as lgb

    pred = np.full(len(X), np.nan)
    ok = day >= 0
    if not ok.any():
        return pred
    d0, d1 = int(day[ok].min()), int(day[ok].max())
    A = X.to_numpy(dtype=np.float32)
    for start in range(d0 + FOLD_DAYS, d1 + 1, FOLD_DAYS):
        tr = ok & (day < start) & (day >= start - TRAIN_DAYS) & np.isfinite(y)
        te = ok & (day >= start) & (day < start + FOLD_DAYS)
        if tr.sum() < MIN_TRAIN or not te.any():
            continue
        m = lgb.LGBMRegressor(n_estimators=N_TREES, learning_rate=0.06, num_leaves=31, min_child_samples=MIN_CHILD,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=7,
                              n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
        rows = order[tr[order]]
        m.fit(A[rows], y[rows])
        pred[te] = m.predict(A[te])
    return pred


def build(df: pd.DataFrame) -> pd.DataFrame:
    X, y, day = inputs(df)
    rk = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    pred = forecast(X, y, day, np.lexsort((_name_codes(df["horse_name"]), rk.to_numpy(), day)))
    p = pd.Series(pred, index=df.index)
    g = p.groupby(rk)
    top = g.transform("max")
    cnt = g.transform("count")
    second = p.where(p < top).groupby(rk).transform("max")
    ties = (p == top).groupby(rk).transform("sum")
    rival = np.where((p == top) & (ties < 2), second, top)
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        lpl = np.asarray(lbs_per_length(np.where(dist > 0, dist, np.nan)), dtype=float)
        e = np.exp((p - top) / WIN_SCALE)
        cols = {"et_pred": pred, "et_gap": np.where(cnt >= 2, p - rival, np.nan),
                "et_z": np.where(cnt >= 3, (p - g.transform("mean")) / g.transform("std"), np.nan),
                "et_rank": g.rank(ascending=False, method="min").to_numpy(float),
                "et_lengths": np.where(lpl > 0, (top - p) / lpl, np.nan),
                "et_win": (e / e.groupby(rk).transform("sum")).to_numpy(float),
                "et_uplift": pred - X["h_fig"].to_numpy(float),
                "et_inputs": np.where(np.isfinite(pred), X[["h_fig", "h_trip_fig", "h_going_fig", "h_course_fig",
                                                            "h_code_fig", "h_l1"]].notna().sum(axis=1), np.nan)}
    new = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=df.index)[FEATURES]
    new["et_z"] = new["et_z"].replace([np.inf, -np.inf], np.nan)
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
