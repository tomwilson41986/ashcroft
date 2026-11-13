"""Improvement: a forecast of how far above or below its own level a horse will run today.

The exposure block gives the trees a horse's exposure and the average gain of horses like it. The owner's ask of
12 Nov 2026 is the projection itself ("research and work on how we can project and forecast improvement: horse
curve, age, unexposed, distance, going, siblings, headgear, gelding, first time at track / with jockey / with
trainer and second, third time, hidden form, track change, field size, race type"). As the expected-time model
does for the figure (iteration 119), a first-stage boosted model forecasts the CHANGE: the run's time figure less
the horse's own decayed mean figure before it, from what is known before the run:

    the curve         earlier runs (the table's career runs and the runs held), runs in this code, age, days off,
                      the horse's level and its last figure against it, the slope of its last figures
    the nth time      earlier runs at this track, with this jockey, with this trainer, in this trip band, on this
                      going group (0 = first time)
    changes           field size, prize, trip and going against the last run; a change of track or of code
    the horse itself  gelded since the last run; the gear moves (model/blocks/gear.py)
    hidden form       the last run against its price on finishing position, lengths and time, and what the market
                      expected of it (model/blocks/market_metrics.py)
    pedigree          how the sire's stock and the dam's other foals improve at this many runs
                      (model/blocks/pedigree_bayes.py)

refitted every fold on the four years before it only (model/blocks/expected_time.forecast), so no forecast has
seen its own day or a later one. A horse with no figure yet has no level to improve on: no forecast.

    im_pred  im_z  im_rank     the forecast change in pounds; within today's race: standardised, and its rank
    im_level                   the horse's level plus the forecast: the figure expected today on this view
    im_level_z                 that standardised within the race
    im_n_track  im_n_jockey  im_n_trainer  im_n_trip  im_n_going     the nth-time counts themselves, capped at 5
    im_first                   how many of those five are a first time today
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import expected_time as _et
from model.blocks import time_figure as _tf
from model.blocks.cond_form import GOING, classify_going
from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.race_shape import DIST_BANDS, asof_decayed_mean, bands, day_index, race_code, race_key

FEATURES = ["im_pred", "im_z", "im_rank", "im_level", "im_level_z", "im_n_track", "im_n_jockey", "im_n_trainer",
            "im_n_trip", "im_n_going", "im_first"]
POST_RACE: set[str] = set()
FROM = ["gr_sev_change", "gr_first", "gr_more", "gr_off", "gr_tt_on", "mm_n_l1", "mm_l_l1", "mm_t_l1", "mm_n_bayes",
        "mm_m_l1", "mm_m_m3", "pb_sire_impr", "pb_dam_impr", "pb_ped_m"]
READS = sorted(set(_tf.READS) | {"trainer", "jockey_name", "number_of_runners", "prize_money", "horse_age", "horse_sex",
                                 "career_runs"} | set(FROM))
AFTER = ("gear", "market_metrics", "pedigree_bayes")
CAP = 5.0


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    fig = np.asarray(_tf.figures(df)["fig"], dtype=float)
    horse = _et._name_codes(df["horse_name"])
    trainer, jockey, course = (_et._name_codes(_col(df, c)) for c in ("trainer", "jockey_name", "track"))
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(float)
    tb = pd.Series(bands(dist, DIST_BANDS)).fillna(-1).to_numpy().astype(np.int64)
    gg = pd.to_numeric(_col(df, "going_description").map(classify_going).map(GOING), errors="coerce").fillna(-1) \
        .to_numpy().astype(np.int64)
    code = pd.factorize(race_code(df), sort=True)[0].astype(np.int64)
    H = _Days(horse, day)
    blocks = np.arange(len(H.u))
    prev = np.clip(blocks - 1, 0, None)
    has_prev = H.valid & (H.pos >= 1)

    def nth(other):
        k = _et._key(horse, other)
        D = _Days(k, day)
        return np.where(k >= 0, D.to_rows(D.pos.astype(float)), np.nan)

    def last(v):
        b = H.per_block(v)
        return H.to_rows(np.where(has_prev, b[prev], np.nan))

    X: dict[str, np.ndarray] = {}
    own, own_n = asof_decayed_mean(horse, day, fig, horse, day, halflife_days=365.0)
    lad = _ladder(H, H.per_block(fig))
    X["level"], X["level_n"] = own, own_n
    X["last_vs_level"] = H.to_rows(lad["l1"]) - own
    X["m3_vs_level"] = H.to_rows(lad["m3"]) - own
    X["runs_held"] = H.to_rows(H.pos.astype(float))
    X["career_runs"] = pd.to_numeric(_col(df, "career_runs"), errors="coerce").to_numpy(float)
    X["runs_code"] = nth(code)
    counts = {"track": nth(course), "jockey": nth(jockey), "trainer": nth(trainer), "trip": nth(tb), "going": nth(gg)}
    for k, v in counts.items():
        X[f"n_{k}"] = v
    X["age"] = pd.to_numeric(_col(df, "horse_age"), errors="coerce").to_numpy(float)
    X["days_off"] = H.to_rows(np.where(has_prev, H.day - H.day[prev], np.nan))
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    prize = pd.to_numeric(_col(df, "prize_money"), errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.where(prize > 0, np.log(np.where(prize > 0, prize, 1.0)), np.nan)
        ld = np.where(dist > 0, np.log(np.where(dist > 0, dist, 1.0)), np.nan)
    X["d_field"], X["d_prize"], X["d_trip"] = n - last(n), lp - last(lp), ld - last(ld)
    g_now = np.where(gg >= 0, gg, np.nan).astype(float)
    X["d_going"] = g_now - last(g_now)
    X["track_change"] = (course.astype(float) != last(course.astype(float))).astype(float)
    X["code_change"] = (code.astype(float) != last(code.astype(float))).astype(float)
    sex = _col(df, "horse_sex").fillna("").astype(str).str.strip().str.lower().str[:1]
    geld = (sex == "g").to_numpy().astype(float)
    X["gelded_since"] = np.where(np.isfinite(last(geld)), ((geld == 1) & (last(geld) == 0)).astype(float), np.nan)
    X["runners"], X["log_trip"], X["going"], X["code"] = n, ld, g_now, code.astype(float)
    for c in FROM:
        X[c] = pd.to_numeric(_col(df, c), errors="coerce").to_numpy(float)

    Xf = pd.DataFrame(X, index=df.index).round(6)       # as debut.py: no summation noise for a tree to split on
    y = fig - own
    rk = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    order = np.lexsort((horse, rk.to_numpy(), day))
    saved = (_et.FOLD_DAYS, _et.MIN_TRAIN, _et.N_TREES, _et.MIN_CHILD)
    try:
        _et.FOLD_DAYS, _et.MIN_TRAIN, _et.N_TREES, _et.MIN_CHILD = FOLD_DAYS, MIN_TRAIN, N_TREES, MIN_CHILD
        pred = _et.forecast(Xf, y, day, order)
    finally:
        _et.FOLD_DAYS, _et.MIN_TRAIN, _et.N_TREES, _et.MIN_CHILD = saved
    pred = np.where(np.isfinite(own), pred, np.nan)

    def within(v):
        s = pd.Series(v, index=df.index)
        g = s.groupby(rk)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = ((s - g.transform("mean")) / g.transform("std")).where(g.transform("count") >= 3)
        return z.replace([np.inf, -np.inf], np.nan).to_numpy(float), g.rank(ascending=False, method="min").to_numpy(float)

    cols = {"im_pred": pred, "im_level": own + pred}
    cols["im_z"], cols["im_rank"] = within(pred)
    cols["im_level_z"], _ = within(own + pred)
    firsts = np.zeros(len(df))
    for k, v in counts.items():
        cols[f"im_n_{k}"] = np.minimum(v, CAP)
        firsts += (v == 0).astype(float)
    cols["im_first"] = firsts
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)


FOLD_DAYS = 365
MIN_TRAIN = 20000
N_TREES = 250
MIN_CHILD = 200
TEST_REACH = {"FOLD_DAYS": 10, "MIN_TRAIN": 60, "N_TREES": 20, "MIN_CHILD": 5}
