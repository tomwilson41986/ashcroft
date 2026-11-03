"""The roadmap's F3, F5 and F6 in the greyhound engine (greyhound/times.py, intent.py, ability.py): each lag-safe (a
day's own results never reach its features), and each doing what it says on small synthetic runs (the engine's
mechanics only, not data a model is trained or scored on)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound import ability, intent, times
from greyhound.data import clean
from greyhound.metrics import DogDays, GreyhoundMetricsEngine
from tests.test_greyhound import _runs

NEW = ("times", "intent", "ability", "etime")


def _vary(runs: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """The synthetic runs with grades, trips and trainers that change, and some trials, so every situation occurs."""
    rng = np.random.default_rng(seed)
    r = runs.copy()
    r["race_class"] = rng.choice(["A3", "A4", "A5", "A6"], len(r))
    r["race_class"] = r.groupby("race_id").race_class.transform("first")
    r["distance_m"] = np.where(r.race_id % 3 == 0, 500.0, 480.0)
    r["trainer"] = [f"T{(d + (day > '2026-01-06')) % 4}" for d, day in zip(r.dog_id, r.race_date)]
    trial = r.race_id % 7 == 0
    r.loc[trial, "race_class"] = "T1"
    return clean(r.drop(columns=["won", "placed2", "field", "grade", "grade_family", "sp_p", "sp_p_norm", "raceid",
                                 "t", "is_trial"]))


def test_the_new_blocks_are_blind_to_their_own_day():
    runs = _vary(_runs(n_days=16))
    eng = GreyhoundMetricsEngine(blocks=("parity", "hrb", "lib") + NEW)
    base = eng.calculate_all(runs)
    feats = [f for b in NEW for f in eng.block_features[b]]
    assert len(feats) > 40
    last_day = runs.race_date.max()
    alt = runs.copy()
    day = alt.race_date == last_day
    alt.loc[day, "position"] = 7 - alt.loc[day, "position"]
    alt.loc[day, "run_time"] += 1.0
    alt.loc[day, "adjusted_time"] += 1.0
    alt.loc[day, "sectional"] -= 0.3
    alt.loc[day, "sp_decimal"] = 2.0
    alt.loc[day, "comment"] = "VSAw,Crd1,Wide,Fdd"
    alt = clean(alt.drop(columns=["won", "placed2", "field", "grade", "grade_family", "sp_p", "sp_p_norm", "raceid",
                                  "t", "is_trial"]))
    other = GreyhoundMetricsEngine(blocks=("parity", "hrb", "lib") + NEW).calculate_all(alt)
    for rid in runs[day].race_id.unique():
        a = base[base.race_id == rid].sort_values("trap")[feats].reset_index(drop=True)
        b = other[other.race_id == rid].sort_values("trap")[feats].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6)


def test_the_rolling_standard_is_the_winners_median_before_the_day():
    runs = _runs(n_days=10, tracks=("Romford",))
    df = runs.assign(adjusted_time=28.0 + runs.race_id % 5 * 0.1)
    s = times.rolling_standard(df, 3)
    day = pd.to_datetime(df.race_date)
    for d in sorted(day.unique())[1:4]:
        win = df[(df.won == 1) & (day < d) & (day >= d - pd.Timedelta(days=3))].adjusted_time
        got = s[day == d].unique()
        assert len(got) == 1 and np.isclose(got[0], win.median())
    assert s[day == day.min()].isna().all()                   # nothing before the first day


def test_the_meeting_variant_shrinks_a_night_by_its_winners():
    runs = _runs(n_days=30, tracks=("Romford", "Hove"))
    std = pd.Series(28.0, index=runs.index)
    rng = np.random.default_rng(4)
    night = runs.meeting_id.map({m: rng.normal(0, 0.1) for m in runs.meeting_id.unique()})
    df = runs.assign(adjusted_time=28.0 + night + rng.normal(0, 0.05, len(runs)))
    v = times.meeting_variant(df, std)
    late = pd.to_datetime(df.race_date) >= pd.Timestamp("2026-01-20")
    rel = v.rel[late]
    assert rel.between(0, 1).all() and rel.notna().all()
    raw = (df.adjusted_time - std).where(df.won == 1).groupby(df.meeting_id).transform("mean")
    # shrunk toward 0: never further from 0 than the night's raw mean
    assert (v.variant[late].abs() <= raw[late].abs() + 1e-12).all()


def test_trainer_intent_reads_the_dogs_last_day():
    runs = _vary(_runs(n_days=12))
    df = GreyhoundMetricsEngine().calculate_all(runs)
    f = intent.flags(df, DogDays)
    first = df.race_date == df.groupby("dog_id").race_date.transform("min")
    assert f[first.to_numpy()].isna().all().all()             # a first run has no last day to compare
    later = df[~first].iloc[:40]
    for i, row in later.iterrows():
        prev = df[(df.dog_id == row.dog_id) & (df.race_date < row.race_date)].sort_values("t").iloc[-1]
        if prev.grade == prev.grade and row.grade == row.grade:
            assert f.loc[i, "class_down"] == float(row.grade > prev.grade)
        assert f.loc[i, "new_trainer"] == float(row.trainer != prev.trainer)


def test_the_chance_of_being_best_sums_to_one_and_favours_the_better_rating():
    m = np.array([5.0, 0.0, 0.0, 2.0, 2.0, np.nan])
    sd = np.array([1.0, 1.0, 1.0, 0.1, 3.0, 1.0])
    race = np.array([1, 1, 1, 2, 2, 2])
    p = ability.p_best(m, sd, race, sims=4000)
    assert np.isclose(np.nansum(p[:3]), 1.0) and np.isclose(np.nansum(p[3:]), 1.0) and np.isnan(p[5])
    assert p[0] > 0.9 and abs(p[3] - 0.5) < 0.05             # equal means: the certain one wins half the time


def test_the_track_ability_is_shrunk_toward_the_dogs_own():
    runs = _runs(n_days=14)
    eng = GreyhoundMetricsEngine(blocks=("lib", "ability"))
    df = eng.calculate_all(runs)
    no_track = df.ab_dt_n == 0
    assert np.allclose(df.ab_dt[no_track], df.ab_dog[no_track], equal_nan=True)
    assert (df.ab_dt_se > 0).all() or df.ab_dt_se.isna().any()
    assert {"ab_kr_var", "ab_p_best", "ab_dt", "ab_dt_n"} <= set(eng.block_features["ability"])


def test_the_expected_time_projects_a_dogs_runs_onto_todays_standard():
    from greyhound import etime
    runs = _vary(_runs(n_days=16))
    eng = GreyhoundMetricsEngine(blocks=("lib", "etime"))
    df = eng.calculate_all(runs)
    f = eng.block_features["etime"]
    assert {"et_eb_s", "et_sd_s", "et_p_fastest", "et_gap_best_len", "et_split_s"} <= set(f)
    late = df[df.race_date >= "2026-01-10"]
    assert late.et_eb_s.notna().all() and (late.et_sd_s > 0).all()
    # each estimate is today's standard scaled by its rel
    np.testing.assert_allclose(late.et_mean3_s, late.std_time * (1 + late.et_mean3_rel), rtol=1e-5)
    # the chance of being fastest: one in each race, the fastest expected time the likeliest
    tot = late.groupby("race_id").et_p_fastest.sum()
    assert np.allclose(tot, 1.0, atol=1e-6)
    top = late.loc[late.groupby("race_id").et_eb_s.idxmin()]
    assert (top.et_gap_best_len == 0).all() and (top.et_rank == 1).all()
    assert (top.et_p_fastest >= late.groupby("race_id").et_p_fastest.transform("median").loc[top.index]).all()
