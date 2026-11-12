"""greyhound/metrics.py _per_run: the time standards and the meeting variant are made of graded races only (the fault
found on 11 Nov 2026: a trial's first dog counted as a winner, a fall counted in the sectional standard)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.data import clean
from greyhound.metrics import LENGTH_S, GreyhoundMetricsEngine


def _runs(trial_time=None, fall_sectional=None):
    """Three days at one track and trip: graded races won in 29.00 s, sectional 4.50; optionally a solo trial on day 2
    (its only dog 'wins' in ``trial_time``) and a faller on day 2 (sectional ``fall_sectional``)."""
    rows, rid = [], 100
    for d in range(3):
        day = f"2025-01-0{d + 1}"
        for r in range(4):
            rid += 1
            for k in range(6):
                rows.append({"race_date": day, "race_time": f"1{r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": k + 1,
                             "dog_id": 10 * r + k + 1, "position": k + 1, "beaten_lengths": 1.0, "sp_decimal": 6.0,
                             "sectional": 4.50, "run_time": 29.0 + 0.08 * k, "adjusted_time": 29.0 + 0.08 * k,
                             "weight_kg": 30.0, "comment": "Crd1" if k == 5 else "QAw"})
        if d == 1 and trial_time is not None:
            rid += 1
            rows.append({"race_date": day, "race_time": "09:00", "meeting_id": d, "track": "Hove", "race_id": rid,
                         "race_class": "T1", "distance_m": 500.0, "trap": 1, "dog_id": 900, "position": 1,
                         "beaten_lengths": 0.0, "sp_decimal": np.nan, "sectional": 4.20, "run_time": trial_time,
                         "adjusted_time": trial_time, "weight_kg": 30.0, "comment": "QAw"})
        if d == 1 and fall_sectional is not None:
            rows[-7 if trial_time is not None else -1].update({"sectional": fall_sectional, "comment": "Fell1"})
    return clean(pd.DataFrame(rows))


def _last_day(df):
    out = GreyhoundMetricsEngine(blocks=())._per_run(df)
    return out[out.race_date == out.race_date.max()]


def test_a_trial_moves_neither_the_standard_nor_the_variant():
    base, with_trial = _last_day(_runs()), _last_day(_runs(trial_time=28.0))
    graded = with_trial[~with_trial.is_trial]
    assert np.allclose(base.std_time, 29.0) and np.allclose(graded.std_time, 29.0)        # not pulled toward 28.0
    assert np.allclose(graded.std_sec, 4.50) and np.allclose(graded.variant, 0.0)
    assert np.allclose(graded.gsr, base.gsr, equal_nan=True)


def test_a_trial_keeps_its_own_figure_against_the_graded_standard():
    df = _runs(trial_time=28.0)
    out = GreyhoundMetricsEngine(blocks=())._per_run(df)
    t = out[out.is_trial].iloc[0]
    assert np.isclose(t.std_time, 29.0) and np.isclose(t.gsr, (29.0 - 28.0) / LENGTH_S)


def test_a_fall_is_not_in_the_sectional_standard():
    out = _last_day(_runs(fall_sectional=9.0))
    assert np.allclose(out.std_sec, 4.50)
    assert not [c for c in out.columns if c.startswith("_std_")]
