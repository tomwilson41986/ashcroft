"""model/blocks/exposure.py by hand: the figures ladder and the gain horses like it go on to make.

Hand-built races with known performance figures (a winner with no margin
returned is credited two pounds over its rating). Mechanics only.
"""
import numpy as np
import pandas as pd
import pytest

from model.blocks import exposure


def _race(day, horse, rating, runs, rival_rating=50, trainer="T1", age=4, track="York"):
    """A two-runner flat race the horse wins by an unknown margin: its figure is rating + 2."""
    d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)
    rid = f"{d.date()}|{track}|{horse}"
    common = dict(race_date=d, race_time="2.30", track=track, raceid=rid, number_of_runners=2,
                  dist_furlongs=6.0, race_type="Handicap", surface_type="Turf", stallion="S1", horse_age=age)
    return [
        dict(common, horse_name=horse, trainer=trainer, official_rating=rating, career_runs=runs,
             placing_numerical=1, LB=0.0, total_dst_bt="0"),
        dict(common, horse_name=f"rival{day}{horse}", trainer="T9", official_rating=rival_rating, career_runs=30,
             placing_numerical=2, LB=np.nan, total_dst_bt=""),
    ]


def _frame(rows):
    return pd.DataFrame([r for race in rows for r in race])


def test_the_figures_ladder():
    df = _frame([_race(0, "a", 58, 0), _race(10, "a", 63, 1), _race(20, "a", 68, 2), _race(30, "a", 75, 3)])
    out = exposure.build(df)
    today = out[(out.horse_name == "a") & (out.race_date == pd.Timestamp("2025-01-31"))].iloc[0]
    assert today.ex_runs_code == 3
    assert today.ex_perf_best == pytest.approx(70.0)            # 68 + 2
    assert today.ex_perf_best3 == pytest.approx(65.0)           # (60 + 65 + 70) / 3
    assert today.ex_perf_slope5 == pytest.approx(5.0)           # five pounds a run
    assert today.ex_perf_d1 == pytest.approx(5.0) and today.ex_perf_d_car == pytest.approx(5.0)
    first = out[(out.horse_name == "a") & (out.race_date == pd.Timestamp("2025-01-01"))].iloc[0]
    assert first.ex_runs_code == 0 and np.isnan(first.ex_perf_best)


def test_the_gain_horses_at_this_exposure_went_on_to_make():
    # horse a improves ten pounds at its second run (career runs 1 before it); b, later, is at the
    # same exposure, age and code: its expected gain is a's
    df = _frame([_race(0, "a", 48, 0), _race(5, "a", 58, 1), _race(9, "b", 60, 0), _race(20, "b", 62, 1)])
    out = exposure.build(df)
    b = out[(out.horse_name == "b") & (out.race_date == pd.Timestamp("2025-01-21"))].iloc[0]
    assert b.ex_gain_cell == pytest.approx(10.0)
    # the same trainer: a's gain is the trainer's only one, shrunk toward the cell's (also 10)
    assert b.ex_gain_trainer == pytest.approx(10.0)
    assert b.ex_headroom == pytest.approx(62.0 + 10.0 - 62.0)   # best (60 + 2) + gain - today's rating
    # a's own second run knows nothing of its own gain
    a2 = out[(out.horse_name == "a") & (out.race_date == pd.Timestamp("2025-01-06"))].iloc[0]
    assert np.isnan(a2.ex_gain_cell)
