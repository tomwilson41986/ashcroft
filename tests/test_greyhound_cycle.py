"""greyhound/pit/cycle.py: a bitch's season cycle read from her absences. Her own cycle is recovered; the phase and
the days to the next season are right the week before she goes; a dog's absences are kept apart as the control; the
day's own results move nothing and a card row gets what its run had."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, cycle
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day


def synth(days=900, seed=5):
    """Twelve dogs racing weekly. Bitches 1-6 have a season every 180 + 15 x id days and miss 70 days; dog 7 misses
    90 days once (an injury); the rest never stop."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 9000
    for d in range(0, days, 7):
        day = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        field = []
        for dog in range(1, 13):
            if dog <= 6 and (d % (180 + 15 * dog)) >= (180 + 15 * dog) - 70:
                continue
            if dog == 7 and 300 <= d < 390:
                continue
            field.append(dog)
        for r in range(0, len(field), 6):
            rid += 1
            f = field[r:r + 6]
            t = 29.5 + rng.normal(0, 0.05, len(f))
            pos = np.argsort(np.argsort(t)) + 1
            for k, dog in enumerate(f):
                rows.append({"race_date": day, "race_time": f"{18 + r // 6}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": k + 1, "dog_id": dog,
                             "dog_name": f"Dog {dog}", "born": "Jan-2022", "sex": "b" if dog <= 6 else "d",
                             "season": "01.Ja.25" if dog <= 6 else None, "sp_decimal": 6.0, "position": int(pos[k]),
                             "beaten_lengths": 0.5, "sectional": 4.5, "run_time": float(t[k]),
                             "adjusted_time": float(t[k]), "weight_kg": 30.0, "comment": "QAw"})
    return clean(pd.DataFrame(rows))


@pytest.fixture()
def built():
    runs = synth()
    tab, notes = cycle.build(runs, core.per_run_figures(runs))
    return runs, runs.merge(tab, on=["race_id", "dog_id"]), notes


def test_every_column_is_registered_and_the_sexes_are_kept_apart(built):
    runs, m, notes = built
    assert {c for c in m.columns if c.startswith(cycle.PREFIX)} == set(cycle.COLUMNS) and len(m) == len(runs)
    assert m[m.sex == "d"][[c for c in cycle.COLUMNS if "_b_" in c]].isna().all().all()
    assert m[m.sex == "b"][[c for c in cycle.COLUMNS if "_d_" in c]].isna().all().all()
    assert notes["absences_a_bitch_year"] > 1.0 and notes["absences_a_dog_year"] < 0.2


def test_her_own_cycle_is_recovered_and_she_is_due_before_she_goes(built):
    _, m, _ = built
    for dog in (1, 4, 6):
        true = 180 + 15 * dog
        b = m[m.dog_id == dog].sort_values("race_date")
        last = b.iloc[-1]
        assert last.oc_b_n >= 2 and abs(last.oc_b_cycle - true) < 25          # shrunk by one interval toward 240
        # the last run before each later absence: nearly a whole cycle since the last season began
        gone = b[(b.oc_b_n >= 2) & (b.oc_b_n.shift(-1) > b.oc_b_n)]
        assert len(gone) and (gone.oc_b_phase > 0.85).all() and (gone.oc_b_due.abs() < 45).all()
        back = b[b.oc_b_runs_back == 0]
        assert (back.oc_b_gap.between(70, 84)).all() and (back.oc_b_back_days == 0).all()


def test_a_maiden_bitch_has_her_age_and_a_dog_his_injury(built):
    _, m, _ = built
    first = m[(m.dog_id == 1) & (m.oc_b_n == 0)]
    assert first.oc_b_none_age.notna().all() and first.oc_b_days.isna().all()
    seven = m[m.dog_id == 7].sort_values("race_date")
    assert seven.oc_d_n.max() == 1 and seven[seven.oc_d_runs_back == 0].oc_d_gap.iloc[0] >= 90


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=400)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = cycle.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = cycle.build(r2, core.per_run_figures(r2))[0]
    cols = list(cycle.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.allclose(x, z, atol=1e-5, equal_nan=True)
