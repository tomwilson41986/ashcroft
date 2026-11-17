"""greyhound/pit/lastrace2.py: the last race's speed less its meeting's variant (review "lastrace"). On a constructed
card where every fifth night runs 0.30 s fast for every dog, the variant reads the fast night (about 3.75 lengths)
and the race speed less it is about 0; it is lr_race_speed less the variant; a day's own results move nothing and a
card row gets what its run had. Constructed data is for these unit tests only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.data import clean
from greyhound.pit import lastrace, lastrace2
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day
from tests.test_greyhound_pit import synth as pit_synth

KEY = ["race_id", "dog_id"]
COLS = list(lastrace2.COLUMNS)


def nights(days=40, seed=2, fast=0.30):
    rng = np.random.default_rng(seed)
    rows, rid = [], 500
    for d in range(days):
        day = (pd.Timestamp("2025-02-03") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        dogs = rng.permutation(np.arange(1, 49))
        f = fast if d % 5 == 0 else 0.0
        for r in range(8):
            rid += 1
            t = 29.5 - f + rng.normal(0, 0.05, 6)
            pos = np.argsort(np.argsort(t)) + 1
            for k, dog in enumerate(dogs[6 * r:6 * r + 6]):
                rows.append({"race_date": day, "race_time": f"{18 + r // 2}:{30 * (r % 2):02d}", "meeting_id": d,
                             "track": "Hove", "race_id": rid, "race_class": "A3", "distance_m": 500.0,
                             "trap": k + 1, "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023",
                             "sex": "d", "sp_decimal": 6.0, "position": int(pos[k]), "beaten_lengths": 1.0,
                             "sectional": 4.5, "run_time": float(t[k]), "adjusted_time": float(t[k]),
                             "weight_kg": 30.0, "comment": "QAw"})
    return clean(pd.DataFrame(rows))


def test_columns_and_keys():
    runs = pit_synth()
    tab, notes = lastrace2.build(runs)
    assert list(tab.columns) == KEY + COLS and (tab[COLS].dtypes == np.float32).all()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and notes["rows_with_a_variant"] > 0.5


def test_a_fast_night_is_taken_out():
    runs = nights()
    tab, _ = lastrace2.build(runs)
    lr, _ = lastrace.build(runs)
    m = runs[KEY + ["race_date"]].merge(tab, on=KEY).merge(lr[KEY + ["lr_race_speed"]], on=KEY)
    m = m[m.lr2_variant.notna() & m.lr_race_speed.notna()]
    prev = (pd.to_datetime(m.race_date) - pd.Timestamp("2025-02-03")).dt.days - 1
    after_fast = m[prev % 5 == 0]                                 # every dog ran the night before (one run a night)
    other = m[prev % 5 != 0]
    assert len(after_fast) > 100 and len(other) > 100
    assert after_fast.lr_race_speed.mean() > 2.0                  # the night looks fast in lr_race_speed
    assert after_fast.lr2_variant.mean() > 2.0
    assert abs(after_fast.lr2_race_speed_v.mean()) < 1.0 and abs(other.lr2_race_speed_v.mean()) < 1.0
    np.testing.assert_allclose(m.lr2_race_speed_v, m.lr_race_speed - m.lr2_variant, atol=1e-3)


def test_the_days_own_results_move_nothing_and_a_card_row_gets_its_runs_features():
    runs = pit_synth()
    last = runs.race_date.max()
    keys = runs.loc[runs.race_date == last, KEY]
    a, _ = lastrace2.build(runs)
    b, _ = lastrace2.build(perturb_last_day(runs))
    c, _ = lastrace2.build(cardify_last_day(runs))
    x = keys.merge(a, on=KEY)
    assert x[COLS].notna().all(axis=1).mean() > 0.5
    pd.testing.assert_frame_equal(x, keys.merge(b, on=KEY))
    pd.testing.assert_frame_equal(x, keys.merge(c, on=KEY))
