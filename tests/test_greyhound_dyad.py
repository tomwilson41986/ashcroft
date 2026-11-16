"""greyhound/pit/dyad.py: the memory of each pair of dogs beyond their ratings. A planted pair effect is found and
signed for both dogs; a rock-paper-scissors triple shows as curl and a consistent one does not; the day's own results
move nothing; a card row gets what its run had; the 'earlier days' sums leave the day itself out."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, dyad, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S


def synth(days=90, seed=3, pair=3.0, cycle=2.5):
    """One track, 12 dogs of equal ability, two races of six a day. Dog 2 runs ``pair`` lengths slower whenever dog 1
    is in its race. Dogs 3, 4, 5 are rock-paper-scissors: 4 loses ``cycle`` lengths to 3, 5 to 4, 3 to 5, when the
    two meet without the third."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 7000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        order = rng.permutation(np.arange(1, 13))
        for r, field in enumerate((order[:6], order[6:])):
            rid += 1
            fs = set(int(x) for x in field)
            t = 29.5 + rng.normal(0, 0.05, 6)
            for k, dog in enumerate(field):
                dog = int(dog)
                if dog == 2 and 1 in fs:
                    t[k] += pair * L
                if not {3, 4, 5} <= fs:
                    for slow, by in ((4, 3), (5, 4), (3, 5)):
                        if dog == slow and by in fs:
                            t[k] += cycle * L
            pos = np.argsort(np.argsort(t)) + 1
            for k, dog in enumerate(field):
                rows.append({"race_date": day, "race_time": f"{18 + r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": k + 1,
                             "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                             "sp_decimal": 6.0, "position": int(pos[k]), "beaten_lengths": 0.5, "sectional": 4.5,
                             "run_time": float(t[k]), "adjusted_time": float(t[k]), "weight_kg": 30.0,
                             "comment": "QAw"})
    return clean(pd.DataFrame(rows))


@pytest.fixture()
def built():
    runs = synth()
    tab, notes = dyad.build(runs, core.per_run_figures(runs))
    return runs, runs.merge(tab, on=["race_id", "dog_id"]), notes


def test_every_column_is_registered(built):
    runs, m, notes = built
    assert {c for c in m.columns if c.startswith(dyad.PREFIX)} == set(dyad.COLUMNS) and len(m) == len(runs)
    assert notes["share_of_pair_rows_with_an_earlier_meeting"] > 0.9


def test_the_planted_pair_is_found_for_both_dogs(built):
    runs, m, _ = built
    late = m[m.race_date >= "2025-03-01"]
    both = late.groupby("race_id").dog_id.transform(lambda s: {1, 2} <= set(s))
    one = late[both & (late.dog_id == 1)]
    two = late[both & (late.dog_id == 2)]
    apart = late[~both & late.dog_id.isin([1, 2])]
    # 3 lengths between them when they meet (about 45% of dog 2's runs, so its expected time carries 1.35 of it and
    # the pair is left about 1.65), shrunk a little, on top of the other pairs' noise
    assert one.dy_edge.mean() > 0.8 and two.dy_edge.mean() < -0.8 and abs(apart.dy_edge.mean()) < 0.6
    assert one.dy_best.mean() > 1.0 and two.dy_worst.mean() < -1.0
    assert one.dy_ahead.mean() > 0.2 and two.dy_ahead.mean() < -0.2


def test_rock_paper_scissors_is_curl_and_consistency_is_not(built):
    runs, m, _ = built
    late = m[m.race_date >= "2025-03-01"]
    fields = late.groupby("race_id").dog_id.agg(set)
    cyc = late[late.race_id.map(fields).map(lambda s: {3, 4, 5} <= s) & late.dog_id.isin([3, 4, 5])]
    rest = late[late.dog_id.isin([6, 7, 8, 9, 10, 11, 12])
                & ~late.race_id.map(fields).map(lambda s: bool({3, 4, 5} & s) or {1, 2} <= s)]
    assert cyc.dy_curl.mean() > 2 * rest.dy_curl.mean() and cyc.dy_curl.mean() > 1.0


def test_earlier_days_only():
    key = np.array([1, 1, 1, 2, 1])
    dn = np.array([5, 5, 6, 5, 7])
    s, = dyad.earlier(key, dn, [np.array([10.0, 20.0, 40.0, 1.0, 80.0])])
    assert list(s) == [0.0, 0.0, 30.0, 0.0, 70.0]              # the day's own rows, both of them, are left out


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=40)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = dyad.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = dyad.build(r2, core.per_run_figures(r2))[0]
    cols = list(dyad.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).mean() > 0.8 and np.allclose(x, z, atol=1e-4, equal_nan=True)


def test_the_expected_time_table_is_taken_from_the_context():
    runs = synth(days=30)
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = dyad.build(runs, figs, {"xtime": xt})[0]
    b = dyad.build(runs, figs)[0]
    assert np.allclose(a[list(dyad.COLUMNS)].to_numpy(float), b[list(dyad.COLUMNS)].to_numpy(float), equal_nan=True)
