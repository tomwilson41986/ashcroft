"""greyhound/pit/draw2.py: the draw with a hierarchy across trips and the dogs' own lead chance taken out. On a planted
card (one track, three trips: at 400 m trap 1 truly leads more; at 500 m there is no trap bias but the racing manager
seeds the fastest starter in trap 1; 600 m opens late and shares 400 m's trap bias): the true bias is read at 400 m,
the seeding is not read as a bias at 500 m (where draw.py's dr_lead reads it), the new trip borrows the other trips'
state on its first day; the day's own results move nothing and a card row gets what its run had. All data here is
planted for the tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, draw, draw2
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S
COLS = list(draw2.COLUMNS)
OPEN600 = 150


def synth(days=200, seed=5, bonus=3.0):
    rng = np.random.default_rng(seed)
    pace = rng.normal(0, 2.0, 61)                    # lengths of early pace a dog
    rows, rid = [], 5000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        trips = (400, 500) + ((600,) if d >= OPEN600 else ())
        dogs = rng.permutation(np.arange(1, 61))
        r = 0
        for trip in trips:
            for _ in range(3):
                field = dogs[(6 * r) % 54:(6 * r) % 54 + 6]
                r += 1
                rid += 1
                if trip == 500:                      # seeded: the fastest starter in trap 1, the rest at random
                    f = field[np.argsort(-pace[field])]
                    field = np.r_[f[0], rng.permutation(f[1:])]
                traps = np.arange(1, 7)
                b = np.where((traps == 1) & (trip != 500), bonus, 0.0)
                early = pace[field] + b + rng.normal(0, 1.0, 6)
                perf = early + rng.normal(0, 1.5, 6)
                pos = np.argsort(np.argsort(-perf)) + 1
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{14 + r}:00", "meeting_id": d, "track": "Hove",
                                 "race_id": rid, "race_class": "A3", "distance_m": float(trip), "trap": int(traps[k]),
                                 "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                                 "sp_decimal": float(rng.choice([4.0, 5.0, 6.0, 8.0])), "position": int(pos[k]),
                                 "beaten_lengths": 0.5, "sectional": float(4.5 - early[k] * L),
                                 "run_time": float(29.5 - perf[k] * L), "adjusted_time": float(29.5 - perf[k] * L),
                                 "weight_kg": 30.0, "comment": "QAw,Mid"})
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def built():
    runs = synth()
    figs = core.per_run_figures(runs)
    tab, notes = draw2.build(runs, figs)
    old, _ = draw.build(runs, figs)
    m = runs[["race_id", "dog_id", "race_date", "trap", "distance_m"]].merge(tab, on=["race_id", "dog_id"]) \
        .merge(old, on=["race_id", "dog_id"])
    m["d"] = (pd.to_datetime(m.race_date) - pd.Timestamp("2025-01-01")).dt.days
    return runs, m, tab, notes


def test_a_row_a_run_and_every_column_registered(built):
    runs, m, tab, notes = built
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS
    assert all(c.startswith(draw2.PREFIX) for c in COLS) and all(draw2.COLUMNS.values())
    assert len(tab) == len(runs) == len(m) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab[COLS].dtypes == np.float32).all()
    assert notes["win"]["q"] in draw.QS and notes["lead"]["tau2"] > 0
    # the dogs' own lead chances: a softmax in each race, rising with early pace
    s = m.groupby("race_id").dr2_p_lead_own.sum()
    assert np.allclose(s, 1.0, atol=1e-5) and notes["own_lead"]["beta"]["pace"] > 0


def test_a_true_bias_is_read_and_seeding_is_not(built):
    _, m, _, _ = built
    end = m[m.d >= 120]
    t1 = end[end.trap == 1]
    at400, at500 = t1[t1.distance_m == 400], t1[t1.distance_m == 500]
    assert at400.dr2_lead.mean() > 0.1 and at400.dr2_win.mean() > 0.05
    # seeded trap 1 at 500 m: draw.py reads it as a lead bias, draw2 takes the dogs' own chances out
    assert at500.dr_lead.mean() > 0.3
    assert abs(at500.dr2_lead.mean()) < 0.5 * at500.dr_lead.mean()
    assert abs(at500.dr2_lead.mean()) < at400.dr2_lead.mean()


def test_a_new_trip_borrows_the_other_trips_on_its_first_day(built):
    _, m, _, _ = built
    first = m[(m.d == OPEN600) & (m.distance_m == 600)]
    t1, rest = first[first.trap == 1], first[first.trap != 1]
    assert t1.dr2_lead_other.mean() > 0.05 and t1.dr2_lead.mean() > 0.05
    assert t1.dr2_lead.mean() > rest.dr2_lead.mean() + 0.05
    assert np.allclose(t1.dr2_lead, t1.dr2_lead.iloc[0])          # one state a trap a trip a day
    # with no reading of its own the trip's state is the other trips' state, its spread wider than theirs
    assert (first.dr2_lead_sd > 0).all()
    assert np.allclose(first.dr2_lead, first.dr2_lead_other, atol=0.02)


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=60)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = draw2.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = draw2.build(r2, core.per_run_figures(r2))[0]
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)
    assert np.isfinite(x).mean() > 0.9 and np.allclose(x, z, atol=1e-5, equal_nan=True)


def test_later_days_move_nothing_earlier(monkeypatch):
    monkeypatch.setattr(draw, "FIT_BEFORE", "2025-01-12")         # the fit's slice fixed by date, not by share
    runs = synth(days=60)
    date = sorted(runs.race_date.unique())[45]
    upto = runs.loc[runs.race_date <= date, ["race_id", "dog_id"]]
    a = draw2.build(runs, core.per_run_figures(runs))[0]
    cut = runs[runs.race_date <= date].reset_index(drop=True)
    b = draw2.build(cut, core.per_run_figures(cut))[0]
    x = upto.merge(a, on=["race_id", "dog_id"])[COLS].to_numpy(float)
    z = upto.merge(b, on=["race_id", "dog_id"])[COLS].to_numpy(float)
    assert np.allclose(x, z, atol=1e-5, equal_nan=True)
