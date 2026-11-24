"""greyhound/pit/bend2.py: the first bend with track-specific trap terms, shrunk inputs, the exact expected position
and a crowding chance. On a planted card (two tracks: at Hove trap 1 leads more, at Romford trap 6 does; each dog an
early pace): the track-and-trap offsets come back with the right signs, the exact expected position is 1 + the sum of
the pairwise sigmoids and matches bend.py's simulation, the positions of a race sum to 1 + 2 + ... + field, the
crowding chance is right by hand and highest between dogs of equal utility; the day's own results move nothing and a
card row gets what its run had. All data here is planted for the tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import bend, bend2, core
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S
COLS = list(bend2.COLUMNS)
START = pd.Timestamp("2025-07-01")


def synth(days=300, seed=11, bonus=3.0):
    rng = np.random.default_rng(seed)
    pace = rng.normal(0, 2.0, 61)
    rows, rid = [], 9000
    for d in range(days):
        day = (START + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        dogs = rng.permutation(np.arange(1, 61))
        for ti, track in enumerate(("Hove", "Romford")):
            for r in range(4):
                field = dogs[6 * (4 * ti + r) % 54:6 * (4 * ti + r) % 54 + 6]
                rid += 1
                traps = np.arange(1, 7)
                b = np.where(traps == (1 if track == "Hove" else 6), bonus, 0.0)
                early = pace[field] + b + rng.normal(0, 1.0, 6)
                perf = early + rng.normal(0, 1.5, 6)
                pos = np.argsort(np.argsort(-perf)) + 1
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{14 + r}:{10 * ti:02d}", "meeting_id": 2 * d + ti,
                                 "track": track, "race_id": rid, "race_class": "A3", "distance_m": 480.0,
                                 "trap": int(traps[k]), "dog_id": int(dog), "dog_name": f"Dog {dog}",
                                 "born": "Jan-2023", "sex": "d", "sp_decimal": 6.0, "position": int(pos[k]),
                                 "beaten_lengths": 0.5, "sectional": float(4.5 - early[k] * L),
                                 "run_time": float(29.5 - perf[k] * L), "adjusted_time": float(29.5 - perf[k] * L),
                                 "weight_kg": 30.0, "comment": "QAw,Mid"})
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def built():
    mp = pytest.MonkeyPatch()
    mp.setattr(bend2, "MIN_FIT", 600)
    runs = synth()
    tab, notes = bend2.build(runs, core.per_run_figures(runs))
    mp.undo()
    m = runs[["race_id", "dog_id", "race_date", "trap", "track", "sectional", "field"]].merge(tab, on=["race_id", "dog_id"])
    return runs, m, tab, notes


def test_a_row_a_run_and_every_column_registered(built):
    runs, m, tab, notes = built
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS
    assert all(c.startswith(bend2.PREFIX) for c in COLS) and all(bend2.COLUMNS.values())
    assert len(tab) == len(runs) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab[COLS].dtypes == np.float32).all()
    assert list(notes["years"]) == [2026]                             # 2025 has no earlier year to fit on
    assert m[m.race_date < "2026-01-01"][COLS].isna().all().all()


def test_the_track_specific_trap_terms(built):
    _, m, _, notes = built
    off = notes["years"][2026]["trap_offsets"]
    assert off["Hove|1"] > 0.2 and off["Romford|6"] > 0.2
    assert off["Hove|6"] < off["Hove|1"] - 0.3 and off["Romford|1"] < off["Romford|6"] - 0.3
    late = m[m.race_date >= "2026-01-01"]
    assert np.allclose(late.groupby("race_id").bd2_p_lead.sum(), 1.0, atol=1e-5)
    first = late.loc[late.groupby("race_id").sectional.idxmin()]
    assert first.bd2_p_lead.mean() > 1 / 6 + 0.1
    assert notes["years"][2026]["lead_beta"]["esr_vs_field"] > 0


def test_the_exact_expected_position(built):
    _, m, _, _ = built
    late = m[m.race_date >= "2026-01-01"]
    s = late.groupby("race_id").bd2_exp_pos.sum()
    f = late.groupby("race_id").field.first()
    assert np.allclose(s, f * (f + 1) / 2, atol=1e-3)
    u, gi = late.bd2_u.to_numpy(float), pd.factorize(late.race_id)[0]
    ex = bend2.expected_position(u, gi)
    for i in np.random.default_rng(0).choice(len(late), 30, replace=False):
        o = u[gi == gi[i]]
        assert np.isclose(ex[i], 1 + (1 / (1 + np.exp(-(o - u[i])))).sum() - 0.5)
    # the closed form is what bend.py's Plackett-Luce simulation tends to
    mc = bend.expected_position(u[:600], gi[:600], sims=4000, seed=3)
    assert np.abs(mc - ex[:600]).max() < 0.1
    # the order of the expected positions is the order of the utilities in a race
    r = late[late.race_id == late.race_id.iloc[0]]
    assert (r.bd2_exp_pos.rank().to_numpy() == (-r.bd2_u).rank().to_numpy()).all()


def test_crowding_by_hand():
    u = np.array([0.0, 0.0, 3.0, -3.0, 0.1, 5.0])
    gi = np.zeros(6, int)
    trap = np.arange(1, 7, dtype=float)
    c2, c1 = bend2.crowding(u, gi, trap)
    p = lambda a, b: bend2.together(np.array([b - a]))[0]  # noqa: E731
    assert np.isclose(c1[0], p(0.0, 0.0))                              # trap 1: only trap 2 beside it
    assert np.isclose(c2[0], 1 - (1 - p(0, 0)) * (1 - p(0, 3.0)))
    assert np.isclose(c1[4], 1 - (1 - p(0.1, -3.0)) * (1 - p(0.1, 5.0)))
    assert np.isclose(bend2.together(np.array([0.0]))[0], 1 / (1 + np.exp(-bend2.DELTA)) - 1 / (1 + np.exp(bend2.DELTA)))
    assert bend2.together(np.array([0.0]))[0] > bend2.together(np.array([2.0]))[0]
    # a vacant trap beside is room: no crowding from it; a trap outside 1 to 6 has no value
    c2v, c1v = bend2.crowding(u[[0, 2]], np.zeros(2, int), np.array([1.0, 3.0]))
    assert np.isclose(c1v[0], 0.0) and np.isclose(c2v[0], p(0.0, 3.0))
    assert np.isnan(bend2.crowding(u[:2], np.zeros(2, int), np.array([1.0, 7.0]))[0][1])


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change, monkeypatch):
    monkeypatch.setattr(bend2, "MIN_FIT", 600)
    runs = synth(days=200)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = bend2.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = bend2.build(r2, core.per_run_figures(r2))[0]
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)
    assert np.isfinite(x).all() and np.allclose(x, z, atol=1e-5)
