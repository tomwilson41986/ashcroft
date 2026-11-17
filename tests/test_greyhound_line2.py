"""greyhound/pit/line2.py: the neighbours' squeeze weighted by arrival, traps 1 and 2 away, and the pace collision.
On a planted card (each dog a railer, a middle or a wide runner and an early pace; a dog loses lengths when a dog
within 2 traps whose line crosses its own reaches the bend with it): every column is right by hand from the line,
expected-time and externality tables; with the neighbours' arrival the same the squeeze 1 trap away is line.py's;
the planted collision cost is read by the collision score better than by line.py's unweighted squeeze; the tables
are read from the context or built; the day's own results move nothing and a card row gets what its run had. All
data here is planted for the tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, ext, line, line2, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S
LINES = ("Rls", "Mid", "W")
WANT = np.array([1.0, 3.5, 6.0])
COLS = list(line2.COLUMNS)


def synth(days=80, seed=7, cost=2.0):
    """One track, 36 dogs, 4 races a day (the last a race of five). Dog % 3 is its line; each dog an early pace
    Each pair 1 or 2 traps apart whose lines cross costs both ``cost`` lengths x their
    closeness of early pace x the boxes of crossing."""
    rng = np.random.default_rng(seed)
    pace = rng.normal(0, 1.0, 37)
    rows, rid = [], 3000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        dogs = rng.permutation(np.arange(1, 37))
        for r in range(4):
            field = dogs[6 * r:6 * r + (5 if r == 3 else 6)]
            rid += 1
            traps = np.arange(1, len(field) + 1)
            lean = WANT[field % 3] - traps
            hit = np.zeros(len(field))
            for a in range(len(field)):
                for b in range(a + 1, min(a + 3, len(field))):
                    gap = b - a
                    conv = max(lean[a] - lean[b] - (gap - 1), 0.0)
                    c = np.exp(-abs(pace[field[a]] - pace[field[b]]))
                    hit[a] += cost * c * conv
                    hit[b] += cost * c * conv
            early = pace[field] + rng.normal(0, 0.3, len(field))
            perf = early + rng.normal(0, 1.0, len(field)) - hit
            pos = np.argsort(np.argsort(-perf)) + 1
            for k, dog in enumerate(field):
                rows.append({"race_date": day, "race_time": f"{14 + r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 480.0, "trap": int(traps[k]),
                             "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                             "sp_decimal": 6.0, "position": int(pos[k]), "beaten_lengths": 0.5,
                             "sectional": float(4.8 - early[k] * L), "run_time": float(29.5 - perf[k] * L),
                             "adjusted_time": float(29.5 - perf[k] * L), "weight_kg": 30.0,
                             "comment": f"QAw,{LINES[dog % 3]}"})
    return clean(pd.DataFrame(rows))


def tables(runs):
    figs = core.per_run_figures(runs)
    ln, _ = line.build(runs, figs)
    xt, _ = xtime.build(runs, figs)
    nx, _ = ext.build(runs, figs, {"xtime": xt})
    return figs, {"line": ln, "xtime": xt, "ext": nx}


@pytest.fixture(scope="module")
def built():
    runs = synth()
    figs, ctx = tables(runs)
    tab, notes = line2.build(runs, figs, ctx)
    k = ["race_id", "dog_id"]
    m = runs[k + ["race_date", "trap", "position"]].merge(tab, on=k).merge(ctx["line"], on=k) \
        .merge(ctx["xtime"][k + ["xt_sec"]], on=k).merge(ctx["ext"][k + ["nx_own_out", "nx_own_in"]], on=k)
    m["d"] = (pd.to_datetime(m.race_date) - pd.Timestamp("2025-01-01")).dt.days
    return runs, figs, ctx, m, tab, notes


def test_a_row_a_run_and_every_column_registered(built):
    runs, _, _, m, tab, notes = built
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS
    assert all(c.startswith(line2.PREFIX) for c in COLS) and all(line2.COLUMNS.values())
    assert len(tab) == len(runs) == len(m) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab[COLS].dtypes == np.float32).all()
    assert notes["rows_with_an_expected_sectional"] > 0.5


def test_every_column_by_hand(built):
    _, _, _, m, _, _ = built
    late = m[m.d >= 20].dropna(subset=["xt_sec"])
    by = m.set_index(["race_id", "trap"])
    for _, r in late.sample(150, random_state=2).iterrows():
        own = -r.ln_trap_mis if np.isfinite(r.ln_trap_mis) else 0.0
        e = dict.fromkeys(("sq1", "sq2", "conflict", "close", "collide", "collide_harm"), 0.0)
        for d in (1, 2):
            for side in (-1, 1):
                if (r.race_id, r.trap + side * d) not in by.index:
                    continue
                k = by.loc[(r.race_id, r.trap + side * d)]
                c = np.exp(-abs(r.xt_sec - k.xt_sec)) if np.isfinite(k.xt_sec) else 0.0
                lk = -k.ln_trap_mis
                lk0 = lk if np.isfinite(lk) else 0.0
                sq = max(lk0 - (d - 1), 0) if side < 0 else max(-lk0 - (d - 1), 0)
                e[f"sq{d}"] += c * sq
                e["conflict"] += c * sq * (max(-own, 0) if side < 0 else max(own, 0))
                e["close"] += c
                conv = (max((lk - own if side < 0 else own - lk) - (d - 1), 0) if np.isfinite(lk) else 0.0)
                e["collide"] += c * conv
                h = k.nx_own_out if side < 0 else k.nx_own_in
                e["collide_harm"] += c * conv * (h if np.isfinite(h) else 0.0)
        for name, v in e.items():
            assert np.isclose(r[f"ln2_{name}"], v, atol=1e-4), (name, r[f"ln2_{name}"], v)
        assert np.isclose(r.ln2_sq, r.ln2_sq1 + r.ln2_sq2, atol=1e-5)


def test_with_the_same_arrival_the_squeeze_beside_is_line_pys(built):
    runs, figs, ctx, _, _, _ = built
    flat = dict(ctx, xtime=ctx["xtime"].assign(xt_sec=1.0))       # every dog expected at the bend together
    tab, _ = line2.build(runs, figs, flat)
    m = tab.merge(ctx["line"], on=["race_id", "dog_id"]).dropna(subset=["ln2_sq1"])
    assert len(m) == len(runs)
    assert np.allclose(m.ln2_sq1, m.ln_nb_squeeze, atol=1e-5)
    one = m[m.ln2_sq2 == 0]                                         # nothing from 2 traps away: line.py's conflict
    assert len(one) > 100 and np.allclose(one.ln2_conflict, one.ln_nb_conflict, atol=1e-5)


def test_the_planted_collision_is_read_better_than_the_unweighted_squeeze(built):
    _, _, _, m, _, _ = built
    late = m[m.d >= 20].dropna(subset=["ln2_collide"])
    pos = late.position
    a = np.corrcoef(late.ln2_collide, pos)[0, 1]
    b = np.corrcoef(late.ln_nb_squeeze, pos)[0, 1]
    assert a > 0.15 and a > b + 0.02


def test_the_tables_are_taken_from_the_context_or_built(built):
    runs, figs, ctx, _, tab, _ = built
    alone, _ = line2.build(runs, figs)
    assert np.allclose(alone[COLS].to_numpy(float), tab[COLS].to_numpy(float), equal_nan=True, atol=1e-5)


def test_a_trap_outside_one_to_six_has_no_columns():
    runs = synth(days=15)
    odd = runs.copy()
    hit = (odd.race_date == odd.race_date.max()) & (odd.trap == 6)
    odd.loc[hit, "trap"] = 7
    figs, ctx = tables(odd)
    tab, notes = line2.build(odd, figs, ctx)
    assert notes["rows_without_a_trap_1_to_6"] == int(hit.sum()) > 0
    assert tab[hit.to_numpy()][COLS].isna().all().all()


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=40)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    figs, ctx = tables(runs)
    a = line2.build(runs, figs, ctx)[0]
    r2 = change(runs)
    figs2, ctx2 = tables(r2)
    b = line2.build(r2, figs2, ctx2)[0]
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)
    assert np.isfinite(x).mean() > 0.8 and np.allclose(x, z, atol=1e-4, equal_nan=True)
