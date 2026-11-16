"""greyhound/pit/line.py: the running line, its change, and the neighbours' lines. On a planted card (every dog a
railer, a middle or a wide runner, and beaten further the further its trap is from its line): the habit is read back,
the trap's distance from the line predicts the finish, a dog that moves from the rails to the wide outside shows the
drift, a railer drawn outside and a wide runner drawn inside squeeze the dog between them, a race of five has a
vacant trap; a day's results move nothing on that day or before it, nor do later days; a card row gets what its run
had; one row a row of runs and no key twice. All data here is planted for the tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, line

LINES = ("Rls", "Mid", "W")                   # dog % 3: the box each wants is 1, 3.5, 6
WANT = np.array([1.0, 3.5, 6.0])
MOVER = 1                                     # runs the rails to day SWITCH, wide after
SWITCH = 40
COLS = list(line.COLUMNS)


def line_of(dog: int, d: int) -> int:
    if dog == MOVER:
        return 0 if d < SWITCH else 2
    return dog % 3


def synth(days=60, seed=3, cost=0.6):
    """One track, 48 dogs, 8 races a day (the last a race of five: trap 6 empty). A dog's run = its ability + noise
    - ``cost`` lengths a box between its trap and its line; the comment gives the line and the break."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.5, 49)
    rows, rid = [], 7000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        dogs = rng.permutation(np.arange(1, 49))
        for r in range(8):
            field = dogs[6 * r:6 * r + (5 if r == 7 else 6)]
            rid += 1
            ln = np.array([line_of(int(g), d) for g in field])
            traps = np.arange(1, len(field) + 1)
            perf = skill[field] + rng.normal(0, 1.0, len(field)) - cost * np.abs(traps - WANT[ln])
            t = 29.5 - perf * core.LENGTH_S
            pos = np.argsort(np.argsort(t)) + 1
            for k, dog in enumerate(field):
                com = f"{'QAw' if dog % 2 else 'SAw'},{LINES[ln[k]]}" + (",ALd" if pos[k] == 1 else "")
                rows.append({"race_date": day, "race_time": f"{14 + r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": int(traps[k]),
                             "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                             "sp_decimal": 6.0, "position": int(pos[k]), "beaten_lengths": 0.5, "sectional": 4.5,
                             "run_time": float(round(t[k], 2)), "adjusted_time": float(round(t[k], 2)),
                             "weight_kg": 30.0, "comment": com})
    return clean(pd.DataFrame(rows))


def table(runs):
    return line.build(runs, core.per_run_figures(runs))


def on(tab, keys):
    return keys.merge(tab, on=["race_id", "dog_id"], how="left")[COLS].to_numpy(float)


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = table(runs)
    m = runs.merge(tab, on=["race_id", "dog_id"])
    m["d"] = (pd.to_datetime(m.race_date) - pd.Timestamp("2025-01-01")).dt.days
    return runs, m, tab, notes


def test_one_row_a_run_and_no_key_twice(built):
    runs, m, tab, notes = built
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS
    assert all(c.startswith(line.PREFIX) for c in COLS) and all(line.COLUMNS.values())
    assert len(tab) == len(runs) == len(m) == notes["rows"]
    assert not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert (tab[COLS].dtypes == np.float32).all()
    twice = pd.concat([runs, runs.iloc[:0]], ignore_index=True)       # the same frame again: the same table
    assert same(table(twice)[0][COLS].to_numpy(float), tab[COLS].to_numpy(float))
    # without the engine's figures the comments are parsed here: the same table
    assert same(line.build(runs)[0][COLS].to_numpy(float), tab[COLS].to_numpy(float))


def test_the_planted_line_is_read_and_the_trap_off_it_costs(built):
    _, m, _, _ = built
    late = m[(m.d >= 10) & (m.dog_id != MOVER)]
    assert late.ln_path_hab.notna().all()
    assert np.allclose(late.ln_path_hab, late.dog_id % 3 + 1)         # rails 1, middle 2, wide 3
    assert np.allclose(late.ln_trap_mis, late.trap - WANT[late.dog_id % 3])
    assert np.allclose(late.ln_trap_mis_abs, late.ln_trap_mis.abs())
    # the planted effect: the further the trap from the line, the worse the finish (the result is never read)
    assert np.corrcoef(late.ln_trap_mis_abs, late.position)[0, 1] > 0.25
    near = late[late.ln_trap_mis_abs <= 1].position.mean()
    far = late[late.ln_trap_mis_abs >= 4].position.mean()
    assert far - near > 1.0
    # a steady dog has no drift; its first day has no history at all
    assert np.allclose(late.ln_path_drift.dropna(), 0.0)
    first = m[m.d == 0]
    assert first[["ln_path_hab", "ln_path_drift", "ln_trap_mis", "ln_first_trap", "ln_nb_brk"]].isna().all().all()
    assert (first.ln_nb_squeeze == 0).all() and (first.ln_nb_conflict == 0).all()


def test_a_dog_that_changes_its_line_drifts(built):
    _, m, _, _ = built
    one = m[m.dog_id == MOVER].sort_values("d")
    before = one[(one.d > 5) & (one.d <= SWITCH)]
    assert np.allclose(before.ln_path_hab, 1.0) and np.allclose(before.ln_path_drift, 0.0)
    after = one[one.d > SWITCH].reset_index(drop=True)
    # its first run after the switch still reads the rails (the day's own comment is not history)
    assert np.isclose(one[one.d == SWITCH].ln_path_hab.iloc[0], 1.0)
    assert np.isclose(after.ln_path_drift.iloc[0], 1.0)               # one wide day and one rails day against rails
    assert np.isclose(after.ln_path_drift.iloc[1], 2.0)               # two wide days against the rails
    assert after.ln_path_hab.is_monotonic_increasing and after.ln_path_hab.iloc[-1] > 1.3


def test_the_neighbours_lines_squeeze_and_a_missing_dog_leaves_a_trap_vacant(built):
    _, m, _, _ = built
    late = m[m.d >= 10].copy()
    by = late.set_index(["race_id", "trap"])
    lean = (2.5 * by.ln_path_hab - 1.5) - by.index.get_level_values("trap")
    exp_sq, exp_cf = [], []
    for (rid, tr), own in lean.items():
        i = lean.get((rid, tr - 1), np.nan)
        o = lean.get((rid, tr + 1), np.nan)
        si = max(i, 0) if np.isfinite(i) else 0.0
        so = max(-o, 0) if np.isfinite(o) else 0.0
        exp_sq.append(si + so)
        exp_cf.append(max(-own, 0) * si + max(own, 0) * so)
    assert np.allclose(by.ln_nb_squeeze, exp_sq, atol=1e-5) and np.allclose(by.ln_nb_conflict, exp_cf, atol=1e-5)
    assert by.ln_nb_squeeze.max() >= 7.0                              # a wide runner inside and a railer outside
    size = late.groupby("race_id").trap.transform("size")
    assert ((late.ln_nb_vacant == 1) == ((size == 5) & (late.trap == 5))).all()
    assert late.ln_nb_vacant.isin([0, 1]).all()
    # the break: quick breakers (odd dogs, 1) and slow ones (even, -1); the neighbours' mean less its own
    brk = pd.Series(np.where(by.dog_id % 2 == 1, 1.0, -1.0), index=by.index)
    for (rid, tr), own in brk.items():
        nb = [brk[(rid, tr + s)] for s in (-1, 1) if (rid, tr + s) in brk.index]
        assert np.isclose(by.ln_nb_brk[(rid, tr)], np.mean(nb) - own, atol=1e-5)


def test_trap_history(built):
    _, m, _, _ = built
    late = m[m.d >= 30]
    for _, r in late.sample(60, random_state=1).iterrows():
        past = m[(m.dog_id == r.dog_id) & (m.d < r.d)]
        assert np.isclose(r.ln_trap_vs_usual, r.trap - past.trap.mean(), atol=1e-5)
        assert r.ln_first_trap == float(not (past.trap == r.trap).any())
        best = past[past.position <= 2]
        if len(best):
            assert np.isclose(r.ln_trap_vs_best, abs(r.trap - best.trap.mean()), atol=1e-5)
        p = (past.position - 1) / (past.field - 1)
        nearby = p[(past.trap - r.trap).abs() <= 1]
        if len(nearby):
            assert np.isclose(r.ln_form_near_trap, nearby.mean() - p.mean(), atol=1e-5)
        else:
            assert np.isnan(r.ln_form_near_trap)


def rewrite_day(runs, date):
    """A day's results rewritten: positions reversed, times and comments changed."""
    r = runs.copy()
    hit = r.race_date == date
    r.loc[hit, "position"] = r.loc[hit, "field"] + 1 - r.loc[hit, "position"]
    r.loc[hit, ["adjusted_time", "run_time"]] += 0.5
    r.loc[hit, "comment"] = "VSAw,EP,Mid,ALd"
    r["won"] = (r.position == 1).astype(float)
    r["placed2"] = (r.position <= 2).astype(float)
    return r


def test_a_days_results_and_later_days_move_nothing_on_that_day(built):
    runs, _, tab, _ = built
    dates = sorted(runs.race_date.unique())
    for date in (dates[30], dates[-1]):
        upto = runs.loc[runs.race_date <= date, ["race_id", "dog_id"]]
        after = runs.loc[runs.race_date > date, ["race_id", "dog_id"]]
        changed, _ = table(rewrite_day(runs, date))
        assert same(on(changed, upto), on(tab, upto))                 # the day's own results are not read
        if len(after):
            assert not same(on(changed, after), on(tab, after))       # and they are read afterwards
        cut, _ = table(runs[runs.race_date <= date].reset_index(drop=True))
        assert same(on(cut, upto), on(tab, upto))                     # the later days change nothing


def test_a_card_row_gets_the_features_its_run_had(built):
    runs, _, tab, _ = built
    last = runs.race_date.max()
    r = runs.copy()
    hit = r.race_date == last
    for col in ("position", "beaten_lengths", "sectional", "run_time", "adjusted_time", "weight_kg", "sp_decimal"):
        r.loc[hit, col] = np.nan
    r.loc[hit, "comment"] = ""
    r["is_card"] = hit
    card = clean(r.drop(columns=["won", "placed2", "t", "field"]))
    keys = runs.loc[runs.race_date == last, ["race_id", "dog_id"]]
    assert card.is_card.sum() == len(keys) and card.loc[card.is_card, "position"].isna().all()
    got, _ = table(card)
    assert len(got) == len(card) and same(on(got, keys), on(tab, keys))
    x = on(got, keys)
    assert np.isfinite(x).mean(axis=0).min() > 0.9                    # made from history, every column
    # a card row is no history
    assert line.build(card)[1]["history_rows"] == int((~card.is_card).sum())


def test_a_trap_outside_one_to_six_has_no_trap_columns():
    runs = synth(days=12)
    odd = runs.copy()
    hit = (odd.race_date == odd.race_date.max()) & (odd.trap == 6)
    odd.loc[hit, "trap"] = 7
    tab, notes = table(odd)
    x = tab[hit.to_numpy()]
    assert notes["rows_without_a_trap_1_to_6"] == int(hit.sum()) > 0
    assert x[["ln_path_hab", "ln_path_drift"]].notna().all().all()
    assert x[["ln_trap_mis", "ln_first_trap", "ln_nb_squeeze", "ln_nb_vacant", "ln_nb_brk"]].isna().all().all()
    five = odd[(odd.race_date == odd.race_date.max()) & (odd.trap == 5)].merge(tab, on=["race_id", "dog_id"])
    assert (five.ln_nb_vacant == 1).all()                             # the dog in "7" is not beside trap 5


def test_a_withdrawn_dog_moves_only_the_neighbour_columns_of_its_race(built):
    """The field is the rows of the race: with a dog taken out (a non-runner: GBGB's results have no row for it)
    nothing moves but the four neighbour columns, and those only in its own race (the module's "field as run")."""
    runs, _, tab, _ = built
    last = runs[runs.race_date == runs.race_date.max()]
    gone = last[last.trap == 3].iloc[0]
    less = runs[~((runs.race_id == gone.race_id) & (runs.dog_id == gone.dog_id))].reset_index(drop=True)
    got, _ = table(less)
    keys = less[["race_id", "dog_id"]]
    a = keys.merge(got, on=["race_id", "dog_id"])
    b = keys.merge(tab, on=["race_id", "dog_id"])
    nb = [c for c in COLS if c.startswith("ln_nb_")]
    own = [c for c in COLS if c not in nb]
    assert len(nb) == 4 and same(a[own].to_numpy(float), b[own].to_numpy(float))
    other = (a.race_id != gone.race_id).to_numpy()
    assert same(a.loc[other, nb].to_numpy(float), b.loc[other, nb].to_numpy(float))
    beside = a[(a.race_id == gone.race_id)].merge(less[["race_id", "dog_id", "trap"]], on=["race_id", "dog_id"])
    assert (beside.loc[beside.trap.isin([2, 4]), "ln_nb_vacant"] == 1).all()
    assert (beside.loc[~beside.trap.isin([2, 4, 5]), "ln_nb_vacant"] == 0).all()
