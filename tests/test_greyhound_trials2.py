"""greyhound/pit/trials2.py (the feature review of 17 Nov 2026, "trials"): planted effects recovered on a constructed
frame (solo trials 2 lengths slower than races, trials in company 1, one session 3 lengths fast, the rest a length
either way), the company figure, the counts, and the point-in-time rule (the day's own results, and the card, move
nothing)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, trials, trials2
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S
NEW, NEW_SKILL = 99, 2.0
SOLO_OFF, COMP_OFF, FAST_DAY, FAST = 2.0, 1.0, 94, 3.0


def raw(days=126, seed=4):
    """One track, 500 m. 40 dogs race weekly (days 0, 7, ...) in fields of 6; each has a fixed skill (lengths). A trial
    session three days after each race day: six established dogs, three alone (2 lengths slower than they race), three
    together (1 length slower). The session of day 94 runs 3 lengths fast. Dog 99 (skill 2) trials alone in that
    session, in company on day 101, and races from day 105."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1.5, 100)
    skill[NEW] = NEW_SKILL
    rows, rid = [], 5000

    def row(day, rt, cls, dog, trap, t, pos):
        return {"race_date": (pd.Timestamp("2025-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                "race_time": rt, "meeting_id": day, "track": "Hove", "race_id": rid, "race_class": cls,
                "distance_m": 500.0, "trap": trap, "dog_id": int(dog), "dog_name": f"Dog {dog}", "sire": "S",
                "dam": "D", "born": "Jan-2023", "sex": "d", "trainer": "T", "sp_decimal": 6.0 if rt else np.nan,
                "position": pos, "beaten_lengths": 0.5, "sectional": 4.5 - 0.02 * skill[dog], "run_time": t,
                "adjusted_time": t, "weight_kg": 30.0, "comment": "QAw,Rls"}

    def heat(day, rt, cls, field, slower):
        nonlocal rid
        rid += 1
        perf = skill[field] + rng.normal(0, 0.3, len(field)) - slower
        t = 29.5 - perf * L
        pos = np.argsort(np.argsort(-perf)) + 1
        for k, dog in enumerate(field):
            rows.append(row(day, rt, cls, dog, k + 1, float(t[k]), int(pos[k])))

    for d in range(0, days, 7):
        pool = rng.permutation(np.arange(40))
        field = list(pool[:35]) + ([NEW] if d >= 105 else [])
        for r in range(6):
            heat(d, f"{18 + r}:00", "A3", np.array(field[r * 6:(r + 1) * 6]), 0.0)
        s = d + 3
        if s >= days:
            continue
        fast = FAST if s == FAST_DAY else float(rng.normal(0, 1.0))
        tri = rng.permutation(np.arange(40))[:6]
        for dog in tri[:3]:
            heat(s, "", "T1", np.array([dog]), SOLO_OFF - fast)
        comp = list(tri[3:]) if s != 101 else [tri[3], tri[4], NEW]
        heat(s, "", "T2", np.array(comp), COMP_OFF - fast)
        if s == FAST_DAY:
            heat(s, "", "T1", np.array([NEW]), SOLO_OFF - fast)
    return pd.DataFrame(rows), skill


@pytest.fixture(scope="module", autouse=True)
def fit_on_more_days():
    """The constructed frame has 18 sessions: the session k is fitted on the first 60% of the days."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trials2, "FIT_SHARE", 0.6)
        mp.setattr(trials2, "MIN_SESSIONS", 5)
        yield


@pytest.fixture(scope="module")
def built(fit_on_more_days):
    r, skill = raw()
    tr = trials.select(r)
    runs = clean(r)
    figs = core.per_run_figures(runs)
    tab, notes = trials2.build(runs, figs, {"trial_runs": tr})
    return r, skill, runs, figs, tr, runs.merge(tab, on=["race_id", "dog_id"]), notes


def scale(built):
    """c: the race scale's zero against skill (established dogs' race levels less their skill)."""
    _, skill, _, _, _, m, _ = built
    late = m[(m.race_date == m.race_date.max()) & (m.tq_race_n >= 5)]
    return float((late.tq_race_lvl - skill[late.dog_id]).mean())


def test_every_column_is_registered(built):
    _, _, runs, _, _, m, _ = built
    assert {c for c in m.columns if c.startswith(trials2.PREFIX)} == set(trials2.COLUMNS) and len(m) == len(runs)


def test_the_offsets_and_the_session_going_are_recovered(built):
    _, skill, _, figs, tr, _, _ = built
    t, _ = trials2.race_scale(tr, figs[~trials.is_trial_class(figs.race_class)], ("gsr",))
    late = t[t.dn >= t.dn.max() - 30]
    solo = late[late["size"] == 1].gsr_off
    comp = late[late["size"] >= 2].gsr_off
    assert abs(solo.mean() - SOLO_OFF) < 0.4 and abs(comp.mean() - COMP_OFF) < 0.4
    day94 = t[t.dn == t.dn.min() + FAST_DAY - 3]
    assert (day94.gsr_sess > 1.5).all()                       # the fast session found, every dog's own left out
    c = scale(built)
    new = t[t.dog_id == NEW]
    assert np.allclose(new.gsr_rs, NEW_SKILL + c, atol=1.0)    # both trials on the race scale
    assert abs(new.gsr_raw.iloc[0] + SOLO_OFF - FAST - (NEW_SKILL + c)) < 1.0      # raw: 1 length fast, net


def test_a_debutant_reads_its_trials_on_the_race_scale(built):
    _, _, _, _, _, m, _ = built
    c = scale(built)
    first = m[m.dog_id == NEW].sort_values("race_date").iloc[0]
    assert first.tq_days == 4 and first.tq_n_since_race == 2 and first.tq_race_n == 0 and np.isnan(first.tq_race_lvl)
    assert abs(first.tq_fig_last - (NEW_SKILL + c)) < 0.8 and abs(first.tq_lvl - (NEW_SKILL + c)) < 0.8
    assert first.tq_solo_last == 0 and first.tq_field_last == 3 and first.tq_comp_n_last == 2
    assert abs(first.tq_comp_fig_last - (NEW_SKILL + c)) < 1.2 and first.tq_comp_days == 4
    assert first.tq_pos_last >= 1 and np.isfinite(first.tq_margin_last)
    assert np.isnan(first.tq_gap) and first.tq_lvl_vs_field > 0


def test_an_established_dog(built):
    _, skill, _, _, _, m, _ = built
    late = m[(m.race_date == m.race_date.max()) & m.tq_lvl.notna() & (m.tq_race_n >= 5)]
    assert len(late) > 5
    assert abs((late.tq_gap).mean()) < 0.6                    # trials and races on one scale
    assert (late.tq_n_since_race >= 0).all() and (late.tq_n_since_race <= 1).all()


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    r, _ = raw()
    tr = trials.select(r)
    runs = clean(r)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = trials2.build(runs, core.per_run_figures(runs), {"trial_runs": tr})[0]
    r2 = change(runs)
    b = trials2.build(r2, core.per_run_figures(r2), {"trial_runs": tr})[0]
    cols = list(trials2.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).any() and np.allclose(x, z, atol=1e-4, equal_nan=True)


def test_a_trial_on_the_day_is_not_read():
    r, _ = raw()
    first = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=105)).strftime("%Y-%m-%d")
    late = r.assign(race_date=r.race_date.where(~((r.dog_id == NEW) & r.race_class.str.startswith("T")), first))
    runs = clean(late)
    tab = trials2.build(runs, core.per_run_figures(runs), {"trial_runs": trials.select(late)})[0]
    row = runs.merge(tab, on=["race_id", "dog_id"]).query("dog_id == @NEW").sort_values("race_date").iloc[0]
    assert np.isnan(row.tq_days) and np.isnan(row.tq_fig_last) and row.tq_n_since_race == 0


def test_no_trials():
    r, _ = raw()
    runs = clean(r)
    tab, notes = trials2.build(runs, core.per_run_figures(runs), {})
    assert notes["trial_rows"] == 0 and tab.tq_race_lvl.notna().mean() > 0.5 and tab.tq_fig_last.isna().all()
