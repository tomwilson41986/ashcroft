"""greyhound/pit/litter.py: the litter as a live sensor. A litter that starts running 2 lengths faster than expected
shows as a negative surprise for the member that has not run since; the runt is found against its own sex; a dog with
no littermates, or no litter, reads nothing, and its own form never enters its own columns; every column equals a
brute-force reading of the earlier days; the day's own results move nothing and a card row gets what its run had."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, dyad, litter, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S
STEP, AWAY, BACK, DEBUT = 60, 55, 85, 50
#: dog -> (sire, dam, born, sex, weight). The spellings of litter A differ in case, spaces and punctuation only.
DOGS = {
    1: ("Fast Sire", "Good Dam", "Jan-2023", "d", 32.0), 2: ("FAST SIRE", "Good  Dam", "Jan-2023", "d", 32.0),
    3: ("Fast-Sire", "good dam", "JAN-2023", "b", 27.0), 4: ("fast sire ", "Good Dam.", "Jan-2023", "b", 27.0),
    5: ("Plain Sire", "Plain Dam", "Mar-2023", "d", 32.0), 6: ("Plain Sire", "Plain Dam", "Mar-2023", "d", 32.0),
    7: ("Plain Sire", "Plain Dam", "Mar-2023", "b", 27.0), 8: ("Plain Sire", "Plain Dam", "Mar-2023", "b", 27.0),
    9: ("Big Sire", "Big Dam", "May-2023", "d", 33.0), 10: ("Big Sire", "Big Dam", "May-2023", "d", 33.0),
    11: ("Big Sire", "Big Dam", "May-2023", "d", 28.0),           # the runt
    12: ("Big Sire", "Big Dam", "May-2023", "b", 26.0),           # its sister: no littermate of her sex
    13: ("Lone Sire", "Lone Dam", "Jun-2023", "d", 32.0),         # no littermates
    14: ("Two Sire", "Two Dam", "Jul-2023", "d", 32.0),           # wins nearly every race
    15: ("Two Sire", "Two Dam", "Jul-2023", "d", 32.0),
    16: (None, "Some Dam", "Aug-2023", "d", 32.0),                # no sire: no litter
    17: ("Late Sire", "Late Dam", "Sep-2023", "d", 32.0),
    18: ("Late Sire", "Late Dam", "Sep-2023", "d", 32.0),         # first runs on day DEBUT
    # the same sire and dam as litter B, another month: a different litter
    19: ("Plain Sire", "Plain Dam", "Nov-2023", "d", 32.0),
}


def synth(days=120, seed=11, lift=2.0):
    """One track, every dog racing daily in fields of six (the last race of the day takes the rest: five or fewer).
    Litter A (dogs 1-4) runs ``lift`` lengths faster from day STEP; dog 4 is away from day AWAY to day BACK. The
    lone dog 13 runs 3 lengths slower from day STEP. Dog 14 is 6 lengths faster than the rest throughout."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 5000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        on = [g for g in DOGS if not (g == 4 and AWAY <= d < BACK) and not (g == 18 and d < DEBUT)]
        order = rng.permutation(on)
        for r in range(0, len(order), 6):
            rid += 1
            field = [int(x) for x in order[r:r + 6]]
            t = 29.5 + rng.normal(0, 0.03, len(field))
            for k, dog in enumerate(field):
                if dog <= 4 and d >= STEP:
                    t[k] -= lift * L
                if dog == 13 and d >= STEP:
                    t[k] += 3.0 * L
                if dog == 14:
                    t[k] -= 6.0 * L
            pos = np.argsort(np.argsort(t)) + 1
            for k, dog in enumerate(field):
                sire, dam, born, sex, wt = DOGS[dog]
                rows.append({"race_date": day, "race_time": f"{18 + r // 6}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": k + 1, "dog_id": dog,
                             "dog_name": f"Dog {dog}", "sire": sire, "dam": dam, "born": born, "sex": sex,
                             "sp_decimal": 6.0, "position": int(pos[k]), "beaten_lengths": 0.5, "sectional": 4.5,
                             "run_time": float(t[k]), "adjusted_time": float(t[k]),
                             "weight_kg": round(wt + float(rng.normal(0, 0.1)), 1), "comment": "QAw"})
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def built():
    runs = synth()
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    tab, notes = litter.build(runs, figs, {"xtime": xt})
    m = runs.merge(tab, on=["race_id", "dog_id"])
    m["d"] = (pd.to_datetime(m.race_date) - pd.Timestamp("2025-01-01")).dt.days
    return runs, figs, xt, tab, m, notes


def test_every_column_is_registered_and_there_is_a_row_a_run(built):
    runs, _, _, tab, m, notes = built
    assert {c for c in tab.columns if c.startswith(litter.PREFIX)} == set(litter.COLUMNS)
    assert list(tab.columns) == ["race_id", "dog_id"] + list(litter.COLUMNS)
    assert len(tab) == len(runs) == len(m) and (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert all(tab[c].dtype == np.float32 for c in litter.COLUMNS)
    assert notes["rows"] == len(runs) and notes["rows_with_a_litter"] == int((runs.dog_id != 16).sum())
    assert notes["litters"] == 7 and notes["litters_with_two_or_more_dogs_run"] == 5
    assert notes["largest_litter_dogs_run"] == 4 and notes["dogs_under_more_than_one_litter_key"] == 0


def test_the_litters_step_reaches_the_dog_that_has_not_run_since(built):
    _, _, _, _, m, _ = built
    four = m[m.dog_id == 4].set_index("d")
    back = four.loc[BACK]
    # dogs 1-3 have run 2 lengths faster than expected for 25 days: just over half the decayed weight (1 - 0.5 **
    # (25 / 21) = 0.56), less what their expected times have learnt of it
    # (the real expected-time model here: the residual is -1.7 lengths in the first week and -0.9 by the fourth)
    assert -2.2 < back.lt_surprise < -0.5
    assert back.lt_size == 4 and back.lt_active == 3 and back.lt_surprise_n > 20
    before = four[(four.index >= 30) & (four.index < AWAY)]
    assert before.lt_surprise.abs().max() < 0.15
    # the litter that did nothing reads nothing on the same days
    plain = m[m.dog_id.isin([5, 6, 7, 8]) & (m.d >= STEP)]
    assert plain.lt_surprise.abs().max() < 0.15 and plain.lt_surprise.notna().all()
    # and the members still running read it too, from the other three
    late = m[m.dog_id.isin([1, 2, 3]) & m.d.between(STEP + 10, STEP + 20)]
    assert late.lt_surprise.mean() < -0.3


def test_the_spellings_of_a_name_are_one_litter_and_another_month_is_not(built):
    _, _, _, _, m, _ = built
    end = m[m.d == m.d.max()].set_index("dog_id")
    assert (end.loc[[1, 2, 3, 4], "lt_size"] == 4).all() and (end.loc[[1, 2, 3, 4], "lt_active"] == 3).all()
    assert (end.loc[[5, 6, 7, 8], "lt_size"] == 4).all()                       # dog 19 is not one of them
    assert end.loc[19, "lt_size"] == 1 and end.loc[19, "lt_active"] == 0 and np.isnan(end.loc[19, "lt_surprise"])


def test_the_runt_is_found_against_its_own_sex(built):
    _, _, _, _, m, _ = built
    late = m[m.d >= 30]
    runt, brothers, sister = late[late.dog_id == 11], late[late.dog_id.isin([9, 10])], late[late.dog_id == 12]
    assert abs(runt.lt_wt.mean() + 5.0) < 0.2                                 # 28 against two of 33
    assert abs(brothers.lt_wt.mean() - 2.5) < 0.2                             # 33 against a 33 and a 28
    assert sister.lt_wt.isna().all()                                          # the dogs' weights are not hers to meet
    assert sister.lt_surprise.notna().all()                                   # though they are her litter
    # two dogs and two bitches of the same weights: nothing between them
    assert late[late.dog_id.isin([5, 6, 7, 8])].lt_wt.abs().mean() < 0.2


def test_a_dog_with_no_littermates_reads_nothing_and_never_itself(built):
    _, _, xt, _, m, _ = built
    lone = m[m.dog_id == 13]
    for c in ("lt_surprise", "lt_level", "lt_level_gap", "lt_wt", "lt_won"):
        assert lone[c].isna().all(), c
    assert (lone.lt_surprise_n == 0).all() and (lone.lt_active == 0).all() and (lone.lt_start_gap == 0).all()
    assert list(lone.sort_values("d").lt_size.iloc[:2]) == [0, 1] and lone.lt_size.max() == 1
    none = m[m.dog_id == 16]
    assert none[list(litter.COLUMNS)].isna().all().all()                      # no sire: no litter, every column NaN
    # a litter of two: each reads the other, not itself. Dog 14 wins nearly every race; the gap between the two is
    # the gap between their levels in the expected-time table (which gives part of dog 14's speed to its preference
    # for the one track and trip of this data set, so the level gap is well under the 6 lengths planted)
    late = m[m.d >= 30]
    winner, other = late[late.dog_id == 14], late[late.dog_id == 15]
    assert winner.lt_won.mean() < 0.15 and other.lt_won.mean() > 0.85
    lv = late[["race_id", "dog_id"]].merge(xt, on=["race_id", "dog_id"]).groupby("dog_id").xt_tot_level.mean()
    apart = float(lv[14] - lv[15])
    assert apart > 0.5
    assert abs(winner.lt_level_gap.mean() + apart) < 0.15 and abs(other.lt_level_gap.mean() - apart) < 0.15
    assert (winner.lt_size == 2).all() and (winner.lt_active == 1).all()


def test_the_start_gap_and_the_first_day(built):
    _, _, _, _, m, _ = built
    first = m[m.d == 0]
    ok = first.dog_id != 16
    assert (first[ok].lt_size == 0).all() and (first[ok].lt_start_gap == 0).all()
    assert first.lt_surprise.isna().all() and (first[ok].lt_surprise_n == 0).all()
    late = m[m.dog_id == 18].sort_values("d")
    assert late.d.iloc[0] == DEBUT and (late.lt_start_gap == DEBUT).all()     # on its first day (to today) and after
    assert late.lt_size.iloc[0] == 1 and late.lt_size.iloc[1] == 2
    assert (m[m.dog_id == 17].lt_start_gap == 0).all()
    # dog 4 away: the others see it go out of the window, and come back
    one = m[m.dog_id == 1].set_index("d")
    assert one.loc[AWAY + litter.WINDOW - 1, "lt_active"] == 3 and one.loc[AWAY + litter.WINDOW, "lt_active"] == 2
    assert one.loc[BACK, "lt_active"] == 2 and one.loc[BACK + 1, "lt_active"] == 3


def test_every_column_is_the_brute_force_reading_of_the_earlier_days(built):
    runs, figs, xt, tab, _, _ = built
    j = figs[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_tot_time", "xt_tot_level"]],
                                         on=["race_id", "dog_id"], how="left")
    resid = dyad.residuals(figs, pd.Series(j.xt_tot_time.to_numpy(), index=figs.index))
    # against the race: less the mean over the race's runners that have one, two or more needed (row by row)
    centred = np.full(len(figs), np.nan)
    race_ids = figs.race_id.to_numpy()
    for i in range(len(figs)):
        same = resid.to_numpy()[race_ids == race_ids[i]]
        if np.isfinite(resid.to_numpy()[i]) and np.isfinite(same).sum() >= 2:
            centred[i] = resid.to_numpy()[i] - np.nanmean(same)
    resid = centred
    level = j.xt_tot_level.to_numpy(float)
    dog = figs.dog_id.to_numpy()
    dn = (pd.to_datetime(figs.race_date) - pd.Timestamp("2025-01-01")).dt.days.to_numpy()
    key = {g: ("".join(ch for ch in str(v[0]).lower() if ch.isalnum()), "".join(ch for ch in v[1].lower() if ch.isalnum()),
               v[2].lower().replace("-", "")) if v[0] else None for g, v in DOGS.items()}
    lit = np.array([hash(key[g]) if key[g] else -1 for g in dog])
    sex = figs.sex.to_numpy()
    wt, won = figs.weight_kg.to_numpy(float), figs.won.to_numpy(float)

    def mean(mask, v, age, hl):
        w = 0.5 ** (age[mask] / hl)
        return ((w * v[mask]).sum() / w.sum() if w.sum() >= litter.MIN_W else np.nan), w.sum()

    exp = np.full((len(figs), len(litter.COLUMNS)), np.nan)
    for i in range(len(figs)):
        if lit[i] == -1:
            continue
        mine = (lit == lit[i]) & (dn < dn[i])
        mates, own = mine & (dog != dog[i]), mine & (dog == dog[i])
        age = (dn[i] - dn).astype(float)
        sur, n = mean(mates & np.isfinite(resid), resid, age, litter.HL_SURPRISE)
        lev, _ = mean(mates & np.isfinite(level), level, age, litter.HL_LEVEL)
        w_m, _ = mean(mates & (sex == sex[i]), wt, age, litter.HL_WT)
        wn, _ = mean(mates, won, age, litter.HL_WON)
        last_wt = wt[own][np.argmax(dn[own])] if own.any() else np.nan
        first_l = dn[mine].min() if mine.any() else np.nan
        first_o = dn[own].min() if own.any() else dn[i]
        exp[i] = [len(set(dog[mine])), len(set(dog[mates & (dn >= dn[i] - litter.WINDOW)])), sur, n, lev,
                  lev - level[i], last_wt - w_m, first_o - first_l if mine.any() else 0.0, wn]
    got = tab[list(litter.COLUMNS)].to_numpy(float)
    assert np.isfinite(exp).mean() > 0.8
    assert (np.isnan(got) == np.isnan(exp)).all()
    assert np.allclose(got, exp, atol=2e-3, equal_nan=True)


def test_earlier_days_only_with_and_without_decay():
    key = np.array([1, 1, 1, 2, 1, -1])
    dn = np.array([5, 5, 6, 5, 7, 7])
    v = np.array([10.0, 20.0, 40.0, 1.0, 80.0, 3.0])
    s, = litter.earlier_by(key, dn, [v])
    assert list(s[:5]) == [0.0, 0.0, 30.0, 0.0, 70.0] and np.isnan(s[5])       # the day's own rows are left out
    h, = litter.earlier_by(key, dn, [v], half_life=1.0)
    assert np.allclose(h[:5], [0.0, 0.0, 15.0, 0.0, 0.25 * 30.0 + 0.5 * 40.0]) and np.isnan(h[5])
    # a long span at a short half-life stays exact (the sums are carried scaled from the key's first day)
    far, = litter.earlier_by(np.zeros(3, int), np.array([0, 4000, 4021]), [np.array([5.0, 8.0, 1.0])], half_life=21.0)
    assert np.allclose(far, [0.0, 5.0 * 0.5 ** (4000 / 21), 4.0 + 5.0 * 0.5 ** (4021 / 21)])
    # the day's own value is never in the arithmetic: one large enough to swallow the earlier sum changes nothing
    # ((30 + 1e30) - 1e30 is 0, not 30: the sum is the running total as it stood the day before, not a subtraction)
    for big in (1e30, -1e30):
        w = v.copy()
        w[4] = big
        for hl in (None, 1.0, 21.0):
            a, b = litter.earlier_by(key, dn, [v], half_life=hl)[0], litter.earlier_by(key, dn, [w], half_life=hl)[0]
            assert np.array_equal(a, b, equal_nan=True), (big, hl)


def test_two_runs_on_a_day_and_a_dog_seen_only_on_the_card():
    runs = synth(days=40)
    last = runs.race_date.max()
    # dog 1 runs twice on the last day; a new dog of litter A is on the last day's card only
    again = runs[(runs.race_date == last) & (runs.dog_id == 1)].assign(race_id=99990, race_time="21:00", trap=1)
    new = again.assign(race_id=99991, dog_id=77, dog_name="Dog 77", trap=2)
    r = clean(pd.concat([runs, again, new], ignore_index=True).drop(columns=["t", "field", "won"]))
    tab = litter.build(r, core.per_run_figures(r))[0]
    m = r.merge(tab, on=["race_id", "dog_id"])
    assert len(tab) == len(r)
    twice = m[(m.race_date == last) & (m.dog_id == 1)]
    cols = list(litter.COLUMNS)
    assert len(twice) == 2 and np.allclose(twice[cols].iloc[0].to_numpy(float), twice[cols].iloc[1].to_numpy(float),
                                           equal_nan=True)
    seven = m[m.dog_id == 77].iloc[0]
    assert seven.lt_size == 4 and seven.lt_active == 4 and seven.lt_start_gap == 39 and np.isfinite(seven.lt_surprise)
    assert np.isfinite(seven.lt_level) and np.isfinite(seven.lt_won)
    assert np.isnan(seven.lt_wt)                                              # no weight of its own before today


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=40)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = litter.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = litter.build(r2, core.per_run_figures(r2))[0]
    cols = list(litter.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).mean() > 0.7 and np.allclose(x, z, atol=1e-4, equal_nan=True)


def rewrite_day(runs, day):
    """One day's results rewritten: the finishing order reversed in each race, the times, the sectionals, the
    weights and the comments changed (``perturb_last_day`` leaves the weights alone)."""
    r = runs.copy()
    k = r.race_date == day
    r.loc[k, "position"] = r[k].groupby("race_id").position.transform(lambda p: p.max() + 1 - p)
    r.loc[k, ["adjusted_time", "run_time"]] += 0.6
    r.loc[k, "sectional"] += 0.2
    r.loc[k, "weight_kg"] += 4.0
    r.loc[k, "comment"] = "SAw,Bmp1,Wide"
    r["won"] = (r.position == 1).astype(float)
    return r


@pytest.mark.parametrize("d", [45, 70, 100])
def test_a_middle_days_results_move_nothing_on_that_day_or_before(built, d):
    """The whole path (the expected-time table rebuilt on the rewritten results): day D's results reach no row of day
    D, in any race, and no earlier row, to the last bit; they do reach the days after."""
    runs, _, _, tab, m, _ = built
    day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
    r2 = rewrite_day(runs, day)
    b = litter.build(r2, core.per_run_figures(r2))[0]
    cols = list(litter.COLUMNS)
    upto, after = (runs.race_date <= day).to_numpy(), (runs.race_date > day).to_numpy()
    x, z = tab[cols].to_numpy(float), b[cols].to_numpy(float)
    assert (b.race_id.to_numpy() == tab.race_id.to_numpy()).all() and (b.dog_id.to_numpy() == tab.dog_id.to_numpy()).all()
    assert upto.sum() > 700 and np.isfinite(x[upto]).mean() > 0.7
    assert np.array_equal(x[upto], z[upto], equal_nan=True)
    moved = [c for j, c in enumerate(cols) if not np.allclose(x[after, j], z[after, j], atol=1e-3, equal_nan=True)]
    assert {"lt_surprise", "lt_won", "lt_wt"} <= set(moved)                    # the rewrite has teeth: later days read it


@pytest.mark.parametrize("d", [30, 61, 90])
def test_the_days_after_are_not_read(built, d):
    """The table of the runs to day D is the table of all the runs, cut at day D, to the last bit (the expected-time
    table held: its filter constants are fitted on the first fifth of whatever days it is given)."""
    runs, _, xt, tab, _, _ = built
    day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
    keep = (runs.race_date <= day).to_numpy()
    r1 = runs[keep]
    a = litter.build(r1, core.per_run_figures(r1), {"xtime": xt})[0]
    cols = list(litter.COLUMNS)
    assert len(a) == keep.sum() and (a.race_id.to_numpy() == tab.race_id.to_numpy()[keep]).all()
    assert np.array_equal(a[cols].to_numpy(float), tab[cols].to_numpy(float)[keep], equal_nan=True)


def test_five_runner_races_odd_traps_and_a_dog_outside_the_litter(built):
    """The table reads no trap and no field size, and a dog of another litter is nothing to this one: the lone dog 13
    taken out of every race on the even days (five-runner races, a race left empty is gone) and trap 6 renumbered 8
    move no other row, and the rows of the short races and the odd traps have their features."""
    runs, _, xt, tab, _, _ = built
    d = (pd.to_datetime(runs.race_date) - pd.Timestamp("2025-01-01")).dt.days
    keep = ~((runs.dog_id == 13) & (d % 2 == 0)).to_numpy()
    r = runs[keep].copy()
    r["trap"] = r.trap.where(r.trap != 6, 8)
    r["field"] = r.groupby("race_id").dog_id.transform("size")
    assert (r.field == 5).any() and (r.trap == 8).any()
    a = litter.build(r, core.per_run_figures(r), {"xtime": xt})[0]
    cols = list(litter.COLUMNS)
    x, z = tab[cols].to_numpy(float)[keep], a[cols].to_numpy(float)
    assert len(a) == len(r) and (a.dog_id.to_numpy() == r.dog_id.to_numpy()).all()
    rest = (r.dog_id != 13).to_numpy()
    free = [i for i, c in enumerate(cols) if c != "lt_surprise"]
    assert np.array_equal(x[rest][:, free], z[rest][:, free], equal_nan=True)
    # the surprise is read against the race's field, so a runner taken out of a race moves it a little and no more
    k = cols.index("lt_surprise")
    assert (np.isnan(x[rest][:, k]) == np.isnan(z[rest][:, k])).all()
    assert np.nanmax(np.abs(x[rest][:, k] - z[rest][:, k])) < 1.0
    odd = ((r.field == 5) | (r.trap == 8)).to_numpy() & (r.dog_id != 16).to_numpy() & (r.race_date > "2025-01-01").to_numpy()
    assert odd.sum() > 100 and np.isfinite(z[odd][:, cols.index("lt_size")]).all()
    assert np.isfinite(z[odd][:, cols.index("lt_surprise")]).mean() > 0.9


def test_the_expected_time_table_is_taken_from_the_context(built):
    runs, figs, xt, tab, _, _ = built
    b = litter.build(runs, figs)[0]
    assert np.allclose(tab[list(litter.COLUMNS)].to_numpy(float), b[list(litter.COLUMNS)].to_numpy(float),
                       equal_nan=True)
