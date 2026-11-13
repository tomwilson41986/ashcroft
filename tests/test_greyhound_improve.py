"""greyhound/pit/improve.py: the dog's own trend, the exposure of today's conditions, the nth-time curves and hidden
form. Planted here (unit-test data, not real): a dog's first outing at a track costs lengths; some dogs improve a
run; a troubled run is followed by a faster one. The table finds each. The day's own results move nothing on that
day (the last day and a middle day), nor do the days after it; a card row gets what its run had; the keys are the
rows of ``runs``, once each, with a dog out twice a day and trials among the outings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, improve
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S
NEW_TRACK = 4.0            # lengths a dog loses on its first outing at a track
GAIN = 0.5                 # lengths a run an improver gains over its first IMPROVE_RUNS runs
IMPROVE_RUNS = 16
BOUNCE = 3.0               # lengths faster the run after a troubled one
TRACKS = ("Hove", "Romford")
COLS = list(improve.COLUMNS)


def synth(days=150, seed=3):
    """Two tracks, 150 dogs first seen over the first 100 days, each with a home track and one day in six away; six
    races of six a track a day. Dogs with an id divisible by 4 improve. A fifth of runs are troubled ("Crd1": no
    cost on the day, the next run faster)."""
    rng = np.random.default_rng(seed)
    n = 150
    skill = rng.normal(0, 1.0, n + 1)
    debut = rng.integers(0, 100, n + 1)
    debut[1:61] = 0
    home = rng.integers(0, 2, n + 1)
    nrun = np.zeros(n + 1, int)
    seen = np.zeros((n + 1, 2), bool)
    owed = np.zeros(n + 1)
    rows, rid = [], 7000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        ready = np.flatnonzero(debut <= d)[1:] if debut[0] <= d else np.flatnonzero(debut <= d)
        ready = ready[ready > 0]
        ready = rng.permutation(ready[rng.random(len(ready)) < 0.6])
        at = np.where(rng.random(len(ready)) < 1 / 6, 1 - home[ready], home[ready])
        for ti, track in enumerate(TRACKS):
            dogs = ready[at == ti]
            for r in range(min(6, len(dogs) // 6)):
                field = dogs[6 * r:6 * r + 6]
                rid += 1
                crd = rng.random(6) < 0.2
                perf = (skill[field] + rng.normal(0, 1.0, 6) - np.where(seen[field, ti], 0.0, NEW_TRACK)
                        + np.where(field % 4 == 0, GAIN * np.minimum(nrun[field], IMPROVE_RUNS), 0.0) + owed[field])
                owed[field] = np.where(crd, BOUNCE, 0.0)
                seen[field, ti] = True
                nrun[field] += 1
                t = 29.5 + 0.3 * ti - perf * L
                order = np.argsort(t)
                pos = np.argsort(order) + 1
                sp = np.argsort(np.argsort(-skill[field])) + 1
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{18 + r}:{10 * ti:02d}", "meeting_id": d * 10 + ti,
                                 "track": track, "race_id": rid, "race_class": f"A{1 + dog % 3}", "distance_m": 480.0,
                                 "trap": k + 1, "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023",
                                 "sex": "b" if dog % 2 else "d", "sp_decimal": float(2 + sp[k]),
                                 "market_pos": int(sp[k]), "position": int(pos[k]), "beaten_lengths": 0.5,
                                 "sectional": float(round(4.5 - perf[k] * L / 4, 3)), "run_time": float(round(t[k], 3)),
                                 "adjusted_time": float(round(t[k], 3)), "weight_kg": 30.0,
                                 "comment": "Crd1,Mid" if crd[k] else "QAw,Rls"})
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def runs():
    return synth()


@pytest.fixture(scope="module")
def built(runs):
    mp = pytest.MonkeyPatch()
    mp.setattr(improve, "MIN_PAIRS", 300)
    tab, notes = improve.build(runs)
    mp.undo()
    return runs.merge(tab, on=["race_id", "dog_id"]), tab, notes


def _build(r, **kw):
    mp = pytest.MonkeyPatch()
    mp.setattr(improve, "MIN_PAIRS", 300)
    try:
        return improve.build(r, None, kw or None)[0]
    finally:
        mp.undo()


def _same(a, b, keys):
    a = keys.merge(a, on=["race_id", "dog_id"], how="left")[COLS].to_numpy(float)
    b = keys.merge(b, on=["race_id", "dog_id"], how="left")[COLS].to_numpy(float)
    assert (np.isnan(a) == np.isnan(b)).all()
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=1e-4, atol=1e-4)
    return a


def test_every_column_is_registered(runs, built):
    m, tab, notes = built
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS and all(c.startswith(improve.PREFIX) for c in COLS)
    assert (tab[COLS].dtypes == np.float32).all() and len(tab) == len(runs) == len(m)
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    late = m[m.race_date >= "2025-04-15"]
    assert (late[COLS].notna().mean() > 0.25).all(), late[COLS].notna().mean()
    assert notes["excuse_months_fitted"] >= 2 and "im_slope_age" not in COLS      # the reviewer's duplicate is gone


def test_the_planted_effects_are_found(built):
    m, _, notes = built
    m = m.sort_values(["dog_id", "t"]).reset_index(drop=True)
    m["n_run"] = m.groupby("dog_id").cumcount()
    m["n_track"] = m.groupby(["dog_id", "track"]).cumcount()
    first_seen = m.groupby("dog_id").race_date.transform("min")
    late = m[(m.race_date >= "2025-04-01") & (first_seen >= "2025-02-20")]            # whole careers in the frame
    # the first outing at a track costs NEW_TRACK lengths against the level: the nth-time curve has learned it
    new, old = late[late.n_track == 0].im_nth_track, late[late.n_track >= 4].im_nth_track
    assert len(new) >= 15 and len(old) > 200
    assert new.mean() < old.mean() - 0.5 * NEW_TRACK, (new.mean(), old.mean())
    assert late[late.n_track == 0].im_exp_sum.mean() < late[late.n_track >= 4].im_exp_sum.mean() - 0.5 * NEW_TRACK
    # an improver's last six figures slope upwards
    mid = m[(m.race_date >= "2025-02-01") & (m.n_run >= 7) & (m.n_run <= IMPROVE_RUNS)]
    up, flat = mid[mid.dog_id % 4 == 0].im_slope_run, mid[mid.dog_id % 4 != 0].im_slope_run
    assert len(up) > 50 and up.mean() > flat.mean() + 0.4 * GAIN, (up.mean(), flat.mean())
    # the run after a troubled one is faster: the excuse weights find the trouble, and the score carries it forward
    assert notes["excuse_weights_last_month"]["z_trouble"] > 0.3 * BOUNCE
    m["crd_before"] = m.groupby("dog_id").comment.shift(1).fillna("").str.startswith("Crd")
    late = m[m.race_date >= "2025-04-01"]
    a, b = late[late.crd_before].im_hidden, late[~late.crd_before & (late.n_run > 0)].im_hidden
    assert a.mean() > b.mean() + 0.3 * BOUNCE, (a.mean(), b.mean())
    assert late[late.crd_before].im_uncashed.mean() > late[~late.crd_before & (late.n_run > 0)].im_uncashed.mean()
    # the exposure of today's conditions: a debutant has met nothing
    assert (m[m.n_run == 0].im_cover == 0).all() and late[late.n_run >= 10].im_cover.mean() > 0.7


def test_the_days_own_results_move_nothing(runs, built):
    _, tab, _ = built
    last = runs.race_date.max()
    keys = runs.loc[runs.race_date == last, ["race_id", "dog_id"]]
    a = _same(tab, _build(perturb_last_day(runs)), keys)
    assert np.isfinite(a).mean() > 0.5
    # a middle day rewritten: that day and every day before it are as they were; later days do move
    mid = "2025-04-20"
    r = runs.copy()
    on = r.race_date == mid
    r.loc[on, "position"] = 7 - r.loc[on, "position"]
    r.loc[on, ["adjusted_time", "run_time"]] += 0.6
    r.loc[on, "sectional"] += 0.2
    r.loc[on, "comment"] = "SAw,Bmp1,Wide"
    r.loc[on, "market_pos"] = 7 - r.loc[on, "market_pos"]
    r["won"] = (r.position == 1).astype(float)
    b = _build(r)
    _same(tab, b, runs.loc[runs.race_date <= mid, ["race_id", "dog_id"]])
    after = runs.loc[runs.race_date > mid, ["race_id", "dog_id"]]
    x, y = after.merge(tab, on=["race_id", "dog_id"])[COLS].to_numpy(float), after.merge(b, on=["race_id", "dog_id"])[COLS].to_numpy(float)
    assert (np.abs(np.nan_to_num(x) - np.nan_to_num(y)) > 1e-3).any()


def test_later_days_move_nothing(runs, built):
    _, tab, _ = built
    for cut in ("2025-03-10", "2025-04-20"):
        early = runs[runs.race_date <= cut].reset_index(drop=True)
        _same(tab, _build(early), early[["race_id", "dog_id"]])


def test_a_card_row_gets_what_its_run_had(runs, built):
    _, tab, _ = built
    card = cardify_last_day(runs)
    assert card.is_card.sum() > 0 and card.loc[card.is_card, "position"].isna().all()
    keys = card.loc[card.is_card, ["race_id", "dog_id"]]
    c = _build(card)
    a = _same(tab, c, keys)
    assert len(c) == len(card) and np.isfinite(a).mean() > 0.5 and np.isfinite(a).any(0).all()
    # whatever a card row carries in a result column is not read
    junk = card.copy()
    junk.loc[junk.is_card, ["position", "market_pos"]] = 1.0
    junk.loc[junk.is_card, ["adjusted_time", "sectional"]] = 1.0
    junk.loc[junk.is_card, "comment"] = "Crd1,Wide"
    _same(c, _build(junk), keys)


def test_no_duplicate_keys_with_two_runs_a_day_and_trials(runs):
    # a dog out twice on the last day (a second race), trial rows handed over, and a T-class row inside runs
    last = runs.race_date.max()
    one = runs[runs.race_date == last].iloc[[0]].copy()
    one["race_id"] = 999001
    one["race_time"], one["position"] = "22:00", 1
    trial_in = runs[runs.race_date == "2025-03-01"].iloc[[0]].copy()
    trial_in["race_id"], trial_in["race_class"], trial_in["position"] = 999002, "T3", 1
    r = clean(pd.concat([runs, one, trial_in], ignore_index=True).drop(columns=["t", "field", "won", "placed2"]))
    dog = int(runs[runs.race_date == "2025-04-10"].dog_id.iloc[0])
    trials = pd.DataFrame({"race_id": [888001, 888002], "dog_id": [dog, dog], "race_date": ["2025-01-20", "2025-04-09"],
                           "track": ["Hove", "Romford"], "distance_m": [480.0, 480.0], "trap": [1, 2]})
    tab = _build(r, trial_runs=trials)
    assert len(tab) == len(r) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == r.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == r.dog_id.to_numpy()).all()
    # the second run of the day reads what the first did (nothing of the day)
    both = tab[(tab.dog_id == int(one.dog_id.iloc[0])) & tab.race_id.isin([999001, int(runs[runs.race_date == last].race_id.iloc[0])])]
    assert len(both) == 2
    for c in ("im_slope_run", "im_curve", "im_hidden", "im_runs", "im_beat_up"):
        v = both[c].to_numpy(float)
        assert (np.isnan(v).all()) or np.isclose(v[0], v[1]), c
    # a trial is an outing, not a race: it counts in the exposure and has no features of its own
    assert tab.loc[tab.race_id == 999002, COLS].isna().all().all()
    base = _build(r)
    k = r.loc[(r.dog_id == dog) & (r.race_date == "2025-04-10"), ["race_id", "dog_id"]]
    with_t, without = k.merge(tab, on=["race_id", "dog_id"]), k.merge(base, on=["race_id", "dog_id"])
    assert (with_t.im_break_exp.to_numpy() != without.im_break_exp.to_numpy()).any() or \
        (with_t.im_cover.to_numpy() >= without.im_cover.to_numpy()).all()
