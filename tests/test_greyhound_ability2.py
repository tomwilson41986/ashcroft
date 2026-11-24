"""greyhound/pit/ability2.py (the feature review of 17 Nov 2026, B2 and "ability"): ability.py's new keyword
arguments leave its output bit-identical; the trials reach the filter (a debutant comes to its first race with a
level); the damping and the band noise do what they say; the day's own results and the card move nothing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import ability, ability2, core, trials
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day, synth
from tests.test_greyhound_trials2 import NEW, NEW_SKILL, raw


# ability.filter_dog_states as it was before the optional arguments (base 95f231b), verbatim
def original_filter(dog: np.ndarray, dayn: np.ndarray, obs: np.ndarray, trial: np.ndarray, prm: dict):
    """Pre-run state (mean, variance, figures seen, days since the last figure) for rows sorted by dog then time.
    Every run on a day reads the state after the dog's EARLIER days; the variance grows by q a day since the last
    figure; a day without a figure changes nothing."""
    n = len(dog)
    mean, var, cnt, since = np.empty(n), np.empty(n), np.zeros(n), np.full(n, np.nan)
    q, r, m0, p0 = prm["q"], prm["r"], prm["m0"], prm["p0"]
    j = 0
    while j < n:
        d0 = dog[j]
        m, p, c, last = m0, p0, 0, None              # the state after the dog's last figure, and that figure's day
        while j < n and dog[j] == d0:
            k = j
            while k < n and dog[k] == d0 and dayn[k] == dayn[j]:
                k += 1
            gap = (dayn[j] - last) if last is not None else None
            p_now = p + q * gap if gap is not None else p
            mean[j:k], var[j:k], cnt[j:k] = m, p_now, c
            since[j:k] = gap if gap is not None else np.nan
            seen = False
            for t in range(j, k):                    # the day's figures, for the days after
                if np.isfinite(obs[t]):
                    pp = p_now if not seen else p
                    rr = r * (1.5 if trial[t] else 1.0)
                    kg = pp / (pp + rr)
                    m = m + kg * (obs[t] - m)
                    p = (1 - kg) * pp
                    c += 1
                    seen = True
            if seen:
                last = dayn[j]
            j = k
    return mean, var, cnt, since


def test_ability_filter_is_bit_identical_with_the_defaults():
    rng = np.random.default_rng(1)
    n = 3000
    dog = np.sort(rng.integers(0, 120, n))
    dayn = np.concatenate([np.sort(rng.integers(0, 400, (dog == d).sum())) for d in np.unique(dog)])
    obs = np.where(rng.random(n) < 0.1, np.nan, rng.normal(0, 3, n))
    trial = rng.random(n) < 0.2
    prm = {"q": 0.05, "r": 4.0, "m0": -1.0, "p0": 9.0}
    for a, b in zip(original_filter(dog, dayn, obs, trial, prm), ability.filter_dog_states(dog, dayn, obs, trial, prm)):
        assert np.array_equal(a, b, equal_nan=True)


def test_ability_table_is_bit_identical(monkeypatch):
    monkeypatch.setattr(ability, "FIT_BEFORE", "2025-01-15")
    runs = synth()
    figs = core.per_run_figures(runs)
    new = ability.build(runs, figs)[0]
    monkeypatch.setattr(ability, "filter_dog_states", original_filter)
    old = ability.build(runs, figs)[0]
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def test_the_damping_and_the_row_noise():
    dog, dayn = np.zeros(3, int), np.array([0, 365, 365 + 730])
    obs, trial = np.array([5.0, np.nan, np.nan]), np.zeros(3, bool)
    prm = {"q": 0.0, "r": 1e-9, "m0": 0.0, "p0": 1.0}
    m = ability.filter_dog_states(dog, dayn, obs, trial, prm, par=np.array([np.nan, 0.0, 1.0]), tau=365.0)[0]
    assert np.isclose(m[1], 5 * np.exp(-1))                      # the level read a year on: pulled to the par
    assert np.isclose(m[2], 1 + 4 * np.exp(-3))                   # no figure since: damped once from 5, not twice
    m = ability.filter_dog_states(dog, np.array([0, 1, 2]), np.array([4.0, 0.0, np.nan]), trial,
                                  {"q": 0.0, "r": 1.0, "m0": 0.0, "p0": 1.0}, r_row=np.array([1.0, 1e6, 1.0]))[0]
    assert np.isclose(m[1], 2.0) and np.isclose(m[2], 2.0, atol=1e-5)     # a noisy band's figure barely moves it


def test_the_band_noise_is_recovered():
    """Two trip bands, one twice as noisy a run (r 4x): the moments put r about 4x apart."""
    rng = np.random.default_rng(3)
    rows = []
    for d in range(3000):
        lvl, band = rng.normal(0, 1.5), d % 2
        days = np.cumsum(rng.integers(5, 15, 12))
        for k in days:
            rows.append({"dog_id": d, "dn": int(k), "trial": False, "band": 1 + 2 * band,
                         "gsr": lvl + rng.normal(0, 0.8 if band == 0 else 1.6), "par_gsr": 0.0,
                         "race_date": (pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(k))).strftime("%Y-%m-%d")})
    fr = pd.DataFrame(rows).sort_values(["dog_id", "dn"]).reset_index(drop=True)
    prm = {"q": 0.001, "r": 1.6, "m0": 0.0, "p0": 2.25}
    out = ability2.fit(fr, "gsr", prm)
    assert 2.5 < out["band_r"][3] / out["band_r"][1] < 6
    assert out["bands"] is not None                               # the likelihood prefers the bands


@pytest.fixture(scope="module")
def built():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ability, "FIT_BEFORE", "2025-02-15")
        r, skill = raw()
        tr = trials.select(r)
        runs = clean(r)
        figs = core.per_run_figures(runs)
        a2, notes = ability2.build(runs, figs, {"trial_runs": tr})
        a1 = ability.build(runs, figs)[0]
    return runs, runs.merge(a2, on=["race_id", "dog_id"]).merge(a1, on=["race_id", "dog_id"]), notes, skill


def test_every_column_is_registered(built):
    runs, m, _, _ = built
    assert {c for c in m.columns if c.startswith(ability2.PREFIX)} == set(ability2.COLUMNS) and len(m) == len(runs)


def test_a_debutant_comes_with_a_level_from_its_trials(built):
    _, m, notes, skill = built
    first = m[m.dog_id == NEW].sort_values("race_date").iloc[0]
    late = m[(m.race_date == m.race_date.max()) & (m.ka_n >= 5)]
    c = float((late.ka_gsr_mean - skill[late.dog_id]).mean())    # the race scale's zero against skill
    assert first.ka_n == 2 and first.ka_n_trials == 2 and first.ka_days == 4 and first.kf_n == 0
    assert abs(first.ka_gsr_mean - (NEW_SKILL + c)) < 1.0         # its trials, on the race scale
    assert first.ka_gsr_mean > first.kf_gsr_mean + 1.0            # ability.py: the debut prior
    assert first.ka_gsr_sd < first.kf_gsr_sd and first.ka_gsr_rank <= 2
    assert notes["trial_rows"] > 0 and notes["params"]["gsr"]["tau"] in ability2.TAUS


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change, monkeypatch):
    monkeypatch.setattr(ability, "FIT_BEFORE", "2025-02-15")
    r, _ = raw()
    tr = trials.select(r)
    runs = clean(r)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = ability2.build(runs, core.per_run_figures(runs), {"trial_runs": tr})[0]
    r2 = change(runs)
    b = ability2.build(r2, core.per_run_figures(r2), {"trial_runs": tr})[0]
    cols = list(ability2.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).any() and np.allclose(x, z, atol=1e-4, equal_nan=True)


def test_no_trials(monkeypatch):
    monkeypatch.setattr(ability, "FIT_BEFORE", "2025-02-15")
    runs = clean(raw()[0])
    tab, notes = ability2.build(runs, core.per_run_figures(runs), {})
    assert notes["trial_rows"] == 0 and (tab.ka_n_trials == 0).all() and tab.ka_gsr_mean.notna().all()


def test_the_cached_constants_give_the_same_table(tmp_path, monkeypatch):
    """The price run reads the nightly build's constants (ctx["fitted"]["ability2"], the stored notes written with
    default=str) instead of fitting again, and gets the same table; a stored table from other code is not read."""
    import json
    from greyhound import pit
    from sources.common import Store
    monkeypatch.setattr(ability, "FIT_BEFORE", "2025-02-15")
    r, _ = raw()
    tr = trials.select(r)
    runs = clean(r)
    figs = core.per_run_figures(runs)
    tab, notes = ability2.build(runs, figs, {"trial_runs": tr})
    stored = json.loads(json.dumps(notes, default=str))
    again, n2 = ability2.build(runs, figs, {"trial_runs": tr, "fitted": {"ability2": stored}})
    assert n2["constants"] == "cached" and notes["constants"] == "fitted"
    pd.testing.assert_frame_equal(tab, again)
    store = Store(root=tmp_path)
    store.put(f"{pit.STORE_PREFIX}/ability2.meta.json",
              json.dumps({"hash": pit.code_hash("ability2"), "notes": stored}, default=str).encode())
    assert "ability2" in pit.stored_fits(store, ["ability2", "draw2"])
    store.put(f"{pit.STORE_PREFIX}/ability2.meta.json", json.dumps({"hash": "other", "notes": stored}).encode())
    assert pit.stored_fits(store, ["ability2"]) == {}
