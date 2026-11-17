"""greyhound/pit/xtime2.py (the review of 17 Nov 2026, "xtime"): xtime.py's new keywords leave the xt_ table as it
was; the forgetting trap effect follows a trap that changes; the constants are fitted and noted; the preference no
longer takes the level's residual twice; P(fastest) matches a simulation; a vacant box is room; the day's own results
move nothing. Constructed frames only (tests/test_greyhound_xtime.py ``synth``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, xtime, xtime2
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day
from tests.test_greyhound_xtime import synth


@pytest.fixture()
def small(monkeypatch):
    monkeypatch.setattr(xtime, "MIN_FIT", 500)
    monkeypatch.setattr(xtime, "REFIT_MONTHS", 1)
    monkeypatch.setattr(xtime, "MIN_COUNT", 50)
    monkeypatch.setattr(xtime, "LAM", 1.0)


def _gappy(days=90, seed=1):
    """synth with trap 4 out of every third race (vacant boxes), cleaned again (the field size is the race's)."""
    runs, skill = synth(days=days, seed=seed)
    r = runs[~((runs.race_id % 3 == 0) & (runs.trap == 4))].reset_index(drop=True)
    return clean(r.drop(columns=[c for c in ("won", "placed2", "sp_p", "sp_p_norm", "t", "field") if c in r])), skill


def _prepared(runs):
    df = core.per_run_figures(runs).sort_values(["t", "race_id", "trap"], kind="stable")
    return xtime.prepare(df), xtime.targets(df)["tot"]


PRM = {"r": 1.6, "q": 0.001, "p0": 0.3}


# ---- the xt_ table is unchanged ------------------------------------------------------------------------------------

def test_run_filter_defaults_equal_explicit_xtime_values():
    runs, _ = _gappy(days=60)
    P, y = _prepared(runs)
    a = xtime.run_filter(P, y, PRM)
    b = xtime.run_filter(P, y, PRM, consts={"k_trap": xtime.K_TRAP, "k_grade": xtime.K_GRADE, "k_pref": xtime.K_PREF,
                                            "std_cap": xtime.STD_CAP, "noisy": xtime.NOISY},
                         forget={"trap": float("inf"), "grade": None}, pref_post=False)
    for k in ("a", "level", "sd", "pref", "trap", "grade", "e", "e1", "e2", "n", "xb"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k


def test_build_defaults_equal_explicit_keywords(small):
    """The xt_ table with every new keyword at its default, and at explicit values equal to the defaults: the same
    bits (the run against the base commit's xtime.py on these frames, 17 Nov 2026, was identical too)."""
    runs, _ = _gappy(days=70)
    figs = core.per_run_figures(runs)
    a, na = xtime.build(runs, figs)

    def tune(P, y, early_to, name):
        prm, grid = xtime.fit_constants(P, y, early_to)
        return prm, {}, {"grid": grid}

    def early_fn(day):
        e = day < pd.Timestamp(xtime.FIT_BEFORE)
        return (day <= day.sort_values().iloc[int(len(day) * 0.2)]) if e.sum() < 0.05 * len(day) else e

    b, nb = xtime.build(runs, figs, None, prefix="xt_", early_fn=early_fn, tune=tune, vacant_room=False)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert set(a.columns) - {"race_id", "dog_id"} == set(xtime.COLUMNS)
    assert na["tot"]["filter"] == nb["tot"]["filter"] and "tuned" not in na["tot"]


# ---- the five changes ----------------------------------------------------------------------------------------------

def test_a_forgetting_trap_effect_follows_a_trap_that_changes():
    """Romford's trap 1 quick (0.10 s) and trap 6 slow until day 50, then the other way round: by the last weeks the
    forgetting effect has turned, the cumulative mean still carries the first 50 days."""
    runs, _ = synth(days=110)
    d = (pd.to_datetime(runs.race_date) - pd.Timestamp("2025-01-01")).dt.days
    flip = (runs.track == "Romford") & (d >= 50)
    shift = np.where(flip & (runs.trap == 1), 0.20, np.where(flip & (runs.trap == 6), -0.20, 0.0))
    runs = runs.assign(adjusted_time=runs.adjusted_time + shift, run_time=runs.run_time + shift)
    P, y = _prepared(runs)
    df = core.per_run_figures(runs).sort_values(["t", "race_id", "trap"], kind="stable")
    late = ((pd.to_datetime(df.race_date) - pd.Timestamp("2025-01-01")).dt.days >= 95).to_numpy() & \
        (df.track == "Romford").to_numpy()
    t1 = late & (df.trap == 1).to_numpy()
    keep = xtime.run_filter(P, y, PRM, fit=False)
    fgt = xtime.run_filter(P, y, PRM, fit=False, forget={"trap": 10.0})
    # percent of the time, above 0 slower: trap 1 is now 0.10 s (0.4%) slow
    assert np.nanmean(fgt["trap"][t1]) > 0.2 and np.nanmean(fgt["trap"][t1]) > np.nanmean(keep["trap"][t1]) + 0.1


def test_the_constants_are_fitted_on_the_early_slice_and_noted(small):
    runs, _ = _gappy(days=90)
    tab, notes = xtime2.build(runs)
    for name in xtime.TARGETS:
        t = notes[name]["tuned"]
        assert set(t["constants"]) == {k for k, _, _ in xtime2.SEARCH} and t["pref_post"] is True
        for key, values, _ in xtime2.SEARCH:
            got = t["constants"][key]
            assert (float("inf") if got == "inf" else got) in values
            scores = {k: v for k, v in t["search"][key].items() if v is not None}
            assert scores[got] == min(scores.values())
        # fitted on the early days only: the last early day is before the frame's last fifth
        assert t["early_to_day"] < xtime.prepare(core.per_run_figures(runs))["dn"].max()


def test_the_preference_no_longer_takes_the_levels_residual_twice():
    """No dog has a preference here: the preference is noise, and smaller once it reads the residual the level's
    update left; the level carries more of the dog."""
    runs, skill = synth(days=120)
    P, y = _prepared(runs)
    late = P["dn"] > np.median(P["dn"])
    for kp in (5.0, 1.0):
        old = xtime.run_filter(P, y, PRM, fit=False, consts={"k_pref": kp})
        new = xtime.run_filter(P, y, PRM, fit=False, consts={"k_pref": kp}, pref_post=True)
        assert np.nanmean(np.abs(new["pref"][late])) < np.nanmean(np.abs(old["pref"][late]))
        assert np.nanstd(new["level"][late]) > np.nanstd(old["level"][late])


def test_p_fastest_matches_a_simulation():
    rng = np.random.default_rng(7)
    f = np.r_[rng.normal(0, 2, 6), rng.normal(0, 1, 5), [1.0, 1.0, 1.0]]
    sd = np.r_[rng.uniform(0.5, 3, 6), rng.uniform(0.5, 3, 5), [2.0, 2.0, 2.0]]
    race = np.r_[[1] * 6, [2] * 5, [3] * 3]
    p = xtime2.p_fastest(f, sd, race)
    for r in (1, 2):
        m = race == r
        sims = rng.normal(f[m], sd[m], size=(200_000, m.sum()))
        mc = np.bincount(sims.argmax(axis=1), minlength=m.sum()) / len(sims)
        assert np.allclose(p[m], mc, atol=0.006), (p[m], mc)
    assert np.allclose(p[race == 3], 1 / 3)
    # a missing expected time is the race's mean, a missing spread the race's median
    q = xtime2.p_fastest(np.r_[f[:5], np.nan], np.r_[sd[:5], np.nan], np.ones(6))
    assert np.isfinite(q).all() and abs(q.sum() - 1) < 1e-9


def test_a_vacant_box_is_room_not_nan(small):
    runs, _ = _gappy(days=70)
    figs = core.per_run_figures(runs)
    old, _ = xtime.build(runs, figs)
    new, _ = xtime2.build(runs, figs)
    m = runs[["race_id", "dog_id", "trap"]].merge(new, on=["race_id", "dog_id"]).merge(old, on=["race_id", "dog_id"])
    gap = m.race_id % 3 == 0
    t3, t5 = m[gap & (m.trap == 3)], m[gap & (m.trap == 5)]
    assert t3.xt_sec_vs_out.isna().all() and t5.xt_sec_vs_in.isna().all()
    ok = t3.xu_sec.notna()
    room = m[gap].groupby("race_id").xu_sec.transform("min")
    assert t3[ok].xu_sec_vs_out.notna().all()
    assert np.allclose(t3[ok].xu_sec_vs_out, (t3.xu_sec - room.loc[t3.index])[ok], atol=1e-5)
    assert (t5[t5.xu_sec.notna()].xu_sec_vs_in >= -1e-6).all()         # room: never behind the slowest
    # no box inside trap 1 or outside trap 6: still NaN
    assert m[m.trap == 1].xu_sec_vs_in.isna().all() and m[m.trap == 6].xu_sec_vs_out.isna().all()


# ---- the table -----------------------------------------------------------------------------------------------------

def test_every_column_is_registered_and_the_prefix_is_its_own(small):
    runs, _ = _gappy(days=60)
    tab, notes = xtime2.build(runs)
    assert set(tab.columns) - {"race_id", "dog_id"} == set(xtime2.COLUMNS) and len(tab) == len(runs)
    assert all(c.startswith(xtime2.PREFIX) for c in xtime2.COLUMNS) and xtime2.DEPENDS == ("xtime",)
    assert np.allclose(tab.groupby("race_id").xu_tot_pfast.sum(), 1.0, atol=1e-5)
    assert "pfast_mult" in notes and notes["fit_before"] == xtime2.FIT_BEFORE


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(small, change):
    runs, _ = _gappy(days=70)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = xtime2.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = xtime2.build(r2, core.per_run_figures(r2))[0]
    cols = list(xtime2.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).mean() > 0.9 and np.allclose(x, z, atol=1e-5, equal_nan=True)
