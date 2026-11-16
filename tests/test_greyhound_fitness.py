"""greyhound/pit/fitness.py: the fitness-fatigue impulse response over the whole load history. The states are the
decayed dose sums over earlier days, trials included, a day's outings added; a planted response (slower the day
after a hard week, faster on a long steady base) is recovered with its sign and rough size once fitted; a period's
fit never reads its own rows, nor the record's first days (their states are still filling); the day's own results
move nothing, on the last day or a middle one; later days, races or trials, move nothing before them; a card row
gets what its run had; a dog with no history, same-day rows, short fields and a record with a silence of years are
read as they should be."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from greyhound import pit
from greyhound.data import clean
from greyhound.pit import core, dyad, trials, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

load = importlib.import_module("greyhound.pit.fitness")
L = core.LENGTH_S
A, B = 1.2, 0.25                                            # the plant: lengths a unit of the 10-day and 45-day states


def row(day, rt, rid, cls, dog, trap, t, pos, dist=480.0, comment="QAw", **extra):
    return {"race_date": day, "race_time": rt, "meeting_id": 1, "track": "Hove", "race_id": rid, "race_class": cls,
            "distance_m": dist, "trap": trap, "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023",
            "sex": "d", "sp_decimal": 6.0 if rt else np.nan, "position": pos, "beaten_lengths": 0.5, "sectional": 4.5,
            "run_time": t, "adjusted_time": t, "weight_kg": 30.0, "comment": comment, **extra}


def raw(days=200, seed=4, dogs=36):
    """One track, 480 m, 36 dogs of equal ability. Each dog is in a busy spell (it races on about half the days) or a
    quiet one (about one day in twelve), changing every fortnight or so; on a quiet day it sometimes trials (a T1 row
    with no off time, as GBGB serves it). Its time is 29.50 plus ``A`` lengths a unit of its true 10-day state less
    ``B`` lengths a unit of its true 45-day state (every earlier outing, trials too) plus noise: slower the day after
    a hard week, faster on a long steady base. Fields of two to six. The true states ride along (true_10, true_45)."""
    rng = np.random.default_rng(seed)
    f10, f45 = np.zeros(dogs + 1), np.zeros(dogs + 1)
    busy = rng.random(dogs + 1) < 0.5
    rows, rid = [], 9000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        f10 *= np.exp(-1 / 10)
        f45 *= np.exp(-1 / 45)
        busy = np.where(rng.random(dogs + 1) < 1 / 14, ~busy, busy)
        u = rng.random(dogs + 1)
        go = [int(k) for k in rng.permutation(np.flatnonzero(u < np.where(busy, 0.5, 0.08))) if k >= 1]
        trial = [int(k) for k in np.flatnonzero(~busy & (u > 0.97)) if k >= 1]
        out, r = list(trial), 0
        while len(go) >= 2:
            field, go = go[:6], go[6:]
            rid += 1
            r += 1
            t = 29.5 + L * (A * f10[field] - B * f45[field]) + rng.normal(0, 0.04, len(field))
            pos = np.argsort(np.argsort(t)) + 1
            for k, dog in enumerate(field):
                rows.append(row(day, f"{12 + r // 4}:{15 * (r % 4):02d}", rid, "A3", dog, k + 1, float(t[k]),
                                int(pos[k]), true_10=f10[dog], true_45=f45[dog]))
            out += field
        for dog in trial:
            rid += 1
            rows.append(row(day, "", rid, "T1", dog, 1, 29.6, 1, comment="Mid"))
        f10[out] += 1
        f45[out] += 1
    return pd.DataFrame(rows)


def flat(runs):
    """An expected-time table that says 29.50 for every run: the residual is then the plant plus noise."""
    return runs[["race_id", "dog_id"]].assign(xt_tot_time=29.5)


def resid(runs, xt):
    j = runs[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_tot_time"]], on=["race_id", "dog_id"], how="left")
    return dyad.residuals(runs, pd.Series(j.xt_tot_time.to_numpy(), index=runs.index)).to_numpy(float)


@pytest.fixture()
def small(monkeypatch):
    """Fits a small record can cross. ``raw`` starts with every dog's first outing (no history before the record),
    so the burn-in is off here; the leak tests and the burn-in's own test turn it on."""
    monkeypatch.setattr(load, "MIN_FIT", 500)
    monkeypatch.setattr(load, "REFIT_MONTHS", 1)
    monkeypatch.setattr(load, "BURN_DAYS", 0)


@pytest.fixture()
def built(small):
    r = raw()
    tr, runs = trials.select(r), clean(r)
    xt = flat(runs)
    tab, notes = load.build(runs, core.per_run_figures(runs), {"xtime": xt, "trial_runs": tr})
    m = runs.merge(tab, on=["race_id", "dog_id"])
    return runs, m.assign(r=resid(m, xt)), notes, tr


def test_every_column_is_registered(built):
    runs, m, notes, tr = built
    assert {c for c in m.columns if c.startswith(load.PREFIX)} == set(load.COLUMNS) and len(m) == len(runs)
    assert len(load.STATES) == 13 and set(load.STATES) < set(load.COLUMNS) and len(load.COLUMNS) == 16
    assert not trials.is_trial_class(runs.race_class).any() and len(tr) > 50          # the trials come by the context
    assert notes["trial_outings"] == len(tr) and notes["race_outings"] == len(runs) and notes["card_rows"] == 0
    assert set(runs.groupby("race_id").size()) == {2, 3, 4, 5, 6}                    # short fields are built too


def test_the_states_are_the_decayed_sums_of_every_earlier_outing(built):
    _, m, _, _ = built
    assert np.allclose(m.ff_out_10, m.true_10, atol=1e-4) and np.allclose(m.ff_out_45, m.true_45, atol=1e-4)
    assert np.allclose(m.ff_m_10, m.ff_out_10, atol=1e-5)                              # every outing here is 480 m
    assert np.allclose(m.ff_tsb, m.true_45 / 45 - m.true_10 / 10, atol=1e-5)
    assert np.allclose(m.ff_mono, m.true_10 / (m.true_45 + 1), atol=1e-5)
    assert (m.ff_trial_21 <= m.ff_out_21 + 1e-6).all() and m.ff_trial_90.max() > 0.5


def test_the_planted_response_is_recovered_with_its_sign_and_size(built):
    _, m, notes, _ = built
    assert m[m.race_date < "2025-02-01"].ff_pred.isna().all()                         # none before the first fit
    k = m[m.ff_pred.notna() & m.r.notna()]
    assert len(k) > 1000 and k.race_date.min() > "2025-02-01"
    assert np.corrcoef(k.ff_pred, k.r)[0, 1] > 0.8                                    # out of sample by construction
    # the response the fit learnt, read off its predictions: about A a unit of the 10-day state, -B of the 45-day
    X = np.c_[np.ones(len(k)), k.ff_out_10, k.ff_out_45]
    _, a, b = np.linalg.lstsq(X, k.ff_pred.to_numpy(float), rcond=None)[0]
    assert 0.8 < a < 1.6 and -0.4 < b < -0.12
    # the day after a hard week it is slower; among the dogs with the quietest ten days, the half with the longer base
    # (about 3.6 more of the 45-day state: -0.9 lengths, less the 0.5 its 0.4 more of the 10-day state costs) is faster
    hard, easy = k[k.ff_out_10 >= k.ff_out_10.quantile(0.75)], k[k.ff_out_10 <= k.ff_out_10.quantile(0.25)]
    assert hard.ff_pred.mean() > easy.ff_pred.mean() + 1.0
    base, none = easy[easy.ff_out_45 >= easy.ff_out_45.median()], easy[easy.ff_out_45 < easy.ff_out_45.median()]
    assert base.ff_pred.mean() < none.ff_pred.mean() - 0.2
    assert np.corrcoef(k.ff_tsb, k.ff_pred)[0, 1] < -0.8
    # the notes carry each fit: its rows, and a coefficient a state
    assert list(notes["fits"])[0] >= "2025-02" and all(set(f["lengths_a_sd"]) == set(load.STATES) for f in notes["fits"].values())
    rows = [f["rows"] for f in notes["fits"].values()]
    assert rows[0] >= 500 and rows == sorted(rows)
    assert notes["read_by_year"][2025]["runs"] == len(k)
    assert notes["read_by_year"][2025]["rmse_less_the_response"] < notes["read_by_year"][2025]["rmse_lengths"]


def test_the_sign_holds_against_the_expected_time_table_and_it_is_taken_from_the_context(small):
    from greyhound.pit import xtime
    r = raw()
    tr, runs = trials.select(r), clean(r)
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = load.build(runs, figs, {"xtime": xt, "trial_runs": tr})[0]
    b = load.build(runs, figs, {"trial_runs": tr})[0]
    cols = list(load.COLUMNS)
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)
    m = runs.merge(a, on=["race_id", "dog_id"])
    m["r"] = resid(m, xt)
    k = m[m.ff_pred.notna() & m.r.notna()]
    # the dog's level takes some of it: the sign and the order of size stay
    assert np.corrcoef(k.ff_pred, k.r)[0, 1] > 0.5
    _, c10, c45 = np.linalg.lstsq(np.c_[np.ones(len(k)), k.ff_out_10, k.ff_out_45], k.ff_pred.to_numpy(float), rcond=None)[0]
    assert 0.3 < c10 < 1.6 and -0.4 < c45 < -0.05


def test_a_fit_reads_the_periods_before_it_only(monkeypatch):
    monkeypatch.setattr(load, "MIN_FIT", 150)
    rng = np.random.default_rng(0)
    k = len(load.STATES)
    X = rng.normal(0, 1, (600, k)) * np.arange(1, k + 1)
    X[:, -1] = 0.0                                                   # a state that never moved (no trials): reads 0
    beta = rng.normal(0, 1, k)
    y = 0.7 + X @ beta + rng.normal(0, 0.1, 600)
    y[::17] = np.nan                                                 # runs with no residual are in no fit
    period = np.repeat([2023, 2024, 2025, 2026], [100, 200, 200, 100])
    pred, fits = load.fit_predict(X, y, period)
    assert np.isnan(pred[:300]).all() and np.isfinite(pred[300:]).all()        # 100 rows, then 300: the first fit
    assert list(fits) == ["2025", "2026"] and fits["2025"]["rows"] == int(np.isfinite(y[:300]).sum())
    assert fits["2025"]["lengths_a_sd"][load.STATES[-1]] == 0.0
    # the ridge, written out: the fit rows' mean and sd, lambda on the standardised columns, a free intercept
    f = np.flatnonzero(np.isfinite(y[:300]))
    mu, sd = X[f].mean(axis=0), X[f].std(axis=0)
    Z = np.divide(X - mu, sd, out=np.zeros_like(X), where=sd > 0)
    b = np.linalg.solve(Z[f].T @ Z[f] + load.LAMBDA * np.eye(k), Z[f].T @ (y[f] - y[f].mean()))
    assert np.allclose(pred[300:500], y[f].mean() + Z[300:500] @ b, atol=1e-8)
    assert np.corrcoef(pred[500:], X[500:] @ beta)[0, 1] > 0.99
    y2 = y.copy()
    y2[500:] += 50.0                                                 # the last period's own results
    assert np.allclose(load.fit_predict(X, y2, period)[0], pred, equal_nan=True)
    y2[300:500] += 50.0                                              # an earlier period's do move the next
    p3 = load.fit_predict(X, y2, period)[0]
    assert np.allclose(p3[300:500], pred[300:500]) and not np.allclose(p3[500:], pred[500:])


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change, small, monkeypatch):
    monkeypatch.setattr(load, "BURN_DAYS", 30)                       # the burn-in is on the path
    r = raw(days=120)
    tr, runs = trials.select(r), clean(r)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = load.build(runs, core.per_run_figures(runs), {"trial_runs": tr})[0]
    r2 = change(runs)
    b = load.build(r2, core.per_run_figures(r2), {"trial_runs": tr})[0]
    cols = list(load.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert len(z) == len(keys) and len(a) == len(runs) and len(b) == len(r2)
    assert np.isfinite(x).mean() > 0.8 and np.isfinite(x[:, cols.index("ff_pred")]).all()
    assert np.allclose(x, z, atol=1e-4, equal_nan=True)


def perturb_day(runs, d):
    """Day ``d``'s results rewritten as ``perturb_last_day`` rewrites the last day's: the finishing order reversed,
    the times, comments (now with interference in them) and SPs changed."""
    r = runs.copy()
    k = r.race_date == d
    r.loc[k, "position"] = r.loc[k, "field"] + 1 - r.loc[k, "position"]
    r.loc[k, "adjusted_time"] += 0.5
    r.loc[k, "run_time"] += 0.5
    r.loc[k, "sectional"] += 0.2
    r.loc[k, "comment"] = "SAw,Bmp1,Crd2,Wide"
    r.loc[k, "sp_decimal"] = 9.0
    r["won"] = (r.position == 1).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


@pytest.fixture()
def whole(small, monkeypatch):
    """150 days built end to end (the expected times built inside, the trials by the context), the burn-in on, a fit
    a month from March. The expected-time table fits its filter's constants on the days before ``FIT_BEFORE``, or on
    the first fifth of a record with none: held to the first 19 days here, so that what is tested is this table."""
    monkeypatch.setattr(load, "BURN_DAYS", 20)
    monkeypatch.setattr(load, "MIN_FIT", 200)
    monkeypatch.setattr(xtime, "FIT_BEFORE", "2025-01-20")
    r = raw(days=150)
    tr, runs = trials.select(r), clean(r)
    tab = load.build(runs, core.per_run_figures(runs), {"trial_runs": tr})[0]
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    return runs, tr, tab[list(load.COLUMNS)].to_numpy(float)


# a day in the middle of a refit period, the last day of one, the first day of the next
DAYS = ["2025-03-12", "2025-03-31", "2025-04-01"]


@pytest.mark.parametrize("d", DAYS)
def test_a_middle_days_results_move_nothing_on_that_day_or_before(d, whole):
    runs, tr, a = whole
    r2 = perturb_day(runs, d)
    b = load.build(r2, core.per_run_figures(r2), {"trial_runs": tr})[0][list(load.COLUMNS)].to_numpy(float)
    upto, cols = (runs.race_date <= d).to_numpy(), list(load.COLUMNS)
    assert (runs.race_date == d).sum() > 5 and np.isfinite(a[(runs.race_date == d).to_numpy(), cols.index("ff_pred")]).all()
    assert np.array_equal(a[upto], b[upto], equal_nan=True)          # the day itself and every day before: not a bit
    # the change is real: the day's bumps are in the states of the days after, and its times in the fits after
    moved = np.abs(a[~upto] - b[~upto]).max(axis=0)
    assert moved[cols.index("ff_hit_10")] > 0.5 and moved[cols.index("ff_pred")] > 1e-3
    assert moved[cols.index("ff_out_10")] == 0                       # an outing is an outing whatever its result


@pytest.mark.parametrize("d", DAYS)
def test_later_days_move_nothing(d, whole):
    """The record cut after day ``d`` gives day ``d`` and every day before exactly what the whole record gives
    them: no state, standardisation or fit reads a later day, race or trial."""
    runs, tr, a = whole
    upto = (runs.race_date <= d).to_numpy()
    cut = runs[upto].reset_index(drop=True)
    figs = core.per_run_figures(cut)
    for trial_runs in (tr, tr[tr.race_date <= d], tr[tr.race_date < d]):       # the later trials, and the day's own
        b = load.build(cut, figs, {"trial_runs": trial_runs})[0]
        assert (b.race_id.to_numpy() == cut.race_id.to_numpy()).all()
        assert np.array_equal(a[upto], b[list(load.COLUMNS)].to_numpy(float), equal_nan=True)


def test_a_trial_on_the_day_or_later_moves_nothing_on_that_day_or_before(whole):
    runs, tr, a = whole
    d = DAYS[0]
    t2 = tr.copy()
    late = t2.race_date >= d
    t2.loc[late, "comment"] = "Bmp1,Crd2"
    t2.loc[late, "distance_m"] = 900.0
    dogs = np.sort(runs.dog_id.unique())
    extra = pd.DataFrame({"race_date": d, "track": "Hove", "distance_m": 480.0, "race_id": 10 ** 7 + np.arange(len(dogs)),
                          "trap": 1, "dog_id": dogs, "comment": "Bmp1"})           # every dog trialled on the day
    b = load.build(runs, core.per_run_figures(runs), {"trial_runs": pd.concat([t2, extra], ignore_index=True)})[0]
    b = b[list(load.COLUMNS)].to_numpy(float)
    upto, cols = (runs.race_date <= d).to_numpy(), list(load.COLUMNS)
    assert late.sum() > 10 and np.array_equal(a[upto], b[upto], equal_nan=True)
    moved = np.abs(a[~upto] - b[~upto]).max(axis=0)
    assert moved[cols.index("ff_trial_21")] > 0.5 and moved[cols.index("ff_out_10")] > 0.5


def test_the_first_days_of_a_record_join_no_fit(small, monkeypatch):
    """A record that starts 60 days into every dog's career: its first states are short of the outings before it, and
    no fit reads a run of the burn-in. The states themselves, and what a fit is read on, are as they were."""
    r = raw()
    start = "2025-03-02"
    r = r[r.race_date >= start]
    tr, runs = trials.select(r), clean(r)
    figs, xt = core.per_run_figures(runs), flat(runs)
    since = (pd.to_datetime(runs.race_date) - pd.Timestamp(start)).dt.days
    res = np.isfinite(resid(runs, xt))
    monkeypatch.setattr(load, "MIN_FIT", 300)
    a, na = load.build(runs, figs, {"xtime": xt, "trial_runs": tr})
    monkeypatch.setattr(load, "BURN_DAYS", 30)
    b, nb = load.build(runs, figs, {"xtime": xt, "trial_runs": tr})
    assert np.array_equal(a[load.STATES].to_numpy(), b[load.STATES].to_numpy())
    # the truncation the burn-in is for: the cut record's 45-day state against the true one (every outing)
    short = (b.ff_out_45 / runs.true_45.where(runs.true_45 > 0)).groupby(since // 30).mean()
    assert short[0] < 0.5 and short[1] < 0.9 and short[4] > 0.93 and (b.ff_out_45 <= runs.true_45 + 1e-4).all()
    assert nb["first_race_day"] == start and nb["burn_days"] == 30 and na["residual_rows_in_the_burn_in"] == 0
    assert nb["residual_rows_in_the_burn_in"] == int((res & (since < 30)).sum()) > 200
    # every fit's rows: the runs with a residual from the end of the burn-in to the start of its period
    for burn, notes in ((0, na), (30, nb)):
        assert len(notes["fits"]) >= 2
        for label, f in notes["fits"].items():
            assert f["rows"] == int((res & (since >= burn) & (runs.race_date < label + "-01")).sum()) >= 300
    assert list(nb["fits"])[0] > list(na["fits"])[0] and set(nb["fits"]) < set(na["fits"])
    assert b.ff_pred[(since < 30).to_numpy()].isna().all() and b.ff_pred.notna().sum() > 300
    both = a.ff_pred.notna() & b.ff_pred.notna()
    assert both.sum() == b.ff_pred.notna().sum() and not np.allclose(a.ff_pred[both], b.ff_pred[both])


def hand():
    """Dog 1: a race on day 0 (480 m, crowded and bumped), a trial on day 3 (240 m, bumped), two races on day 5 (640 m
    and 480 m), a race and, the same day, a trial on day 9, a race on day 12. Dog 2 runs beside it; dog 3 has its
    first race on day 12."""
    day = lambda d: (pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")  # noqa: E731
    rows = [row(day(0), "18:00", 1, "A3", 1, 1, 29.5, 1, comment="Crd1,Bmp2,Mid"), row(day(0), "18:00", 1, "A3", 2, 2, 29.6, 2),
            row(day(3), "", 2, "T1", 1, 1, 15.0, 1, dist=240.0, comment="Bmp1"),
            row(day(5), "14:00", 3, "A3", 1, 1, 40.1, 1, dist=640.0), row(day(5), "14:00", 3, "A3", 2, 2, 40.3, 2, dist=640.0),
            row(day(5), "21:00", 4, "A3", 1, 1, 29.5, 1), row(day(5), "21:00", 4, "A3", 2, 2, 29.6, 2),
            row(day(9), "18:00", 5, "A3", 1, 1, 29.5, 1), row(day(9), "18:00", 5, "A3", 2, 2, 29.6, 2),
            row(day(9), "", 6, "T1", 1, 1, 29.7, 1),
            row(day(12), "18:00", 7, "A3", 1, 1, 29.5, 1), row(day(12), "18:00", 7, "A3", 2, 2, 29.6, 2),
            row(day(12), "18:00", 7, "A3", 3, 3, 29.7, 3)]
    r = pd.DataFrame(rows)
    return trials.select(r), clean(r)


def test_doses_same_day_rows_a_trial_on_the_day_and_a_dog_with_no_history():
    tr, runs = hand()
    tab = load.build(runs, core.per_run_figures(runs), {"xtime": flat(runs), "trial_runs": tr})[0]
    m = runs.merge(tab, on=["race_id", "dog_id"])
    one = m[m.dog_id == 1].set_index("race_id")
    e = np.exp
    assert (one.loc[1, load.STATES] == 0).all() and one.loc[1, "ff_tsb"] == 0 and one.loc[1, "ff_mono"] == 0
    # both of day 5's rows read days 0 and 3, and neither reads the other
    for rid in (3, 4):
        assert np.isclose(one.loc[rid, "ff_out_4"], e(-5 / 4) + e(-2 / 4), atol=1e-5)
        assert np.isclose(one.loc[rid, "ff_m_10"], e(-5 / 10) + 0.5 * e(-2 / 10), atol=1e-5)
        assert np.isclose(one.loc[rid, "ff_hit_10"], 2 * e(-5 / 10) + e(-2 / 10), atol=1e-5)
        assert np.isclose(one.loc[rid, "ff_trial_21"], e(-2 / 21), atol=1e-5)
    # day 9: the day's two outings of day 5 added; the trial of day 9 itself is not read
    assert np.isclose(one.loc[5, "ff_out_10"], e(-0.9) + e(-0.6) + 2 * e(-0.4), atol=1e-5)
    assert np.isclose(one.loc[5, "ff_m_45"], e(-9 / 45) + 0.5 * e(-6 / 45) + (640 / 480 + 1) * e(-4 / 45), atol=1e-5)
    assert np.isclose(one.loc[5, "ff_trial_90"], e(-6 / 90), atol=1e-5)
    # day 12: it is, with the race of day 9
    assert np.isclose(one.loc[7, "ff_trial_21"], e(-9 / 21) + e(-3 / 21), atol=1e-5)
    assert np.isclose(one.loc[7, "ff_out_180"], e(-12 / 180) + e(-9 / 180) + 2 * e(-7 / 180) + 2 * e(-3 / 180), atol=1e-5)
    assert np.isclose(one.loc[7, "ff_hit_45"], 2 * e(-12 / 45) + e(-9 / 45), atol=1e-5)
    assert np.isclose(one.loc[7, "ff_tsb"], one.loc[7, "ff_out_45"] / 45 - one.loc[7, "ff_out_10"] / 10, atol=1e-6)
    # a dog with no history: no load, and no fitted response on so short a record
    three = m[m.dog_id == 3].iloc[0]
    assert (three[load.STATES] == 0).all() and three.ff_mono == 0 and m.ff_pred.isna().all()
    # without the trial channel the trials are not there
    bare = runs.merge(load.build(runs, core.per_run_figures(runs), {"xtime": flat(runs)})[0], on=["race_id", "dog_id"])
    assert (bare.ff_trial_21 == 0).all() and bare.ff_out_180.max() < m.ff_out_180.max()


def test_a_card_row_is_no_outing_and_a_trial_row_in_the_runs_is_counted_once():
    tr, runs = hand()
    card = cardify_last_day(runs)
    frame, notes = load.outings(core.per_run_figures(card), tr)
    assert notes["card_rows"] == 3 and (frame.w.to_numpy()[:len(card)][card.is_card.to_numpy()] == 0).all()
    assert notes["race_outings"] == len(runs) - 3 and notes["trial_outings"] == 2
    # a T-class row that is in the runs (it has an off time) and in the trial channel as well: one outing, a trial
    both = pd.concat([runs, tr.assign(race_time="10:00", meeting_id=1, sp_decimal=np.nan, beaten_lengths=0.0)],
                     ignore_index=True)
    both = clean(both.drop(columns=["t", "won", "placed2", "sp_p", "sp_p_norm", "field"]))
    frame, notes = load.outings(core.per_run_figures(both), tr)
    assert notes["trial_rows_in_runs"] == 2 and notes["trial_outings"] == 0 and len(frame) == len(both)
    assert frame.trial.sum() == 2


def test_a_silence_of_years_does_not_break_the_sums():
    day = lambda d: pd.Timestamp("2015-01-01") + pd.Timedelta(days=d)  # noqa: E731
    frame = pd.DataFrame({"dog_id": [1, 1, 1, 2], "race_date": [day(0), day(3000), day(3004), day(3004)], "w": 1.0,
                          "m": 1.0, "hit": 0.0, "trial": 0.0})
    st = load.states(frame)
    assert np.isfinite(st.to_numpy()).all()
    assert np.isclose(st.ff_out_4[2], np.exp(-1.0)) and np.isclose(st.ff_out_180[2], np.exp(-3004 / 180) + np.exp(-4 / 180))
    assert st.ff_out_4[1] == 0 and (st.loc[3] == 0).all()
