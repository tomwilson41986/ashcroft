"""greyhound/pit/debut.py: the prior for a dog with little race form. Planted data (unit tests only): debutants whose
trial time and sire set their price, and a mark-down the first time at a second track. The table finds each; a day's
own results and later days move nothing; a card row gets the run's features; one row a (race, dog)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, debut, trials
from tests.test_greyhound_pit import cardify_last_day, synth

L = core.LENGTH_S
START = pd.Timestamp("2025-01-01")
FIRST_DEBUT = 95                       # after the 92 days in which a first outing is not known to be a start
FT_COST = 1.2                          # the log-strength the market takes off a dog first time at Romford


def date(d):
    return (START + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")


def raw(days=250, seed=4):
    """Hove 500 m, three races a day; Romford 400 m, one. 24 old Hove dogs and 12 old Romford dogs (born 2021, racing
    from day 0: their starts are before the data). From day 95 a new dog debuts each day at Hove (ids 1000+): it
    trialled there 5 days before in 29.5 - 0.08 q seconds (q its quality in lengths), is by "Fast Sire" (even ids,
    +0.8 log strength) or "Slow Sire" (-0.8), runs every 3 days, 12 runs; its 5th run is at Romford, where the market
    takes FT_COST off it. The SP is the softmax of the strengths; the result follows the strengths with noise."""
    rng = np.random.default_rng(seed)
    old_h, old_r = np.arange(1, 25), np.arange(50, 62)
    skill = {int(d): rng.normal(0, 0.6) for d in np.concatenate([old_h, old_r])}
    q = {}
    rows, rid = [], [5000]

    def strength(dog):
        if dog < 1000:
            return skill[dog]
        return 0.25 * q[dog] + (0.8 if dog % 2 == 0 else -0.8)

    def race(d, hh, track, dist, cls, field, penalty=()):
        rid[0] += 1
        u = np.array([strength(x) - (FT_COST if x in penalty else 0.0) for x in field]) + rng.normal(0, 0.1, len(field))
        share = np.exp(u) / np.exp(u).sum()
        perf = u + rng.normal(0, 0.7, len(field))
        pos = np.empty(len(field), int)
        pos[np.argsort(-perf)] = np.arange(1, len(field) + 1)
        base = 29.5 if dist == 500.0 else 24.0
        for k, dog in enumerate(field):
            new = dog >= 1000
            t = round(base - L * perf[k] * 2.0, 2)
            rows.append({"race_date": date(d), "race_time": f"{hh}:00", "meeting_id": d * 10 + (track == "Romford"),
                         "track": track, "race_id": rid[0], "race_class": cls, "distance_m": dist, "trap": k + 1,
                         "dog_id": int(dog), "dog_name": f"Dog {dog}",
                         "sire": ("Fast Sire" if dog % 2 == 0 else "Slow Sire") if new else f"Old Sire {dog % 5}",
                         "dam": f"Dam {dog // 2}" if new else f"Old Dam {dog % 7}",
                         "born": "Feb-2024" if new else "Jan-2021", "sex": "b" if dog % 3 == 0 else "d",
                         "trainer": "T", "sp_decimal": float(1.0 / share[k]), "position": int(pos[k]),
                         "beaten_lengths": 0.5, "sectional": 4.5, "run_time": t, "adjusted_time": t,
                         "weight_kg": 30.0, "comment": "Mid"})

    for d in range(days):
        new_today = {}                                               # dog -> its run number today
        for dog, deb in ((1000 + i, FIRST_DEBUT + i) for i in range(max(0, d - FIRST_DEBUT + 1))):
            if (d - deb) % 3 == 0 and (d - deb) // 3 < 12:
                new_today[dog] = (d - deb) // 3
        if d + 5 >= FIRST_DEBUT:                                     # the dog debuting in 5 days trials today
            dog = 1000 + d + 5 - FIRST_DEBUT
            q[dog] = float(rng.normal(0, 3.0))
            rid[0] += 1
            t = round(29.5 - L * q[dog], 2)
            rows.append({"race_date": date(d), "race_time": "", "meeting_id": d * 10, "track": "Hove",
                         "race_id": rid[0], "race_class": "T1", "distance_m": 500.0, "trap": 1, "dog_id": dog,
                         "dog_name": f"Dog {dog}", "sire": "Fast Sire" if dog % 2 == 0 else "Slow Sire",
                         "dam": f"Dam {dog // 2}", "born": "Feb-2024", "sex": "b" if dog % 3 == 0 else "d",
                         "trainer": "T", "sp_decimal": np.nan, "position": 1, "beaten_lengths": 0.0,
                         "sectional": 4.5, "run_time": t, "adjusted_time": t, "weight_kg": 30.0, "comment": "Mid"})
        away = [x for x, k in new_today.items() if k == 4]
        home = [x for x, k in new_today.items() if k != 4]
        fill = list(rng.permutation(old_h))
        entries = home + fill[:18 - len(home)]
        entries = list(rng.permutation(entries))
        for r in range(3):
            race(d, 18 + r, "Hove", 500.0, "A3", entries[r * 6:(r + 1) * 6])
        rfield = away + list(rng.permutation(old_r))[:6 - len(away)]
        race(d, 21, "Romford", 400.0, "D2", rfield, penalty=set(away))
    return pd.DataFrame(rows), q


@pytest.fixture(scope="module")
def planted():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(debut, "MIN_FIT", 150)
        r, q = raw()
        runs = clean(r)
        tab, notes = debut.build(runs, None, {"trial_runs": trials.select(r)})
    m = runs.merge(tab, on=["race_id", "dog_id"], validate="one_to_one")
    m["day"] = (pd.to_datetime(m.race_date) - START).dt.days
    m["n"] = m.groupby("dog_id").cumcount()
    lp = np.log(m.sp_p_norm)
    m["lsp"] = lp - lp.groupby(m.race_id).transform("mean")
    m["q"] = m.dog_id.map(q)
    return r, runs, tab, m, notes


def build(r, runs=None):
    runs = clean(r) if runs is None else runs
    return runs, debut.build(runs, None, {"trial_runs": trials.select(r)})[0]


def test_every_column_is_registered_one_row_a_runner(planted):
    _, runs, tab, _, notes = planted
    assert list(tab.columns) == ["race_id", "dog_id"] + list(debut.COLUMNS)
    assert all(c.startswith(debut.PREFIX) for c in debut.COLUMNS) and all(debut.COLUMNS.values())
    assert all(tab[c].dtype == np.float32 for c in debut.COLUMNS)
    assert len(tab) == len(runs) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert notes["rows"] == len(runs) and notes["trial_rows"] == len(trials.select(planted[0]))


def test_no_duplicate_keys_when_the_runs_hold_trials_and_odd_indexes():
    runs = synth(days=20)                                            # T3 rows inside runs, as an old frame may hold
    odd = runs.sample(frac=1.0, random_state=1)                      # shuffled, a non-range index
    tab, _ = debut.build(odd)
    assert len(tab) == len(odd) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == odd.race_id.to_numpy()).all()
    back = odd[["race_id", "dog_id"]].merge(tab, on=["race_id", "dog_id"]).set_index(["race_id", "dog_id"])
    straight = debut.build(runs)[0].set_index(["race_id", "dog_id"]).reindex(back.index)
    assert np.allclose(back.to_numpy(float), straight.to_numpy(float), atol=1e-5, equal_nan=True)   # order moves nothing


def test_the_weight_fades_with_the_dogs_own_runs(planted):
    m = planted[3]
    new = m[m.dog_id >= 1000]
    assert np.allclose(new.db_w, 1.0 / (1.0 + new.n), atol=1e-6)                 # 1, 1/2, 1/3 ... from the debut
    old = m[m.dog_id < 1000]
    assert np.allclose(old.db_w, 1.0 / (1.0 + old.n + debut.CENSOR_ADD), atol=1e-6)   # a start before the data
    deb = new[new.n == 0]
    assert np.allclose(deb.db_age_debut, (pd.to_datetime(deb.race_date) - pd.Timestamp("2024-02-15")).dt.days)
    assert old.db_age_debut.isna().all()
    # the opposition's inexperience: the other runners' weights
    tot = m.groupby("race_id").db_w.transform("sum")
    assert np.allclose(m.db_field_inexp, tot - m.db_w, atol=1e-5)
    assert np.allclose(new.db_bitch_w, np.where(new.dog_id % 3 == 0, new.db_w, 0.0), atol=1e-6)


def test_the_planted_sire_is_found_on_a_debut(planted):
    m = planted[3]
    deb = m[(m.dog_id >= 1000) & (m.n == 0) & (m.day >= 150)]
    fast, slow = deb[deb.dog_id % 2 == 0], deb[deb.dog_id % 2 == 1]
    assert fast.db_sire_first_lsp_w.min() > 0.3 and slow.db_sire_first_lsp_w.max() < -0.3
    assert fast.db_sire_early_ae_w.mean() > slow.db_sire_early_ae_w.mean()
    assert fast.db_sire_early_fld_w.mean() > 0.02 > -0.02 > slow.db_sire_early_fld_w.mean()
    first = m[(m.dog_id == 1000)].iloc[0]                             # the first debutant: no debut before it
    assert first.db_sire_first_lsp_w == 0 and first.db_trk_deb_lsp_w == 0 and first.db_trap_deb_lsp_w == 0
    assert np.isnan(first.db_grade_seed_w) and np.isnan(first.db_prior_lsp)
    # the track's debutants so far against the market: the planted debutants average out near the field
    late = m[(m.dog_id >= 1000) & (m.n == 0) & (m.day >= 200)]
    assert np.allclose(late.db_grade_seed_w, 0.0, atol=1e-6)         # every debut here was in A3


def test_the_time_gap_reads_the_trial_against_the_field(planted):
    r, _, _, m, _ = planted
    deb = m[(m.dog_id >= 1000) & (m.n == 0) & (m.day >= 120)]
    assert deb.db_tt_gap.notna().all() and np.corrcoef(deb.db_tt_gap, deb.q)[0, 1] > 0.9
    assert np.allclose(deb.db_trial_gap_w, deb.db_tt_gap, atol=1e-5)  # w is 1 on a debut
    assert m[m.dog_id < 1000].db_trial_gap_w.isna().all()            # not for a dog whose start was not seen
    # one row by hand: the others' mean earlier time at Hove 500 m less the trial time, in lengths
    row = deb.iloc[10]
    hist = r[(r.race_date < row.race_date) & (r.track == "Hove") & (r.race_class != "T1")]
    others = m[(m.race_id == row.race_id) & (m.dog_id != row.dog_id)].dog_id
    means = hist[hist.dog_id.isin(others)].groupby("dog_id").adjusted_time.mean()
    own = r[(r.dog_id == row.dog_id) & (r.race_class == "T1")].adjusted_time.min()
    assert len(means) >= 2 and np.isclose(row.db_tt_gap, np.clip((means.mean() - own) / L, -40, 40), atol=1e-3)


def test_the_price_map_forecasts_a_debutants_price_from_earlier_months(planted):
    _, _, _, m, notes = planted
    assert notes["ridge_fits"] >= 2 and m[m.day < 120].db_prior_lsp.isna().all()      # no fit, no forecast
    assert m[m.dog_id < 1000].db_prior_lsp.isna().all()                               # only a start that was seen
    first_fit = (pd.Period(notes["ridge_first_fit"]).to_timestamp() - START).days
    new = m[(m.dog_id >= 1000) & (m.day >= first_fit) & (m.n < debut.MAP_RUNS)]
    assert new.db_prior_lsp.notna().all()
    assert m[(m.dog_id >= 1000) & (m.n >= debut.MAP_RUNS)].db_prior_lsp.isna().all()
    deb = new[new.n == 0]
    assert len(deb) > 60 and np.corrcoef(deb.db_prior_lsp, deb.lsp)[0, 1] > 0.8
    assert deb.db_prior_gap.isna().all() and new[new.n > 0].db_prior_gap.notna().all()


def test_the_first_time_mark_down_is_found(planted):
    m = planted[3]
    ft = m[(m.dog_id >= 1000) & (m.track == "Romford")]
    assert (ft.n == 4).all()
    late = ft[ft.day >= 200]
    assert len(late) > 20 and late.db_ft_track.max() < -0.4          # the market marked the earlier ones down
    assert late.db_ft_trip.max() < -0.4 and late.db_ft_fam.max() < -0.4   # the same run: first 400 m, first D grade
    rest = m[~((m.dog_id >= 1000) & (m.n.isin([4, 5])))]
    assert (rest.db_ft_track == 0).all() and (rest.db_ft_fam == 0).all()


def test_a_days_results_and_later_days_move_nothing(planted, monkeypatch):
    monkeypatch.setattr(debut, "MIN_FIT", 150)
    r, runs, tab, m, _ = planted
    day = date(215)                                                   # inside a fitted month, 35 days from the end
    cols = list(debut.COLUMNS)
    base = runs[["race_id", "dog_id", "race_date"]].merge(tab, on=["race_id", "dog_id"])
    # (a) the day's own results rewritten: positions reversed, times, SPs
    p = r.copy()
    on = (p.race_date == day) & (p.race_class != "T1")
    assert on.sum() == 24
    p.loc[on, "position"] = 7 - p.loc[on, "position"]
    p.loc[on, ["adjusted_time", "run_time"]] += 0.6
    p.loc[on, "sp_decimal"] = np.where(p.loc[on, "trap"] == 1, 1.5, 12.0)
    runs_p, tab_p = build(p)
    a = base[base.race_date == day].set_index(["race_id", "dog_id"])[cols]
    b = runs_p[["race_id", "dog_id", "race_date"]].merge(tab_p, on=["race_id", "dog_id"]) \
        .query("race_date == @day").set_index(["race_id", "dog_id"])[cols].reindex(a.index)
    assert len(a) == 24 and np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-6, equal_nan=True)
    later = base[base.race_date > day].set_index(["race_id", "dog_id"])[cols]
    after = runs_p[["race_id", "dog_id"]].merge(tab_p, on=["race_id", "dog_id"]).set_index(["race_id", "dog_id"])[cols]
    assert not np.allclose(later.to_numpy(float), after.reindex(later.index).to_numpy(float), atol=1e-6, equal_nan=True)
    # (b) the later days (races and trials) taken away: every day up to it is as it was
    runs_t, tab_t = build(r[r.race_date <= day])
    a = base[base.race_date <= day].set_index(["race_id", "dog_id"])[cols]
    b = tab_t.set_index(["race_id", "dog_id"])[cols].reindex(a.index)
    assert len(tab_t) == len(a) and np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-6, equal_nan=True)


def test_a_card_row_gets_the_features_its_run_had(planted, monkeypatch):
    monkeypatch.setattr(debut, "MIN_FIT", 150)
    r, runs, tab, _, _ = planted
    card = cardify_last_day(runs)
    last = runs.race_date.max()
    is_card = card.is_card.to_numpy(bool)
    assert is_card.sum() == 24 and card.loc[is_card, "sp_p_norm"].isna().all() and card.loc[is_card, "position"].isna().all()
    tab_c, notes = debut.build(card, None, {"trial_runs": trials.select(r)})
    assert notes["card_rows"] == 24 and len(tab_c) == len(card)
    cols = list(debut.COLUMNS)
    keys = runs.loc[runs.race_date == last, ["race_id", "dog_id"]]
    a = keys.merge(tab, on=["race_id", "dog_id"])[cols].to_numpy(float)
    b = keys.merge(tab_c, on=["race_id", "dog_id"])[cols].to_numpy(float)
    assert len(b) == 24 and np.allclose(a, b, atol=1e-6, equal_nan=True)
    c = keys.merge(tab_c, on=["race_id", "dog_id"])
    assert c.db_w.notna().all() and c.db_field_inexp.notna().all() and c.db_tt_gap.notna().any()
    deb = c[c.dog_id >= 1000]
    assert deb.db_prior_lsp.notna().any() and deb.db_sire_first_lsp_w.abs().max() > 0.05


def test_the_cold_start_is_finite(planted):
    tab = planted[2]
    assert np.isfinite(tab[list(debut.COLUMNS)].to_numpy(float)[~tab[list(debut.COLUMNS)].isna().to_numpy()]).all()
