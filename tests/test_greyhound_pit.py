"""greyhound/pit: the metrics database's tables are point-in-time (the last day's results move none of its rows), and
each table's mechanics (the Kalman spread, the learnt trouble prices, the identity match, the windows)."""
import numpy as np
import pandas as pd
import pytest

from greyhound import pit
from greyhound.data import clean
from greyhound.pit import ability, core, ids, life, trouble

TRACKS = ("Romford", "Hove")


def synth(days=40, seed=0, trouble_cost=3.0):
    """A synthetic card: 2 tracks a day, 6 races of 6 from 80 dogs; each dog an ability; 'Crd1' costs lengths."""
    rng = np.random.default_rng(seed)
    dogs = np.arange(1, 81)
    skill = rng.normal(0, 1.5, len(dogs) + 1)
    rows, rid = [], 1000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for ti, track in enumerate(TRACKS):
            for r in range(6):
                rid += 1
                field = rng.choice(dogs, 6, replace=False)
                crd = rng.random(6) < 0.25
                perf = skill[field] + rng.normal(0, 1.0, 6) - np.where(crd, trouble_cost, 0.0)
                time_ = 29.0 - perf * core.LENGTH_S
                order = np.argsort(-perf)
                pos = np.empty(6, int)
                pos[order] = np.arange(1, 7)
                sp = np.exp(-skill[field])
                sp = 1 + 5 * sp / sp.sum() * 6
                trial = (r == 0 and d % 7 == 0)
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{18 + r}:{10 * ti:02d}", "meeting_id": d * 10 + ti,
                                 "track": track, "race_id": rid, "race_class": "T3" if trial else f"A{1 + dog % 5}",
                                 "distance_m": 480.0, "trap": k + 1, "dog_id": int(dog), "dog_name": f"Dog {dog}",
                                 "sire": f"Sire {dog % 7}", "dam": f"Dam {dog % 11}", "born": "Jan-2023",
                                 "sex": "b" if dog % 2 else "d", "season": "01.Ja.25" if dog % 2 else None,
                                 "trainer": f"Trainer {dog % 5}", "sp_decimal": float(sp[k]),
                                 "market_pos": int(np.argsort(np.argsort(sp))[k]) + 1, "position": int(pos[k]),
                                 "beaten_lengths": 0.5, "sectional": 4.8 - skill[dog] * 0.02,
                                 "run_time": float(time_[k]), "adjusted_time": float(time_[k]),
                                 "weight_kg": 30.0 + dog % 5, "comment": "Crd1,Mid" if crd[k] else "QAw,Rls"})
    return clean(pd.DataFrame(rows))


def perturb_last_day(runs):
    """The last day's results rewritten: positions reversed, times, comments, SPs and weights changed."""
    r = runs.copy()
    last = r.race_date == r.race_date.max()
    r.loc[last, "position"] = 7 - r.loc[last, "position"]
    r.loc[last, "adjusted_time"] += 0.5
    r.loc[last, "run_time"] += 0.5
    r.loc[last, "sectional"] += 0.2
    r.loc[last, "comment"] = "SAw,Bmp1,Wide"
    r.loc[last, "sp_decimal"] = 9.0
    r["won"] = (r.position == 1).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


@pytest.mark.parametrize("name", ["ability", "trouble", "connections", "market", "breeding", "life"])
def test_no_table_reads_the_days_own_results(name, monkeypatch):
    monkeypatch.setattr(trouble, "MIN_FIT", 100)
    monkeypatch.setattr(ability, "FIT_BEFORE", "2025-01-15")
    runs = synth()
    last = runs.race_date.max()
    keys = runs.loc[runs.race_date == last, ["race_id", "dog_id"]]
    a = pit.build_tables([name], runs)[name]
    ctx = {"fetch_lag": pd.Series(0, index=sorted(runs.race_date.unique()))}      # read at the time: trainers known
    tab_a = pit.module(name).build(runs, core.per_run_figures(runs), dict(ctx))[0]
    tab_b = pit.module(name).build(perturb_last_day(runs), core.per_run_figures(perturb_last_day(runs)), dict(ctx))[0]
    if name == "connections":
        assert tab_a.cx_tr_ae365.notna().mean() > 0.9                             # the trainer columns are tested
    cols = [c for c in tab_a.columns if c.startswith(pit.module(name).PREFIX)
            and c not in getattr(pit.module(name), "CARD_UNSAFE", ())]
    x = keys.merge(tab_a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    y = keys.merge(tab_b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.allclose(x, y, equal_nan=True), name
    assert a["rows"] == len(runs)


def test_the_kalman_spread_narrows_with_runs_and_widens_with_time_off():
    prm = {"q": 0.1, "r": 4.0, "m0": 0.0, "p0": 25.0}
    dog = np.array([1, 1, 1, 1, 2])
    dayn = np.array([0, 7, 14, 100, 0])
    obs = np.array([5.0, 5.0, 5.0, 5.0, np.nan])
    m, v, c, since = ability.filter_dog_states(dog, dayn, obs, np.zeros(5, bool), prm)
    assert v[0] == 25.0 and v[1] < v[0] and v[2] < v[1] and v[3] > v[2]       # 86 days off widen it
    assert m[0] == 0.0 and 0 < m[1] < m[2] < 5.0 and list(c) == [0, 1, 2, 3, 0]
    assert since[1] == 7 and np.isnan(since[0])


def test_same_day_runs_share_the_state_of_the_earlier_days():
    prm = {"q": 0.1, "r": 4.0, "m0": 0.0, "p0": 25.0}
    m, v, c, _ = ability.filter_dog_states(np.array([1, 1, 1]), np.array([0, 5, 5]), np.array([3.0, 9.0, 9.0]),
                                           np.zeros(3, bool), prm)
    assert m[1] == m[2] and c[1] == c[2] == 1


def test_trouble_prices_are_learnt_from_earlier_years():
    toks = [["crd1", "mid"]] * 300 + [["qaw", "rls"]] * 300
    y = np.r_[np.full(300, -3.0), np.full(300, 0.0)] + np.random.default_rng(0).normal(0, 0.1, 600)
    costs = trouble.fit_costs(toks, y, min_count=50, lam=1.0)
    assert costs["crd1"] + costs["mid"] < costs["qaw"] + costs["rls"] - 2.5


def test_prior_window_counts_only_earlier_days_in_the_window():
    df = pd.DataFrame({"k": ["a"] * 4, "race_date": ["2025-01-01", "2025-01-05", "2025-01-05", "2025-02-20"],
                       "v": [1.0, 2.0, 4.0, 8.0]})
    w = core.prior_window(df, ["k"], ["v"], 30)
    assert list(w.v) == [0.0, 1.0, 1.0, 0.0] and list(w.n) == [0, 1, 1, 0]
    a = core.prior_window(df, ["k"], ["v"], None)
    assert list(a.v) == [0.0, 1.0, 1.0, 7.0]


def test_the_identity_match_needs_the_slot_and_the_name():
    runs = synth(days=2)
    r0 = runs.iloc[:3]
    bf = pd.DataFrame({"race_date": r0.race_date, "race_time": r0.race_time, "track": r0.track, "trap": r0.trap,
                       "selection_id": [11, 12, 13], "event_id": 99,
                       "dog": [r0.dog_name.iloc[0].upper(), r0.dog_name.iloc[1] + "x", "Someone Else"]})
    tab, meta = ids.build(r0, None, {"runners": bf})
    assert list(tab.id_match) == ["exact", "near", "slot_only"]
    assert tab.id_selection_id.iloc[2] is pd.NA or pd.isna(tab.id_selection_id.iloc[2])


def test_a_season_dated_after_the_run_is_dropped():
    assert life.season_date("16.Au.26") == pd.Timestamp("2026-08-16")
    assert life.season_date("Unknown") is None and life.season_date("31.Fe.26") is None
    runs = synth(days=3)
    runs.loc[runs.index[0], "season"] = "01.De.30"
    tab, meta = life.build(runs, core.per_run_figures(runs))
    assert meta["season_after_run"] == 1 and "lf_season_days" not in tab                 # never a column (current value)
    assert life.season_date("16.Sp.26") == pd.Timestamp("2026-09-16")


def test_the_trainer_is_known_only_where_gbgb_was_read_at_the_time():
    runs = synth(days=3)
    days = sorted(runs.race_date.unique())
    lag = pd.Series([1, 30, 0], index=days)
    tr = core.trainer_at_the_time(runs, lag)
    assert tr[runs.race_date == days[1]].isna().all() and tr[runs.race_date == days[0]].notna().all()
    assert core.trainer_at_the_time(runs, None).isna().all()


def test_the_registry_lists_every_column_with_a_source():
    reg = pit.registry()
    assert len(reg) > 60 and all(r["source"] and r["definition"] for r in reg)
    assert {r["table"] for r in reg} == set(pit.TABLES)
