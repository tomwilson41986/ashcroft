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


@pytest.mark.parametrize("name", ["ability", "trouble", "connections", "market", "breeding", "life", "figs", "form", "draw", "bend"])
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


# ---- the figs table (the BFSP plan of 10 Nov 2026: corrected figures, evidence-aware levels, the market rating)

def test_the_figs_table_reads_ability_and_the_market_reads_it_better():
    from greyhound.pit import figs
    runs = synth(days=60, seed=1)
    tab, notes = figs.build(runs, core.per_run_figures(runs))
    d = runs[["race_id", "dog_id", "race_date"]].merge(tab, on=["race_id", "dog_id"])
    rng = np.random.default_rng(1)
    skill = rng.normal(0, 1.5, 81)                                   # synth's abilities (same seed, same draw)
    late = d[d.race_date >= sorted(d.race_date.unique())[40]]
    s = skill[late.dog_id.to_numpy()]
    assert np.corrcoef(late.fg_tot, s)[0, 1] > 0.6                    # the speed level finds the abilities
    assert np.corrcoef(late.fg_mr, s)[0, 1] > np.corrcoef(late.fg_tot, s)[0, 1]   # the market's rating, more so
    shares = late.groupby("race_id")[["fg_mr_share", "fg_tot_share"]].sum()
    assert np.allclose(shares, 1.0, atol=1e-4)
    assert notes["tau"] in figs.TAUS and set(notes["market_rating"]) >= {"r", "q", "v0"}


def test_a_debutant_reads_the_grade_par_and_evidence_moves_it():
    from greyhound.pit import figs
    runs = synth(days=40, seed=2)
    tab, _ = figs.build(runs, core.per_run_figures(runs))
    d = runs[["race_id", "dog_id", "t"]].merge(tab, on=["race_id", "dog_id"]).sort_values("t")
    first = d.groupby("dog_id").head(1)
    assert (first.fg_tot_n == 0).all()                               # no evidence on a dog's first day
    later = d[d.fg_tot_n > 3]
    assert later.fg_tot.std() > first.fg_tot.std()                   # with evidence the levels spread apart


def test_the_variant_is_measured_against_the_figures_own_standard():
    # a track whose times fall 0.5 s over a year: a winner running to the recent standard scores near 0 on the
    # corrected figure (the engine's track-normalised times counted the drift twice)
    from greyhound.pit import figs
    rows, rid = [], 0
    for d in range(500):
        day = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        base = 29.0 - 0.5 * d / 365
        for r in range(6):
            rid += 1
            for k in range(6):
                t = base + 0.08 * k + (0.0 if k else 0.0)
                rows.append({"race_date": day, "race_time": f"19:{10 * r:02d}", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 480.0, "trap": k + 1,
                             "dog_id": rid * 10 + k, "won": float(k == 0), "adjusted_time": t, "run_time": t,
                             "sectional": 4.8 + 0.01 * k, "comment": "", "is_trial": False})
    df = pd.DataFrame(rows)
    f = figs.figures(df)
    last = df.race_date >= "2025-03-01"
    assert f.f_tot[last & (df.won == 1)].abs().mean() < 0.5          # within half a length of 0
    assert (f.rel_tot.dropna() <= 1).all()


def test_the_market_rating_ranks_the_field_like_the_sp():
    from greyhound.pit import figs
    runs = synth(days=50, seed=3)
    prm = {"r": 0.3, "q": 0.002, "v0": 1.0}
    mr, ce = figs.market_rating(runs.sort_values(["t", "race_id", "trap"]).reset_index(drop=True), prm)
    assert np.isfinite(ce) and ce > 0
    d = runs.sort_values(["t", "race_id", "trap"]).reset_index(drop=True).assign(m=mr.m.to_numpy())
    late = d[d.race_date >= sorted(d.race_date.unique())[30]].dropna(subset=["m"])
    rho = late.groupby("race_id").apply(lambda g: g.m.rank().corr(g.sp_p_norm.rank()) if len(g) > 2 else np.nan)
    assert rho.mean() > 0.5                                           # before each race it orders the field as the SP does



def test_the_bsp_rating_reads_the_betfair_sp_where_the_race_has_it():
    from greyhound.pit import figs
    runs = synth(days=40, seed=5)
    prices = runs[["race_date", "race_time", "track", "trap"]].assign(
        market="win", country="GB", bsp=runs.sp_decimal.to_numpy() * 1.05)
    prices = prices[runs.race_date < "2025-01-30"]                    # the price files stop: the SP after
    tab, notes = figs.build(runs, core.per_run_figures(runs), {"prices": prices})
    assert {"fg_bmr", "fg_bmr_share"} <= set(tab.columns) and notes["bsp_rating"]["races_on_bsp"] > 0
    d = runs[["race_id", "dog_id"]].merge(tab, on=["race_id", "dog_id"])
    assert np.allclose(d.groupby("race_id").fg_bmr_share.sum(), 1.0, atol=1e-4)



def test_the_trend_filter_finds_a_rising_dog_and_ignores_a_wild_run():
    from greyhound.pit import form
    rows = []
    for d in range(60):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * d)).strftime("%Y-%m-%d")
        for dog, (base, rise) in enumerate(((0.0, 0.0), (0.0, 0.15))):        # dog 1 improves 0.15 lengths a week
            fig = base + rise * d + (25.0 if (dog == 0 and d == 30) else 0.0)  # dog 0: one wild run
            rows.append({"dog_id": dog, "race_id": d * 10 + dog, "race_date": day, "t": pd.Timestamp(day),
                         "born": "Jan-2023", "sex": "d"})
            rows[-1]["_f"] = fig
    df = pd.DataFrame(rows)
    curve = {0: {float(a): 0.0 for a in range(12, 73)}, 1: {float(a): 0.0 for a in range(12, 73)}}
    prm = {"ql": 0.01, "qs": 1e-4, "r": 1.0, "m0": 0.0, "p0": 4.0, "s0": 1e-3}
    st, ll = form.trend_filter(df, df._f, np.zeros(len(df), bool), curve, prm)
    last = df.race_date == df.race_date.max()
    rise = st[last & (df.dog_id == 1)].iloc[0]
    flat = st[last & (df.dog_id == 0)].iloc[0]
    assert rise.slope > 0.3 and abs(flat.slope) < 0.2                          # lengths a month
    assert abs(rise.level - 0.15 * 59) < 1.5                                   # it carries the level forward
    after = st[(df.dog_id == 0) & (df.race_date == sorted(df.race_date.unique())[31])].iloc[0]
    assert abs(after.level) < 2.0 and np.isfinite(ll)                          # the wild run barely moved it


def test_the_form_table_fits_its_levels_and_reads_the_market():
    from greyhound.pit import form
    runs = synth(days=50, seed=6)
    tab, notes = form.build(runs, core.per_run_figures(runs))
    assert {"fm_tot", "fm_level", "fm_slope", "fm_mr_innov"} <= set(tab.columns)
    assert notes["levels"]["tot"]["half_life"] in form.HALF_LIVES and notes["levels"]["tot"]["k"] in form.PRIORS
    d = runs[["race_id", "dog_id", "race_date"]].merge(tab, on=["race_id", "dog_id"])
    skill = np.random.default_rng(6).normal(0, 1.5, 81)
    late = d[d.race_date >= sorted(d.race_date.unique())[35]].dropna(subset=["fm_level"])
    assert np.corrcoef(late.fm_level, skill[late.dog_id.to_numpy()])[0, 1] > 0.5



def test_the_trap_state_follows_a_planted_bias():
    from greyhound.pit import draw
    runs = synth(days=120, seed=7)
    # trap 1 at Romford wins far more than its SP says from day 40 on
    late = (runs.race_date >= "2025-02-10") & (runs.track == "Romford")
    r = runs.copy()
    r.loc[late, "won"] = (r.loc[late, "trap"] == 1).astype(float)
    tab, notes = draw.build(r, core.per_run_figures(r))
    d = r[["race_id", "dog_id", "trap", "track", "race_date"]].merge(tab, on=["race_id", "dog_id"])
    end = d[(d.race_date == d.race_date.max()) & (d.track == "Romford")]
    assert end[end.trap == 1].dr_win_vs_sp.mean() > end[end.trap != 1].dr_win_vs_sp.mean() + 0.1
    assert notes["win"]["q"] in draw.QS


def test_the_bend_model_ranks_a_planted_fast_starter_first(monkeypatch):
    from greyhound.pit import bend
    monkeypatch.setattr(bend, "MIN_FIT", 500)
    monkeypatch.setattr(bend, "SIMS", 20)
    runs = synth(days=400, seed=8)            # 2025 and into 2026: 2026 is priced by a model fitted on 2025
    tab, notes = bend.build(runs, core.per_run_figures(runs))
    assert 2026 in notes["years"] and notes["years"][2026]["lead_beta"]["esr_vs_field"] > 0
    d = runs[["race_id", "dog_id", "race_date", "sectional"]].merge(tab, on=["race_id", "dog_id"])
    d = d[d.race_date >= "2026-01-01"].dropna(subset=["bd_p_lead"])
    assert np.allclose(d.groupby("race_id").bd_p_lead.sum(), 1.0, atol=1e-6)
    first = d.loc[d.groupby("race_id").sectional.idxmin()]
    assert first.bd_p_lead.mean() > 1 / 6 + 0.05                       # the dog that led got more than a fair chance


def cardify_last_day(runs):
    """The last day as the live path sees it before racing (greyhound/card.py ``cards``, ``with_cards``, ``clean``):
    no position, time, sectional, weight, SP or comment; ``is_card`` set."""
    r = runs.copy()
    last = r.race_date == r.race_date.max()
    for col in ("position", "beaten_lengths", "sectional", "run_time", "adjusted_time", "weight_kg", "sp_decimal",
                "going", "market_pos"):
        if col in r:
            r.loc[last, col] = np.nan
    r.loc[last, "comment"] = ""
    r["is_card"] = last
    return clean(r.drop(columns=[c for c in ("won", "placed2", "sp_p", "sp_p_norm", "t", "field") if c in r]))


SERVED_TABLES = ("ability", "trouble", "connections", "market", "breeding", "life", "figs", "form", "draw", "bend")


@pytest.mark.parametrize("name", SERVED_TABLES)
def test_a_card_gets_the_features_its_run_had(name, monkeypatch):
    """The live model builds the tables with today's card as the last day (greyhound/live.py ``card_tables``): a card
    row, with no result yet, must get exactly the features the nightly build gives the run once it is in."""
    monkeypatch.setattr(trouble, "MIN_FIT", 100)
    monkeypatch.setattr(ability, "FIT_BEFORE", "2025-01-15")
    runs = synth()
    last = runs.race_date.max()
    keys = runs.loc[runs.race_date == last, ["race_id", "dog_id"]]
    ctx = {"fetch_lag": pd.Series(0, index=sorted(runs.race_date.unique()))}
    names = [t for t in pit.TABLES if t in SERVED_TABLES and pit.TABLES.index(t) <= pit.TABLES.index(name)]
    full = pit.build_tables(names, runs, None, dict(ctx), keep=True)[name]
    card = pit.build_tables(names, cardify_last_day(runs), None, dict(ctx), keep=True)[name]
    mod = pit.module(name)
    cols = [c for c in full.columns if c.startswith(mod.PREFIX) and c not in getattr(mod, "CARD_UNSAFE", ())]
    x = keys.merge(full, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    y = keys.merge(card, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert len(y) == len(keys)
    bad = [c for j, c in enumerate(cols) if not np.allclose(x[:, j], y[:, j], equal_nan=True, atol=1e-5)]
    assert not bad, f"{name}: {bad}"
