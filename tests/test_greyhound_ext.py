"""greyhound/pit/ext.py: what the dogs drawn beside a dog do to its time. A planted culprit (it costs the dog on its
outside 2 lengths) is found on its neighbours' rows and nowhere else; a dog that runs slower in company and one that
runs slower pressed from the outside are found; the heavy dog's mass ratio is above 0 and the light dog's below; a
vacant trap is flagged and costs nothing; the sums are right by hand, a day's two races read the same earlier days;
the day's own results move nothing and a card row gets what its run had."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, ext, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day
from tests.test_greyhound_pit import synth as pit_synth

L = core.LENGTH_S
HEAVY, LIGHT, CULPRIT, SHY, LEANT = 2, 3, 1, 5, 6
KG = {HEAVY: 36.0, LIGHT: 26.0}


def pace(dog: int) -> float:
    """The dog's early pace in lengths (to the line the first time): dog 1 the slowest, dog 12 the fastest."""
    return (dog - 6.5) * 0.3


def synth(days=90, seed=3, harm=2.0, crowd=1.5, press=2.0):
    """One track, 12 dogs of equal ability over the trip and different early pace, two races a day, the traps drawn
    at random; on about a third of the days one of the second race's middle traps is vacant. Dog 1 costs the dog
    drawn immediately outside it ``harm`` lengths every time. Dog 5 runs ``crowd`` lengths slower with both adjacent
    traps occupied. Dog 6 runs ``press`` lengths slower when the dog on its outside is 0.5 lengths or more faster
    early. Dog 2 weighs 36 kg, dog 3 26 kg, the rest 30."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 7000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        order = rng.permutation(np.arange(1, 13))
        vacant = int(rng.integers(2, 6)) if rng.random() < 0.35 else None
        for r, field in enumerate((order[:6], order[6:])):
            rid += 1
            at = {k + 1: int(dog) for k, dog in enumerate(field) if not (r == 1 and k + 1 == vacant)}
            t = {trap: 29.5 + rng.normal(0, 0.05) for trap in at}
            for trap, dog in at.items():
                if at.get(trap - 1) == CULPRIT:
                    t[trap] += harm * L
                if dog == SHY and trap - 1 in at and trap + 1 in at:
                    t[trap] += crowd * L
                if dog == LEANT and trap + 1 in at and pace(at[trap + 1]) - pace(dog) >= 0.5:
                    t[trap] += press * L
            rank = {trap: i + 1 for i, trap in enumerate(sorted(t, key=t.get))}
            for trap, dog in at.items():
                rows.append({"race_date": day, "race_time": f"{18 + r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": trap, "dog_id": dog,
                             "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d", "sp_decimal": 6.0,
                             "position": rank[trap], "beaten_lengths": 0.5,
                             "sectional": 4.5 - pace(dog) * L + rng.normal(0, 0.01), "run_time": float(t[trap]),
                             "adjusted_time": float(t[trap]), "weight_kg": KG.get(dog, 30.0), "comment": "QAw"})
    return clean(pd.DataFrame(rows))


def beside(m: pd.DataFrame, step: int, cols: list[str]) -> pd.DataFrame:
    """The row of the runner ``step`` traps away in the same race (NaN with none), aligned to ``m``."""
    o = m[["race_id", "trap"] + cols].assign(trap=m.trap - step)
    return m[["race_id", "trap"]].merge(o, on=["race_id", "trap"], how="left")[cols].set_axis(m.index)


@pytest.fixture(scope="module")
def built():
    runs = synth()
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    tab, notes = ext.build(runs, figs, {"xtime": xt})
    m = runs.merge(tab, on=["race_id", "dog_id"]).merge(xt[["race_id", "dog_id", "xt_sec"]], on=["race_id", "dog_id"])
    return runs, m, notes, tab


def test_every_column_is_registered_and_there_is_a_row_a_run(built):
    runs, m, notes, tab = built
    assert {c for c in tab.columns if c.startswith(ext.PREFIX)} == set(ext.COLUMNS) and len(tab) == len(runs)
    assert list(tab.columns) == ["race_id", "dog_id"] + list(ext.COLUMNS)
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert all(tab[c].dtype == np.float32 for c in ext.COLUMNS)
    assert notes["rows"] == len(runs) and notes["rows_without_a_trap_1_to_6"] == 0
    assert notes["runs_with_a_rated_neighbour"] > 0.9 and 0 < notes["inside_trap_vacant_share"] < 0.1


def test_the_planted_culprit_is_charged_on_its_neighbours_rows(built):
    _, m, notes, _ = built
    late = m[m.race_date >= "2025-03-01"]
    inside = beside(late, -1, ["dog_id"]).dog_id
    outside = beside(late, 1, ["dog_id"]).dog_id
    hit = late[inside == CULPRIT]                                   # drawn immediately outside the culprit today
    rest = late[(inside != CULPRIT) & (late.dog_id != CULPRIT)]
    # 2 lengths every time: less the race's mean (a sixth of it) and the shrinkage, about 1.4
    assert len(hit) > 15 and 1.0 < hit.nx_in.mean() < 2.0 and (hit.nx_in > 0.7).all()
    assert abs(rest.nx_in.mean()) < 0.2 and rest.nx_in.abs().quantile(0.9) < 0.6
    assert hit.nx_sum.mean() > rest.nx_sum.mean() + 1.0 and hit.nx_n.min() >= 30
    # it does nothing to the dog on its inside: the rating is one-sided
    assert late[outside == CULPRIT].nx_out.abs().mean() < 0.6
    own = late[late.dog_id == CULPRIT]
    assert own.nx_own_out.mean() > 1.0 and own.nx_own_in.abs().mean() < 0.6
    assert late[late.dog_id != CULPRIT].nx_own_out.abs().mean() < 0.4
    assert notes["corr_nx_sum_with_the_runs_residual_on_those"] > 0.1   # the neighbours' record tells the residual


def test_a_dog_that_minds_company_and_one_pressed_from_the_outside(built):
    _, m, _, _ = built
    late = m[m.race_date >= "2025-03-01"]
    shy, others = late[late.dog_id == SHY], late[~late.dog_id.isin([SHY, LEANT, CULPRIT])]
    assert shy.nx_victim.mean() > 0.5 and abs(others.nx_victim.mean()) < 0.3
    leant = late[late.dog_id == LEANT]
    assert leant.nx_lat_out.mean() > 0.4 and abs(others.nx_lat_out.mean()) < 0.25
    assert leant.nx_lat_out.mean() > leant.nx_lat_in.mean() + 0.3
    # today's pressure: its record applies only when the outside neighbour is expected that much faster
    out_sec = beside(late, 1, ["xt_sec"]).xt_sec.loc[leant.index]
    pressed = (out_sec - leant.xt_sec) >= ext.PRESS
    assert pressed.sum() > 5 and (~pressed).sum() > 5
    assert leant[pressed].nx_press.mean() > leant[~pressed].nx_press.mean() + 0.3


def test_the_heavy_dog_and_the_light_dog(built):
    _, m, _, _ = built
    late = m[m.race_date >= "2025-01-10"]
    assert late[late.dog_id == HEAVY].nx_mass.mean() > 0.05 and late[late.dog_id == LIGHT].nx_mass.mean() < -0.05
    assert late[~late.dog_id.isin([HEAVY, LIGHT])].nx_mass.abs().mean() < 0.05
    # the construction itself: log of the mass ratio by exp(-|difference of the expected sectionals|)
    late = late.assign(w=late.dog_id.map(KG).fillna(30.0))
    for side, step in (("in", -1), ("out", 1)):
        nb = beside(late, step, ["w", "xt_sec"])
        want = np.where(nb.w.notna(), np.log(late.w / nb.w) * np.exp(-(late.xt_sec - nb.xt_sec).abs()), 0.0)
        assert np.allclose(late[f"nx_mass_{side}"].to_numpy(float), want, atol=1e-5)
    assert np.allclose(late.nx_mass, late.nx_mass_in + late.nx_mass_out, atol=1e-6)
    heavy_light = late[(late.dog_id == HEAVY) & (beside(late, -1, ["dog_id"]).dog_id == LIGHT)]
    assert len(heavy_light) and (heavy_light.nx_mass_in > 0).all() and (heavy_light.nx_mass_in <= np.log(36 / 26) + 1e-6).all()
    assert m[m.race_date == m.race_date.min()].nx_mass.isna().all()          # no earlier weight on the first day


def test_a_vacant_trap_is_flagged_and_costs_nothing(built):
    _, m, _, _ = built
    five = m[m.groupby("race_id").dog_id.transform("size") == 5]
    assert five.race_id.nunique() > 10
    has_in = beside(m, -1, ["dog_id"]).dog_id.notna()
    has_out = beside(m, 1, ["dog_id"]).dog_id.notna()
    assert (m.nx_free_in == (~has_in).astype(float)).all() and (m.nx_free_out == (~has_out).astype(float)).all()
    assert (m[m.trap == 1].nx_free_in == 1).all() and (m[m.trap == 6].nx_free_out == 1).all()
    gap = five[(five.trap > 1) & (five.nx_free_in == 1)]                       # drawn just outside the vacant trap
    assert len(gap) == five.race_id.nunique() and (gap.nx_in == 0).all() and (gap.nx_mass_in == 0).all()
    full = m[m.groupby("race_id").dog_id.transform("size") == 6]
    assert (full[full.trap.between(2, 5)][["nx_free_in", "nx_free_out"]] == 0).all().all()
    assert np.allclose(m.nx_sum, m.nx_in + m.nx_out, atol=1e-6)


def by_hand():
    """Three days by hand: each race's ``lengths slower than expected`` sum to 0, so they are the race-centred
    residuals; dog 1 runs twice on day 2; day 3 has trap 4 vacant."""
    races = [  # day, time, race, [(trap, dog, lengths slower, weight, expected sectional in lengths)]
        ("2025-01-01", "18:00", 101, [(1, 1, 0, 36.0, 0.0), (2, 2, 3, 27.0, 1.0), (3, 3, 0, 30.0, 0.0),
                                      (4, 4, 0, 30.0, 0.0), (5, 5, -3, 30.0, 0.0), (6, 6, 0, 30.0, 0.0)]),
        ("2025-01-02", "18:00", 102, [(1, 1, 0, 37.0, 0.0), (2, 7, 6, 30.0, 0.0), (3, 8, 0, 30.0, 0.0),
                                      (4, 9, 0, 30.0, 0.0), (5, 10, 0, 30.0, 0.0), (6, 11, -6, 30.0, 0.0)]),
        ("2025-01-02", "19:00", 103, [(1, 12, -2, 30.0, 0.2), (2, 1, 3, 5.0, 0.0), (3, 13, 4, 30.0, 2.0),
                                      (4, 14, 0, 30.0, 0.0), (5, 15, -3, 30.0, 0.0), (6, 16, -2, 30.0, 0.0)]),
        ("2025-01-03", "18:00", 104, [(1, 3, 1, 30.0, 0.8), (2, 1, -1, 30.0, 1.0), (3, 2, 0, 30.0, 1.5),
                                      (5, 6, 0, 30.0, 0.0), (6, 5, 0, 30.0, 0.0)]),
    ]
    rows, xt = [], []
    for day, tm, rid, field in races:
        order = sorted(field, key=lambda f: f[2])
        for trap, dog, slower, kg, sec in field:
            t = 29.5 + slower * L
            rows.append({"race_date": day, "race_time": tm, "meeting_id": int(day[-2:]), "track": "Hove",
                         "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": trap, "dog_id": dog,
                         "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d", "sp_decimal": 6.0,
                         "position": 1 + [f[1] for f in order].index(dog), "beaten_lengths": 0.5, "sectional": 4.5,
                         "run_time": t, "adjusted_time": t, "weight_kg": kg, "comment": "QAw"})
            xt.append({"race_id": rid, "dog_id": dog, "xt_tot_time": 29.5, "xt_sec": sec})
    return clean(pd.DataFrame(rows)), pd.DataFrame(xt)


def test_the_sums_by_hand_and_a_days_two_races_read_the_same_earlier_days():
    runs, xt = by_hand()
    tab, _ = ext.build(runs, core.per_run_figures(runs), {"xtime": xt})
    m = runs[["race_id", "dog_id", "trap"]].merge(tab, on=["race_id", "dog_id"]).set_index(["race_id", "dog_id"])
    K = ext.K
    # day 2: dog 1 cost the dog outside it 3 lengths on day 1; both its races of day 2 read that and not each other
    assert m.loc[(102, 7), "nx_in"] == pytest.approx(3 / (1 + K)) and m.loc[(103, 13), "nx_in"] == pytest.approx(3 / (1 + K))
    assert m.loc[(102, 1), "nx_own_out"] == m.loc[(103, 1), "nx_own_out"] == np.float32(3 / (1 + K))
    assert m.loc[(103, 13), "nx_n"] == 1 and m.loc[(103, 1), "nx_own_in"] == 0
    # a dog with no history: no rating, no evidence, no weight, no record in company or under pressure
    new = m.loc[(103, 16)]
    assert new.nx_own_out == 0 and new.nx_own_in == 0 and new.nx_in == 0 and new.nx_n == 0 and new.nx_press == 0
    assert np.isnan(new.nx_victim) and np.isnan(new.nx_mass) and np.isnan(new.nx_lat_out) and np.isnan(new.nx_lat_in)
    assert new.nx_free_out == 1 and new.nx_free_in == 0
    # the weight is the last EARLIER day's: 36 on day 2 (not that day's 37), against a neighbour with none
    assert np.isnan(m.loc[(103, 1), "nx_mass_in"]) and np.isnan(m.loc[(103, 1), "nx_mass"])
    # day 3. Dog 2, trap 3, outside dog 1 (3 + 6 + 4 over 3 neighbour-runs), trap 4 vacant
    two = m.loc[(104, 2)]
    assert two.nx_in == pytest.approx(13 / (3 + K)) and two.nx_out == 0 and two.nx_sum == pytest.approx(13 / (3 + K))
    assert two.nx_n == 3 and two.nx_free_out == 1 and two.nx_free_in == 0
    assert two.nx_mass_in == pytest.approx(np.log(27 / 37) * np.exp(-0.5), abs=1e-6) and two.nx_mass_out == 0
    assert two.nx_mass == pytest.approx(two.nx_mass_in)
    # dog 3, trap 1, inside dog 1 (the dog inside dog 1 ran 2 lengths faster, once)
    three = m.loc[(104, 3)]
    assert three.nx_free_in == 1 and three.nx_in == 0 and three.nx_out == pytest.approx(-2 / (1 + K)) and three.nx_n == 1
    # dog 1 itself: its ratings, its record in company (3 slower, once) against alone (0, twice), its weight 37 (the
    # 5 kg of its last run is a keying error and is passed over), pressed from outside twice (0, 3), inside once (3)
    one = m.loc[(104, 1)]
    assert one.nx_own_out == pytest.approx(13 / (3 + K)) and one.nx_own_in == pytest.approx(-2 / (1 + K))
    assert one.nx_in == 0 and one.nx_out == 0 and one.nx_n == 2
    assert one.nx_victim == pytest.approx(3 / (1 + K) - 0 / (2 + K))
    assert one.nx_mass_in == pytest.approx(np.log(37 / 30) * np.exp(-0.2), abs=1e-6)
    assert one.nx_mass_out == pytest.approx(np.log(37 / 27) * np.exp(-0.5), abs=1e-6)
    base = 3 / (3 + K)
    assert one.nx_lat_out == pytest.approx(3 / (2 + K) - base, abs=1e-6)
    assert one.nx_lat_in == pytest.approx(3 / (1 + K) - base, abs=1e-6)
    assert one.nx_press == pytest.approx(3 / (2 + K) + 3 / (1 + K) - 2 * base, abs=1e-6)      # pressed from both today
    # beside the vacant trap, and the widest trap
    assert m.loc[(104, 6), "nx_free_in"] == 1 and m.loc[(104, 6), "nx_in"] == 0 and m.loc[(104, 5), "nx_free_out"] == 1
    # day 3's own times (dog 3 a length slow, dog 1 a length fast) are in nothing: dog 3 was level on day 1
    assert m.loc[(104, 3), "nx_own_out"] == 0 and m.loc[(104, 3), "nx_own_in"] == pytest.approx(3 / (1 + K))


def test_no_expected_sectional_no_mass_and_no_pressure():
    """A track and trip whose sectionals are not timed (the sprints) has no expected sectional: the columns that
    need it are NaN, the neighbours' ratings are still read."""
    runs, xt = by_hand()
    tab, notes = ext.build(runs, core.per_run_figures(runs), {"xtime": xt.assign(xt_sec=np.nan)})
    m = runs[["race_id", "dog_id"]].merge(tab, on=["race_id", "dog_id"]).set_index(["race_id", "dog_id"])
    one, two = m.loc[(104, 1)], m.loc[(104, 2)]
    assert np.isnan(one.nx_press) and np.isnan(one.nx_lat_out) and np.isnan(one.nx_lat_in)
    assert np.isnan(one.nx_mass_in) and np.isnan(one.nx_mass_out) and np.isnan(one.nx_mass)
    assert two.nx_mass_out == 0 and np.isnan(two.nx_mass_in) and np.isnan(two.nx_mass)   # a vacant side is no reading
    assert two.nx_in == pytest.approx(13 / (3 + ext.K)) and one.nx_victim == pytest.approx(3 / (1 + ext.K))
    assert notes["runs_with_an_expected_sectional"] == 0


def test_a_trap_outside_one_to_six_is_read_by_nothing():
    runs, xt = by_hand()
    odd = runs.copy()
    odd.loc[(odd.race_id == 101) & (odd.dog_id == 2), "trap"] = 8          # the dog dog 1 cost 3 lengths on day 1
    tab, notes = ext.build(odd, core.per_run_figures(odd), {"xtime": xt})
    m = odd[["race_id", "dog_id"]].merge(tab, on=["race_id", "dog_id"]).set_index(["race_id", "dog_id"])
    assert len(tab) == len(odd) and notes["rows_without_a_trap_1_to_6"] == 1
    assert m.loc[(101, 2)][list(ext.COLUMNS)].isna().all()
    assert m.loc[(101, 1), "nx_free_out"] == 1 and m.loc[(102, 7), "nx_in"] == 0 and m.loc[(102, 7), "nx_n"] == 0


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=40)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = ext.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = ext.build(r2, core.per_run_figures(r2))[0]
    cols = list(ext.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).mean() > 0.8 and np.allclose(x, z, atol=1e-4, equal_nan=True)


def test_the_expected_time_table_is_taken_from_the_context():
    runs = synth(days=30)
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = ext.build(runs, figs, {"xtime": xt})[0]
    b = ext.build(runs, figs)[0]
    assert np.allclose(a[list(ext.COLUMNS)].to_numpy(float), b[list(ext.COLUMNS)].to_numpy(float), equal_nan=True)


# --- the reviewer's leak tests (11 Nov 2026): a uniform shift of a day's times leaves a race-centred residual where it
# was, so the day's results are rewritten row by row, on a MIDDLE day, and every day up to it must stay put


def scramble_day(runs: pd.DataFrame, day: str, seed: int = 1) -> pd.DataFrame:
    """One day's results rewritten row by row: each time moved by its own amount (so the race-centred residuals
    change), the sectionals and the weights too, the positions re-ranked, three runs in ten a fall."""
    rng = np.random.default_rng(seed)
    r = runs.copy()
    m = (r.race_date == day).to_numpy()
    k = int(m.sum())
    shift = rng.normal(0, 0.4, k)
    r.loc[m, "adjusted_time"] = r.loc[m, "adjusted_time"].to_numpy() + shift
    r.loc[m, "run_time"] = r.loc[m, "run_time"].to_numpy() + shift
    r.loc[m, "sectional"] = r.loc[m, "sectional"].to_numpy() + rng.normal(0, 0.1, k)
    r.loc[m, "weight_kg"] = r.loc[m, "weight_kg"].to_numpy() + rng.normal(0, 3, k)
    r.loc[m, "position"] = r[m].groupby("race_id").adjusted_time.rank(method="first").to_numpy()
    r.loc[m, "comment"] = np.where(rng.random(k) < 0.3, "Fell", "SAw,Bmp1,Wide")
    r.loc[m, "sp_decimal"] = 9.0
    r["won"] = (r.position == 1).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


def _moved(a: pd.DataFrame, b: pd.DataFrame, rows: np.ndarray) -> dict:
    """The cells of each column that differ between two builds on ``rows`` (a NaN against a number counts)."""
    cols = list(ext.COLUMNS)
    x, z = a.loc[rows, cols].to_numpy(float), b.loc[rows, cols].to_numpy(float)
    diff = ~np.isclose(x, z, atol=1e-6, equal_nan=True)
    return {c: int(diff[:, j].sum()) for j, c in enumerate(cols) if diff[:, j].any()}


@pytest.mark.parametrize("rebuild", [False, True], ids=["same expected times", "expected times rebuilt"])
@pytest.mark.parametrize("make", [lambda: synth(days=50), pit_synth], ids=["planted", "two tracks, dogs twice a day"])
def test_a_middle_days_results_move_nothing_on_that_day_or_before(make, rebuild):
    """A result of day D may only enter a feature of a LATER day: every column of every row of day D (the same race,
    the other races of the day, a dog's second race of the day) and of the days before is what it was; the later
    days do move (the test can see a result being read)."""
    runs = make()
    days = sorted(runs.race_date.unique())
    mid = days[int(len(days) * 0.6)]
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = ext.build(runs, figs, {"xtime": xt})[0]
    r2 = scramble_day(runs, mid)
    assert (r2.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (r2.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    f2 = core.per_run_figures(r2)
    b = ext.build(r2, f2, {"xtime": xtime.build(r2, f2)[0] if rebuild else xt})[0]
    upto, after = (runs.race_date <= mid).to_numpy(), (runs.race_date > mid).to_numpy()
    assert (runs.race_date == mid).sum() >= 10 and np.isfinite(a.loc[upto, list(ext.COLUMNS)].to_numpy(float)).mean() > 0.8
    assert _moved(a, b, upto) == {}
    later = _moved(a, b, after)
    for c in ("nx_in", "nx_out", "nx_sum", "nx_own_out", "nx_own_in", "nx_victim", "nx_lat_out", "nx_lat_in",
              "nx_press", "nx_mass_in", "nx_mass_out", "nx_mass"):
        assert later.get(c, 0) > 0, c                              # each reads results (or the weigh-in) of earlier days
    assert "nx_free_in" not in later and "nx_free_out" not in later          # the card alone


def test_the_days_after_are_not_read():
    """Built on the runs up to a day, or on all of them, with the same expected times: the same features up to it."""
    runs = pit_synth()
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = ext.build(runs, figs, {"xtime": xt})[0]
    days = sorted(runs.race_date.unique())
    for cut in (days[5], days[len(days) // 2], days[-2]):
        head = runs[runs.race_date <= cut].reset_index(drop=True)
        b = ext.build(head, core.per_run_figures(head), {"xtime": xt})[0]
        x = a.loc[(runs.race_date <= cut).to_numpy(), list(ext.COLUMNS)].to_numpy(float)
        assert len(b) == len(head) and np.allclose(x, b[list(ext.COLUMNS)].to_numpy(float), atol=1e-6, equal_nan=True)


def test_a_dogs_two_races_of_a_day_read_the_same_earlier_days():
    """In the two-track data a dog runs more than once a day: its own ratings, its record in company and under
    pressure and its last weight are the same on each of the day's rows (none reads the day's earlier race)."""
    runs = pit_synth()
    tab = ext.build(runs, core.per_run_figures(runs))[0]
    m = runs[["race_date", "dog_id"]].join(tab[["nx_own_out", "nx_own_in", "nx_victim", "nx_lat_out", "nx_lat_in"]])
    g = m.groupby(["race_date", "dog_id"])
    twice = g.nx_own_out.transform("size") > 1
    assert twice.sum() > 200
    for c in ("nx_own_out", "nx_own_in", "nx_victim", "nx_lat_out", "nx_lat_in"):
        spread = g[c].transform(lambda s: s.max() - s.min() if s.notna().all() else (0.0 if s.isna().all() else np.inf))
        assert (spread[twice] == 0).all(), c


def test_a_runner_taken_out_moves_only_the_rows_beside_it():
    """The vacancy is read from the rows of the race (a result has the field that ran; a card has the field as it
    stood): a runner taken out of a last-day race frees the trap for its two neighbours, and changes no other row
    of the race, of the day or before."""
    runs = synth(days=40)
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = runs[["race_id", "dog_id", "trap"]].merge(ext.build(runs, figs, {"xtime": xt})[0], on=["race_id", "dog_id"])
    last = runs[runs.race_date == runs.race_date.max()]
    rid = int(last.groupby("race_id").dog_id.size().loc[lambda s: s == 6].index[0])
    less = runs[~((runs.race_id == rid) & (runs.trap == 3))].reset_index(drop=True)
    b = less[["race_id", "dog_id", "trap"]].merge(ext.build(less, core.per_run_figures(less), {"xtime": xt})[0],
                                                  on=["race_id", "dog_id"])
    m = a.merge(b, on=["race_id", "dog_id", "trap"], suffixes=("", "_b"))
    assert len(m) == len(runs) - 1
    cols = list(ext.COLUMNS)
    same = np.isclose(m[cols].to_numpy(float), m[[f"{c}_b" for c in cols]].to_numpy(float), atol=1e-6, equal_nan=True)
    beside_it = ((m.race_id == rid) & m.trap.isin([2, 4])).to_numpy()
    assert same[~beside_it].all()
    two, four = m[(m.race_id == rid) & (m.trap == 2)].iloc[0], m[(m.race_id == rid) & (m.trap == 4)].iloc[0]
    assert two.nx_free_out == 0 and two.nx_free_out_b == 1 and two.nx_out_b == 0 and two.nx_mass_out_b == 0
    assert four.nx_free_in == 0 and four.nx_free_in_b == 1 and four.nx_in_b == 0 and four.nx_mass_in_b == 0
    # on those two rows only the columns of the freed side (and the sums over both sides) may move
    may = {2: {"nx_out", "nx_sum", "nx_n", "nx_free_out", "nx_mass_out", "nx_mass", "nx_press"},
           4: {"nx_in", "nx_sum", "nx_n", "nx_free_in", "nx_mass_in", "nx_mass", "nx_press"}}
    for trap, allowed in may.items():
        row = np.flatnonzero(((m.race_id == rid) & (m.trap == trap)).to_numpy())[0]
        assert {c for j, c in enumerate(cols) if not same[row, j]} <= allowed, trap
