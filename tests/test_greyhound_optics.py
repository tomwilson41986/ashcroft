"""greyhound/pit/optics.py: the look of the printed card less the substance. A dog whose printed times are fast only
because its nights were fast has a positive surplus; a 29.95 best is just under the second and a 30.02 just over; a
run of short-head wins is counted as photos and as luck; the card is six lines, at today's track and trip; a void
race is no line; a second run the same day reads what the first did; a dog with no line has no look; the day's own
results move nothing and a card row gets what its run had. The review of 11 Nov 2026 added: a MIDDLE day's results
move nothing on that day or before it (and do move later days); the days after a day move nothing on it; a slow,
row-by-row reading of the specification gives the same table on a card where dogs run twice a day and at two tracks;
trap numbers are not read; an empty distance in GBGB's text is no photo; a calculated time no greyhound runs (a
keying error in GBGB) is no best time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, optics, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day
from tests.test_greyhound_pit import synth as pit_synth

L = core.LENGTH_S
FAST = 0.30                                   # seconds every dog runs faster on a fast night (3.75 lengths)


def synth(days=120, seed=4, fast_by=FAST):
    """One track, 36 dogs with abilities 1 length apart (sd); dog 1 is an average dog. Every tenth night the
    track runs ``fast_by`` seconds fast for every dog. Dog 1 runs only on the fast nights and two nights before the
    next; the other 35 run every night, so a night without dog 1 has a race of five. On dog 1's ordinary nights its
    card shows fast-night times and its rivals' six lines show none."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1.0 * L, 37)                               # seconds; above 0 slower
    skill[1] = 0.0
    rows, rid = [], 5000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        fast = d % 10 == 0
        dogs = rng.permutation(np.arange(2, 37))
        if d % 10 in (0, 8):
            dogs = rng.permutation(np.r_[1, dogs])
        for r in range(6):
            field = dogs[6 * r:6 * r + 6]
            rid += 1
            t = 29.5 + skill[field] + rng.normal(0, 0.05, len(field)) - (fast_by if fast else 0.0)
            order = np.argsort(t)
            pos = np.argsort(order) + 1
            gap = np.r_[0.0, np.diff(t[order])] / L
            beaten = np.empty(len(field))
            beaten[order] = gap
            for k, dog in enumerate(field):
                rows.append({"race_date": day, "race_time": f"{18 + r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": k + 1,
                             "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                             "sp_decimal": 6.0, "position": int(pos[k]), "beaten_lengths": float(round(beaten[k], 2)),
                             "sectional": 4.5, "run_time": float(round(t[k], 2)),
                             "adjusted_time": float(round(t[k], 2)), "weight_kg": 30.0, "comment": "QAw"})
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = optics.build(runs, core.per_run_figures(runs))
    return runs, runs.merge(tab, on=["race_id", "dog_id"]), tab, notes


def test_every_column_is_registered(built):
    runs, m, tab, notes = built
    assert {c for c in tab.columns if c.startswith(optics.PREFIX)} == set(optics.COLUMNS)
    assert list(tab.columns) == ["race_id", "dog_id"] + list(optics.COLUMNS)
    assert len(tab) == len(runs) == len(m) and (tab[list(optics.COLUMNS)].dtypes == np.float32).all()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert notes["rows"] == len(runs) and notes["rows_with_a_surplus"] > 0.9


def _dog_one(m):
    """Dog 1 on its ordinary nights (two before a fast night) once the card is full, and its rivals there."""
    d = (pd.to_datetime(m.race_date) - pd.Timestamp("2025-01-01")).dt.days
    late = m[(d >= 40) & (d % 10 == 8)]
    one = late[late.dog_id == 1]
    return late, one, late[late.race_id.isin(one.race_id) & (late.dog_id != 1)]


def test_fast_nights_flatter_the_card_and_not_the_expected_time(built):
    _, m, _, _ = built
    late, one, rest = _dog_one(m)
    plain = synth(fast_by=0.0)                                       # the same dogs and noise, no fast night
    tab0, _ = optics.build(plain, core.per_run_figures(plain))
    _, one0, _ = _dog_one(plain.merge(tab0, on=["race_id", "dog_id"]))
    assert len(one) == len(one0) == 8
    # its best is from a fast night (0.30 s = 3.75 lengths); its rivals' six lines are all ordinary nights
    assert (one.op_best < 29.3).all() and (one0.op_best > 29.3).all()
    assert (one.op_fastest == 1).all() and one0.op_fastest.mean() < 0.5
    assert 2.5 < one.op_best_vs.mean() - one0.op_best_vs.mean() < 4.5
    # the expected time took the night out: the substance has not moved, the look has, and the surplus with it
    assert abs(one.op_real.mean() - one0.op_real.mean()) < 0.3
    assert one.op_look.mean() - one0.op_look.mean() > 0.3
    assert one.op_surplus.mean() - one0.op_surplus.mean() > 0.3
    assert one.op_surplus.mean() > 0.15 and one.op_surplus.mean() > rest.op_surplus.mean() + 0.15
    assert np.allclose(one.op_surplus, one.op_look - one.op_real, atol=1e-5)
    # the look and the substance are z-scores within the race
    g = late.groupby("race_id")
    assert np.allclose(g.op_real.mean(), 0.0, atol=1e-5) and np.allclose(g.op_real.std(ddof=0), 1.0, atol=1e-4)
    # over every dog the luck is none, and the better dogs look better
    assert abs(m.op_luck6.mean()) < 0.2 and late.op_look.corr(late.op_real) > 0.3


def test_a_race_of_five_is_read_against_its_own_runners(built):
    _, m, _, _ = built
    five = m[(m.field == 5) & (m.race_date >= "2025-02-10")]
    assert len(five) and five.op_best_vs.notna().all()
    for _, g in list(five.groupby("race_id"))[:5]:
        want = [(g.op_best.drop(i).mean() - g.op_best[i]) / L for i in g.index]
        assert np.allclose(g.op_best_vs, want, atol=1e-3)
        assert g.op_fastest.sum() >= 1 and abs(g.op_real.mean()) < 1e-5


# ---- a card by hand: the digits, the photos, the six lines

def hand(days=9):
    """Two races a day at Hove over 500 m. Race A: dog 1 runs 29.95 (29.50 on the first day), dog 2 runs 30.02, dogs
    7 to 10 slower; on the sixth day race A is over 285 m. Race B: dog 3 beats dog 4 a short head every day, dog 5 is
    third 2.5 lengths back, then 6, 11, 12; on the fifth day race B is void (position 0, no times). On the last day
    dog 1 runs again later with five dogs that have never run."""
    rows = []

    def add(day, d, rid, hhmm, dist, field):
        for trap, (dog, pos, beaten, t) in enumerate(field, start=1):
            rows.append({"race_date": day, "race_time": hhmm, "meeting_id": d, "track": "Hove", "race_id": rid,
                         "race_class": "A3", "distance_m": dist, "trap": trap, "dog_id": dog, "dog_name": f"Dog {dog}",
                         "born": "Jan-2023", "sex": "d", "sp_decimal": 6.0, "position": pos, "beaten_lengths": beaten,
                         "sectional": 4.5, "run_time": t, "adjusted_time": t, "weight_kg": 30.0, "comment": "QAw"})

    for d in range(days):
        day = (pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        short = d == 5
        off = -12.6 if short else 0.0
        add(day, d, 100 + d, "18:00", 285.0 if short else 500.0,
            [(1, 1, 0.0, (29.50 if d == 0 else 29.95) + off), (2, 2, 0.75, 30.02 + off), (7, 3, 4.5, 30.38 + off),
             (8, 4, 1.0, 30.46 + off), (9, 5, 1.0, 30.54 + off), (10, 6, 1.0, 30.62 + off)])
        if d == 4:
            add(day, d, 200 + d, "18:15", 500.0, [(k, 0, np.nan, np.nan) for k in (3, 4, 5, 6, 11, 12)])
        else:
            add(day, d, 200 + d, "18:15", 500.0,
                [(3, 1, 0.0, 30.10), (4, 2, 0.05, 30.10), (5, 3, 2.5, 30.30), (6, 4, 1.5, 30.42),
                 (11, 5, 1.0, 30.50), (12, 6, 1.0, 30.58)])
        if d == days - 1:
            add(day, d, 300 + d, "21:00", 500.0,
                [(1, 2, 1.0, 30.20), (13, 1, 0.0, 30.12), (14, 3, 1.0, 30.28), (15, 4, 1.0, 30.36),
                 (16, 5, 1.0, 30.44), (17, 6, 1.0, 30.52)])
    return clean(pd.DataFrame(rows))


def flat_xt(runs, slow=(3,)):
    """An expected-time table by hand: every dog equal, but the dogs of ``slow`` three lengths slower."""
    return pd.DataFrame({"race_id": runs.race_id, "dog_id": runs.dog_id,
                         "xt_tot": np.where(runs.dog_id.isin(slow), -3.0, 0.0)})


@pytest.fixture(scope="module")
def card():
    runs = hand()
    tab, notes = optics.build(runs, core.per_run_figures(runs), {"xtime": flat_xt(runs)})
    m = runs.merge(tab, on=["race_id", "dog_id"])
    return m, notes


def row(m, dog, d, race_time=None):
    day = (pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
    r = m[(m.dog_id == dog) & (m.race_date == day)]
    return (r[r.race_time == race_time] if race_time else r).iloc[0]


def test_a_best_just_under_the_second_and_one_just_over(card):
    m, _ = card
    one, two = row(m, 1, 8, "18:00"), row(m, 2, 8)
    assert one.op_best == pytest.approx(29.95) and two.op_best == pytest.approx(30.02)
    assert (one.op_round_under, one.op_round_over) == (1.0, 0.0) and one.op_round_dist == pytest.approx(-0.05)
    assert (two.op_round_under, two.op_round_over) == (0.0, 1.0) and two.op_round_dist == pytest.approx(0.02)
    others = np.mean([30.02, 30.38, 30.46, 30.54, 30.62])
    assert one.op_best_vs == pytest.approx((others - 29.95) / L, abs=1e-3) and one.op_fastest == 1 and two.op_fastest == 0
    seven = row(m, 7, 8)
    assert (seven.op_round_under, seven.op_round_over) == (0.0, 0.0) and seven.op_round_dist == pytest.approx(0.38)


def test_the_card_is_six_lines_at_todays_track_and_trip(card):
    m, _ = card
    # the seventh day reads the first six, 29.50 among them (the 285 m line is not a time over 500 m)
    assert row(m, 1, 6).op_best == pytest.approx(29.50) and row(m, 1, 6).op_round_under == 0
    # the eighth day reads days 2 to 7: the 29.50 has scrolled off
    assert row(m, 1, 7).op_best == pytest.approx(29.95)
    # over 285 m no dog has a line at the trip: no best, no pick, and the look is the form figures and the wins
    short = row(m, 1, 5)
    assert np.isnan(short.op_best) and np.isnan(short.op_best_vs) and np.isnan(short.op_fastest)
    assert np.isnan(short.op_round_under) and np.isnan(short.op_round_dist)
    assert short.op_wins6 == 5 and short.op_pos_sum == 6 and short.op_look > 0
    # form figures: fewer than three lines is no total; four lines are counted at the rate of six
    assert np.isnan(row(m, 2, 2).op_pos_sum) and row(m, 2, 3).op_pos_sum == pytest.approx(12.0)
    assert row(m, 2, 8).op_pos_sum == pytest.approx(12.0) and row(m, 10, 8).op_pos_sum == pytest.approx(36.0)
    assert row(m, 2, 8).op_pos_last == 2 and row(m, 2, 8).op_wins6 == 0 and row(m, 2, 8).op_lto_won == 0


def test_a_run_of_short_head_wins_is_photos_and_luck(card):
    m, notes = card
    three, four, five, six = (row(m, k, 8) for k in (3, 4, 5, 6))
    assert three.op_wins6 == 6 and three.op_lto_won == 1 and three.op_pos_last == 1 and three.op_pos_sum == 6
    assert three.op_photo6 == 6 and four.op_photo6 == 6             # the winner and the dog it beat a short head
    assert five.op_photo6 == 0 and six.op_photo6 == 0               # 2.5 lengths back, and 1.5 behind that
    # the expected-time table had dog 3 the slowest of the six and it won every time: five places of luck
    assert three.op_luck6 == pytest.approx(5.0) and four.op_luck6 == pytest.approx(3.0 - 2.0)
    assert three.op_real < 0 and three.op_look > 0 and three.op_surplus > 2.0
    # the void race of the fifth day is no line: on the sixth the card still shows four runs, and on the ninth six
    assert row(m, 3, 5).op_wins6 == 4 and row(m, 3, 5).op_photo6 == 4 and row(m, 3, 4).op_wins6 == 4
    assert notes["lines_settled_by_a_photo"] > 0.1 and notes["lines"] == int((m.position >= 1).sum()) - 1


def test_a_dead_heat_and_an_empty_distance():
    rc = np.array([0, 0, 0, 0, 1, 1, 1])
    pos = np.array([1.0, 1.0, 3.0, 4.0, 1.0, 2.0, np.nan])
    beaten = np.array([0.0, 0.0, 2.0, 0.1, 0.0, np.nan, np.nan])
    f = optics.photo_flags(rc, pos, beaten)
    assert list(f[:6]) == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0] and np.isnan(f[6])
    z = optics.race_z(np.array([1.0, 3.0, np.nan, 2.0, 2.0]), np.array([0, 0, 0, 1, 1]), 2)
    assert list(z[:2]) == [-1.0, 1.0] and np.isnan(z[2]) and list(z[3:]) == [0.0, 0.0]


def test_a_second_run_the_same_day_reads_what_the_first_did(card):
    m, _ = card
    a, b = row(m, 1, 8, "18:00"), row(m, 1, 8, "21:00")
    for c in ("op_best", "op_pos_last", "op_pos_sum", "op_wins6", "op_lto_won", "op_luck6", "op_photo6",
              "op_round_under", "op_round_over", "op_round_dist"):
        assert a[c] == b[c], c
    assert b.op_wins6 == 6 and b.op_lto_won == 1                     # the win at 18:00 today is not on the card


def test_a_dog_with_no_line_has_no_look(card):
    m, _ = card
    new = m[m.dog_id.isin([13, 14, 15, 16, 17])]
    look = [c for c in optics.COLUMNS if c != "op_real"]
    assert len(new) == 5 and new[look].isna().all().all() and new.op_real.notna().all()
    one = row(m, 1, 8, "21:00")                                      # the only runner with a card: its z-scores are 0
    assert np.isnan(one.op_best_vs) and one.op_fastest == 1 and one.op_look == 0 and one.op_surplus == 0


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=40)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = optics.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = optics.build(r2, core.per_run_figures(r2))[0]
    cols = list(optics.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).mean() > 0.8 and np.allclose(x, z, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_hand_card_does_not_move_either(change):
    runs = hand()
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = optics.build(runs, core.per_run_figures(runs), {"xtime": flat_xt(runs)})[0]
    r2 = change(runs)
    b = optics.build(r2, core.per_run_figures(r2), {"xtime": flat_xt(r2)})[0]
    cols = list(optics.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert len(z) == len(keys) == 18 and np.allclose(x, z, atol=1e-5, equal_nan=True)


# ---- the review's tests (11 Nov 2026): a middle day, the future, and a slow reading of the specification

def scatter(days=30, seed=5):
    """The pit suite's card: two tracks, 80 dogs drawn afresh for every race (so a dog often runs twice a day, at
    one track or both), a trial a week; with the distances beaten drawn from GBGB's short ones and long ones."""
    runs = pit_synth(days=days, seed=seed)
    rng = np.random.default_rng(seed)
    runs["beaten_lengths"] = np.where(runs.position == 1, 0.0, rng.choice([0.05, 0.1, 0.5, 1.0, 2.5], len(runs)))
    return runs


def perturb_day(runs, day):
    """One day's results rewritten, whichever day: positions reversed, times, distances, comments and SPs changed."""
    r = runs.copy()
    m = r.race_date == day
    r.loc[m, "position"] = 7 - r.loc[m, "position"]
    r.loc[m, "adjusted_time"] -= 0.7
    r.loc[m, "run_time"] -= 0.7
    r.loc[m, "sectional"] += 0.2
    r.loc[m, "beaten_lengths"] = np.where(r.loc[m, "position"] == 1, 0.0, 0.05)
    r.loc[m, "weight_kg"] += 1.0
    r.loc[m, "comment"] = "SAw,Bmp1,Wide"
    r.loc[m, "sp_decimal"] = 9.0
    r["won"] = (r.position == 1).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


def _table(runs, xt=None):
    figs = core.per_run_figures(runs)
    return runs[["race_date"]].join(optics.build(runs, figs, None if xt is None else {"xtime": xt})[0])


def test_a_middle_days_results_move_nothing_on_that_day_or_before(monkeypatch):
    """The last-day tests cannot see a feature that reads its own day and is then overwritten by nothing: here the
    day rewritten has 14 days after it, the expected-time table is rebuilt inside, and every row on the day or before
    must be untouched while the later rows (which read that day as a line) must move."""
    monkeypatch.setattr(xtime, "FIT_BEFORE", "2025-01-08")           # the constants' window is before the day
    runs = scatter()
    days = sorted(runs.race_date.unique())
    day = days[15]
    twice = runs[runs.race_date == day].dog_id.duplicated().sum()
    assert twice > 5                                                 # dogs with a second run on the day rewritten
    a, b = _table(runs), _table(perturb_day(runs, day))
    cols = list(optics.COLUMNS)
    upto = (a.race_date <= day).to_numpy()
    x, y = a[cols].to_numpy(float), b[cols].to_numpy(float)
    assert np.isfinite(x[(a.race_date == day).to_numpy()]).mean() > 0.8
    assert np.allclose(x[upto], y[upto], atol=1e-5, equal_nan=True)
    moved = pd.DataFrame(~np.isclose(x[~upto], y[~upto], atol=1e-5, equal_nan=True), columns=cols).sum()
    assert (moved > 0).all(), moved.to_dict()                        # the test has teeth: every column reads results


@pytest.mark.parametrize("rebuild", [False, True])
def test_the_days_after_move_nothing(rebuild, monkeypatch):
    """No statistic is fitted on the whole table and applied to earlier rows: the table built from the days up to a
    day is the full table's rows up to that day, with the expected-time table handed in and with it rebuilt from the
    shorter frame. The expected-time table chooses its three filter constants on the days before
    ``xtime.FIT_BEFORE`` and, where a store starts later, on the first fifth of whatever frame it is given
    (greyhound/pit/xtime.py ``build``), so that window is pinned here: without the pin the shorter frame has a
    different first fifth and can price every row with other constants (the review, 11 Nov 2026)."""
    monkeypatch.setattr(xtime, "FIT_BEFORE", "2025-01-08")
    runs = scatter()
    day = sorted(runs.race_date.unique())[15]
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = runs[["race_date"]].join(optics.build(runs, figs, {"xtime": xt})[0])
    b = _table(runs[runs.race_date <= day], None if rebuild else xt)
    cols = list(optics.COLUMNS)
    assert len(b) == int((a.race_date <= day).sum()) and b[cols].notna().mean().min() > 0.6
    assert np.allclose(a.loc[a.race_date <= day, cols].to_numpy(float), b[cols].to_numpy(float), atol=1e-5,
                       equal_nan=True)


def _z(v):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    out = np.full(len(v), np.nan)
    if ok.any():
        sd = v[ok].std()
        out[ok] = (v[ok] - v[ok].mean()) / sd if sd > 1e-9 else 0.0
    return out


def slow_reading(runs, xt, races):
    """The specification read a row at a time, sharing no code with the module: the dog's last six earlier days with
    a position (a day's last run), the photo of each line from its race's finishing order, the luck from the rank of
    xt_tot in the line's race, then today's race from those."""
    r = runs.merge(xt[["race_id", "dog_id", "xt_tot"]].drop_duplicates(["race_id", "dog_id"]), on=["race_id", "dog_id"],
                   how="left")
    assert len(r) == len(runs)
    r["pos"] = r.position.where(r.position >= 1)
    r["erank"] = r.groupby("race_id").xt_tot.rank(ascending=False, method="average")
    by_race = {k: g for k, g in r.groupby("race_id")}
    by_dog = {k: g for k, g in r.groupby("dog_id")}

    def photo(line):
        g = by_race[line.race_id]
        g = g[g.pos.notna()].sort_values("pos", kind="stable")
        i = list(g.index).index(line.Index)
        own = line.pos > 1 and line.beaten_lengths <= optics.PHOTO
        behind = i + 1 < len(g) and g.beaten_lengths.iloc[i + 1] <= optics.PHOTO
        return float(own or behind or (g.pos == line.pos).sum() > 1)

    out = []
    for race in races:
        g, rec = by_race[race], []
        for a in g.itertuples():
            h = by_dog[a.dog_id]
            h = h[(h.race_date < a.race_date) & h.pos.notna()].sort_values("t", kind="stable")
            h = h.groupby("race_date").tail(1).tail(optics.N_CARD)
            here = h[(h.track == a.track) & (h.distance_m == a.distance_m)]
            best = here.adjusted_time.min() if len(here) else np.nan
            hund = np.rint(best * 100) % 100
            luck = (h.erank - h.pos).dropna()
            some = len(h) > 0
            rec.append({"op_best": best, "op_pos_last": h.pos.iloc[-1] if some else np.nan,
                        "op_pos_sum": 6 * h.pos.mean() if len(h) >= optics.MIN_POS else np.nan,
                        "op_wins6": float((h.pos == 1).sum()) if some else np.nan,
                        "op_lto_won": float(h.pos.iloc[-1] == 1) if some else np.nan,
                        "op_round_under": float(hund >= 90) if np.isfinite(best) else np.nan,
                        "op_round_over": float(hund <= 9) if np.isfinite(best) else np.nan,
                        "op_round_dist": (hund - 100 if hund >= 50 else hund) / 100 if np.isfinite(best) else np.nan,
                        "op_luck6": luck.mean() if len(luck) else np.nan,
                        "op_photo6": float(sum(photo(x) for x in h.itertuples())) if some else np.nan,
                        "xt": a.xt_tot})
        b = pd.DataFrame(rec, index=g.index)
        k = b.op_best.notna()
        b["op_best_vs"] = [(b.op_best[k].drop(i).mean() - b.op_best[i]) / L if k[i] and k.sum() > 1 else np.nan
                           for i in b.index]
        b["op_fastest"] = [float(b.op_best[i] <= b.op_best.min() + 1e-9) if k[i] else np.nan for i in b.index]
        z = np.vstack([_z(b.op_best_vs), _z(-b.op_pos_sum), _z(b.op_wins6)])
        parts = np.isfinite(z).sum(axis=0)
        b["op_look"] = np.where(parts > 0, np.nansum(z, axis=0) / np.maximum(parts, 1), np.nan)
        b["op_real"] = _z(b.xt)
        b["op_surplus"] = b.op_look - b.op_real
        out.append(b.assign(race_id=g.race_id, dog_id=g.dog_id))
    return pd.concat(out)


def test_the_table_is_the_specification_read_a_row_at_a_time():
    runs = scatter()
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    tab = optics.build(runs, figs, {"xtime": xt})[0]
    late = runs[runs.race_date >= sorted(runs.race_date.unique())[20]]
    races = late.race_id.drop_duplicates().iloc[::3].tolist()        # 40 races, both tracks, the weekly trial among them
    want = slow_reading(runs, xt, races)
    got = want[["race_id", "dog_id"]].merge(tab, on=["race_id", "dog_id"], how="left")
    assert len(got) == len(want) == 6 * len(races) and len(races) == 40
    # the sample has what the construction must get right: second runs the same day, both tracks, photos, no best
    day_runs = late[late.race_id.isin(races)].merge(runs[["dog_id", "race_date", "race_id"]], on=["dog_id", "race_date"])
    assert (day_runs.race_id_x != day_runs.race_id_y).sum() > 20
    assert want.op_best.isna().sum() > 0 and want.op_photo6.max() >= 4 and want.op_luck6.abs().max() > 1.0
    for c in optics.COLUMNS:
        assert np.allclose(got[c].to_numpy(float), want[c].to_numpy(float), atol=2e-3, equal_nan=True), c
        assert want[c].notna().mean() > 0.6, c


def test_trap_numbers_are_not_read():
    """A card with traps numbered 7 to 12 (a reserve's number, a hurdle's): the table is the same."""
    runs = scatter(days=20)
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    a = optics.build(runs, figs, {"xtime": xt})[0]
    r2 = runs.assign(trap=runs.trap + 6)
    b = optics.build(r2, core.per_run_figures(r2), {"xtime": xt})[0]
    cols = list(optics.COLUMNS)
    assert (a.race_id.to_numpy() == b.race_id.to_numpy()).all() and (a.dog_id.to_numpy() == b.dog_id.to_numpy()).all()
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)


def test_an_empty_distance_in_the_text_is_no_photo(card):
    """sources/gbgb.py reads an empty distance as 0.0 lengths: with GBGB's text beside it, a non-winner's empty
    distance is unknown, not a dead heat, and neither it nor the dog in front is given a photo for it."""
    m, _ = card
    runs = hand()
    text = runs.beaten_lengths.map({0.0: "", 0.05: "SH", 0.75: "3/4", 1.0: "1", 1.5: "1 1/2", 2.5: "2 1/2",
                                    4.5: "4 1/2"})
    lost = (runs.dog_id == 4) & (runs.position == 2)                 # the short head, lost in GBGB's text
    runs["beaten"] = text.where(~lost, "")
    runs.loc[lost, "beaten_lengths"] = 0.0
    tab = optics.build(runs, core.per_run_figures(runs), {"xtime": flat_xt(runs)})
    assert tab[1]["non_winners_with_no_distance"] == int(lost.sum()) == 8
    t = runs.merge(tab[0], on=["race_id", "dog_id"])
    assert row(m, 3, 8).op_photo6 == 6 and row(m, 4, 8).op_photo6 == 6
    assert row(t, 3, 8).op_photo6 == 0 and row(t, 4, 8).op_photo6 == 0 and row(t, 5, 8).op_photo6 == 0
    with_text = runs.assign(beaten=text)                              # the text kept: the photos are as before
    t2 = with_text.merge(optics.build(with_text, core.per_run_figures(with_text), {"xtime": flat_xt(runs)})[0],
                         on=["race_id", "dog_id"])
    assert np.allclose(t2[list(optics.COLUMNS)].to_numpy(float), m[list(optics.COLUMNS)].to_numpy(float), equal_nan=True)


def test_a_time_no_greyhound_runs_is_no_best_time(card):
    """GBGB's keying errors (the real store, 1 Jul 2025 to 5 Nov 2026: a going allowance of 514 s at Valley, of 20 s
    at Yarmouth, a 238 m time under a 415 m race at Harlow) would make the dog the time-finder's pick by 150 lengths,
    or its rivals by 1,600. Such a time is not read; the line's position still is."""
    m, _ = card
    runs = hand()
    day = (pd.Timestamp("2025-03-01") + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    slow, fast = (runs.race_date == day) & (runs.dog_id == 2), (runs.race_date == day) & (runs.dog_id == 7)
    runs.loc[slow, "adjusted_time"] = 544.34                         # 30.02 with 514.32 keyed as the going
    runs.loc[fast, "adjusted_time"] = 16.20                          # a sprint's time under a 500 m race
    tab, notes = optics.build(runs, core.per_run_figures(runs), {"xtime": flat_xt(runs)})
    t = runs.merge(tab, on=["race_id", "dog_id"])
    assert notes["times_no_greyhound_runs_left_out"] == 2
    cols = [c for c in optics.COLUMNS]
    after = (t.race_date > day).to_numpy()
    assert np.allclose(t.loc[after, cols].to_numpy(float), m.loc[after, cols].to_numpy(float), equal_nan=True)
    assert row(t, 2, 8).op_best == pytest.approx(30.02) and row(t, 7, 8).op_best == pytest.approx(30.38)
    assert row(t, 1, 8, "18:00").op_fastest == 1 and row(t, 7, 8).op_fastest == 0
    assert optics.PACE[0] * 500 < 29.5 and optics.PACE[1] * 500 > 31.0 and optics.PACE[1] * 285 > 18.0
