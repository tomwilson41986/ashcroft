"""greyhound/pit/clock.py: the body clock, the wait in the racing kennels and the calendar of noise. A dog that runs
worse the later it races in the card has its slope found and signed; a dog moved from afternoons to an evening is new
to the session and to the dark and hours off its habit; the clock-change week, a 5 November evening and sunset are
right against values derived here; the meeting is read from the off times, so a card's day-wide meeting id changes
nothing; the day's own results move nothing and a card row gets what its run had."""
from __future__ import annotations

import calendar as pycal
import math

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import clock, core, dyad, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S
START = "2025-08-01"
AFTERNOON, EVENING = 14 * 60 + 8, 18 * 60 + 30        # the two meetings' first offs
STEP = 20                                             # minutes between races
SLOPE = 3.0                                           # lengths an hour of wait, planted on dog 1
MOVE_DAY = 100                                        # 9 Nov 2025: dog 40 to the evening, dog 30 to the afternoon
HOVE = (50.8422, -0.17595)                            # research/greyhound/reference/gb_tracks.csv


def synth(days=120, seed=11, twice=None, track="Hove"):
    """One track, two meetings a day of six races of six, 20 minutes apart: dogs 1-36 race in the evening meeting
    (first off 18:30), dogs 37-72 in the afternoon's (14:08), in a random race each day. Dog 1 runs ``SLOPE`` lengths
    slower for every hour after its meeting's first race. From day ``MOVE_DAY`` dog 40 races in the evening and dog 30
    in the afternoon. Dog 72 runs every other day (a five-runner race) and dog 71 not before day 20 (a debutant).
    ``twice``: a dog that on the last day also runs in the evening's last race, in trap 6's place."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 5000
    for d in range(days):
        day = (pd.Timestamp(START) + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        eve, aft = list(range(1, 37)), list(range(37, 73))
        if d >= MOVE_DAY:
            eve[eve.index(30)], aft[aft.index(40)] = 40, 30
        for mi, (first, dogs) in enumerate(((AFTERNOON, aft), (EVENING, eve))):
            order = rng.permutation(dogs)
            for r in range(6):
                rid += 1
                field = [int(x) for x in order[6 * r:6 * r + 6]]
                if twice is not None and d == days - 1 and mi == 1 and r == 5:
                    field[5] = twice
                wait = STEP * r
                t = 29.5 + rng.normal(0, 0.05, 6)
                t = t + np.array([SLOPE * wait / 60.0 * L if dog == 1 else 0.0 for dog in field])
                keep = [k for k, dog in enumerate(field) if not (dog == 72 and d % 2) and not (dog == 71 and d < 20)]
                pos = np.argsort(np.argsort(t[keep])) + 1
                for p, k in zip(pos, keep):
                    dog = field[k]
                    rows.append({"race_date": day, "race_time": f"{(first + wait) // 60:02d}:{(first + wait) % 60:02d}",
                                 "meeting_id": 2 * d + mi, "track": track, "race_id": rid, "race_number": r + 1,
                                 "race_class": "A3", "distance_m": 500.0, "trap": k + 1, "dog_id": dog,
                                 "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d", "sp_decimal": 6.0,
                                 "position": int(p), "beaten_lengths": 0.5, "sectional": 4.5, "run_time": float(t[k]),
                                 "adjusted_time": float(t[k]), "weight_kg": 30.0, "comment": "QAw"})
    return clean(pd.DataFrame(rows))


def frame(runs, tab):
    m = runs.merge(tab, on=["race_id", "dog_id"])
    m["minutes"] = m.race_time.str[:2].astype(int) * 60 + m.race_time.str[3:5].astype(int)
    return m


@pytest.fixture(scope="module")
def built():
    runs = synth()
    figs = core.per_run_figures(runs)
    xt, _ = xtime.build(runs, figs)
    tab, notes = clock.build(runs, figs, {"xtime": xt})
    return runs, figs, xt, tab, frame(runs, tab), notes


def own_sunset(day: pd.Timestamp, lat: float, lon: float, summer_time: bool, fine: bool = True) -> float:
    """This test's own astronomy, in UK clock minutes: the sun's declination 23.44 sin(2 pi (N - 80) / 365) degrees,
    setting at the hour angle w with cos w = -tan(latitude) tan(declination), w / 15 hours after local solar noon,
    which is 12:00 less four minutes a degree of longitude east. ``fine`` adds the two small terms: the sun's upper
    edge seen 0.833 degrees below the horizon (0.833 / (cos latitude cos declination sin w) degrees of hour angle
    later) and the equation of time by its two-term formula on the mean anomaly (the sundial ahead of the clock by
    -7.659 sin M + 9.863 sin(2 M + 3.5932) minutes, M = 6.24004077 + 0.01720197 x days since noon on 1 Jan 2000)."""
    decl = math.radians(23.44) * math.sin(2 * math.pi * (day.dayofyear - 80) / 365.0)
    phi = math.radians(lat)
    w = math.degrees(math.acos(-math.tan(phi) * math.tan(decl)))
    t = 720.0 + 4.0 * w - 4.0 * lon + (60.0 if summer_time else 0.0)
    if fine:
        t += 4.0 * 0.833 / (math.cos(phi) * math.cos(decl) * math.sin(math.radians(w)))
        mean_anomaly = 6.24004077 + 0.01720197 * ((day - pd.Timestamp("2000-01-01")).days - 0.5)
        t -= -7.659 * math.sin(mean_anomaly) + 9.863 * math.sin(2 * mean_anomaly + 3.5932)
    return t


def last_sunday(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month,
                        day=max(w[pycal.SUNDAY] for w in pycal.monthcalendar(year, month)))


def test_every_column_is_registered_and_one_row_a_run(built):
    runs, _, _, tab, m, notes = built
    assert [c for c in tab.columns if c.startswith(clock.PREFIX)] == list(clock.COLUMNS)
    assert len(tab) == len(runs) == len(m) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert all(tab[c].dtype == np.float32 for c in clock.COLUMNS)
    assert notes["rows"] == len(runs) and notes["meetings_by_off_time"] == 240 == notes["meetings_by_meeting_id"]
    assert notes["rows_first_off_differs_from_meeting_id"] == 0 and notes["tracks_without_coordinates"] == []
    assert (runs.groupby("race_id").dog_id.size() == 5).any()                     # the five-runner races are in


def test_the_wait_and_the_slot_are_the_meetings_own(built):
    _, _, _, _, m, _ = built
    first = np.where(m.minutes < 17 * 60, AFTERNOON, EVENING)
    assert np.allclose(m.ck_wait, m.minutes - first)                # not counted from the afternoon's first race
    assert np.allclose(m.ck_slot, m.race_number / 6.0)
    assert set(m.ck_wait.unique()) == {0.0, 20.0, 40.0, 60.0, 80.0, 100.0}


def test_the_slope_of_the_dog_that_tires_of_waiting_is_found_and_signed(built):
    _, figs, xt, _, m, _ = built
    late = m[m.race_date >= "2025-10-15"]
    one, rest = late[late.dog_id == 1], late[(late.dog_id > 1) & (late.dog_id <= 36)]
    # 3 lengths an hour planted; 75 to 119 earlier runs over waits of 0 to 100 minutes shrink it by 0.83 to 0.89
    assert 1.8 < one.ck_wait_slope.mean() < 3.3 and one.ck_wait_slope.min() > 1.2
    assert rest.groupby("dog_id").ck_wait_slope.mean().abs().max() < 1.0
    assert abs(rest.ck_wait_slope.mean()) < 0.2
    # the lengths today's wait costs: above 0 in the last race, below 0 in the first, and 100 minutes of slope apart
    hi, lo = one[one.ck_wait == 100].ck_wait_pred, one[one.ck_wait == 0].ck_wait_pred
    assert len(hi) and len(lo) and hi.min() > 1.0 and lo.max() < -1.0
    assert abs((hi.mean() - lo.mean()) - one.ck_wait_slope.mean() * 100 / 60.0) < 0.4
    # the construction, by hand, at dog 1's last run: its earlier runs' residuals on their waits
    j = figs[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_tot_time"]], on=["race_id", "dog_id"], how="left")
    res = dyad.residuals(figs, pd.Series(j.xt_tot_time.to_numpy(), index=figs.index))
    h = m.assign(r=res.to_numpy())
    last = h[h.dog_id == 1].sort_values("race_date").iloc[-1]
    e = h[(h.dog_id == 1) & (h.race_date < last.race_date) & h.r.notna()]
    w, r = e.ck_wait.to_numpy(float), e.r.to_numpy(float)
    b = ((w - w.mean()) * r).sum() / (((w - w.mean()) ** 2).sum() + 3600.0 * 5)
    assert len(e) > 100 and abs(last.ck_wait_slope - 60 * b) < 1e-3
    assert abs(last.ck_wait_pred - b * (last.ck_wait - w.mean())) < 1e-3
    # the planted effect is in the residuals the slope reads: dog 1's later runs are slower than expected
    assert np.corrcoef(w, r)[0, 1] > 0.6


def test_a_dog_moved_from_afternoons_to_an_evening(built):
    _, _, _, _, m, _ = built
    move = (pd.Timestamp(START) + pd.Timedelta(days=MOVE_DAY)).strftime("%Y-%m-%d")
    d40 = m[m.dog_id == 40].sort_values("race_date")
    before, at, after = d40[d40.race_date < move], d40[d40.race_date == move].iloc[0], d40[d40.race_date > move]
    assert (before.ck_session == 1).all() and (before.ck_session_new == 0).all()
    assert before.ck_habit_abs.iloc[1:].max() <= 100 / 60.0 + 1e-6               # inside its own meeting's span
    assert at.ck_session == 2 and at.ck_session_new == 1
    assert 2.5 < at.ck_habit_gap < 6.1 and at.ck_habit_abs == pytest.approx(at.ck_habit_gap)
    assert at.ck_dark == 1 and at.ck_dark_share == 0 and at.ck_dark_new == 1     # 9 November: its first night race
    assert (after.ck_session_new == 0).all()                                     # one evening is on its last six days
    assert after.ck_dark_new.iloc[0] == pytest.approx(1 - 1 / (MOVE_DAY + 1), abs=1e-4)
    assert after.ck_habit_gap.iloc[-1] < at.ck_habit_gap - 0.5                   # the habit follows the new hour
    d30 = m[(m.dog_id == 30) & (m.race_date == move)].iloc[0]                    # the evening dog sent to afternoons
    assert d30.ck_session == 1 and d30.ck_session_new == 1 and d30.ck_habit_gap < -2.5 and d30.ck_dark_new == 0
    # the habit by hand: on a dog's second day it is the first day's off time, and its weight one day of decay
    d2 = m[m.dog_id == 2].sort_values("race_date")
    assert d2.ck_habit_gap.iloc[1] == pytest.approx((d2.minutes.iloc[1] - d2.minutes.iloc[0]) / 60.0, abs=1e-5)
    assert d2.ck_habit_n.iloc[1] == pytest.approx(0.5 ** (1 / clock.HALF_LIFE), abs=1e-5)
    # the dogs that never move are never new to their session
    others = m[(m.dog_id > 1) & (m.dog_id <= 29) & (m.race_date > "2025-08-01")]
    assert (others.ck_session_new == 0).all() and (others.ck_session == 2).all()


def test_the_clock_change_weeks(built):
    _, _, _, _, m, _ = built
    change = last_sunday(2025, 10)
    assert change == pd.Timestamp("2025-10-26") and clock.clock_changes(2025)[1] == change
    for year in (2014, 2021, 2025, 2026, 2027):
        assert clock.clock_changes(year) == (last_sunday(year, 3), last_sunday(year, 10))
    off = (pd.to_datetime(m.race_date) - change).dt.days
    assert (m.ck_dst[(off >= 0) & (off <= 6)] == 1).all() and (m.ck_dst[(off >= -7) & (off <= -1)] == -1).all()
    assert (m.ck_dst[(off < -7) | (off > 6)] == 0).all()
    assert sorted(m.ck_dst.unique()) == [-1.0, 0.0, 1.0]
    spring = clock.calendar(pd.date_range("2026-03-20", "2026-04-07"))            # 29 March 2026
    assert list(spring.dst) == [0.0, 0.0] + [-1.0] * 7 + [1.0] * 7 + [0.0, 0.0, 0.0]
    assert list(spring.bst) == [False] * 9 + [True] * 10


def test_a_fifth_of_november_evening(built):
    _, _, _, _, m, _ = built
    on = lambda d: m[m.race_date == d]                                            # noqa: E731
    night = on("2025-11-05")
    assert (night.ck_fireworks[night.minutes >= 18 * 60] == 1).all()
    assert (night.ck_fireworks[night.minutes < 18 * 60] == 0).all() and (night.ck_days_since_fw == 0).all()
    assert (on("2025-11-04").ck_days_since_fw == 308).all() and (on("2025-11-06").ck_days_since_fw == 1).all()
    for d, flag in (("2025-11-01", 0), ("2025-11-02", 1), ("2025-11-08", 1), ("2025-11-09", 0), ("2025-10-05", 0)):
        assert (on(d).ck_fireworks[on(d).minutes >= 18 * 60] == flag).all(), d
    assert (m.ck_fireworks[m.minutes < 18 * 60] == 0).all()
    cal = clock.calendar(pd.to_datetime(["2025-12-29", "2025-12-30", "2025-12-31", "2026-01-01", "2026-01-02",
                                         "2026-11-04", "2026-11-05"]))
    assert list(cal.fw_date) == [False, True, True, True, False, True, True]
    assert list(cal.since_fw) == [54.0, 55.0, 0.0, 1.0, 2.0, 308.0, 0.0]


def test_sunset_against_this_tests_own_astronomy(built):
    _, _, _, _, m, _ = built
    # an equinox: the sun is over the equator and sets six hours after local solar noon, at any latitude (the
    # sinusoid's declination is a degree off zero on the day: a few minutes)
    eq = pd.Timestamp("2025-09-22")
    assert abs(own_sunset(eq, *HOVE, True, fine=False) - (19 * 60 + 4 * 0.17595)) < 8
    assert abs(float(clock.sunset_minutes(eq.dayofyear, *HOVE, True)) - (19 * 60 + 4 * 0.17595)) < 15
    cases = [(eq, HOVE[0], HOVE[1], True), (pd.Timestamp("2025-06-21"), 52.0, 0.0, True),
             (pd.Timestamp("2025-12-21"), 52.0, 0.0, False), (pd.Timestamp("2025-11-05"), 54.94, -1.41, False),
             (pd.Timestamp("2026-03-01"), 51.57, 0.165, False)]
    for day, lat, lon, summer in cases:
        got = float(clock.sunset_minutes(day.dayofyear, lat, lon, summer))
        assert abs(got - own_sunset(day, lat, lon, summer)) < 6, (day, got, own_sunset(day, lat, lon, summer))
        assert abs(got - own_sunset(day, lat, lon, summer, fine=False)) < 20       # the bare geometry: the same hour
    # midsummer at 52 north: the sun sets after 21:00 summer time; midwinter: before 16:00
    assert 21 * 60 < float(clock.sunset_minutes(172, 52.0, 0.0, True)) < 21 * 60 + 40
    assert 15 * 60 + 30 < float(clock.sunset_minutes(355, 52.0, 0.0, False)) < 16 * 60
    # the table: every race more than 20 minutes from this test's own sunset is on the right side of it
    days = pd.to_datetime(m.race_date)
    summer = (days < last_sunday(2025, 10)).to_numpy()
    own = np.array([own_sunset(d, HOVE[0], HOVE[1], s) for d, s in zip(days, summer)])
    far = np.abs(m.minutes.to_numpy() - own) > 20
    assert far.mean() > 0.85 and (m.ck_dark.to_numpy()[far] == (m.minutes.to_numpy()[far] > own[far])).all()
    assert (m.ck_dark[m.race_date == "2025-08-01"] == 0).all()                    # sunset near 20:50
    nov = m[m.race_date == "2025-11-20"]
    assert (nov.ck_dark[nov.minutes >= EVENING] == 1).all() and (nov.ck_dark[nov.minutes < 16 * 60] == 0).all()
    # the share of a dog's earlier race days in the dark rises through the autumn
    d5 = m[m.dog_id == 5].sort_values("race_date")
    assert d5.ck_dark_share.iloc[1] == 0 and d5.ck_dark_share.iloc[-1] > 0.4
    assert np.allclose(d5.ck_dark_new.iloc[1:], d5.ck_dark.iloc[1:] * (1 - d5.ck_dark_share.iloc[1:]), atol=1e-6)


def test_a_dog_with_no_history(built):
    _, _, _, _, m, _ = built
    first = m[m.race_date == START]
    assert first.ck_habit_gap.isna().all() and (first.ck_habit_n == 0).all() and first.ck_wait_slope.isna().all()
    assert first.ck_wait_pred.isna().all() and (first.ck_session_new == 0).all() and first.ck_dark_share.isna().all()
    assert first[["ck_wait", "ck_slot", "ck_session", "ck_dst", "ck_dark", "ck_fireworks",
                  "ck_days_since_fw"]].notna().all().all()
    deb = m[m.dog_id == 71].sort_values("race_date").iloc[0]                      # the debutant of day 20
    assert deb.race_date > START and np.isnan(deb.ck_habit_gap) and deb.ck_habit_n == 0 and deb.ck_session_new == 0


def test_a_cards_day_wide_meeting_id_and_numbering_change_nothing(built):
    runs, figs, xt, tab, _, _ = built
    cols = list(clock.COLUMNS)
    # greyhound/card.py: one meeting id a track's day, races numbered by off time through both meetings
    lump = figs.copy()
    lump["meeting_id"] = -(pd.factorize(lump.race_date)[0] + 1)
    lump["race_number"] = lump.groupby("race_date").race_time.rank(method="dense")
    assert lump.race_number.max() == 12
    a = clock.build(runs, lump, {"xtime": xt})[0]
    assert np.allclose(a[cols].to_numpy(float), tab[cols].to_numpy(float), equal_nan=True)
    # no race number: the race's rank by off time in its meeting
    b, notes = clock.build(runs, figs.drop(columns="race_number"), {"xtime": xt})
    assert np.allclose(b[cols].to_numpy(float), tab[cols].to_numpy(float), equal_nan=True)
    assert notes["rows_slot_from_race_number"] == 0


def test_the_meeting_is_split_at_a_long_gap_only():
    tk, dn = np.zeros(7, dtype=int), np.array([1, 1, 1, 1, 1, 2, 2])
    mi = np.array([600.0, 618.0, 700.0, 821.0, 840.0, 600.0, np.nan])           # 121 minutes begins a new meeting
    rn = np.array([1.0, 2.0, 4.0, 5.0, 6.0, np.nan, np.nan])
    mt = clock.meetings(tk, dn, mi, rn)
    assert list(mt["first"]) == [600.0, 600.0, 600.0, 821.0, 821.0, 600.0, 600.0]
    assert mt.meeting.nunique() == 3
    # race 3 was void: the numbers stand in the first meeting; the second is numbered from its own first race
    assert np.allclose(mt.slot[:6], [1 / 4, 2 / 4, 4 / 4, 1 / 2, 2 / 2, 1.0]) and np.isnan(mt.slot[6])
    # the input's order is kept whatever it is
    p = np.array([4, 0, 6, 2, 5, 1, 3])
    mp = clock.meetings(tk[p], dn[p], mi[p], rn[p])
    assert np.allclose(mp["first"].to_numpy(), mt["first"].to_numpy()[p])
    assert np.allclose(mp.slot.to_numpy(), mt.slot.to_numpy()[p], equal_nan=True)


def test_earlier_days_only():
    dog = np.array([1, 1, 1, 2, 1, 1])
    dn = np.array([5, 5, 6, 5, 7, 9])
    v = np.array([[1.0], [0.0], [1.0], [1.0], [np.nan], [1.0]])
    s, c, days = clock.earlier_days(dog, dn, v, None)
    # the day's own rows, both of them, are left out; day 5 counts once, at its mean; the day with no value not at all
    assert list(s[:, 0]) == [0.0, 0.0, 0.5, 0.0, 1.5, 1.5] and list(c[:, 0]) == [0.0, 0.0, 1.0, 0.0, 2.0, 2.0]
    assert list(days) == [0.0, 0.0, 1.0, 0.0, 2.0, 3.0]
    s2, _, d2 = clock.earlier_days(dog, dn, v, 2)
    assert list(s2[:, 0]) == [0.0, 0.0, 0.5, 0.0, 1.5, 1.0] and list(d2) == [0.0, 0.0, 1.0, 0.0, 2.0, 2.0]


def test_a_run_earlier_the_same_day_moves_nothing():
    runs = synth(days=30, twice=37)                       # dog 37 runs in the afternoon and again that evening
    last = runs.race_date.max()
    both = runs[(runs.dog_id == 37) & (runs.race_date == last)].sort_values("race_time")
    assert len(both) == 2
    eve_race, aft_race = int(both.race_id.iloc[1]), int(both.race_id.iloc[0])
    a = clock.build(runs, core.per_run_figures(runs))[0]
    r2 = runs[~((runs.dog_id == 37) & (runs.race_id == aft_race))].reset_index(drop=True)
    b = clock.build(r2, core.per_run_figures(r2))[0]
    cols = list(clock.COLUMNS)
    x = a[(a.dog_id == 37) & (a.race_id == eve_race)][cols].to_numpy(float)
    z = b[(b.dog_id == 37) & (b.race_id == eve_race)][cols].to_numpy(float)
    assert x.shape == (1, len(cols)) and np.isfinite(x).all() and np.allclose(x, z, atol=1e-5)
    row = a[(a.dog_id == 37) & (a.race_id == eve_race)].iloc[0]
    assert row.ck_session == 2 and row.ck_session_new == 1 and row.ck_habit_gap > 2.5   # the afternoon is not read


def test_a_track_without_coordinates_has_no_darkness(built):
    runs, figs, xt, tab, _, _ = built
    f2 = figs.assign(track="Nowhere")
    b, notes = clock.build(runs.assign(track="Nowhere"), f2, {"xtime": xt})
    assert b[["ck_dark", "ck_dark_share", "ck_dark_new"]].isna().all().all()
    assert notes["tracks_without_coordinates"] == ["Nowhere"] and notes["races_dark_unknown"] == notes["races"]
    rest = [c for c in clock.COLUMNS if "dark" not in c]
    assert np.allclose(b[rest].to_numpy(float), tab[rest].to_numpy(float), equal_nan=True)


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    runs = synth(days=40)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = clock.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = clock.build(r2, core.per_run_figures(r2))[0]
    cols = list(clock.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert len(z) == len(keys) and np.isfinite(x).mean() > 0.8 and np.allclose(x, z, atol=1e-4, equal_nan=True)


def test_the_expected_time_table_is_taken_from_the_context(built):
    runs, figs, xt, tab, _, _ = built
    b = clock.build(runs, figs)[0]
    assert np.allclose(tab[list(clock.COLUMNS)].to_numpy(float), b[list(clock.COLUMNS)].to_numpy(float), equal_nan=True)


# ---- the review of 11 Nov 2026: leak tests that reach past the last day -------------------------------------------

RESULT_READERS = ["ck_wait_slope", "ck_wait_pred"]            # the only columns that read a result (of earlier days)


def perturb_from(runs, day):
    """The results of ``day`` and of every later day rewritten, as ``perturb_last_day`` rewrites the last day's
    (positions reversed, comments, SPs and weights changed), each time moved by its own random amount."""
    rng = np.random.default_rng(5)
    r = runs.copy()
    m = (r.race_date >= day).to_numpy()
    r.loc[m, "position"] = 7 - r.loc[m, "position"]
    r.loc[m, "adjusted_time"] += rng.normal(0.3, 0.2, int(m.sum()))
    r.loc[m, "run_time"] += rng.normal(0.3, 0.2, int(m.sum()))
    r.loc[m, "sectional"] += 0.2
    r.loc[m, "weight_kg"] += 1.5
    r.loc[m, "comment"] = "SAw,Bmp1,Wide"
    r.loc[m, "sp_decimal"] = 9.0
    r["won"] = (r.position == 1).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


def test_a_middle_days_results_move_nothing_on_that_day_or_before():
    runs = synth(days=40)
    day = sorted(runs.race_date.unique())[25]
    a = clock.build(runs, core.per_run_figures(runs))[0]
    r2 = perturb_from(runs, day)
    b = clock.build(r2, core.per_run_figures(r2))[0]
    assert (a.race_id.to_numpy() == b.race_id.to_numpy()).all() and (a.dog_id.to_numpy() == b.dog_id.to_numpy()).all()
    cols = list(clock.COLUMNS)
    upto, after = (runs.race_date <= day).to_numpy(), (runs.race_date > day).to_numpy()
    assert upto.sum() > 1500 and after.sum() > 800
    x, z = a[cols].to_numpy(float), b[cols].to_numpy(float)
    # the day itself and every day before it: nothing moves, to the last float32 bit
    assert np.array_equal(x[upto], z[upto], equal_nan=True)
    # the days after: the rewrite is seen (so the test above could have failed), and only by the wait slope
    moved = ~np.isclose(a.loc[after, "ck_wait_slope"], b.loc[after, "ck_wait_slope"], atol=1e-3, equal_nan=True)
    assert moved.mean() > 0.9
    rest = [c for c in cols if c not in RESULT_READERS]
    assert np.array_equal(a[rest].to_numpy(float), b[rest].to_numpy(float), equal_nan=True)


def test_the_days_after_might_as_well_not_exist(built):
    runs, figs, xt, tab, _, _ = built
    cols = list(clock.COLUMNS)
    day = sorted(runs.race_date.unique())[70]
    keep = (runs.race_date <= day).to_numpy()
    r2, f2 = runs[keep].reset_index(drop=True), figs[keep].reset_index(drop=True)
    # the expected times handed in are the full build's: on a short store xtime fits its constants on the first fifth
    # of the days it is given, so its own table is not the same on a shorter one
    b = clock.build(r2, f2, {"xtime": xt})[0]
    assert len(b) == keep.sum() and np.array_equal(b[cols].to_numpy(float), tab.loc[keep, cols].to_numpy(float),
                                                   equal_nan=True)


def test_the_days_later_races_move_only_the_slot():
    """GBGB's results hold only the races that were run: with a day's later races gone (a meeting abandoned after
    race 3) the races before them keep every feature but the slot, which is counted against the last race run."""
    runs = synth(days=30)
    last = runs.race_date == runs.race_date.max()
    a = clock.build(runs, core.per_run_figures(runs))[0]
    r2 = runs[~(last & (runs.race_number > 3))].reset_index(drop=True)
    b = clock.build(r2, core.per_run_figures(r2))[0]
    keys = runs.loc[last & (runs.race_number <= 3), ["race_id", "dog_id", "race_number"]]
    assert len(keys) > 30
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])
    rest = [c for c in clock.COLUMNS if c != "ck_slot"]
    assert np.isfinite(x[rest].to_numpy(float)).mean() > 0.8
    assert np.allclose(x[rest].to_numpy(float), z[rest].to_numpy(float), atol=1e-5, equal_nan=True)
    assert np.allclose(x.ck_slot, x.race_number / 6.0) and np.allclose(z.ck_slot, z.race_number / 3.0)
    # and the earlier days read none of it
    before = (runs.race_date < runs.race_date.max()).to_numpy()
    cols = list(clock.COLUMNS)
    assert np.array_equal(a.loc[before, cols].to_numpy(float), b.loc[(r2.race_date < r2.race_date.max()).to_numpy(), cols]
                          .to_numpy(float), equal_nan=True)


def test_a_card_as_the_live_path_makes_it_gets_what_its_run_had():
    """greyhound/card.py: no results, one meeting id a track's day, the races numbered by the off through both
    meetings, the off as HH:MM:SS, a negative race id, and a dog GBGB has no run for under a negative id of its own."""
    runs = synth(days=40)
    day = runs.race_date.max()
    keys = runs.loc[runs.race_date == day, ["race_id", "dog_id"]]
    a = clock.build(runs, core.per_run_figures(runs))[0]
    c = cardify_last_day(runs)
    last = (c.race_date == day).to_numpy()
    assert c.loc[last, "is_card"].all() and c.loc[last, "adjusted_time"].isna().all()
    c.loc[last, "meeting_id"] = -7
    c.loc[last, "race_number"] = pd.Series(clock.minutes_of(c.loc[last, "race_time"])).rank(method="dense").to_numpy()
    assert c.loc[last, "race_number"].max() == 12
    c.loc[last, "race_time"] = c.loc[last, "race_time"] + ":00"
    c.loc[last, "race_id"] = -c.loc[last, "race_id"]
    c.loc[last & (c.dog_id == 5).to_numpy(), "dog_id"] = -5                # unknown to GBGB: no history under this id
    c = clean(c.drop(columns=[x for x in ("won", "placed2", "sp_p", "sp_p_norm", "t", "field") if x in c]))
    assert (c.loc[c.race_date == day, "race_time"].str.len() == 5).all()
    b = clock.build(c, core.per_run_figures(c))[0]
    b = b[b.race_id < 0].assign(race_id=lambda d: -d.race_id)
    cols = list(clock.COLUMNS)
    known = keys[keys.dog_id != 5]
    x = known.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = known.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert len(z) == len(known) and np.isfinite(x).mean() > 0.8 and np.allclose(x, z, atol=1e-4, equal_nan=True)
    # the unknown dog: the race's own columns as its run had them, nothing of a history
    run5 = a[(a.dog_id == 5) & a.race_id.isin(keys.race_id)].iloc[0]
    new5 = b[b.dog_id == -5]
    assert len(new5) == 1
    new5 = new5.iloc[0]
    race_level = ["ck_wait", "ck_slot", "ck_session", "ck_dst", "ck_dark", "ck_fireworks", "ck_days_since_fw"]
    assert all(new5[k] == run5[k] for k in race_level) and np.isfinite(new5[race_level].to_numpy(float)).all()
    assert np.isnan(new5.ck_habit_gap) and new5.ck_habit_n == 0 and new5.ck_session_new == 0
    assert np.isnan(new5.ck_wait_slope) and np.isnan(new5.ck_wait_pred) and np.isnan(new5.ck_dark_share)


def test_the_rows_order_and_the_traps_are_not_read(built):
    runs, figs, xt, tab, _, _ = built
    cols = list(clock.COLUMNS)
    # the figures handed in shuffled: every row still gets its own features, back in the input's order
    b = clock.build(runs, figs.sample(frac=1.0, random_state=4), {"xtime": xt})[0]
    assert (b.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (b.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert np.array_equal(b[cols].to_numpy(float), tab[cols].to_numpy(float), equal_nan=True)
    # traps 7 and 8 (a reserve's number, an eight-dog race): nothing here is by trap
    c = clock.build(runs, figs.assign(trap=figs.trap + 2), {"xtime": xt})[0]
    assert np.array_equal(c[cols].to_numpy(float), tab[cols].to_numpy(float), equal_nan=True)


def test_the_clock_changes_and_summer_time_against_the_time_zone_database():
    days = pd.date_range("2014-01-01", "2027-12-31")
    cal = clock.calendar(days)
    evening = (days + pd.Timedelta(hours=18)).tz_localize("Europe/London")
    hours = np.array([t.utcoffset().total_seconds() / 3600.0 for t in evening])
    assert (cal.bst.to_numpy() == (hours == 1.0)).all()                           # summer time at an evening off
    change = np.flatnonzero(np.diff(hours) != 0) + 1                              # the first day on the new clock
    assert len(change) == 28
    want = np.zeros(len(days))
    for c in change:
        want[c - 7:c] = -1.0
    for c in change:
        want[c:c + 7] = 1.0
    assert (cal.dst.to_numpy() == want).all()
