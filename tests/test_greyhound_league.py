"""greyhound/pit/league.py: the league table. A planted strong league is recovered and a dog leaving it for a weak
one gets its class move (today less last: below zero going down, above zero going up); every result on and after a
refit Monday rewritten moves nothing in that week or before it; a card row gets what its run had; one row a
(race, dog) in the input order; no column reads the trainer. Planted data only."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import league
from tests.test_greyhound_pit import cardify_last_day

COLS = list(league.COLUMNS)
KEY = ["race_id", "dog_id"]
LEAGUES = (("Hove", "A1", 1.2), ("Romford", "A3", 0.6), ("Hove", "A5", 0.0))      # track, class, the dogs' mean skill
STRONG_DOWN, WEAK_UP, TRAVELLER = 1, 61, 62          # home A1, A5, A5: never away from home before the last day


def synth(days=120, seed=5):
    """Three leagues of 30 dogs, 12 points of skill apart. Each day a dog runs at home, or (15 in 100) visits another
    league: the visitors are the links. On the last day dog 1 (A1) drops to A5, dog 61 (A5) rises to A1 and dog 62
    (A5 at Hove) goes to Romford for the first time."""
    rng = np.random.default_rng(seed)
    home = np.repeat(np.arange(3), 30)
    skill = np.array([LEAGUES[h][2] for h in home]) + rng.normal(0, 0.3, 90)
    fixed = {STRONG_DOWN - 1, WEAK_UP - 1, TRAVELLER - 1}
    rows, rid = [], 9000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        where = home.copy()
        away = rng.random(90) < 0.15
        where[away] = (home[away] + rng.integers(1, 3, away.sum())) % 3
        for k in fixed:
            where[k] = home[k]
        if d == days - 1:
            where[STRONG_DOWN - 1], where[WEAK_UP - 1], where[TRAVELLER - 1] = 2, 0, 1
        for j, (track, cls, _) in enumerate(LEAGUES):
            pool = rng.permutation(np.flatnonzero(where == j))
            if d == days - 1:                                    # the three movers run (first in the pool)
                first = [k for k in pool if k in fixed]
                pool = np.array(first + [k for k in pool if k not in fixed])
            for r in range(len(pool) // 6):
                rid += 1
                field = pool[6 * r:6 * r + 6]
                u = skill[field] + rng.normal(0, 0.2, 6)
                p = np.exp(u) / np.exp(u).sum()
                perf = skill[field] + rng.normal(0, 0.8, 6)
                pos = np.argsort(np.argsort(-perf)) + 1
                t = 29.5 - 0.08 * perf * 3
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{11 + j * 3 + r // 4}:{15 * (r % 4):02d}",
                                 "meeting_id": d * 10 + (track == "Romford"), "track": track, "race_id": rid,
                                 "race_class": cls, "distance_m": 500.0, "trap": k + 1, "dog_id": int(dog) + 1,
                                 "dog_name": f"Dog {dog + 1}", "born": "Jan-2023", "sex": "d",
                                 "trainer": f"T{dog % 7}", "owner": f"O{dog % 5}",
                                 "sp_decimal": float(1.0 / (1.15 * p[k])),
                                 "market_pos": int(np.argsort(np.argsort(-p))[k]) + 1, "position": int(pos[k]),
                                 "beaten_lengths": 0.5, "sectional": 4.5, "run_time": float(round(t[k], 2)),
                                 "adjusted_time": float(round(t[k], 2)), "weight_kg": 30.0, "comment": "RanOn"})
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = league.build(runs)
    return runs, tab, notes


def rewrite_results(runs: pd.DataFrame, since: str) -> pd.DataFrame:
    """Every result dated ``since`` or later rewritten: the finishing order reversed, times and SPs changed."""
    r = runs.copy()
    m = r.race_date >= since
    r.loc[m, "position"] = r.loc[m, "field"] + 1 - r.loc[m, "position"]
    r.loc[m, "run_time"] = 59.0 - r.loc[m, "run_time"]
    r.loc[m, "adjusted_time"] = 59.0 - r.loc[m, "adjusted_time"]
    r.loc[m, "sp_decimal"] = 2.0 + r.loc[m, "trap"]
    r["won"] = (r.position == 1).astype(float)
    r["placed2"] = (r.position <= 2).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


def same(a: pd.DataFrame, b: pd.DataFrame, keys: pd.DataFrame) -> bool:
    x = keys.merge(a, on=KEY, how="left")[COLS].to_numpy()
    y = keys.merge(b, on=KEY, how="left")[COLS].to_numpy()
    return bool(np.array_equal(x, y, equal_nan=True))


def test_the_contract(built):
    runs, tab, notes = built
    assert list(tab.columns) == KEY + COLS and all(c.startswith(league.PREFIX) for c in COLS)
    assert 10 <= len(COLS) <= 14 and all(tab[c].dtype == np.float32 for c in COLS)
    assert len(tab) == len(runs) and not tab.duplicated(KEY).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    late = (runs.race_date >= "2025-03-01").to_numpy()
    for c in COLS:
        assert tab.loc[late, c].notna().mean() > 0.95, c
    assert notes["passes"]["league"]["at_the_cap"] == 0 and notes["passes"]["prb2"]["at_the_cap"] == 0
    # the input order is not the build's: the same rows in another order get the same values
    sh = runs.sample(frac=1.0, random_state=1)
    tab2, _ = league.build(sh)
    assert (tab2.race_id.to_numpy() == sh.race_id.to_numpy()).all() and not tab2.duplicated(KEY).any()
    x, y = runs[KEY].merge(tab, on=KEY)[COLS].to_numpy(), runs[KEY].merge(tab2, on=KEY)[COLS].to_numpy()
    assert np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(x, y, atol=1e-5, equal_nan=True)   # sums in another order


def test_the_refit_calendar():
    d = pd.date_range("2024-12-25", "2025-02-10")
    dn = ((d - pd.Timestamp("1970-01-01")).days).to_numpy()
    mon = (d - pd.to_timedelta(d.dayofweek, unit="D"))
    assert (league.monday(dn) == (mon - pd.Timestamp("1970-01-01")).days.to_numpy()).all()


def test_a_planted_strong_league_is_recovered_and_the_class_move_is_signed(built):
    runs, tab, _ = built
    m = runs.merge(tab, on=KEY)
    last = m[m.race_date == m.race_date.max()]
    s = last.groupby(["track", "race_class"]).lg_str.first()
    a1, a3, a5 = s[("Hove", "A1")], s[("Romford", "A3")], s[("Hove", "A5")]
    assert a1 > a3 + 0.05 and a3 > a5 + 0.05                    # the planted order, well apart
    st = last.groupby(["track", "race_class"]).lg_move_t.count()
    assert st.min() > 0
    down = last[last.dog_id == STRONG_DOWN].iloc[0]
    up = last[last.dog_id == WEAK_UP].iloc[0]
    trav = last[last.dog_id == TRAVELLER].iloc[0]
    # today less last: the dog dropping out of the strong league moves DOWN the scale, the one rising moves up
    assert down.race_class == "A5" and np.isclose(down.lg_str_last, a1) and np.isclose(down.lg_move, a5 - a1, atol=1e-6)
    assert down.lg_move < -0.1 and down.lg_move3 < -0.1 and down.lg_move_avg < -0.1 and down.lg_move_t < 0
    assert up.race_class == "A1" and up.lg_move > 0.1 and up.lg_move3 > 0.1 and up.lg_move_avg > 0.1 and up.lg_move_t > 0
    # the dropper is the class of its new field; the riser is not
    assert down.lg_rating_vs > 0.05 and up.lg_rating_vs < -0.05
    # a dog that stayed has not moved
    stay = last[(last.race_class == "A3") & np.isclose(last.lg_str_last, a3)]
    assert len(stay) and np.allclose(stay.lg_move, 0.0, atol=1e-6)
    # strength of schedule: the strong league's dogs have met better (their results flatter them less)
    homeA1, homeA5 = last[last.dog_id <= 30], last[(last.dog_id > 60)]
    for c in ("lg_sos_prb2", "lg_sos_nfp", "lg_mkt_diff"):
        assert homeA1[c].mean() > homeA5[c].mean() + 0.02, c
    # the track
    assert trav.track == "Romford" and trav.lg_new_trk == 1 and trav.lg_n_trk == 0
    before = runs[(runs.dog_id == STRONG_DOWN) & (runs.track == "Hove") & (runs.race_date < down.race_date)]
    assert down.lg_new_trk == 0 and down.lg_n_trk == len(before) > 50


def test_results_on_and_after_a_refit_monday_move_nothing_in_its_week_or_before(built):
    runs, tab, _ = built
    monday = "2025-04-07"
    assert pd.Timestamp(monday).dayofweek == 0
    sunday = "2025-04-13"
    tab2, _ = league.build(rewrite_results(runs, monday))
    upto = runs.loc[runs.race_date <= sunday, KEY]
    assert (runs.race_date >= monday).sum() > 1000 and len(upto) > 5000
    assert same(tab, tab2, upto)
    # the test can fail: the later weeks, which may read those results, do move
    later = runs.loc[runs.race_date > sunday, KEY]
    assert not same(tab, tab2, later)
    x, y = later.merge(tab, on=KEY)[COLS], later.merge(tab2, on=KEY)[COLS]
    moved = {c for c in COLS if not np.array_equal(x[c].to_numpy(), y[c].to_numpy(), equal_nan=True)}
    assert moved >= set(COLS) - {"lg_new_trk", "lg_n_trk"}       # every fitted column reads results, a week late
    # and the days after a week are not read by it: the store cut at the Sunday gives the same rows
    tab3, _ = league.build(runs[runs.race_date <= sunday].reset_index(drop=True))
    assert same(tab, tab3, upto)
    # a mid-week day's own results, and the earlier days of its week, are not read either
    wed = runs.loc[runs.race_date == "2025-04-09", KEY]
    assert len(wed) and same(tab, league.build(rewrite_results(runs, "2025-04-09"))[0], wed)


def test_a_card_row_gets_the_features_its_run_had(built):
    runs, tab, _ = built
    card = cardify_last_day(runs)
    last = runs.race_date.max()
    assert pd.Timestamp(last).dayofweek == 2                    # a Wednesday: two earlier days of its own week
    assert card.is_card.sum() == (runs.race_date == last).sum() and card.loc[card.is_card, "position"].isna().all()
    tab2, notes = league.build(card)
    assert notes["card_rows"] == card.is_card.sum() and len(tab2) == len(card) and not tab2.duplicated(KEY).any()
    keys = runs.loc[runs.race_date == last, KEY]
    a = keys.merge(tab, on=KEY)[COLS].to_numpy()
    b = keys.merge(tab2, on=KEY)[COLS].to_numpy()
    assert np.allclose(a, b, atol=1e-6, equal_nan=True)
    assert np.isfinite(b).mean() > 0.95


@pytest.mark.parametrize("weeks", (2, 4, 8))
def test_the_card_build_from_a_few_weeks_back_gives_the_full_builds_card_rows(built, weeks):
    """The live card build (``ctx["card_weeks"]``, greyhound/live.py ``card_tables``) refits only from ``weeks``
    Mondays before the card's week: every card row within 1e-3 of the full build on every column, the track counts
    exact; fewer refits; and without card rows (the nightly build) the flag changes nothing."""
    runs, tab, notes = built
    card = cardify_last_day(runs)
    keys = runs.loc[runs.race_date == runs.race_date.max(), KEY]
    short, sn = league.build(card, None, {"card_weeks": weeks})
    assert sn["weeks"] == weeks + 1 < notes["weeks"] and sn["card_weeks"] == weeks
    a = keys.merge(tab, on=KEY)[COLS]
    b = keys.merge(short, on=KEY)[COLS]
    assert len(b) == len(keys)
    for c in COLS:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), c
        assert np.nanmax(np.abs(x - y)) <= 1e-3, (c, weeks)
    for c in ("lg_new_trk", "lg_n_trk", "lg_trap_wax", "lg_mkt_open", "lg_mkt_diff"):   # no warm start: exact
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True), c
    assert np.isfinite(b.to_numpy(float)).mean() > 0.95
    # the history rows before the first refit are left empty: the card build is read on the card only
    assert short.loc[(runs.race_date < "2025-02-01").to_numpy(), "lg_str"].isna().all()
    # no card rows: the nightly build, every week fitted
    whole, wn = league.build(runs, None, {"card_weeks": weeks})
    assert wn["weeks"] == notes["weeks"] and same(tab, whole, runs[KEY])
    assert league.CARD_WEEKS >= 26

def test_no_column_reads_the_trainer(built):
    runs, tab, _ = built
    r = runs.copy()
    rng = np.random.default_rng(0)
    r["trainer"] = rng.permutation(r.trainer.to_numpy())
    r["owner"] = "one owner"
    assert same(tab, league.build(r)[0], runs[KEY])
    assert same(tab, league.build(runs.drop(columns=["trainer", "owner"]))[0], runs[KEY])
    code = "".join(inspect.getsource(f) for f in (league.build, league.league_mm, league.closed_form,
                                                    league.market_pass, league._codes, league._earlier))
    for word in ("trainer", "kennel", "owner"):
        assert word not in code
        assert not any(word in c or word in what.split("(")[0].lower() for c, what in league.COLUMNS.items())
