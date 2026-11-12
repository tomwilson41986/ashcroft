"""greyhound/pit/followed.py: the followed dog. A dog planted as an always-beaten odds-on favourite reads as one (and
as the only name in its races), a dog planted as a winning outsider as the reverse; a row-at-a-time reading of the
specification gives the same table on a card where dogs run twice a day; a day's results rewritten move nothing on
that day or before it (and do move later days); the days after a day move nothing on it; a card row, with no result,
gets what its run had; a dog with no priced run has a count of 0 and no rates; one row a race and dog."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import followed
from tests.test_greyhound_pit import cardify_last_day

P = followed.PREFIX
COLS = list(followed.COLUMNS)
FAV, OUT = 1, 2                               # the planted beaten favourite and the planted winning outsider


def synth(days=30, seed=3):
    """One track, 4 races of 6 a day from 26 dogs, each race drawn afresh (a dog can run twice a day). Dog 1 runs
    only in the first race, at 1.5 and market rank 1, and finishes last; dog 2 only in the second, at 12.0 and
    market rank 6, and wins. The rest have random prices and positions. Every tenth run has no SP."""
    rng = np.random.default_rng(seed)
    rows, rid, k = [], 7000, 0
    for d in range(days):
        day = (pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for r in range(4):
            rid += 1
            rest = rng.choice(np.arange(3, 27), 6, replace=False)
            field = rest.copy()
            sp = np.round(rng.uniform(2.0, 9.0, 6), 2)
            pos = rng.permutation(6) + 1
            if r == 0:
                field[0], sp[0] = FAV, 1.5
                pos = np.r_[6, rng.permutation(5) + 1]
            if r == 1:
                field[0], sp[0] = OUT, 12.0
                pos = np.r_[1, rng.permutation(5) + 2]
            mp = np.argsort(np.argsort(sp)) + 1
            for j, dog in enumerate(field):
                k += 1
                miss = k % 10 == 0 and dog > 2
                rows.append({"race_date": day, "race_time": f"{18 + r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": j + 1,
                             "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                             "sp_decimal": np.nan if miss else float(sp[j]),
                             "market_pos": np.nan if miss else float(mp[j]), "position": int(pos[j]),
                             "beaten_lengths": 0.5, "sectional": 4.5, "run_time": 29.5, "adjusted_time": 29.5,
                             "weight_kg": 30.0, "comment": "QAw"})
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def runs():
    return synth()


@pytest.fixture(scope="module")
def tab(runs):
    return followed.build(runs)[0]


def keyed(t):
    return t.set_index(["race_id", "dog_id"])[COLS]


def same(a, b, keys):
    ix = pd.MultiIndex.from_frame(keys[["race_id", "dog_id"]])
    np.testing.assert_allclose(keyed(a).loc[ix].to_numpy(float), keyed(b).loc[ix].to_numpy(float), rtol=1e-6,
                               atol=1e-6, equal_nan=True)


def slow(runs):
    """The specification read a row at a time."""
    day = pd.to_datetime(runs.race_date)
    rows = []
    for i in range(len(runs)):
        me = runs.iloc[i]
        h = runs[(runs.dog_id == me.dog_id) & (day < day.iloc[i]) & runs.sp_decimal.notna() & runs.market_pos.notna()
                 & runs.position.notna()].sort_values("t")
        v = dict.fromkeys(COLS, np.nan)
        v[f"{P}n"] = float(len(h))
        if len(h):
            fav = (h.market_pos == 1).to_numpy(float)
            short = (h.sp_decimal <= followed.SHORT).to_numpy(float)
            won = (h.position == 1).to_numpy(float)
            over = (h.position - h.market_pos).to_numpy(float)
            run_of = lambda f: int(len(f) - 1 - np.flatnonzero(f == 0).max()) if (f == 0).any() else len(f)  # noqa: E731
            v[f"{P}fav_rate"] = fav.mean()
            v[f"{P}fav_rate_l5"] = fav[-5:].mean()
            v[f"{P}short_streak"] = run_of(short)
            v[f"{P}fav_streak"] = run_of(fav)
            v[f"{P}beaten_fav_rate"] = (fav * (1 - won)).sum() / fav.sum() if fav.sum() else np.nan
            v[f"{P}beaten_fav_l5"] = (fav * (1 - won))[-5:].sum()
            v[f"{P}last_beaten_fav"] = fav[-1] * (1 - won[-1])
            v[f"{P}overbet"] = over.mean()
            v[f"{P}overbet_l5"] = over[-5:].mean()
            v[f"{P}ae"] = (won.sum() - h.sp_p_norm.fillna(0).sum()) / len(h)
            v[f"{P}logsp_l5"] = np.log(h.sp_decimal.to_numpy(float))[-5:].mean()
            sw = pd.to_datetime(h.race_date[(short == 1) & (won == 1)])
            if len(sw):
                v[f"{P}days_short_win"] = min((day.iloc[i] - sw.max()).days, 365)
        rows.append({"race_id": me.race_id, "dog_id": me.dog_id, **v})
    out = pd.DataFrame(rows)
    for rid, g in out.groupby("race_id"):
        for i in g.index:
            oth = g.loc[g.index != i, f"{P}fav_rate_l5"].dropna()
            out.loc[i, f"{P}oth_fav_rate_l5"] = oth.mean() if len(oth) else np.nan
            if len(g) > 1:
                out.loc[i, f"{P}name_gap"] = out.loc[i, f"{P}fav_rate_l5"] - (oth.max() if len(oth) else 0.0)
    return out


def test_the_planted_names_read_as_planted(runs, tab):
    t = tab.merge(runs[["race_id", "dog_id", "race_date"]], on=["race_id", "dog_id"])
    late = t[t.race_date >= "2025-03-10"]
    f, o, rest = late[late.dog_id == FAV], late[late.dog_id == OUT], late[late.dog_id > 2]
    assert len(f) and len(o)
    assert (f[f"{P}fav_rate"] == 1).all() and (f[f"{P}fav_rate_l5"] == 1).all()
    assert (f[f"{P}beaten_fav_rate"] == 1).all() and (f[f"{P}beaten_fav_l5"] == 5).all()
    assert (f[f"{P}last_beaten_fav"] == 1).all()
    assert (f[f"{P}fav_streak"] == f[f"{P}n"]).all() and (f[f"{P}short_streak"] == f[f"{P}n"]).all()
    assert np.allclose(f[f"{P}overbet"], 5.0) and np.allclose(f[f"{P}logsp_l5"], np.log(1.5), atol=1e-6)
    assert (f[f"{P}ae"] < -0.3).all() and f[f"{P}days_short_win"].isna().all()
    assert (f[f"{P}name_gap"] > 0.2).all()                                   # the only name in its races
    assert (o[f"{P}fav_rate"] == 0).all() and o[f"{P}beaten_fav_rate"].isna().all()
    assert np.allclose(o[f"{P}overbet"], -5.0) and (o[f"{P}ae"] > 0.8).all()
    assert (o[f"{P}short_streak"] == 0).all() and (o[f"{P}name_gap"] <= 0).all() and o[f"{P}name_gap"].mean() < -0.1
    # the favourite's rivals see a name among the others; the rest of the card does not
    with_fav = late.race_id.isin(f.race_id) & (late.dog_id != FAV)
    assert late.loc[with_fav, f"{P}oth_fav_rate_l5"].mean() > rest[f"{P}oth_fav_rate_l5"].mean() + 0.1
    assert abs(rest[f"{P}overbet"].mean()) < 0.5 < f[f"{P}overbet"].mean()
    # the first day has no history: a count of 0, no rates
    first = t[t.race_date == t.race_date.min()]
    assert (first[f"{P}n"] == 0).all() and first[[c for c in COLS if c != f"{P}n"]].isna().all().all()


def test_the_table_is_the_specification_read_a_row_at_a_time(runs, tab):
    assert runs.groupby(["dog_id", "race_date"]).size().max() > 1            # dogs run twice a day here
    want = slow(runs)
    assert (want[["race_id", "dog_id"]].to_numpy() == tab[["race_id", "dog_id"]].to_numpy()).all()
    for c in COLS:
        np.testing.assert_allclose(tab[c].to_numpy(float), want[c].to_numpy(float), rtol=1e-5, atol=1e-5,
                                   equal_nan=True, err_msg=c)


def rewrite_day(runs, day):
    """``day``'s results rewritten: positions reversed, every dog 2.0 and market rank 1."""
    r = runs.copy()
    on = r.race_date == day
    r.loc[on, "position"] = 7 - r.loc[on, "position"]
    r.loc[on, "sp_decimal"] = 2.0
    r.loc[on, "market_pos"] = 1.0
    r["won"] = (r.position == 1).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


def test_a_days_results_move_nothing_on_that_day_or_before(runs, tab):
    days = sorted(runs.race_date.unique())
    for day in (days[len(days) // 2], days[-1]):
        other = followed.build(rewrite_day(runs, day))[0]
        same(tab, other, runs[runs.race_date <= day])
        later = runs[runs.race_date > day]
        if len(later):                                                       # and the test can see a change
            ix = pd.MultiIndex.from_frame(later[["race_id", "dog_id"]])
            assert not np.allclose(keyed(tab).loc[ix].to_numpy(float), keyed(other).loc[ix].to_numpy(float),
                                   equal_nan=True)


def test_later_days_move_nothing(runs, tab):
    days = sorted(runs.race_date.unique())
    for day in (days[5], days[len(days) // 2]):
        cut = runs[runs.race_date <= day].reset_index(drop=True)
        same(tab, followed.build(cut)[0], cut)


def test_a_card_row_gets_what_its_run_had(runs, tab):
    card = cardify_last_day(runs)
    last = card[card.is_card]
    assert len(last) and last.position.isna().all() and last.sp_decimal.isna().all()
    t, notes = followed.build(card)
    assert notes["card_rows"] == len(last) and len(t) == len(card)
    same(tab, t, last)
    got = keyed(t).loc[pd.MultiIndex.from_frame(last[["race_id", "dog_id"]])]
    assert (got[f"{P}n"] > 0).mean() > 0.9 and got[f"{P}fav_rate_l5"].notna().mean() > 0.9
    # a card row with a price written on it (a forecast SP) is still no history for a later day
    priced = card.copy()
    priced.loc[priced.is_card, ["sp_decimal", "market_pos", "position"]] = [2.0, 1.0, 1.0]
    extra = priced[priced.is_card].assign(race_date="2025-12-01", race_id=lambda x: x.race_id + 100000,
                                          t=pd.Timestamp("2025-12-01 18:00"))
    both = pd.concat([priced, extra], ignore_index=True)
    plain = pd.concat([card, extra], ignore_index=True)
    same(followed.build(both)[0], followed.build(plain)[0], extra)


def test_one_row_a_race_and_dog_every_row_float32(runs, tab):
    assert len(tab) == len(runs) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab[["race_id", "dog_id"]].to_numpy() == runs[["race_id", "dog_id"]].to_numpy()).all()
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS and all(c.startswith(P) for c in COLS)
    assert (tab[COLS].dtypes == np.float32).all()
    assert not set(getattr(followed, "CARD_UNSAFE", ())) and followed.SOURCE
    # shuffled rows give the same table, in the rows' order
    sh = runs.sample(frac=1.0, random_state=1).reset_index(drop=True)
    t = followed.build(sh)[0]
    assert (t[["race_id", "dog_id"]].to_numpy() == sh[["race_id", "dog_id"]].to_numpy()).all()
    same(tab, t, runs)


def test_a_frame_with_no_prices_and_an_empty_frame(runs):
    bare = runs.drop(columns=["market_pos"])
    t, notes = followed.build(bare)
    assert not notes["has_market_pos"] and (t[f"{P}n"] == 0).all() and t[f"{P}name_gap"].isna().all()
    t, _ = followed.build(runs.iloc[:0])
    assert len(t) == 0 and list(t.columns) == ["race_id", "dog_id"] + COLS


def test_a_run_with_no_time_of_day_stays_on_its_day(runs, tab):
    """A history run whose time is missing is still read on its own day: the later runs are not in its day's state."""
    days = sorted(runs.race_date.unique())
    r = runs.copy()
    r.loc[(r.race_date == days[4]) & (r.dog_id == FAV), "t"] = pd.NaT
    assert r.t.isna().sum() == 1
    same(tab, followed.build(r)[0], runs)
