"""greyhound/pit/readers.py: six fast-and-frugal rules ranking the field off the printed card. A planted champion is
every rule's pick and a planted fast trap is the trap rule's; a slow, row-by-row reading of the specification gives
the same table on a card where dogs run twice a day and at two tracks and trips; a day's own results move nothing on
that day or before it (and do move later days); the days after a day move nothing on it; a card row, with no result
yet, gets what its run had and is never a line of history; the keys are the runs', one row each."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import readers
from tests.test_greyhound_pit import cardify_last_day
from tests.test_greyhound_pit import synth as pit_synth

KEY = ["race_id", "dog_id"]
COLS = list(readers.COLUMNS)
FAST_TRAP = 0.06                              # seconds trap 1 runs faster at Hove (planted)


def synth(days=60, seed=7, champion=0.5):
    """Hove over 500 m and 285 m, 6 races of 6 a day from 37 dogs. Dog 1 (the champion) is ``champion`` seconds
    faster than an average dog and runs every third day, drawn in traps 2 to 6; trap 1 runs FAST_TRAP faster."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.08, 38)
    skill[1] = -champion
    rows, rid = [], 7000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        dogs = rng.permutation(np.arange(2, 38))
        for r in range(6):
            field = dogs[6 * r:6 * r + 6].copy()
            if r == 2 and d % 3 == 0:
                field[rng.integers(1, 6)] = 1
            rid += 1
            trip = 285.0 if r == 5 else 500.0
            t = (29.5 if trip == 500.0 else 16.8) + skill[field] + rng.normal(0, 0.05, 6)
            t[0] -= FAST_TRAP
            order = np.argsort(t)
            pos = np.argsort(order) + 1
            for k, dog in enumerate(field):
                rows.append({"race_date": day, "race_time": f"{18 + r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": trip, "trap": k + 1,
                             "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                             "sp_decimal": 6.0, "position": int(pos[k]),
                             "beaten_lengths": float(round((t[k] - t.min()) / 0.08, 2)), "sectional": 4.5,
                             "run_time": float(round(t[k], 2)), "adjusted_time": float(round(t[k], 2)),
                             "weight_kg": 30.0, "comment": "QAw"})
    return clean(pd.DataFrame(rows))


def perturb_day(runs, day):
    """One day's results rewritten: positions reversed, the times and the distances changed."""
    r = runs.copy()
    on = r.race_date == day
    r.loc[on, "position"] = r.loc[on, "field"] + 1 - r.loc[on, "position"]
    r.loc[on, "adjusted_time"] -= 0.7 * (r.loc[on, "trap"] % 2)
    r.loc[on, "run_time"] = r.loc[on, "adjusted_time"]
    r.loc[on, "beaten_lengths"] = 9.0 - r.loc[on, "beaten_lengths"].clip(upper=9.0)
    r["won"] = (r.position == 1).astype(float)
    r["placed2"] = (r.position <= 2).astype(float)
    return r


def same(a, b):
    return np.allclose(a[COLS].to_numpy(float), b[COLS].to_numpy(float), equal_nan=True, atol=1e-6)


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = readers.build(runs)
    return runs, runs.merge(tab, on=KEY), tab, notes


def test_every_column_is_registered_and_the_keys_are_the_runs(built):
    runs, m, tab, notes = built
    assert list(tab.columns) == KEY + COLS and all(c.startswith(readers.PREFIX) for c in COLS)
    assert (tab[COLS].dtypes == np.float32).all()
    assert len(tab) == len(runs) == len(m)
    assert not tab.duplicated(KEY).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert notes["rows"] == len(runs) and notes["rows_with_a_rule"] > 0.9
    assert not getattr(readers, "CARD_UNSAFE", ())


def test_no_duplicate_keys_on_a_shuffled_frame_with_a_dog_twice_a_day():
    runs = pit_synth(days=20)                                        # two tracks a day: dogs run twice a day
    assert runs.groupby(["dog_id", "race_date"]).size().max() > 1
    shuffled = runs.sample(frac=1.0, random_state=3)
    tab, _ = readers.build(shuffled)
    assert len(tab) == len(runs) and not tab.duplicated(KEY).any()
    assert (tab.race_id.to_numpy() == shuffled.race_id.to_numpy()).all()
    assert (tab.dog_id.to_numpy() == shuffled.dog_id.to_numpy()).all()
    a = readers.build(runs)[0].sort_values(KEY).reset_index(drop=True)
    assert same(a, tab.sort_values(KEY).reset_index(drop=True))     # the row order moves nothing


def test_the_planted_champion_is_every_form_rules_pick_and_the_fast_trap_the_trap_rules(built):
    runs, m, _, _ = built
    late = m[m.race_date >= "2025-02-01"]
    one = late[late.dog_id == 1]
    rest = late[late.race_id.isin(one.race_id) & (late.dog_id != 1)]
    assert len(one) >= 8
    # half a second clear: it won last time (first or joint first with the other last-time winners in the field),
    # has the most wins and the best time here
    assert (one.rd_rk_pos <= 2).all() and (one.rd_rk_beat <= 2).all()
    assert (one.rd_rk_pos < rest.groupby("race_id").rd_rk_pos.mean().mean()).all()
    assert (one.rd_rk_wins <= 1.5).all() and (one.rd_rk_wins == 1).mean() > 0.8 and (one.rd_rk_time == 1).all()
    assert (one.rd_ntop >= 4).all() and one.rd_ntop.mean() > rest.rd_ntop.mean() + 2
    assert (one.rd_mean_rk <= 3).all() and one.rd_mean.mean() < rest.rd_mean.mean() - 1.0
    # it ran three days ago and its rivals the day before: the days rule alone ranks it last
    assert (one.rd_rk_days == 6).all()
    # most days it is the best on the mean rank too (a last-time winner in the fast trap can match its four picks:
    # the flags are checked by hand below)
    assert (one.rd_mean_rk == 1).mean() > 0.6
    assert (one.rd_ntop == late[late.race_id.isin(one.race_id)].groupby("race_id").rd_ntop.max().to_numpy()).all()
    # the trap rule: nothing until the trap has 100 runs at the track, then the planted fast trap
    assert m[m.race_date <= "2025-01-10"].rd_rk_trap.isna().all()
    assert (late[late.trap == 1].rd_rk_trap == 1).mean() > 0.95
    assert late[late.trap != 1].rd_rk_trap.min() >= 2
    # over the other dogs the rules find the better dog: the mean rank falls as the dog wins
    plain = late[~late.race_id.isin(one.race_id)]
    assert plain.rd_mean_rk.corr(plain.position, method="spearman") > 0.1
    assert plain.groupby("race_id").rd_ntop.max().min() >= 1         # every race has some rule's pick


def test_the_flags_and_the_spread_by_hand(built):
    _, m, _, _ = built
    late = m[m.race_date >= "2025-02-01"]
    rk = [f"rd_rk_{k}" for k in readers.RULES]
    assert np.allclose(late.rd_mean, late[rk].mean(axis=1), atol=1e-5)
    assert np.allclose(late.rd_sd, late[rk].std(axis=1, ddof=0), atol=1e-5)
    seen = {0: 0, 1: 0}
    for _, g in late.groupby("race_id"):
        assert np.allclose(g.rd_race_sd, g.rd_sd.mean(), atol=1e-5)
        most = g.rd_ntop.max()
        alone = (g.rd_ntop == most).sum() == 1
        best = g.rd_mean == g.rd_mean.min()
        obv = (g.rd_ntop == most) & alone
        assert (g.rd_obv_notbest == (obv & ~best).astype(float)).all()
        assert (g.rd_best_notobv == (best & ~obv).astype(float)).all()
        seen[int(g.rd_best_notobv.max())] += 1
    assert seen[0] > 20 and seen[1] > 20                              # both cases are met


def slow_values(runs):
    """The specification read row by row: each rule's raw reading of every row."""
    hist = runs[~runs.is_card].sort_values(["dog_id", "t"])
    by_dog = {d: g for d, g in hist.groupby("dog_id")}
    out = []
    for i, r in runs.iterrows():
        h = by_dog.get(r.dog_id)
        h = h[h.race_date < r.race_date].tail(readers.LINES) if h is not None else hist.iloc[:0]
        v = dict.fromkeys(readers.RULES, np.nan)
        if len(h):
            last = h.iloc[-1]
            v["pos"], v["beat"] = last.position, last.beaten_lengths
            v["days"] = (pd.Timestamp(r.race_date) - pd.Timestamp(last.race_date)).days
            v["wins"] = -h.won.sum()
            here = h[(h.track == r.track) & (h.distance_m == r.distance_m)]
            pace = here.adjusted_time / here.distance_m
            here = here[(pace >= readers.PACE[0]) & (pace <= readers.PACE[1])]
            v["time"] = here.adjusted_time.min() if len(here) else np.nan
        tr = hist[(hist.track == r.track) & (hist.trap == r.trap) & (hist.race_date < r.race_date)]
        if len(tr) >= readers.TRAP_MIN:
            v["trap"] = -tr.won.mean()
        out.append(v)
    return pd.DataFrame(out, index=runs.index)[list(readers.RULES)]


def test_a_slow_reading_of_the_specification_gives_the_same_table(monkeypatch):
    monkeypatch.setattr(readers, "TRAP_MIN", 20)
    runs = cardify_last_day(pit_synth(days=14))                      # two tracks, dogs twice a day, a card last
    fast = readers.rule_values(runs)
    slow = slow_values(runs)
    assert fast.notna().mean().min() > 0.3
    assert np.allclose(fast.to_numpy(float), slow.to_numpy(float), equal_nan=True, atol=1e-9)
    tab, _ = readers.build(runs)
    g = slow.groupby(runs.race_id.to_numpy())
    for k in readers.RULES:
        assert np.allclose(tab[f"rd_rk_{k}"].to_numpy(float), g[k].rank(method="average").to_numpy(float),
                           equal_nan=True)
    assert np.allclose(tab.rd_ntop.to_numpy(float),
                       (g.rank(method="min") == 1).sum(axis=1).where(slow.notna().any(axis=1)).to_numpy(float),
                       equal_nan=True)


def test_a_days_results_move_nothing_on_that_day_or_before(built):
    runs, _, tab, _ = built
    day = "2025-02-10"                                               # a middle day
    tab_b, _ = readers.build(perturb_day(runs, day))
    upto = (runs.race_date <= day).to_numpy()
    assert same(tab[upto], tab_b[upto])
    after = tab[~upto][COLS].to_numpy(float), tab_b[~upto][COLS].to_numpy(float)
    assert not np.allclose(after[0], after[1], equal_nan=True)       # the results are read, from the next day on
    # and the last day's own results move nothing on it
    last = runs.race_date.max()
    tab_c, _ = readers.build(perturb_day(runs, last))
    assert same(tab, tab_c)


def test_later_days_move_nothing_on_a_day(built):
    runs, _, tab, _ = built
    day = "2025-02-10"
    upto = (runs.race_date <= day).to_numpy()
    cut, _ = readers.build(runs[upto])
    assert (cut.race_id.to_numpy() == tab.race_id.to_numpy()[upto]).all()
    assert same(tab[upto].reset_index(drop=True), cut)


def test_a_card_row_gets_the_features_its_run_had(built):
    runs, _, tab, _ = built
    card = cardify_last_day(runs)
    is_card = card.is_card.to_numpy(bool)
    assert is_card.sum() == 36 and card.loc[is_card, "position"].isna().all()
    tab_c, _ = readers.build(card)
    a = runs[KEY].merge(tab, on=KEY)
    b = runs[KEY].merge(tab_c, on=KEY)
    assert len(a) == len(b) == len(runs) and same(a, b)
    rows = tab_c[is_card]
    # made from history: every rule reads the card (the time rule only a dog with a line at the card's trip)
    assert rows[[c for c in COLS if c != "rd_rk_time"]].notna().all().all() and rows.rd_rk_time.notna().mean() > 0.8
    # a card row is never a line: a second card the next day reads the run before the first card
    nxt = card[is_card].assign(race_date="2025-03-02", race_id=lambda x: x.race_id + 100_000)
    two = clean(pd.concat([card, nxt.drop(columns=["t", "won", "placed2", "field"])], ignore_index=True))
    tab_2, _ = readers.build(two)
    first = two[KEY].merge(tab_2, on=KEY)[(two.race_date == "2025-03-01").to_numpy()]
    second = two[KEY].merge(tab_2, on=KEY)[(two.race_date == "2025-03-02").to_numpy()]
    for k in ("pos", "beat", "wins", "time"):
        assert np.allclose(first[f"rd_rk_{k}"].to_numpy(), second[f"rd_rk_{k}"].to_numpy(), equal_nan=True)


def test_a_dog_with_no_line_has_no_form_rank():
    runs = synth(days=3)
    tab, _ = readers.build(runs)
    first = (runs.race_date == runs.race_date.min()).to_numpy()
    assert tab[first][COLS].isna().all().all()                       # no history, no trap record: nothing
    assert tab[~first].rd_rk_pos.notna().any() and tab.rd_rk_trap.isna().all()


def test_a_time_no_greyhound_runs_is_not_read_and_a_repeated_index_moves_nothing(built):
    runs, _, tab, _ = built
    bad = runs.copy()
    day = "2025-02-10"
    hit = ((bad.race_date == day) & (bad.distance_m == 500.0) & (bad.trap == 3)).to_numpy()
    bad.loc[hit, "adjusted_time"] = 16.9                             # a sprint's time keyed under a 500 m race
    tab_b, _ = readers.build(bad)
    assert same(tab, tab_b)                                          # the best time of every later card is as it was
    twice = runs.set_index(np.arange(len(runs)) // 2)                # an index that repeats (a concat not re-indexed)
    tab_t, _ = readers.build(twice)
    assert len(tab_t) == len(runs) and same(tab, tab_t)
    assert (tab_t.race_id.to_numpy() == runs.race_id.to_numpy()).all()
