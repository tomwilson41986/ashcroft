"""greyhound/pit/cells.py: the price's past record in each cell the card shows. A trap that wins more than its SP
says at one track has a positive A/E there and not at the other; a yard always sent off short has a negative price
habit; a day's own results, and every later day, move nothing on that day; a card row gets what its run gets; a
row-by-row reading of the specification gives the same table; one row a (race, dog). The data is planted (a
synthetic card), as a unit test's may be."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import cells, core
from tests.test_greyhound_pit import cardify_last_day

TRACKS = ("Hove", "Romford")
COLS = list(cells.COLUMNS)
KEY = ["race_id", "dog_id"]


def synth(days=70, seed=3):
    """Two tracks, 5 races of 6 a track a day from 90 dogs. Planted: at Hove trap 1 wins 60% of its races at an
    SP like any other trap's; Trainer 0's dogs are sent off at 2.5 where the others are 7.0."""
    rng = np.random.default_rng(seed)
    dogs = np.arange(1, 91)
    rows, rid = [], 7000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for ti, track in enumerate(TRACKS):
            pool = rng.permutation(dogs)
            for r in range(5):
                rid += 1
                field = pool[6 * r:6 * r + 6]
                order = rng.permutation(6)                            # order[0] wins
                if track == "Hove" and rng.random() < 0.6:
                    order = np.r_[0, rng.permutation(np.arange(1, 6))]
                pos = np.empty(6, int)
                pos[order] = np.arange(1, 7)
                cls = f"A{1 + int(rng.integers(0, 4))}"
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{18 + r}:{10 * ti:02d}", "meeting_id": d * 10 + ti,
                                 "track": track, "race_id": rid, "race_class": cls,
                                 "distance_m": 480.0 if r < 4 else 270.0, "trap": k + 1, "dog_id": int(dog),
                                 "dog_name": f"Dog {dog}", "sire": f"Sire {dog % 7}", "dam": f"Dam {dog % 11}",
                                 "born": f"{('Jan', 'Jun', 'Oct')[dog % 3]}-{2021 + dog % 3}",
                                 "sex": "b" if dog % 2 else "d", "trainer": f"Trainer {dog % 6}",
                                 "sp_decimal": 2.5 if dog % 6 == 0 else 7.0, "position": int(pos[k]),
                                 "beaten_lengths": float(pos[k] - 1), "sectional": 4.5, "run_time": 29.5,
                                 "adjusted_time": 29.5, "weight_kg": 30.0, "comment": "QAw"})
    return clean(pd.DataFrame(rows))


def rewrite_day(runs, day):
    """``day``'s results rewritten: the positions reversed and every SP changed."""
    r = runs.copy()
    m = r.race_date == day
    r.loc[m, "position"] = 7 - r.loc[m, "position"]
    r.loc[m, "sp_decimal"] = np.where(r.loc[m, "trap"] % 2 == 0, 3.0, 11.0)
    r["won"] = (r.position == 1).astype(float)
    r["placed2"] = (r.position <= 2).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


def on(tab, runs, mask):
    k = runs.loc[mask, KEY]
    return k.merge(tab, on=KEY, how="left").sort_values(KEY)[COLS].to_numpy(float)


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = cells.build(runs, None, None)
    return runs, tab, notes


def test_every_column_is_registered_and_every_row_is_there(built):
    runs, tab, notes = built
    assert list(tab.columns) == KEY + COLS and all(c.startswith(cells.PREFIX) for c in COLS)
    assert len(tab) == len(runs) and (tab[COLS].dtypes == np.float32).all()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert notes["rows"] == len(runs) and notes["trainer_read"].startswith("as served")
    # the trainer cells read the trainer as served: never the card's until a fetch record is passed
    assert set(cells.CARD_UNSAFE) == {c for c in COLS if "_tr_" in c or "trtrack" in c} and len(cells.CARD_UNSAFE) == 4


def test_no_duplicate_keys(built):
    runs, tab, notes = built
    assert not tab.duplicated(KEY).any()
    doubled = pd.concat([runs, runs.tail(50)], ignore_index=True)
    t2, n2 = cells.build(doubled, None, None)
    assert not t2.duplicated(KEY).any() and len(t2) == len(runs) and n2["duplicate_keys_dropped"] == 50
    np.testing.assert_allclose(t2[COLS].to_numpy(float), tab[COLS].to_numpy(float), atol=1e-6, equal_nan=True)


def test_the_planted_habits_are_found(built):
    runs, tab, notes = built
    m = runs.merge(tab, on=KEY)
    late = m[m.race_date >= "2025-02-20"]
    hove1 = late[(late.track == "Hove") & (late.trap == 1)]
    hove_rest = late[(late.track == "Hove") & (late.trap != 1)]
    rom1 = late[(late.track == "Romford") & (late.trap == 1)]
    # trap 1 at Hove wins 60% at an SP worth about a sixth: its record against the price is far above the others'
    for c in ("ch_tdt_ae", "ch_tt_ae"):
        assert hove1[c].min() > 0.5
        assert hove1[c].mean() > hove_rest[c].mean() + 0.6
        assert abs(rom1[c].mean()) < 0.3
    assert hove_rest.ch_tt_ae.mean() < 0                              # the traps that paid for it
    # Trainer 0 is always the short one; the others are a little longer than their races' mean
    short = late[late.trainer == "Trainer 0"]
    rest = late[late.trainer != "Trainer 0"]
    assert short.ch_trtrack_sp.max() < -0.4 and rest.ch_trtrack_sp.min() > 0
    # the first day has no history: every cell at its prior
    first = m[m.race_date == m.race_date.min()]
    assert (first[COLS].fillna(0.0).to_numpy() == 0).all()
    assert (first.ch_move_ae.notna()).all() and notes["moves"]["first"] >= 90 - 30
    assert notes["moves"]["up"] > 0 and notes["moves"]["down"] > 0 and notes["moves"]["newtrack"] > 0


def test_a_slow_reading_of_the_specification_agrees(built):
    """A row at a time: the cell's earlier-day rows, their winners, expected winners and centred log SPs."""
    runs, tab, notes = built
    r = runs.copy()
    lsp = np.log(r.sp_decimal)
    r["_c"] = lsp - lsp.groupby(r.race_id).transform("mean")
    r["_dist"] = r.distance_m.round()
    m = r.merge(tab, on=KEY)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(m), 60, replace=False):
        x = m.iloc[i]
        past = r[r.race_date < x.race_date]
        for name, sel in (("tdt", (past.track == x.track) & (past._dist == x._dist) & (past.trap == x.trap)),
                          ("tt", (past.track == x.track) & (past.trap == x.trap)),
                          ("trtrack", (past.trainer == x.trainer) & (past.track == x.track)),
                          ("tr", past.trainer == x.trainer), ("sire", past.sire == x.sire)):
            h = past[sel]
            if f"ch_{name}_ae" in COLS:
                want = np.log((h.won.sum() + cells.K_AE) / (h.sp_p_norm.sum() + cells.K_AE))
                assert x[f"ch_{name}_ae"] == pytest.approx(want, abs=1e-5)
            if f"ch_{name}_sp" in COLS:
                assert x[f"ch_{name}_sp"] == pytest.approx(h._c.sum() / (len(h) + cells.K_SP), abs=1e-5)
            if f"ch_{name}_n" in COLS:
                assert x[f"ch_{name}_n"] == pytest.approx(np.log1p(len(h)), abs=1e-5)


def test_a_result_with_no_sp_is_left_out_of_the_price_columns(built):
    """A winner with no SP has no expected winners to be set against: the A/E and the price habit read the priced
    results only, the trainer's count every result."""
    runs, tab, notes = built
    r = runs.copy()
    day = sorted(r.race_date.unique())[10]
    hit = (r.race_date == day) & (r.track == "Hove") & (r.won == 1)
    r.loc[hit, ["sp_decimal", "sp_p", "sp_p_norm"]] = np.nan         # the day's Hove winners unpriced
    t2, n2 = cells.build(r, None, None)
    assert n2["results_with_no_sp_left_out_of_the_price_columns"] == int(hit.sum()) > 0
    assert notes["results_with_no_sp_left_out_of_the_price_columns"] == 0
    lsp = np.log(r.sp_decimal)
    r["_c"] = lsp - lsp.groupby(r.race_id).transform("mean")
    m = r.merge(t2, on=KEY)
    x = m[(m.race_date > day) & (m.trainer == r.loc[hit, "trainer"].iloc[0])].iloc[-1]
    past = r[(r.race_date < x.race_date) & (r.trainer == x.trainer)]
    priced = past[past.sp_decimal.notna()]
    assert len(priced) < len(past)
    assert x.ch_tr_ae == pytest.approx(np.log((priced.won.sum() + cells.K_AE) / (priced.sp_p_norm.sum() + cells.K_AE)),
                                       abs=1e-5)
    assert x.ch_tr_n == pytest.approx(np.log1p(len(past)), abs=1e-5)
    here = priced[priced.track == x.track]
    assert x.ch_trtrack_sp == pytest.approx(here._c.sum() / (len(here) + cells.K_SP), abs=1e-5)


def test_the_grade_move_reads_the_last_earlier_day(built):
    runs, tab, notes = built
    code = pd.Series(cells.grade_move(runs.reset_index(drop=True)), index=runs.index)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(runs), 80, replace=False):
        x = runs.iloc[i]
        past = runs[(runs.dog_id == x.dog_id) & (runs.race_date < x.race_date)].sort_values("t")
        if not len(past):
            want = "first"
        else:
            p = past.iloc[-1]
            want = ("newtrack" if p.track != x.track else "other" if p.grade_family != x.grade_family
                    else "up" if x.grade < p.grade else "down" if x.grade > p.grade else "same")
        assert cells.MOVES[int(code.iloc[i])] == want


def test_a_days_results_and_later_days_move_nothing_on_it(built):
    runs, tab, notes = built
    days = sorted(runs.race_date.unique())
    mid, last = days[40], days[-1]
    for day in (mid, last):
        tab_b, _ = cells.build(rewrite_day(runs, day), None, None)
        upto = runs.race_date <= day
        np.testing.assert_allclose(on(tab_b, runs, upto), on(tab, runs, upto), atol=1e-6, equal_nan=True)
        if day == mid:                                               # and the rewrite is seen from the next day on
            after = runs.race_date > day
            assert np.nanmax(np.abs(on(tab_b, runs, after) - on(tab, runs, after))) > 0.01
    # the later days cut off (or, read the other way, added): the days up to ``mid`` are as they were
    cut = runs[runs.race_date <= mid].reset_index(drop=True)
    tab_c, _ = cells.build(cut, None, None)
    np.testing.assert_allclose(on(tab_c, cut, cut.race_date <= mid), on(tab, runs, runs.race_date <= mid),
                               atol=1e-6, equal_nan=True)
    # a second race the same day is not read either: the day's first and last races read the same cells
    tab_d, _ = cells.build(runs[(runs.race_date < mid) | (runs.race_time >= "20:00")].reset_index(drop=True),
                           None, None)
    lateones = (runs.race_date == mid) & (runs.race_time >= "20:00")
    got = runs.loc[lateones, KEY].merge(tab_d, on=KEY, how="left").sort_values(KEY)
    want = runs.loc[lateones, KEY].merge(tab, on=KEY, how="left").sort_values(KEY)
    same_day_free = [c for c in COLS if "move" not in c]             # the move reads only earlier days too
    np.testing.assert_allclose(got[same_day_free].to_numpy(float), want[same_day_free].to_numpy(float), atol=1e-6,
                               equal_nan=True)
    np.testing.assert_allclose(got[COLS].to_numpy(float), want[COLS].to_numpy(float), atol=1e-6, equal_nan=True)


def test_a_card_row_gets_the_features_its_run_gets(built):
    runs, tab, notes = built
    card = cardify_last_day(runs)
    last = card.is_card
    assert last.any() and card.loc[last, "won"].isna().all() and card.loc[last, "sp_decimal"].isna().all()
    tab_c, notes_c = cells.build(card, None, None)
    assert notes_c["card_rows"] == int(last.sum()) and len(tab_c) == len(card)
    got = on(tab_c, card, last)
    assert np.isfinite(got).all()                                    # made from history, every column
    assert np.abs(got).sum() > 0
    np.testing.assert_allclose(got, on(tab, runs, runs.race_date == runs.race_date.max()), atol=1e-6)
    # and through the engine's figures, as build_tables calls it
    tab_f, _ = cells.build(card, core.per_run_figures(card), {})
    np.testing.assert_allclose(tab_f[COLS].to_numpy(float), tab_c[COLS].to_numpy(float), atol=1e-6, equal_nan=True)


def test_the_trainer_is_read_as_known_at_the_time_with_a_fetch_record(built):
    """With a fetch record a past row's trainer counts only where GBGB was read at the time (core.FRESH_DAYS)."""
    runs, tab, notes = built
    days = sorted(runs.race_date.unique())
    fresh = pd.Series(0, index=days)
    t_fresh, n_fresh = cells.build(runs, None, {"fetch_lag": fresh})
    assert n_fresh["trainer_read"].startswith("at the time")
    np.testing.assert_allclose(t_fresh[COLS].to_numpy(float), tab[COLS].to_numpy(float), atol=1e-6, equal_nan=True)
    stale = pd.Series(400, index=days)                               # every day backfilled: no trainer history
    t_stale, _ = cells.build(runs, None, {"fetch_lag": stale})
    tr = [c for c in COLS if c.startswith(("ch_tr_", "ch_trtrack_"))]
    assert (t_stale[tr].to_numpy() == 0).all()
    rest = [c for c in COLS if c not in tr]
    np.testing.assert_allclose(t_stale[rest].to_numpy(float), tab[rest].to_numpy(float), atol=1e-6, equal_nan=True)
