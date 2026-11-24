"""greyhound/pit/lastrace.py: the last race as an event with a market. A market that shortens a dog after a good
finish is found by the reaction map (and is not found where none was planted); the columns are what a slow, row by
row reading of the specification gives; a day's own results, a middle day's results and the days after a day move
nothing on that day; a card row gets what its run had, from history; a second run the same day does not read the
first; one row a (race, dog), every column registered. Planted data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import lastrace
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

COLS = list(lastrace.COLUMNS)
REACT = 1.0                                    # log-odds the market adds a dog that won last time over one that was 6th


def synth(days=100, seed=3, react=REACT, twice=True):
    """One track, 60 dogs, 10 races of 6 a day. The SP is the dog's ability plus ``react`` x (3.5 - its last finish)
    / 5: the market follows the last finishing position, whatever the dog is. On every seventh day dog 1 runs a
    second race late in the evening (with five dogs that never run otherwise)."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.6, 61)
    last_pos = np.full(61, np.nan)
    rows, rid = [], 7000

    def race(d, day, hhmm, field, new_pos=True):
        nonlocal rid
        rid += 1
        bonus = np.where(np.isnan(last_pos[field]), 0.0, react * (3.5 - last_pos[field]) / 5.0) if max(field) <= 60 else np.zeros(len(field))
        sk = skill[field] if max(field) <= 60 else np.zeros(len(field))
        u = sk + bonus
        p = np.exp(u) / np.exp(u).sum()
        perf = sk + rng.normal(0, 1.0, len(field))
        pos = np.argsort(np.argsort(-perf)) + 1
        early = np.argsort(np.argsort(rng.normal(0, 1, len(field)) - 0.3 * perf))
        t = 29.5 - perf * 0.08
        for k, dog in enumerate(field):
            rows.append({"race_date": day, "race_time": hhmm, "meeting_id": d, "track": "Hove", "race_id": rid,
                         "race_class": f"A{1 + (rid % 3)}", "distance_m": 500.0 if rid % 4 else 285.0, "trap": k + 1,
                         "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                         "sp_decimal": float(1.0 / (1.15 * p[k])), "market_pos": int(np.argsort(np.argsort(-p))[k]) + 1,
                         "position": int(pos[k]), "beaten_lengths": 0.5, "sectional": float(4.4 + 0.05 * early[k]),
                         "run_time": float(round(t[k] - (12.6 if rid % 4 == 0 else 0.0), 2)),
                         "adjusted_time": float(round(t[k] - (12.6 if rid % 4 == 0 else 0.0), 2)), "weight_kg": 30.0,
                         "comment": ("Crd1" if pos[k] > 4 else "Led1" if pos[k] == 1 else "RanOn")})
        if new_pos and max(field) <= 60:
            last_pos[field] = pos

    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        dogs = rng.permutation(np.arange(1, 61))
        for r in range(10):
            race(d, day, f"{11 + r}:00", dogs[6 * r:6 * r + 6])
        if twice and d % 7 == 6:
            race(d, day, "22:00", np.array([1, 61, 62, 63, 64, 65]), new_pos=False)
    return clean(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = lastrace.build(runs)
    return runs, runs.merge(tab, on=["race_id", "dog_id"]), tab, notes


def _same(a: pd.DataFrame, b: pd.DataFrame, keys: pd.DataFrame) -> None:
    x = keys.merge(a, on=["race_id", "dog_id"], how="left")[COLS].to_numpy()
    y = keys.merge(b, on=["race_id", "dog_id"], how="left")[COLS].to_numpy()
    assert x.shape == y.shape and np.array_equal(x, y, equal_nan=True)


def test_every_column_is_registered_and_no_key_twice(built):
    runs, m, tab, notes = built
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS and all(c.startswith(lastrace.PREFIX) for c in COLS)
    assert len(tab) == len(runs) == len(m) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert (tab[COLS].dtypes == np.float32).all()
    assert notes["rows"] == len(runs) and notes["rows_with_a_last_run"] > 0.95 and notes["months_with_a_map"] >= 2
    late = m[m.race_date >= "2025-03-01"]
    late = late[late.dog_id <= 60]
    assert late[COLS].notna().mean().min() > 0.6                     # every column is made (grade_chg: same family)
    assert not hasattr(lastrace, "CARD_UNSAFE") or not lastrace.CARD_UNSAFE


def test_the_planted_reaction_is_in_the_map_and_none_where_none_was_planted(built):
    runs, m, _, _ = built
    late = m[(m.race_date >= "2025-03-01") & (m.dog_id <= 60)]

    def first_less_last(mm):
        """The map's move after a win less its move after a sixth, read off the rows whose last run was each."""
        lp = lastpos(mm)
        return float(mm.lr_exp_react[lp == 1].mean() - mm.lr_exp_react[lp == 6].mean())

    def lastpos(mm):
        r = runs_of[id(mm)]
        day_last = r.sort_values(["dog_id", "t"]).drop_duplicates(["dog_id", "race_date"], keep="last")
        prev = day_last.groupby("dog_id").position.shift(1)
        j = mm[["dog_id", "race_date"]].merge(day_last.assign(prev=prev)[["dog_id", "race_date", "prev"]],
                                              on=["dog_id", "race_date"], how="left")
        return j.prev.to_numpy()

    plain = synth(react=0.0)
    tab0, _ = lastrace.build(plain)
    m0 = plain.merge(tab0, on=["race_id", "dog_id"])
    late0 = m0[(m0.race_date >= "2025-03-01") & (m0.dog_id <= 60)]
    runs_of = {id(late): runs, id(late0): plain}
    planted, none = first_less_last(late), first_less_last(late0)
    # the market adds REACT to a last-time winner over a last-time sixth; with nothing planted a winner (who was a
    # short price in a weak field more often than not) is, if anything, a longer price next time
    # (seed 3: 0.42 planted, -0.16 with none; the cells are shrunk and the dog's bonus before that run comes off)
    assert planted > 0.3 * REACT and none < 0.0 and planted - none > 0.4 * REACT
    # and the map, fitted on earlier months, forecasts this row's own move in its SP share (never read)
    lsp = np.log(late.sp_p_norm * late.field)
    prev_lsp = late.lr_sp_vs_prev                                     # only to pick rows with a history
    r = runs.assign(lsp=np.log(runs.sp_p_norm * runs.field)).sort_values(["dog_id", "t"])
    day_last = r.drop_duplicates(["dog_id", "race_date"], keep="last")
    day_last = day_last.assign(prev=day_last.groupby("dog_id").lsp.shift(1))
    j = late[["dog_id", "race_date"]].merge(day_last[["dog_id", "race_date", "prev"]], on=["dog_id", "race_date"], how="left")
    move = lsp.to_numpy() - j.prev.to_numpy()
    ok = ~np.isnan(move) & late.lr_exp_react.notna().to_numpy() & prev_lsp.notna().to_numpy()
    assert np.corrcoef(move[ok], late.lr_exp_react.to_numpy()[ok])[0, 1] > 0.3


def test_the_columns_are_the_specification_read_a_row_at_a_time(built):
    runs, m, _, _ = built
    r = runs.assign(lsp=np.log(runs.sp_p_norm * runs.field), d=pd.to_datetime(runs.race_date))
    pick = m[(m.race_date >= "2025-03-05") & (m.dog_id <= 60)].sample(25, random_state=1)
    pick = pd.concat([pick, m[(m.dog_id == 1) & (m.race_date >= "2025-03-05")].head(6)])
    assert len(pick) == 31
    for _, row in pick.iterrows():
        h = r[(r.dog_id == row.dog_id) & (r.race_date < row.race_date)].sort_values("t")
        h = h.drop_duplicates("race_date", keep="last")              # a day is read by its last run
        la, l2 = h.iloc[-1], h.iloc[-2]
        race = r[r.race_id == la.race_id]
        want = {"lr_mkt_beat": la.market_pos - la.position, "lr_ae_last": la.won - la.sp_p_norm,
                "lr_trap_chg": row.trap - la.trap, "lr_field_chg": row.field - la.field,
                "lr_dist_chg": np.log(row.distance_m / la.distance_m), "lr_track_chg": 0.0,
                "lr_pos_contrast": l2.position - la.position,
                "lr_pos_gain": race.sectional.rank(method="min")[race.dog_id == row.dog_id].iloc[0] - la.position,
                "lr_sp_vs_prev": la.lsp - h.iloc[-4:-1].lsp.mean()}
        home = race.run_time - race.sectional
        want["lr_fin_speed"] = (home.mean() - home[race.dog_id == row.dog_id].iloc[0]) / 0.08
        opp = []
        for q in race.dog_id[race.dog_id != row.dog_id]:
            hh = r[(r.dog_id == q) & (r.race_date < la.race_date)].sort_values("t").drop_duplicates("race_date", keep="last")
            if len(hh) >= lastrace.MIN_PRIOR:
                opp.append(hh.lsp.mean())
        want["lr_opp_fancied"] = np.mean(opp) if len(opp) >= 2 else np.nan
        # the course standard before the last race's day and its winner against it
        w = r[(r.position == 1) & (r.track == la.track) & (r.distance_m == la.distance_m) & (r.race_date < la.race_date)]
        wg = w[w.race_class == la.race_class]
        mc = w.adjusted_time.mean() if len(w) >= lastrace.MIN_STD else np.nan
        mg = wg.adjusted_time.mean() if len(wg) else mc
        std = (len(wg) * mg + lastrace.K_STD * mc) / (len(wg) + lastrace.K_STD)
        want["lr_race_speed"] = (std - race[race.position == 1].adjusted_time.iloc[0]) / 0.08
        # the reaction map of the row's month: pairs whose NEXT run is dated before the first of the month
        p = r[r.race_date < row.race_date[:7] + "-01"].sort_values(["dog_id", "t"]).drop_duplicates(["dog_id", "race_date"], keep="last")
        g = p.groupby("dog_id")
        p = p.assign(pp=g.position.shift(1), pm=g.market_pos.shift(1), dl=p.lsp - g.lsp.shift(1)).dropna(subset=["pp", "pm", "dl"])
        c = p[(p.pp == la.position) & (p.pm == la.market_pos)].dl
        want["lr_exp_react"] = (c.sum() + lastrace.K_CELL * p[p.pp == la.position].dl.mean()) / (len(c) + lastrace.K_CELL)
        c2 = p[(p.pp == l2.position) & (p.pm == l2.market_pos)].dl
        want["lr_react_resid"] = (la.lsp - l2.lsp) - (c2.sum() + lastrace.K_CELL * p[p.pp == l2.position].dl.mean()) / (len(c2) + lastrace.K_CELL)
        for k, v in want.items():
            assert (np.isnan(v) and np.isnan(row[k])) or abs(v - row[k]) < 2e-4, (k, v, row[k], row.race_id, row.dog_id)


def test_the_days_own_results_move_nothing(built):
    runs, _, tab, _ = built
    last = runs.race_date.max()
    keys = runs.loc[runs.race_date == last, ["race_id", "dog_id"]]
    tab2, _ = lastrace.build(perturb_last_day(runs))
    _same(tab, tab2, keys)
    _same(tab, tab2, runs[["race_id", "dog_id"]])                    # nor any earlier row


def test_a_middle_days_results_move_nothing_on_it_or_before_and_do_move_later_days(built):
    runs, _, tab, _ = built
    mid = "2025-03-11"                                               # dog 1 runs twice on it (day 69)
    assert (runs[(runs.race_date == mid) & (runs.dog_id == 1)].shape[0]) == 2
    r = runs.copy()
    on = r.race_date == mid
    r.loc[on, "position"] = r.loc[on, "field"] + 1 - r.loc[on, "position"]
    r.loc[on, ["adjusted_time", "run_time"]] += 0.4
    r.loc[on, "sectional"] = 4.5 + 0.05 * r.loc[on, "trap"]
    r.loc[on, "comment"] = "SAw,Bmp1"
    r.loc[on, "sp_decimal"] = 3.0 + r.loc[on, "trap"]
    r.loc[on, "market_pos"] = r.loc[on, "trap"]
    r["won"] = (r.position == 1).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    tab2, _ = lastrace.build(r)
    _same(tab, tab2, runs.loc[runs.race_date <= mid, ["race_id", "dog_id"]])
    later = runs.loc[runs.race_date == "2025-03-12", ["race_id", "dog_id"]]      # every dog runs every day
    a, b = later.merge(tab, on=["race_id", "dog_id"]), later.merge(tab2, on=["race_id", "dog_id"])
    assert (a.lr_mkt_beat.fillna(-99) != b.lr_mkt_beat.fillna(-99)).mean() > 0.5
    # the maps are of the months before: the rest of March reads the same maps, April reads the changed pairs
    march = (runs.race_date > mid) & (runs.race_date < "2025-04-01")
    untouched = runs.loc[march, ["race_id", "dog_id"]].merge(tab, on=["race_id", "dog_id"])
    touched = runs.loc[march, ["race_id", "dog_id"]].merge(tab2, on=["race_id", "dog_id"])
    assert set(np.round(touched.lr_exp_react.dropna(), 6)) <= set(np.round(untouched.lr_exp_react.dropna(), 6))
    april = runs.loc[runs.race_date >= "2025-04-01", ["race_id", "dog_id"]]
    assert not np.array_equal(april.merge(tab, on=["race_id", "dog_id"]).lr_exp_react.to_numpy(),
                              april.merge(tab2, on=["race_id", "dog_id"]).lr_exp_react.to_numpy(), equal_nan=True)


def test_the_days_after_a_day_move_nothing_on_it(built):
    runs, _, tab, _ = built
    for cut in ("2025-02-20", "2025-03-12", "2025-04-02"):
        short = runs[runs.race_date <= cut].reset_index(drop=True)
        tab2, _ = lastrace.build(short)
        assert len(tab2) == len(short)
        _same(tab, tab2, short[["race_id", "dog_id"]])


def test_a_card_row_gets_the_features_its_run_had(built):
    runs, _, tab, _ = built
    card = cardify_last_day(runs)
    on = card.is_card.to_numpy(bool)
    assert on.sum() == (runs.race_date == runs.race_date.max()).sum() and card.position[on].isna().all()
    tab2, notes = lastrace.build(card)
    keys = card.loc[on, ["race_id", "dog_id"]]
    _same(tab, tab2, keys)
    got = keys.merge(tab2, on=["race_id", "dog_id"])
    assert got[["lr_mkt_beat", "lr_exp_react", "lr_exp_ae", "lr_exp_ae_cmt", "lr_trap_chg", "lr_opp_fancied",
                "lr_race_speed", "lr_pos_gain"]].notna().mean().min() > 0.9
    assert notes["card_rows"] == on.sum() and notes["card_rows_with_a_last_run"] > 0.9
    # a card alone, with no history, is a table of rows with nothing in them (and no error)
    alone, _ = lastrace.build(card[on].reset_index(drop=True))
    assert len(alone) == on.sum() and alone[COLS].isna().all().all()


def test_a_second_run_the_same_day_does_not_read_the_first(built):
    runs, m, _, _ = built
    one = m[(m.dog_id == 1) & (m.race_date >= "2025-02-01")].sort_values("t")
    two = one[one.groupby("race_date").race_id.transform("size") == 2]
    assert len(two) >= 10
    for _, g in two.groupby("race_date"):
        first, second = g.iloc[0], g.iloc[1]
        assert np.array_equal(first[["lr_mkt_beat", "lr_ae_last", "lr_exp_react", "lr_race_speed", "lr_opp_fancied"]].to_numpy(float),
                              second[["lr_mkt_beat", "lr_ae_last", "lr_exp_react", "lr_race_speed", "lr_opp_fancied"]].to_numpy(float),
                              equal_nan=True)
        nxt = one[one.race_date > g.race_date.iloc[0]].iloc[0]       # the next day reads the day's LAST run
        assert nxt.lr_mkt_beat == second.market_pos - second.position
        assert nxt.lr_field_chg == nxt.field - second.field


def test_no_map_before_there_are_pairs_enough(built):
    runs, m, _, notes = built
    jan = m[m.race_date < "2025-02-01"]
    assert jan.lr_exp_react.isna().all() and jan.lr_exp_ae_cmt.isna().all() and jan.lr_react_resid.isna().all()
    assert jan[jan.race_date > "2025-01-10"].lr_mkt_beat.notna().mean() > 0.9
    assert notes["first_month_with_a_map"] == "2025-02"
