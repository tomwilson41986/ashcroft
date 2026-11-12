"""greyhound/pit/salience.py: each printed attribute against today's field. A dog planted with the field's best last
time, best time, last position, wins and sectional is the race's standout and its rivals are not; the numbers are the
specification's, read a race at a time; a day's own results move nothing on that day or before it, and the days after
a day move nothing on it; a card row gets what its run had, from history; one row a race and dog; the field is every
row of the race; a time at another track or distance is not read; today's weight is not read."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import salience
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day
from tests.test_greyhound_pit import synth as pit_synth

COLS = list(salience.COLUMNS)
KEY = ["race_id", "dog_id"]


def planted(days=30, seed=3, star=1):
    """One track and trip, 36 dogs of equal ability; six races of six a night, dog ``star`` always in a race. From
    day 10 on the star is planted 0.40 s faster, 0.10 s faster to the bend and wins every race."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 7000
    for d in range(days):
        day = (pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        dogs = rng.permutation(np.arange(1, 37))
        for r in range(6):
            field = dogs[6 * r:6 * r + 6]
            rid += 1
            t = 29.5 + rng.normal(0, 0.08, 6)
            s = 4.50 + rng.normal(0, 0.03, 6)
            if d >= 10:
                t = np.where(field == star, 29.5 - 0.40, t)
                s = np.where(field == star, 4.50 - 0.10, s)
            pos = np.argsort(np.argsort(t)) + 1
            for k, dog in enumerate(field):
                rows.append({"race_date": day, "race_time": f"{18 + r}:00", "meeting_id": d, "track": "Hove",
                             "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": k + 1,
                             "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                             "sp_decimal": 6.0, "position": int(pos[k]), "beaten_lengths": 1.0,
                             "sectional": float(round(s[k], 2)), "run_time": float(round(t[k], 2)),
                             "adjusted_time": float(round(t[k], 2)),
                             "weight_kg": float(round(30.0 + rng.normal(0, 0.2), 1)), "comment": "QAw"})
    return clean(pd.DataFrame(rows))


def table(runs):
    tab, notes = salience.build(runs)
    return tab, notes


def test_every_column_is_registered_and_keys_are_unique():
    runs = pit_synth()
    tab, notes = table(runs)
    assert list(tab.columns) == KEY + COLS and all(c.startswith(salience.PREFIX) for c in COLS)
    assert (tab[COLS].dtypes == np.float32).all()
    assert len(tab) == len(runs) == notes["rows"]
    assert not tab.duplicated(KEY).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    assert (tab.dog_id.to_numpy() == runs.dog_id.to_numpy()).all()
    assert notes["rows_with_any_form_attribute"] > 0.8


def test_the_planted_dog_stands_out():
    runs = planted()
    tab, _ = table(runs)
    m = runs.merge(tab, on=KEY)
    d = (pd.to_datetime(m.race_date) - pd.Timestamp("2025-03-01")).dt.days
    late = m[d >= 20]                                           # ten planted days on the card by then
    one = late[late.dog_id == 1]
    rivals = late[late.race_id.isin(one.race_id) & (late.dog_id != 1)]
    assert len(one) == 10
    assert (one.sl_is_standout == 1).all() and (rivals.sl_is_standout == 0).all()
    assert (one.sl_n_top >= 3).all() and (one.sl_n_top > rivals.groupby("race_id").sl_n_top.max().to_numpy()).all()
    assert (one.sl_lt_best == 0).all() and (one.sl_bt3_best == 0).all() and (one.sl_sec3_best == 0).all()
    assert (rivals.sl_bt3_best < 0).all() and (rivals.sl_lt_best < 0).all()
    assert one.sl_bt3_med.min() > 1.0 and one.sl_max_for.min() > 1.0 and one.sl_lead.min() > 1.0
    assert one.sl_w6_med.min() > 1.0 and one.sl_lp_med.min() >= 0
    assert one.sl_max_for.mean() > rivals.sl_max_for.mean() + 1.0
    # before the plant the star is a dog like the others
    early = m[(d >= 4) & (d < 10) & (m.dog_id == 1)]
    assert early.sl_is_standout.mean() < 0.5 and early.sl_bt3_med.abs().mean() < 1.5


def slow(runs, race_id):
    """The specification read a row at a time for one race: each dog's attributes from its earlier days, then the
    field's median, sample sd (floored, at least three values), best and best under the top."""
    r = runs.assign(_d=pd.to_datetime(runs.race_date))
    race = r[r.race_id == race_id]
    vals = {}
    for _, row in race.iterrows():
        h = r[(r.dog_id == row.dog_id) & (r._d < row._d)].sort_values("t")
        v = dict.fromkeys(salience.FLOOR, np.nan)
        if len(h):
            last = h.iloc[-1]
            v["lp"] = last.position if last.position >= 1 else np.nan
            v["w6"] = float((h.position.tail(6) == 1).sum())
            v["dw"] = last.weight_kg - h.iloc[-2].weight_kg if len(h) > 1 else np.nan
            c = h[(h.track == row.track) & (h.distance_m == row.distance_m)]
            if len(c):
                v["lt"] = c.adjusted_time.iloc[-1]
                v["bt3"] = c.adjusted_time.tail(3).min()
                v["sec3"] = c.sectional.tail(3).mean()
        vals[row.dog_id] = v
    out = {dog: {} for dog in vals}
    tops = {dog: 0 for dog in vals}
    for a, floor in salience.FLOOR.items():
        x = pd.Series({dog: v[a] * salience.SIGN[a] for dog, v in vals.items()}).dropna()
        sd = max(x.std(), floor) if len(x) >= 3 else np.nan
        under = x[x < x.max()] if len(x) else x
        for dog in vals:
            has = dog in x.index
            out[dog][f"{a}_med"] = (x[dog] - x.median()) / sd if has else np.nan
            out[dog][f"{a}_best"] = (x[dog] - x.max()) / sd if has else np.nan
            top = has and x[dog] == x.max() and len(under) > 0
            out[dog][f"{a}_lead"] = (x[dog] - under.max()) / sd if top else np.nan
            if a in salience.FORM:
                tops[dog] += int(top)
    res = {}
    for dog in vals:
        z = [out[dog][f"{a}_med"] for a in salience.FORM]
        some = bool(np.isfinite(z).any())
        leads = [out[dog][f"{a}_lead"] for a in salience.FORM]
        res[dog] = {"sl_lt_med": out[dog]["lt_med"], "sl_lt_best": out[dog]["lt_best"],
                    "sl_bt3_med": out[dog]["bt3_med"], "sl_bt3_best": out[dog]["bt3_best"],
                    "sl_lp_med": out[dog]["lp_med"], "sl_w6_med": out[dog]["w6_med"],
                    "sl_sec3_med": out[dog]["sec3_med"], "sl_sec3_best": out[dog]["sec3_best"],
                    "sl_dw_med": out[dog]["dw_med"], "sl_dw_abs": abs(out[dog]["dw_med"]),
                    "sl_max_for": np.nanmax(z) if some else np.nan,
                    "sl_max_against": np.nanmin(z) if some else np.nan,
                    "sl_n_top": float(tops[dog]) if some else np.nan,
                    "sl_lead": (np.nanmax(leads) if np.isfinite(leads).any() else 0.0) if some else np.nan}
    have = [t for dog, t in tops.items() if np.isfinite(res[dog]["sl_n_top"])]
    for dog in vals:
        if not have:
            res[dog]["sl_is_standout"] = np.nan
        else:
            mx = max(have)
            res[dog]["sl_is_standout"] = float(mx >= salience.STANDOUT and have.count(mx) == 1
                                               and np.isfinite(res[dog]["sl_n_top"]) and tops[dog] == mx)
    return pd.DataFrame(res).T[COLS]


def test_the_table_is_the_specification_read_a_row_at_a_time():
    runs = pit_synth(days=25, seed=5)                            # two tracks, dogs twice a day, fields overlapping
    tab, _ = table(runs)
    ids = runs.race_id.drop_duplicates()
    for race_id in list(ids.iloc[::23]) + list(ids.tail(3)):
        want = slow(runs, race_id)
        got = tab[tab.race_id == race_id].set_index("dog_id")[COLS].reindex(want.index)
        np.testing.assert_allclose(got.to_numpy(float), want.to_numpy(float), rtol=1e-4, atol=1e-4, equal_nan=True)


def test_the_days_own_results_move_nothing():
    runs = pit_synth()
    last = runs.race_date.max()
    a, _ = table(runs)
    b, _ = table(perturb_last_day(runs))
    keys = runs.loc[runs.race_date == last, KEY]
    x, y = keys.merge(a, on=KEY), keys.merge(b, on=KEY)
    assert len(x) == len(keys) and x[COLS].notna().any(axis=1).mean() > 0.8
    pd.testing.assert_frame_equal(x, y)


def test_a_middle_days_results_move_nothing_on_it_or_before_and_later_days_move_nothing_on_it():
    runs = pit_synth()
    days = sorted(runs.race_date.unique())
    mid = days[len(days) // 2]
    a, _ = table(runs)
    upto = runs.race_date <= mid
    # the days after the middle day cut off: nothing on or before it moves
    cut, _ = table(runs[upto].reset_index(drop=True))
    keys = runs.loc[upto, KEY]
    pd.testing.assert_frame_equal(keys.merge(a, on=KEY), keys.merge(cut, on=KEY))
    # the middle day's results rewritten: it and the days before are unmoved, the days after do move
    r = runs.copy()
    on = r.race_date == mid
    r.loc[on, "position"] = 7 - r.loc[on, "position"]
    r.loc[on, "adjusted_time"] += 0.7
    r.loc[on, "sectional"] += 0.3
    r.loc[on, "weight_kg"] += 1.5
    r["won"] = (r.position == 1).astype(float)
    b, _ = table(r)
    pd.testing.assert_frame_equal(keys.merge(a, on=KEY), keys.merge(b, on=KEY))
    after = runs.loc[~upto, KEY]
    x, y = after.merge(a, on=KEY)[COLS].to_numpy(float), after.merge(b, on=KEY)[COLS].to_numpy(float)
    assert (~np.isclose(x, y, equal_nan=True)).any()


def test_a_card_row_gets_the_features_its_run_had():
    runs = pit_synth()
    last = runs.race_date.max()
    card = cardify_last_day(runs)
    assert card.loc[card.race_date == last, "is_card"].all() and card.loc[card.race_date == last, "position"].isna().all()
    a, _ = table(runs)
    b, _ = table(card)
    keys = runs.loc[runs.race_date == last, KEY]
    x, y = keys.merge(a, on=KEY), keys.merge(b, on=KEY)
    assert len(y) == len(keys) and y[COLS].notna().any(axis=1).mean() > 0.8
    pd.testing.assert_frame_equal(x, y)
    # and a card for a day on which no dog has run before has no features, rather than made-up ones
    first = card[card.race_date == card.race_date.min()].assign(is_card=True)
    c, _ = table(clean(first.drop(columns=["won", "placed2", "t", "field"], errors="ignore")))
    assert len(c) == len(first) and c[[k for k in COLS if k != "sl_is_standout"]].isna().all().all()


def test_the_field_is_every_row_of_the_race_and_other_courses_are_not_read():
    runs = planted()
    last = runs.race_date.max()
    race = runs.loc[(runs.race_date == last) & (runs.dog_id == 1), "race_id"].iloc[0]
    a, _ = table(runs)
    # a runner with no price is still in the field: its row's presence is what counts
    r = runs.copy()
    r.loc[r.race_id == race, "sp_decimal"] = np.nan
    b, _ = table(r)
    pd.testing.assert_frame_equal(a[a.race_id == race], b[b.race_id == race])
    # drop a rival's row and the field, so the others' numbers, change
    rival = runs.loc[(runs.race_id == race) & (runs.dog_id != 1), "dog_id"].iloc[0]
    c, _ = table(runs[~((runs.race_id == race) & (runs.dog_id == rival))].reset_index(drop=True))
    x = a[(a.race_id == race) & (a.dog_id != rival)][COLS].to_numpy(float)
    y = c[c.race_id == race][COLS].to_numpy(float)
    assert x.shape == y.shape and (~np.isclose(x, y, equal_nan=True)).any()
    # the race moved to another trip: no dog has a time there, the positions and wins still read
    r = runs.copy()
    r.loc[r.race_id == race, "distance_m"] = 270.0
    d, _ = table(r)
    d = d[d.race_id == race]
    assert d[["sl_lt_med", "sl_bt3_med", "sl_sec3_med"]].isna().all().all() and d.sl_w6_med.notna().all()


def test_todays_weight_is_not_read():
    runs = pit_synth()
    last = runs.race_date.max()
    r = runs.copy()
    r.loc[r.race_date == last, "weight_kg"] += np.arange((r.race_date == last).sum()) % 3
    a, _ = table(runs)
    b, _ = table(r)
    pd.testing.assert_frame_equal(a, b)
    assert not getattr(salience, "CARD_UNSAFE", ())
