"""In-day features: only races run earlier the same day, and never the race itself."""

import numpy as np
import pandas as pd
import pytest

from model.inday_features import INDAY_FEATURES, add_inday_features, earlier_sums, early_position, off_minutes

TIMES = ["1.30", "2.05", "2.40", "3.15", "3.50"]


def _card(low_draws_win=True, date="2025-06-01", track="Chester", trainer_first="T"):
    """Five sprints at one meeting. In each, stall 1 finishes first although it is the 8.0 outsider."""
    rows = []
    for r, t in enumerate(TIMES):
        for s in range(1, 7):
            bsp = [8.0, 2.5, 4.0, 6.0, 10.0, 15.0][s - 1]
            pos = s if low_draws_win else 7 - s
            rows.append({"race_date": date, "race_time": t, "track": track, "horse_name": f"{track}{r}{s}",
                         "trainer": trainer_first if s == 1 else f"tr{s}", "jockey_name": f"j{s}",
                         "bfsp": bsp, "placing_numerical": pos, "stall": s, "number_of_runners": 6,
                         "dist_furlongs": 5.0,
                         "comment": "made all" if s == 1 else ("held up in rear" if s >= 5 else "tracked leaders")})
    return pd.DataFrame(rows)


def _feats(df, **kw):
    out, names = add_inday_features(df, **kw)
    return out, names


def test_first_race_sees_nothing_and_later_races_see_the_bias():
    out, names = _feats(_card())
    assert set(names) == set(INDAY_FEATURES)
    first = out[out["race_time"] == "1.30"]
    assert (first["id_race_index"] == 0).all()
    assert (first["id_draw_n"] == 0).all() and (first["id_draw_slope"] == 0).all()
    last = out[out["race_time"] == "3.50"]
    assert (last["id_race_index"] == 4).all()
    # low draws have beaten their market order all afternoon: negative slope on centred draw
    assert (last["id_draw_slope"] < 0).all()
    low = last[last["stall"] == 1].iloc[0]
    high = last[last["stall"] == 6].iloc[0]
    assert low["id_draw_edge_mine"] > 0 > high["id_draw_edge_mine"]
    assert low["id_draw_ae_mine"] > 0                        # the 8.0 shots in the low third kept winning
    # front-runners too: the "made all" runner is the one that kept winning
    assert (last["id_pace_slope"] > 0).all()
    assert (last["id_front_ae"] > 0).all()
    # trainer T's earlier runners today all won at 8.0
    t_row = last[last["trainer"] == "T"].iloc[0]
    assert t_row["id_trainer_runs_today"] == 4
    assert t_row["id_trainer_ae_today"] > 0
    assert t_row["id_trainer_bmr_today"] > 0


def test_the_bias_reverses_sign_with_the_results():
    out, _ = _feats(_card(low_draws_win=False))
    last = out[out["race_time"] == "3.50"]
    assert (last["id_draw_slope"] > 0).all()


@pytest.mark.parametrize("target", TIMES)
def test_nothing_from_this_race_or_later_moves_a_feature(target):
    d = _card()
    out1, names = _feats(d)
    d2 = d.copy()
    t_target = off_minutes(pd.Series([target])).iloc[0]
    later_or_same = off_minutes(d2["race_time"]) >= t_target
    rng = np.random.default_rng(0)
    for (rt,), g in d2[later_or_same].groupby(["race_time"]):
        d2.loc[g.index, "placing_numerical"] = rng.permutation(np.arange(1, 7))
        d2.loc[g.index, "bfsp"] = rng.uniform(1.5, 30, size=len(g))
        d2.loc[g.index, "comment"] = rng.choice(["led", "held up", "prominent", ""], size=len(g))
    out2, _ = _feats(d2)
    sel = d["race_time"] == target
    for c in names:
        a, b = out1.loc[sel, c].to_numpy(), out2.loc[sel, c].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), c


def test_other_meetings_move_only_the_connection_features_and_respect_the_gap():
    here = _card()
    # trainer T also runs at Kempton at 2.00 (before the 2.40 by 40 min) and at 2.35 (only 5 min before)
    away = pd.DataFrame([
        {"race_date": "2025-06-01", "race_time": "2.00", "track": "Kempton", "horse_name": "kA",
         "trainer": "T", "jockey_name": "jk", "bfsp": 20.0, "placing_numerical": 1, "stall": 1,
         "number_of_runners": 2, "dist_furlongs": 6.0, "comment": "led"},
        {"race_date": "2025-06-01", "race_time": "2.00", "track": "Kempton", "horse_name": "kB",
         "trainer": "X", "jockey_name": "jx", "bfsp": 1.05, "placing_numerical": 2, "stall": 2,
         "number_of_runners": 2, "dist_furlongs": 6.0, "comment": "held up"},
        {"race_date": "2025-06-01", "race_time": "2.35", "track": "Kempton", "horse_name": "kC",
         "trainer": "T", "jockey_name": "jk", "bfsp": 30.0, "placing_numerical": 1, "stall": 1,
         "number_of_runners": 2, "dist_furlongs": 6.0, "comment": "led"},
        {"race_date": "2025-06-01", "race_time": "2.35", "track": "Kempton", "horse_name": "kD",
         "trainer": "Y", "jockey_name": "jy", "bfsp": 1.04, "placing_numerical": 2, "stall": 2,
         "number_of_runners": 2, "dist_furlongs": 6.0, "comment": "held up"},
    ])
    out_alone, names = _feats(here)
    out_both, _ = _feats(pd.concat([here, away], ignore_index=True))
    row = lambda o: o[(o["track"] == "Chester") & (o["race_time"] == "2.40") & (o["trainer"] == "T")].iloc[0]
    a, b = row(out_alone), row(out_both)
    # the 2.00 counts (two Chester races + one Kempton), the 2.35 does not: it is within the gap
    assert a["id_trainer_runs_today"] == 2 and b["id_trainer_runs_today"] == 3
    for c in names:
        if not c.startswith(("id_trainer", "id_jockey")):
            assert np.array_equal([a[c]], [b[c]], equal_nan=True), c


def test_an_unparsed_off_time_is_neither_source_nor_target():
    d = _card()
    d.loc[d["race_time"] == "3.15", "race_time"] = "TBC"
    out, names = _feats(d)
    bad = out[out["race_time"] == "TBC"]
    assert bad[names].isna().all().all()
    last = out[out["race_time"] == "3.50"]
    assert (last["id_race_index"] == 3).all()               # the TBC race is not counted as run


def test_days_do_not_mix():
    d = pd.concat([_card(date="2025-06-01"), _card(date="2025-06-02", low_draws_win=False)], ignore_index=True)
    out, _ = _feats(d)
    first_next_day = out[(out["race_date"] == "2025-06-02") & (out["race_time"] == "1.30")]
    assert (first_next_day["id_race_index"] == 0).all()
    assert (first_next_day["id_draw_n"] == 0).all()
    assert (first_next_day["id_trainer_runs_today"] == 0).all()


def test_earlier_sums_is_exclusive_and_grouped():
    s, n = earlier_sums(["a", "a", "b", "a"], [10.0, 20.0, 5.0, 40.0], {"v": [1.0, 2.0, 100.0, 4.0]},
                        ["a", "a", "a", "b", "c"], [10.0, 25.0, 60.0, 30.0, 30.0], gap=5.0)
    # at 25 with a 5-minute gap both the 10 and the 20 have been run; at 60, all three of "a"
    assert list(n) == [0, 2, 3, 1, 0]
    assert list(s["v"]) == [0.0, 3.0, 7.0, 100.0, 0.0]


def test_early_position_reads_the_first_phrase_and_admits_no_phrase():
    assert early_position("made all, ridden out") == 6.0
    assert early_position("held up in rear, led final 100yds") == 1.0
    assert np.isnan(early_position("ran on well"))
    assert np.isnan(early_position(None))


def test_the_three_day_window_counts_the_last_72_hours_only():
    """A jockey's rides 1 and 2 days back count toward a race today; one 4 days back does not;
    and today's own race never does."""
    def ride(date, time, horse, jockey, bsp, pos):
        return {"race_date": date, "race_time": time, "track": "Wolverhampton", "horse_name": horse,
                "trainer": "t" + horse, "jockey_name": jockey, "bfsp": bsp, "placing_numerical": pos,
                "stall": pos, "number_of_runners": 2, "dist_furlongs": 8.5, "comment": ""}
    rows = []
    for date, won in (("2025-01-06", True), ("2025-01-09", True), ("2025-01-10", False), ("2025-01-11", None)):
        # J rides a 6.0 shot against a 1.2 favourite ridden by K
        pj = 1 if won else 2
        rows += [ride(date, "6.00", f"a{date}", "J", 6.0, pj if won is not None else 1),
                 ride(date, "6.00", f"b{date}", "K", 1.2, (3 - pj) if won is not None else 2)]
    out, names = add_inday_features(pd.DataFrame(rows))
    assert {"id_jockey_ae_3d", "id_jockey_runs_3d", "id_trainer_bmr_3d"} <= set(names)
    j = out[out["jockey_name"] == "J"].set_index("race_date")
    # on the 11th: the 9th (won) and the 10th (lost) are inside 72 hours, the 6th is not
    assert j.loc["2025-01-11", "id_jockey_runs_3d"] == 2
    assert j.loc["2025-01-10", "id_jockey_runs_3d"] == 1          # the 9th only
    assert j.loc["2025-01-09", "id_jockey_runs_3d"] == 0          # the 6th is 3 days back to the minute: out
    assert j.loc["2025-01-10", "id_jockey_ae_3d"] > 0             # a 6.0 winner the day before
    # changing the 11th's own result moves nothing on the 11th
    d2 = pd.DataFrame(rows)
    d2.loc[d2["race_date"] == "2025-01-11", "placing_numerical"] = [2, 1]
    out2, _ = add_inday_features(d2)
    for c in names:
        a = out.loc[out["race_date"] == "2025-01-11", c].to_numpy()
        b = out2.loc[out2["race_date"] == "2025-01-11", c].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), c
