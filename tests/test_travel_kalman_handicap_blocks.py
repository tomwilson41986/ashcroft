"""The travel, Kalman-rating and handicap-angle blocks on small hand-built histories.

Synthetic frames test mechanics only; nothing is trained or evaluated on them.
tests/test_feature_blocks.py runs all three through the generic lag, card, order and
READS tests; these check the values.
"""
import numpy as np
import pandas as pd
import pytest

from model.blocks import handicap_angles, kalman, travel

NEWMARKET, YORK = travel.COURSES["newmarket"][:2], travel.COURSES["york"][:2]


def _run(day, track, horse, trainer="T1", pos=1, runners=8, time="2.00", **kw):
    row = {"race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day), "race_time": time, "track": track,
           "horse_name": horse, "trainer": trainer, "placing_numerical": pos, "number_of_runners": runners,
           "official_rating": 80, "median_or": 75, "pounds": 130, "total_dst_bt": "0" if pos == 1 else "2",
           "dist_furlongs": 8.0, "race_type": "Handicap", "race_name": "A Handicap"}
    row.update(kw)
    row["raceid"] = f"{row['race_date']:%Y-%m-%d}_{row['race_time']}_{row['track']}"
    return row


def _frame(rows):
    return pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------------------------------- travel

def test_a_yard_based_at_newmarket_sent_to_york_travels_the_distance_between_them():
    rows = [_run(d, "Newmarket (Rowley)", f"h{d}") for d in range(1, 15)]     # a base after five, a usual trip
    rows.append(_run(20, "York", "hY"))                                         # after five more
    out = travel.build(_frame(rows))
    york = out.iloc[-1]
    assert york["tv_km"] == pytest.approx(travel.km(*NEWMARKET, *YORK), rel=1e-9)
    assert 200 < york["tv_km"] < 225
    assert york["tv_usual"] == pytest.approx(0.0, abs=1e-6)          # the yard's runners never left home
    assert york["tv_km_vs_usual"] == pytest.approx(york["tv_km"])
    assert york["tv_raid"] == 0.0
    assert york["tv_track_share"] == 0.0                              # never ran at York before
    assert york["tv_n_meeting"] == 1 and york["tv_km_single"] == pytest.approx(york["tv_km"])
    assert np.isnan(out.iloc[0]["tv_km"])                             # no base before five runners


def test_an_irish_yard_at_cheltenham_is_a_raid_and_at_naas_is_not():
    rows = [_run(d, "Leopardstown", f"h{d}", trainer="IRE") for d in range(1, 9)]
    rows += [_run(20, "Cheltenham", "hC", trainer="IRE"), _run(21, "Naas", "hN", trainer="IRE")]
    out = travel.build(_frame(rows))
    assert out.iloc[-2]["tv_raid"] == 1.0
    assert out.iloc[-1]["tv_raid"] == 0.0
    assert out.iloc[-2]["tv_km"] > 300


def test_a_course_with_no_position_has_no_distances_and_does_not_move_the_base():
    rows = [_run(d, "Newmarket", f"h{d}") for d in range(1, 9)]
    rows.append(_run(10, "Nowhere Park", "hX"))
    rows.append(_run(20, "York", "hY"))
    out = travel.build(_frame(rows))
    assert np.isnan(out.iloc[-2]["tv_km"]) and np.isnan(out.iloc[-2]["tv_track_share"])
    assert out.iloc[-1]["tv_km"] == pytest.approx(travel.km(*NEWMARKET, *YORK), rel=1e-9)


def test_the_share_of_the_yards_runners_at_the_course_and_its_runners_at_the_meeting():
    rows = [_run(d, "Newmarket", f"h{d}") for d in range(1, 9)] + [_run(d, "York", f"y{d}") for d in (9, 10)]
    rows += [_run(30, "York", "a", time="1.00"), _run(30, "York", "b", time="3.00")]
    out = travel.build(_frame(rows))
    last = out.iloc[-1]
    w = 2.0 ** (-(30 - np.arange(1, 11)) / travel.HALFLIFE_DAYS)       # decayed runners
    assert last["tv_track_share"] == pytest.approx(w[8:].sum() / (w.sum() + 1.0), rel=1e-9)
    assert last["tv_n_meeting"] == 2 and last["tv_km_single"] == 0.0


def test_the_horses_own_trip_is_from_its_last_course_on_an_earlier_day():
    rows = [_run(1, "Newmarket", "h"), _run(5, "York", "h", time="1.00"), _run(5, "York", "h", time="4.00")]
    out = travel.build(_frame(rows))
    assert np.isnan(out.iloc[0]["tv_horse_km"])
    d = travel.km(*NEWMARKET, *YORK)
    assert out.iloc[1]["tv_horse_km"] == pytest.approx(d) and out.iloc[2]["tv_horse_km"] == pytest.approx(d)


def test_course_names_are_read_through_their_variants():
    names = pd.Series(["Newmarket (July)", "Chelmsford City", "Bangor-On-Dee", "Kempton (AW)", "Down Royal"])
    assert list(travel.course_name(names)) == ["newmarket", "chelmsford", "bangor", "kempton", "downroyal"]
    assert all(c in travel.COURSES for c in travel.course_name(names))


# --------------------------------------------------------------------------- kalman

def _two_runs(gap=30):
    # the horse wins a mile race off 80 by a length (2 lb at the trip): its figure is 82
    rows = [_run(1, "York", "h", pos=1, total_dst_bt="0"), _run(1, "York", "r", pos=2, total_dst_bt="1")]
    rows += [_run(1 + gap, "York", "h", pos=3, official_rating=84, median_or=75, total_dst_bt="3")]
    return _frame(rows)


def test_a_first_run_is_rated_at_its_races_median_mark_with_the_full_uncertainty():
    out = kalman.build(_two_runs())
    first = out.iloc[0]
    assert first["kr_rating"] == 75.0 and first["kr_sd"] == pytest.approx(10.0) and first["kr_n"] == 0


def test_the_next_prior_is_the_filters_update_carried_across_the_gap():
    out = kalman.build(_two_runs(gap=30))
    second = out.iloc[2]
    k = kalman.INIT_VAR / (kalman.INIT_VAR + kalman.R)
    mean = 75.0 + k * (82.0 - 75.0)
    var = (1 - k) * kalman.INIT_VAR + kalman.Q * kalman.Q_CAREER_MULT * 30 / 365.0
    assert second["kr_rating"] == pytest.approx(mean)
    assert second["kr_sd"] == pytest.approx(np.sqrt(var))
    assert second["kr_n"] == 1
    assert second["kr_vs_or"] == pytest.approx(mean - 84.0)


def test_two_runs_on_one_day_share_the_days_prior():
    rows = [_run(1, "York", "h", pos=1, total_dst_bt="0"), _run(1, "York", "r", pos=2, total_dst_bt="1"),
            _run(9, "York", "h", time="1.00", pos=1, total_dst_bt="0"),
            _run(9, "York", "q", time="1.00", pos=2, total_dst_bt="4"),
            _run(9, "Ayr", "h", time="5.00", pos=4, total_dst_bt="9")]
    out = kalman.build(_frame(rows))
    a, b = out.iloc[2], out.iloc[4]
    assert a["kr_rating"] == b["kr_rating"] and a["kr_sd"] == b["kr_sd"] and a["kr_n"] == b["kr_n"] == 1


def test_the_within_race_readings():
    rows = [_run(1, "York", h, pos=i + 1, total_dst_bt=str(float(i)), official_rating=70 + 5 * i)
            for i, h in enumerate("abc")]
    rows += [_run(40, "Ayr", h, time="3.00", pos=i + 1) for i, h in enumerate("abc")]
    out = kalman.build(_frame(rows)).iloc[3:]
    assert out["kr_rank"].tolist() == out["kr_rating"].rank(ascending=False, method="min").tolist()
    assert out["kr_vs_max"].max() == 0.0
    assert out["kr_z"].mean() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- handicap angles

def test_weight_against_the_mark_within_a_handicap():
    rows = [_run(1, "York", "top", official_rating=90, pounds=140, pos=1),
            _run(1, "York", "pen", official_rating=80, pounds=132, pos=2),
            _run(1, "York", "cl", official_rating=80, pounds=127, pos=3)]
    out = handicap_angles.build(_frame(rows))
    assert out["hca_wt_vs_mark"].tolist() == [0.0, 2.0, -3.0]


def test_well_in_after_a_win_off_an_unchanged_mark():
    rows = [_run(1, "York", "h", pos=1, total_dst_bt="0"), _run(1, "York", "r", pos=2, total_dst_bt="5"),
            _run(8, "Ayr", "h", pos=2, total_dst_bt="1")]
    out = handicap_angles.build(_frame(rows)).iloc[2]
    assert out["hca_lr_won"] == 1.0 and out["hca_lr_handicap"] == 1.0 and out["hca_days_since_lr"] == 7
    assert out["hca_lr_win_margin_lbs"] == pytest.approx(10.0)      # five lengths at a mile, 2 lb a length
    assert out["hca_mark_unchanged_after_win"] == 1.0
    assert out["hca_penalty_edge_lbs"] == pytest.approx(10.0 - handicap_angles.ASSUMED_PENALTY_LBS)


def test_marks_against_the_horses_wins_and_a_same_day_run_reads_the_day_before():
    rows = [_run(1, "York", "h", pos=1, official_rating=90), _run(20, "York", "h", pos=1, official_rating=80),
            _run(40, "York", "h", time="1.00", pos=5, official_rating=85, total_dst_bt="6"),
            _run(40, "Ayr", "h", time="4.00", pos=2, official_rating=85, total_dst_bt="1")]
    out = handicap_angles.build(_frame(rows))
    third, fourth = out.iloc[2], out.iloc[3]
    assert third["hca_mark_vs_last_win"] == 5.0 and third["hca_wins_off_higher_mark"] == 1.0
    for c in handicap_angles.FEATURES[1:]:                        # the second run of the day reads day 20, too
        assert third[c] == fourth[c] or (np.isnan(third[c]) and np.isnan(fourth[c])), c
    assert fourth["hca_lr_won"] == 1.0
