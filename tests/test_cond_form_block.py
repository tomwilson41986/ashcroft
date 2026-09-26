"""model/blocks/cond_form.py by hand: form in today's trip, going, course, race type and headgear."""
import numpy as np
import pandas as pd

from model.blocks import cond_form


def _race(day, track, going, rtype, order, gear=(), dist=6.0):
    """One race of four: `order` is the finishing order; `gear` the horses wearing headgear."""
    d = pd.Timestamp("2025-05-01") + pd.Timedelta(days=day)
    rows = []
    for place, h in enumerate(order, start=1):
        rows.append(dict(raceid=f"r{day}{track}", race_date=d, race_time="2.30", track=track, horse_name=h,
                         number_of_runners=len(order), placing_numerical=float(place),
                         total_dst_bt="0" if place == 1 else str(place - 1), dist_furlongs=dist,
                         official_rating=70.0, bfsp=2.0 * place, going_description=going, race_type=rtype,
                         race_name=rtype, headgear="b" if h in gear else ""))
    return rows


def _card(day, track, going, rtype, horses, gear=(), dist=6.0):
    rows = _race(day, track, going, rtype, horses, gear, dist)
    for r in rows:
        r.update(placing_numerical=np.nan, total_dst_bt="", bfsp=np.nan)
    return rows


def test_windows_read_only_earlier_runs_in_the_same_condition():
    rows = (_race(0, "York", "Good", "Handicap", ["A", "B", "C", "D"])             # A wins: nfp 1
            + _race(1, "York", "Soft", "Handicap", ["B", "C", "D", "A"])           # A last: nfp 0
            + _race(2, "Kempton", "Standard", "Maiden", ["B", "A", "C", "D"], dist=7.0)   # nfp 2/3
            + _card(3, "York", "Good", "Handicap", ["A", "B", "C", "D"], gear=("A",)))
    out = cond_form.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = out.loc[(pd.Timestamp("2025-05-04"), "A")]
    assert a["cf_course_n"] == 2 and np.isclose(a["cf_course_nfp_car"], 0.5)      # York twice
    assert a["cf_going_n"] == 1 and np.isclose(a["cf_going_nfp_car"], 1.0)        # good once, won
    assert a["cf_hcap_n"] == 2 and np.isclose(a["cf_hcap_nfp_m3"], 0.5)
    assert a["cf_trip_n"] == 2                                                      # 6f twice; the 7f apart
    assert a["cf_gear_n"] == 0 and np.isnan(a["cf_gear_nfp_car"])                   # first time in headgear
    overall = (1.0 + 0.0 + 2 / 3) / 3
    assert np.isclose(a["cf_going_nfp_d"], (1.0 - overall) * 1 / (1 + cond_form.K))
    assert np.isclose(a["cf_course_nfp_d"], (0.5 - overall) * 2 / (2 + cond_form.K))


def test_the_day_itself_and_later_days_never_count():
    rows = (_race(0, "York", "Good", "Handicap", ["A", "B", "C", "D"])
            + _race(1, "York", "Good", "Handicap", ["D", "C", "B", "A"]))
    out = cond_form.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    first = out.loc[pd.Timestamp("2025-05-01")]
    assert (first["cf_course_n"] == 0).all() and first["cf_course_nfp_car"].isna().all()
    second = out.loc[(pd.Timestamp("2025-05-02"), "A")]
    assert second["cf_course_n"] == 1 and np.isclose(second["cf_course_nfp_car"], 1.0)   # day 0 only


def test_conditions_from_the_card():
    df = pd.DataFrame(_card(0, "York", "Good to Firm", "Handicap Chase", ["A"], gear=("A",), dist=20.0)
                      + _card(0, "Wolverhampton", "Standard", "Novice Stakes", ["B"])
                      + _card(0, "Ayr", "Heavy", "Nursery", ["C"])
                      + _card(0, "Ayr", None, "Maiden", ["D"]))
    c = cond_form.conditions(df)
    assert list(c["going"]) == [0, 4, 3, -1]
    assert list(c["hcap"]) == [1, 0, 1, 0] and list(c["gear"]) == [1, 0, 0, 0]
    assert c["trip"][0] != c["trip"][1]
