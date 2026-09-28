"""model/blocks/elo.py by hand: ratings from finishing orders, read as they stood before the day."""
import numpy as np
import pandas as pd

from model.blocks import elo


def _races(results, code="Flat"):
    """results: list of (day, [horses in finishing order], [non-finishers])."""
    rows = []
    for day, order, dnf in results:
        d = pd.Timestamp("2025-04-01") + pd.Timedelta(days=day)
        for place, h in enumerate(order, start=1):
            rows.append(dict(raceid=f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h,
                             placing_numerical=float(place), race_code=code))
        for h in dnf:
            rows.append(dict(raceid=f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h,
                             placing_numerical=np.nan, race_code=code))
    return pd.DataFrame(rows)


def test_ratings_follow_the_form_and_carry_it_through_rivals_who_never_met():
    # A beats B, B beats C, repeatedly; X beats Y, Y beats Z: two groups that never meet
    hist = []
    for i in range(6):
        hist.append((2 * i, ["A", "B", "C"], []))
        hist.append((2 * i + 1, ["X", "Y", "Z"], []))
    # today: C against Z (each last in its group), and a newcomer N; results unknown
    hist.append((20, [], ["C", "Z", "N"]))
    out = elo.build(_races(hist)).set_index(["race_date", "horse_name"])
    last = pd.Timestamp("2025-04-11")                       # day 10: A-B-C's sixth race
    assert out.loc[(last, "A"), "elo"] > out.loc[(last, "B"), "elo"] > out.loc[(last, "C"), "elo"]
    today = pd.Timestamp("2025-04-21")
    c, z, nn = out.loc[(today, "C")], out.loc[(today, "Z")], out.loc[(today, "N")]
    assert np.isnan(nn["elo"]) and nn["elo_runs"] == 0                # never finished: no rating
    assert c["elo_runs"] == 6 and z["elo_runs"] == 6
    assert abs(c["elo"] - z["elo"]) < 1e-9                             # the same record in mirror groups
    assert c["elo"] < 1500 and c["elo_gap"] == 0.0 and np.isclose(c["elo_z"], 0.0)
    assert np.isclose(c["elo_field"], c["elo"])                        # rated runners only
    assert c["elo_trend"] < 0                                          # still losing ground


def test_a_fall_moves_nothing_and_the_first_day_knows_nobody():
    out = elo.build(_races([(0, ["A", "B"], ["F"]), (1, ["A"], ["F", "B"]), (2, [], ["A", "B", "F"])]))
    first = out[out.race_date == pd.Timestamp("2025-04-01")]
    assert first["elo"].isna().all()
    day2 = out[out.race_date == pd.Timestamp("2025-04-03")].set_index("horse_name")
    assert np.isnan(day2.loc["F", "elo"])                              # never finished
    assert day2.loc["A", "elo_runs"] == 1                              # a lone finisher beats nobody: no update
    assert day2.loc["A", "elo"] > 1500 > day2.loc["B", "elo"]


def test_the_flat_and_jumps_are_rated_apart():
    df = pd.concat([_races([(0, ["A", "B"], [])], "Flat"), _races([(1, ["B", "A"], [])], "National Hunt"),
                    _races([(2, [], ["A"])], "Flat"), _races([(3, [], ["A"])], "National Hunt")])
    out = elo.build(df.reset_index(drop=True)).set_index(["race_date", "race_code", "horse_name"])
    assert out.loc[(pd.Timestamp("2025-04-03"), "Flat", "A"), "elo"] > 1500
    assert out.loc[(pd.Timestamp("2025-04-04"), "National Hunt", "A"), "elo"] < 1500
