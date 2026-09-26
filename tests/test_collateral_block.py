"""model/blocks/collateral.py by hand: today's runners compared through the opponents they have both met."""
import numpy as np
import pandas as pd

from model.blocks import collateral
from model.form_windows import run_measures


def _race(day, order, beaten, rid=None, dist=8.0):
    """One race: `order` finishes in that order, `beaten` their lengths behind the winner."""
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    return [dict(raceid=rid or f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h,
                 number_of_runners=len(order), placing_numerical=float(i + 1),
                 total_dst_bt=str(beaten[h]), dist_furlongs=dist)
            for i, h in enumerate(order)]


def _card(day, horses, rid=None, dist=8.0):
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    return [dict(raceid=rid or f"c{day}", race_date=d, race_time="3.00", track="York", horse_name=h,
                 number_of_runners=len(horses), placing_numerical=np.nan, total_dst_bt="", dist_furlongs=dist)
            for h in horses]


def _at(out, day, horse):
    return out.loc[(pd.Timestamp("2025-06-01") + pd.Timedelta(days=day), horse)]


def test_a_common_opponent_gives_the_line_between_two_runners():
    rows = (_race(0, ["A", "C"], {"A": 0.0, "C": 2.0})             # A beat C by 2 lengths
            + _race(1, ["C", "B"], {"C": 0.0, "B": 1.0})           # C beat B by a length
            + _card(2, ["A", "B", "X"]))
    df = pd.DataFrame(rows)
    out = collateral.build(df).set_index(["race_date", "horse_name"])
    lbs = pd.Series(run_measures(df)["lbs"], index=pd.MultiIndex.from_frame(df[["race_date", "horse_name"]]))
    a_over_c = lbs[(pd.Timestamp("2025-06-01"), "C")] - lbs[(pd.Timestamp("2025-06-01"), "A")]
    b_over_c = lbs[(pd.Timestamp("2025-06-02"), "C")] - lbs[(pd.Timestamp("2025-06-02"), "B")]
    a = _at(out, 2, "A")
    b = _at(out, 2, "B")
    assert a["cl_rivals"] == 1 and a["cl_links"] == 1
    assert np.isclose(a["cl_lbs_max"], a_over_c - b_over_c) and a_over_c - b_over_c > 0
    assert np.isclose(a["cl_lbs"], (a_over_c - b_over_c) / (1 + collateral.K_RIVALS))
    assert np.isclose(b["cl_lbs_min"], -(a_over_c - b_over_c))            # the same line, from B's side
    assert a["cl_net"] == 1 and b["cl_net"] == -1
    assert np.isclose(a["cl_ahead_share"], 2 / 3) and np.isclose(b["cl_ahead_share"], 1 / 3)
    x = _at(out, 2, "X")
    assert x["cl_rivals"] == 0 and np.isnan(x["cl_lbs"]) and x["cl_ahead_share"] == 0.5


def test_only_each_horses_last_three_runs_and_earlier_days_count():
    rows = _race(0, ["A", "C"], {"A": 0.0, "C": 2.0})                     # A met C, then ran three more times
    for d in (1, 2, 3):
        rows += _race(d, ["A", f"z{d}"], {"A": 0.0, f"z{d}": 1.0})
    rows += _race(4, ["C", "B"], {"C": 0.0, "B": 1.0}) + _card(5, ["A", "B"])
    out = collateral.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    assert _at(out, 5, "A")["cl_rivals"] == 0                              # the C race is A's fourth run back
    # the day itself: B meets C on day 2 while A's card is day 2: no line from the day's own race
    rows2 = _race(0, ["A", "C"], {"A": 0.0, "C": 2.0}) + _race(2, ["C", "B"], {"C": 0.0, "B": 1.0}, rid="r2") \
        + _card(2, ["A", "B"], rid="c2")
    out2 = collateral.build(pd.DataFrame(rows2)).set_index(["race_date", "horse_name"])
    card_a = out2.loc[(pd.Timestamp("2025-06-03"), "A")]
    assert (np.atleast_1d(card_a["cl_rivals"]) == 0).all()


def test_two_opponents_average_into_one_rival():
    rows = (_race(0, ["A", "C", "D"], {"A": 0.0, "C": 1.0, "D": 3.0})
            + _race(1, ["C", "B", "D"], {"C": 0.0, "B": 2.0, "D": 2.5})
            + _card(2, ["A", "B"]))
    out = collateral.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = _at(out, 2, "A")
    assert a["cl_rivals"] == 1 and a["cl_links"] == 2 and np.isclose(a["cl_lbs_min"], a["cl_lbs_max"])
