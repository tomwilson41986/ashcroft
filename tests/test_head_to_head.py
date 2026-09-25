"""model/blocks/head_to_head.py by hand: form lines between today's rivals, from earlier days only."""
import numpy as np
import pandas as pd
import pytest

from model.blocks import head_to_head as h2h
from model.perf_figures import lbs_per_length


def _race(day, time, runners):
    """runners: (horse, placing or None, lengths beaten)."""
    d = pd.Timestamp("2025-03-01") + pd.Timedelta(days=day)
    return [dict(race_date=d, race_time=time, track="York", raceid=f"{d.date()}|York|{time}",
                 horse_name=h, number_of_runners=len(runners), placing_numerical=p, LB=lb,
                 total_dst_bt=str(lb), dist_furlongs=8.0) for h, p, lb in runners]


def _frame():
    rows = _race(0, "2.30", [("A", 1, 0.0), ("B", 2, 1.0), ("C", 3, 3.0)])
    rows += _race(1, "3.00", [("A", 2, 0.5), ("B", 1, 0.0)])
    rows += _race(2, "4.00", [("A", None, np.nan), ("B", None, np.nan), ("C", None, np.nan),
                              ("D", None, np.nan)])            # today's card: no results
    return pd.DataFrame(rows)


def test_the_form_lines_between_todays_runners():
    out = h2h.build(_frame())
    lpl = float(lbs_per_length(8.0))
    a = out[(out.horse_name == "A") & (out.race_time == "4.00")].iloc[0]
    # A met B twice (ahead by 1 length, then behind by half a length) and C once (ahead by 3)
    assert a.h2h_rivals_met == 2 and a.h2h_meetings == 3
    assert a.h2h_net == 1 and a.h2h_win_share == pytest.approx(3 / 5)
    assert a.h2h_lbs == pytest.approx((1.0 + 3.0 - 0.5) * lpl / 5)
    assert a.h2h_last_lbs == pytest.approx((-0.5 + 3.0) / 2 * lpl)   # B's latest meeting, C's only one
    assert a.h2h_last_net == 0
    c = out[(out.horse_name == "C") & (out.race_time == "4.00")].iloc[0]
    assert c.h2h_rivals_met == 2 and c.h2h_net == -2 and c.h2h_last_net == -2
    assert c.h2h_lbs == pytest.approx(-(3.0 + 2.0) * lpl / 4)
    d = out[out.horse_name == "D"].iloc[0]                          # met nobody
    assert d.h2h_meetings == 0 and d.h2h_win_share == 0.5 and d.h2h_lbs == 0 and np.isnan(d.h2h_last_lbs)


def test_a_days_own_result_never_enters_its_features():
    df = _frame()
    first = h2h.build(df)
    assert (first[first.race_time == "2.30"]["h2h_meetings"] == 0).all()   # nobody had met on day 1
    # the day-2 result reversed: day 2's own rows unchanged, day 3's move
    flipped = df.copy()
    flipped.loc[flipped.race_time == "3.00", "placing_numerical"] = [1, 2]
    flipped.loc[flipped.race_time == "3.00", "LB"] = [0.0, 0.5]
    second = h2h.build(flipped)
    cols = h2h.FEATURES
    day2 = first.race_time == "3.00"
    pd.testing.assert_frame_equal(first.loc[day2, cols], second.loc[day2, cols])
    a3 = (first.horse_name == "A") & (first.race_time == "4.00")
    assert first.loc[a3, "h2h_net"].item() == 1 and second.loc[a3, "h2h_net"].item() == 3


def test_every_ordered_pair_once():
    race = np.array([0, 0, 1, 1, 1, 2])
    a, b = h2h.pairs(race)
    got = sorted(zip(a.tolist(), b.tolist()))
    assert got == [(0, 1), (1, 0), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3)]
