"""model/blocks/bookings.py by hand: jockeys and yards as the market rated their earlier runners."""
import numpy as np
import pandas as pd
import pytest

from model.blocks import bookings


def _race(day, time, runners):
    """runners: (horse, jockey, trainer, bsp, placing)."""
    d = pd.Timestamp("2025-03-01") + pd.Timedelta(days=day)
    return [dict(race_date=d, race_time=time, track="York", raceid=f"{d.date()}|York|{time}", horse_name=h,
                 jockey_name=j, trainer=t, number_of_runners=len(runners), bfsp=b, placing_numerical=p)
            for h, j, t, b, p in runners]


def _frame():
    rows = _race(0, "2.30", [("A", "J1", "T1", 2.0, 1), ("B", "J2", "T2", 6.0, 2), ("C", "J3", "T2", 6.0, 3)])
    rows += _race(1, "3.00", [("A", "J2", "T1", 3.0, 2), ("B", "J1", "T2", 3.0, 1)])
    rows += _race(2, "4.00", [("A", "J1", "T1", np.nan, None), ("B", "J3", "T2", np.nan, None),
                              ("D", "J9", "T9", np.nan, None)])            # today's card
    return pd.DataFrame(rows)


def test_jockeys_and_yards_read_from_earlier_days_only():
    out = bookings.build(_frame())
    card = out[out.race_time == "4.00"].set_index("horse_name")
    # J1 rode a 2.0 shot among 2/6/6 (backed) and a 3.0 shot in a two-horse race (level): above 0
    assert card.loc["A", "bk_jk_mkt"] > 0
    # J3 rode one 6.0 shot among 2/6/6: below 0
    assert card.loc["B", "bk_jk_mkt"] < 0
    assert np.isnan(card.loc["D", "bk_jk_mkt"]) or card.loc["D", "bk_jk_mkt"] == 0   # never rode: the prior
    # A rode J1 then J2; today J1 again: not the last run's jockey, one earlier ride on the horse
    assert card.loc["A", "bk_jk_same"] == 0 and card.loc["A", "bk_jk_rides_on_horse"] == 1
    assert card.loc["B", "bk_jk_rides_on_horse"] == 0
    assert np.isnan(card.loc["D", "bk_jk_same"])                                      # no earlier run
    # the upgrade is today's jockey's reading less the mean of the last runs' jockeys, each as of its day
    first = out[(out.horse_name == "A") & (out.race_time == "2.30")].iloc[0]
    assert np.isnan(first.bk_jk_upgrade)                                             # no earlier run


def test_a_days_own_prices_never_enter_its_features():
    df = _frame()
    a = bookings.build(df)
    b_df = df.copy()
    b_df.loc[b_df.race_time == "3.00", "bfsp"] = [9.0, 1.2]                          # day 2's market reversed
    b = bookings.build(b_df)
    day2 = a.race_time == "3.00"
    pd.testing.assert_frame_equal(a.loc[day2, bookings.FEATURES], b.loc[day2, bookings.FEATURES])
    card = a.race_time == "4.00"
    assert not np.allclose(a.loc[card, "bk_jk_mkt"].to_numpy(float), b.loc[card, "bk_jk_mkt"].to_numpy(float),
                           equal_nan=True)
