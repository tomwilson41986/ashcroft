"""model/blocks/stable.py by hand: a yard's runners today, their order, and whose jockey it books."""
import numpy as np
import pandas as pd

from model.blocks import stable


def _rows():
    def r(day, time, track, horse, trainer, jockey, rating):
        d = pd.Timestamp("2025-05-01") + pd.Timedelta(days=day)
        return dict(race_date=d, race_time=time, track=track, raceid=f"{d.date()}|{track}|{time}", horse_name=horse,
                    trainer=trainer, jockey_name=jockey, official_rating=rating)
    rows = [r(0, "2.00", "York", "a", "T1", "J1", 80), r(0, "2.00", "York", "b", "T1", "J1", 70),   # J1 rides for T1
            r(1, "3.00", "York", "a", "T1", "J1", 80), r(1, "3.00", "York", "c", "T2", "J3", 75),
            r(2, "4.00", "York", "b", "T1", "J2", 72)]                                          # J2 once for T1
    # today: T1 runs three at York, two in the 2.30 (J1 on the lower-rated one), one in the 3.30; one at Ripon
    rows += [r(5, "2.30", "York", "a", "T1", "J2", 82), r(5, "2.30", "York", "b", "T1", "J1", 74),
             r(5, "2.30", "York", "c", "T2", "J3", 0), r(5, "3.30", "York", "d", "T1", "J4", 60),
             r(5, "4.10", "Ripon", "e", "T1", "J1", 65)]
    return pd.DataFrame(rows)


def test_a_yards_runners_their_order_and_its_jockey():
    out = stable.build(_rows())
    today = out[out.race_date == pd.Timestamp("2025-05-06")].set_index("horse_name")
    assert today.loc["a", "st_n_race"] == 2 and today.loc["d", "st_n_race"] == 1
    assert today.loc["a", "st_n_meeting"] == 3 and today.loc["e", "st_n_meeting"] == 1
    assert today.loc["a", "st_n_day"] == 4 and today.loc["c", "st_n_day"] == 1
    # a is rated 82, b 74: a first, b 8 behind; c runs alone for T2 and is unrated
    assert today.loc["a", "st_or_rank"] == 1 and today.loc["b", "st_or_rank"] == 2
    assert today.loc["b", "st_or_gap"] == -8 and today.loc["a", "st_or_gap"] == 0
    assert np.isnan(today.loc["c", "st_or_rank"]) and np.isnan(today.loc["d", "st_or_rank"])
    # T1's earlier rides: J1 three, J2 one (decayed, so about 3/4 and 1/4); J4 never
    assert 0.7 < today.loc["b", "st_jk_share"] < 0.8 and 0.2 < today.loc["a", "st_jk_share"] < 0.3
    assert today.loc["d", "st_jk_share"] == 0
    # the yard's jockey rides b, the lower-rated one
    assert today.loc["b", "st_jk_first"] == 1 and today.loc["a", "st_jk_first"] == 0
    assert today.loc["a", "st_jk_gap"] < 0 and today.loc["b", "st_jk_gap"] == 0
    assert np.isnan(today.loc["d", "st_jk_first"])                     # alone in its race


def test_the_jockeys_standing_reads_earlier_days_only():
    df = _rows()
    a = stable.build(df)
    b = stable.build(df.assign(jockey_name=np.where(df.race_date == pd.Timestamp("2025-05-06"), "J9",
                                                    df.jockey_name)))
    first = a.race_date < pd.Timestamp("2025-05-06")
    pd.testing.assert_frame_equal(a.loc[first, stable.FEATURES], b.loc[first, stable.FEATURES])
    # on the first day nothing is known of any jockey's standing
    day0 = a.race_date == pd.Timestamp("2025-05-01")
    assert a.loc[day0, "st_jk_share"].isna().all()
