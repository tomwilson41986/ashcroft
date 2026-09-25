"""model/blocks/shrunk_rates.py by hand: a small cell is pulled toward the level above it. Mechanics only."""
import numpy as np
import pandas as pd
import pytest

from model.blocks import shrunk_rates


def _runs(rows):
    out = []
    for i, (day, track, horse, trainer, won, n) in enumerate(rows):
        d = pd.Timestamp("2025-03-01") + pd.Timedelta(days=day)
        rid = f"{d.date()}|{track}|{i}"
        for k in range(n):
            me = k == 0
            out.append(dict(race_date=d, race_time="2.30", track=track, raceid=rid, number_of_runners=n,
                            horse_name=horse if me else f"r{i}_{k}", trainer=trainer if me else "T9",
                            jockey_name="J1" if me else "J9", stallion="S1", going_description="Good",
                            placing_numerical=(1 if won else 2) if me else (2 if won else 1) + (k > 1) * k,
                            LB=0.0, total_dst_bt="0", dist_furlongs=6.0, official_rating=70, bfsp=4.0))
    return pd.DataFrame(out)


def test_a_trainers_course_record_is_pulled_to_its_overall():
    # trainer T1: at York 1 win from 2; elsewhere 0 from 8 -> overall 1 from 10 (shrunk to 10% at k=20);
    # the York cell is 1 from 2 pulled toward that
    rows = [(0, "York", "a", "T1", True, 4), (1, "York", "b", "T1", False, 4)]
    rows += [(2 + i, "Ascot", f"c{i}", "T1", False, 4) for i in range(8)]
    rows += [(20, "York", "z", "T1", False, 4)]                      # the query
    out = shrunk_rates.build(_runs(rows))
    q = out[(out.horse_name == "z")].iloc[0]
    overall = (1 + 20 * 0.1) / (10 + 20)                             # ~0.1: decay over 20 days is ~2%
    expected = (1 + 20 * overall) / (2 + 20)
    assert q.sr_tr_course_win == pytest.approx(expected, rel=0.03)
    assert 0.1 < q.sr_tr_course_win < 0.5                            # nowhere near the raw 50%


def test_a_horses_one_run_on_the_going_says_little():
    rows = [(0, "York", "h", "T1", True, 4), (10, "York", "h", "T1", False, 4), (20, "York", "h", "T1", False, 4)]
    df = _runs(rows)
    df.loc[(df.horse_name == "h") & (df.race_date == pd.Timestamp("2025-03-01")), "going_description"] = "Soft"
    df.loc[df.race_date == pd.Timestamp("2025-03-21"), "going_description"] = "Soft"
    out = shrunk_rates.build(df)
    q = out[(out.horse_name == "h") & (out.race_date == pd.Timestamp("2025-03-21"))].iloc[0]
    # career: a win (NFP 1) and a second of four (NFP 2/3), shrunk to 0.5 at k = 3; its one soft run won
    career = (1.0 + 2 / 3 + 3 * 0.5) / (2 + 3)
    assert q.sr_horse_going_nfp == pytest.approx((1.0 + 2 * career) / (1 + 2))
    assert q.sr_horse_going_n == pytest.approx(1.0)
