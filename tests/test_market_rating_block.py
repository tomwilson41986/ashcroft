"""The market rating block on a hand-worked history: the same share of the book against strong
rivals reads higher than against weak ones (model/blocks/market_rating.py)."""
import numpy as np
import pandas as pd
import pytest

from model.blocks import market_rating as mr


def _row(day, time, horse, bsp, race_type="Handicap"):
    return {"race_date": pd.Timestamp("2025-03-01") + pd.Timedelta(days=day), "race_time": time,
            "track": "Kempton", "horse_name": horse, "bfsp": bsp, "race_type": race_type,
            "surface_type": "Standard"}


def _history():
    rows = [
        # day 0: two strong (40% of the book each) and two weak (10% each); everyone new
        _row(0, "2.00", "R1", 2.5), _row(0, "2.00", "R2", 2.5), _row(0, "2.00", "W1", 10.0), _row(0, "2.00", "W2", 10.0),
        # day 1: N shares a book 50/50 with a strong horse, M 50/50 with a weak one
        _row(1, "2.00", "N", 2.0), _row(1, "2.00", "R1", 2.0),
        _row(1, "3.00", "M", 2.0), _row(1, "3.00", "W1", 2.0),
        # day 2: N and M meet
        _row(2, "2.00", "N", 3.0), _row(2, "2.00", "M", 1.5),
    ]
    return pd.DataFrame(rows)


def test_the_same_share_against_stronger_rivals_rates_higher():
    out = mr.build(_history()).set_index(["race_date", "horse_name"])
    d0, d2 = pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-03")
    # nobody is rated before their first reading
    assert out.loc[(d0, "R1"), "mr_rating"] != out.loc[(d0, "R1"), "mr_rating"]
    assert out.loc[(d0, "R1"), "mr_n"] == 0
    n, m = out.loc[(d2, "N")], out.loc[(d2, "M")]
    assert n["mr_n"] == 1 and m["mr_n"] == 1
    assert n["mr_rating"] > 0.3 > -0.3 > m["mr_rating"]
    assert n["mr_gap"] == pytest.approx(n["mr_rating"] - m["mr_rating"])
    assert n["mr_share"] + m["mr_share"] == pytest.approx(1.0)
    assert n["mr_share"] > 0.6
    # day 2's own prices (M the favourite) reach none of day 2's features
    alt = _history()
    alt.loc[alt["race_date"] == d2, "bfsp"] = [1.5, 3.0]
    alt_out = mr.build(alt).set_index(["race_date", "horse_name"])
    pd.testing.assert_frame_equal(out.loc[[(d2, "N"), (d2, "M")], mr.FEATURES],
                                  alt_out.loc[[(d2, "N"), (d2, "M")], mr.FEATURES])


def test_a_race_too_thinly_priced_is_not_read():
    df = _history()
    df.loc[(df["race_date"] == pd.Timestamp("2025-03-02")) & (df["horse_name"] == "R1"), "bfsp"] = np.nan
    out = mr.build(df).set_index(["race_date", "horse_name"])
    assert out.loc[(pd.Timestamp("2025-03-03"), "N"), "mr_n"] == 0


def test_flat_and_jumps_are_separate_pools():
    df = _history()
    df.loc[df["race_date"] == pd.Timestamp("2025-03-03"), "race_type"] = "Handicap Hurdle"
    df.loc[df["race_date"] == pd.Timestamp("2025-03-03"), "surface_type"] = "Turf"
    out = mr.build(df).set_index(["race_date", "horse_name"])
    assert out.loc[(pd.Timestamp("2025-03-03"), "N"), "mr_n"] == 0
