"""model/blocks/sire_windows.py by hand: the sire's and damsire's progeny over recent windows."""
import numpy as np
import pandas as pd

from model.blocks import sire_windows as sw


def _race(day, runners, rid=None, result=True):
    """runners: (horse, sire, damsire, age) in finishing order; equal BSPs."""
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    n = len(runners)
    return [dict(raceid=rid or f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h, stallion=s,
                 dam_stallion=ds, horse_age=a, number_of_runners=n,
                 placing_numerical=float(i + 1) if result else np.nan, bfsp=float(n) if result else np.nan,
                 total_dst_bt=str(float(i)) if result else "", dist_furlongs=8.0)
            for i, (h, s, ds, a) in enumerate(runners)]


def _at(out, day, horse):
    return out.loc[(pd.Timestamp("2025-06-01") + pd.Timedelta(days=day), horse)]


def test_the_sires_progeny_make_its_windows_and_the_young_ones_their_own():
    rows = (_race(0, [("a1", "S1", "D1", 2), ("b1", "S2", "D2", 5)])          # S1's two-year-old won
            + _race(1, [("b2", "S2", "D2", 5), ("a2", "S1", "D1", 6)])        # S1's six-year-old last
            + _race(2, [("a3", "S1", "D2", 2), ("b3", "S2", "D1", 4)], result=False))
    out = sw.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a3 = _at(out, 2, "a3")
    assert a3["sw_sr_n"] == 2
    assert np.isclose(a3["sw_sr_nfp_l100"], 0.5) and np.isclose(a3["sw_sr_nfp_l500"], 0.5)
    assert np.isclose(a3["sw_sr_nfp_car"], (1.0 + 0.0 + sw.K_CAR * 0.5) / (2 + sw.K_CAR))
    assert np.isclose(a3["sw_sr_ae_l300"], (0.5 - 0.5) / (2 + sw.K_CAR))
    # the young progeny: one run, a win, shrunk to S1's own last-500 level (0.5) by twenty
    assert np.isclose(a3["sw_sr_young_nfp"], (1.0 + sw.K_YOUNG * 0.5) / (1 + sw.K_YOUNG))
    assert np.isnan(_at(out, 2, "b3")["sw_sr_young_nfp"])                   # a four-year-old gets none
    # the damsire reads its own grandchildren: D2's runs were b1 (last) and b2 (won)
    assert np.isclose(a3["sw_ds_nfp_l100"], 0.5)
    # S2's progeny: last, won: level with S1, so both rank first
    assert a3["sw_sr_nfp_rank"] == 1 and _at(out, 2, "b3")["sw_sr_nfp_rank"] == 1


def test_the_day_itself_never_counts():
    rows = _race(0, [("a", "S1", "D1", 3), ("b", "S2", "D2", 3)]) + _race(0, [("c", "S1", "D1", 3)], rid="r0b")
    out = sw.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    first = out.loc[pd.Timestamp("2025-06-01")]
    assert (first["sw_sr_n"] == 0).all() and first["sw_sr_nfp_l100"].isna().all()
