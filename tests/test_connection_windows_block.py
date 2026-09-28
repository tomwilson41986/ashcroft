"""model/blocks/connection_windows.py by hand: the trainer's and jockey's earlier runners, windowed and ranked."""
import numpy as np
import pandas as pd

from model.blocks import connection_windows as cw

K = cw.K


def _race(day, runners, rid=None, result=True):
    """runners: (horse, trainer, jockey) in finishing order; equal BSPs."""
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    n = len(runners)
    return [dict(raceid=rid or f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h, trainer=t,
                 jockey_name=j, number_of_runners=n, placing_numerical=float(i + 1) if result else np.nan,
                 bfsp=float(n) if result else np.nan, total_dst_bt=str(float(i)) if result else "",
                 dist_furlongs=8.0)
            for i, (h, t, j) in enumerate(runners)]


def _at(out, day, horse):
    return out.loc[(pd.Timestamp("2025-06-01") + pd.Timedelta(days=day), horse)]


def test_the_trainers_earlier_runners_make_its_windows_and_rank():
    rows = (_race(0, [("a1", "T1", "J1"), ("b1", "T2", "J2")])               # T1 won, T2 last
            + _race(1, [("b2", "T2", "J2"), ("a2", "T1", "J1")])             # T2 won, T1 last
            + _race(2, [("a3", "T1", "J1"), ("b3", "T2", "J2"), ("c3", "T3", "J3")])
            + _race(3, [("a4", "T1", "J2"), ("b4", "T2", "J1"), ("c4", "T3", "J3")], result=False))
    out = cw.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a4 = _at(out, 3, "a4")
    # T1 before day 3: won (nfp 1), last of 2 (nfp 0), won of 3 (nfp 1)
    assert a4["cw_tr_n"] == 3
    assert np.isclose(a4["cw_tr_nfp_car"], (2.0 + K * cw.NFP_PRIOR) / (3 + K))
    assert np.isclose(a4["cw_tr_nfp_l20"], 2 / 3) and np.isclose(a4["cw_tr_nfp_l100"], 2 / 3)
    # T2: last (0), won (1), second of three (0.5)
    b4 = _at(out, 3, "b4")
    assert np.isclose(b4["cw_tr_nfp_l20"], 0.5)
    assert a4["cw_tr_nfp_rank"] == 1 and b4["cw_tr_nfp_rank"] == 2
    # T3 ran once, third of three
    assert _at(out, 3, "c4")["cw_tr_n"] == 1
    # the jockey J2 rode for T2 until day 3: its record is T2's
    assert np.isclose(a4["cw_jk_nfp_l20"], 0.5)


def test_the_day_itself_never_counts_and_a_first_runner_has_none():
    rows = _race(0, [("a", "T1", "J1"), ("b", "T2", "J2")]) + _race(0, [("c", "T1", "J1"), ("d", "T2", "J2")], rid="r0b")
    out = cw.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    first = out.loc[pd.Timestamp("2025-06-01")]
    assert (first["cw_tr_n"] == 0).all() and first["cw_tr_nfp_l20"].isna().all()
    assert np.allclose(first["cw_tr_nfp_car"], cw.NFP_PRIOR)


def test_the_windows_hold_the_last_twenty_runners_only():
    rows = []
    for d in range(25):                                    # T1 wins 20 times, then loses 5
        rows += _race(d, [("x%d" % d, "T1", "J1"), ("y%d" % d, "T9", "J9")] if d < 20
                      else [("y%d" % d, "T9", "J9"), ("x%d" % d, "T1", "J1")])
    rows += _race(25, [("z", "T1", "J1"), ("w", "T9", "J9")], result=False)
    out = cw.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    z = _at(out, 25, "z")
    assert z["cw_tr_n"] == 25
    assert np.isclose(z["cw_tr_nfp_l20"], 15 / 20)        # the last 20: 15 wins, 5 last places
    assert np.isclose(z["cw_tr_nfp_l100"], 20 / 25)
