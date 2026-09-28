"""model/blocks/connection_grains.py by hand: the connections' fortnight, race code, course and pairing."""
import numpy as np
import pandas as pd

from model.blocks import connection_grains as cg

K = cg.K_RECENT


def _race(day, runners, rid=None, result=True, track="York", race_type="Handicap", surface="Turf"):
    """runners: (horse, trainer, jockey) in finishing order; two runners at equal BSPs unless more are given."""
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    n = len(runners)
    return [dict(raceid=rid or f"r{day}{track}", race_date=d, race_time="2.30", track=track, horse_name=h,
                 trainer=t, jockey_name=j, number_of_runners=n,
                 placing_numerical=float(i + 1) if result else np.nan, bfsp=float(n) if result else np.nan,
                 total_dst_bt=str(float(i)) if result else "", dist_furlongs=8.0, race_type=race_type,
                 surface_type=surface)
            for i, (h, t, j) in enumerate(runners)]


def _at(out, day, horse):
    return out.loc[(pd.Timestamp("2025-06-01") + pd.Timedelta(days=day), horse)]


def test_the_fortnight_reads_only_the_last_fourteen_days_against_the_connections_own_level():
    rows = (_race(0, [("a0", "T1", "J1"), ("b0", "T2", "J2")])            # T1 won: 15+ days before day 20
            + _race(10, [("b1", "T2", "J2"), ("a1", "T1", "J1")])         # T1 last: inside the fortnight
            + _race(20, [("a2", "T1", "J1"), ("b2", "T2", "J2")], result=False))
    out = cg.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a2 = _at(out, 20, "a2")
    base = 0.5                                                             # T1's last 100: a win (1) and a last (0)
    assert a2["cg_tr_d14_n"] == 1
    assert np.isclose(a2["cg_tr_d14_nfp"], (0.0 + K * base) / (1 + K))
    assert np.isclose(a2["cg_tr_d14_dnfp"], (0.0 + K * base) / (1 + K) - base)
    assert np.isclose(a2["cg_tr_d14_ae"], -0.5 / (1 + K))                  # lost at a chance of one half
    assert np.isclose(a2["cg_jk_d14_nfp"], a2["cg_tr_d14_nfp"])           # J1 rode every T1 runner
    # the day itself never counts, and a key with nothing before has its prior
    first = _at(out, 0, "a0")
    assert first["cg_tr_d14_n"] == 0 and np.isclose(first["cg_tr_d14_nfp"], cg.NFP_PRIOR)


def test_the_race_code_the_course_and_the_pairing_each_read_their_own_runners():
    rows = (_race(0, [("a0", "T1", "J1"), ("b0", "T2", "J2")])                                  # flat: T1 won
            + _race(1, [("b1", "T2", "J2"), ("a1", "T1", "J1")], race_type="Handicap Hurdle",
                    track="Cheltenham")                                                           # hurdle: T1 last
            + _race(2, [("a2", "T1", "J3"), ("b2", "T2", "J2")], track="Kempton", surface="Standard")  # aw: T1 won
            + _race(3, [("a3", "T1", "J1"), ("b3", "T2", "J2")], race_type="Handicap Hurdle", track="Cheltenham",
                    result=False))
    out = cg.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a3 = _at(out, 3, "a3")
    base = 2 / 3                                                           # T1's last 100: won, last, won
    assert a3["cg_tr_code_n"] == 1                                         # one hurdle runner before
    assert np.isclose(a3["cg_tr_code_nfp"], (0.0 + cg.K_CODE * base) / (1 + cg.K_CODE))
    assert a3["cg_tr_trk_n"] == 1                                          # one Cheltenham runner before
    assert np.isclose(a3["cg_tr_trk_nfp"], (0.0 + K * base) / (1 + K))
    assert a3["cg_tj_n"] == 2                                              # T1 with J1 on days 0 and 1, not J3
    assert np.isclose(a3["cg_tj_nfp"], (1.0 + 0.0 + K * base) / (2 + K))
    assert np.isclose(a3["cg_tj_ae"], (0.5 - 0.5) / (2 + K))
    # the jockey's course record is its own: J1 rode once at Cheltenham, last
    j_base = 0.5                                                           # J1: won, last
    assert a3["cg_jk_trk_n"] == 1 and np.isclose(a3["cg_jk_trk_nfp"], (0.0 + K * j_base) / (1 + K))


def test_a_race_without_a_result_counts_as_a_runner_but_gives_no_reading():
    rows = (_race(0, [("a0", "T1", "J1"), ("b0", "T2", "J2")], result=False)   # void: no placings at all
            + _race(1, [("a1", "T1", "J1"), ("b1", "T2", "J2")], result=False))
    out = cg.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a1 = _at(out, 1, "a1")
    assert a1["cg_tr_d14_n"] == 1 and np.isclose(a1["cg_tr_d14_nfp"], cg.NFP_PRIOR) and a1["cg_tr_d14_ae"] == 0
