"""model/blocks/form_lines_close.py by hand: rivals' results since, weighted by how close they finished."""
import numpy as np
import pandas as pd

from model.blocks import form_lines_close as flc

K, L = flc.K, flc.L


def _race(day, order, beaten=None, dnf=(), rid=None):
    """One race: `order` finishes in that order, `beaten` their lengths behind the winner (0 for the winner)."""
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    horses = list(order) + list(dnf)
    beaten = beaten or {h: float(i) for i, h in enumerate(order)}
    return [dict(raceid=rid or f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h,
                 placing_numerical=float(i + 1) if i < len(order) else np.nan,
                 total_dst_bt=str(beaten[h]) if i < len(order) else "", bfsp=float(len(horses)))
            for i, h in enumerate(horses)]


def _card(day, horses, rid=None):
    rows = _race(day, horses, rid=rid)
    for r in rows:
        r.update(placing_numerical=np.nan, total_dst_bt="", bfsp=np.nan)
    return rows


def _at(out, day, horse):
    return out.loc[(pd.Timestamp("2025-06-01") + pd.Timedelta(days=day), horse)]


def test_a_close_rival_counts_more_than_a_distant_one():
    rows = (_race(0, ["A", "B", "C"], beaten={"A": 0.0, "B": 1.0, "C": 12.0})
            + _race(1, ["B", "X"]) + _race(2, ["C", "Y"])                    # both rivals win since
            + _card(3, ["A", "Z"]))
    out = flc.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = _at(out, 3, "A")
    wb, wc = np.exp(-1.0 / L), np.exp(-12.0 / L)
    assert np.isclose(a["flc_l1_n"], wb + wc) and np.isclose(a["flc_l1_wins"], wb + wc)
    assert np.isclose(a["flc_l1_wr"], (wb + wc + K * flc.P_WIN) / (wb + wc + K))
    # each race had two runners at equal prices: a win is 1 - 0.5 above the market's chance
    assert np.isclose(a["flc_l1_ae"], 0.5 * (wb + wc) / (wb + wc + K))
    assert np.isnan(_at(out, 3, "Z")["flc_l1_n"])                             # a debutant


def test_non_finishers_and_the_day_itself_count_for_nothing():
    rows = (_race(0, ["A", "B"], dnf=["D"]) + _race(1, ["D", "q"]) + _race(2, ["B", "r"], rid="r2")
            + _card(2, ["A", "s"], rid="c2"))
    out = flc.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = _at(out, 2, "A")
    assert a["flc_l1_n"] == 0 and a["flc_l1_wins"] == 0          # D did not finish; B's run is today's


def test_the_last_three_add_each_race_and_leave_out_the_horses_own_runs():
    rows = (_race(0, ["A", "C"], beaten={"A": 0.0, "C": 2.0}) + _race(1, ["A", "D"], beaten={"A": 0.0, "D": 0.5})
            + _race(2, ["C", "e"]) + _race(3, ["D", "f"]) + _card(4, ["A", "g"]))
    out = flc.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = _at(out, 4, "A")
    wd, wc = np.exp(-0.5 / L), np.exp(-2.0 / L)
    assert np.isclose(a["flc_l1_n"], wd)                                       # the last race: D's win on day 3
    assert np.isclose(a["flc_l3_n"], wd + wc) and np.isclose(a["flc_l3_wins"], wd + wc)   # A's own day-1 run left out
