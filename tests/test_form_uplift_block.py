"""model/blocks/form_uplift.py by hand: what the rivals from a horse's recent races have run since, in figures."""
import numpy as np
import pandas as pd

from model.blocks import form_uplift
from model.form_windows import run_measures

K = form_uplift.K


def _race(day, order, ratings, beaten=None, bsp=None, dnf=(), rid=None):
    """One race: `order` finishes in that order (then `dnf`); `ratings` the official ratings (0 = none)."""
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    horses = list(order) + list(dnf)
    beaten = beaten or {h: float(i) for i, h in enumerate(order)}
    bsp = bsp or {h: float(len(horses)) for h in horses}
    return [dict(raceid=rid or f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h,
                 number_of_runners=len(horses), placing_numerical=float(i + 1) if i < len(order) else np.nan,
                 total_dst_bt=str(beaten[h]) if h in beaten and i < len(order) else "",
                 dist_furlongs=6.0, official_rating=float(ratings[h]), bfsp=bsp[h])
            for i, h in enumerate(horses)]


def _card(day, ratings, rid=None):
    rows = _race(day, list(ratings), ratings, rid=rid)
    for r in rows:
        r.update(placing_numerical=np.nan, total_dst_bt="", bfsp=np.nan)
    return rows


def _at(out, day, horse):
    return out.loc[(pd.Timestamp("2025-06-01") + pd.Timedelta(days=day), horse)]


def test_the_last_race_is_revised_by_its_rivals_runs_since():
    rows = (_race(0, ["A", "B", "C", "D"], {"A": 70, "B": 72, "C": 68, "D": 65},
                  bsp={"A": 3.0, "B": 4.0, "C": 6.0, "D": 12.0})
            + _race(1, ["B", "E"], {"B": 75, "E": 60}, bsp={"B": 2.0, "E": 2.5})       # B raised 3, wins
            + _race(2, ["G", "C"], {"G": 70, "C": 66}, bsp={"G": 1.5, "C": 5.0})       # C dropped 2, second
            + _card(3, {"A": 74, "X": 50}))
    df = pd.DataFrame(rows)
    out = form_uplift.build(df).set_index(["race_date", "horse_name"])
    a = _at(out, 3, "A")
    assert np.isclose(a["fu_l1_or"], (3 - 2) / (2 + K))                     # the handicapper's reassessment
    m = pd.DataFrame(run_measures(df)).assign(day=df.race_date.dt.day - 1, horse=df.horse_name).set_index(
        ["day", "horse"])
    for s in ("perf", "mkt"):
        change = (m.loc[(1, "B"), s] - m.loc[(0, "B"), s]) + (m.loc[(2, "C"), s] - m.loc[(0, "C"), s])
        assert np.isclose(a[f"fu_l1_{s}"], change / (2 + K)), s
    assert np.isclose(a["fu_l1_nres"], (m.loc[(1, "B"), "nres"] + m.loc[(2, "C"), "nres"]) / (2 + K))
    assert np.isclose(a["fu_l1_adj_vs_or"], m.loc[(0, "A"), "perf"] + a["fu_l1_perf"] - 74)
    assert np.isclose(a["fu_l3_or"], a["fu_l1_or"])                         # one race so far
    x = _at(out, 3, "X")
    assert np.isnan(x["fu_l1_or"]) and np.isnan(x["fu_l3_perf"])          # a debutant
    assert out.loc[pd.Timestamp("2025-06-01"), "fu_l1_perf"].isna().all()


def test_the_day_itself_never_counts_and_each_rival_only_its_next_three():
    ratings = {"A": 70, "B": 70}
    rows = _race(0, ["A", "B"], ratings)
    for d in range(1, 6):                                                    # B raised 1 lb a run
        rows += _race(d, ["B", f"z{d}"], {"B": 70 + d, f"z{d}": 50})
    rows += _card(6, {"A": 70, "B": 76})
    out = form_uplift.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    assert np.isclose(_at(out, 6, "A")["fu_l1_or"], (1 + 2 + 3) / (3 + K))  # B's next three only
    # on day 2 (a card for A beside B's second run): only B's day-1 run is before the day
    rows2 = _race(0, ["A", "B"], ratings) + _race(1, ["B", "z1"], {"B": 72, "z1": 50}) \
        + _race(2, ["B", "z2"], {"B": 75, "z2": 50}, rid="r2b") + _card(2, {"A": 70, "q": 50})
    out2 = form_uplift.build(pd.DataFrame(rows2)).set_index(["race_date", "horse_name"])
    assert np.isclose(_at(out2, 2, "A")["fu_l1_or"], 2 / (1 + K))


def test_the_last_three_leave_out_the_horses_own_runs_and_unknown_pairs():
    rows = (_race(0, ["A", "C"], {"A": 70, "C": 60})
            + _race(1, ["A", "D"], {"A": 72, "D": 0})                        # D unrated: no mark to compare
            + _race(2, ["C", "E"], {"C": 64, "E": 50})                        # C up 4 since day 0
            + _race(3, ["D", "F"], {"D": 55, "F": 50})
            + _card(4, {"A": 73, "Z": 50}))
    out = form_uplift.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = _at(out, 4, "A")
    assert np.isclose(a["fu_l1_or"], 0.0)                                   # D's pair unknown: nothing counted
    # the last three: day 0's C (+4); A's own run on day 1 (+2 on day 0's mark) is left out
    assert np.isclose(a["fu_l3_or"], 4 / (1 + K))
