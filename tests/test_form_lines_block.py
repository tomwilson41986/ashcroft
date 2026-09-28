"""model/blocks/form_lines.py by hand: what the rivals from a horse's recent races have done since."""
import numpy as np
import pandas as pd

from model.blocks import form_lines


def _race(day, order, dnf=(), bsp=None, rid=None):
    """One race: `order` finishes in that order, `dnf` did not finish; BSPs equal unless given."""
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    horses = list(order) + list(dnf)
    bsp = bsp or {h: float(len(horses)) for h in horses}
    return [dict(raceid=rid or f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h,
                 placing_numerical=float(i + 1) if i < len(order) else np.nan, bfsp=bsp[h])
            for i, h in enumerate(horses)]


def _card(day, horses, rid=None):
    rows = _race(day, horses, rid=rid)
    for r in rows:
        r.update(placing_numerical=np.nan, bfsp=np.nan)
    return rows


def test_the_last_race_is_read_through_its_rivals_next_runs_before_today():
    rows = (_race(0, ["A", "B", "C", "D"])
            + _race(1, ["B", "C", "E", "F"])                   # B wins, C second: their next runs
            + _race(2, ["C", "D", "G", "H"])                   # C's second run since, D's first
            + _race(3, ["E", "G", "F", "H"])                   # nobody from day 0
            + _card(4, ["A", "X", "Y"]))
    out = form_lines.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = out.loc[(pd.Timestamp("2025-06-05"), "A")]
    assert a["fl_l1_n"] == 4 and a["fl_l1_wins"] == 2                  # B@1 won, C@1 2nd, C@2 won, D@2 2nd
    assert np.isclose(a["fl_l1_wr"], (2 + form_lines.K * form_lines.P_WIN) / (4 + form_lines.K))
    assert np.isclose(a["fl_l1_plc"], (4 + form_lines.K * form_lines.P_PLC) / (4 + form_lines.K))
    assert np.isclose(a["fl_l1_ae"], (2 - 4 * 0.25) / (4 + form_lines.K))  # four runners at equal prices each time
    x = out.loc[(pd.Timestamp("2025-06-05"), "X")]
    assert np.isnan(x["fl_l1_n"]) and np.isnan(x["fl_l3_n"])           # a debutant
    first = out.loc[pd.Timestamp("2025-06-01")]
    assert first["fl_l1_n"].isna().all()


def test_each_rival_counts_its_next_three_runs_only_and_the_day_itself_never():
    rows = _race(0, ["A", "B"]) + [r for d in range(1, 6) for r in _race(d, ["B", f"z{d}"])] + _race(6, ["A", "B"])
    out = form_lines.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = out.loc[(pd.Timestamp("2025-06-07"), "A")]
    assert a["fl_l1_n"] == form_lines.NEXT and a["fl_l1_wins"] == form_lines.NEXT   # B won five; three count
    # B's own later runs never count for B: its last race (day 5) had z5 in it, who never ran again
    b = out.loc[(pd.Timestamp("2025-06-07"), "B")]
    assert b["fl_l1_n"] == 0


def test_the_last_three_leave_out_the_horses_own_runs():
    rows = (_race(0, ["A", "C"]) + _race(1, ["A", "D"]) + _race(2, ["C", "E"]) + _race(3, ["D", "F"])
            + _card(4, ["A", "Z"]))
    out = form_lines.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = out.loc[(pd.Timestamp("2025-06-05"), "A")]
    # last race (day 1, with D): D's next run, day 3, won. Day 0 (with C): C won on day 2; A's own run on
    # day 1 is its own and does not count
    assert a["fl_l1_n"] == 1 and a["fl_l1_wins"] == 1
    assert a["fl_l3_n"] == 2 and a["fl_l3_wins"] == 2


def test_a_non_finisher_is_a_run_not_won():
    rows = _race(0, ["A", "B"]) + _race(1, ["C"], dnf=["B"]) + _card(2, ["A", "Q"])
    out = form_lines.build(pd.DataFrame(rows)).set_index(["race_date", "horse_name"])
    a = out.loc[(pd.Timestamp("2025-06-03"), "A")]
    assert a["fl_l1_n"] == 1 and a["fl_l1_wins"] == 0
