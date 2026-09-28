"""model/blocks/form_lines_ab.py by hand: form lines split by rivals ahead and behind, and against the field."""
import numpy as np
import pandas as pd

from model import blocks
from model.blocks import form_lines_ab


def _race(day, order, dnf=(), rid=None):
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    horses = list(order) + list(dnf)
    return [dict(raceid=rid or f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h,
                 placing_numerical=float(i + 1) if i < len(order) else np.nan, bfsp=float(len(horses)))
            for i, h in enumerate(horses)]


def _card(day, horses, rid=None):
    rows = _race(day, horses, rid=rid)
    for r in rows:
        r.update(placing_numerical=np.nan, bfsp=np.nan)
    return rows


def _build(rows):
    out, _ = blocks.attach(pd.DataFrame(rows), ["form_lines_ab"])
    return out.set_index(["race_date", "horse_name"])


def test_rivals_are_split_by_where_they_finished():
    rows = (_race(0, ["B", "A", "C"], dnf=["D"])                  # B beat A; C and D (pulled up) behind it
            + _race(1, ["B", "C", "E", "F"])                       # B wins; C second
            + _race(2, ["C", "G", "H", "I"], rid="r2a")            # C wins
            + _race(2, ["D", "J", "K", "L"], rid="r2b")            # D wins
            + _card(3, ["A", "Z"]))
    a = _build(rows).loc[(pd.Timestamp("2025-06-04"), "A")]
    assert a["fa_ahead_n"] == 1 and a["fa_ahead_wins"] == 1
    assert a["fa_behind_n"] == 3 and a["fa_behind_wins"] == 2 and a["fa_beat_winner"] == 1
    assert np.isclose(a["fa_ahead_ae"], 0.75 / (1 + form_lines_ab.K))           # four runners at equal prices
    assert np.isclose(a["fa_behind_ae"], (-0.25 + 0.75 + 0.75) / (3 + form_lines_ab.K))
    z = _build(rows).loc[(pd.Timestamp("2025-06-04"), "Z")]
    assert np.isnan(z["fa_ahead_n"]) and np.isnan(z["fa_beat_winner"])          # a debutant


def test_a_horse_that_did_not_finish_has_no_split_and_today_never_counts():
    rows = _race(0, ["B"], dnf=["A"]) + _race(1, ["B", "Q"]) + _card(1, ["A", "Y"], rid="c1")
    a = _build(rows).loc[(pd.Timestamp("2025-06-02"), "A")]
    assert np.isnan(a["fa_ahead_n"]) and np.isnan(a["fa_behind_n"])           # pulled up last time
    rows = _race(0, ["A", "B"]) + _race(1, ["B", "Q"]) + _card(1, ["A", "Y"], rid="c1")
    a = _build(rows).loc[(pd.Timestamp("2025-06-02"), "A")]
    assert a["fa_behind_n"] == 0 and a["fa_beat_winner"] == 0                   # B's win is today's


def test_the_field_readings_set_form_lines_against_todays_runners():
    rows = (_race(0, ["A", "B", "C", "D"], rid="x") + _race(0, ["E", "F", "G", "H"], rid="y")
            + _race(1, ["B", "C", "Q1", "Q2"], rid="x1") + _race(1, ["Q3", "F", "Q4", "Q5"], rid="y1")
            + _card(2, ["A", "E"], rid="today"))
    out = _build(rows)
    a, e = out.loc[(pd.Timestamp("2025-06-03"), "A")], out.loc[(pd.Timestamp("2025-06-03"), "E")]
    assert a["fl_l1_ae"] > e["fl_l1_ae"]                                         # A's race has produced a winner
    assert a["fa_l1_ae_gap"] == 0.0 and e["fa_l1_ae_gap"] < 0
    assert np.isclose(a["fa_l1_ae_z"], -e["fa_l1_ae_z"]) and a["fa_l1_ae_z"] > 0
