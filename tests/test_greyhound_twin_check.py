"""research/greyhound/twin_check.py: a dog with two rows on a day, or a run a day after its last, marks its race."""

import pandas as pd

from research.greyhound.twin_check import audit, flags, markdown


def frame():
    rows = []
    # race A and its copy B on the same day share dog 1; race C is clean; race D holds a run a day after the last
    for race, day, dogs, gaps in (("A", "2024-03-01", (1, 2), (7, 7)), ("B", "2024-03-01", (1, 3), (0, 9)),
                                  ("C", "2024-03-02", (4, 5), (6, None)), ("D", "2025-01-05", (6, 7), (1, 14))):
        for dog, gap in zip(dogs, gaps):
            rows.append({"race_id": race, "day": day, "dog_id": dog, "days_since": gap})
    return pd.DataFrame(rows)


def test_a_twin_and_a_short_gap_mark_their_races_and_a_clean_race_is_left():
    d = frame()
    f = flags(d)
    assert f.twin.tolist() == [True, False, True, False, False, False, False, False]
    assert f.short.tolist() == [False, False, True, False, False, False, True, False]
    touched = d.race_id[f.touched].unique().tolist()
    assert touched == ["A", "B", "D"]                       # C: no twin, and a first run (no gap) is not a short gap


def test_the_audit_counts_by_year_and_the_report_says_when_nothing_is_touched():
    d = frame()
    a = audit(d, flags(d))
    by = {r["year"]: r for r in a["by_year"]}
    assert by["2024"]["races"] == 3 and by["2024"]["twin_rows"] == 2 and by["2024"]["races_touched"] == 2
    assert by["2025"]["short_gap_rows"] == 1 and by["all"]["races_touched"] == 3 == a["races_touched"]
    clean = d[d.race_id == "C"]
    res = {"audit": audit(clean, flags(clean))}
    assert res["audit"]["races_touched"] == 0 and "No race is touched" in markdown(res)
