"""The Betfair-native model joined to the lab's features: only linked matches, never one played on its collection day."""

from __future__ import annotations

import pandas as pd

from research.open_close import football_bf_lab_join as lj


def rows(kickoffs):
    return pd.DataFrame({"event_id": [f"e{i}" for i in range(len(kickoffs))],
                         "kickoff": pd.to_datetime(kickoffs, utc=True), "y_logit": 0.0})


def test_only_linked_matches_with_lab_rows_are_kept_and_collection_days_are_left_out():
    # Sat 2026-11-21 (collected Fri 20th): kept. Fri 2026-11-20 is its own collection day: out.
    # Wed 2026-11-25 (collected Tue 24th): kept. Tue 2026-11-24 is its own collection day: out.
    d = rows(["2026-11-21 15:00", "2026-11-20 19:45", "2026-11-25 19:45", "2026-11-24 19:45", "2026-11-22 13:00"])
    link = pd.DataFrame({"event_id": ["e0", "e1", "e2", "e3", "e4"], "match_id": ["m0", "m1", "m2", "m3", None]})
    labt = pd.DataFrame({"match_id": ["m0", "m1", "m2", "m3"], "lab_x": [1.0, 2.0, 3.0, 4.0]})
    got = lj.join(d, link, labt)
    assert list(got["event_id"]) == ["e0", "e2"]          # e4 has no canonical match
    assert list(got["lab_x"]) == [1.0, 3.0]


def test_an_event_linked_twice_counts_once():
    d = rows(["2026-11-21 15:00"])
    link = pd.DataFrame({"event_id": ["e0", "e0"], "match_id": ["m0", "m0"]})
    labt = pd.DataFrame({"match_id": ["m0"], "lab_x": [1.0]})
    assert len(lj.join(d, link, labt)) == 1
