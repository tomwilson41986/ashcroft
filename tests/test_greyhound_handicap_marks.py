"""research/greyhound/handicap_marks.py: card marks before the off, matched to GBGB by day, dog and trap."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.greyhound.handicap_marks import card_rows, compare, per_runner, snapshot_time


def race(rid, off, grade, marks, names):
    return {"race_id": rid, "region": "GB", "track": {"name": "Romford"}, "grade": grade, "is_handicap": False,
            "scheduled_start": {"utc": off},
            "runners": [{"trap": t, "dog_name": n, "runner_status": "runner", "handicap_metres": m}
                        for t, (n, m) in enumerate(zip(names, marks), start=1)]}


def test_rows_before_the_off_only_and_the_first_lead_and_last_mark():
    early = snapshot_time("greyhoundapi/raw/cards/2026-11-20/0843.json.gz")
    late = snapshot_time("greyhoundapi/raw/cards/2026-11-20/1243.json.gz")
    hp = race(1, "2026-11-20T12:00:00Z", "HP", [None, None], ["Ann's Dog", "Bob"])
    hp2 = race(1, "2026-11-20T12:00:00Z", "HP", [12, 0], ["Ann's Dog", "Bob"])
    a5 = race(2, "2026-11-20T14:00:00Z", "A5", [0, 0], ["Cid", "Dot"])
    rows = card_rows({"today": [hp, a5]}, early) + card_rows({"today": [hp2, a5]}, late)
    c = pd.DataFrame(rows)
    assert set(c.race_id) == {"1", "2"} and (c.lead_h > 0).all()     # race 1 is past its off at 12:43: not read
    r = per_runner(c)
    one = r[r.race_id == "1"].set_index("trap")
    assert one.mark.isna().all() and one.first_seen_lead_h.iloc[0] == (12 * 60 - (8 * 60 + 43)) / 60
    two = r[r.race_id == "2"]
    assert two.mark.tolist() == [0.0, 0.0] and not two.mark_changed.any()


def test_compare_counts_equal_marks_and_mismatches():
    at = datetime(2026, 11, 20, 8, 43, tzinfo=timezone.utc)
    rows = card_rows({"today": [race(1, "2026-11-20T12:00:00Z", "HP", [12, 0], ["Ann's Dog", "Bob"]),
                                race(2, "2026-11-20T14:00:00Z", "A5", [0, 0], ["Cid", "Dot"])]}, at)
    r = per_runner(pd.DataFrame(rows))
    g = pd.DataFrame({"race_date": ["2026-11-20"] * 4, "race_id": [10, 10, 11, 11], "trap": [1, 2, 1, 2],
                      "dog_name": ["Anns Dog", "BOB", "Cid", "Dot"], "track": "Romford", "race_time": "12:00",
                      "handicap": [True, True, False, False], "trap_handicap": ["R12", "R3", None, None],
                      "hcp_start": [12.0, 3.0, np.nan, np.nan]})
    h = compare(r, g)
    assert h["handicaps"]["races"] == 1 and h["handicaps"]["runners_compared"] == 2
    assert h["handicaps"]["marks_equal"] == 1 and h["handicaps"]["mismatches"][0]["dog_name"] == "BOB"
    assert h["other_races"]["mark"] == {"0.0": 2}
