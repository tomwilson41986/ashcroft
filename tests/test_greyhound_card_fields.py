"""research/greyhound/card_fields.py on planted nested card JSON in a local store: every path walked with list
positions folded, files/records/fill rate counted, the seeding, season, bends, comments, going, reserves and weight
paths flagged, long strings cut, contact-like keys redacted, files spread across days and tracks, and an empty store
reported as no data found. Hand-built fixtures, no real data."""
import json

from research.greyhound import card_fields as cf
from sources.common import Store

LONG = "x" * 200


def card(track, weights=True, comment="QAway,Rails"):
    runners = []
    for trap in range(1, 4):
        runners.append({
            "trap": trap, "name": f"Planted Dog {trap}", "seed": ["R", "M", "RM"][trap - 1],
            "bitchSeason": "01-Sep-26" if trap == 2 else None,
            "weight": {"weight": 30.5 + trap if weights else None},
            "form": [{"date": "2026-10-01", "bends": "1123", "going": "-10", "comment": comment, "remarks": LONG}],
            "trainer": {"name": "Planted Trainer", "email": "someone@example.org"},
        })
    return {"meeting": {"track": track, "going": "N"},
            "races": [{"raceNo": 1, "grade": "A5", "reserves": [{"name": "Planted Reserve", "trap": 0}],
                       "nonRunners": [], "runners": runners}]}


def store_with(tmp_path, files):
    s = Store(root=tmp_path)
    for key, doc in files.items():
        s.put(key, json.dumps(doc).encode())
    s.put("greyhounds_today/2026-10-08/polls.jsonl", b'{"utc": "x"}\n')
    return s


def test_paths_counts_and_flags(tmp_path):
    s = store_with(tmp_path, {
        "greyhounds_today/2026-10-08/101500Z_Romford_card1.json": card("Romford"),
        "greyhounds_today/2026-10-09/111500Z_Central_Park_card_2.json": card("Central Park", weights=False),
    })
    r = cf.run(s, n=40)
    assert r["read"] == 2 and r["card_files_held"] == 2
    assert r["tracks"] == ["Central Park", "Romford"] and r["days"] == ["2026-10-08", "2026-10-09"]
    p = {x["path"]: x for x in r["paths"]}
    w = p["$.races[].runners[].weight.weight"]
    assert (w["files"], w["records"], w["filled"]) == (2, 6, 3) and w["fill_rate"] == 0.5
    assert p["$.races[].runners[].trap"]["records"] == 6 and p["$.races[].runners[].trap"]["fill_rate"] == 1.0
    assert p["$.races[].nonRunners"]["fill_rate"] == 0.0
    fl = r["flagged"]
    assert "$.races[].runners[].seed" in fl["seeding"]
    assert p["$.races[].runners[].seed"]["codes"] == {"R": 2, "M": 2, "RM": 2}
    assert "$.races[].runners[].bitchSeason" in fl["season"]
    assert "$.races[].runners[].form[].bends" in fl["bends_running_line"]
    assert "$.races[].runners[].form[].comment" in fl["comments"]
    assert "$.races[].runners[].form[].remarks" in fl["comments"]
    assert {"$.meeting.going", "$.races[].runners[].form[].going"} <= set(fl["going"])
    assert {"$.races[].reserves", "$.races[].nonRunners"} <= set(fl["reserves_non_runners"])
    assert "$.races[].runners[].weight.weight" in fl["weights"]
    assert all(len(e) <= cf.CUT for e in p["$.races[].runners[].form[].remarks"]["examples"])
    assert p["$.races[].runners[].trainer.email"]["examples"] == ["<redacted>"]
    md = cf.markdown(r)
    assert "someone@example.org" not in md and "someone@example.org" not in json.dumps(r)
    assert "| `$.races[].runners[].weight.weight` | 2 | 6 | 0.500 |" in md


def test_files_spread_across_days_and_tracks_last_version_first():
    keys = [f"greyhounds_today/2026-10-{d:02d}/{h:02d}0000Z_{t}_c.json"
            for d in range(1, 11) for t in ("Hove", "Kinsley") for h in (9, 12, 15)]
    got = cf.choose(keys, 4)
    assert len(got) == 4
    assert all("150000Z" in k for k in got)                      # the last version of each day and track
    assert {cf.parse_key(k)[1] for k in got} == {"Hove", "Kinsley"}
    assert cf.parse_key(got[0])[0] == "2026-10-01" and cf.parse_key(got[-1])[0] == "2026-10-10"
    assert len(cf.choose(keys, 100)) == len(keys)
    assert cf.choose(keys, 5, first="2026-10-10") and all("2026-10-10" in k for k in cf.choose(keys, 5, first="2026-10-10"))


def test_no_cards_is_no_data_found(tmp_path):
    r = cf.run(Store(root=tmp_path), n=40)
    assert r["read"] == 0 and r["paths"] == []
    assert "No data found" in cf.markdown(r)
