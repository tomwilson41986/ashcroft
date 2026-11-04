"""cricket/state/match_state.py against fifty Cricsheet scorecards (tests/fixtures/cricket/README.md).

The ball-by-ball replay must reproduce what Cricsheet records apart from it: the second innings' target is the
first innings' total plus one, and the result margin (by runs: first total minus second; by wickets: ten minus the
wickets the chasing side lost). The live reducer and the vectorised table builder must give the same states.
"""

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from cricket.ingest.cricsheet import parse_match
from cricket.state.match_state import overs_to_balls, phase_of, replay, states_frame

FIX = Path(__file__).parent / "fixtures" / "cricket"
FILES = sorted(FIX.glob("*.json.gz"))


def _load(path):
    mid = path.name[: -len(".json.gz")]
    return parse_match(mid, json.loads(gzip.decompress(path.read_bytes())))


def _innings(part):
    inns = [i for i in part["innings"] if not i["super_over"]]
    dels = {i["innings"]: [d for d in part["deliveries"] if d["innings"] == i["innings"]] for i in inns}
    return inns, dels


def test_fifty_fixtures_across_ipl_t20i_odi():
    assert len(FILES) == 50
    groups = {"T20": 0, "ODI": 0}
    for p in FILES:
        groups[_load(p)["matches"][0]["match_type"]] += 1
    assert groups == {"T20": 35, "ODI": 15}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name.split(".")[0])
def test_replay_reproduces_the_scorecard(path):
    part = _load(path)
    m = part["matches"][0]
    inns, dels = _innings(part)
    finals = {i["innings"]: replay(m, i, dels[i["innings"]])[-1] for i in inns}
    first, second = finals[1], finals[2]
    # the second innings target is the first innings total plus one (no D/L in the fixtures)
    assert inns[1]["target_runs"] == first.runs + 1
    if m["by_runs"] is not None:
        assert m["winner"] == inns[0]["team"]
        assert first.runs - second.runs == m["by_runs"]
    else:
        assert m["winner"] == inns[1]["team"]
        assert second.runs >= second.target
        assert 10 - second.wickets == m["by_wickets"]
    # never more legal balls than the scheduled overs (plus any over Cricsheet flags as miscounted by the umpires,
    # as 1486225's seven-ball 19th over), never more than ten wickets
    for i in inns:
        s = finals[i["innings"]]
        extra = sum(int(v["balls"]) - 6 for v in json.loads(i["miscounted_overs"] or "{}").values())
        assert s.wickets <= 10
        assert s.max_balls is None or s.legal_balls <= s.max_balls + extra


@pytest.mark.parametrize("path", FILES[::5], ids=lambda p: p.name.split(".")[0])
def test_vectorised_states_equal_the_live_reducer(path):
    part = _load(path)
    m = part["matches"][0]
    inns, dels = _innings(part)
    frame = states_frame(pd.DataFrame(part["deliveries"]), pd.DataFrame(part["innings"]), pd.DataFrame([m]))
    for inn in inns:
        states = replay(m, inn, dels[inn["innings"]])[:-1]
        f = frame[frame.innings == inn["innings"]].reset_index(drop=True)
        assert len(f) == len(states)
        for row, s in zip(f.itertuples(), states):
            assert (row.runs_before, row.wickets_before, row.legal_before) == (s.runs, s.wickets, s.legal_balls)
            assert row.batter == s.striker and row.bowler == s.bowler
            assert row.phase == s.phase
            assert row.last6_runs == sum(r for r, _ in s.last6)
            assert row.last6_wkts == sum(w for _, w in s.last6)
            if s.target is None:
                assert pd.isna(row.target)
            else:
                assert row.runs_needed == s.runs_needed
                if s.balls_left:
                    assert row.required_rate == pytest.approx(s.required_rate)


def test_balls_since_wicket_resets_after_each_wicket():
    part = _load(FILES[0])
    frame = states_frame(pd.DataFrame(part["deliveries"]), pd.DataFrame(part["innings"]),
                         pd.DataFrame(part["matches"]))
    f = frame[frame.innings == 1].reset_index(drop=True)
    since = 0
    for row in f.itertuples():
        assert row.balls_since_wicket == since
        since = 0 if row.team_wickets > 0 else since + int(row.legal)


def test_overs_notation_and_phases():
    assert overs_to_balls(17.4) == 106
    assert overs_to_balls(20) == 120
    assert phase_of("T20", 35) == "powerplay" and phase_of("T20", 36) == "middle" and phase_of("T20", 90) == "death"
    assert phase_of("ODI", 59) == "powerplay" and phase_of("ODI", 240) == "death"
