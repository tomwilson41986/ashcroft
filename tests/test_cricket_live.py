"""The live path: a Cricsheet replay through the engine (tests/fixtures/cricket), with a fixed-probability
stand-in for the next-ball model (the engine's bookkeeping is tested here, not the model)."""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cricket.ingest.ball_feed import CricsheetReplay, FeedRecorder
from cricket.live.engine import Engine

FIX = Path(__file__).parent / "fixtures" / "cricket"
T20 = []
for p in sorted(FIX.glob("*.json.gz")):
    raw = json.loads(gzip.decompress(p.read_bytes()))
    if raw["info"]["match_type"] == "T20" and raw["info"].get("overs") == 20:
        T20.append((p.name.split(".")[0], raw))


class FixedModel:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float32)

    def inplace_predict(self, x):
        return np.tile(self.p, (len(x), 1))


def _bundle():
    cols = ["player_id", "n_eff"] + [f"p{k}" for k in range(8)]
    probs = [0.3, 0.38, 0.07, 0.004, 0.105, 0.045, 0.054, 0.042]
    probs = list(np.array(probs) / sum(probs))
    return {"bundle": {"version": "test", "prior": probs, "population": {"rpb": 1.3, "wpb": 0.05},
                       "extras_runs": {"values": [1], "p": [1.0]}, "calibration": {}},
            "booster": FixedModel(probs), "bat_ratings": pd.DataFrame(columns=cols),
            "bowl_ratings": pd.DataFrame(columns=cols),
            "venues": pd.DataFrame(columns=["venue_id", "venue_rpb", "venue_wpb"]),
            "comps": pd.DataFrame(columns=["event_name", "comp_rpb"])}


def test_replay_state_matches_the_scorecard_and_quotes_are_probabilities(tmp_path):
    mid, raw = T20[0]
    eng = Engine(_bundle(), paths=50)
    rec = FeedRecorder(tmp_path / "feed.jsonl")
    quotes = []
    totals = {}
    for ev in rec.wrap(CricsheetReplay(mid, raw)):
        eng.on_event(ev)
        if ev.kind == "delivery" and len(quotes) < 12 or ev.kind == "innings" and ev.payload["innings"] == 2:
            q = eng.price()
            quotes.append(q)
            assert abs(sum(q["p_raw"].values()) - 1) < 1e-9 and all(0 <= v <= 1 for v in q["p_raw"].values())
        if ev.kind == "delivery":
            totals[eng.inn.number] = eng.inn.state.runs
    tgt = [i for i in raw["innings"] if not i.get("super_over")][1]["target"]["runs"]
    assert totals[1] + 1 == tgt
    assert quotes[0]["innings"] == 1 and "innings_total" in quotes[0]
    assert quotes[-1]["innings"] == 2 and quotes[-1]["target"] == tgt
    # every event kept with the time it reached us
    lines = (tmp_path / "feed.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "setup" and all(json.loads(x)["receipt_utc"].endswith("Z") for x in lines)


def test_strike_rotation_names_the_next_ball_striker():
    """Between two deliveries of the same innings with no wicket, the engine's inferred striker for the coming ball
    should be the batter the next delivery records."""
    right = total = 0
    for mid, raw in T20:
        eng = Engine(_bundle(), paths=1)
        prev_out = True
        for ev in CricsheetReplay(mid, raw).events():
            if ev.kind == "delivery" and eng.inn is not None and eng.inn.number == ev.payload["innings"] \
                    and not prev_out and eng.inn.state.legal_balls > 0:
                total += 1
                right += eng.inn.striker == ev.payload["batter_id"]
            eng.on_event(ev)
            if ev.kind == "delivery":
                prev_out = ev.payload["team_wickets"] > 0
    assert total > 5000
    assert right / total > 0.97, right / total
