"""The cricket model parts: ratings stay as-of (no match rates itself), the simulator's mechanics, the metrics.

Real Cricsheet fixtures only (tests/fixtures/cricket). The simulator test uses a fixed-probability stand-in for the
next-ball model: it checks the roll-out's bookkeeping, it trains and evaluates nothing.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cricket.ingest.cricsheet import parse_match
from cricket.model import simulator as sim
from cricket.model.features import BAT, BOWL, CONTEXT, FEATURES, add_state_features
from cricket.model.metrics import ece, log_loss
from cricket.model.outcomes import CLASSES, EXTRA, W, classify
from cricket.ratings.player_ratings import C_PRIOR, ratings
from cricket.ratings.shrinkage import decayed_asof
from cricket.state.match_state import states_frame

FIX = Path(__file__).parent / "fixtures" / "cricket"


def _tables():
    parts = [parse_match(p.name[:-8], json.loads(gzip.decompress(p.read_bytes()))) for p in sorted(FIX.glob("*.gz"))]
    return {k: pd.DataFrame([r for part in parts for r in part[k]]) for k in ("matches", "innings", "deliveries",
                                                                              "players")}


def test_classes():
    d = pd.DataFrame({"runs_total": [0, 1, 5, 7, 1, 0], "team_wickets": [0, 0, 0, 0, 0, 1],
                      "legal": [True, True, True, True, False, True]})
    assert [CLASSES[c] for c in classify(d)] == ["0", "1", "4", "6", "extra", "W"]


def test_decay_counts_only_earlier_matches():
    df = pd.DataFrame({"e": ["a", "a", "a", "b"], "d": [1, 2, 3, 1], "x": [1.0, 2.0, 4.0, 9.0]})
    out = decayed_asof(df, "e", ["d"], ["x"], 0.5)[:, 0]
    assert out.tolist() == [0.0, 1.0, 0.5 * 1.0 + 2.0, 0.0]


def test_player_ratings_are_as_of_the_match_start():
    t = _tables()
    f = states_frame(t["deliveries"], t["innings"], t["matches"])
    f["cls"] = classify(f)
    prior = np.full(8, 1 / 8)
    r = ratings(f, t["players"], t["matches"], "bat", prior)
    first = r.merge(t["matches"][["match_id", "start_date"]], on="match_id").sort_values("start_date")
    first = first.groupby("player_id").head(1)
    # a player's first match in the data carries no experience and rates at the prior exactly
    assert (first.n_eff == 0).all()
    assert np.allclose(first[[f"p{k}" for k in range(8)]].to_numpy(), prior)
    # a later match counts the earlier match's balls only (r = 0.88 decays them from the next match on)
    pid = f.groupby("batter_id").match_id.nunique().idxmax()
    rows = r[r.player_id == pid].merge(t["matches"][["match_id", "start_date"]], on="match_id")
    rows = rows.sort_values("start_date").reset_index(drop=True)
    faced = f[f.batter_id == pid].groupby("match_id").size()
    if len(rows) > 1:
        assert rows.n_eff[1] == pytest.approx(faced.get(rows.match_id[0], 0))
    assert C_PRIOR == 60.0


class FixedModel:
    """Next-ball probabilities fixed for every row: a stand-in to test the roll-out's bookkeeping."""

    def __init__(self, p):
        self.p = np.asarray(p, dtype=np.float32)

    def inplace_predict(self, x):
        return np.tile(self.p, (len(x), 1))


def _prep(runs_before, wickets_before, legal_before, target, max_balls=120):
    s = pd.DataFrame({"match_id": ["m"], "innings": [2], "runs_before": [runs_before],
                      "wickets_before": [wickets_before], "legal_before": [legal_before], "max_balls": [max_balls],
                      "target": [target], "balls_since_wicket": [0]})
    zero9 = np.r_[np.full(8, 1 / 8), 0.0][None, :]
    return {"s": s, "striker": zero9.copy(), "non": zero9.copy(), "bowler": zero9.copy(),
            "queue": np.tile(zero9, (1, 10, 1)), "team_bowl": zero9.copy(), "lag_r": np.zeros((1, 6)),
            "lag_w": np.zeros((1, 6)), "context": np.zeros((1, len(CONTEXT)))}


def test_roll_out_bookkeeping():
    ex_v, ex_p = np.array([1]), np.array([1.0])
    ones = np.zeros(8)
    ones[1] = 1.0                       # every ball a single
    # 10 needed, 12 balls left: always won
    assert sim.win_probability(FixedModel(ones), _prep(100, 0, 108, 110), 50, ex_v, ex_p)[0] == 1.0
    # 14 needed off 12 singles: always lost; 13 needed ends level on 109: a tie scores one half
    assert sim.win_probability(FixedModel(ones), _prep(96, 0, 108, 110), 50, ex_v, ex_p)[0] == 0.0
    assert sim.win_probability(FixedModel(ones), _prep(97, 0, 108, 110), 50, ex_v, ex_p)[0] == 0.5
    wk = np.zeros(8)
    wk[W] = 1.0                         # every ball a wicket: nine down, all out next ball, short of the target
    assert sim.win_probability(FixedModel(wk), _prep(100, 9, 60, 105), 20, ex_v, ex_p)[0] == 0.0
    assert sim.win_probability(FixedModel(wk), _prep(100, 9, 60, 101), 20, ex_v, ex_p)[0] == 0.5   # all out level
    ex = np.zeros(8)
    ex[EXTRA], ex[0] = 0.5, 0.5         # wides use no legal ball: half of all deliveries add a run
    p = sim.win_probability(FixedModel(ex), _prep(100, 0, 114, 104), 2000, ex_v, ex_p, seed=1)[0]
    # 4 needed off 6 legal balls: won with 4 wides before the 6th dot, P(Bin(9, .5) >= 4) = 382/512; level on 3
    # wides, a tie, P = C(8, 3) / 2^9 = 56/512, scores one half
    assert p == pytest.approx(382 / 512 + 0.5 * 56 / 512, abs=0.03)


def test_feature_layout():
    assert FEATURES[: 2] == ["innings", "legal_before"] and BAT[-1] == "bat_n" and BOWL[-1] == "bowl_n"
    f = pd.DataFrame({"legal_before": [0, 6], "runs_before": [0, 9], "required_rate": [np.nan, 8.0],
                      "phase": ["powerplay", "powerplay"], "innings": [1, 2], "runs_needed": [5.0, 100.0],
                      "target": [200.0, 200.0]})
    g = add_state_features(f)
    # the target is never a feature in the first innings
    assert g.loc[0, ["runs_needed", "rrr", "target"]].isna().all()
    assert g.crr[1] == 9.0


def test_metrics():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert log_loss(y, p) == pytest.approx(np.log(2))
    assert ece(y, p) == pytest.approx(0.0)
    assert ece(np.array([1, 1]), np.array([0.25, 0.25])) == pytest.approx(0.75)
