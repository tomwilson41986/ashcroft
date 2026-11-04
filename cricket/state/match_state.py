"""Deterministic match state from a delivery sequence (build spec 6.1).

The state is what is known *before* a delivery: innings, runs, wickets, legal balls bowled, balls left, the target
(second innings only), the striker, non-striker and bowler, the phase, and the last six legal balls. Two forms that
must agree (tests/test_cricket_state.py):

- ``step(state, delivery)``: the live reducer, one delivery at a time;
- ``states_frame(deliveries, innings, matches)``: the same states for a whole table at once, for training.

Wickets count against the team as in a scorecard: retired hurt and retired not out do not (cricket/ingest).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# phase boundaries in legal balls bowled: (powerplay ends, middle ends); the death overs follow
PHASES = {"T20": (36, 90), "IT20": (36, 90), "ODI": (60, 240), "ODM": (60, 240)}


def overs_to_balls(overs, balls_per_over: int = 6) -> int | None:
    """Cricket overs notation (17.4 = 17 overs and 4 balls) to balls."""
    if overs is None or (isinstance(overs, float) and np.isnan(overs)):
        return None
    whole = int(overs)
    part = int(round((float(overs) - whole) * 10))
    return whole * balls_per_over + part


def phase_of(match_type: str, legal_balls: int) -> str:
    pp, mid = PHASES.get(match_type, (36, 90))
    return "powerplay" if legal_balls < pp else ("middle" if legal_balls < mid else "death")


@dataclass(frozen=True)
class State:
    match_type: str
    innings: int
    max_balls: int | None              # legal balls the innings may last (scheduled or revised overs)
    target: int | None = None          # runs to win, second innings only (None in the first: never a feature there)
    runs: int = 0
    wickets: int = 0
    legal_balls: int = 0
    striker: str | None = None
    non_striker: str | None = None
    bowler: str | None = None
    last6: tuple = ()                  # (runs, wicket) of the last six legal balls, oldest first

    @property
    def balls_left(self) -> int | None:
        return None if self.max_balls is None else self.max_balls - self.legal_balls

    @property
    def phase(self) -> str:
        return phase_of(self.match_type, self.legal_balls)

    @property
    def runs_needed(self) -> int | None:
        return None if self.target is None else self.target - self.runs

    @property
    def required_rate(self) -> float | None:
        """Runs needed per over (six balls); None in the first innings or with no balls left."""
        bl = self.balls_left
        if self.target is None or not bl:
            return None
        return 6.0 * self.runs_needed / bl

    @property
    def over(self) -> int:
        return self.legal_balls // 6


def step(s: State, d: dict) -> State:
    """The state after delivery ``d`` (keys: runs_total, legal, team_wickets, batter, non_striker, bowler)."""
    legal = bool(d["legal"])
    last6 = s.last6
    if legal:
        last6 = (last6 + ((int(d["runs_total"]), int(d["team_wickets"]) > 0),))[-6:]
    return replace(s, runs=s.runs + int(d["runs_total"]), wickets=s.wickets + int(d["team_wickets"]),
                   legal_balls=s.legal_balls + int(legal), striker=d.get("batter"),
                   non_striker=d.get("non_striker"), bowler=d.get("bowler"), last6=last6)


def start(match: dict, innings: dict) -> State:
    """The state before the first ball of an innings, from its ``matches`` and ``innings`` rows."""
    bpo = int(match.get("balls_per_over") or 6)
    max_balls = None
    if match.get("overs") is not None and not pd.isna(match.get("overs")):
        max_balls = int(match["overs"]) * bpo
    target = None
    if innings.get("target_runs") is not None and not pd.isna(innings.get("target_runs")):
        target = int(innings["target_runs"])
        revised = overs_to_balls(innings.get("target_overs"), bpo)
        if revised:
            max_balls = revised
    return State(match_type=match["match_type"], innings=int(innings["innings"]), max_balls=max_balls, target=target)


def replay(match: dict, innings: dict, deliveries: list[dict]) -> list[State]:
    """The state before each delivery of one innings, and the final state last (len = deliveries + 1)."""
    s = start(match, innings)
    out = [s]
    for d in deliveries:
        s = step(s, d)
        out.append(s)
    # the "before" view names who is on strike for the coming ball: shift the names forward one delivery
    named = [replace(out[i], striker=deliveries[i]["batter"], non_striker=deliveries[i]["non_striker"],
                     bowler=deliveries[i]["bowler"]) for i in range(len(deliveries))]
    return named + [out[-1]]


def states_frame(deliveries: pd.DataFrame, innings: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """For every delivery, the state before it (vectorised; the same numbers ``replay`` gives).

    Columns added: runs_before, wickets_before, legal_before, max_balls, balls_left, target, runs_needed,
    required_rate (per six balls), phase, last6_runs, last6_wkts, balls_since_wicket. Rows keep their order within
    each innings; the frame is sorted by (match_id, innings, over, repeat, ball).
    """
    d = deliveries.sort_values(["match_id", "innings", "over", "repeat", "ball"], kind="stable").reset_index(drop=True)
    g = d.groupby(["match_id", "innings"], sort=False)
    legal = d["legal"].astype(np.int32)
    d["runs_before"] = g["runs_total"].cumsum().astype(np.int32) - d["runs_total"].astype(np.int32)
    d["wickets_before"] = g["team_wickets"].cumsum().astype(np.int32) - d["team_wickets"].astype(np.int32)
    d["legal_before"] = legal.groupby([d.match_id, d.innings], sort=False).cumsum().astype(np.int32) - legal

    m = matches.set_index("match_id")
    bpo = d["match_id"].map(m["balls_per_over"]).fillna(6).astype(int)
    sched = d["match_id"].map(m["overs"]) * bpo
    inn = innings.set_index(["match_id", "innings"])
    idx = pd.MultiIndex.from_arrays([d.match_id, d.innings])
    target = inn["target_runs"].reindex(idx).to_numpy()
    tov = inn["target_overs"].reindex(idx).to_numpy()
    revised = np.array([overs_to_balls(x) if not pd.isna(x) else np.nan for x in tov], dtype=float)
    d["target"] = target
    d["max_balls"] = np.where(~np.isnan(revised) & (revised > 0), revised, sched.to_numpy(dtype=float))
    d["balls_left"] = d["max_balls"] - d["legal_before"]
    d["runs_needed"] = d["target"] - d["runs_before"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["required_rate"] = np.where(d["balls_left"] > 0, 6.0 * d["runs_needed"] / d["balls_left"], np.nan)
    mt = d["match_id"].map(m["match_type"])
    pp = mt.map(lambda x: PHASES.get(x, (36, 90))[0])
    mid = mt.map(lambda x: PHASES.get(x, (36, 90))[1])
    d["phase"] = np.where(d.legal_before < pp, "powerplay", np.where(d.legal_before < mid, "middle", "death"))

    # last six legal balls before this delivery, and legal balls since the last wicket
    key = d["match_id"].astype(str) + "|" + d["innings"].astype(str)
    lr = d["runs_total"].where(d["legal"], 0).astype(np.int64)
    lw = ((d["team_wickets"] > 0) & d["legal"]).astype(np.int64)
    d["last6_runs"] = _last_n_legal(lr, key, d["legal_before"], 6)
    d["last6_wkts"] = _last_n_legal(lw, key, d["legal_before"], 6)
    # legal balls bowled at the delivery after the latest wicket, carried forward
    after = (d["legal_before"] + legal).where(d["team_wickets"] > 0)
    after = after.groupby(key, sort=False).shift(1).groupby(key, sort=False).ffill().fillna(0)
    d["balls_since_wicket"] = (d["legal_before"] - after).astype(np.int32)
    return d


def _last_n_legal(vals: pd.Series, key: pd.Series, legal_before: pd.Series, n: int) -> np.ndarray:
    """Sum of ``vals`` (zero on illegal deliveries) over the last ``n`` legal deliveries before each row.

    Every row with the same count of legal balls before it carries the same running sum, so the sum ``n`` legal
    balls back is a lookup of that running sum at ``legal_before - n``.
    """
    before = vals.groupby(key, sort=False).cumsum() - vals
    look = pd.DataFrame({"key": key, "lb": legal_before, "s": before}).drop_duplicates(["key", "lb"])
    q = pd.DataFrame({"key": key, "lb": legal_before - n})
    back = q.merge(look, on=["key", "lb"], how="left")["s"].fillna(0).to_numpy()
    return (before.to_numpy() - back).astype(np.int32)
