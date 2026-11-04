"""Vectorised Monte Carlo roll-out (build spec 6.4): win probability and the runs distribution from the next-ball model.

Every path of every state in a batch is stepped together: one model call per step for all live paths.

- ``win_probability``: a chase. A path ends when the target is reached (win), or on the tenth wicket or the last legal
  ball (loss; level at the end is a tie and scores one half).
- ``match_probability``: a first-innings state. Each path plays out the rest of the first innings, then the whole
  chase of its own total plus one. Returns P(the side batting first wins) and every path's first-innings total, the
  runs distribution the innings-runs line is priced from (no second model needed).

Who faces and who bowls, inside a path:
- the striker, non-striker and bowler of the state are the real ones; the current bowler finishes the over;
- after a wicket the new batter is the next listed player of the batting XI who has not yet batted (Cricsheet's
  list order; not guaranteed to be the batting order: stated in the reports), on strike;
- after the current over, the bowler is the bowling XI's rating averaged by each player's bowling experience;
- a simulated chase opens with the first two listed players of the chasing XI against that XI-average bowling;
- strike changes on odd runs from a legal ball and at the end of each over.
Simplifications from cricket/model/outcomes.py apply (no runs on wicket balls, no wickets on wides or no-balls).

The batch (``Batch``) is plain arrays, one row a state, so the historical evaluation (``prepare``) and the live
engine (cricket/live/engine.py) build the same thing from their own sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cricket.model.features import BAT, BOWL, CONTEXT, FEATURES
from cricket.model.next_ball import predict
from cricket.model.outcomes import EXTRA, RUNS, W

NR = len(BAT)            # 8 probabilities and the experience n
COL = {c: i for i, c in enumerate(FEATURES)}
QUEUE = 10
T20_BALLS = 120


@dataclass
class Batch:
    """One row a state. Rating vectors are the 8 outcome probabilities and the experience n (9 numbers)."""
    innings: np.ndarray          # (S,) 1 or 2
    runs: np.ndarray             # (S,)
    wickets: np.ndarray
    legal: np.ndarray            # legal balls bowled
    max_balls: np.ndarray
    target: np.ndarray           # NaN in the first innings
    bsw: np.ndarray              # legal balls since the last wicket
    lag_r: np.ndarray            # (S, 6) runs of the last six legal balls, oldest first
    lag_w: np.ndarray            # (S, 6) wickets of the last six legal balls
    striker: np.ndarray          # (S, 9)
    non: np.ndarray              # (S, 9)
    bowler: np.ndarray           # (S, 9)
    queue: np.ndarray            # (S, QUEUE, 9) batters to come, in order
    team_bowl: np.ndarray        # (S, 9) the bowling XI's experience-weighted rating
    ctx: np.ndarray              # (S, len(CONTEXT))
    # the chase that follows a first-innings state
    open2: np.ndarray | None = None        # (S, 2, 9) the chasing XI's first two listed players
    queue2: np.ndarray | None = None       # (S, QUEUE, 9) the rest of the chasing XI
    team_bowl2: np.ndarray | None = None   # (S, 9) the side batting first, as bowlers
    max_balls2: np.ndarray | None = None   # (S,)

    def __len__(self) -> int:
        return len(self.runs)


# ---------------------------------------------------------------------------------------------------------------
# ratings helpers shared with the live engine
# ---------------------------------------------------------------------------------------------------------------

def prior_vector(prior: np.ndarray) -> np.ndarray:
    return np.r_[prior, 0.0]


def team_bowling(vectors: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """The XI's bowling rating weighted by each player's bowling experience (the prior for a side with none)."""
    if len(vectors) == 0 or vectors[:, -1].sum() <= 0:
        return prior_vector(prior)
    w = vectors[:, -1]
    return np.r_[(vectors[:, :8] * w[:, None]).sum(0) / w.sum(), w.mean()]


def fill_queue(vectors: np.ndarray, prior: np.ndarray, n: int = QUEUE) -> np.ndarray:
    q = np.tile(prior_vector(prior), (n, 1))
    k = min(len(vectors), n)
    if k:
        q[:k] = vectors[:k]
    return q


# ---------------------------------------------------------------------------------------------------------------
# historical batches (evaluation)
# ---------------------------------------------------------------------------------------------------------------

def _rating_table(r: pd.DataFrame) -> pd.DataFrame:
    cols = [f"p{k}" for k in range(8)] + ["n_eff"]
    return r.set_index(["player_id", "match_id"])[cols]


def prepare(states: pd.DataFrame, deliveries: pd.DataFrame, players: pd.DataFrame, bat: pd.DataFrame,
            bowl: pd.DataFrame, prior: np.ndarray) -> Batch:
    """A batch from rows of the feature frame (cricket/model/features.py), one innings of one or more matches."""
    s = states.reset_index(drop=True)
    S = len(s)
    bt, bw = _rating_table(bat), _rating_table(bowl)
    pv = prior_vector(prior)

    def look(tab, pids, mid):
        idx = pd.MultiIndex.from_arrays([np.asarray(pids, dtype=object), np.full(len(pids), mid, dtype=object)])
        v = tab.reindex(idx).to_numpy(dtype=float).reshape(len(pids), 9)
        v[np.isnan(v).any(1)] = pv
        return v

    non = np.tile(pv, (S, 1))
    queue = np.tile(pv, (S, QUEUE, 1))
    team_bowl = np.tile(pv, (S, 1))
    first = bool((s.innings == 1).all())
    open2 = np.tile(pv, (S, 2, 1)) if first else None
    queue2 = np.tile(pv, (S, QUEUE, 1)) if first else None
    team_bowl2 = np.tile(pv, (S, 1)) if first else None
    pl = players.sort_values("list_order")
    for (mid, inn), g in s.groupby(["match_id", "innings"], sort=False):
        bat_team, bowl_team = g.team.iloc[0], g.bowl_team.iloc[0]
        dl = deliveries[(deliveries.match_id == mid) & (deliveries.innings == inn)].sort_values(
            ["over", "repeat", "ball"])
        xi = pl[(pl.match_id == mid) & (pl.team == bat_team)].player_id.tolist()
        bxi = pl[(pl.match_id == mid) & (pl.team == bowl_team)].player_id.tolist()
        team_bowl[g.index] = team_bowling(look(bw, bxi, mid), prior)
        if first:
            chase_bat = look(bt, bxi, mid)
            open2[g.index] = fill_queue(chase_bat[:2], prior, 2)
            queue2[g.index] = fill_queue(chase_bat[2:], prior)
            team_bowl2[g.index] = team_bowling(look(bw, xi, mid), prior)
        names = list(zip(dl.batter_id, dl.non_striker_id))
        pos = {k: i for i, k in enumerate(zip(dl.over, dl.repeat, dl.ball))}
        for i, row in zip(g.index, g.itertuples()):
            j = pos[(row.over, row.repeat, row.ball)]
            seen = {p for pair in names[: j + 1] for p in pair}
            todo = [p for p in xi if p not in seen]
            queue[i] = fill_queue(look(bt, todo, mid) if todo else np.zeros((0, 9)), prior)
            non[i] = look(bt, [row.non_striker_id], mid)[0]
    lag_r, lag_w = _lags(s, deliveries)
    return Batch(innings=s.innings.to_numpy(), runs=s.runs_before.to_numpy(), wickets=s.wickets_before.to_numpy(),
                 legal=s.legal_before.to_numpy(), max_balls=s.max_balls.to_numpy(),
                 target=s.target.to_numpy(dtype=float), bsw=s.balls_since_wicket.to_numpy(), lag_r=lag_r,
                 lag_w=lag_w, striker=s[BAT].to_numpy(dtype=float), non=non, bowler=s[BOWL].to_numpy(dtype=float),
                 queue=queue, team_bowl=team_bowl, ctx=s[CONTEXT].to_numpy(dtype=float), open2=open2,
                 queue2=queue2, team_bowl2=team_bowl2,
                 max_balls2=np.full(S, T20_BALLS) if first else None)


def _lags(s: pd.DataFrame, deliveries: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The (runs, wicket) of the six legal balls before each state, oldest first; zeros before the innings."""
    ids = set(s.match_id)
    d = deliveries[deliveries.match_id.isin(ids) & deliveries.legal.astype(bool)]
    d = d.sort_values(["match_id", "innings", "over", "repeat", "ball"])
    d = d.assign(j=d.groupby(["match_id", "innings"]).cumcount())
    look = d.set_index(["match_id", "innings", "j"])[["runs_total", "team_wickets"]]
    R = np.zeros((len(s), 6))
    Wk = np.zeros((len(s), 6))
    for k in range(6):  # column 5 is the latest legal ball
        j = s.legal_before.to_numpy() - (6 - k)
        idx = pd.MultiIndex.from_arrays([s.match_id.to_numpy(), s.innings.to_numpy(), j])
        v = look.reindex(idx).to_numpy(dtype=float)
        R[:, k] = np.nan_to_num(v[:, 0])
        Wk[:, k] = (np.nan_to_num(v[:, 1]) > 0)
    return R, Wk


# ---------------------------------------------------------------------------------------------------------------
# the roll-out
# ---------------------------------------------------------------------------------------------------------------

def _roll(booster, innings: int, st: np.ndarray, P: dict, queue: np.ndarray, team_bowl: np.ndarray,
          ctx: np.ndarray, ex_v: np.ndarray, ex_p: np.ndarray, rng) -> np.ndarray:
    """Step the per-path arrays ``P`` to the end of the innings (in place). Returns each path's result for a chase
    (1 won, 0 lost, 0.5 tied) or NaN in a first innings. ``st`` maps a path to its state's row of ``queue``,
    ``team_bowl`` and ``ctx``."""
    N = len(st)
    runs, wk, lb, maxb, target = P["runs"], P["wk"], P["lb"], P["maxb"], P["target"]
    bsw, lag_r, lag_w, strk, non, bowl, ptr = P["bsw"], P["lag_r"], P["lag_w"], P["strk"], P["non"], P["bowl"], P["ptr"]
    chase = innings == 2
    result = np.full(N, np.nan)
    alive = lb < maxb
    if chase:
        result[runs >= target] = 1.0
        alive &= runs < target
    alive &= wk < 10
    X = np.zeros((N, len(FEATURES)), dtype=np.float32)
    ecum = np.cumsum(ex_p)
    while alive.any():
        a = np.flatnonzero(alive)
        x = X[: len(a)]
        bl = maxb[a] - lb[a]
        x[:] = np.nan
        x[:, COL["innings"]] = innings
        x[:, COL["legal_before"]] = lb[a]
        x[:, COL["balls_left"]] = bl
        x[:, COL["wickets_before"]] = wk[a]
        x[:, COL["runs_before"]] = runs[a]
        x[:, COL["crr"]] = np.where(lb[a] > 0, 6.0 * runs[a] / np.maximum(lb[a], 1), np.nan)
        if chase:
            need = target[a] - runs[a]
            x[:, COL["runs_needed"]] = need
            x[:, COL["rrr"]] = np.clip(6.0 * need / np.maximum(bl, 1), 0, 36)
        x[:, COL["phase_code"]] = np.where(lb[a] < 36, 0, np.where(lb[a] < 90, 1, 2))
        x[:, COL["last6_runs"]] = lag_r[a].sum(1)
        x[:, COL["last6_wkts"]] = lag_w[a].sum(1)
        x[:, COL["balls_since_wicket"]] = bsw[a]
        x[:, COL[BAT[0]]: COL[BAT[-1]] + 1] = strk[a]
        x[:, COL[BOWL[0]]: COL[BOWL[-1]] + 1] = bowl[a]
        x[:, COL[CONTEXT[0]]: COL[CONTEXT[-1]] + 1] = ctx[st[a]]
        prob = predict(booster, x)
        u = rng.random(len(a))
        cls = np.minimum((prob.cumsum(1) < u[:, None]).sum(1), EXTRA)
        legal = cls != EXTRA
        r = RUNS[cls].astype(np.int32)
        ex = ~legal
        if ex.any():
            r[ex] = ex_v[np.minimum(np.searchsorted(ecum, rng.random(ex.sum())), len(ex_v) - 1)]
        out = cls == W
        runs[a] += r
        wk[a] += out
        lb[a] += legal
        la = a[legal]
        lag_r[la] = np.c_[lag_r[la][:, 1:], r[legal]]
        lag_w[la] = np.c_[lag_w[la][:, 1:], out[legal]]
        bsw[a] = np.where(out, 0, bsw[a] + legal)
        oa = a[out]
        if len(oa):
            q = np.minimum(ptr[oa], queue.shape[1] - 1)
            strk[oa] = queue[st[oa], q]
            ptr[oa] += 1
        odd = legal & (r % 2 == 1) & ~out
        endo = legal & (lb[a] % 6 == 0)
        swap = a[odd ^ endo]
        strk[swap], non[swap] = non[swap], strk[swap].copy()
        eo = a[endo]
        bowl[eo] = team_bowl[st[eo]]
        for k, arr in P.get("marks", {}).items():     # runs when a path first reaches k legal balls
            hit = a[(lb[a] >= k) & np.isnan(arr[a])]
            arr[hit] = runs[hit]
        over = (wk[a] >= 10) | (lb[a] >= maxb[a])
        if chase:
            won = runs[a] >= target[a]
            lost = ~won & over
            result[a[won]] = 1.0
            result[a[lost]] = np.where(runs[a[lost]] == target[a[lost]] - 1, 0.5, 0.0)
            alive[a[won | lost]] = False
        else:
            alive[a[over]] = False
    for arr in P.get("marks", {}).values():        # an innings over before the mark: its final score
        gone = np.isnan(arr)
        arr[gone] = runs[gone]
    if chase:   # a path that could not start (no balls or wickets left) and has not won
        undecided = np.isnan(result)
        result[undecided] = np.where(runs[undecided] == target[undecided] - 1, 0.5, 0.0)
    return result


def smooth(wins: np.ndarray, paths: int) -> np.ndarray:
    """The posterior mean under a Jeffreys prior, (wins + 1/2) / (paths + 1): never exactly 0 or 1. When every path
    agrees the raw share says certain, and one such call that is wrong costs log loss without limit (seen in the
    2025 evaluation: 10 rows took the season's log loss from 0.417 to 0.445)."""
    return (np.asarray(wins, dtype=float) + 0.5) / (paths + 1.0)


def _paths(B: Batch, paths: int) -> tuple[np.ndarray, dict]:
    st = np.repeat(np.arange(len(B)), paths)

    def rep(x, dtype=None):
        return np.repeat(np.asarray(x), paths, axis=0).astype(dtype) if dtype else np.repeat(np.asarray(x), paths,
                                                                                               axis=0).copy()
    P = {"runs": rep(B.runs, np.int32), "wk": rep(B.wickets, np.int32), "lb": rep(B.legal, np.int32),
         "maxb": rep(B.max_balls, np.int32), "target": rep(np.nan_to_num(B.target, nan=0), np.int32),
         "bsw": rep(B.bsw, np.int32), "lag_r": rep(B.lag_r, float), "lag_w": rep(B.lag_w, float),
         "strk": rep(B.striker, float), "non": rep(B.non, float), "bowl": rep(B.bowler, float),
         "ptr": np.zeros(len(st), dtype=np.int32)}
    return st, P


def win_probability(booster, B: Batch, paths: int, ex_v: np.ndarray, ex_p: np.ndarray, seed: int = 0) -> np.ndarray:
    """P(chasing side wins) for every (second-innings) state: the mean over ``paths`` roll-outs."""
    rng = np.random.default_rng(seed)
    st, P = _paths(B, paths)
    res = _roll(booster, 2, st, P, B.queue, B.team_bowl, B.ctx, ex_v, ex_p, rng)
    return smooth(res.reshape(len(B), paths).sum(1), paths)


def first_innings_runs(booster, B: Batch, paths: int, ex_v: np.ndarray, ex_p: np.ndarray, marks=(36,),
                       seed: int = 0) -> dict:
    """For first-innings states: each path's runs at each legal-ball mark (an innings ended before it: its final
    score) and at the end, as {mark: (S, paths)} with key "total" for the end."""
    rng = np.random.default_rng(seed)
    st, P = _paths(B, paths)
    P["marks"] = {k: np.full(len(st), np.nan) for k in marks}
    _roll(booster, 1, st, P, B.queue, B.team_bowl, B.ctx, ex_v, ex_p, rng)
    out = {k: v.reshape(len(B), paths) for k, v in P["marks"].items()}
    out["total"] = P["runs"].reshape(len(B), paths)
    return out


def match_probability(booster, B: Batch, paths: int, ex_v: np.ndarray, ex_p: np.ndarray,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """For every first-innings state: P(the side batting first wins) and each path's first-innings total (S, paths)."""
    rng = np.random.default_rng(seed)
    st, P = _paths(B, paths)
    _roll(booster, 1, st, P, B.queue, B.team_bowl, B.ctx, ex_v, ex_p, rng)
    total1 = P["runs"].copy()
    N = len(st)
    o = B.open2[st]
    P2 = {"runs": np.zeros(N, np.int32), "wk": np.zeros(N, np.int32), "lb": np.zeros(N, np.int32),
          "maxb": B.max_balls2[st].astype(np.int32), "target": total1 + 1, "bsw": np.zeros(N, np.int32),
          "lag_r": np.zeros((N, 6)), "lag_w": np.zeros((N, 6)), "strk": o[:, 0].copy(), "non": o[:, 1].copy(),
          "bowl": B.team_bowl2[st].copy(), "ptr": np.zeros(N, np.int32)}
    chase = _roll(booster, 2, st, P2, B.queue2, B.team_bowl2, B.ctx, ex_v, ex_p, rng)
    return smooth((1.0 - chase).reshape(len(B), paths).sum(1), paths), total1.reshape(len(B), paths)
