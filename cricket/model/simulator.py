"""Vectorised Monte Carlo roll-out of a chase (build spec 6.4, v1): win probability from the next-ball model.

Every path of every state in a batch is stepped together: one model call per step for all live paths. A path ends
when the target is reached (win), or on the tenth wicket or the last legal ball (loss; a tie scores one half).

Who faces and who bowls, inside a path:
- the striker, non-striker and bowler of the state are the real ones; the current bowler finishes the over;
- after a wicket the new batter is the next listed player of the batting XI who has not yet batted (Cricsheet's
  list order; not guaranteed to be the batting order: stated in the report), on strike;
- after the current over, the bowler is the bowling XI's rating averaged by each player's bowling experience;
- strike changes on odd runs from a legal ball and at the end of each over.
Simplifications from cricket/model/outcomes.py apply (no runs on wicket balls, no wickets on wides or no-balls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cricket.model.features import BAT, BOWL, CONTEXT, FEATURES
from cricket.model.next_ball import predict
from cricket.model.outcomes import EXTRA, RUNS, W

NR = len(BAT)            # 8 probabilities and the experience n
COL = {c: i for i, c in enumerate(FEATURES)}


def _rating_table(r: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [f"p{k}" for k in range(8)] + ["n_eff"]
    return r.set_index(["player_id", "match_id"])[cols].rename(
        columns={**{f"p{k}": f"{prefix}_p{k}" for k in range(8)}, "n_eff": f"{prefix}_n"})


def prepare(states: pd.DataFrame, deliveries: pd.DataFrame, players: pd.DataFrame, bat: pd.DataFrame,
            bowl: pd.DataFrame, prior: np.ndarray, queue_len: int = 10) -> dict:
    """Arrays the roll-out needs for each chase state (rows of the feature frame, second innings)."""
    s = states.reset_index(drop=True)
    S = len(s)
    bt = _rating_table(bat, "bat")
    bw = _rating_table(bowl, "bowl")
    prior_row = np.r_[prior, 0.0]

    def look(tab, pids, mids):
        idx = pd.MultiIndex.from_arrays([pids, mids])
        v = tab.reindex(idx).to_numpy(dtype=float)
        v[np.isnan(v).any(1)] = prior_row
        return v

    striker = s[BAT].to_numpy(dtype=float)
    non = look(bt, s.non_striker_id.to_numpy(), s.match_id.to_numpy())
    bowler = s[BOWL].to_numpy(dtype=float)

    # who has batted before each state (striker and non-striker included), the rest of the XI queued in list order
    queue = np.tile(prior_row, (S, queue_len, 1))
    team_bowl = np.tile(prior_row, (S, 1))
    pl = players.sort_values("list_order")
    for (mid, inn), g in s.groupby(["match_id", "innings"], sort=False):
        bat_team, bowl_team = g.team.iloc[0], g.bowl_team.iloc[0]
        dl = deliveries[(deliveries.match_id == mid) & (deliveries.innings == inn)]
        dl = dl.sort_values(["over", "repeat", "ball"])
        xi = pl[(pl.match_id == mid) & (pl.team == bat_team)].player_id.tolist()
        bxi = pl[(pl.match_id == mid) & (pl.team == bowl_team)].player_id.tolist()
        bv = look(bw, np.array(bxi), np.full(len(bxi), mid))
        wts = bv[:, -1]
        tb = (bv[:, :8] * wts[:, None]).sum(0) / wts.sum() if wts.sum() > 0 else prior
        team_bowl[g.index] = np.r_[tb, wts.mean() if len(wts) else 0.0]
        seen: set = set()
        names = list(zip(dl.batter_id, dl.non_striker_id))
        pos = {k: i for i, k in enumerate(zip(dl.over, dl.repeat, dl.ball))}
        for i, row in zip(g.index, g.itertuples()):
            j = pos[(row.over, row.repeat, row.ball)]
            for a, b in names[: j + 1]:
                seen.add(a)
                seen.add(b)
            todo = [p for p in xi if p not in seen][:queue_len]
            if todo:
                queue[i, : len(todo)] = look(bt, np.array(todo), np.full(len(todo), mid))
            seen.clear()
    lag_r, lag_w = _lags(s, deliveries)
    return {"s": s, "striker": striker, "non": non, "bowler": bowler, "queue": queue, "team_bowl": team_bowl,
            "lag_r": lag_r, "lag_w": lag_w, "context": s[CONTEXT].to_numpy(dtype=float)}


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


def win_probability(booster, prep: dict, paths: int, extras_vals: np.ndarray, extras_p: np.ndarray,
                    seed: int = 0) -> np.ndarray:
    """P(chasing side wins) for every state in ``prep``: the mean over ``paths`` roll-outs."""
    rng = np.random.default_rng(seed)
    s = prep["s"]
    S = len(s)
    N = S * paths
    st = np.repeat(np.arange(S), paths)
    runs = np.repeat(s.runs_before.to_numpy(), paths).astype(np.int32)
    wk = np.repeat(s.wickets_before.to_numpy(), paths).astype(np.int32)
    lb = np.repeat(s.legal_before.to_numpy(), paths).astype(np.int32)
    maxb = np.repeat(s.max_balls.to_numpy(), paths).astype(np.int32)
    target = np.repeat(s.target.to_numpy(), paths).astype(np.int32)
    bsw = np.repeat(s.balls_since_wicket.to_numpy(), paths).astype(np.int32)
    lag_r = np.repeat(prep["lag_r"], paths, axis=0)
    lag_w = np.repeat(prep["lag_w"], paths, axis=0)
    strk = np.repeat(prep["striker"], paths, axis=0)
    non = np.repeat(prep["non"], paths, axis=0)
    bowl = np.repeat(prep["bowler"], paths, axis=0)
    team_bowl = prep["team_bowl"]
    queue = prep["queue"]
    ptr = np.zeros(N, dtype=np.int32)
    ctx = prep["context"]
    result = np.full(N, np.nan)
    alive = np.ones(N, dtype=bool)
    # a state already decided (cannot happen before a delivery, but kept safe)
    X = np.zeros((N, len(FEATURES)), dtype=np.float32)
    ecum = np.cumsum(extras_p)
    while alive.any():
        a = np.flatnonzero(alive)
        x = X[: len(a)]
        bl = maxb[a] - lb[a]
        need = target[a] - runs[a]
        x[:, COL["innings"]] = 2
        x[:, COL["legal_before"]] = lb[a]
        x[:, COL["balls_left"]] = bl
        x[:, COL["wickets_before"]] = wk[a]
        x[:, COL["runs_before"]] = runs[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            x[:, COL["crr"]] = np.where(lb[a] > 0, 6.0 * runs[a] / np.maximum(lb[a], 1), np.nan)
            x[:, COL["rrr"]] = np.clip(np.where(bl > 0, 6.0 * need / np.maximum(bl, 1), np.nan), 0, 36)
        x[:, COL["runs_needed"]] = need
        x[:, COL["phase_code"]] = np.where(lb[a] < 36, 0, np.where(lb[a] < 90, 1, 2))
        x[:, COL["last6_runs"]] = lag_r[a].sum(1)
        x[:, COL["last6_wkts"]] = lag_w[a].sum(1)
        x[:, COL["balls_since_wicket"]] = bsw[a]
        x[:, COL[BAT[0]]: COL[BAT[-1]] + 1] = strk[a]
        x[:, COL[BOWL[0]]: COL[BOWL[-1]] + 1] = bowl[a]
        x[:, COL[CONTEXT[0]]: COL[CONTEXT[-1]] + 1] = ctx[st[a]]
        P = predict(booster, x)
        u = rng.random(len(a))
        cls = np.minimum((P.cumsum(1) < u[:, None]).sum(1), EXTRA)
        legal = cls != EXTRA
        r = RUNS[cls].astype(np.int32)
        ex = ~legal
        if ex.any():
            r[ex] = extras_vals[np.minimum(np.searchsorted(ecum, rng.random(ex.sum())), len(extras_vals) - 1)]
        out = cls == W
        runs[a] += r
        wk[a] += out
        lb[a] += legal
        # last six legal balls and balls since the last wicket
        la = a[legal]
        lag_r[la] = np.c_[lag_r[la][:, 1:], r[legal]]
        lag_w[la] = np.c_[lag_w[la][:, 1:], out[legal]]
        bsw[a] = np.where(out, 0, bsw[a] + legal)
        # a wicket: the next listed batter comes in on strike
        oa = a[out]
        if len(oa):
            q = np.minimum(ptr[oa], queue.shape[1] - 1)
            strk[oa] = queue[st[oa], q]
            ptr[oa] += 1
        # strike: odd runs off a legal ball, then the end of the over; a new over brings the team's bowling
        odd = legal & (r % 2 == 1) & ~out
        endo = legal & (lb[a] % 6 == 0)
        swap = a[odd ^ endo]
        strk[swap], non[swap] = non[swap], strk[swap].copy()
        eo = a[endo]
        bowl[eo] = team_bowl[st[eo]]
        # ends
        won = runs[a] >= target[a]
        lost = ~won & ((wk[a] >= 10) | (lb[a] >= maxb[a]))
        tie = lost & (runs[a] == target[a] - 1)
        result[a[won]] = 1.0
        result[a[lost]] = 0.0
        result[a[tie]] = 0.5
        alive[a[won | lost]] = False
    return result.reshape(S, paths).mean(1)
