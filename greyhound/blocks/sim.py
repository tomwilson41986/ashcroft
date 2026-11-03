"""A race simulator (reports/greyhound_roadmap.md F2, the horse model's ``model/abm`` idea for greyhounds): each race
run many times from its dogs' pre-race form, in lengths.

1. The run to the first bend: each dog's early pace (``esr_mean3``, the first sectional against the track and
   distance's standard, faster is higher; ``esr_max3`` blended in) plus a draw of noise.
2. Crowding: a dog that reaches the bend level with a dog drawn next to it (within one trap, within ``CLOSE`` lengths
   of it) and behind it is checked, losing up to ``CHECK`` lengths (more the closer they arrive), as in
   ``greyhound/blocks/first_bend.py``'s crowding.
3. The run-in: the finishing speed (``fsr_mean3``; where missing, the overall rating ``gsr_mean6`` less the early pace)
   plus a draw of noise scaled by the dog's own variability (``gsr_std6``).

A dog with no history (a missing input) takes the race's median with wider noise. The outputs, over ``DRAWS``
simulations a race (seeded, so the same race always gives the same numbers): the chance of winning, of leading at
the bend, of finishing in the first two, the mean lengths lost to checks, and the chance of winning against the
field's best. Pre-race values of the row's own race only: no result is read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREFIX = "sim_"
DRAWS = 300
EARLY_SD = 1.0          # lengths: the run to the bend from one day to the next
RUN_SD = 1.5            # lengths: the run-in, before the dog's own variability
CLOSE = 1.0             # lengths: two dogs this close at the bend, drawn side by side, can collide
CHECK = 2.0             # lengths lost by the dog behind in a collision at level
UNKNOWN_SD = 1.5        # the noise multiplier for a dog with a missing input
CHUNK = 4000            # races simulated at a time


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce").astype(float) if name in df.columns else \
        pd.Series(np.nan, index=df.index)


def _inputs(df: pd.DataFrame) -> pd.DataFrame:
    race = df.race_id
    esr = _col(df, "esr_mean3")
    esr_max = _col(df, "esr_max3")
    early = esr.where(esr_max.isna(), 0.75 * esr + 0.25 * esr_max.where(esr_max.notna(), esr))
    run = _col(df, "fsr_mean3")
    run = run.fillna(_col(df, "gsr_mean6") - esr)
    sd = _col(df, "gsr_std6").clip(lower=0.5, upper=8.0)
    unknown = early.isna() | run.isna()
    out = pd.DataFrame({"race": race.to_numpy(), "trap": _col(df, "trap").to_numpy(), "early": early.to_numpy(),
                        "run": run.to_numpy(), "sd": sd.to_numpy(), "unknown": unknown.to_numpy()}, index=df.index)
    for c in ("early", "run", "sd"):
        med = out.groupby("race")[c].transform("median")
        out[c] = out[c].fillna(med).fillna(0.0 if c != "sd" else 2.0)
    return out


def _simulate(early, run, sd, unk, trap, rng) -> dict[str, np.ndarray]:
    """``early``, ``run``, ``sd``, ``unk``, ``trap``: arrays [races, k] for races of ``k`` runners."""
    n, k = early.shape
    mult = np.where(unk, UNKNOWN_SD, 1.0)[:, None, :]
    bend = early[:, None, :] + rng.normal(0.0, EARLY_SD, (n, DRAWS, k)) * mult
    # the checks at the bend: a dog with a neighbour (within one trap) ahead of it and within CLOSE lengths
    lost = np.zeros((n, DRAWS, k))
    for j in range(k):
        for i in range(k):
            if i == j:
                continue
            side = np.abs(trap[:, i] - trap[:, j]) <= 1.0 + 1e-9          # [n]
            if not side.any():
                continue
            gap = bend[:, :, i] - bend[:, :, j]                           # > 0: i ahead of j
            hit = (gap > 0) & (gap < CLOSE) & side[:, None]
            lost[:, :, j] += np.where(hit, CHECK * (1.0 - gap / CLOSE), 0.0)
    noise = rng.normal(0.0, 1.0, (n, DRAWS, k)) * np.sqrt(RUN_SD ** 2 + (0.5 * sd[:, None, :]) ** 2) * mult
    final = bend - lost + run[:, None, :] + noise
    order = np.argsort(-final, axis=2)
    first, second = order[:, :, 0], order[:, :, 1]
    eye = np.arange(k)
    win = (first[:, :, None] == eye).mean(axis=1)
    top2 = ((first[:, :, None] == eye) | (second[:, :, None] == eye)).mean(axis=1)
    lead = (np.argmax(bend, axis=2)[:, :, None] == eye).mean(axis=1)
    return {"win_p": win, "lead_p": lead, "top2_p": top2, "lost": lost.mean(axis=1)}


def build(df: pd.DataFrame) -> list[str]:
    x = _inputs(df)
    names = ("win_p", "lead_p", "top2_p", "lost", "win_vs_best", "win_logit")
    out = {n: np.full(len(df), np.nan) for n in names}
    pos = pd.Series(np.arange(len(df)), index=df.index)
    rng = np.random.default_rng(0)
    order = x.sort_values(["race", "trap"], kind="stable")
    size = order.groupby("race").race.transform("size")
    for k, part in order.groupby(size, sort=True):
        k = int(k)
        if k < 2:
            continue
        races = part.race.to_numpy()
        n = len(races) // k
        idx = pos.reindex(part.index).to_numpy().reshape(n, k)
        arr = {c: part[c].to_numpy(float).reshape(n, k) for c in ("early", "run", "sd", "trap")}
        unk = part["unknown"].to_numpy(bool).reshape(n, k)
        for s in range(0, n, CHUNK):
            sl = slice(s, s + CHUNK)
            r = _simulate(arr["early"][sl], arr["run"][sl], arr["sd"][sl], unk[sl], arr["trap"][sl], rng)
            for c in ("win_p", "lead_p", "top2_p", "lost"):
                out[c][idx[sl].ravel()] = r[c].ravel()
    win = pd.Series(out["win_p"], index=df.index)
    best = win.groupby(df.race_id).transform("max")
    out["win_vs_best"] = (win / best.replace(0, np.nan)).to_numpy()
    p = win.clip(0.5 / DRAWS, 1 - 0.5 / DRAWS)
    out["win_logit"] = np.log(p / (1 - p)).to_numpy()
    cols = []
    for n in names:
        df[f"{PREFIX}{n}"] = out[n].astype(np.float32)
        cols.append(f"{PREFIX}{n}")
    return cols
