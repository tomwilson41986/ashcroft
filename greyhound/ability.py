"""Ability with its uncertainty (reports/greyhound_roadmap.md F6): a dog with three runs is not a dog with forty.

- **The Kalman rating with its variance.** ``greyhound.lib.kalman`` already reads each dog's speed figure (GSR) as a
  state with a posterior sd (``lb_kr``, ``lb_kr_sd``). Here its variance and what it implies against today's field: a
  pessimistic and an optimistic reading (mean -+ sd), the rating's z in the field scaled by its own uncertainty, and
  the chance the dog is the best in the race when every rating is read with its uncertainty (normal draws).
- **A dog x track hierarchical ability.** The dog's mean GSR at today's track, shrunk toward its mean everywhere by its
  runs there, ``(n_t * m_t + K_TRACK * m_dog) / (n_t + K_TRACK)``; the dog's mean everywhere itself shrunk toward the
  population by ``K_DOG`` runs. With the runs each rests on and a standard error from the dog's own spread of figures
  (the population's where it has fewer than 3), all from earlier days only.

    ab_kr_var, ab_kr_lo, ab_kr_hi, ab_kr_zu, ab_p_best, ab_dog, ab_dog_n, ab_dt, ab_dt_n, ab_dt_minus_dog, ab_dt_se,
    ab_dt_z, ab_dt_gap
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_count, race_lagged_expanding_mean

PREFIX = "ab_"
K_DOG = 3.0                   # runs of prior toward the population, a dog's overall ability
K_TRACK = 3.0                 # runs of prior toward the dog's overall ability, at a track
SIMS = 200                    # draws for the chance of being the best in the race
SEED = 20261103


def p_best(m: np.ndarray, sd: np.ndarray, race: np.ndarray, sims: int = SIMS, seed: int = SEED,
           chunk: int = 5_000) -> np.ndarray:
    """Each runner's chance of the highest draw in its race, ratings read as normal(m, sd). A runner without a rating
    gets NaN and is left out of its race's draws; a race with one rated runner gives it 1."""
    out = np.full(len(m), np.nan)
    ok = np.isfinite(m) & np.isfinite(sd)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return out
    codes, r = np.unique(race[idx], return_inverse=True)
    order = np.argsort(r, kind="stable")
    r_sorted = r[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r_sorted)) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    width = int(sizes.max())
    pos = np.arange(len(order)) - np.repeat(starts, sizes)
    M = np.full((len(codes), width), -np.inf)
    S = np.zeros((len(codes), width))
    M[r_sorted, pos] = m[idx][order]
    S[r_sorted, pos] = sd[idx][order]
    # the same draws for every race (a runner's are those of its place among the race's rated runners), so a race's
    # numbers are its own inputs' and nothing else's. Until 11 Nov 2026 one stream ran down the whole frame, and the
    # same race had other numbers on the card than in its result row (the audit of 11 Nov 2026, findings 17 and 42)
    Z = np.random.default_rng(seed).standard_normal((width, sims))
    P = np.zeros((len(codes), width))
    for a in range(0, len(codes), chunk):
        mm, ss = M[a:a + chunk], S[a:a + chunk]
        draws = mm[:, :, None] + ss[:, :, None] * Z[None, :, :]
        best = draws.argmax(axis=1)                            # (races, sims)
        for j in range(width):
            P[a:a + chunk, j] = (best == j).mean(axis=1)
    out[idx[order]] = P[r_sorted, pos]
    return out


def calculate(df: pd.DataFrame, dog_days=None) -> tuple[pd.DataFrame, list[str]]:
    """The block for the engine's frame (after the base features; reads ``lb_kr``/``lb_kr_sd`` when the lib block ran,
    else runs the same filter)."""
    race = df.race_id
    cols = {}
    if {"lb_kr", "lb_kr_sd"} <= set(df.columns):
        km, ks = df.lb_kr.astype(float), df.lb_kr_sd.astype(float)
    else:
        from greyhound.lib import kalman
        from greyhound.lib import debut_prior
        k = kalman(df, debut_prior(df))
        km, ks = k.m, k.sd
    cols[f"{PREFIX}kr_var"] = ks ** 2
    cols[f"{PREFIX}kr_lo"] = km - ks
    cols[f"{PREFIX}kr_hi"] = km + ks
    g = km.groupby(race)
    cols[f"{PREFIX}kr_zu"] = (km - g.transform("mean")) / np.sqrt(ks ** 2 + g.transform("var").fillna(0.0)).replace(0, np.nan)
    cols[f"{PREFIX}p_best"] = p_best(km.to_numpy(float), ks.to_numpy(float), race.to_numpy())
    # the hierarchy: population <- dog <- dog at today's track, earlier days only
    gsr = pd.to_numeric(df.gsr, errors="coerce")
    d = df.assign(_g=gsr, _g2=gsr ** 2, _pop=0)
    pop = race_lagged_expanding_mean(d, "_pop", "_g")
    pop_sq = race_lagged_expanding_mean(d, "_pop", "_g2")
    pop_sd = np.sqrt((pop_sq - pop ** 2).clip(lower=0))
    m_dog = race_lagged_expanding_mean(d, "dog_id", "_g")
    sq_dog = race_lagged_expanding_mean(d, "dog_id", "_g2")
    n_dog = race_lagged_expanding_count(d, "dog_id", "_g")
    dog = (m_dog.fillna(pop) * n_dog + pop * K_DOG) / (n_dog + K_DOG)
    m_dt = race_lagged_expanding_mean(d, ["dog_id", "track"], "_g")
    n_dt = race_lagged_expanding_count(d, ["dog_id", "track"], "_g")
    dt = (m_dt.fillna(dog) * n_dt + dog * K_TRACK) / (n_dt + K_TRACK)
    sd_dog = np.sqrt((sq_dog - m_dog ** 2).clip(lower=0)).where(n_dog >= 3, pop_sd)
    cols[f"{PREFIX}dog"] = dog
    cols[f"{PREFIX}dog_n"] = n_dog
    cols[f"{PREFIX}dt"] = dt
    cols[f"{PREFIX}dt_n"] = n_dt
    cols[f"{PREFIX}dt_minus_dog"] = dt - dog
    cols[f"{PREFIX}dt_se"] = sd_dog / np.sqrt(n_dt + K_TRACK)
    gd = dt.groupby(race)
    cols[f"{PREFIX}dt_z"] = (dt - gd.transform("mean")) / gd.transform("std").replace(0, np.nan)
    cols[f"{PREFIX}dt_gap"] = dt - gd.transform("max")
    out = pd.DataFrame(cols, index=df.index).astype(np.float32)
    return out, list(out.columns)
