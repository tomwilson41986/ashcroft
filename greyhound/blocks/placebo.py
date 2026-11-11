"""A placebo block: ten columns of noise, fixed by the race and the dog, never served (the horse fast loop's
``model/blocks/placebo.py``; docs/FAST_LOOP.md: the seed floor and the placebo).

A new block's paired gain is read against two floors on the same recipe: what ten columns of pure noise "gain" (the
trees' feature sampling moves the fit a little whatever is added), and how far the baseline moves from one seed to
the next. A gain that does not clear both is not a gain. The noise is a hash of (race_id, dog_id, column), so a race
gets the same values in any frame and on any day; it carries no information by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREFIX = "pb_"
N = 10


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser: a well-mixed 64-bit hash of each element."""
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def noise(race_id, dog_id, col: int) -> np.ndarray:
    """A standard normal a (race, dog, column), by Box-Muller on two hashes."""
    r = pd.to_numeric(pd.Series(race_id), errors="coerce").fillna(0).to_numpy(np.int64).astype(np.uint64)
    d = pd.to_numeric(pd.Series(dog_id), errors="coerce").fillna(0).to_numpy(np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        base = r * np.uint64(0x9E3779B97F4A7C15) + d * np.uint64(0xC2B2AE3D27D4EB4F) + np.uint64(col + 1) * np.uint64(
            0x165667B19E3779F9)
    u1 = (_mix(base) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u2 = (_mix(base ^ np.uint64(0xD6E8FEB86659FD93)) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.sqrt(-2.0 * np.log(np.clip(u1, 1e-300, 1.0))) * np.cos(2.0 * np.pi * u2)


def build(df: pd.DataFrame) -> list[str]:
    cols = []
    for k in range(N):
        c = f"{PREFIX}{k}"
        df[c] = noise(df.race_id, df.dog_id, k).astype(np.float32)
        cols.append(c)
    return cols
