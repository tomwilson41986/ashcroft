"""A control: columns of noise that carry no information, to measure what adding a block moves by chance.

The quick screening recipe samples half the features for every tree, so adding
any block reshuffles which features each tree sees, much as changing the seed
does (iteration 39: the seed alone moves the paired error by up to 0.0016).
A block's measured gain is then its information plus that reshuffle. This
block is the reshuffle alone: a variant with it and nothing else shows the
change a block of no value produces on the recipe being used, which is the
bar a real block has to clear.

    pb_noise_<k>   a standard normal draw fixed by the runner (race and horse),
                   k = 1..10: the same row always gets the same values, in any
                   order, on the live path as in training

Never served.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from model.race_shape import race_key

N_COLS = 10
FEATURES = [f"pb_noise_{k}" for k in range(1, N_COLS + 1)]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name"]
READS_RESULTS = False


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def build(df: pd.DataFrame) -> pd.DataFrame:
    keys = (race_key(df).astype(str) + "|" + df["horse_name"].astype(str)).to_numpy()
    uniq, inv = np.unique(keys, return_inverse=True)
    draws = np.vstack([np.random.default_rng(_seed(k)).standard_normal(N_COLS) for k in uniq])
    new = pd.DataFrame(draws[inv], columns=FEATURES, index=df.index)
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
