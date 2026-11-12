"""greyhound/blocks/placebo.py: noise fixed by the race and the dog, the same in any frame, and never served."""
from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound import blocks, stack_model
from greyhound.blocks import placebo


def test_the_noise_is_fixed_by_the_race_and_the_dog_and_is_noise():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"race_id": rng.integers(1, 10**6, 60000), "dog_id": rng.integers(-10**7, 10**6, 60000)})
    a = df.copy()
    cols = blocks.attach(a, "placebo")
    b = df.sample(frac=0.5, random_state=1).copy()                       # another frame, another order
    blocks.attach(b, "placebo")
    assert len(cols) == placebo.N and np.allclose(a.loc[b.index, cols].to_numpy(), b[cols].to_numpy())
    x = a[cols].to_numpy(float)
    assert np.all(np.abs(x.mean(axis=0)) < 0.02) and np.all(np.abs(x.std(axis=0) - 1) < 0.02)
    c = np.corrcoef(x, rowvar=False)
    assert np.abs(c - np.eye(placebo.N)).max() < 0.02                     # the columns are unrelated to each other


def test_the_placebo_is_never_served():
    assert "placebo" not in stack_model.MATRIX_BLOCKS and "placebo" in blocks.NAMES
