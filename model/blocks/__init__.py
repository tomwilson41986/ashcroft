"""
Drop-in feature blocks: one file each, evaluated without rebuilding the matrix.

The metrics engine's blocks live inside `CustomMetricsEngine.calculate_all`, and
the feature cache is keyed on every file the engine imports, so a change to any
of them rebuilds the whole matrix (11-17 minutes) before the first fit can
start. A block in this package is imported by nothing the engine reaches: it is
computed on the cached matrix at evaluation time, so adding or editing one costs
the block's own seconds, not the rebuild.

A block is a module here that defines

    FEATURES : list[str]
        The model inputs it adds. Name them with a short prefix of the block's
        own so they cannot collide with the engine's.
    POST_RACE : set[str]   (optional)
        Per-run readings it leaves on the frame that describe the race itself
        (a run's own position, margin, pace). They feed later days' windows and
        are refused as inputs.
    build(df) -> df
        Returns the frame with FEATURES added. It may read the race records and
        the metrics engine's columns, not the context features (the matrix has
        them, the engine's output does not, and a served block runs inside the
        engine). Earlier days only: nothing from a row's own day enters its
        features, not even as rounding.

and whose docstring says what it measures. `tests/test_feature_blocks.py` runs
every block found here through the lag and card tests automatically: a day's
results scrambled or blanked must move none of that day's features, and later
days' features must move (the block reads history at all). Nothing to register.

    python scripts/new_block.py my_idea           # the template, ready to fill
    pytest tests/test_feature_blocks.py -k my_idea
    research/loop.json: {"recipe": "quick", "blocks": "my_idea", ...}  -> push

On the cached matrix a block sees the rows with a usable price (99.9% of runs:
697,903 of 698,625 on 24 Sep); the engine sees every run. The difference is
immaterial to a screen and is why a block that is promoted moves into the engine
and is measured again there before it is served.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent


def names() -> list[str]:
    """Every block in the package, by module name (private modules excluded)."""
    return sorted(m.name for m in pkgutil.iter_modules([str(_HERE)])
                  if not m.name.startswith("_") and not m.ispkg)


def load(name: str):
    """Import a block and check it keeps the contract."""
    if name not in names():
        raise KeyError(f"no block {name!r} in model/blocks ({', '.join(names()) or 'none'})")
    mod = importlib.import_module(f"model.blocks.{name}")
    feats = getattr(mod, "FEATURES", None)
    if not isinstance(feats, (list, tuple)) or not feats:
        raise TypeError(f"model/blocks/{name}.py: FEATURES must be a non-empty list")
    if len(set(feats)) != len(feats):
        raise TypeError(f"model/blocks/{name}.py: FEATURES has duplicates")
    if not callable(getattr(mod, "build", None)):
        raise TypeError(f"model/blocks/{name}.py: no build(df)")
    clash = set(feats) & set(getattr(mod, "POST_RACE", set()))
    if clash:
        raise TypeError(f"model/blocks/{name}.py: {sorted(clash)} are both features and post-race")
    return mod


def features(name: str) -> list[str]:
    return list(load(name).FEATURES)


def post_race(name: str) -> set[str]:
    return set(getattr(load(name), "POST_RACE", set()))


def attach(df: pd.DataFrame, block_names) -> tuple[pd.DataFrame, list[str]]:
    """Build each named block on `df`, in order. Returns (frame, features added)."""
    cols: list[str] = []
    for name in block_names:
        mod = load(name)
        out = mod.build(df)
        missing = [c for c in mod.FEATURES if c not in out.columns]
        if missing:
            raise ValueError(f"block {name}: build() did not add {missing[:5]}"
                             f"{' ...' if len(missing) > 5 else ''}")
        if len(out) != len(df):
            raise ValueError(f"block {name}: build() changed the row count ({len(df)} -> {len(out)})")
        df = out
        cols += list(mod.FEATURES)
    return df, cols
