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
697,903 of 698,625 on 24 Sep); the engine sees every run.

Serving a block is the same computation, not a port of it. A model that reads a
block's features names the block by them (`used_by`); the live path switches on
the engine blocks it declares it reads (`ENGINE`, e.g. ("form_windows",)) and
builds it with `attach_as_trained`: on the rows the matrix holds, the runs with
a usable price, plus the card's, exactly as training built it on the matrix.
Optional attributes:

    READS : list[str]
        Every column build() reads. The live path then copies only those, not
        the whole history (tests/test_feature_blocks.py checks that a build on
        READS alone equals the build on the whole frame).
    ENGINE : tuple[str, ...]
        CustomMetricsEngine flags whose columns build() reads.
    AFTER : tuple[str, ...]
        Drop-in blocks whose features build() reads; they are built first.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from pathlib import Path

import numpy as np
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
    """Build each named block on `df`, in order, after any drop-in block it reads
    (AFTER) that the frame does not carry yet. Returns (frame, the named blocks'
    features): a block built only as another's input is not a feature."""
    named = list(dict.fromkeys(block_names))
    cols: list[str] = []
    for name in build_order(named):
        mod = load(name)
        if name not in named and all(c in df.columns for c in mod.FEATURES):
            continue
        out = mod.build(df)
        missing = [c for c in mod.FEATURES if c not in out.columns]
        if missing:
            raise ValueError(f"block {name}: build() did not add {missing[:5]}"
                             f"{' ...' if len(missing) > 5 else ''}")
        if len(out) != len(df):
            raise ValueError(f"block {name}: build() changed the row count ({len(df)} -> {len(out)})")
        df = out
        if name in named:
            cols += list(mod.FEATURES)
    return df, cols


def build_order(block_names) -> list[str]:
    """The named blocks and every drop-in block they read (AFTER), each after
    the blocks it reads, otherwise in the order given."""
    want: list[str] = []

    def add(n: str, path: tuple = ()) -> None:
        if n in path:
            raise ValueError(f"drop-in blocks read each other in a cycle: {' -> '.join(path + (n,))}")
        for dep in getattr(load(n), "AFTER", ()):
            add(dep, path + (n,))
        if n not in want:
            want.append(n)

    for n in block_names:
        add(n)
    return want


def used_by(feature_cols) -> list[str]:
    """The drop-in blocks a model reads, by its feature names, in build order."""
    cols = set(feature_cols)
    return build_order([n for n in names() if cols & set(features(n))])


def engine_flags(block_names) -> dict[str, bool]:
    """The CustomMetricsEngine flags the named blocks read columns of (ENGINE)."""
    return {flag: True for n in build_order(block_names) for flag in getattr(load(n), "ENGINE", ())}


def matrix_rows(df: pd.DataFrame) -> np.ndarray:
    """The rows the cached matrix keeps (evaluate_oos.build_feature_frame): runs
    with a usable price. A block is built on these in research and training."""
    if "bfsp" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return (pd.to_numeric(df["bfsp"], errors="coerce") > 1.0).to_numpy()


def attach_as_trained(df: pd.DataFrame, block_names, card=None) -> tuple[pd.DataFrame, list[str]]:
    """Build blocks on a live frame exactly as training built them on the matrix.

    The live frame holds every run of the history and the card; the matrix a
    model was trained on holds the runs with a usable price. A block's windows,
    race means and cell averages would differ between the two, so each block is
    built on the matrix's rows plus the card's (`card`, a mask; the card has no
    price yet) and every other row gets NaN: no model is asked to price those.
    With READS declared only those columns are copied. A drop-in block another
    reads (AFTER) is built first. Returns (frame, the named blocks' features)."""
    keep = matrix_rows(df)
    if card is not None:
        keep = keep | np.asarray(card, dtype=bool)
    idx = np.flatnonzero(keep)
    named = list(dict.fromkeys(block_names))
    added: list[str] = []
    for name in build_order(named):
        mod = load(name)
        reads = getattr(mod, "READS", None)
        if reads is not None:
            lacking = [c for c in dict.fromkeys(reads) if c not in df.columns]
            if lacking:
                raise ValueError(f"block {name} reads {lacking[:5]}, which this frame lacks: its features "
                                 f"would be built on nothing")
        use = list(dict.fromkeys(reads)) if reads is not None else list(df.columns)
        sub = df[use].iloc[idx]
        out = mod.build(sub)
        missing = [c for c in mod.FEATURES if c not in out.columns]
        if missing or len(out) != len(sub):
            raise ValueError(f"block {name}: build() broke the contract "
                             f"({'missing ' + str(missing[:3]) if missing else 'row count changed'})")
        with warnings.catch_warnings():      # in place: the live frame is the whole history
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            for c in mod.FEATURES:
                arr = np.full(len(df), np.nan)
                arr[idx] = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)
                df[c] = arr
        if name in named:
            added += list(mod.FEATURES)
    return df, added
