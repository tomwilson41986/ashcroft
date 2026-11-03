"""Drop-in feature blocks for the greyhound fast loop (``greyhound.fastloop``): each computed on the cached feature
matrix, no rebuild, the horse model's ``model/blocks`` pattern. A block is a module with ``PREFIX`` and
``build(df) -> list[str]`` that adds its columns to ``df`` in place and returns their names. **Earlier days only**:
a block may read the pre-race features of the row's own race and the results of earlier days, never the row's own
day's results (``tests/test_greyhound_fastloop.py`` rewrites the last day's results and checks nothing moves).

- ``race_rel2``: the strongest ratings and form windows read against the field (z-score, rank, gap to the best).
- ``first_bend``: a model of the run to the first bend from early speed, break and leading history and the traps:
  each dog's chance of leading, its projected order, the crowding around it.
- ``style_draw``: how dogs of this running style (railer, middle, wide) have done from this trap at this track and
  distance on earlier days, against the trap's own record.
- ``sim``: a race simulator (roadmap F2): each race run 300 times from early pace, the checks at the first bend and the
  run-in; the chance of winning, leading at the bend, finishing in the first two, the lengths lost to checks.
- ``fcast``: Betfair's forecast price as a prior (roadmap L2), from the recorded catalogues (6 Oct 2026 on; NaN
  before).
"""

from __future__ import annotations

import importlib

NAMES = ("race_rel2", "first_bend", "style_draw", "sim", "fcast")


def attach(df, name: str) -> list[str]:
    """Add block ``name``'s columns to ``df`` (once) and return their names."""
    if name not in NAMES:
        raise ValueError(f"unknown block {name!r}; known: {NAMES}")
    mod = importlib.import_module(f"greyhound.blocks.{name}")
    have = [c for c in df.columns if c.startswith(mod.PREFIX)]
    return have if have else mod.build(df)
