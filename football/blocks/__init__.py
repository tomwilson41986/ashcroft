"""Drop-in feature blocks for the football lab (docs/FOOTBALL_FAST_LOOP.md).

A block is one module here with ``FEATURES`` (the columns it adds, each with the block's prefix) and
``build(m) -> DataFrame`` (``match_id`` + ``FEATURES``), computed from the lab's cached match matrix in seconds.
Only information known at the match's early price may enter: opening prices, the fixture list, and results or
closes of matches before the match's collection day (``football.blocks._util``). ``tests/test_football_lab.py``
checks every block for that automatically.
"""

from __future__ import annotations

import importlib
import pkgutil


def available() -> dict:
    """{block name: module} for every block in this package."""
    out = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{info.name}")
        if hasattr(mod, "FEATURES") and hasattr(mod, "build"):
            out[info.name] = mod
    return out
