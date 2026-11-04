"""The eight next-ball outcomes (build spec 3, 6.3): 0, 1, 2, 3, 4, 6, W, extras.

A delivery is ``extra`` when it is a wide or a no-ball (it does not use up a legal ball), else ``W`` when the team
loses a wicket on it, else its total runs: 5 is counted as 4 and 7 or more as 6 (both rare: legal 5s are
overthrows and penalties). Byes and leg byes count as runs: the outcome is the team's, not the batter's.
Simplifications the simulator inherits, stated here so a report can quote them: runs scored on a wicket ball are
dropped (almost always zero), and wickets on wides and no-balls (run outs, stumpings; 0.3% of them in Cricsheet)
are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ["0", "1", "2", "3", "4", "6", "W", "extra"]
K = len(CLASSES)
W, EXTRA = 6, 7
RUNS = np.array([0, 1, 2, 3, 4, 6, 0, 0], dtype=np.int16)   # runs of each legal class (extras sampled apart)
_RUN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5}


def classify(d: pd.DataFrame) -> np.ndarray:
    runs = d["runs_total"].to_numpy().astype(int)
    cls = np.array([_RUN_TO_CLASS.get(r, 5) for r in np.minimum(runs, 7)], dtype=np.int8)
    cls = np.where(d["team_wickets"].to_numpy() > 0, W, cls)
    cls = np.where(~d["legal"].to_numpy(dtype=bool), EXTRA, cls)
    return cls.astype(np.int8)


def extras_runs_distribution(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Empirical runs on a wide or no-ball (values, probabilities), from the rows given (the training rows)."""
    v = d.loc[~d["legal"].astype(bool), "runs_total"].astype(int).value_counts(normalize=True).sort_index()
    return v.index.to_numpy(dtype=np.int16), v.to_numpy(dtype=float)
