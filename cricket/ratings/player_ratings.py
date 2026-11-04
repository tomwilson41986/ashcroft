"""Per-player, per-format next-ball ratings as of the start of each match (build spec 6.2, v1).

Each batter's and bowler's eight-outcome distribution (cricket/model/outcomes.py) from their earlier matches in the
format, each match decayed by r = 0.88 per later match, shrunk towards the format's population distribution with a
Dirichlet prior of c = 60 balls (the SFU recipe). One row per (player, match) the player took part in, plus a row for
each listed player of a match who did not bat or bowl, so the simulator can rate a batter yet to come in.

Not in v1 (stated, not filled): phase x innings priors, bowler type (no data found in Cricsheet), opposing-player
matchups (Bandyopadhyay 2026). The next-ball model sees phase and state directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cricket.model.outcomes import K
from cricket.ratings.shrinkage import decayed_asof, dirichlet

C_PRIOR = 60.0
DECAY = 0.88
COLS = [f"c{k}" for k in range(K)]


def _counts(d: pd.DataFrame, who: str) -> pd.DataFrame:
    """Outcome counts per (player, match) for the role ``who`` (batter_id or bowler_id)."""
    oh = pd.get_dummies(d["cls"]).reindex(columns=range(K), fill_value=0).astype(np.int32)
    oh.columns = COLS
    oh[who] = d[who].to_numpy()
    oh["match_id"] = d["match_id"].to_numpy()
    return oh.groupby([who, "match_id"], sort=False)[COLS].sum().reset_index()


def ratings(d: pd.DataFrame, players: pd.DataFrame, matches: pd.DataFrame, role: str,
            prior: np.ndarray | None = None, c: float = C_PRIOR, r: float = DECAY) -> pd.DataFrame:
    """As-of ratings for ``role`` in {"bat", "bowl"}: columns player_id, match_id, n_eff, p0..p7.

    ``d`` carries a ``cls`` column; ``players`` the match lists (so non-batting listed players get a rating);
    ``matches`` gives each match's start date for the order. ``prior`` defaults to the population of ``d``.
    """
    who = "batter_id" if role == "bat" else "bowler_id"
    cnt = _counts(d, who).rename(columns={who: "player_id"})
    lst = players[players.match_id.isin(matches.match_id)][["player_id", "match_id"]].dropna().drop_duplicates()
    rows = lst.merge(cnt, on=["player_id", "match_id"], how="outer").fillna({c_: 0 for c_ in COLS})
    rows = rows.merge(matches[["match_id", "start_date"]], on="match_id", how="left").reset_index(drop=True)
    if prior is None:
        tot = d["cls"].value_counts().reindex(range(K), fill_value=0).to_numpy(dtype=float)
        prior = tot / tot.sum()
    s = decayed_asof(rows, "player_id", ["start_date", "match_id"], COLS, r)
    p = dirichlet(s, prior, c)
    out = rows[["player_id", "match_id"]].copy()
    out["n_eff"] = s.sum(axis=1)
    for k in range(K):
        out[f"p{k}"] = p[:, k]
    return out
