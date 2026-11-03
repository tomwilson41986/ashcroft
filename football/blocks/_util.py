"""Helpers for football blocks: the per-side table and windows lagged to the collection day."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.team_features import collection_day, windows


def sides(m: pd.DataFrame, cols: dict | None = None) -> pd.DataFrame:
    """Two rows a match (home ``H``, away ``A``): team, opponent, date, season, competition, goals for and against,
    and ``cols`` = {name: (home column, away column)} taken from the side's point of view."""
    base = ["match_id", "date", "season", "competition_id", "country"]
    out = []
    for venue, me, op in (("H", "home", "away"), ("A", "away", "home")):
        p = m[base].copy()
        p["venue"], p["team"], p["opp"] = venue, m[f"{me}_team_id"], m[f"{op}_team_id"]
        p["gf"], p["ga"] = m[f"ft_{me}"], m[f"ft_{op}"]
        for name, (hc, ac) in (cols or {}).items():
            p[name] = m[hc] if venue == "H" else m[ac]
        out.append(p)
    s = pd.concat(out, ignore_index=True)
    s["date"] = pd.to_datetime(s["date"])
    s["own_r"] = 0.0
    return s.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def lagged(s: pd.DataFrame, measures: list[str], key: str = "team") -> pd.DataFrame:
    """Windows (last 5, last 10, EWM, season to date) of ``measures`` over the side's earlier matches, lagged to the
    collection day. ``key``: the column that defines a side (e.g. team and venue together)."""
    t = s.copy()
    t["_orig_team"] = t["team"]
    t["team"] = t[key]
    w = windows(t, measures)
    w["team"] = t["_orig_team"].to_numpy()
    return w


def home_minus_away(s: pd.DataFrame, w: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    """One row a match: home minus away of ``cols`` (each named ``prefix + col``)."""
    w = w.assign(venue=s["venue"].to_numpy(), match_id=s["match_id"].to_numpy())
    h = w[w["venue"] == "H"].set_index("match_id")[cols]
    a = w[w["venue"] == "A"].set_index("match_id")[cols]
    idx = h.index.union(a.index)
    out = pd.DataFrame({"match_id": idx})
    for c in cols:
        out[prefix + c] = (h[c].reindex(idx) - a[c].reindex(idx)).to_numpy()
    return out


__all__ = ["sides", "lagged", "home_minus_away", "collection_day", "np", "pd"]
