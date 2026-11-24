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


def prior_sum(group, date, coll, value, window: int) -> tuple[np.ndarray, np.ndarray]:
    """For each row: the sum and the count of ``value`` over the last ``window`` rows of its ``group`` dated strictly
    before the row's collection day ``coll`` (rows with a missing value do not count). Arrays in the rows' order."""
    group = pd.Series(np.asarray(group)).reset_index(drop=True)
    date = pd.to_datetime(pd.Series(np.asarray(date))).to_numpy("datetime64[ns]")
    coll = pd.to_datetime(pd.Series(np.asarray(coll))).to_numpy("datetime64[ns]")
    value = np.asarray(value, float)
    s, n = np.zeros(len(value)), np.zeros(len(value))
    for _, pos in group.groupby(group, sort=False):
        i = pos.index.to_numpy()
        ok = i[np.isfinite(value[i])]
        ok = ok[np.argsort(date[ok], kind="stable")]
        cs = np.concatenate([[0.0], np.cumsum(value[ok])])
        hi = np.searchsorted(date[ok], coll[i], side="left")             # strictly before the collection day
        lo = np.maximum(0, hi - window)
        s[i], n[i] = cs[hi] - cs[lo], hi - lo
    return s, n


__all__ += ["prior_sum"]


def at_collection(s: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """``values`` (one row a row of ``s``, each the side's state before that match) as they stood at the side's first
    match of its collection window: a side's later matches in the window take that row's values, blank or not."""
    cut = collection_day(s["date"]).to_numpy()
    g = pd.Series(pd.MultiIndex.from_arrays([s["team"].to_numpy(), cut]).factorize()[0])
    order = np.lexsort((np.arange(len(s)), pd.to_datetime(s["date"]).to_numpy("datetime64[ns]"), g.to_numpy()))
    first = pd.Series(order, index=g.to_numpy()[order]).groupby(level=0).first()
    return values.iloc[first.reindex(g.to_numpy()).to_numpy()].set_axis(values.index)


def decayed(s: pd.DataFrame, cols: list[str], half_days: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Each side's mean of ``cols`` over its earlier matches, weighted by exp(-ln2 * days ago / ``half_days``), and
    the weight behind it (the effective number of matches), both lagged to the collection day. ``s`` is ``sides``'
    table (sorted by team and date). A match with a missing value does not count for that column; the weight
    returned is the first column's."""
    lam = np.log(2.0) / half_days
    day = pd.to_datetime(s["date"]).to_numpy("datetime64[D]").astype(float)
    team = s["team"].to_numpy()
    out, wt = {}, None
    for c in cols:
        v = s[c].to_numpy(float)
        mean, w_out = np.full(len(s), np.nan), np.zeros(len(s))
        S = W = 0.0
        for i in range(len(s)):
            if i == 0 or team[i] != team[i - 1]:
                S = W = 0.0
            else:
                f = np.exp(-lam * (day[i] - day[i - 1]))
                S, W = S * f, W * f
            if W > 0:
                mean[i] = S / W
            w_out[i] = W
            if np.isfinite(v[i]):
                S, W = S + v[i], W + 1.0
        out[c] = mean
        if wt is None:
            wt = w_out
    vals = at_collection(s, pd.DataFrame(out, index=s.index).assign(_w=wt))
    return vals[cols], vals["_w"].to_numpy()


def split_record(s: pd.DataFrame, value: str, kind: np.ndarray, kinds: tuple, window: int) -> dict:
    """{k: (sum, count)} of ``value`` over each side's last ``window`` earlier matches of kind ``k`` (``kind``: each
    row's own kind), dated strictly before the row's collection day, for every row whatever its own kind."""
    coll = collection_day(s["date"])
    out = {}
    for k in kinds:
        v = np.where(kind == k, s[value].to_numpy(float), np.nan)
        out[k] = prior_sum(s["team"], s["date"], coll, v, window)
    return out


def per_match(s: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """One row a match from per-side arrays: ``cols`` = {name: array}; gives ``<name>_h`` and ``<name>_a``."""
    f = pd.DataFrame(cols, index=s.index).assign(match_id=s["match_id"].to_numpy(), venue=s["venue"].to_numpy())
    h = f[f["venue"] == "H"].drop(columns="venue").set_index("match_id").add_suffix("_h")
    a = f[f["venue"] == "A"].drop(columns="venue").set_index("match_id").add_suffix("_a")
    return h.join(a, how="outer").reset_index()


__all__ += ["at_collection", "decayed", "split_record", "per_match"]
