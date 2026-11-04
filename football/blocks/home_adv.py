"""Home advantage as it stands now, promoted and relegated sides, the early season. ``ha_``.

racing² football research, 4 Nov 2026. Ted Knutson's lessons:
- Home advantage drifts by league and season. Premier League home goal difference went 0.41, 0.32, 0.09 over three
  seasons while the Championship's held, and models built on history suffer.
- Promoted sides are overrated and slow to adjust ("a free quarter goal" against a promoted side).
- The early season inflates metrics.
(The Transfer Flow, 22 Jul 2025; Variance Betting.)

From the fixture list and the results before each match's collection day only:

- ``ha_hgd_190`` / ``ha_hgd_380`` / ``ha_hgd_1900``: the competition's home goal difference a match over its last 190,
  380 and 1,900 matches (about half a season, a season, five seasons) before the collection day.
- ``ha_drift``: the last season's level less the five-season level.
- ``ha_promoted_*`` / ``ha_relegated_*``: the side moved up (down) a tier since last season (blank when last season's
  tier is unknown). ``ha_played_*``: its matches this season before the collection day.
- ``ha_promo_early_*``: promoted times the early-season weight max(0, 1 - played / 10). The ``*_diff`` columns are home
  less away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import collection_day, sides

FEATURES = ["ha_hgd_190", "ha_hgd_380", "ha_hgd_1900", "ha_drift",
            "ha_promoted_h", "ha_promoted_a", "ha_promoted_diff", "ha_relegated_diff",
            "ha_played_h", "ha_played_a", "ha_promo_early_diff"]
WINDOWS = (190, 380, 1900)


def league_home_gd(m: pd.DataFrame) -> pd.DataFrame:
    """Each match's competition home goal difference over the last N matches finished before its collection day."""
    d = m[["match_id", "competition_id", "date", "ft_home", "ft_away"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["coll"] = pd.to_datetime(collection_day(d["date"]))
    out = []
    for _, g in d.groupby("competition_id", sort=False):
        played = g[g["ft_home"].notna() & g["ft_away"].notna()].sort_values("date")
        dates = played["date"].to_numpy("datetime64[ns]")
        cs = np.concatenate([[0.0], np.cumsum((played["ft_home"] - played["ft_away"]).to_numpy(float))])
        idx = np.searchsorted(dates, g["coll"].to_numpy("datetime64[ns]"), side="left")    # strictly before
        r = pd.DataFrame({"match_id": g["match_id"].to_numpy()})
        for w in WINDOWS:
            lo = np.maximum(0, idx - w)
            n = idx - lo
            r[f"ha_hgd_{w}"] = np.where(n >= min(w, 100) // 2, (cs[idx] - cs[lo]) / np.maximum(n, 1), np.nan)
        out.append(r)
    r = pd.concat(out, ignore_index=True)
    r["ha_drift"] = r["ha_hgd_380"] - r["ha_hgd_1900"]
    return r


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m)
    tier = m.set_index("match_id")["tier"] if "tier" in m.columns else None
    s["tier"] = s["match_id"].map(tier) if tier is not None else np.nan
    # each team's tier a season (its most common), and last season's
    ts = s.groupby(["team", "season"])["tier"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    prev = ts.copy()
    prev.index = pd.MultiIndex.from_arrays([prev.index.get_level_values(0), prev.index.get_level_values(1) + 1])
    key = pd.MultiIndex.from_arrays([s["team"], s["season"]])
    now, before = ts.reindex(key).to_numpy(float), prev.reindex(key).to_numpy(float)
    s["promoted"] = np.where(np.isfinite(before), (before > now).astype(float), np.nan)
    s["relegated"] = np.where(np.isfinite(before), (before < now).astype(float), np.nan)
    # matches played this season before the collection day (the fixture count, not results)
    s["coll"] = pd.to_datetime(collection_day(s["date"]))
    played = []
    for _, g in s.groupby(["team", "season"], sort=False):
        dates = np.sort(g["date"].to_numpy("datetime64[ns]"))
        played.append(pd.Series(np.searchsorted(dates, g["coll"].to_numpy("datetime64[ns]"), side="left"),
                                index=g.index))
    s["played"] = pd.concat(played).reindex(s.index).astype(float)
    s["promo_early"] = s["promoted"] * np.clip(1 - s["played"] / 10, 0, 1)
    h = s[s["venue"] == "H"].set_index("match_id")
    a = s[s["venue"] == "A"].set_index("match_id")
    idx = h.index.union(a.index)
    out = pd.DataFrame({"match_id": idx})
    out["ha_promoted_h"], out["ha_promoted_a"] = h["promoted"].reindex(idx).to_numpy(), a["promoted"].reindex(idx).to_numpy()
    out["ha_promoted_diff"] = out["ha_promoted_h"] - out["ha_promoted_a"]
    out["ha_relegated_diff"] = (h["relegated"].reindex(idx) - a["relegated"].reindex(idx)).to_numpy()
    out["ha_played_h"], out["ha_played_a"] = h["played"].reindex(idx).to_numpy(), a["played"].reindex(idx).to_numpy()
    out["ha_promo_early_diff"] = (h["promo_early"].reindex(idx) - a["promo_early"].reindex(idx)).to_numpy()
    out = out.merge(league_home_gd(m), on="match_id", how="left")
    return out[["match_id"] + FEATURES]
