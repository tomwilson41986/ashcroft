"""Fatigue and fixture congestion beyond ``rest_days``: matches in the last 7 and 10 days, short turnarounds, and
the days to the side's next match (rotation ahead of it). ``fat_``.

racing² football research, 4 Nov 2026 (the owner: "features around tiredness and fatigue: days since last game,
number of games in X days"). From the fixture dates of the matrix's league matches only: the cup and European
fixtures are not in our history (API-Football history holds the 22 league competitions), so congestion from them is
"no data found" here. No result or price enters.

- ``fat_n7_*`` / ``fat_n10_*``: the side's league matches in the 7 (10) days before the match.
- ``fat_short_*``: the side's previous match was 3 days or fewer before (a midweek turnaround).
- ``fat_next_*``: days to the side's next league match, capped at 14. The fixture list is public weeks ahead, but
  the date used is the one the match was played on: a postponement announced after the early price is a small
  look-ahead (flagged; the lag test covers results and prices, not fixture dates).
- ``*_diff``: home less away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import sides

FEATURES = ["fat_n7_h", "fat_n7_a", "fat_n7_diff", "fat_n10_diff", "fat_short_h", "fat_short_a", "fat_short_diff",
            "fat_next_h", "fat_next_a", "fat_next_diff"]


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m)
    g = s.groupby("team", sort=False)["date"]
    for n in (7, 10):
        def count(x, n=n):
            d = x.to_numpy("datetime64[D]").astype(np.int64)
            return pd.Series(np.searchsorted(d, d, side="left") - np.searchsorted(d, d - n, side="left"),
                             index=x.index)
        s[f"n{n}"] = g.transform(count).astype(float)
    rest = (s["date"] - g.shift(1)).dt.days
    s["short"] = np.where(rest.notna(), (rest <= 3).astype(float), np.nan)
    s["next"] = (g.shift(-1) - s["date"]).dt.days.clip(upper=14)
    h = s[s["venue"] == "H"].set_index("match_id")
    a = s[s["venue"] == "A"].set_index("match_id")
    idx = h.index.union(a.index)
    out = pd.DataFrame({"match_id": idx})
    for k in ("n7", "short", "next"):
        out[f"fat_{k}_h"], out[f"fat_{k}_a"] = h[k].reindex(idx).to_numpy(), a[k].reindex(idx).to_numpy()
        out[f"fat_{k}_diff"] = out[f"fat_{k}_h"] - out[f"fat_{k}_a"]
    out["fat_n10_diff"] = (h["n10"].reindex(idx) - a["n10"].reindex(idx)).to_numpy()
    return out[["match_id"] + FEATURES]
