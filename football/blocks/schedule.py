"""Rest and congestion: days since each side's previous league match and its matches in the last 14 and 28 days,
from the fixture dates (public in advance; no result enters). ``sch_``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import sides

FEATURES = ["sch_rest_diff", "sch_rest_h", "sch_rest_a", "sch_n14_diff", "sch_n28_diff"]


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m)
    g = s.groupby("team", sort=False)["date"]
    s["rest"] = (s["date"] - g.shift(1)).dt.days.clip(upper=21)
    for n in (14, 28):
        def count(x, n=n):
            d = x.to_numpy("datetime64[D]").astype(np.int64)
            return pd.Series(np.searchsorted(d, d, side="left") - np.searchsorted(d, d - n, side="left"),
                             index=x.index)
        s[f"n{n}"] = g.transform(count)
    h = s[s["venue"] == "H"].set_index("match_id")
    a = s[s["venue"] == "A"].set_index("match_id")
    idx = h.index.union(a.index)
    out = pd.DataFrame({"match_id": idx})
    out["sch_rest_h"], out["sch_rest_a"] = h["rest"].reindex(idx).to_numpy(), a["rest"].reindex(idx).to_numpy()
    out["sch_rest_diff"] = out["sch_rest_h"] - out["sch_rest_a"]
    for n in (14, 28):
        out[f"sch_n{n}_diff"] = (h[f"n{n}"].reindex(idx) - a[f"n{n}"].reindex(idx)).to_numpy()
    return out
