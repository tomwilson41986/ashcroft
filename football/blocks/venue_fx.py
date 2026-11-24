"""Each side's own home and away effect. ``hv_``.

racing² football research, 24 Nov 2026 (the owner: home / away adjustments). The market prices a league's home
advantage; some sides are better at home, or travel better, than that. The measure is the goal difference less the
closing market's expected one (so the league's own home advantage is already out), over the side's last 15 home and
last 15 away matches before the collection day, each shrunk towards zero (sum / (n + 5)):

- ``hv_home_split``: the home side's record at home less its record away;
- ``hv_away_split``: the away side's record away less its record at home;
- ``hv_net``: the first less the second: the venue's pull on today's match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import per_match, sides, split_record

FEATURES = ["hv_home_split", "hv_away_split", "hv_net"]
WINDOW, SHRINK = 15, 5.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m, {"xg": ("mclam_h", "mclam_a"), "xga": ("mclam_a", "mclam_h")})
    s["gae"] = (s["gf"] - s["ga"]) - (s["xg"] - s["xga"])
    rec = split_record(s, "gae", s["venue"].to_numpy(), ("H", "A"), WINDOW)
    mean = {k: np.where(n > 0, t / (n + SHRINK), np.nan) for k, (t, n) in rec.items()}
    p = per_match(s, {"hma": mean["H"] - mean["A"]})
    out = p[["match_id"]].copy()
    out["hv_home_split"] = p["hma_h"]
    out["hv_away_split"] = -p["hma_a"]
    out["hv_net"] = out["hv_home_split"] - out["hv_away_split"]
    return out
