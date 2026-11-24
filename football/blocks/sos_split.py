"""A side's record against better sides and against worse ones. ``ss_``.

racing² football research, 24 Nov 2026 (the owner: metrics by team strength, by strength of schedule). An average
hides who it was earned against. Each earlier match is typed by the opponent's market rating against the side's own
at the time (``better`` or ``worse``); the measure is the goal difference less the closing market's expected one, over
the side's last 10 matches of each type before the collection day, shrunk towards zero (sum / (n + 2)):

- ``ss_better_diff`` / ``ss_worse_diff``: the record against better / worse sides, home less away;
- ``ss_fit_diff``: each side's record in the type of match it plays today (the home side against a better or worse
  visitor, and the reverse), home less away;
- ``ss_bully_diff``: record against worse sides less record against better ones (a flat-track bully), home less away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import per_match, sides, split_record

FEATURES = ["ss_better_diff", "ss_worse_diff", "ss_fit_diff", "ss_bully_diff"]
WINDOW, SHRINK = 10, 2.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m, {"xg": ("mclam_h", "mclam_a"), "xga": ("mclam_a", "mclam_h"), "r_own": ("r_h", "r_a"),
                  "r_opp": ("r_a", "r_h")})
    s["gae"] = (s["gf"] - s["ga"]) - (s["xg"] - s["xga"])
    known = s["r_own"].notna() & s["r_opp"].notna()
    kind = np.where(~known, "none", np.where(s["r_opp"] > s["r_own"], "better", "worse"))
    rec = split_record(s, "gae", kind, ("better", "worse"), WINDOW)
    mean = {k: np.where(n > 0, t / (n + SHRINK), np.nan) for k, (t, n) in rec.items()}
    fit = np.where(kind == "better", mean["better"], np.where(kind == "worse", mean["worse"], np.nan))
    p = per_match(s, {"better": mean["better"], "worse": mean["worse"], "fit": fit,
                      "bully": mean["worse"] - mean["better"]})
    out = p[["match_id"]].copy()
    for k in ("better", "worse", "fit", "bully"):
        out[f"ss_{k}_diff"] = p[f"{k}_h"] - p[f"{k}_a"]
    return out
