"""Sportmonks' own match prediction against the opening price. ``sp_``.

racing² football research, 24 Nov 2026 (the owner: "use these in our models and prices" - the Odds & Predictions
bundle). Sportmonks publishes a probability for each match's result, over/under 2.5 and both teams to score. Each
feature is Sportmonks' chance less the opening price's fair chance:

- ``sp_h`` / ``sp_a``: home / away win, in log odds; ``sp_d``: the draw, in chance;
- ``sp_o25``: over 2.5 goals, in log odds (the opening over/under price); ``sp_btts``: both teams to score (no
  opening price to set it against: the chance itself).

**Caveat on time.** Sportmonks gives no time for a prediction, and its historical value may have been revised up to
the kick-off (line-ups, late news). A test on the history can therefore credit the block with information that came
after the early price: it is read as an upper bound. ``sources/sportmonks.py`` keeps timestamped snapshots from
24 Nov 2026 for a clean forward read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["sp_h", "sp_a", "sp_d", "sp_o25", "sp_btts"]
NEED = ["smx_pr_ft_h", "smx_pr_ft_d", "smx_pr_ft_a", "smx_pr_o25", "smx_pr_btts", "qo_h", "qo_a", "qo_o"]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in NEED if c not in m.columns})
    out = m[["match_id"]].copy()
    tot = m["smx_pr_ft_h"] + m["smx_pr_ft_d"] + m["smx_pr_ft_a"]
    ph, pd_, pa = (m[f"smx_pr_ft_{k}"] / tot.where(tot > 0) for k in ("h", "d", "a"))
    out["sp_h"] = _logit(ph) - _logit(m["qo_h"])
    out["sp_a"] = _logit(pa) - _logit(m["qo_a"])
    out["sp_d"] = pd_ - (1 - m["qo_h"] - m["qo_a"])
    out["sp_o25"] = _logit(m["smx_pr_o25"]) - _logit(m["qo_o"])
    out["sp_btts"] = m["smx_pr_btts"]
    for f in FEATURES:
        out[f] = out[f].where(np.isfinite(out[f]))
    return out
