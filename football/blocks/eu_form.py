"""European matches in team form. ``eu_``.

racing² football research, 24 Nov 2026 (the owner: "use the European matches in team form"). From
``football/europe_form.py`` (each side's Champions League, Europa League and Conference League matches before the
collection day, 2024/25 on):

- ``eu_xgd_diff``: European non-penalty xG difference (120-day half-life), home less away; blank for a side with
  none counts as 0 (a side not in Europe has no European form, not an unknown one);
- ``eu_pae_diff``: European points above the market's expectation, home less away (same convention);
- ``eu_n90_diff``: European matches in the 90 days before, home less away (the load);
- ``eu_rest_min``: the fewer days since either side's last European match (capped at 30; 30 when neither played):
  a midweek trip before a league match;
- ``eu_any``: 1 where either side has played in Europe in the last year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.europe_form import COLUMNS

FEATURES = ["eu_xgd_diff", "eu_pae_diff", "eu_n90_diff", "eu_rest_min", "eu_any"]
CAP = 30.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in COLUMNS if c not in m.columns})
    out = m[["match_id"]].copy()
    known = m["eux_n90_h"].notna() & m["eux_n90_a"].notna()        # the European stage covers this match
    w_h, w_a = m["eux_w_h"].fillna(0), m["eux_w_a"].fillna(0)
    for f, c in (("eu_xgd_diff", "eux_xgd"), ("eu_pae_diff", "eux_pae")):
        out[f] = (m[f"{c}_h"].where(w_h > 0, 0.0).fillna(0) - m[f"{c}_a"].where(w_a > 0, 0.0).fillna(0)).where(known)
    out["eu_n90_diff"] = (m["eux_n90_h"] - m["eux_n90_a"]).where(known)
    out["eu_rest_min"] = np.fmin(m["eux_days_h"].fillna(CAP), m["eux_days_a"].fillna(CAP)).clip(upper=CAP) \
        .where(known)
    out["eu_any"] = ((w_h > 0) | (w_a > 0)).astype(float).where(known)
    return out
