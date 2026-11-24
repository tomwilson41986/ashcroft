"""The referee's record. ``rf_``.

racing² football research, 24 Nov 2026 (the metric library: officials). Over the referee's last 60 matches before
the collection day (football-data names the referee in the English leagues and a few others; blank elsewhere):

- ``rf_home_gae``: the home side's goal difference less the closing market's expected one, shrunk (sum / (n + 20));
- ``rf_cards``: cards a match (yellow plus two a red), shrunk towards 4 (20 matches' weight);
- ``rf_n``: how many of his matches are behind them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import collection_day, prior_sum

FEATURES = ["rf_home_gae", "rf_cards", "rf_n"]
WINDOW, SHRINK, CARDS_PRIOR = 60, 20.0, 4.0


def build(m: pd.DataFrame) -> pd.DataFrame:
    out = m[["match_id"]].copy()
    if "referee" not in m.columns or m["referee"].notna().sum() == 0:
        return out.assign(**{f: np.nan for f in FEATURES})
    ref = m["referee"].astype("string").str.strip().str.lower()
    ref = ref.where(ref.str.len() > 0)
    coll = collection_day(m["date"])
    gae = (m["ft_home"] - m["ft_away"]) - (m["mclam_h"] - m["mclam_a"])
    zero = pd.Series(0.0, index=m.index)
    cards = sum((m[c] if c in m.columns else zero).fillna(0) * w for c, w in
                (("yellow_home", 1), ("yellow_away", 1), ("red_home", 2), ("red_away", 2)))
    cards = cards.where(m["yellow_home"].notna() if "yellow_home" in m.columns else False)
    key = ref.fillna("?").to_numpy()
    t, n = prior_sum(key, m["date"], coll, gae.to_numpy(float), WINDOW)
    ct, cn = prior_sum(key, m["date"], coll, np.asarray(cards, float), WINDOW)
    has = ref.notna().to_numpy()
    out["rf_home_gae"] = np.where(has & (n > 0), t / (n + SHRINK), np.nan)
    out["rf_cards"] = np.where(has & (cn > 0), (ct + SHRINK * CARDS_PRIOR) / (cn + SHRINK), np.nan)
    out["rf_n"] = np.where(has, n, np.nan)
    return out
