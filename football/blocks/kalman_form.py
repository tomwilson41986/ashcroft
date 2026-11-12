"""Form as a filtered state with its uncertainty, in place of a ladder of raw windows. ``kf_``.

racing² football research, 10 Nov 2026 (the model review, findings A6 and A7: the windows are raw means of 2 to 10
matches with no allowance for sample size or noise, and they run across the summer break). A local-level Kalman
filter (``model.state_space.kalman_filter_runs``, the horse and greyhound models' filter, read-only) of each side's
performance against the closing market:

- ``gae``: goal difference less the market's expected one (the implied closing goal rates);
- ``sotm``: shots-on-target difference less the market's expected one (the expected goal difference times the
  league's shots on target a goal over its earlier seasons): process data against the market, in every league with
  shot counts, back to the first season.

The state starts at zero (a side new to the data is what the market says it is) and its variance grows with the days
since the side's last match, so a summer break widens it. The drift and noise variances are fitted by maximum
likelihood on the first three seasons of the data only, then held. Per match, lagged to the collection day:

- ``kf_<m>_diff``: the filtered level before the match, home less away;
- ``kf_<m>_sd``: the root of the two sides' prior variances summed: how unsure the level is;
- ``kf_<m>_innov_diff``: each side's surprise in its latest match (the result less the level before it), home less
  away: momentum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import collection_day, sides

MEASURES = ["gae", "sotm"]
FEATURES = [f"kf_{c}_{k}" for c in MEASURES for k in ("diff", "sd", "innov_diff")]
FIT_SEASONS = 3
FIT_ROWS = 15000


def measures(m: pd.DataFrame) -> pd.DataFrame:
    s = sides(m, {"mf": ("mclam_h", "mclam_a"), "ma": ("mclam_a", "mclam_h"), "sot_f": ("sot_home", "sot_away"),
                  "sot_a": ("sot_away", "sot_home")})
    xgd = s["mf"] - s["ma"]
    s["gae"] = (s["gf"] - s["ga"]) - xgd
    # the league's shots on target a goal over its earlier seasons (none for its first season in the data)
    home = s[s["venue"] == "H"]
    tot = home.assign(g=home["gf"] + home["ga"], t=home["sot_f"] + home["sot_a"]).dropna(subset=["g", "t"]) \
        .groupby(["competition_id", "season"])[["g", "t"]].sum().sort_index()
    cum = tot.groupby(level=0).cumsum().groupby(level=0).shift(1)
    rate = (cum["t"] / cum["g"].where(cum["g"] > 0)).rename("rate").reset_index()
    s = s.merge(rate, on=["competition_id", "season"], how="left").sort_values(["team", "date", "match_id"]) \
        .reset_index(drop=True)
    s["sotm"] = (s["sot_f"] - s["sot_a"]) - s["rate"] * xgd
    return s


def params(s: pd.DataFrame, c: str) -> tuple[float, float, float]:
    """(drift variance a year, noise variance, starting variance), fitted on the data's first seasons."""
    from model.state_space import fit_kalman_params
    first = s[s["season"] < s["season"].min() + FIT_SEASONS]
    v = float(np.nanvar(first[c])) if first[c].notna().any() else float("nan")
    if not np.isfinite(v) or v <= 0:
        return 0.1, 1.0, 0.2
    start = (0.1 * v, v, 0.2 * v)
    if first[c].notna().sum() < 2000:
        return start
    if len(first) > FIT_ROWS:                               # a fixed sample of sides keeps the block to seconds
        keep = first.groupby("team").size().sort_index().cumsum()
        first = first[first["team"].isin(keep[keep <= FIT_ROWS].index)]
    f = fit_kalman_params(first, c, horse_col="team", date_col="date", time_col=None, q_career_mult=1.0, x0=start)
    return f["q"], f["r"], f["init_var"]


def build(m: pd.DataFrame) -> pd.DataFrame:
    from model.state_space import kalman_filter_runs
    s = measures(m)
    days = s["date"].to_numpy("datetime64[D]").astype(np.int64)
    teams = s["team"].astype(str).to_numpy()
    cut = collection_day(s["date"]).to_numpy()
    for c in MEASURES:
        q, r, v0 = params(s, c)
        vals = np.array(s[c], dtype=float)
        pm, pv, _qm, _qv, _n, _ll = kalman_filter_runs(vals, days, teams, q, r, v0, 0.0)
        s[f"{c}_lvl"], s[f"{c}_var"] = pm, pv
        s[f"{c}_innov"] = pd.Series(vals - pm).groupby(teams).shift(1).to_numpy()
        cols = [f"{c}_lvl", f"{c}_var", f"{c}_innov"]
        s[cols] = s.groupby(["team", cut])[cols].transform("first")      # the side's first match of the window
    h = s[s["venue"] == "H"].set_index("match_id")
    a = s[s["venue"] == "A"].set_index("match_id")
    idx = pd.Index(m["match_id"])
    out = pd.DataFrame({"match_id": idx})
    for c in MEASURES:
        out[f"kf_{c}_diff"] = (h[f"{c}_lvl"].reindex(idx) - a[f"{c}_lvl"].reindex(idx)).to_numpy()
        out[f"kf_{c}_sd"] = np.sqrt(h[f"{c}_var"].reindex(idx) + a[f"{c}_var"].reindex(idx)).to_numpy()
        out[f"kf_{c}_innov_diff"] = (h[f"{c}_innov"].reindex(idx) - a[f"{c}_innov"].reindex(idx)).to_numpy()
    return out[["match_id"] + FEATURES]
