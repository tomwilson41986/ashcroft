"""The next-ball feature frame (build spec 6.3): state, ratings, venue and competition, last-six-balls momentum.

The target is never a feature in the first innings: ``runs_needed`` and ``rrr`` are NaN there by construction.
Every rating is as of the match's start (cricket/ratings), so nothing from the match itself or later reaches a row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cricket.model.data import ball_frame, scope_matches
from cricket.model.outcomes import K, classify
from cricket.ratings.player_ratings import ratings
from cricket.ratings.venue_ratings import rates

PHASE_CODE = {"powerplay": 0, "middle": 1, "death": 2}
STATE = ["innings", "legal_before", "balls_left", "wickets_before", "runs_before", "crr", "runs_needed", "rrr",
         "phase_code", "last6_runs", "last6_wkts", "balls_since_wicket"]
BAT = [f"bat_p{k}" for k in range(K)] + ["bat_n"]
BOWL = [f"bowl_p{k}" for k in range(K)] + ["bowl_n"]
CONTEXT = ["venue_rpb", "venue_wpb", "comp_rpb"]
FEATURES = STATE + BAT + BOWL + CONTEXT


def t20_matches(m: pd.DataFrame) -> pd.DataFrame:
    """Men's 20-over, six-ball T20s (The Hundred's five-ball sets left out), decided, no revised target."""
    s = scope_matches(m, ("T20", "IT20"), "male")
    return s[(s.overs == 20) & (s.balls_per_over == 6)]


def add_state_features(f: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        f["crr"] = np.where(f.legal_before > 0, 6.0 * f.runs_before / f.legal_before, np.nan)
    f["rrr"] = f["required_rate"].clip(lower=0, upper=36)
    f["phase_code"] = f["phase"].map(PHASE_CODE).astype(np.int8)
    f.loc[f.innings == 1, ["runs_needed", "rrr", "target"]] = np.nan
    return f


def build(t: dict[str, pd.DataFrame], since: str = "2012-01-01", pop_before: str = "2023-01-01"):
    """The ball frame of every in-scope delivery since ``since`` with features and the outcome class ``cls``.

    Returns (frame, bat ratings, bowl ratings, prior). Ratings use every earlier in-scope match, also before
    ``since``; the population prior and rates come from matches before ``pop_before`` only."""
    m = t20_matches(t["matches"])
    f = ball_frame(t, m)
    f["cls"] = classify(f)
    early = f[pd.to_datetime(f.start_date) < pd.Timestamp(pop_before)]
    tot = early["cls"].value_counts().reindex(range(K), fill_value=0).to_numpy(dtype=float)
    prior = tot / tot.sum()
    br = ratings(f, t["players"], m, "bat", prior)
    bw = ratings(f, t["players"], m, "bowl", prior)
    vr = rates(f, m, pop_before)
    f = f[f.start_date >= since].copy()
    f = f.merge(br.rename(columns={"player_id": "batter_id", "n_eff": "bat_n",
                                   **{f"p{k}": f"bat_p{k}" for k in range(K)}}), on=["batter_id", "match_id"],
                how="left")
    f = f.merge(bw.rename(columns={"player_id": "bowler_id", "n_eff": "bowl_n",
                                   **{f"p{k}": f"bowl_p{k}" for k in range(K)}}), on=["bowler_id", "match_id"],
                how="left")
    f = f.merge(vr, on="match_id", how="left")
    f = add_state_features(f)
    return f, br, bw, prior
