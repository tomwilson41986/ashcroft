"""Each match's shapes and game states from API-Football's line-ups and events: the raw columns of two blocks.

racing² football research, 24 Nov 2026 (the owner: "build the match simulator and formation matchups"). One row a
canonical match, what happened in it (nothing here is lagged: ``football/blocks/formation.py`` and
``football/blocks/match_sim.py`` take only matches before the collection day):

- ``tq_def_h`` / ``tq_mid_h`` / ``tq_fwd_h`` (and ``_a``): the starting eleven's defenders, midfielders and forwards,
  counted from each starter's listed position (G, D, M, F). Blank unless eleven starters have a position and one is
  the goalkeeper. A coarse shape: a 4-2-3-1 and a 4-5-1 both read 4-5-1.
- ``gs_min_level`` / ``gs_min_lead`` / ``gs_min_trail``: minutes of the first 90 the home side spent level, ahead
  and behind; ``gs_hg_<state>`` / ``gs_ag_<state>``: the home and away goals scored in each (the state is the home
  side's, read before the goal). From the goal events; blank where the match has no events or they do not add up to
  the full-time score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ("level", "lead", "trail")
SHAPE_COLS = [f"tq_{p}_{s}" for s in ("h", "a") for p in ("def", "mid", "fwd")]
STATE_COLS = [f"gs_min_{s}" for s in STATES] + [f"gs_{t}g_{s}" for t in ("h", "a") for s in STATES]
COLUMNS = SHAPE_COLS + STATE_COLS


def shapes(hp: pd.DataFrame) -> pd.DataFrame:
    """One row a fixture: the two elevens' defenders, midfielders and forwards."""
    s = hp[hp["role"] == "starter"].drop_duplicates(["apif_fixture_id", "side", "apif_player_id", "player"])
    pos = s["pos"].astype("string").str.upper().str[:1]
    s = s.assign(pos=pos.where(pos.isin(["G", "D", "M", "F"])))
    n = s.groupby(["apif_fixture_id", "side"]).agg(starters=("pos", "size"), known=("pos", "count"))
    c = s.groupby(["apif_fixture_id", "side", "pos"]).size().unstack("pos", fill_value=0)
    c = c.reindex(columns=["G", "D", "M", "F"], fill_value=0).join(n)
    ok = (c["starters"] == 11) & (c["known"] == 11) & (c["G"] == 1)
    c = c[ok].rename(columns={"D": "def", "M": "mid", "F": "fwd"})[["def", "mid", "fwd"]].astype(float)
    w = c.unstack("side")
    out = pd.DataFrame(index=w.index)
    for side, tag in (("home", "h"), ("away", "a")):
        for p in ("def", "mid", "fwd"):
            out[f"tq_{p}_{tag}"] = w[(p, side)] if (p, side) in w.columns else np.nan
    return out.reset_index()


def goal_states(hm: pd.DataFrame, he: pd.DataFrame, full: int = 90) -> pd.DataFrame:
    """One row a fixture with goal events that add up to its score: minutes and goals by the home side's state."""
    g = he[(he["type"] == "Goal") & ~he["detail"].astype(str).str.lower().str.contains("missed")]
    by_fix = {k: v.sort_values("minute", kind="stable") for k, v in g.groupby("apif_fixture_id")}
    rows = []
    for r in hm.itertuples(index=False):
        if not getattr(r, "has_events", True) or r.ft_home != r.ft_home or r.ft_away != r.ft_away:
            continue
        acc = dict.fromkeys(STATE_COLS, 0.0)
        diff, t0, hg, ag = 0, 0, 0, 0
        ev = by_fix.get(r.apif_fixture_id)
        for e in ([] if ev is None else ev.itertuples(index=False)):
            m = min(int(e.minute) if e.minute == e.minute else 0, full)
            state = "level" if diff == 0 else ("lead" if diff > 0 else "trail")
            acc[f"gs_min_{state}"] += m - t0
            t0 = m
            if e.apif_team_id == r.apif_home_id:
                acc[f"gs_hg_{state}"] += 1
                diff, hg = diff + 1, hg + 1
            else:
                acc[f"gs_ag_{state}"] += 1
                diff, ag = diff - 1, ag + 1
        acc["gs_min_" + ("level" if diff == 0 else ("lead" if diff > 0 else "trail"))] += full - t0
        if hg == int(r.ft_home) and ag == int(r.ft_away):          # the events tell the whole score
            rows.append({"apif_fixture_id": r.apif_fixture_id, **acc})
    return pd.DataFrame(rows, columns=["apif_fixture_id"] + STATE_COLS)


def build(linked: pd.DataFrame, hm: pd.DataFrame, hp: pd.DataFrame, he: pd.DataFrame) -> pd.DataFrame:
    """``match_id`` + ``COLUMNS`` for the linked matches (``football.player_ratings.link_matches``)."""
    ids = linked[["match_id", "apif_fixture_id"]].dropna().drop_duplicates("match_id")
    out = ids.merge(shapes(hp), on="apif_fixture_id", how="left")
    out = out.merge(goal_states(hm[hm["apif_fixture_id"].isin(ids["apif_fixture_id"])], he), on="apif_fixture_id",
                    how="left")
    return out[["match_id"] + COLUMNS]
