"""The match played minute by minute, the scoring rates reacting to the score. ``gx_``.

racing² football research, 24 Nov 2026 (the owner: "build the match simulator"). ``sim`` plays a match as two
independent Poisson counts. Sides do not play that way: a side behind pushes on, a side ahead sits in or picks off.
This block plays the 90 minutes as a chain over the goal difference, each side's rate a minute depending on whether
it is level, ahead or behind:

- The base rates are ``sim``'s (the opening price's implied rates updated by each side's venue record).
- A side's state multipliers: its goals scored (and conceded) while level, ahead and behind over its last 30 matches
  before the collection day, against what its overall rate would have given in those minutes, shrunk towards the
  multiplier of all sides in earlier seasons (a prior worth 3 goals; a set constant, not fitted).
- The home side scores at: base rate x its scoring multiplier in its state x the away side's conceding multiplier in
  the opposite state / the all-sides multiplier (which both carry). The away side the same way round.
- The chain is solved exactly (no random draws): the chance of each goal difference after 90 minutes.

``gx_h`` / ``gx_a``: the chain's home / away chance less the plain Poisson's from the same base rates, in log odds
(what the game states add). ``gx_d``: the draw chance, chain less plain. ``gx_total``: expected goals, chain less
base. ``gx_lead_for_diff`` / ``gx_trail_for_diff``: the scoring multiplier ahead / behind, home less away.
``gx_lead_against_diff``: the conceding multiplier when ahead (holding a lead), home less away.

Goal minutes come from API-Football's events (2020 on, the linked leagues); blank where neither side has any.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks import sim
from football.blocks._util import collection_day, per_match, prior_sum, sides
from football.tactics import STATE_COLS

FEATURES = ["gx_h", "gx_a", "gx_d", "gx_total", "gx_lead_for_diff", "gx_trail_for_diff", "gx_lead_against_diff"]
STATES = ("trail", "level", "lead")
OPP = {"trail": "lead", "level": "level", "lead": "trail"}
WINDOW, PRIOR_GOALS, MAX_DIFF, MINUTES = 30, 3.0, 6, 90


def play(lh: np.ndarray, la: np.ndarray, fh: np.ndarray, fa: np.ndarray):
    """(home, draw, away chances, expected goals) of the chain. ``fh`` / ``fa``: (rows, 3) multipliers of the home /
    away scoring rate while the home side trails, is level, leads."""
    st = np.array([0] * MAX_DIFF + [1] + [2] * MAX_DIFF)
    rh = (np.asarray(lh, float) / MINUTES)[:, None] * fh[:, st]
    ra = (np.asarray(la, float) / MINUTES)[:, None] * fa[:, st]
    p = np.zeros_like(rh)
    p[:, MAX_DIFF] = 1.0
    goals = np.zeros(len(rh))
    for _ in range(MINUTES):
        up, dn = p * rh, p * ra
        goals += (up + dn).sum(axis=1)
        nxt = p - up - dn
        nxt[:, 1:] += up[:, :-1]
        nxt[:, -1] += up[:, -1]
        nxt[:, :-1] += dn[:, 1:]
        nxt[:, 0] += dn[:, 0]
        p = nxt
    return p[:, MAX_DIFF + 1:].sum(axis=1), p[:, MAX_DIFF], p[:, :MAX_DIFF].sum(axis=1), goals


def side_table(m: pd.DataFrame) -> pd.DataFrame:
    """Per side: minutes, goals for and goals against in each of its own states."""
    cols = {"min_level": ("gs_min_level", "gs_min_level"), "min_lead": ("gs_min_lead", "gs_min_trail"),
            "min_trail": ("gs_min_trail", "gs_min_lead"),
            "gf_level": ("gs_hg_level", "gs_ag_level"), "gf_lead": ("gs_hg_lead", "gs_ag_trail"),
            "gf_trail": ("gs_hg_trail", "gs_ag_lead"),
            "ga_level": ("gs_ag_level", "gs_hg_level"), "ga_lead": ("gs_ag_lead", "gs_hg_trail"),
            "ga_trail": ("gs_ag_trail", "gs_hg_lead")}
    return sides(m, cols)


def all_sides(s: pd.DataFrame) -> pd.DataFrame:
    """Per season: the scoring and conceding multipliers of all sides in the seasons before it (1 where none)."""
    cols = [f"{k}_{st}" for k in ("min", "gf", "ga") for st in STATES]
    tot = s.groupby("season")[cols].sum().sort_index()
    cum = tot.cumsum().shift(1)
    out = pd.DataFrame(index=tot.index)
    mins = cum[[f"min_{st}" for st in STATES]].sum(axis=1)
    for k, name in (("gf", "for"), ("ga", "against")):
        overall = cum[[f"{k}_{st}" for st in STATES]].sum(axis=1) / mins.where(mins > 0)
        for st in STATES:
            rate = cum[f"{k}_{st}"] / cum[f"min_{st}"].where(cum[f"min_{st}"] > 0)
            out[f"{name}_{st}"] = (rate / overall.where(overall > 0)).fillna(1.0)
    return out


def multipliers(s: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """{'for_<state>' / 'against_<state>': each side row's multiplier}, and the minutes of state record behind it."""
    coll = collection_day(s["date"])
    lag = {c: prior_sum(s["team"], s["date"], coll, s[c].to_numpy(float), WINDOW)[0]
           for c in (f"{k}_{st}" for k in ("min", "gf", "ga") for st in STATES)}
    glob = all_sides(s).reindex(s["season"]).reset_index(drop=True)
    mins = sum(lag[f"min_{st}"] for st in STATES)
    out = {}
    for k, name in (("gf", "for"), ("ga", "against")):
        overall = sum(lag[f"{k}_{st}"] for st in STATES) / np.where(mins > 0, mins, np.nan)
        for st in STATES:
            g = glob[f"{name}_{st}"].to_numpy(float)
            expected = np.nan_to_num(overall * lag[f"min_{st}"])
            out[f"{name}_{st}"] = (lag[f"{k}_{st}"] + PRIOR_GOALS * g) / (expected + PRIOR_GOALS)
            out[f"glob_{name}_{st}"] = g
    return out, mins


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in STATE_COLS if c not in m.columns})
    s = side_table(m)
    mult, mins = multipliers(s)
    p = per_match(s, {**mult, "mins": mins})
    d = sim.rates(m).merge(p, on="match_id", how="left")
    ok = (d["ok"].fillna(False) & ((d["mins_h"].fillna(0) > 0) | (d["mins_a"].fillna(0) > 0))).to_numpy()
    out = d[["match_id"]].copy()
    for f in FEATURES:
        out[f] = np.nan
    if not ok.any():
        return out
    w = d[ok]
    fh = np.column_stack([w[f"for_{st}_h"] * w[f"against_{OPP[st]}_a"] / w[f"glob_for_{st}_h"] for st in STATES])
    fa = np.column_stack([w[f"for_{OPP[st]}_a"] * w[f"against_{st}_h"] / w[f"glob_for_{OPP[st]}_a"] for st in STATES])
    fh, fa = np.nan_to_num(fh, nan=1.0), np.nan_to_num(fa, nan=1.0)
    lh, la = w["lh"].to_numpy(float), w["la"].to_numpy(float)
    h1, d1, a1, g1 = play(lh, la, fh, fa)
    h0, d0, a0, _ = play(lh, la, np.ones_like(fh), np.ones_like(fa))
    out.loc[ok, "gx_h"] = sim._logit(h1) - sim._logit(h0)
    out.loc[ok, "gx_a"] = sim._logit(a1) - sim._logit(a0)
    out.loc[ok, "gx_d"] = d1 - d0
    out.loc[ok, "gx_total"] = g1 - (lh + la)
    out.loc[ok, "gx_lead_for_diff"] = (w["for_lead_h"] - w["for_lead_a"]).to_numpy()
    out.loc[ok, "gx_trail_for_diff"] = (w["for_trail_h"] - w["for_trail_a"]).to_numpy()
    out.loc[ok, "gx_lead_against_diff"] = (w["against_lead_h"] - w["against_lead_a"]).to_numpy()
    return out
