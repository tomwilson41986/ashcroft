"""Formation matchups: the shape each side is expected to play, and how such shapes have met. ``fm_``.

racing² football research, 24 Nov 2026 (the owner: tactical fit, formation matchups). The shapes are the coarse ones
of ``football/tactics.py`` (defenders, midfielders, forwards counted from the starters' positions), sorted into six
kinds: a back three, four or five (three or fewer, four, five or more defenders) with one forward or with two or more.
The shape a side is expected to play is the one it last started before the collection day. Performance is the goal
difference less the closing market's expected one, so a shape's edge is what the price missed.

- ``fm_backs_diff`` / ``fm_fwd_diff``: the expected defenders (3 to 5) and forwards (1 or 2), home less away;
- ``fm_changes_diff``: how often the side changed kind in its last 10 matches, home less away;
- ``fm_fit_diff``: each side's record in its last 10 matches against the opponent's expected back line (three, four
  or five), shrunk (sum / (n + 3)), home less away;
- ``fm_table``: the home side's edge in the last 3,000 matches anywhere that set the two expected kinds against each
  other, home kind against away kind, shrunk (sum / (n + 100)).

Line-ups come from API-Football (2020 on, the linked leagues); blank elsewhere and where a shape is not known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import at_collection, collection_day, per_match, prior_sum, sides, split_record
from football.tactics import SHAPE_COLS

FEATURES = ["fm_backs_diff", "fm_fwd_diff", "fm_changes_diff", "fm_fit_diff", "fm_table"]
OWN_WINDOW, OWN_SHRINK, TABLE_WINDOW, TABLE_SHRINK = 10, 3.0, 3000, 100.0


def kind(defenders, forwards) -> np.ndarray:
    """backs x 10 + forwards (31, 32, 41, 42, 51, 52); blank where the shape is not known."""
    d, f = np.asarray(defenders, float), np.asarray(forwards, float)
    return np.where(np.isfinite(d) & np.isfinite(f), np.clip(d, 3, 5) * 10 + np.where(f >= 2, 2, 1), np.nan)


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in SHAPE_COLS if c not in m.columns})
    s = sides(m, {"d": ("tq_def_h", "tq_def_a"), "f": ("tq_fwd_h", "tq_fwd_a"), "od": ("tq_def_a", "tq_def_h"),
                  "of": ("tq_fwd_a", "tq_fwd_h"), "xg": ("mclam_h", "mclam_a"), "xga": ("mclam_a", "mclam_h")})
    s["gae"] = (s["gf"] - s["ga"]) - (s["xg"] - s["xga"])
    s["kind"], s["opp_kind"] = kind(s["d"], s["f"]), kind(s["od"], s["of"])
    coll = collection_day(s["date"])
    # the kind the side last started (carried over matches with no line-up), as it stood at the collection day
    last = s.groupby("team")["kind"].shift(1).groupby(s["team"]).ffill()
    exp = at_collection(s, pd.DataFrame({"k": last}, index=s.index))["k"].to_numpy()
    prev = s.groupby("team")["kind"].shift(1).to_numpy()
    changed = np.where(np.isfinite(s["kind"]) & np.isfinite(prev), (s["kind"].to_numpy() != prev).astype(float), np.nan)
    ct, cn = prior_sum(s["team"], s["date"], coll, changed, OWN_WINDOW)
    # today's opponent's expected kind, on each side's row
    p = per_match(s, {"exp": exp})
    opp = s[["match_id", "venue"]].merge(p, on="match_id", how="left")
    opp_exp = np.where(opp["venue"] == "H", opp["exp_a"], opp["exp_h"])
    opp_backs = np.floor(s["opp_kind"].to_numpy() / 10)
    rec = split_record(s, "gae", np.where(np.isfinite(opp_backs), opp_backs, -1), (3.0, 4.0, 5.0), OWN_WINDOW)
    fit = np.full(len(s), np.nan)
    for b, (t, n) in rec.items():
        pick = (np.floor(opp_exp / 10) == b) & (n > 0)
        fit[pick] = (t / (n + OWN_SHRINK))[pick]
    q = per_match(s, {"exp": exp, "chg": np.where(cn >= 3, ct / np.maximum(cn, 1), np.nan), "fit": fit})
    out = q[["match_id"]].copy()
    out["fm_backs_diff"] = np.floor(q["exp_h"] / 10) - np.floor(q["exp_a"] / 10)
    out["fm_fwd_diff"] = q["exp_h"] % 10 - q["exp_a"] % 10
    out["fm_changes_diff"] = q["chg_h"] - q["chg_a"]
    out["fm_fit_diff"] = q["fit_h"] - q["fit_a"]
    # the table of kinds: home rows only, the pair as played against the pair expected today
    h = s[s["venue"] == "H"].reset_index(drop=True)
    hq = h[["match_id"]].merge(q[["match_id", "exp_h", "exp_a"]], on="match_id", how="left")
    played = (h["kind"] * 100 + h["opp_kind"]).to_numpy()
    expected = (hq["exp_h"] * 100 + hq["exp_a"]).to_numpy()
    table = np.full(len(h), np.nan)
    one, hcoll = np.zeros(len(h)), collection_day(h["date"])
    for k in np.unique(played[np.isfinite(played)]):
        t, n = prior_sum(one, h["date"], hcoll, np.where(played == k, h["gae"].to_numpy(float), np.nan), TABLE_WINDOW)
        pick = (expected == k) & (n > 0)
        table[pick] = (t / (n + TABLE_SHRINK))[pick]
    return out.merge(pd.DataFrame({"match_id": h["match_id"], "fm_table": table}), on="match_id", how="left")
