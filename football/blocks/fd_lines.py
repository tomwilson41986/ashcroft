"""Betfair's handicap and goals prices against the market's, the handicap line's moves, and how many books priced the
match. ``fl_``.

racing² football research, 24 Nov 2026 (the owner: "work on adding this" - football-data.co.uk columns we used too
little). At the early price:

- ``fl_bfe_ah_gap``: Betfair's opening handicap chance for the home side less the bookmaker average's, both fair
  (two-way, margin split evenly), in log odds; ``fl_bfe_ou_gap``: the same for over 2.5 goals;
- ``fl_n_books``: the log of the number of bookmakers that priced the 1X2 (Betbrain's count to 2018/19, the
  bookmaker columns with an opening price after): a liquidity proxy; ``fl_n_ah``: Betbrain's handicap count (to
  2018/19 only).

From earlier matches, lagged to the collection day (each side's last 10, shrunk: sum / (n + 3)):

- ``fl_line_move_diff``: the handicap line's move from the open to the close, from the side's point of view (positive:
  the market moved towards the side), home less away;
- ``fl_ah_price_move_diff``: the side's handicap chance at the close less at the open (the same line), home less away;
- ``fl_line_moved_share``: the share of the league's matches in its last 200 whose line moved at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.blocks._util import collection_day, per_match, prior_sum, sides

FEATURES = ["fl_bfe_ah_gap", "fl_bfe_ou_gap", "fl_n_books", "fl_n_ah", "fl_line_move_diff", "fl_ah_price_move_diff",
            "fl_line_moved_share"]
OWN_WINDOW, SHRINK, LEAGUE_WINDOW = 10, 3.0, 200
NEED = ["bk_bfe_ahh", "bk_bfe_aha", "bk_avg_ahh", "bk_avg_aha", "bk_avg_o", "bk_avg_u", "bfe_open_o25",
        "bfe_open_u25", "bk_n_1x2", "bk_n_ah", "open_ah_line", "close_ah_line", "open_ah_h", "open_ah_a",
        "close_ah_h", "close_ah_a"]


def fair_logit(a, b) -> np.ndarray:
    """The first outcome's fair chance in log odds from two decimal prices (the margin split evenly)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = (a > 1.0) & (b > 1.0)
    p = np.where(ok, (1 / a) / (1 / np.where(ok, a, 2) + 1 / np.where(ok, b, 2)), np.nan)
    return np.log(p / (1 - p))


def build(m: pd.DataFrame) -> pd.DataFrame:
    m = m.assign(**{c: np.nan for c in NEED if c not in m.columns})
    out = m[["match_id"]].copy()
    out["fl_bfe_ah_gap"] = fair_logit(m["bk_bfe_ahh"], m["bk_bfe_aha"]) - fair_logit(m["bk_avg_ahh"], m["bk_avg_aha"])
    out["fl_bfe_ou_gap"] = fair_logit(m["bfe_open_o25"], m["bfe_open_u25"]) - fair_logit(m["bk_avg_o"], m["bk_avg_u"])
    out["fl_n_books"] = np.log(m["bk_n_1x2"].where(m["bk_n_1x2"] > 0))
    out["fl_n_ah"] = np.log(m["bk_n_ah"].where(m["bk_n_ah"] > 0))
    # what each match's handicap did, from the home side's view (a home line of -0.5 moving to -0.75: towards home)
    line_move = -(m["close_ah_line"] - m["open_ah_line"])
    same = m["close_ah_line"] == m["open_ah_line"]
    price_move = np.where(same, fair_logit(m["close_ah_h"], m["close_ah_a"]) - fair_logit(m["open_ah_h"], m["open_ah_a"]),
                          np.nan)
    s = sides(m.assign(_lm=line_move, _pm=price_move), {"lm": ("_lm", "_lm"), "pm": ("_pm", "_pm")})
    sign = np.where(s["venue"] == "H", 1.0, -1.0)
    coll = collection_day(s["date"])
    cols = {}
    for name, c in (("lm", "lm"), ("pm", "pm")):
        t, n = prior_sum(s["team"], s["date"], coll, sign * s[c].to_numpy(float), OWN_WINDOW)
        cols[name] = np.where(n > 0, t / (n + SHRINK), np.nan)
    p = per_match(s, cols)
    out = out.merge(p.assign(fl_line_move_diff=p["lm_h"] - p["lm_a"],
                             fl_ah_price_move_diff=p["pm_h"] - p["pm_a"])[
        ["match_id", "fl_line_move_diff", "fl_ah_price_move_diff"]], on="match_id", how="left")
    moved = np.where(m["close_ah_line"].notna() & m["open_ah_line"].notna(), (~same).astype(float), np.nan)
    t, n = prior_sum(m["competition_id"], m["date"], collection_day(m["date"]), moved, LEAGUE_WINDOW)
    out["fl_line_moved_share"] = np.where(n >= 20, t / np.maximum(n, 1), np.nan)
    return out
