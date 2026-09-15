"""
Feature-stage registry — Part 9 / principle P5 of the racing² master framework.

    stage F  fundamental: market-free, legal in the fundamental model
    stage C  combination: built from odds / BSP / implied probability —
             only ever enters the market-combination stage
    stage S  selection / staking context (race-level constants)

"Putting BFSP-derived stats into Stage F double-counts the market and
destroys your ΔR² measurement." Enforce it in code: the Stage F matrix is
``stage_f_columns(cols)``, not developer discipline.

Ashcroft's existing metric code derives these from BSP:
    ORR2 family (odds-to-runner ratio), PFD, OFS/OFS1, and the
    expectation-residual family (expected_NFP, NFP_residual,
    career_residual, win_surprise, career_win_surprise), plus their
    within-race rank versions and anything from the Betfair price files
    (mkt_*, LR_mkt_*, horse_*mkt*, *_steam_*), the ABM's market-ability
    mode, and the Blandford in-play / BSP-advantage lags.

``leakage_audit`` is the tautology detector from Part 4.5: any Stage-F
feature whose race-demeaned correlation with ln(market probability) is
extreme is flagged for review.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

STAGE_C_EXACT = {
    "ORR2", "LR_ORR2", "preracehorsecareerORR2", "PFD", "OFS", "OFS1", "NFP_residual", "expected_NFP",
    "career_residual", "career_win_surprise", "win_surprise", "bfsp", "bfsp_place", "odds", "fav", "log_bfsp",
    "predicted_bfsp", "predicted_win_prob", "predicted_win_prob_norm", "overlay_pct",
    "LR_tf_ipmin_ratio", "tf_ipmin_3r", "LR_tf_bsp_adv",
}
STAGE_C_PATTERNS = [
    re.compile(r"^r?(ORR|OFS|PFD)\d*", re.I), re.compile(r"(^|_)mkt(_|$)"), re.compile(r"steam"), re.compile(r"bfsp|bsp", re.I),
    re.compile(r"(^|_)odds", re.I), re.compile(r"market", re.I), re.compile(r"_residual$|_surprise$|^expected_NFP$"),
    re.compile(r"^r(NFPResid|WinSurprise|ExpectedNFP|CareerResid)", re.I), re.compile(r"today_morning"),
    re.compile(r"ipmin|ip_low|ip_hit", re.I), re.compile(r"outperform", re.I),
]
# race-level constants: cancel in the softmax; use for segmentation / selection only
STAGE_S_PATTERNS = [
    re.compile(r"^(number_of_runners|dist_furlongs|race_class|going_description|prize_money|surface_type|race_type|track|"
               r"abm_pace_contest|abm_pace_collapse_p|abm_race_entropy|pred_race_pace|pred_n_front|pred_front_pct|pred_n_back|"
               r"pred_back_pct|pred_pace_scenario|pred_pace_spread|pred_max_front|RPS|pace_pressure|Race_avgOR|td_front_win_share|"
               r"td_holdup_win_share|td_avg_winner_pos|track_front_win_share|n_eff.*)$"),
]


def stage_of(name: str) -> str:
    if name in STAGE_C_EXACT or any(p.search(name) for p in STAGE_C_PATTERNS):
        return "C"
    if any(p.search(name) for p in STAGE_S_PATTERNS):
        return "S"
    return "F"


def classify(cols) -> pd.DataFrame:
    return pd.DataFrame({"feature": list(cols), "stage": [stage_of(c) for c in cols]})


def stage_f_columns(cols) -> list[str]:
    return [c for c in cols if stage_of(c) == "F"]


def stage_c_columns(cols) -> list[str]:
    return [c for c in cols if stage_of(c) == "C"]


def leakage_audit(df: pd.DataFrame, cols, price_col: str = "bfsp", race_col: str = "raceid", threshold: float = 0.9) -> pd.DataFrame:
    """Race-demeaned |corr| of each feature with ln(1/BSP); flags tautologies."""
    lp = -np.log(pd.to_numeric(df[price_col], errors="coerce").clip(1.01, None))
    lp = lp - lp.groupby(df[race_col]).transform("mean")
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        x = x - x.groupby(df[race_col]).transform("mean")
        ok = x.notna() & lp.notna()
        corr = float(np.corrcoef(x[ok], lp[ok])[0, 1]) if ok.sum() > 100 and x[ok].std() > 0 else np.nan
        rows.append({"feature": c, "stage": stage_of(c), "corr_with_ln_market": corr,
                     "flag": bool(abs(corr) >= threshold) if corr == corr else False})
    return pd.DataFrame(rows).sort_values("corr_with_ln_market", key=np.abs, ascending=False)
