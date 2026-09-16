"""
Phase 1 of the racing² master framework, run end to end: an honest ΔR².

Given a results frame with pre-race ratings, positions and Betfair SP
(the Blandford / Timeform feed is enough), this builds a MARKET-FREE
Stage F feature set, fits the race-grouped conditional logit walk-forward,
fits Stage C on out-of-fold fundamentals, and reports:

    R²_market, R²_fundamental, R²_combined, ΔR² with a race-bootstrap CI
    Benter-style conditional calibration (model > market / model <= market)
    Kalman vs fixed-λ ratings on next-run performance
    ordering parameters γ, δ (Benter / Lo–Bacon-Shone) fitted on 1-2-3 finishes,
        globally and per field-size band (§IV.3 says they are not constants)
    ΔR² by OFSR_eff band — §4.3's claim that edge is easier to find in some
        market shapes than others, measured rather than assumed

Stage F features (all lag-safe or pre-race):
    kf_z            Kalman rating on the performance-rating history (within-race z)
    kf_sd           its uncertainty
    tf_master_z     pre-race master rating (within-race z), tf_master_vs_max
    tfig_ewm_z      recency-weighted timefigure
    perf_max3_z, perf_min5_z, perf_iqm5_z   best / consistency floor / typical
    nmfp_mean3_z, nmfp_max_career_z
    lto_tfig_above_ability, lto_vs_career
    sos_vs_today    class move vs recent strength of schedule
    dslr_ln, runs_count_ln, age, draw_pct, first_run

The Part 4 odds block (`model.odds_metrics`) is attached to the same frame —
pi_market, OTRR and its variants, N_eff, OFSR_eff, OFSR_conc — but it is
**Stage C**: it is listed in STAGE_C_ODDS_FEATURES, never in
STAGE_F_FEATURES, and `model.feature_registry.stage_of` classifies every one
of those names as "C" so the two lists cannot quietly merge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.odds_metrics import ODDS_FEATURES, add_odds_metrics
from model.ordering import fit_ordering_params_by_band
from model.primitives import (aggregate_history, lto_contradiction_features, nmfp, strength_of_schedule,
                              within_race_transforms)
from model.stage_f import conditional_calibration, delta_r2, mcfadden_r2, walk_forward_stage_f
from model.state_space import add_kalman_features, ewm_baseline, fit_kalman_params

STAGE_F_FEATURES = ["kf_z", "kf_sd", "kf_vs_max", "tf_master_z", "tf_master_vs_max", "tfig_ewm_z", "perf_max3_z", "perf_min5_z",
                    "perf_iqm5_z", "nmfp_mean3_z", "nmfp_max_career_z", "lto_tfig_above_ability", "lto_vs_career",
                    "sos_vs_today", "dslr_ln", "runs_count_ln", "age_z", "draw_pct", "first_run"]
#: Market-derived, Stage C only (Part 4). Never add any of these to STAGE_F_FEATURES.
STAGE_C_ODDS_FEATURES = list(ODDS_FEATURES)
#: The §4.3 selection variables: race-level, so they segment and filter rather
#: than entering the softmax, where they would cancel anyway.
SELECTION_VARIABLES = ["n_eff_market", "OFSR_eff", "OFSR_conc", "n_priced", "overround"]


def prepare_blandford_frame(bf: pd.DataFrame, flat_only: bool = True) -> pd.DataFrame:
    """Blandford feed rows -> race frame with raceid, won, market probability."""
    d = bf.copy()
    d = d[d["position"].notna() & (d["n_runners"] >= 4)]
    if flat_only and "race_type" in d.columns:
        d = d[d["race_type"].astype(str).str.lower().str.contains("flat")]
    d["race_date"] = pd.to_datetime(d["meeting_date"]); d["race_time"] = d["race_time"].fillna("00:00")
    d["raceid"] = d["meeting_date"].astype(str) + "_" + d["course_bf"].astype(str) + "_" + d["race_number"].astype(str)
    d["won"] = (d["position"] == 1).astype(float)
    # the feed codes missing ratings / figures as 0, and unrated pre-race ratings as 999
    for c in ("performance_rating", "pre_race_master_rating", "pre_race_adjusted_rating", "timefigure"):
        if c in d.columns:
            v = pd.to_numeric(d[c], errors="coerce").replace(0, np.nan)
            d[c] = v.where(v < 900)
    ok_race = d.groupby("raceid")["won"].transform("sum") == 1
    d = d[ok_race].copy()
    bsp = pd.to_numeric(d["betfair_win_sp"], errors="coerce").where(lambda s: s > 1.0)
    d["pi_market"] = (1 / bsp) / (1 / bsp).groupby(d["raceid"]).transform("sum")
    # a race is usable only if every runner has a BSP (else the market vector is not a distribution)
    d = d[d.groupby("raceid")["pi_market"].transform(lambda s: s.notna().all())].copy()
    # the whole Part 4 odds block: OTRR and its variants, N_eff, OFSR_eff /
    # OFSR_conc. Stage C — see STAGE_C_ODDS_FEATURES.
    d = add_odds_metrics(d, price_col="betfair_win_sp", race_col="raceid")
    d["nmfp"] = nmfp(d["n_runners"].values, d["position"].values)
    d["horse_name"] = d["horse_name"].astype(str)
    return d.sort_values(["race_date", "race_time", "raceid"]).reset_index(drop=True)


def build_stage_f_features(d: pd.DataFrame, kalman_fit_end: str, half_life_days: float = 270.0) -> tuple[pd.DataFrame, dict]:
    """Market-free features; Kalman q/r fitted on rows before ``kalman_fit_end`` only."""
    d = d.copy()
    fit_rows = d[d["race_date"] < pd.Timestamp(kalman_fit_end)]
    kf = fit_kalman_params(fit_rows, "performance_rating")
    d = add_kalman_features(d, "performance_rating", prefix="kf", q=kf["q"], r=kf["r"], init_var=kf["init_var"])
    d["ewm_perf"] = ewm_baseline(d, "performance_rating", half_life_days=half_life_days)
    d["tfig_ewm"] = ewm_baseline(d, "timefigure", half_life_days=half_life_days)
    d = aggregate_history(d, ["performance_rating", "nmfp", "timefigure"], windows=(3, 5), weight_col=None, aggs=("mean", "iqm", "max", "min"))
    d = lto_contradiction_features(d, nmfp_col="nmfp", tfig_col="timefigure", perf_col="performance_rating")
    d = strength_of_schedule(d, "pre_race_master_rating", window=3)
    g = d.groupby("horse_name")
    prev_date = g["race_date"].shift(1)
    d["dslr_ln"] = np.log1p((d["race_date"] - prev_date).dt.days.clip(lower=0))
    d["runs_count_ln"] = np.log1p(d["runs_count"]); d["first_run"] = (d["runs_count"] == 0).astype(float)
    d["draw_pct"] = (pd.to_numeric(d["draw"], errors="coerce") - 1) / (d["n_runners"] - 1).replace(0, np.nan)
    d = d.rename(columns={"performance_rating_max_L3": "perf_max3", "performance_rating_min_L5": "perf_min5",
                          "performance_rating_iqm_L5": "perf_iqm5", "nmfp_mean_L3": "nmfp_mean3"})
    d = within_race_transforms(d, ["pre_race_master_rating", "tfig_ewm", "perf_max3", "perf_min5", "perf_iqm5",
                                   "nmfp_mean3", "nmfp_max_career", "age"], suffixes=("z", "vs_max"))
    d = d.rename(columns={"pre_race_master_rating_z": "tf_master_z", "pre_race_master_rating_vs_max": "tf_master_vs_max"})
    return d, kf


def run_phase1(bf: pd.DataFrame, test_start: str = "2025-07-01", n_folds: int = 8, l2: float = 2.0, flat_only: bool = True) -> dict:
    d = prepare_blandford_frame(bf, flat_only=flat_only)
    d, kf = build_stage_f_features(d, kalman_fit_end=test_start)
    feats = [c for c in STAGE_F_FEATURES if c in d.columns]
    d = d[d["pi_market"].notna()].copy()
    # the walk-forward covers the whole span; we report on races from test_start on
    wf = walk_forward_stage_f(d, feats, "won", "raceid", "race_date", market_col="pi_market", l2=l2, n_folds=n_folds, min_train_races=300)
    te = (wf["race_date"] >= pd.Timestamp(test_start)) & wf["f_oof"].notna()
    out = {"kalman": kf, "features": feats, "n_races_total": int(d["raceid"].nunique())}
    y = wf.loc[te, "won"].values; g = wf.loc[te, "raceid"].values
    f = wf.loc[te, "f_oof"].values; pi = wf.loc[te, "pi_market"].values
    out["r2_market"] = mcfadden_r2(pi, y, g); out["r2_fundamental"] = mcfadden_r2(f, y, g)
    out["delta_fundamental_vs_market"] = delta_r2(f, pi, y, g)
    tc = te & wf["c_oof"].notna()
    if tc.sum() > 0:
        yc = wf.loc[tc, "won"].values; gc = wf.loc[tc, "raceid"].values
        out["r2_combined"] = mcfadden_r2(wf.loc[tc, "c_oof"].values, yc, gc)
        out["delta_combined_vs_market"] = delta_r2(wf.loc[tc, "c_oof"].values, wf.loc[tc, "pi_market"].values, yc, gc)
        out["conditional_calibration"] = conditional_calibration(wf.loc[tc, "f_oof"].values, wf.loc[tc, "pi_market"].values, yc, gc)
        out["n_races_combined"] = int(len(np.unique(gc)))
    out["n_races_test"] = int(len(np.unique(g)))
    # Kalman vs EWM on next-run performance (test period)
    m = te & wf["performance_rating"].notna() & (wf["kf_n"] >= 1) & wf["ewm_perf"].notna()
    out["next_run_mse_kalman"] = float(np.mean((wf.loc[m, "performance_rating"] - wf.loc[m, "kf_rating"]) ** 2))
    out["next_run_mse_ewm270"] = float(np.mean((wf.loc[m, "performance_rating"] - wf.loc[m, "ewm_perf"]) ** 2))
    out["next_run_mse_master_rating"] = float(np.mean((wf.loc[m, "performance_rating"] - wf.loc[m, "pre_race_master_rating"]).dropna() ** 2))
    # ordering parameters on the market vector (positions 1-2-3)
    races = []
    for rid, grp in wf[te].groupby("raceid"):
        p = grp["pi_market"].values; pos = grp["position"].values
        if len(p) >= 4 and np.isfinite(p).all() and {1, 2, 3} <= set(pos):
            races.append((p, (int(np.where(pos == 1)[0][0]), int(np.where(pos == 2)[0][0]), int(np.where(pos == 3)[0][0]))))
    if len(races) > 200:
        fit = fit_ordering_params_by_band(races[:6000], min_races=200)
        out["ordering"] = fit["global"]
        out["ordering_by_field_band"] = fit["bands"]
    if "OFSR_eff" in wf.columns and tc.sum() > 0:
        out["delta_r2_by_ofsr_eff"] = delta_r2_by_band(wf[tc], n_bands=3)
    out["frame"] = wf
    return out


def delta_r2_by_band(wf: pd.DataFrame, band_col: str = "OFSR_eff", n_bands: int = 3, race_col: str = "raceid") -> pd.DataFrame:
    """ΔR² of the combined model over the market, split by a race-level market
    descriptor (§4.3).

    The spec's claim is that some market shapes are unplayable and others are
    tremendously +EV; OFSR_eff — the share of the field that is genuinely live
    — is the variable it names. This measures the claim instead of asserting
    it, so bet selection can be set from the answer.
    """
    d = wf[[race_col, "won", "pi_market", "c_oof", band_col]].dropna()
    if d.empty:
        return pd.DataFrame()
    per_race = d.groupby(race_col)[band_col].first()
    try:
        bands = pd.qcut(per_race, n_bands, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    d = d.join(bands.rename("_band"), on=race_col)
    rows = []
    for band, grp in d.groupby("_band", observed=True):
        y = grp["won"].values; g = grp[race_col].values
        rows.append({"band": str(band), "n_races": int(len(np.unique(g))), "n_runners": int(len(grp)),
                     f"mean_{band_col}": float(grp[band_col].mean()),
                     "r2_market": mcfadden_r2(grp["pi_market"].values, y, g),
                     "r2_combined": mcfadden_r2(grp["c_oof"].values, y, g),
                     "delta_r2": mcfadden_r2(grp["c_oof"].values, y, g) - mcfadden_r2(grp["pi_market"].values, y, g)})
    return pd.DataFrame(rows)
