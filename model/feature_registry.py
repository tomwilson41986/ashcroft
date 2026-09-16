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

Part 9 also asks every feature to carry eight fields beyond its name:
definition, source_tables, as_of_rule, stage, shrinkage, missing_policy,
transform, version. ``describe`` is the lookup; ``feature_dictionary``
renders the table; ``validate_dictionary`` is the Phase 0 exit test — it
names the features nobody has written an entry for, because a backtest is
only reproducible if the definitions it ran on are pinned.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

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
    # Part 4 odds-derived block (model/odds_metrics.py). OTRR = pi*N and its
    # variants, the implied-probability vectors, the favourite flags and the
    # book descriptors are all functions of the price and are Stage C by
    # construction; OFSR_* already matches the OFS rule above.
    re.compile(r"(^|_)(ln_)?OTRR"), re.compile(r"(^|_)OFSR"), re.compile(r"(^|_)pi_(raw|market|shin)$"),
    re.compile(r"(^|_)(is_fav|is_jt_fav|fav_rank)$"), re.compile(r"(^|_)(overround|n_priced)$"),
    re.compile(r"(^|_)shin(_|$)", re.I),
    # Part 4.4 price-path descriptors whose names carry no "mkt": traded-volume
    # share, the drift *rate* (spatial.py's going drift_smooth / drift_sd are a
    # different, market-free thing) and the pre-off volatility estimators.
    re.compile(r"(^|_)vol_share(_|$)"), re.compile(r"(^|_)drift_rate(_|$)"), re.compile(r"(^|_)pp_vol"),
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


# ---------------------------------------------------------------------------
# Part 9: the feature dictionary
# ---------------------------------------------------------------------------
#
# The dictionary is infrastructure, not paperwork. ``stage`` is the field that
# mechanically keeps the market out of Stage F; the other eight are what make a
# backtest reproducible — a 2019 re-run has to know which *version* of a
# definition it ran on, what timestamp bounded it, and what happened to the
# rows where it was missing. Entries are written once per feature *family*
# (the suffix/prefix grammar of the names does the rest) so the table stays
# maintainable across several hundred columns.

REGISTRY_VERSION = "3.1.0"      # bump when any definition below changes meaning

# the five as-of rules Ashcroft actually has; anything else is a bug
AS_OF = {
    "PRIOR_RUNS": "rows of this horse strictly before this race's off time",
    "PRIOR_RACES": "races of this group strictly before this race (model.lagsafe)",
    "PRIOR_WINDOW": "rows of this entity within the trailing window, strictly before this race",
    "CARD": "race card, known at declaration time (draw, age, sex, field size)",
    "MARKET": "market prices as at the quoted time (Stage C only)",
    "POST_RACE": "known only after the race has run — never a model input",
}

MISSING = {
    "ROUTE": "route: debutant / thin-record sub-model, no imputation",
    "ZERO": "impute 0 (the population mean of a centred quantity)",
    "PRIOR": "impute the shrinkage prior (parent-level mean)",
    "FLAG": "leave NaN and carry an explicit indicator column",
    "DROP": "drop the row from the fold",
    "EXCLUDE": "never enters the feature matrix",
}


@dataclass(frozen=True)
class FeatureSpec:
    """One row of the Part 9 dictionary."""

    name: str
    definition: str = ""
    source_tables: tuple[str, ...] = ()
    as_of_rule: str = ""
    stage: str = "F"
    shrinkage: str = "none"
    missing_policy: str = ""
    transform: str = "raw"
    version: str = ""
    origin: str = "unregistered"        # entry | family | unregistered

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source_tables"] = ",".join(self.source_tables)
        return d


def _spec(name: str, definition: str, as_of_rule: str, *, source_tables=("blandford_results",), shrinkage: str = "none",
          missing_policy: str = MISSING["FLAG"], transform: str = "raw", version: str = REGISTRY_VERSION) -> FeatureSpec:
    return FeatureSpec(name=name, definition=definition, source_tables=tuple(source_tables), as_of_rule=as_of_rule,
                       stage=stage_of(name), shrinkage=shrinkage, missing_policy=missing_policy, transform=transform,
                       version=version, origin="entry")


# Explicit entries: the Stage F block that train_stage_f.py and model.phase1 fit on.
FEATURE_DICTIONARY: dict[str, FeatureSpec] = {f.name: f for f in [
    _spec("kf_rating", "Kalman posterior-mean ability on performance_rating; theta_t = theta_{t-1} + w, y_t = theta_t + v (Section IIA.1)",
          AS_OF["PRIOR_RUNS"], shrinkage="Kalman gain from fitted q/r; wide prior on debut", missing_policy=MISSING["ROUTE"]),
    _spec("kf_sd", "sqrt of the Kalman prior variance P_t — the per-horse uncertainty that sizes the Kelly shrink",
          AS_OF["PRIOR_RUNS"], shrinkage="Kalman", missing_policy=MISSING["PRIOR"]),
    _spec("kf_n", "count of runs the filter has absorbed for this horse; the ratability gate of Section IV.1",
          AS_OF["PRIOR_RUNS"], missing_policy=MISSING["ZERO"], transform="raw"),
    _spec("kf_z", "kf_rating centred and scaled within race (Part 8.1)", AS_OF["PRIOR_RUNS"], shrinkage="Kalman", transform="z",
          missing_policy=MISSING["ROUTE"]),
    _spec("kf_vs_max", "kf_rating minus the race's best kf_rating", AS_OF["PRIOR_RUNS"], shrinkage="Kalman", transform="vs_max",
          missing_policy=MISSING["ROUTE"]),
    _spec("tf_master_z", "pre-race master rating, within-race z", AS_OF["CARD"], transform="z", missing_policy=MISSING["FLAG"]),
    _spec("tf_master_vs_max", "pre-race master rating minus the race best", AS_OF["CARD"], transform="vs_max",
          missing_policy=MISSING["FLAG"]),
    _spec("tfig_ewm_z", "exponentially weighted timefigure (270d half-life), within-race z", AS_OF["PRIOR_RUNS"],
          shrinkage="recency decay, lambda from half_life_days", transform="z", missing_policy=MISSING["FLAG"]),
    _spec("nmfp", "normalised finishing position, 1 for the winner to 0 for last (Part 1.4)", AS_OF["POST_RACE"],
          missing_policy=MISSING["EXCLUDE"]),
    _spec("perf_max3", "best performance_rating of the last 3 runs (Part 3.2 max aggregator)", AS_OF["PRIOR_RUNS"],
          missing_policy=MISSING["ROUTE"]),
    _spec("perf_min5", "worst performance_rating of the last 5 runs — the consistency floor", AS_OF["PRIOR_RUNS"],
          missing_policy=MISSING["ROUTE"]),
    _spec("perf_iqm5", "interquartile mean performance_rating over the last 5 runs — the typical run, outliers trimmed",
          AS_OF["PRIOR_RUNS"], missing_policy=MISSING["ROUTE"]),
    _spec("nmfp_mean3", "mean NMFP of the last 3 runs", AS_OF["PRIOR_RUNS"], missing_policy=MISSING["ROUTE"]),
    _spec("nmfp_max_career", "best career NMFP", AS_OF["PRIOR_RUNS"], missing_policy=MISSING["ROUTE"]),
    _spec("sos_vs_today", "today's class minus the mean pre-race master rating of the last 3 opponents' races (Part 2.3)",
          AS_OF["PRIOR_RUNS"], missing_policy=MISSING["ZERO"]),
    _spec("dslr_ln", "log1p(days since last run)", AS_OF["PRIOR_RUNS"], missing_policy=MISSING["FLAG"], transform="raw"),
    _spec("runs_count_ln", "log1p(number of prior career runs) — Benter's 'number of past races'", AS_OF["PRIOR_RUNS"],
          missing_policy=MISSING["ZERO"]),
    _spec("first_run", "1 if this is the horse's debut (runs_count == 0)", AS_OF["PRIOR_RUNS"], missing_policy=MISSING["ZERO"]),
    _spec("age_z", "age in years, within-race z", AS_OF["CARD"], transform="z", missing_policy=MISSING["FLAG"]),
    _spec("draw_pct", "(draw - 1) / (field size - 1), 0 = innermost stall", AS_OF["CARD"], transform="rankpct",
          missing_policy=MISSING["ZERO"]),
    _spec("is_female", "1 for fillies and mares", AS_OF["CARD"], missing_policy=MISSING["ZERO"]),
    _spec("draw_x_lnN", "draw_pct x ln(field size) — draw bias scales with how many rivals are inside (Part 6.5)",
          AS_OF["CARD"], transform="raw", missing_policy=MISSING["ZERO"]),
    _spec("draw_x_sprint", "draw_pct restricted to sprints (<= 6.5f), where the bias lives", AS_OF["CARD"],
          missing_policy=MISSING["ZERO"]),
    _spec("kf_sd_x_runs", "kf_sd x log1p(runs) — how far to trust the rating given how thin the record is",
          AS_OF["PRIOR_RUNS"], shrinkage="Kalman", missing_policy=MISSING["ZERO"]),
    _spec("lto_tfig_above_ability", "last-time-out timefigure minus the horse's ability estimate (LTO contradiction, Part 7.2)",
          AS_OF["PRIOR_RUNS"], missing_policy=MISSING["ZERO"]),
    _spec("lto_vs_career", "last-time-out NMFP minus career mean NMFP", AS_OF["PRIOR_RUNS"], missing_policy=MISSING["ZERO"]),
    _spec("dist_apt", "mean NMFP in today's distance band minus overall mean, shrunk by n/(n+2) (Part 8.4)",
          AS_OF["PRIOR_RUNS"], shrinkage="k = 2 toward the horse's overall mean", missing_policy=MISSING["ZERO"]),
    _spec("surf_apt", "mean NMFP on today's surface minus overall mean, shrunk by n/(n+2)", AS_OF["PRIOR_RUNS"],
          shrinkage="k = 2 toward the horse's overall mean", missing_policy=MISSING["ZERO"]),
    _spec("jockey_booking_upgrade", "today's jockey shrunk SR minus the horse's usual jockey SR — a stable-intent signal",
          AS_OF["PRIOR_RUNS"], shrinkage="k = 30 on each strike rate", missing_policy=MISSING["ZERO"]),
    _spec("pi_market", "overround-removed Betfair SP probability, proportional normalisation within race (Section IV.1)",
          AS_OF["MARKET"], source_tables=("blandford_results", "betfair_prices"), missing_policy=MISSING["DROP"],
          transform="share"),
]}

# Family rules: name grammar -> the fields that follow from it. First match wins,
# so put the specific patterns first.
FAMILY_RULES: list[tuple[re.Pattern, dict]] = [
    (re.compile(r"^(EPF\d?|NFP|placing_numerical|comment|early_pos|mid_move|late_move|finishing_effort|was_keen|had_trouble|"
                r"led_at_furlong|total_move|early_pos_pct)$"),
     {"definition": "in-running / finishing-position reading of THIS race", "as_of_rule": AS_OF["POST_RACE"],
      "missing_policy": MISSING["EXCLUDE"], "source_tables": ("race_results",)}),
    (re.compile(r"_form_(\d+)d$"),
     {"definition": "recency-weighted mean NMFP of this entity over the trailing window (Part 5)",
      "as_of_rule": AS_OF["PRIOR_WINDOW"], "shrinkage": "n/(n+6) toward 0", "missing_policy": MISSING["ZERO"]}),
    (re.compile(r"_form_vs_career$"),
     {"definition": "trailing-window form minus the entity's shrunk career mean", "as_of_rule": AS_OF["PRIOR_WINDOW"],
      "shrinkage": "n/(n+6) toward 0", "missing_policy": MISSING["ZERO"]}),
    (re.compile(r"_sr_shrunk$"),
     {"definition": "entity strike rate, (wins + k*prior)/(runs + k) (Part 8.4)", "as_of_rule": AS_OF["PRIOR_RACES"],
      "shrinkage": "k = 30 toward the population strike rate (k = 20 toward the trainer's own for tj_)",
      "missing_policy": MISSING["PRIOR"]}),
    (re.compile(r"_nmfp_shrunk$"),
     {"definition": "entity mean NMFP shrunk toward 0", "as_of_rule": AS_OF["PRIOR_RACES"], "shrinkage": "k = 15 toward 0",
      "missing_policy": MISSING["ZERO"]}),
    (re.compile(r"_runs$"),
     {"definition": "count of this entity's prior runs — carried alongside every shrunk rate (Part 8.4)",
      "as_of_rule": AS_OF["PRIOR_RACES"], "missing_policy": MISSING["ZERO"]}),
    (re.compile(r"^(perf|nmfp|timefigure|performance_rating)_.*_(career|L\d+)"),
     {"definition": "window aggregate (mean / IQM / max / min) of a performance primitive (Part 3.2)",
      "as_of_rule": AS_OF["PRIOR_RUNS"], "missing_policy": MISSING["ROUTE"]}),
    (re.compile(r"^LR_"),
     {"definition": "last-run value of the underlying metric", "as_of_rule": AS_OF["PRIOR_RUNS"],
      "missing_policy": MISSING["FLAG"], "source_tables": ("race_results",)}),
    (re.compile(r"^abm_"),
     {"definition": "Monte-Carlo race-simulator summary for this runner (model/abm)", "as_of_rule": AS_OF["CARD"],
      "missing_policy": MISSING["ZERO"], "source_tables": ("race_results",)}),
    (re.compile(r"_z$"), {"transform": "z", "definition": "within-race z of the underlying metric (Part 8.1)",
                          "as_of_rule": AS_OF["PRIOR_RUNS"], "missing_policy": MISSING["ZERO"]}),
    (re.compile(r"_vs_max$"), {"transform": "vs_max", "definition": "underlying metric minus the race maximum",
                               "as_of_rule": AS_OF["PRIOR_RUNS"], "missing_policy": MISSING["ZERO"]}),
    (re.compile(r"_rank$|_rankpct$|^r[A-Z]"), {"transform": "rankpct", "definition": "within-race rank of the underlying metric",
                                               "as_of_rule": AS_OF["PRIOR_RUNS"], "missing_policy": MISSING["ZERO"]}),
]


# Part 8's transformation layer is a suffix on an existing feature, so resolve
# it against the base rather than as a rule of its own: trainer_sr_shrunk_z is
# a within-race z *of a shrunk strike rate*, and should keep the shrinkage.
TRANSFORM_SUFFIXES = (("_rankpct", "rankpct"), ("_rank", "rankpct"), ("_vs_max", "vs_max"), ("_share", "share"), ("_z", "z"))


def describe(name: str) -> FeatureSpec:
    """The dictionary entry for one feature: the explicit entry if there is
    one, else the base feature's entry re-transformed, else the fields implied
    by its name grammar, else a bare spec whose empty ``version`` marks it as
    undocumented."""
    if name in FEATURE_DICTIONARY:
        return FEATURE_DICTIONARY[name]
    for suffix, transform in TRANSFORM_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            under = describe(name[: -len(suffix)])
            if under.origin != "unregistered":
                return replace(under, name=name, stage=stage_of(name), transform=transform)
    base = FeatureSpec(name=name, stage=stage_of(name), version=REGISTRY_VERSION, origin="family",
                       source_tables=("race_results", "blandford_results"))
    for pat, fields in FAMILY_RULES:
        if pat.search(name):
            return replace(base, **fields)
    if base.stage == "C":       # the stage detector is itself the entry for market-derived names
        return replace(base, definition="market-derived (STAGE_C_EXACT / STAGE_C_PATTERNS) — Stage C only",
                       as_of_rule=AS_OF["MARKET"], missing_policy=MISSING["DROP"], transform="raw")
    return FeatureSpec(name=name, stage=base.stage, origin="unregistered")


def feature_dictionary(cols) -> pd.DataFrame:
    """The Part 9 table for a feature list, one row per feature."""
    return pd.DataFrame([describe(c).to_dict() for c in cols])


REQUIRED_FIELDS = ("definition", "source_tables", "as_of_rule", "stage", "shrinkage", "missing_policy", "transform", "version")


def validate_dictionary(cols, required=REQUIRED_FIELDS, stage: str | None = None) -> pd.DataFrame:
    """Phase 0 exit test: every feature in ``cols`` has a dictionary entry with
    every required field filled.

    Returns the offending rows only — empty means the matrix is documented.
    ``missing_fields`` names what is absent; ``post_race`` marks a feature that
    is only known after the race and must not be in a model matrix at all;
    ``stage_mismatch`` marks a feature whose stage is not the one asked for
    (pass ``stage='F'`` when validating a Stage F matrix)."""
    rows = []
    for c in cols:
        s = describe(c)
        missing = [f for f in required if not getattr(s, f)]
        post_race = s.as_of_rule == AS_OF["POST_RACE"]
        mismatch = stage is not None and s.stage != stage
        if missing or post_race or mismatch:
            rows.append({"feature": c, "stage": s.stage, "origin": s.origin, "missing_fields": ",".join(missing),
                         "post_race": post_race, "stage_mismatch": mismatch, "version": s.version})
    return pd.DataFrame(rows, columns=["feature", "stage", "origin", "missing_fields", "post_race", "stage_mismatch", "version"])
