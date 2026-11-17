"""Today's race as a field (review section 5, race-context features, reports/greyhound_feature_review.md): how spread
the field's ratings are, the gap between the top two, how many runners the form book cannot read yet, how much early
pace there is, and where the dog stands in it.

The rating is the dog's decayed mean of the engine's race-time figure ``gsr`` (greyhound/metrics.py: lengths faster
than the track and trip's standard after the meeting's going; higher is faster) over its race runs on days strictly
before the race's day, half-life ``HL_RUNS`` runs (pricepath.history: merge_asof, allow_exact_matches False). The early
pace is the same over the engine's first-sectional figure ``esr`` (lengths faster than the sectional standard). Trials
and card rows add nothing to a dog's history. Debutants and dogs with fewer than 3 runs count earlier race days (any
race run, with or without a figure).

"Strong early pace" is a prior early-pace level at or above ``EP_Q`` of the levels seen in the first ``FIT_SHARE`` of
the days (one constant, fitted on that early slice only and reported in the notes); ``rx_n_ep_near_top`` needs no fit:
the runners within ``EP_NEAR`` lengths of the field's best prior early pace.

Card-safe: every column reads the dogs drawn in the race and their earlier days only. On a card the field is the
declared field (reserves included until they come out); in the historic build it is the field that ran, so a late
non-runner makes the two differ for that race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core
from greyhound.pit.pricepath import history

PREFIX = "rx_"
SOURCE = "GBGB results through the metrics engine's per-run figures (greyhound/metrics.py: gsr, esr)"
DEPENDS = ("pricepath",)
HL_RUNS = 3.0
EP_Q = 0.75
FIT_SHARE = 0.2
EP_NEAR = 1.0


def _top2(v: pd.Series, race: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The race's best and second-best of ``v`` (NaN ignored), aligned to ``v``."""
    rk = v.groupby(race).rank(ascending=False, method="first")
    top = v.where(rk == 1).groupby(race).transform("max")
    second = v.where(rk == 2).groupby(race).transform("max")
    return top, second


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    figs = core.per_run_figures(runs) if figs is None else figs
    racing = ~runs.is_trial.fillna(False).astype(bool)
    if "is_card" in runs:
        racing &= ~runs.is_card.fillna(False).astype(bool)
    per = pd.DataFrame({"dog_id": runs.dog_id, "race_date": runs.race_date}, index=runs.index)
    per["gsr"] = pd.to_numeric(figs.gsr, errors="coerce").to_numpy()
    per["esr"] = pd.to_numeric(figs.esr, errors="coerce").to_numpy()
    per.loc[~racing.to_numpy(), ["gsr", "esr"]] = np.nan
    per["one"] = np.where(racing, 1.0, np.nan)
    h = history(per, ["gsr", "esr", "one"], hl=HL_RUNS)
    race = runs.race_id
    rating, ep, nruns = h.gsr_ewm, h.esr_ewm, h.one_n
    out = pd.DataFrame({"race_id": runs.race_id, "dog_id": runs.dog_id}, index=runs.index)
    g = rating.groupby(race)
    out[f"{PREFIX}rating"] = rating
    out[f"{PREFIX}n_rated"] = g.transform("count").astype(float)
    out[f"{PREFIX}field_sd"] = g.transform("std")
    top, second = _top2(rating, race)
    out[f"{PREFIX}top2_gap"] = top - second
    out[f"{PREFIX}rating_rank"] = g.rank(ascending=False, method="min")
    out[f"{PREFIX}rating_vs_mean"] = rating - g.transform("mean")
    out[f"{PREFIX}gap_to_top"] = top - rating
    out[f"{PREFIX}n_debut"] = (nruns == 0).groupby(race).transform("sum").astype(float)
    out[f"{PREFIX}n_lt3"] = (nruns < 3).groupby(race).transform("sum").astype(float)
    # strong early pace: the early slice's quantile of the prior early-pace level (one constant)
    d = core.day(runs)
    days = np.sort(d.dropna().unique())
    cut = days[min(int(len(days) * FIT_SHARE), len(days) - 1)] if len(days) else None
    early = ep[(d <= cut)] if cut is not None else ep.iloc[:0]
    thr = float(early.quantile(EP_Q)) if early.notna().sum() >= 20 else np.nan
    ge = ep.groupby(race)
    ep_top, _ = _top2(ep, race)
    out[f"{PREFIX}ep"] = ep
    out[f"{PREFIX}ep_rank"] = ge.rank(ascending=False, method="min")
    out[f"{PREFIX}ep_gap_to_top"] = ep_top - ep
    out[f"{PREFIX}n_ep_strong"] = ((ep >= thr).groupby(race).transform("sum").astype(float)
                                   if np.isfinite(thr) else np.nan)
    out[f"{PREFIX}n_ep_near_top"] = (ep >= ep_top - EP_NEAR).groupby(race).transform("sum").astype(float) \
        .where(ep_top.notna())
    return out, {"rows": int(len(out)), "rating": "decayed mean of the engine's gsr (half-life 3 race runs)",
                 "ep_strong_threshold_lengths": None if not np.isfinite(thr) else round(thr, 4),
                 "ep_threshold_fitted_to": None if cut is None else str(pd.Timestamp(cut).date()),
                 "card_safe": "every column (the race's drawn field and its earlier days only)"}


COLUMNS = {
    f"{PREFIX}rating": "the dog's prior rating: decayed mean of the engine's gsr over its earlier race days (lengths)",
    f"{PREFIX}n_rated": "runners in the race with a prior rating",
    f"{PREFIX}field_sd": "the spread (sd) of the field's prior ratings",
    f"{PREFIX}top2_gap": "the best prior rating in the field less the second best",
    f"{PREFIX}rating_rank": "the dog's rank on prior rating in the field (1 best; NaN unrated)",
    f"{PREFIX}rating_vs_mean": "the dog's prior rating less the field's mean",
    f"{PREFIX}gap_to_top": "the field's best prior rating less the dog's",
    f"{PREFIX}n_debut": "runners with no earlier race run",
    f"{PREFIX}n_lt3": "runners with fewer than 3 earlier race runs (debutants included)",
    f"{PREFIX}ep": "the dog's prior early pace: decayed mean of the engine's esr over its earlier race days (lengths)",
    f"{PREFIX}ep_rank": "the dog's rank on prior early pace in the field (1 quickest)",
    f"{PREFIX}ep_gap_to_top": "the field's best prior early pace less the dog's",
    f"{PREFIX}n_ep_strong": "runners whose prior early pace is at or above the early slice's 75th percentile",
    f"{PREFIX}n_ep_near_top": "runners within 1 length of the field's best prior early pace (the early-pace pressure)",
}
