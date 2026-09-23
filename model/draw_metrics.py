"""
Draw Bias & Stall Position Feature Engineering.

Extends the existing minimal draw features (draw_relative, draw_quartile)
with a comprehensive framework capturing:

1. **Track+Distance Draw Bias** — Lag-safe expanding mean of low vs high stall
   NFP at each track+distance combination.
2. **Going-Adjusted Draw Bias** — How ground conditions shift the draw advantage.
3. **Field-Size Scaling** — Bigger fields amplify draw effects.
4. **Stall x Pace Interaction** — Draw matters differently for front-runners
   vs hold-up horses. Low stalls enable front-running but the advantage is
   concentrated on prominent/midfield runners.
5. **Horse Personal Draw Preference** — Does this horse perform better from
   low or high stalls? (lag-safe career stats)
6. **Draw Advantage Score** — Composite: how much does this stall help/hurt
   given the track, distance, going, and field size?

Framework §6.3–§6.5 add four things the low/high split above cannot express:

7. **`draw_adj`** — the stall renumbered after the non-runners are taken out.
   Stall 8 with three withdrawals inside it is stall 5 out of the gate, and
   every draw feature here is built on the adjusted number, not the card one.
8. **Expected-versus-actual bias.** §6.3 is explicit that estimating draw bias
   from raw finishing positions by stall band is "the standard trap": a course
   where the low stalls happen to draw the better horses looks biased when it
   is not. The bias here is ``mean(nmfp) − mean(expected nmfp from ratings)``,
   so what is left is the part the horses do not explain. The expectation is
   the field ranked by rating, scaled by a lag-safe estimate of how well
   ratings have actually ordered a field to date, so a rating that predicts
   nothing shrinks the expectation to nothing rather than inventing a bias.
9. **A per-stall surface with hierarchical shrinkage.** Stall 1 and stall 7 of
   a fourteen-runner field are not the same thing. Estimates are formed per
   stall and pooled, thin cell toward thick: the §6.3 cell
   (course × distance × going × field-size band × stall) shrinks toward
   course × distance × field band × stall, then toward course × distance ×
   stall, then toward course × draw quartile, then toward zero.
10. **Time decay and a confidence flag.** §6.3's guard: draw bias "reverses
   with rail movement, watering and going changes, and has weakened over time".
   Every cell mean is weighted by ``2 ** (−age / half-life)``, carries the
   resulting effective sample size, and is marked unconfident when the sample
   is thin or the rail has been moved for today's meeting.

`track_direction` (handedness) and `rail_move` are read here — they are scraped
and stored, and before this nothing in the feature path touched them. Handedness
is a first-order determinant of which side of the track is the inside, and rail
movement is the main reason a course's bias is not stationary.

Lag safety: a track, a track+distance and a track+distance+going key all repeat
inside a race, so every cell statistic goes through `race_lagged_expanding_mean`
or `race_lagged_decayed_mean`, which aggregate to races before lagging. A
row-wise ``shift(1)`` on those keys steps back to a rival in the same race.

The per-run columns that need the result of the race being predicted —
`nmfp_actual`, `draw_resid` and the split-half screen columns — are listed in
`DRAW_POST_RACE_ONLY` and never appear in the exported feature lists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import lagsafe
from model.lagsafe import ensure_race_key, race_lagged_expanding_mean
from model.primitives import nmfp

#: A track x distance x going cell needs this many earlier races before its draw
#: bias is reported at all; below it the estimate is left NaN.
MIN_CELL_RACES = 5

#: Half-life in days of the weight on a past race in any draw-bias cell (§6.3:
#: "any draw feature must be time-decayed"). Two years: long enough to fill a
#: per-stall cell, short enough that a resurfacing or a permanent rail change
#: washes out of the estimate within a handful of seasons.
DRAW_BIAS_HALFLIFE_DAYS = 730.0

#: Shrinkage strengths, in effective runners, for the hierarchy in §6.3. Each is
#: the sample size at which a level's own estimate carries half the weight and
#: the level above it carries the other half. Heavy on purpose: "most
#: course/distance/going/field-size cells are thin".
SHRINK_K_COURSE = 60.0     # course x draw quartile, shrunk toward zero
SHRINK_K_STALL = 40.0      # course x distance x stall, toward the course level
SHRINK_K_FIELD = 30.0      # + field-size band, toward course x distance x stall
SHRINK_K_CELL = 20.0       # + going (the §6.3 cell), toward the field band level

#: Effective sample size at which a per-stall estimate is called confident.
MIN_CONFIDENT_N_EFF = 30.0

#: Yards of rail movement that halve the trust in a stored draw bias.
RAIL_TRUST_YARDS = 6.0

#: Field-size bands for the §6.3 cell.
FIELD_SIZE_BAND_EDGES = (0, 7, 11, 15, 99)

#: Stalls above this are pooled: they exist only in a handful of huge fields.
MAX_STALL = 24

#: A race whose split-half gap exceeds this multiple of its course/distance
#: baseline is flagged for the full expected-vs-actual study (§6.4: "a fast
#: indicator, not a full bias study").
SPLIT_HALF_FLAG_MULT = 1.5

#: Ratings used to build the expected finishing position, best coverage first.
#: `official_rating` is the handicap mark, known at declaration; the career NFP
#: is the lag-safe fallback for races where few runners carry a mark.
RATING_COLS = ("official_rating", "preracehorsecareerNFP")

#: A rating must cover this share of the field before it is used for that race.
MIN_RATED_SHARE = 0.6

#: Races needed before the rating->finishing-position calibration is trusted.
#: Below it no residual is emitted at all: with no idea how well ratings order a
#: field, "actual minus expected" is not a measurement.
RATING_SLOPE_MIN_RACES = 10

#: The calibration slope is clipped here — it is a regression slope through the
#: origin on bounded data, and a wild value means a thin or degenerate window.
RATING_SLOPE_CLIP = (0.0, 1.5)

#: Columns below describe the race being predicted. They exist so the lagged
#: cell statistics can be built and must never reach a model.
DRAW_POST_RACE_ONLY = frozenset({
    "nmfp_actual", "draw_resid", "dsh_front", "dsh_back", "dsh_diff",
    "dsh_diff_norm", "dsh_level", "dsh_abnormal",
})


# ---------------------------------------------------------------------------
# Course descriptors: going bucket, handedness, rail movement
# ---------------------------------------------------------------------------

def classify_going(going: object) -> str:
    """Bucket a free-text going description into the cell label."""
    if not isinstance(going, str):
        return "Unknown"
    g = going.lower()
    if "heavy" in g:
        return "Heavy"
    if "soft" in g and "good" not in g:
        return "Soft"
    if "good to soft" in g or "yielding" in g:
        return "GtS"
    if "good to firm" in g:
        return "GtF"
    if "firm" in g and "good" not in g:
        return "Firm"
    if "standard" in g or "slow" in g or "fast" in g:
        return "AW"
    return "Good"


def parse_track_direction(values) -> pd.Series:
    """Handedness as +1 left-handed, −1 right-handed, 0 straight, NaN unknown.

    The column is free text from the card ("Left", "L/H", "Right-handed",
    "Straight"), and a signed code is what an interaction term wants: which way
    the field turns decides which end of the stalls is the inside.
    """
    s = pd.Series(values).astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[s.str.contains(r"^(?:l|lh|l/h|left)", regex=True, na=False)] = 1.0
    out[s.str.contains(r"^(?:r|rh|r/h|right)", regex=True, na=False)] = -1.0
    out[s.str.contains(r"straight|^s$", regex=True, na=False)] = 0.0
    return out


def parse_rail_move(values) -> tuple[pd.Series, pd.Series]:
    """(yards moved, moved-at-all flag) from the free-text rail column.

    Signed: positive = rail out from the true line (the inside ground is being
    rested, so the inner is longer and usually slower), negative = rail in.
    A blank, "nil" or "no change" is zero yards and not moved.
    """
    s = pd.Series(values).astype(str).str.strip().str.lower()
    blank = s.isin(["", "nan", "none", "nil", "no", "n/a", "-", "0", "no change", "true line"])
    num = s.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False).astype(float)
    inward = s.str.contains(r"\bin\b|inward|inside", regex=True, na=False) & \
        ~s.str.contains(r"\bout\b|outward", regex=True, na=False)
    yards = num.abs() * np.where(inward, -1.0, 1.0)
    yards = yards.where(~blank, 0.0).fillna(0.0)
    moved = ((~blank) & (yards.abs() > 0)).astype(float)
    return yards, moved


# ---------------------------------------------------------------------------
# Time-decayed, race-lagged cell means
# ---------------------------------------------------------------------------

def _race_when(df: pd.DataFrame) -> np.ndarray:
    """Race time as a float number of days, so same-day cards keep their order."""
    d = pd.to_datetime(df["race_date"], errors="coerce")
    day = d.values.astype("datetime64[D]").astype(float)
    if "race_time" in df.columns:
        hm = df["race_time"].astype(str).str.extract(r"(\d{1,2}):(\d{2})")
        mins = pd.to_numeric(hm[0], errors="coerce") * 60 + pd.to_numeric(hm[1], errors="coerce")
        day = day + np.nan_to_num(mins.to_numpy(dtype=float), nan=0.0) / 1440.0
    return np.nan_to_num(day, nan=0.0)


def _codes(values) -> np.ndarray:
    """Integer codes for a key column, missing values getting their own code."""
    codes = pd.factorize(values)[0].astype(np.int64)
    if (codes < 0).any():
        codes = np.where(codes < 0, codes.max() + 1, codes)
    return codes


def race_lagged_decayed_mean(df: pd.DataFrame, group_col: str | list[str], value_col: str,
                             halflife_days: float = DRAW_BIAS_HALFLIFE_DAYS,
                             race_col: str = "raceid",
                             min_races: int = 0) -> tuple[pd.Series, pd.Series]:
    """Time-decayed mean of `value_col` over *earlier races* in the group.

    Returns ``(mean, n_eff)``: a past race counts ``2 ** (−age_days /
    halflife_days)`` of an observation, and ``n_eff`` is the sum of those
    weights — the effective sample size the shrinkage and the confidence flag
    are built on. ``halflife_days=inf`` gives a plain expanding mean, and
    ``min_races`` withholds an estimate until the group has that many earlier
    races behind it.

    Same lagging rule as `model.lagsafe.race_lagged_expanding_mean`, and the
    same reason for it: the keys used here (course, course+distance,
    course+distance+going, and per stall) repeat inside a race, so the frame is
    aggregated to races first and each race then steps back a whole race, never
    a row.

    Written on integer codes and flat numpy rather than a groupby-and-merge:
    the finest cell here has tens of thousands of groups and this runs on every
    row of the history, four times over.
    """
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    idx = df.index
    if len(df) == 0:
        empty = pd.Series(dtype=float, index=idx)
        return empty, empty.copy()

    if len(keys) == 1:
        gcode = _codes(df[keys[0]])
    else:
        gcode = _codes(pd.MultiIndex.from_arrays([df[k] for k in keys]))
    race = _codes(ensure_race_key(df, race_col))
    when = _race_when(df)
    # The cell a prior steps back by: a whole day under the default lag rule
    # (see model.lagsafe), so nothing from the card being priced enters it.
    by_day = lagsafe.LAG_UNIT == "day"
    if by_day:
        when = np.floor(when)
        rcode = _codes(when)
    else:
        rcode = race

    # one cell per (group, unit): a race (or a day) contributes to a group once
    n_r = int(rcode.max()) + 1
    pair = gcode * np.int64(n_r) + rcode
    upair, pidx = np.unique(pair, return_inverse=True)
    m = len(upair)

    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(v)
    sums = np.bincount(pidx[ok], weights=v[ok], minlength=m)
    cnts = np.bincount(pidx[ok], minlength=m).astype(float)
    if by_day:                                  # races per cell, so min_races still counts races
        rp = np.unique(pidx.astype(np.int64) * np.int64(int(race.max()) + 1) + race)
        races_in_cell = np.bincount(rp // np.int64(int(race.max()) + 1), minlength=m).astype(float)
    else:
        races_in_cell = np.ones(m)
    # earliest time seen for each (group, race); every row of a race carries the
    # same one, and taking the minimum keeps the result independent of row order
    # if a file ever disagrees with itself.
    o = np.lexsort((when, pidx))
    p_sorted = pidx[o]
    firsts = np.empty(len(p_sorted), dtype=bool)
    firsts[0] = True
    firsts[1:] = p_sorted[1:] != p_sorted[:-1]
    pwhen = np.zeros(m)
    pwhen[p_sorted[firsts]] = when[o][firsts]
    pgroup = upair // np.int64(n_r)

    order = np.lexsort((upair, pwhen, pgroup))
    g_s, w_s, s_s, c_s = pgroup[order], pwhen[order], sums[order], cnts[order]
    r_s = races_in_cell[order]

    is_new = np.empty(m, dtype=bool)
    is_new[0] = True
    is_new[1:] = g_s[1:] != g_s[:-1]
    starts = np.maximum.accumulate(np.where(is_new, np.arange(m), 0))

    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip((w_s - w_s[starts]) / float(halflife_days), 0.0, 300.0)
    t = np.nan_to_num(t, nan=0.0)
    grow = np.exp2(t)
    ws, wn = grow * s_s, grow * c_s
    cs, cn = np.cumsum(ws), np.cumsum(wn)
    prior_ws = cs - ws - (cs[starts] - ws[starts])       # exclusive, within group
    prior_wn = cn - wn - (cn[starts] - wn[starts])
    cr = np.cumsum(r_s)
    prior_races = cr - r_s - (cr[starts] - r_s[starts])

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_s = np.where((prior_wn > 0) & (prior_races >= min_races), prior_ws / prior_wn, np.nan)
    neff_s = np.exp2(-t) * prior_wn

    mean_p, neff_p = np.empty(m), np.empty(m)
    mean_p[order], neff_p[order] = mean_s, neff_s
    return (pd.Series(mean_p[pidx], index=idx), pd.Series(neff_p[pidx], index=idx))


def _shrink(raw: pd.Series, n_eff: pd.Series, prior, k: float) -> pd.Series:
    """Empirical-Bayes pull of a cell mean toward the level above it."""
    prior = pd.Series(prior, index=raw.index) if not isinstance(prior, pd.Series) else prior
    n = n_eff.fillna(0.0)
    val = raw.fillna(0.0) * n + prior.fillna(0.0) * k
    return (val / (n + k)).where(n + k > 0, prior)


# ---------------------------------------------------------------------------
# Expected finishing position from ratings (§6.3)
# ---------------------------------------------------------------------------

def expected_nmfp_from_ratings(df: pd.DataFrame, rating_cols=RATING_COLS,
                               min_rated_share: float = MIN_RATED_SHARE,
                               race_col: str = "raceid") -> tuple[pd.Series, pd.Series]:
    """Rating-implied NMFP, plus the name of the rating each race used.

    The field is ranked by the best-covered rating and each runner is given the
    NMFP of the position that rank implies; unrated runners in a usable race sit
    at the middle of the field. A race where no rating covers
    ``min_rated_share`` of the runners gets NaN — it can say nothing about draw
    bias, and guessing would put the trap §6.3 warns about straight back in.

    NMFP rather than the repo's NFP because it is the framework's primitive for
    this and is centred within a race by construction, which makes the residual
    below a clean per-race contrast.
    """
    rk = ensure_race_key(df, race_col)
    n = pd.to_numeric(df["number_of_runners"], errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    src = pd.Series(np.nan, index=df.index, dtype=object)
    for col in rating_cols:
        if col not in df.columns:
            continue
        r = pd.to_numeric(df[col], errors="coerce")
        cover = r.notna().astype(float).groupby(rk).transform("mean")
        rank = r.groupby(rk).rank(ascending=False, method="average")
        rank = rank.fillna((n + 1) / 2.0)
        cand = pd.Series(nmfp(n.values, rank.values), index=df.index)
        take = out.isna() & (cover >= min_rated_share) & cand.notna()
        out = out.where(~take, cand)
        src = src.where(~take, col)
    return out, src


def rating_calibration_slope(df: pd.DataFrame, expected_col: str, actual_col: str,
                             min_races: int = RATING_SLOPE_MIN_RACES,
                             clip: tuple[float, float] = RATING_SLOPE_CLIP) -> pd.Series:
    """How much of the rating-implied finishing position actually materialises.

    The least-squares slope through the origin of actual NMFP on rating-implied
    NMFP, over every race run before this one. Both series are within-race
    centred by construction, so the slope is ``Σxy / Σx²``.

    Without it, "expected" would assume ratings order a field perfectly. They do
    not, so the raw expectation is over-dispersed, and in a cell where the low
    stalls held the better horses the over-dispersion would come back out as a
    draw bias of the opposite sign — the same trap, mirrored.
    """
    d = df[[expected_col, actual_col]].copy()
    d["_all"] = "all"
    d["race_date"] = df["race_date"]
    if "race_time" in df.columns:
        d["race_time"] = df["race_time"]
    if "raceid" in df.columns:
        d["raceid"] = df["raceid"]
    x = pd.to_numeric(d[expected_col], errors="coerce")
    y = pd.to_numeric(d[actual_col], errors="coerce")
    d["_xy"] = x * y
    d["_xx"] = x * x
    xy, _ = race_lagged_decayed_mean(d, "_all", "_xy", halflife_days=np.inf, min_races=min_races)
    xx, _ = race_lagged_decayed_mean(d, "_all", "_xx", halflife_days=np.inf, min_races=min_races)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = xy / xx.where(xx > 0)
    return slope.clip(*clip)


# ---------------------------------------------------------------------------
# §6.4 draw-bias split-half screen
# ---------------------------------------------------------------------------

def draw_bias_split_half(df: pd.DataFrame, stall_col: str = "draw_adj",
                         finish_col: str = "placing_numerical",
                         race_col: str = "raceid") -> pd.DataFrame:
    """The §6.4 split-half screen, one row per race.

    Split the field in half by finishing position; in each half take
    ``δ_i = stall_i − finish_pos_i`` and report ``√(Σ δ_i²) / n_half``. A large
    difference between the halves says the draw sorted one end of the result and
    not the other. ``dsh_level`` is the same statistic over the whole field: it
    is small when stall order and finish order agree (the draw decided the race)
    and large when they are reversed.

    The odd-field rule is the framework's: the median horse is reused in both
    halves, so each half holds ``N/2 + 1`` horses and that is the divisor. The
    general formula in §6.4 divides by N; dividing each half by its own size is
    the reading that makes the odd-field rule consistent, and it is the only one
    that leaves the two halves comparable.

    A screen over completed races, not a feature: the stall-versus-finish
    contrast needs the finishing positions of the race it describes.
    """
    d = pd.DataFrame({
        "_race": ensure_race_key(df, race_col),
        "_stall": pd.to_numeric(df[stall_col], errors="coerce") if stall_col in df.columns else np.nan,
        "_fin": pd.to_numeric(df[finish_col], errors="coerce") if finish_col in df.columns else np.nan,
    }).dropna()
    if d.empty:
        return pd.DataFrame(columns=["raceid", "n", "dsh_front", "dsh_back", "dsh_diff",
                                     "dsh_diff_norm", "dsh_level"])
    g = d.groupby("_race", sort=False)
    d["_n"] = g["_fin"].transform("size")
    d["_rank"] = g["_fin"].rank(method="first")
    d["_d2"] = (d["_stall"] - d["_fin"]) ** 2
    front = d["_rank"] <= np.ceil(d["_n"] / 2.0)
    back = d["_rank"] >= np.floor(d["_n"] / 2.0) + 1
    d["_f2"] = d["_d2"].where(front)
    d["_b2"] = d["_d2"].where(back)
    agg = d.groupby("_race", sort=False).agg(
        n=("_fin", "size"), _all=("_d2", "sum"),
        _fs=("_f2", "sum"), _fn=("_f2", "count"),
        _bs=("_b2", "sum"), _bn=("_b2", "count"),
    ).reset_index()
    agg["dsh_front"] = np.sqrt(agg["_fs"]) / agg["_fn"].replace(0, np.nan)
    agg["dsh_back"] = np.sqrt(agg["_bs"]) / agg["_bn"].replace(0, np.nan)
    agg["dsh_diff"] = agg["dsh_front"] - agg["dsh_back"]
    agg["dsh_level"] = np.sqrt(agg["_all"]) / agg["n"]
    # scale-free version, so courses with different field sizes are comparable
    agg["dsh_diff_norm"] = agg["dsh_diff"] / (agg["n"] - 1).replace(0, np.nan)
    return agg.rename(columns={"_race": "raceid"})[
        ["raceid", "n", "dsh_front", "dsh_back", "dsh_diff", "dsh_diff_norm", "dsh_level"]
    ]


class DrawMetricsEngine:
    """Advanced draw/stall feature engineering.

    Call ``calculate(df)`` after NFP, EPF, draw_relative and draw_quartile
    have been computed (i.e. after _calc_draw_bias in CustomMetricsEngine).

    ``gp_draw_surface=True`` adds the Gaussian-process per-stall surface from
    `model.spatial` (columns `GP_DRAW_FEATURES`) on top of the closed-form
    hierarchy below. It borrows strength between *adjacent* stalls, which the
    cell hierarchy cannot do, and costs one GP fit per course per period — off
    by default because the closed-form surface is what the daily pipeline needs
    and this is not cheap on a full history.
    """

    def __init__(self, gp_draw_surface: bool = False, gp_keys=("track", "_gp_dist"),
                 gp_freq: str = "Y") -> None:
        self.gp_draw_surface = gp_draw_surface
        self.gp_keys = tuple(gp_keys)
        self.gp_freq = gp_freq

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all draw feature calculations in order."""
        df = df.copy()

        # Ensure prerequisites
        if "stall_num_raw" not in df.columns:
            df["stall_num_raw"] = pd.to_numeric(df.get("stall"), errors="coerce")
        if "draw_relative" not in df.columns:
            nr = df["number_of_runners"].replace(0, np.nan)
            df["draw_relative"] = (df["stall_num_raw"] - 1) / (nr - 1).replace(0, np.nan)

        # Identify flat races (stall data only meaningful for flat)
        df["_is_flat"] = (df["stall_num_raw"] > 0) & df["stall_num_raw"].notna()

        # Step 0a: the stall renumbered after the non-runners (§6.3 draw_adj)
        df = self._calc_draw_adj(df)

        # Step 0b: handedness and rail movement
        df = self._calc_track_geometry(df)

        # Step 1: Track+distance draw bias
        df = self._calc_track_distance_draw_bias(df)

        # Step 2: Going-adjusted draw bias
        df = self._calc_going_draw_bias(df)

        # Step 3: Field-size draw scaling
        df = self._calc_field_size_draw(df)

        # Step 3b: expected-vs-actual per-stall bias surface (§6.3)
        df = self._calc_expected_vs_actual_bias(df)

        # Step 3c: the §6.4 split-half screen and its course baselines
        df = self._calc_split_half_screen(df)

        # Step 3d (opt-in): GP smoothing across adjacent stalls
        if self.gp_draw_surface:
            df = self._calc_gp_draw_surface(df)

        # Step 4: Stall x pace interaction
        df = self._calc_stall_pace_interaction(df)

        # Step 5: Horse personal draw preference
        df = self._calc_horse_draw_preference(df)

        # Step 5b: the §6.5 draw x position interactions
        df = self._calc_spec_interactions(df)

        # Step 6: Draw advantage composite
        df = self._calc_draw_advantage(df)

        # Step 7: Within-race draw rankings
        df = self._calc_draw_ranks(df)

        # Cleanup
        df.drop(columns=["_is_flat"], errors="ignore", inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 0a: draw adjusted for non-runners (§6.3)
    # ------------------------------------------------------------------
    def _calc_draw_adj(self, df: pd.DataFrame) -> pd.DataFrame:
        """`draw_adj`: the stall's rank among the horses that actually ran.

        Stall 8 with three withdrawals inside it leaves the gate fifth, and on a
        course where the inside is worth something it is a stall-5 draw, not a
        stall-8 draw. `draw_relative` off the card number has that wrong for
        every race with a non-runner, which in British racing is most of them.

        ``draw_vacancy_inside`` (card stall − adjusted stall) is the count of
        non-runners drawn inside this horse, and is informative in its own right:
        it is also the space the horse has to cross.
        """
        rk = ensure_race_key(df)
        stall = pd.to_numeric(df["stall_num_raw"], errors="coerce").where(df["_is_flat"])
        nr = pd.to_numeric(df["number_of_runners"], errors="coerce").replace(0, np.nan)

        df["draw_adj"] = stall.groupby(rk).rank(method="min")
        df["draw_pct_adj"] = ((df["draw_adj"] - 1) / (nr - 1)).clip(0, 1)
        # fall back to the card-number version where the field is not all here
        df["draw_pct_adj"] = df["draw_pct_adj"].fillna(df["draw_relative"])
        df["draw_vacancy_inside"] = stall - df["draw_adj"]
        df["draw_nr_shift"] = df["draw_relative"] - df["draw_pct_adj"]
        return df

    # ------------------------------------------------------------------
    # Step 0b: handedness and rail movement
    # ------------------------------------------------------------------
    def _calc_track_geometry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Read `track_direction` and `rail_move`, which nothing else does.

        Handedness decides which end of the stalls is the inside, so the draw
        effect has no reason to share a sign across left- and right-handed
        courses: the interaction is what carries the information, and the sign
        is fitted, not assumed (§6.5).

        Rail movement is the main reason a course's draw bias is not stationary:
        the ground everyone raced on last season is not the ground being raced
        on today. `rail_trust` is the multiplier the stored bias is worth when
        the rail has moved, and it also gates the confidence flag.
        """
        df["track_handed"] = parse_track_direction(
            df.get("track_direction", pd.Series(np.nan, index=df.index))
        ).values
        df["is_left_handed"] = (df["track_handed"] > 0).astype(float).where(df["track_handed"].notna())
        df["is_right_handed"] = (df["track_handed"] < 0).astype(float).where(df["track_handed"].notna())

        yards, moved = parse_rail_move(df.get("rail_move", pd.Series("", index=df.index)))
        df["rail_move_yards"] = yards.values
        df["rail_moved"] = moved.values
        df["rail_trust"] = 1.0 / (1.0 + df["rail_move_yards"].abs() / RAIL_TRUST_YARDS)

        dp = df["draw_pct_adj"]
        df["draw_x_handed"] = dp * df["track_handed"]
        # the form FEATURE_ENGINEERING_RESEARCH.md §E3 proposes, kept verbatim
        df["draw_x_direction"] = dp * df["is_left_handed"]
        return df

    # ------------------------------------------------------------------
    # Step 1: Track+Distance Draw Bias (lag-safe expanding mean)
    # ------------------------------------------------------------------
    def _calc_track_distance_draw_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute historical draw bias per track+distance.

        For each track+distance, we track the expanding mean NFP
        for low-stall and high-stall runners separately. The difference
        gives the draw bias at that course configuration.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        # Track-distance key
        df["_td_key"] = (
            df["track"].astype(str) + "_" + df["dist_furlongs"].round(0).astype(str)
        )

        # Low/high stall NFP (only for flat races)
        is_low = (df["draw_relative"] <= 0.5) & df["_is_flat"]
        is_high = (df["draw_relative"] > 0.5) & df["_is_flat"]

        nfp = df.get("NFP", pd.Series(np.nan, index=df.index))
        df["_low_stall_nfp"] = np.where(is_low, nfp, np.nan)
        df["_high_stall_nfp"] = np.where(is_high, nfp, np.nan)

        # Expanding mean NFP for each stall group at this track+distance, lagged by
        # RACE. A row-wise shift(1) here would step back to a rival in the same
        # race, and since these columns are otherwise constant across a race that
        # leak would be their only within-race variation. See model/lagsafe.py.
        df["td_low_stall_nfp"] = race_lagged_expanding_mean(df, "_td_key", "_low_stall_nfp")
        df["td_high_stall_nfp"] = race_lagged_expanding_mean(df, "_td_key", "_high_stall_nfp")

        # Draw bias at this track+distance (positive = low stalls favoured)
        df["td_draw_bias"] = df["td_low_stall_nfp"] - df["td_high_stall_nfp"]

        # This horse's stall vs the bias: positive = horse has the favoured draw
        df["draw_bias_alignment"] = np.where(
            is_low,
            df["td_draw_bias"],   # Low stall: positive bias helps
            np.where(is_high, -df["td_draw_bias"], 0)  # High stall: bias hurts
        )

        # Track-level (all distances) draw bias
        df = df.sort_values(["track", "race_date", "race_time"]).reset_index(drop=True)
        t_grp = df.groupby("track", group_keys=False)
        # Vectorized expanding mean (avoid slow apply per-group)
        low_shifted = t_grp["_low_stall_nfp"].shift(1)
        high_shifted = t_grp["_high_stall_nfp"].shift(1)
        low_cs = low_shifted.groupby(df["track"], sort=False).cumsum()
        low_cc = low_shifted.notna().astype(float).groupby(df["track"], sort=False).cumsum()
        high_cs = high_shifted.groupby(df["track"], sort=False).cumsum()
        high_cc = high_shifted.notna().astype(float).groupby(df["track"], sort=False).cumsum()
        df["track_draw_bias"] = (
            low_cs / low_cc.replace(0, np.nan)
            - high_cs / high_cc.replace(0, np.nan)
        )

        # Cleanup intermediates
        df.drop(columns=["_td_key", "_low_stall_nfp", "_high_stall_nfp"],
                inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 2: Going-Adjusted Draw Bias
    # ------------------------------------------------------------------
    def _calc_going_draw_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw bias modulated by going/ground conditions.

        Going shifts draw advantage: heavy ground nearly neutralises bias,
        soft amplifies it at some tracks, AW (all-weather) is consistent.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

        df["_going_cat"] = df["going_description"].apply(classify_going)

        # Track+distance+going key
        df["_tdg_key"] = (
            df["track"].astype(str) + "_"
            + df["dist_furlongs"].round(0).astype(str) + "_"
            + df["_going_cat"]
        )

        is_low = (df["draw_relative"] <= 0.5) & df["_is_flat"]
        is_high = (df["draw_relative"] > 0.5) & df["_is_flat"]
        nfp = df.get("NFP", pd.Series(np.nan, index=df.index))

        df["_low_nfp_g"] = np.where(is_low, nfp, np.nan)
        df["_high_nfp_g"] = np.where(is_high, nfp, np.nan)

        # Lagged by race, and only once the cell has a few races behind it: a
        # track x distance x going bucket is small, and an unshrunk difference of
        # two means over one prior race is noise with a large number attached.
        df["tdg_low_stall_nfp"] = race_lagged_expanding_mean(df, "_tdg_key", "_low_nfp_g", min_races=MIN_CELL_RACES)
        df["tdg_high_stall_nfp"] = race_lagged_expanding_mean(df, "_tdg_key", "_high_nfp_g", min_races=MIN_CELL_RACES)
        df["tdg_draw_bias"] = df["tdg_low_stall_nfp"] - df["tdg_high_stall_nfp"]

        # Going-adjusted alignment: use tdg if enough data, fallback to td
        df["going_draw_alignment"] = np.where(
            is_low,
            df["tdg_draw_bias"].fillna(df.get("td_draw_bias", 0)),
            np.where(
                is_high,
                -df["tdg_draw_bias"].fillna(df.get("td_draw_bias", 0)),
                0,
            )
        )

        # Going draw shift: difference between going-specific and overall bias
        td_bias = df.get("td_draw_bias", pd.Series(0, index=df.index))
        df["going_draw_shift"] = df["tdg_draw_bias"] - td_bias

        df.drop(
            columns=["_going_cat", "_tdg_key", "_low_nfp_g", "_high_nfp_g",
                     "tdg_low_stall_nfp", "tdg_high_stall_nfp"],
            inplace=True,
        )

        return df

    # ------------------------------------------------------------------
    # Step 3: Field-Size Draw Scaling
    # ------------------------------------------------------------------
    def _calc_field_size_draw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw bias amplified by field size.

        Data shows: 13-16 runners: bias +0.019, 6-8 runners: +0.009.
        Scale the draw bias by field size bucket.
        """
        nr = df["number_of_runners"].replace(0, np.nan)

        # Field-size weight: bigger fields = stronger draw effect
        # Normalise: ~1.0 at typical field of 10
        df["draw_field_weight"] = np.clip(nr / 10.0, 0.5, 2.0)

        # Field-adjusted draw advantage
        td_bias = df.get("td_draw_bias", pd.Series(0, index=df.index)).fillna(0)
        is_low = (df["draw_relative"] <= 0.5) & df["_is_flat"]

        raw_alignment = np.where(is_low, td_bias, -td_bias)
        df["draw_advantage_field_adj"] = raw_alignment * df["draw_field_weight"]

        # Edge position: is the horse on the extreme inside (stall 1) or
        # extreme outside (last stall)? These have distinct profiles.
        df["is_stall_1"] = (df["stall_num_raw"] == 1).astype(float)
        df["is_widest_stall"] = (
            df["stall_num_raw"] == df["number_of_runners"]
        ).astype(float)

        return df

    # ------------------------------------------------------------------
    # Step 3b: expected-vs-actual draw bias, per stall, shrunk and decayed
    # ------------------------------------------------------------------
    def _calc_expected_vs_actual_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """§6.3 done the way §6.3 asks for it.

        ``draw_bias(cell) = mean(nmfp) − mean(expected nmfp from ratings)``.
        Taking raw finishing positions by stall band instead — which is what
        every other draw feature in this module does — measures the horses as
        much as the track: "using expected-vs-actual rather than raw win rates
        controls for the possibility that low stalls simply held the best horses
        on that day, which is the standard trap".

        Estimated per stall and pooled down a four-level hierarchy, every level
        time-decayed, with the effective sample size kept so thin cells can be
        shrunk hard and flagged rather than believed.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
        rk = ensure_race_key(df)
        is_flat = df["_is_flat"]
        n = pd.to_numeric(df["number_of_runners"], errors="coerce")
        place = (pd.to_numeric(df["placing_numerical"], errors="coerce")
                 if "placing_numerical" in df.columns
                 else pd.Series(np.nan, index=df.index))

        df["nmfp_actual"] = nmfp(n.values, place.values)
        exp_raw, rating_src = expected_nmfp_from_ratings(df)
        df["_exp_nmfp_raw"] = exp_raw
        df["rating_slope"] = rating_calibration_slope(df, "_exp_nmfp_raw", "nmfp_actual")
        df["expected_nmfp"] = df["rating_slope"] * df["_exp_nmfp_raw"]
        df["rating_source"] = rating_src

        # The residual is the whole point: what the horses did not explain.
        df["draw_resid"] = (df["nmfp_actual"] - df["expected_nmfp"]).where(is_flat)

        # --- cell keys ------------------------------------------------------
        track = df["track"].astype(str)
        dist = df["dist_furlongs"].round(0).astype(str)
        going = df["going_description"].apply(classify_going).astype(str)
        fband = pd.cut(n, FIELD_SIZE_BAND_EDGES, labels=False).astype("float").astype(str)
        stall = df["draw_adj"].clip(1, MAX_STALL).round(0).astype("float").astype(str)
        quart = pd.cut(df["draw_pct_adj"], bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
                       labels=False).astype("float").astype(str)

        df["_k_course"] = track + "|" + quart
        df["_k_stall"] = track + "|" + dist + "|" + stall
        df["_k_field"] = df["_k_stall"] + "|" + fband
        df["_k_cell"] = df["_k_field"] + "|" + going

        m1, n1 = race_lagged_decayed_mean(df, "_k_course", "draw_resid")
        m2, n2 = race_lagged_decayed_mean(df, "_k_stall", "draw_resid")
        m3, n3 = race_lagged_decayed_mean(df, "_k_field", "draw_resid")
        m4, n4 = race_lagged_decayed_mean(df, "_k_cell", "draw_resid")

        # Hierarchical pooling: each level is believed only in proportion to its
        # own effective sample and otherwise inherits the level above it.
        e1 = _shrink(m1, n1, 0.0, SHRINK_K_COURSE)
        e2 = _shrink(m2, n2, e1, SHRINK_K_STALL)
        e3 = _shrink(m3, n3, e2, SHRINK_K_FIELD)
        e4 = _shrink(m4, n4, e3, SHRINK_K_CELL)

        df["draw_bias_ev_course"] = e1.where(is_flat)
        df["draw_bias_ev_stall"] = e2.where(is_flat)
        df["draw_bias_ev"] = e4.where(is_flat)
        df["draw_bias_ev_n_eff"] = n4.where(is_flat)
        df["draw_bias_stall_n_eff"] = n2.where(is_flat)

        # §6.3 guard: decay is not enough on its own, the feature has to say
        # when it does not know — thin sample, or the rail has been moved.
        df["draw_bias_confident"] = (
            (n2 >= MIN_CONFIDENT_N_EFF) & (df["rail_moved"] == 0)
        ).astype(float).where(is_flat)
        df["draw_bias_ev_adj"] = df["draw_bias_ev"] * df["rail_trust"]

        # Race-level shape of the surface: how strongly the draw is tilted today
        # and how far apart the best and worst stalls are. Both are built from
        # lagged quantities, so they are race constants with nothing of today in
        # them.
        x = df["draw_pct_adj"].where(is_flat)
        y = df["draw_bias_ev"]
        xm = x.groupby(rk).transform("mean")
        ym = y.groupby(rk).transform("mean")
        cov = ((x - xm) * (y - ym)).groupby(rk).transform("mean")
        var = ((x - xm) ** 2).groupby(rk).transform("mean")
        df["draw_bias_gradient"] = (cov / var.where(var > 0))
        df["draw_bias_spread"] = y.groupby(rk).transform("std")

        df.drop(columns=["_exp_nmfp_raw", "_k_course", "_k_stall", "_k_field", "_k_cell"],
                inplace=True)
        return df

    # ------------------------------------------------------------------
    # Step 3c: §6.4 split-half screen
    # ------------------------------------------------------------------
    def _calc_split_half_screen(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the cheap §6.4 screen over every race and average it by course.

        The per-race numbers are a screen over completed races. What a model can
        use is the course/distance baseline built from earlier races: how far
        apart stall order and finish order usually end up here, and how lopsided
        the two halves of the result usually are. A course where they agree is a
        course where the draw decides races.
        """
        df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
        rk = ensure_race_key(df)

        scr = draw_bias_split_half(df)
        if scr.empty:
            for c in ("dsh_front", "dsh_back", "dsh_diff", "dsh_diff_norm", "dsh_level"):
                df[c] = np.nan
        else:
            m = scr.set_index("raceid")
            for c in ("dsh_front", "dsh_back", "dsh_diff", "dsh_diff_norm", "dsh_level"):
                df[c] = rk.map(m[c])
        n = pd.to_numeric(df["number_of_runners"], errors="coerce")
        df["_dsh_abs"] = df["dsh_diff_norm"].abs()
        df["_dsh_level_norm"] = df["dsh_level"] / np.sqrt(n.where(n > 0))

        df["_td_key"] = df["track"].astype(str) + "_" + df["dist_furlongs"].round(0).astype(str)
        # plain expanding means over earlier races (the same rule as
        # race_lagged_expanding_mean, on the faster path used above)
        flat = dict(halflife_days=np.inf)
        df["td_draw_screen"] = race_lagged_decayed_mean(df, "_td_key", "_dsh_abs", **flat)[0]
        df["track_draw_screen"] = race_lagged_decayed_mean(df, "track", "_dsh_abs", **flat)[0]
        df["td_dsh_level"] = race_lagged_decayed_mean(df, "_td_key", "_dsh_level_norm", **flat)[0]
        df["dsh_abnormal"] = (df["_dsh_abs"] > SPLIT_HALF_FLAG_MULT * df["td_draw_screen"]).astype(float)
        df.drop(columns=["_td_key", "_dsh_abs", "_dsh_level_norm"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Step 3d: GP per-stall surface (opt-in)
    # ------------------------------------------------------------------
    def _calc_gp_draw_surface(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooth the residual across neighbouring stalls, refit period by period.

        The hierarchy in step 3b pools a stall toward coarser *cells*; this pools
        it toward its *neighbours*, which is the other half of the problem —
        stall 6 and stall 7 of the same course share ground and should not be
        estimated independently. Lag-safe by construction: each period is scored
        by a surface fitted only on races before it started.
        """
        from model.spatial import walk_forward_stall_surface

        df["_gp_dist"] = df["dist_furlongs"].round(0)
        surf = walk_forward_stall_surface(
            df, value_col="draw_resid", stall_col="draw_adj",
            keys=list(self.gp_keys), freq=self.gp_freq,
        )
        for c in GP_DRAW_FEATURES:
            df[c] = surf[c].where(df["_is_flat"])
        df.drop(columns=["_gp_dist"], inplace=True)
        return df

    # ------------------------------------------------------------------
    # Step 4: Stall x Pace Interaction
    # ------------------------------------------------------------------
    def _calc_stall_pace_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw effects interact with running style.

        Key insight from data: draw bias primarily affects Prominent/Midfield
        runners (NFP delta ~0.03-0.04), not Leaders or Held-up horses.
        Low stalls enable front-running (23.9% vs 21.3% Leader rate).
        """
        dr = df["draw_relative"].fillna(0.5)

        # Predicted early position from pace features (use if available)
        pred_ep = df.get("pred_early_pos", df.get("horse_career_early_pos",
                         df.get("Horse_Career_EPF", pd.Series(3.0, index=df.index))))
        pred_ep = pred_ep.fillna(3.0)

        # Front-runner from low stall advantage
        # Low stall + front-runner style = rail advantage
        df["draw_front_advantage"] = np.where(
            pred_ep >= 4.0,
            (0.5 - dr) * 2,  # Scales -1 (widest) to +1 (innermost)
            0,
        )

        # Prominent runner draw effect (strongest bias in the data)
        df["draw_prominent_effect"] = np.where(
            (pred_ep >= 3.0) & (pred_ep < 5.0),
            (0.5 - dr) * 2,
            0,
        )

        # Hold-up runner: draw matters less (near-zero bias in data)
        df["draw_holdup_effect"] = np.where(
            pred_ep < 2.5,
            (0.5 - dr) * 0.5,  # Dampened effect
            0,
        )

        # Draw enables front-running: probability of leading given stall
        # Low stall -> more likely to get to the front
        df["draw_front_enable"] = np.clip(1.0 - dr, 0, 1)

        return df

    # ------------------------------------------------------------------
    # Step 5: Horse Personal Draw Preference
    # ------------------------------------------------------------------
    def _calc_horse_draw_preference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Does this horse historically perform better from low or high stalls?

        Some horses need the rail, others cope fine wide. Track this
        as a personal preference metric (lag-safe expanding mean).
        """
        df = df.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
        grp = df.groupby("horse_name", group_keys=False)

        nfp = df.get("NFP", pd.Series(np.nan, index=df.index))
        is_flat = df["_is_flat"]

        # Low-stall NFP for this horse
        is_low = (df["draw_relative"] <= 0.5) & is_flat
        is_high = (df["draw_relative"] > 0.5) & is_flat

        df["_h_low_nfp"] = np.where(is_low, nfp, np.nan)
        df["_h_high_nfp"] = np.where(is_high, nfp, np.nan)

        # Career average NFP from low/high stalls (lag-safe)
        df["horse_low_stall_nfp"] = grp["_h_low_nfp"].apply(
            lambda x: x.shift(1).expanding().mean()
        )
        df["horse_high_stall_nfp"] = grp["_h_high_nfp"].apply(
            lambda x: x.shift(1).expanding().mean()
        )

        # Personal draw preference (positive = prefers low stalls)
        df["horse_draw_pref"] = df["horse_low_stall_nfp"] - df["horse_high_stall_nfp"]

        # Current draw matches preference?
        df["draw_pref_match"] = np.where(
            is_low,
            df["horse_draw_pref"].fillna(0),     # Low stall: positive pref = good match
            np.where(
                is_high,
                -df["horse_draw_pref"].fillna(0), # High stall: negative pref = good match
                0,
            )
        )

        # Last-run draw relative (was horse wide or inside last time?)
        df["LR_draw_relative"] = grp["draw_relative"].shift(1)

        # Draw change from last run
        df["draw_change"] = df["draw_relative"] - df["LR_draw_relative"]

        df.drop(columns=["_h_low_nfp", "_h_high_nfp"], inplace=True)

        return df

    # ------------------------------------------------------------------
    # Step 5b: §6.5 draw x position interactions
    # ------------------------------------------------------------------
    def _calc_spec_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """The §6.5 interactions, by their framework names.

        "A low draw at a track where low draws win *because* they get an early
        tactical advantage only helps a horse that will actually use it. A
        hold-up horse drawn low at such a course gets less of the benefit and
        may get worse - trapped on the rail with nowhere to go."

        Every term is a product of two lag-safe quantities, and the sign of each
        is left to the model: §6.5 is explicit that it should be fitted rather
        than assumed.
        """
        idx = df.index
        nan = pd.Series(np.nan, index=idx)
        dp = df["draw_pct_adj"]
        mean_epf = pd.to_numeric(df.get("horse_epf_norm_mean", nan), errors="coerce")
        pred_epf = pd.to_numeric(df.get("pred_epf_norm", nan), errors="coerce")
        pressure = pd.to_numeric(df.get("pred_pace_pressure", nan), errors="coerce")
        n = pd.to_numeric(df["number_of_runners"], errors="coerce")

        df["draw_x_style"] = dp * mean_epf
        df["draw_x_pace"] = dp * pressure
        df["draw_x_fieldsize"] = dp * n
        # draw_pos_fit = draw_bias(context) x predicted early-position advantage,
        # the advantage being +1 for a confirmed front-runner and −1 for a horse
        # that will be last of the field early.
        df["draw_pos_fit"] = df["draw_bias_ev_adj"] * (1 - 2 * pred_epf)
        df["draw_lead_fit"] = df["draw_bias_ev_adj"] * pd.to_numeric(
            df.get("predicted_lead_prob", nan), errors="coerce"
        )
        return df

    # ------------------------------------------------------------------
    # Step 6: Draw Advantage Composite
    # ------------------------------------------------------------------
    def _calc_draw_advantage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite draw advantage score combining all factors.

        Combines: track bias, going adjustment, field size scaling,
        pace interaction, and personal preference.
        """
        # Component weights (based on data analysis signal strength)
        td_bias = df.get("draw_bias_alignment", pd.Series(0, index=df.index)).fillna(0)
        going_adj = df.get("going_draw_alignment", pd.Series(0, index=df.index)).fillna(0)
        field_adj = df.get("draw_advantage_field_adj", pd.Series(0, index=df.index)).fillna(0)
        pace_front = df.get("draw_front_advantage", pd.Series(0, index=df.index)).fillna(0)
        pref_match = df.get("draw_pref_match", pd.Series(0, index=df.index)).fillna(0)

        # Weighted composite
        df["draw_advantage_composite"] = (
            0.30 * td_bias
            + 0.25 * going_adj
            + 0.20 * field_adj
            + 0.15 * pace_front
            + 0.10 * pref_match
        )

        return df

    # ------------------------------------------------------------------
    # Step 7: Within-race draw rankings
    # ------------------------------------------------------------------
    def _calc_draw_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank horses within race on draw-related features."""
        df = df.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)

        rank_cols = {
            "rDrawBiasAlign": "draw_bias_alignment",
            "rGoingDrawAlign": "going_draw_alignment",
            "rDrawAdvComposite": "draw_advantage_composite",
            "rDrawFieldAdj": "draw_advantage_field_adj",
            "rHorseDrawPref": "horse_draw_pref",
            "rDrawBiasEV": "draw_bias_ev_adj",
            "rDrawPosFit": "draw_pos_fit",
            "rDrawPctAdj": "draw_pct_adj",
        }

        for rank_name, source_col in rank_cols.items():
            if source_col in df.columns:
                df[rank_name] = df.groupby("raceid")[source_col].rank(
                    ascending=False, method="min", na_option="bottom"
                )

        return df


# ---------------------------------------------------------------------------
# Feature lists for train_bfsp.py integration
# ---------------------------------------------------------------------------

# §6.3 draw adjusted for non-runners
DRAW_ADJ_FEATURES = [
    "draw_adj",
    "draw_pct_adj",
    "draw_vacancy_inside",
    "draw_nr_shift",
]

# Course geometry: handedness and rail movement (previously unread columns)
TRACK_GEOMETRY_FEATURES = [
    "track_handed",
    "is_left_handed",
    "is_right_handed",
    "rail_move_yards",
    "rail_moved",
    "rail_trust",
    "draw_x_handed",
    "draw_x_direction",
]

# §6.3 expected-vs-actual per-stall bias surface
EV_DRAW_FEATURES = [
    "draw_bias_ev",
    "draw_bias_ev_adj",
    "draw_bias_ev_stall",
    "draw_bias_ev_course",
    "draw_bias_ev_n_eff",
    "draw_bias_stall_n_eff",
    "draw_bias_confident",
    "draw_bias_gradient",
    "draw_bias_spread",
]

# §6.4 bias screens, as course/distance baselines over earlier races
DRAW_SCREEN_FEATURES = [
    "td_draw_screen",
    "track_draw_screen",
    "td_dsh_level",
]

# Opt-in: the Gaussian-process per-stall surface (DrawMetricsEngine(
# gp_draw_surface=True)). Kept out of ALL_DRAW_FEATURES because the columns only
# exist when the flag is on.
GP_DRAW_FEATURES = [
    "gp_draw_bias",
    "gp_draw_sd",
    "gp_draw_n",
]

# §6.5 draw x position interactions
DRAW_INTERACTION_FEATURES = [
    "draw_x_style",
    "draw_x_pace",
    "draw_x_fieldsize",
    "draw_pos_fit",
    "draw_lead_fit",
]

# Track/distance draw bias
TRACK_DRAW_FEATURES = [
    "td_draw_bias",
    "draw_bias_alignment",
    "track_draw_bias",
    "td_low_stall_nfp",
    "td_high_stall_nfp",
]

# Going-adjusted draw
GOING_DRAW_FEATURES = [
    "tdg_draw_bias",
    "going_draw_alignment",
    "going_draw_shift",
]

# Field-size draw features
FIELD_DRAW_FEATURES = [
    "draw_field_weight",
    "draw_advantage_field_adj",
    "is_stall_1",
    "is_widest_stall",
]

# Stall x pace interaction
STALL_PACE_FEATURES = [
    "draw_front_advantage",
    "draw_prominent_effect",
    "draw_holdup_effect",
    "draw_front_enable",
]

# Horse personal draw preference
HORSE_DRAW_FEATURES = [
    "horse_low_stall_nfp",
    "horse_high_stall_nfp",
    "horse_draw_pref",
    "draw_pref_match",
    "LR_draw_relative",
    "draw_change",
]

# Composite and rankings
DRAW_COMPOSITE_FEATURES = [
    "draw_advantage_composite",
    "rDrawBiasAlign",
    "rGoingDrawAlign",
    "rDrawAdvComposite",
    "rDrawFieldAdj",
    "rHorseDrawPref",
    "rDrawBiasEV",
    "rDrawPosFit",
    "rDrawPctAdj",
]

# Combined list for easy import
ALL_DRAW_FEATURES = (
    DRAW_ADJ_FEATURES
    + TRACK_GEOMETRY_FEATURES
    + EV_DRAW_FEATURES
    + DRAW_SCREEN_FEATURES
    + DRAW_INTERACTION_FEATURES
    + TRACK_DRAW_FEATURES
    + GOING_DRAW_FEATURES
    + FIELD_DRAW_FEATURES
    + STALL_PACE_FEATURES
    + HORSE_DRAW_FEATURES
    + DRAW_COMPOSITE_FEATURES
)

assert not (set(ALL_DRAW_FEATURES) & DRAW_POST_RACE_ONLY), \
    "a column that needs the result of the race being predicted reached the feature list"
