"""
Performance primitives and transforms from the racing² master framework
(Section III, Parts 1–3, 7, 8) that Ashcroft's custom metrics do not yet carry.

Per run (Part 1):
    prb, prb2, prb2_par, prb2_fsa     % rivals beaten, squared, field-size par, adjusted
    nmfp                              normalised finishing position (= uniform-rank z / 3)
    fss_cred                          field-size credibility weight √((N−1)/(N+1))
    plc_fsa                           placed flag minus places/N (UK each-way terms)
    bl_trunc, dabl                    15·tanh(BL/15), distance-adjusted (÷ (dist_m/1000)^0.25)
    bl_seconds, bl_pct_wintime        lengths → seconds (LPS by surface/going) ÷ winning time
    bl_prop_figure                    proportion-method beaten figure, (1−(t_win/t_horse)^(d/1000))·1000
    bl_signed, bl_next, bl_behind     cluster geometry: the *shape* of the finish (§1.5.a)
    bl_gap_ahead, bl_isolation
    bl_vs_par, bl_par_ratio           DABL vs an empirical (position × field-size × context) par surface
    bl_per_position, won_by_vs_par
    bl_censored + the five separate    eased / pulled up / tailed off / hampered / slipped
      flags of §1.5.f

Aggregation (Part 3): windows × {mean, iqm, max, min, slope, first, %above par},
FSS-credibility weighting, λ-decay in days and γ^k decay in runs, DSLR and
season counters; the "consistency floor" MIN over L10.

Ratings (Part 7): handicapper-gap features (well_in, mark_vs_form, tf_vs_or,
rating_dispersion); LTO-contradiction features (lto_vs_career,
lto_tfig_above_ability).

Context (Part 2): rating-based and market effective field size, field quality
strength (mean / IQM / max of the opponents' ratings) and its dispersion,
strength of schedule with SoS_delta and SoS_vs_today, combined race strength,
field-size-adjusted placed flags for the horse, jockey and trainer, and PvS —
performance relative to schedule, §2.4's "single most important correction to
naive form aggregates".

Transforms (Part 8.1): within-race z, rankpct, vs_max, vs_median, share,
norm_spread and vs_2nd for any feature.

Lag safety. Aggregates grouped on ``horse_name`` use shift-1 over the horse's
runs ordered by date/time, which is safe because a horse runs once in a race.
Anything grouped on a key that repeats inside a race (trainer, jockey) goes
through ``model.lagsafe.race_lagged_expanding_mean`` instead. Fitted surfaces
(the par surfaces, the PvS regression) are fitted on a training window and
applied forward.

Within-race geometry — ``bl_next``, ``bl_isolation``, ``bl_z`` and the rest —
describes the race it is computed from, so it is *post-race*: legitimate as the
input to a lagged feature about the horse's previous runs, never a feature of
the race being predicted. The names are collected in ``POST_RACE_PRIMITIVES``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.abm.features import place_terms
from model.lagsafe import ensure_race_key, race_lagged_expanding_mean
from model.perf_figures import parse_beaten_lengths, winning_margin

LPS_STANDARD = 6.0  # lengths per second on standard going (BHA scale; re-fit locally)

#: BHA published lengths-per-second scale (§1.5.b). A length is a made-up unit
#: and its time value moves with the ground: one second covers six lengths on
#: good ground and five on heavy. Flat and jumps run a length apart throughout.
#: These are the published figures, not fitted ones — §X.5/P11 asks for our own
#: LPS by surface and going once there is enough timing data to fit it.
LPS_SCALE = {
    "flat": {"firm": 6.0, "good to firm": 6.0, "good": 6.0, "good to soft": 5.5,
             "soft": 5.5, "heavy": 5.0, "aw": 6.0},
    "jumps": {"firm": 5.0, "good to firm": 5.0, "good": 5.0, "good to soft": 4.5,
              "soft": 4.5, "heavy": 4.0, "aw": 5.0},
}

_JUMPS_RX = re.compile(r"hurdle|chase|steeple|\bnh\b|jump|bumper|nhf", re.I)


# ---------------------------------------------------------------------------
# Per-run primitives
# ---------------------------------------------------------------------------

def prb(N, P):
    N = np.asarray(N, float); P = np.asarray(P, float)
    return np.where(N > 1, (N - P) / (N - 1), np.nan)


def prb2_par(N):
    N = np.asarray(N, float)
    return np.where(N > 1, (2 * N - 1) / (6 * (N - 1)), np.nan)


def nmfp(N, P):
    N = np.asarray(N, float); P = np.asarray(P, float)
    return np.where(N > 1, (N + 1 - 2 * P) / np.sqrt(3 * (N * N - 1)), np.nan)


def fss_cred(N):
    N = np.asarray(N, float)
    return np.where(N > 1, np.sqrt((N - 1) / (N + 1)), np.nan)


def truncate_bl(bl, c: float = 15.0):
    return c * np.tanh(np.asarray(bl, float) / c)


def dist_adjust_bl(bl_trunc, dist_metres, exponent: float = 0.25):
    return np.asarray(bl_trunc, float) / (np.asarray(dist_metres, float) / 1000.0) ** exponent


def going_group(going) -> pd.Series:
    """Going strings collapsed to the seven buckets the BHA LPS scale uses."""
    g = pd.Series(going).fillna("").astype(str).str.lower().str.strip()
    out = pd.Series("good", index=g.index, dtype=object)
    out[g.str.contains("standard|polytrack|tapeta|fibresand|slow|fast", regex=True)] = "aw"
    out[g.str.contains("good to firm|gd-fm|gf")] = "good to firm"
    out[g.str.contains("good to soft|gd-sft|gs|yielding")] = "good to soft"
    out[g.str.contains("soft") & ~g.str.contains("good")] = "soft"
    out[g.str.contains("firm") & ~g.str.contains("good")] = "firm"
    out[g.str.contains("heavy")] = "heavy"
    out[g.eq("")] = "good"
    return out


def lps_scale(going=None, code=None, default: float = LPS_STANDARD) -> np.ndarray | float:
    """Lengths per second for the going and the code (§1.5.b, D16).

    ``code`` is the race code/type string; anything reading as hurdle, chase or
    bumper takes the jumps column, everything else the flat column. Unknown
    going falls back to ``default`` rather than silently pricing heavy ground
    as good.
    """
    scalar = np.ndim(going) == 0 and (code is None or np.ndim(code) == 0)
    g = going_group(pd.Series([going]) if scalar else pd.Series(going))
    if code is None:
        jumps = pd.Series(False, index=g.index)
    else:
        c = pd.Series([code] if scalar else code).fillna("").astype(str)
        c.index = g.index
        jumps = c.str.contains(_JUMPS_RX)
    flat = g.map(LPS_SCALE["flat"]); jmp = g.map(LPS_SCALE["jumps"])
    out = pd.Series(np.where(jumps, jmp, flat), index=g.index).astype(float).fillna(default).values
    return float(out[0]) if scalar else out


def proportion_beaten_figure(win_time, horse_time, dist_metres) -> np.ndarray:
    """§1.5.b / D13: (1 − (t_win / t_horse) ^ (dist_m / 1000)) · 1000.

    The properly normalised margin measure: scale-free, distance-aware and
    directly comparable across the database. ``horse_time`` is the individual
    horse's time where it exists; the schema carries only the race's winning
    time, so ``add_run_primitives`` reconstructs it from the margin and the LPS
    scale and the figure inherits that reconstruction's error.
    """
    w = np.asarray(win_time, float); h = np.asarray(horse_time, float)
    d = np.asarray(dist_metres, float) / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (1.0 - (w / h) ** d) * 1000.0
    return np.where((w > 0) & (h > 0) & (d > 0), out, np.nan)


CENSOR_RX = re.compile(r"eased|tailed off|pulled up|hampered|badly hampered|brought down|unseated|fell|slipped|"
                       r"not persevered|virtually pulled up|lost action|refused|ran out|saddle slipped", re.I)

#: §1.5.f asks for the censoring reasons separately, not one combined flag: a
#: horse eased when held is a different observation from one hampered at a
#: crucial point, and the model should be able to discount them differently.
COMMENT_FLAG_RX = {
    "eased_flag": re.compile(r"eased|not persevered|nothing more asked|no extra pressure", re.I),
    "pulled_up_flag": re.compile(r"pulled up|virtually pulled up|refused|ran out", re.I),
    "tailed_off_flag": re.compile(r"tailed off|well behind|detached|lost touch|soon behind", re.I),
    "hampered_flag": re.compile(r"hamper|badly hamper|short of room|no room|squeezed|impeded|checked|barged|carried", re.I),
    "slipped_flag": re.compile(r"slipped|lost action|lost (its |his |her )?action|stumbl|fell|brought down|unseated", re.I),
    # §3.3's lto_* suite wants these two as well; same source, same caveat
    "wide_flag": re.compile(r"\bwide\b|raced wide|wide of|outer", re.I),
    "slow_start_flag": re.compile(r"slowly away|slow to start|missed the break|dwelt|reluctant to race|never a factor from", re.I),
}


def comment_flags(df: pd.DataFrame, comment_col: str = "comment") -> pd.DataFrame:
    """The §1.5.f censoring flags plus lto_wide / lto_slow_start, from the
    in-running comment. POST-RACE: today's card has no comments."""
    out = df.copy()
    c = out[comment_col].fillna("").astype(str) if comment_col in out.columns else pd.Series("", index=out.index)
    for name, rx in COMMENT_FLAG_RX.items():
        out[name] = c.str.contains(rx).astype(int)
    out["bl_censored"] = c.str.contains(CENSOR_RX).astype(int)
    return out


def add_run_primitives(df: pd.DataFrame, pos_col: str = "placing_numerical", n_col: str = "number_of_runners",
                       beaten_col: str = "total_dst_bt", dist_col: str = "dist_furlongs", win_time_col: str = "comptime_numeric",
                       comment_col: str = "comment", race_type_col: str = "race_type", lps: float = LPS_STANDARD) -> pd.DataFrame:
    d = df.copy()
    N = pd.to_numeric(d[n_col], errors="coerce").values; P = pd.to_numeric(d[pos_col], errors="coerce").values
    d["prb"] = prb(N, P); d["prb2"] = d["prb"] ** 2; d["prb2_par"] = prb2_par(N); d["prb2_fsa"] = d["prb2"] - d["prb2_par"]
    d["nmfp"] = nmfp(N, P); d["fss_cred"] = fss_cred(N)
    hcp = d[race_type_col].astype(str).str.lower().str.contains("h", regex=False) if race_type_col in d.columns else pd.Series(False, index=d.index)
    places = np.array([place_terms(int(n), bool(h)) if n == n else 0 for n, h in zip(N, hcp)])
    d["plc_fsa"] = np.where(np.isnan(P), np.nan, (P <= places).astype(float) - places / np.where(N > 0, N, np.nan))
    bl = d[beaten_col].map(parse_beaten_lengths).values.astype(float) if beaten_col in d.columns else np.full(len(d), np.nan)
    bl = np.where(P == 1, 0.0, bl)
    d["bl_winner"] = bl
    d["bl_trunc"] = truncate_bl(bl)
    dist_m = pd.to_numeric(d[dist_col], errors="coerce").values * 201.168
    d["dabl"] = dist_adjust_bl(d["bl_trunc"].values, dist_m)
    if win_time_col in d.columns:
        wt = pd.to_numeric(d[win_time_col], errors="coerce").values
        d["bl_pct_wintime"] = np.where(wt > 0, (bl / lps) / wt * 100.0, np.nan)
    if comment_col in d.columns:
        d["bl_censored"] = d[comment_col].fillna("").astype(str).str.contains(CENSOR_RX).astype(int)
        d.loc[d["bl_censored"] == 1, "dabl"] = np.nan  # option 3 of §1.5.f for the margin; position primitives stay
    return d


class BeatenLengthsPar:
    """Empirical par surface: median DABL by finishing position × field-size band
    (monotone in position within band). Fit on a training window, apply forward."""

    def __init__(self, n_bands=(2, 5, 8, 11, 15, 100)):
        self.n_bands = list(n_bands); self.table_ = None

    def _band(self, N):
        return pd.cut(pd.Series(np.asarray(N, float)), self.n_bands, right=True, labels=False)

    def fit(self, df: pd.DataFrame, pos_col: str = "placing_numerical", n_col: str = "number_of_runners", dabl_col: str = "dabl"):
        d = pd.DataFrame({"P": pd.to_numeric(df[pos_col], errors="coerce"), "b": self._band(df[n_col].values), "v": df[dabl_col]}).dropna()
        t = d.groupby(["b", "P"])["v"].median().reset_index()
        t["v"] = t.groupby("b")["v"].cummax()  # monotone in position
        self.table_ = t.set_index(["b", "P"])["v"]
        return self

    def par(self, N, P):
        b = self._band(N).values; P = pd.to_numeric(pd.Series(P), errors="coerce").values
        idx = pd.MultiIndex.from_arrays([b, P])
        return self.table_.reindex(idx).values

    def transform(self, df: pd.DataFrame, pos_col: str = "placing_numerical", n_col: str = "number_of_runners", dabl_col: str = "dabl") -> pd.DataFrame:
        d = df.copy()
        d["bl_par"] = self.par(d[n_col].values, d[pos_col].values)
        d["bl_vs_par"] = d["bl_par"] - d[dabl_col]  # positive = beaten less far than typical for that slot
        return d


# ---------------------------------------------------------------------------
# Aggregation (lag-safe)
# ---------------------------------------------------------------------------

def _iqm(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return float(np.mean(x)) if len(x) else np.nan
    q1, q3 = np.quantile(x, [0.25, 0.75])
    m = x[(x >= q1) & (x <= q3)]
    return float(np.mean(m)) if len(m) else float(np.mean(x))


def _slope(x: np.ndarray) -> float:
    ok = ~np.isnan(x)
    if ok.sum() < 3:
        return np.nan
    t = np.arange(len(x))[ok]
    return float(np.polyfit(t, x[ok], 1)[0])


def aggregate_history(df: pd.DataFrame, cols, windows=(3, 5, 10), horse_col: str = "horse_name",
                      date_col: str = "race_date", time_col: str = "race_time", weight_col: str | None = "fss_cred",
                      aggs=("mean", "iqm", "max", "min", "slope")) -> pd.DataFrame:
    """{col}_{agg}_L{w} and {col}_{agg}_career from PRIOR runs only; plus
    {col}_wmean_career weighted by FSS credibility when ``weight_col`` is set."""
    d = df.sort_values([horse_col, date_col, time_col]).copy()
    g = d.groupby(horse_col, group_keys=False)
    for c in cols:
        s = g[c].shift(1)
        d[f"{c}_last"] = s
        for w in windows:
            r = s.groupby(d[horse_col]).rolling(w, min_periods=1)
            if "mean" in aggs: d[f"{c}_mean_L{w}"] = r.mean().reset_index(level=0, drop=True)
            if "max" in aggs: d[f"{c}_max_L{w}"] = r.max().reset_index(level=0, drop=True)
            if "min" in aggs: d[f"{c}_min_L{w}"] = r.min().reset_index(level=0, drop=True)
            if "iqm" in aggs and w >= 5: d[f"{c}_iqm_L{w}"] = r.apply(_iqm, raw=True).reset_index(level=0, drop=True)
            if "slope" in aggs and w >= 5: d[f"{c}_slope_L{w}"] = r.apply(_slope, raw=True).reset_index(level=0, drop=True)
        e = s.groupby(d[horse_col]).expanding(min_periods=1)
        d[f"{c}_mean_career"] = e.mean().reset_index(level=0, drop=True)
        d[f"{c}_max_career"] = e.max().reset_index(level=0, drop=True)
        if weight_col and weight_col in d.columns:
            wv = g[weight_col].shift(1)
            num = (s * wv).groupby(d[horse_col]).cumsum(); den = wv.where(s.notna()).groupby(d[horse_col]).cumsum()
            d[f"{c}_wmean_career"] = num / den.replace(0, np.nan)
    d["runs_count"] = g.cumcount()
    return d.sort_index()


# ---------------------------------------------------------------------------
# Ratings, gaps, LTO contradictions, context
# ---------------------------------------------------------------------------

def handicapper_gap_features(d: pd.DataFrame, or_col: str = "official_rating", perf_col: str = "perf_lbs",
                             tf_col: str | None = "tf_master_pre", rating_cols=None) -> pd.DataFrame:
    """well_in = best perf L3 − OR; mark_vs_form = OR − perf IQM L5 (or mean L3);
    tf_vs_or; rating_dispersion across available rating sources (same-day)."""
    out = d.copy()
    orr = pd.to_numeric(out[or_col], errors="coerce").replace(0, np.nan)
    if f"{perf_col}_max_L3" in out.columns:
        out["well_in"] = out[f"{perf_col}_max_L3"] - orr
    form = out.get(f"{perf_col}_iqm_L5", out.get(f"{perf_col}_mean_L3"))
    if form is not None:
        out["mark_vs_form"] = orr - form
    if tf_col and tf_col in out.columns:
        out["tf_vs_or"] = out[tf_col] - orr
    cols = [c for c in (rating_cols or [or_col, tf_col, f"{perf_col}_mean_L3"]) if c and c in out.columns]
    if len(cols) >= 2:
        out["rating_dispersion"] = out[cols].std(axis=1)
    return out


def lto_contradiction_features(d: pd.DataFrame, nmfp_col: str = "nmfp", tfig_col: str | None = "timefigure",
                               perf_col: str = "perf_lbs") -> pd.DataFrame:
    """lto_vs_career = LTO nmfp − career mean excluding LTO;
    lto_tfig_above_ability = LTO timefigure − prior performance mean (ran above ability last time)."""
    out = d.copy()
    if f"{nmfp_col}_last" in out.columns and f"{nmfp_col}_mean_career" in out.columns and "runs_count" in out.columns:
        n = out["runs_count"].replace(0, np.nan)
        ex_lto = (out[f"{nmfp_col}_mean_career"] * n - out[f"{nmfp_col}_last"]) / (n - 1).replace(0, np.nan)
        out["lto_vs_career"] = out[f"{nmfp_col}_last"] - ex_lto
    if tfig_col and f"{tfig_col}_last" in out.columns and f"{perf_col}_mean_L5" in out.columns:
        out["lto_tfig_above_ability"] = out[f"{tfig_col}_last"] - out[f"{perf_col}_mean_L5"]
    return out


def n_eff_from_ratings(d: pd.DataFrame, rating_col: str, race_col: str = "raceid", temperature: float = 5.0) -> pd.Series:
    """Market-free effective field size: exp(entropy of softmax(rating/T)) per race."""
    r = pd.to_numeric(d[rating_col], errors="coerce")
    z = (r - r.groupby(d[race_col]).transform("max")) / temperature
    e = np.exp(z.fillna(z.min() if z.notna().any() else 0.0))
    p = e / e.groupby(d[race_col]).transform("sum")
    ent = -(p * np.log(p.clip(1e-12))).groupby(d[race_col]).transform("sum")
    return np.exp(ent)


def strength_of_schedule(d: pd.DataFrame, rating_col: str, race_col: str = "raceid", horse_col: str = "horse_name",
                         date_col: str = "race_date", time_col: str = "race_time", window: int = 3) -> pd.DataFrame:
    """FSS_qual = IQM of opponents' pre-race ratings; SoS_L{w} over prior runs;
    sos_vs_today = FSS_qual(today) − SoS_L{w} (class move)."""
    out = d.copy()
    r = pd.to_numeric(out[rating_col], errors="coerce")
    grp = r.groupby(out[race_col])
    tot = grp.transform("sum"); cnt = grp.transform("count")
    out["fss_qual"] = (tot - r.fillna(0)) / (cnt - r.notna().astype(int)).replace(0, np.nan)  # mean of opponents
    out = out.sort_values([horse_col, date_col, time_col])
    prior = out.groupby(horse_col)["fss_qual"].shift(1)
    out[f"sos_L{window}"] = prior.groupby(out[horse_col]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    out["sos_vs_today"] = out["fss_qual"] - out[f"sos_L{window}"]
    return out.sort_index()


def within_race_transforms(d: pd.DataFrame, cols, race_col: str = "raceid", suffixes=("z", "rankpct", "vs_max")) -> pd.DataFrame:
    """Part 8.1: x_z, x_rankpct, x_vs_max for each column."""
    out = d.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce"); g = x.groupby(out[race_col])
        if "z" in suffixes: out[f"{c}_z"] = (x - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        if "rankpct" in suffixes:
            n = g.transform("count"); out[f"{c}_rankpct"] = (g.rank(ascending=False, method="average") - 1) / (n - 1).replace(0, np.nan)
        if "vs_max" in suffixes: out[f"{c}_vs_max"] = x - g.transform("max")
    return out
