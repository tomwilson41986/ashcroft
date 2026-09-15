"""
Performance primitives and transforms from the racing² master framework
(Section III, Parts 1–3, 7, 8) that Ashcroft's custom metrics do not yet carry.

Per run (Part 1):
    prb, prb2, prb2_par, prb2_fsa     % rivals beaten, squared, field-size par, adjusted
    nmfp                              normalised finishing position (= uniform-rank z / 3)
    fss_cred                          field-size credibility weight √((N−1)/(N+1))
    plc_fsa                           placed flag minus places/N (UK each-way terms)
    bl_trunc, dabl                    15·tanh(BL/15), distance-adjusted (÷ (dist_m/1000)^0.25)
    bl_pct_wintime                    lengths → seconds (LPS) ÷ winning time
    bl_vs_par                         DABL vs an empirical (position × field-size) par surface
    bl_censored                       eased / tailed-off / hampered flag from comments

Aggregation (Part 3): windows × {mean, iqm, max, min, slope}, FSS-credibility
weighting, run-count features; the "consistency floor" MIN over L10.

Ratings (Part 7): handicapper-gap features (well_in, mark_vs_form, tf_vs_or,
rating_dispersion); LTO-contradiction features (lto_vs_career,
lto_tfig_above_ability).

Context (Part 2): rating-based effective field size (Stage-F-legal),
strength of schedule and SoS_vs_today.

Transforms (Part 8.1): within-race z, rankpct and vs_max for any feature.

All aggregates are lag-safe (shift-1 per horse ordered by date/time). The
par surface is fitted on a training window and applied forward.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.abm.features import place_terms
from model.perf_figures import parse_beaten_lengths

LPS_STANDARD = 6.0  # lengths per second on standard going (BHA scale; re-fit locally)


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


CENSOR_RX = re.compile(r"eased|tailed off|pulled up|hampered|badly hampered|brought down|unseated|fell|slipped|"
                       r"not persevered|virtually pulled up|lost action|refused|ran out|saddle slipped", re.I)


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
