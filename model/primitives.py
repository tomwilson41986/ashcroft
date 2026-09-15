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
    bl_censored + five flags          eased / pulled up / tailed off / hampered / slipped (§1.5.f)

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
from model.perf_figures import parse_beaten_lengths

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
    "slipped_flag": re.compile(r"slipped|lost action|lost (?:its |his |her )?action|stumbl|fell|brought down|unseated", re.I),
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
                       comment_col: str = "comment", race_type_col: str = "race_type", lps: float | None = None,
                       going_col: str = "going_description", code_col: str = "race_code",
                       censor_policy: str = "missing", geometry: bool = True, race_col: str = "raceid") -> pd.DataFrame:
    """The §1.7 primitive vector for every past run.

    ``lps`` fixes the lengths-per-second scale; leave it None to take the BHA
    scale for each row's going and code. ``censor_policy`` follows §1.5.f:
    ``"missing"`` drops the margin for a censored run (option 3, the historical
    default here), ``"ceiling"`` substitutes the truncation ceiling and leaves
    ``bl_censored`` to tell the model to discount it (option 2, the spec's
    pragmatic default), ``"raw"`` leaves the recorded margin alone.

    ``geometry`` adds the §1.5.a cluster geometry, which needs a race key —
    ``race_col`` or the date/time/track the key is rebuilt from. Everything
    this returns describes the run it is computed from; see
    ``POST_RACE_PRIMITIVES``.
    """
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
    scale = float(lps) if lps is not None else lps_scale(d[going_col] if going_col in d.columns else None,
                                                         d[code_col] if code_col in d.columns else None)
    d["lps"] = scale
    d["bl_seconds"] = bl / scale
    if win_time_col in d.columns:
        wt = pd.to_numeric(d[win_time_col], errors="coerce").values
        d["bl_pct_wintime"] = np.where(wt > 0, d["bl_seconds"].values / wt * 100.0, np.nan)
        # No individual times in the schema, so the horse's time is the winner's
        # plus the margin converted through LPS (the §1.5.b reconstruction).
        d["bl_prop_figure"] = proportion_beaten_figure(wt, wt + d["bl_seconds"].values, dist_m)
    if geometry:
        d = add_margin_geometry(d, race_col=race_col, pos_col=pos_col, beaten_col=beaten_col)
    if comment_col in d.columns:
        d = comment_flags(d, comment_col)
        if censor_policy == "missing":
            d.loc[d["bl_censored"] == 1, "dabl"] = np.nan   # option 3 of §1.5.f; position primitives stay
        elif censor_policy == "ceiling":
            ceiling = truncate_bl(np.full(len(d), np.inf))  # 15·tanh(∞) = the truncation ceiling
            d.loc[d["bl_censored"] == 1, "dabl"] = dist_adjust_bl(ceiling, dist_m)[d["bl_censored"].values == 1]
        elif censor_policy != "raw":
            raise ValueError(f"censor_policy must be 'missing', 'ceiling' or 'raw', got {censor_policy!r}")
    return d


def add_margin_geometry(df: pd.DataFrame, race_col: str = "raceid", pos_col: str = "placing_numerical",
                        beaten_col: str = "total_dst_bt", winner_gap_ahead: float = 0.0,
                        last_gap_behind: float = 0.0) -> pd.DataFrame:
    """§1.5.a: the shape of the finish, not just the margin (D7–D10, D12).

    ``total_dst_bt`` is cumulative distance behind the winner, so the margin to
    the horse immediately ahead is own minus the next-ahead *finishing
    position* within the same race::

        bl_won_by     winning margin (race-level, 0 on a dead-heated win)
        bl_signed     +bl_won_by for the winner, −bl_winner for everyone else
        bl_next       margin to the horse immediately ahead   (= bl_gap_ahead)
        bl_behind     margin to the horse immediately behind   (= bl_gap_behind)
        bl_isolation  bl_gap_behind − bl_gap_ahead
        bl_to_last    lengths ahead of the last-placed finisher
        bl_spread     first-to-last lengths in the race (race-level)

    A horse beaten four lengths in a blanket finish is in a different position
    from one beaten four lengths of which three and a half are the gap to the
    horse in front; ``bl_isolation`` is positive when the horse was clear of
    those behind and close to those ahead, i.e. travelling with the principals.

    Dead heats share a finishing position, so they share a cluster and get the
    same gaps. Rows with no numeric finishing position — non-runners, fallers,
    pulled up — take no part in the ordering and get NaN geometry; ``finished``
    and ``n_finishers`` record that. Margins are floored at zero against the
    odd non-monotone published figure.

    POST-RACE. Every column here describes the race it is computed from. Use
    the lagged aggregates (``bl_isolation_mean_L3`` and friends) as features.
    """
    d = df.copy()
    key = ensure_race_key(d, race_col)
    place = pd.to_numeric(d[pos_col], errors="coerce")
    bl = d["bl_winner"] if "bl_winner" in d.columns else d[beaten_col].map(parse_beaten_lengths).where(place != 1, 0.0)
    bl = pd.to_numeric(bl, errors="coerce").astype(float)
    d["bl_winner"] = bl.values

    ok = place.notna() & (place > 0) & bl.notna()
    d["finished"] = ok.astype(int)
    d["n_finishers"] = ok.astype(int).groupby(key).transform("sum")
    d["is_dead_heat"] = (ok & (place.where(ok).groupby([key, place]).transform("size") > 1)).astype(int)

    # one row per (race, finishing position): the finish's distinct clusters
    u = (pd.DataFrame({"_r": key[ok], "_p": place[ok], "_b": bl[ok]})
         .groupby(["_r", "_p"], sort=True)["_b"].max().reset_index())
    g = u.groupby("_r", sort=False)["_b"]
    u["_ahead"] = g.diff().clip(lower=0.0)
    u["_behind"] = (-g.diff(-1)).clip(lower=0.0)
    u["_max"] = g.transform("max")
    u["_i"] = u.groupby("_r", sort=False).cumcount()

    # positional left-merge: pandas keeps the left frame's row order, and
    # (_r, _p) is unique in u, so no row is duplicated or reordered
    m = pd.DataFrame({"_r": key.values, "_p": place.values}).merge(u, on=["_r", "_p"], how="left")
    fin = ok.values
    d["bl_gap_ahead"] = np.where(fin, m["_ahead"].fillna(winner_gap_ahead).values, np.nan)
    d["bl_gap_behind"] = np.where(fin, m["_behind"].fillna(last_gap_behind).values, np.nan)
    d["bl_next"] = d["bl_gap_ahead"]                      # §1.5.a names both
    d["bl_behind"] = d["bl_gap_behind"]
    d["bl_isolation"] = d["bl_gap_behind"] - d["bl_gap_ahead"]
    d["bl_to_last"] = np.where(fin, m["_max"].values - bl.values, np.nan)
    d["bl_spread"] = key.map(u.groupby("_r")["_max"].max()).values

    # the gap from the winner's cluster to the second is the winning margin;
    # a dead-heated win is a margin of zero however far back the third is
    won_by = key.map(u.loc[u["_i"] == 1].set_index("_r")["_ahead"])
    dead_heated_win = (ok & (place == 1)).groupby(key).transform("sum") > 1
    won_by = won_by.where(~dead_heated_win, 0.0)
    d["bl_won_by"] = won_by.values
    d["bl_signed"] = np.where(fin, np.where(place == 1, won_by.values, -bl.values), np.nan)
    return d


def harmonic_bl_par(N, P, normalise: bool = False):
    """§1.5.e theoretical par: E[BL(P, N)] ∝ H(N−1) − H(N−P).

    Under the independent-exponential running-time model behind the Harville
    formula, the gaps between consecutive finishers are independent
    exponentials with rate proportional to the number of runners still behind,
    so the expected cumulative deficit at position P is a difference of
    harmonic numbers. Shape only — the units are the unknown scale parameter,
    which is why ``normalise`` divides through by the last-placed value to put
    the curve on [0, 1]. The spec is explicit that this is a cross-check on the
    shape of the empirical surface, not a replacement for it.
    """
    from scipy.special import digamma

    N = np.asarray(N, float); P = np.asarray(P, float)
    H = lambda m: np.where(m >= 1, digamma(np.maximum(m, 0) + 1) + np.euler_gamma, 0.0)   # exact harmonic number
    par = H(N - 1) - H(N - P)
    par = np.where((N > 1) & (P >= 1) & (P <= N), par, np.nan)
    return par / np.where(N > 1, H(N - 1), np.nan) if normalise else par


class BeatenLengthsPar:
    """Empirical par surfaces for the margin and the winning margin (§1.5.e).

    Expected beaten lengths at a given finishing position grows mechanically
    with field size — 6th of 8 and 6th of 20 carry different expected deficits,
    and the difference is structural, not performance. The winning margin moves
    the other way: more runners means a higher chance someone is close, so a
    three-length win in a twenty-runner handicap is the bigger performance and
    raw winning margin gets that backwards.

    Two surfaces, both median-based and both fitted on a training window and
    applied forward:

    ``bl_par(P, N, dist, going, class)``
        median DABL by finishing position × field-size band. Most
        (P, N, going, class) cells are thin, so the base surface is fitted over
        (band, P) alone and smoothed monotone in P with isotonic regression
        (weighted by cell counts, which also lets it extrapolate to positions
        the window never saw); the context cells are then shrunk toward that
        base with weight ``k``, the standard §P3 credibility form.
    ``won_by_par(N)``
        median winning margin by field size, smoothed monotone *decreasing*
        in N.

    Produces ``bl_vs_par`` (headline field-size-adjusted margin, positive =
    beaten less far than typical for the slot), ``bl_par_ratio``,
    ``bl_per_position`` and ``won_by_vs_par``.
    """

    def __init__(self, n_bands=(2, 5, 8, 11, 15, 100), dist_bands=(0, 7, 9.5, 12, 16, 100),
                 k: float = 25.0, smoother: str = "isotonic", context: bool = True):
        self.n_bands = list(n_bands); self.dist_bands = list(dist_bands)
        self.k = float(k); self.smoother = smoother; self.context = context
        self.table_ = None; self.cell_ = None; self.won_by_ = None; self._models = {}

    # -- keys ---------------------------------------------------------------
    def _band(self, N):
        return pd.cut(pd.Series(np.asarray(N, float)), self.n_bands, right=True, labels=False)

    def _context(self, df: pd.DataFrame, dist_col: str, going_col: str, class_col: str) -> pd.Series:
        """dist band × going group × class, as one string key. Missing columns
        simply drop out of the key rather than voiding the whole cell."""
        parts = []
        if self.context and dist_col in df.columns:
            parts.append(pd.cut(pd.to_numeric(df[dist_col], errors="coerce"), self.dist_bands,
                                right=True, labels=False).astype("string").fillna("?"))
        if self.context and going_col in df.columns:
            parts.append(going_group(df[going_col]).astype("string"))
        if self.context and class_col in df.columns:
            parts.append(df[class_col].astype("string").fillna("?"))
        if not parts:
            return pd.Series("*", index=df.index, dtype=object)
        out = parts[0].astype(str)
        for p in parts[1:]:
            out = out + "|" + p.astype(str)
        return out

    # -- fitting ------------------------------------------------------------
    def _monotone(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, increasing: bool):
        """Weighted monotone smoother over one axis; falls back to a running
        max/min if scikit-learn is unavailable."""
        if self.smoother == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
            except ImportError:                                     # pragma: no cover
                pass
            else:
                return IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x, y, sample_weight=w)
        order = np.argsort(x)
        run = np.maximum.accumulate(y[order]) if increasing else np.minimum.accumulate(y[order])
        xs, ys = x[order], run

        class _Step:
            def predict(self, q):
                return np.interp(np.asarray(q, float), xs, ys)
        return _Step()

    def fit(self, df: pd.DataFrame, pos_col: str = "placing_numerical", n_col: str = "number_of_runners",
            dabl_col: str = "dabl", won_by_col: str = "bl_won_by", dist_col: str = "dist_furlongs",
            going_col: str = "going_description", class_col: str = "race_class", race_col: str = "raceid"):
        d = pd.DataFrame({"P": pd.to_numeric(df[pos_col], errors="coerce"),
                          "b": self._band(df[n_col].values).values,
                          "c": self._context(df, dist_col, going_col, class_col).values,
                          "v": pd.to_numeric(df[dabl_col], errors="coerce").values}).dropna(subset=["P", "b", "v"])

        base = d.groupby(["b", "P"])["v"].agg(["median", "count"]).reset_index()
        smooth = []; self._models = {}
        for b, grp in base.groupby("b"):
            m = self._monotone(grp["P"].values.astype(float), grp["median"].values,
                               grp["count"].values.astype(float), increasing=True)
            smooth.append(pd.DataFrame({"b": b, "P": grp["P"].values, "base": m.predict(grp["P"].values)}))
            self._models[b] = m
        base = base.merge(pd.concat(smooth, ignore_index=True), on=["b", "P"], how="left")
        self.table_ = base.set_index(["b", "P"])["base"]

        cell = d.groupby(["b", "P", "c"])["v"].agg(["median", "count"]).reset_index()
        cell = cell.merge(base[["b", "P", "base"]], on=["b", "P"], how="left")
        # §P3: a thin (position, field size, going, class) cell is shrunk back
        # toward the pooled surface rather than trusted
        cell["par"] = (cell["count"] * cell["median"] + self.k * cell["base"]) / (cell["count"] + self.k)
        cell = cell.sort_values(["b", "c", "P"])
        cell["par"] = cell.groupby(["b", "c"])["par"].cummax()      # monotone in P within context
        self.cell_ = cell.set_index(["b", "c", "P"])["par"]

        if won_by_col in df.columns:
            w = pd.DataFrame({"N": pd.to_numeric(df[n_col], errors="coerce"),
                              "r": ensure_race_key(df, race_col),
                              "v": pd.to_numeric(df[won_by_col], errors="coerce")}).dropna()
            w = w.drop_duplicates("r").groupby("N")["v"].agg(["median", "count"]).reset_index()
            if len(w):
                self.won_by_ = self._monotone(w["N"].values.astype(float), w["median"].values,
                                              w["count"].values.astype(float), increasing=False)
        return self

    # -- application --------------------------------------------------------
    def par(self, N, P, context=None) -> np.ndarray:
        b = self._band(N).values
        P = pd.to_numeric(pd.Series(P), errors="coerce").values
        out = self.table_.reindex(pd.MultiIndex.from_arrays([b, P])).values.astype(float)
        if context is not None and self.cell_ is not None:
            fine = self.cell_.reindex(pd.MultiIndex.from_arrays([b, np.asarray(context), P])).values.astype(float)
            out = np.where(np.isnan(fine), out, fine)
        # positions the fit window never saw fall back to the smoother
        miss = np.isnan(out) & ~np.isnan(P) & ~pd.isna(b)
        if miss.any() and self._models:
            fill = np.array([self._models[bb].predict([pp])[0] if bb in self._models else np.nan
                             for bb, pp in zip(b[miss], P[miss])])
            out[miss] = fill
        return out

    def won_by_par(self, N) -> np.ndarray:
        if self.won_by_ is None:
            return np.full(len(np.atleast_1d(N)), np.nan)
        n = pd.to_numeric(pd.Series(np.asarray(N)), errors="coerce").values.astype(float)
        out = np.full(len(n), np.nan)
        ok = ~np.isnan(n)
        if ok.any():
            out[ok] = self.won_by_.predict(n[ok])
        return out

    def transform(self, df: pd.DataFrame, pos_col: str = "placing_numerical", n_col: str = "number_of_runners",
                  dabl_col: str = "dabl", won_by_col: str = "bl_won_by", dist_col: str = "dist_furlongs",
                  going_col: str = "going_description", class_col: str = "race_class") -> pd.DataFrame:
        d = df.copy()
        ctx = self._context(d, dist_col, going_col, class_col).values if self.context else None
        P = pd.to_numeric(d[pos_col], errors="coerce")
        d["bl_par"] = self.par(d[n_col].values, P.values, context=ctx)
        v = pd.to_numeric(d[dabl_col], errors="coerce")
        d["bl_vs_par"] = d["bl_par"] - v                     # positive = beaten less far than typical for that slot
        d["bl_par_ratio"] = v / d["bl_par"].replace(0, np.nan)
        d["bl_per_position"] = v / (P - 1).where(P > 1)      # mean gap per place lost; undefined for the winner
        if won_by_col in d.columns:
            d["won_by_par"] = self.won_by_par(d[n_col].values)
            d["won_by_vs_par"] = pd.to_numeric(d[won_by_col], errors="coerce") - d["won_by_par"]
            # the winning margin belongs to the winner's performance: carried
            # race-wide it would credit the beaten horses with the win as well
            d["won_by_vs_par_win"] = d["won_by_vs_par"].where(P == 1)
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
                      aggs=("mean", "iqm", "max", "min", "slope"), iqm_min_window: int = 5,
                      par: float = 0.0) -> pd.DataFrame:
    """{col}_{agg}_L{w} and {col}_{agg}_career from PRIOR runs only; plus
    {col}_wmean_career weighted by FSS credibility when ``weight_col`` is set.

    Aggregators (§3.2): mean, iqm, max, min, slope on run index, slope on date,
    first value, and the count / share of prior runs above ``par`` — the
    primitives here are par-centred, so ``par`` defaults to zero. ``iqm`` and
    ``slope`` need a few points to mean anything and are skipped for windows
    below ``iqm_min_window``; L3 is opened up by passing ``iqm_min_window=3``.

    Lag-safe by construction: the group key is the horse, which runs once in a
    race, and every statistic is taken over ``shift(1)``.
    """
    d = df.sort_values([horse_col, date_col, time_col]).copy()
    g = d.groupby(horse_col, group_keys=False)
    flat = lambda r: r.reset_index(level=0, drop=True)
    prior_runs = g.cumcount()
    # rows are contiguous and in order per horse after the sort, so a global
    # counter increments by one per run within every horse
    run_index = pd.Series(np.arange(len(d), dtype=float), index=d.index)
    new: dict[str, pd.Series] = {}                 # one concat at the end; column-at-a-time fragments the frame
    for c in cols:
        s = g[c].shift(1)
        new[f"{c}_last"] = s
        hs = s.groupby(d[horse_col])
        for w in windows:
            r = hs.rolling(w, min_periods=1)
            if "mean" in aggs: new[f"{c}_mean_L{w}"] = flat(r.mean())
            if "max" in aggs: new[f"{c}_max_L{w}"] = flat(r.max())
            if "min" in aggs: new[f"{c}_min_L{w}"] = flat(r.min())
            if "iqm" in aggs and w >= iqm_min_window: new[f"{c}_iqm_L{w}"] = flat(r.apply(_iqm, raw=True))
            if "slope" in aggs and w >= iqm_min_window:
                new[f"{c}_slope_L{w}"] = _rolling_slope(s, run_index, d[horse_col], w)
            if "slope_date" in aggs and w >= iqm_min_window:
                days = (pd.to_datetime(g[date_col].shift(1), errors="coerce") - pd.Timestamp("2000-01-01")).dt.days.astype(float)
                new[f"{c}_slopedate_L{w}"] = _rolling_slope(s, days, d[horse_col], w)
            if "above_par" in aggs:
                ab = (s > par).where(s.notna()).astype(float).groupby(d[horse_col]).rolling(w, min_periods=1)
                new[f"{c}_n_above_par_L{w}"] = flat(ab.sum())
                new[f"{c}_pct_above_par_L{w}"] = flat(ab.mean())
        e = hs.expanding(min_periods=1)
        new[f"{c}_mean_career"] = flat(e.mean())
        new[f"{c}_max_career"] = flat(e.max())
        if "min" in aggs: new[f"{c}_min_career"] = flat(e.min())
        # "first value" skips missing figures but stays NaN on the horse's own
        # first run, where there is no prior value to be first
        if "first" in aggs: new[f"{c}_first"] = s.groupby(d[horse_col]).transform("first").where(prior_runs > 0)
        if weight_col and weight_col in d.columns:
            wv = g[weight_col].shift(1)
            num = (s * wv).groupby(d[horse_col]).cumsum(); den = wv.where(s.notna()).groupby(d[horse_col]).cumsum()
            new[f"{c}_wmean_career"] = num / den.replace(0, np.nan)
    new["runs_count"] = prior_runs
    d = pd.concat([d.drop(columns=[k for k in new if k in d.columns]), pd.DataFrame(new, index=d.index)], axis=1)
    return d.sort_index()


def _rolling_slope(values: pd.Series, t: pd.Series, horse: pd.Series, window: int, min_points: int = 3) -> pd.Series:
    """OLS slope of ``values`` on ``t`` over a rolling window (§3.2).

    cov(t, x) / var(t) from rolling sums rather than a Python fit per window:
    the same number, two orders of magnitude faster over a database-sized
    frame. With ``t`` the run index this is the slope on run number; with ``t``
    in days it is the slope on date, so a horse whose runs are bunched and one
    whose runs are spread over two years do not get the same trajectory from
    the same sequence of figures.
    """
    ok = values.notna() & t.notna()
    tm = t.where(ok); x = values.where(ok)
    roll = lambda z: z.groupby(horse).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    n = roll(ok.astype(float))
    st, sx, stx, stt = roll(tm), roll(x), roll(tm * x), roll(tm * tm)
    den = (n * stt - st ** 2).replace(0, np.nan)
    return ((n * stx - st * sx) / den).where(n >= min_points)


def recency_weighted_mean(df: pd.DataFrame, col: str, lam: float = 1.0, horse_col: str = "horse_name",
                          date_col: str = "race_date", time_col: str = "race_time") -> pd.Series:
    """§3.1 first decay axis: mean of prior runs weighted w = exp(−λ·Δt/365).

    λ is a parameter, not a constant of nature — ``fit_decay_lambda`` fits it
    per primitive, and §IIA.1's Kalman filter generalises the whole scheme.
    """
    d = df[[horse_col, date_col, col]].copy()
    d["_t"] = pd.to_datetime(d[date_col], errors="coerce")
    order = df.sort_values([horse_col, date_col, time_col]).index
    pos = df.index.get_indexer(order)
    horses = d[horse_col].astype(str).values[pos]
    days = (d["_t"].values[pos] - np.datetime64("2000-01-01")).astype("timedelta64[D]").astype(float)
    vals = pd.to_numeric(d[col], errors="coerce").values[pos]
    out = np.full(len(pos), np.nan)
    cur = None; s_w = s_wx = 0.0; last = 0.0
    for i in range(len(pos)):
        if horses[i] != cur:
            cur = horses[i]; s_w = s_wx = 0.0; last = days[i]
        decay = np.exp(-lam * (days[i] - last) / 365.0) if s_w > 0 else 1.0
        s_w *= decay; s_wx *= decay; last = days[i]
        if s_w > 0:
            out[i] = s_wx / s_w
        if not np.isnan(vals[i]):
            s_w += 1.0; s_wx += vals[i]
    return pd.Series(out, index=order).reindex(df.index)


def run_decayed_mean(df: pd.DataFrame, col: str, gamma: float = 0.85, horse_col: str = "horse_name",
                     date_col: str = "race_date", time_col: str = "race_time") -> pd.Series:
    """§3.1 second decay axis: weights γ^k in *runs* since, not days.

    A horse with three runs in the last month is differently placed from one
    with three runs in the last year even when the calendar decay matches;
    both axes belong in the model.
    """
    d = df.sort_values([horse_col, date_col, time_col])
    prior = d.groupby(horse_col)[col].shift(1)
    out = prior.groupby(d[horse_col]).apply(lambda s: s.ewm(alpha=1.0 - gamma, ignore_na=True).mean())
    out = out.reset_index(level=0, drop=True) if isinstance(out.index, pd.MultiIndex) else out
    return out.reindex(df.index)


def fit_decay_lambda(df: pd.DataFrame, col: str, target_col: str | None = None, lambdas=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
                     horse_col: str = "horse_name", date_col: str = "race_date", time_col: str = "race_time") -> dict:
    """Fit λ per primitive (§3.1, P7): the decay whose weighted history best
    predicts the *next* run's outcome.

    The history is lagged, the target is the current run, so the search is
    honest; run it on a training window and carry the winner forward.
    """
    y = pd.to_numeric(df[target_col or col], errors="coerce")
    scores = {}
    for lam in lambdas:
        m = recency_weighted_mean(df, col, lam=lam, horse_col=horse_col, date_col=date_col, time_col=time_col)
        ok = m.notna() & y.notna()
        scores[float(lam)] = float(np.corrcoef(m[ok], y[ok])[0, 1]) if ok.sum() > 10 else np.nan
    best = max((v, k) for k, v in scores.items() if v == v)[1] if any(v == v for v in scores.values()) else np.nan
    return {"lambda": best, "scores": scores}


def add_dslr_features(df: pd.DataFrame, horse_col: str = "horse_name", date_col: str = "race_date",
                      time_col: str = "race_time", window: int = 10) -> pd.DataFrame:
    """§3.1/§3.4: days since last run and the counters around it (F3, F4, F18).

        dslr                 days since the horse's previous run
        dslr_cum_L10         cumulative DSLR over the last ten runs — "there is
                             no fitness like race fitness"
        dslr_mean_career     the horse's typical gap, from prior runs only
        dslr_vs_horse_norm   this gap relative to that typical gap: a horse off
                             90 days that always runs off 90 days is not the
                             same animal as one that always runs off 21
        runs_this_season     runs so far this season, excluding today
        days_since_first_run one of Benter's most significant factors, with no
                             obvious common-sense reason
    """
    d = df.sort_values([horse_col, date_col, time_col]).copy()
    dt = pd.to_datetime(d[date_col], errors="coerce")
    g = d.groupby(horse_col, group_keys=False)
    d["dslr"] = (dt - g[date_col].transform(lambda s: pd.to_datetime(s, errors="coerce").shift(1))).dt.days
    prior = g["dslr"].shift(1)
    d[f"dslr_cum_L{window}"] = prior.groupby(d[horse_col]).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
    d["dslr_mean_career"] = prior.groupby(d[horse_col]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    d["dslr_vs_horse_norm"] = d["dslr"] / d["dslr_mean_career"].replace(0, np.nan)
    season = dt.dt.year                                   # calendar year: crude, and the only split the schema supports
    d["season"] = season
    d["runs_this_season"] = d.groupby([d[horse_col], season]).cumcount()
    d["days_since_first_run"] = (dt - g[date_col].transform(lambda s: pd.to_datetime(s, errors="coerce").min())).dt.days
    return d.sort_index()


def season_aggregates(df: pd.DataFrame, cols, horse_col: str = "horse_name", date_col: str = "race_date",
                      time_col: str = "race_time") -> pd.DataFrame:
    """§3.2 season windows: {col}_mean_this_season and _last_season, both from
    prior runs only. Season = calendar year; the schema carries no season
    marker and the turf/AW split does not follow one."""
    d = df.sort_values([horse_col, date_col, time_col]).copy()
    d["season"] = pd.to_datetime(d[date_col], errors="coerce").dt.year
    for c in cols:
        prior = d.groupby(horse_col)[c].shift(1)
        this = prior.groupby([d[horse_col], d["season"]]).expanding(min_periods=1).mean()
        d[f"{c}_mean_this_season"] = this.reset_index(level=[0, 1], drop=True)
        # last season is complete before today's race starts, so the whole
        # season's mean is available without lagging inside it
        prev = (d.groupby([horse_col, "season"])[c].mean().rename("v").reset_index()
                 .assign(season=lambda x: x["season"] + 1).set_index([horse_col, "season"])["v"])
        d[f"{c}_mean_last_season"] = prev.reindex(pd.MultiIndex.from_arrays([d[horse_col], d["season"]])).values
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


def _loo_iqm(values: np.ndarray) -> np.ndarray:
    """Interquartile mean of every element's *opponents*, one race at a time.

    §2.1b puts the IQM first among the FSS_qual aggregators — it is robust and
    the internal research has it accounting for more than half the variance in
    segmented regression. There is no closed form for the leave-one-out version
    so this loops over the race's runners; races are small, but the cost is
    linear in runners rather than free, so callers opt in via ``qual_aggs``.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    for i in range(n):
        rest = np.delete(values, i)
        rest = rest[~np.isnan(rest)]
        if not len(rest):
            continue
        if len(rest) < 4:
            out[i] = rest.mean(); continue
        q1, q3 = np.quantile(rest, [0.25, 0.75])
        mid = rest[(rest >= q1) & (rest <= q3)]
        out[i] = mid.mean() if len(mid) else rest.mean()
    return out


def strength_of_schedule(d: pd.DataFrame, rating_col: str, race_col: str = "raceid", horse_col: str = "horse_name",
                         date_col: str = "race_date", time_col: str = "race_time", window: int = 3,
                         qual_agg: str = "mean", qual_aggs=("mean", "max"), lam: float | None = None,
                         prefix: str = "") -> pd.DataFrame:
    """§2.1b field quality strength and §2.3 strength of schedule.

        fss_qual_mean   leave-one-out mean of the opponents' pre-race ratings
        fss_qual_iqm    leave-one-out interquartile mean (§2.1b's first choice)
        fss_qual_max    the best opponent in the race
        race_rating_sd  dispersion of the ratings in the race, with _iqr — a
                        race where every runner is within 3lb is a different
                        prediction problem from one with a 25lb spread
        fss_qual        whichever aggregator ``qual_agg`` names
        sos_L{w}        recency-ordered rolling mean of FSS_qual over the
                        horse's prior runs; sos_wL{w} is the λ-decayed version
        sos_career      the same over the whole career
        sos_delta       sos_L{w} − sos_career: stepping up or dropping down
        sos_vs_today    FSS_qual(today) − sos_L{w}: the class move for today,
                        "one of the highest-value features in the whole suite"

    Ratings are pre-race, so the opponents' ratings are legitimate inputs; the
    horse's own history is lagged per horse, which is safe because a horse runs
    once in a race. ``prefix`` lets several rating variants (OR, Timeform,
    class rating) coexist in one frame, as §2.3 asks.
    """
    out = d.copy()
    r = pd.to_numeric(out[rating_col], errors="coerce")
    grp = r.groupby(out[race_col])
    aggs = set(qual_aggs) | {qual_agg}

    tot = grp.transform("sum"); cnt = grp.transform("count")
    out[f"{prefix}fss_qual_mean"] = (tot - r.fillna(0)) / (cnt - r.notna().astype(int)).replace(0, np.nan)
    if "max" in aggs:
        mx = grp.transform("max")
        second = r.where(r.groupby(out[race_col]).rank(method="min", ascending=False) == 2).groupby(out[race_col]).transform("max")
        unique_top = (r.eq(mx)).groupby(out[race_col]).transform("sum").eq(1)
        out[f"{prefix}fss_qual_max"] = mx.where(~(r.eq(mx) & unique_top), second)
    if "iqm" in aggs:
        out[f"{prefix}fss_qual_iqm"] = r.groupby(out[race_col], group_keys=False).transform(
            lambda s: pd.Series(_loo_iqm(s.values.astype(float)), index=s.index))
    out[f"{prefix}race_rating_sd"] = grp.transform("std")
    q = grp.quantile([0.25, 0.75]).unstack()                  # per race, not per row: a lambda here is a Python
    out[f"{prefix}race_rating_iqr"] = out[race_col].map(q[0.75] - q[0.25]).values   # loop over every race
    out[f"{prefix}fss_qual"] = out[f"{prefix}fss_qual_{qual_agg}"]

    out = out.sort_values([horse_col, date_col, time_col])
    prior = out.groupby(horse_col)[f"{prefix}fss_qual"].shift(1)
    hs = prior.groupby(out[horse_col])
    out[f"{prefix}sos_L{window}"] = hs.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    out[f"{prefix}sos_career"] = hs.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    out[f"{prefix}sos_delta"] = out[f"{prefix}sos_L{window}"] - out[f"{prefix}sos_career"]
    out[f"{prefix}sos_vs_today"] = out[f"{prefix}fss_qual"] - out[f"{prefix}sos_L{window}"]
    out = out.sort_index()
    if lam is not None:
        out[f"{prefix}sos_wL{window}"] = recency_weighted_mean(out, f"{prefix}fss_qual", lam=lam, horse_col=horse_col,
                                                              date_col=date_col, time_col=time_col)
    return out


def within_race_transforms(d: pd.DataFrame, cols, race_col: str = "raceid", suffixes=("z", "rankpct", "vs_max"),
                           pos_col: str | None = None) -> pd.DataFrame:
    """Part 8.1 / §1.5.d: within-race normalisations for each column.

        z            (x − race mean) / race SD
        rankpct      rank within the race, scaled to [0, 1]
        vs_max       x − race max
        vs_median    x − race median, robust to one tailed-off horse
        share        x / Σ_race x, the share of the total race deficit
        norm_spread  (x − race min) / race spread; 0 = winner, 1 = last
        vs_2nd       x − the runner-up's x

    §1.5.d is explicit that ``vs_median`` and ``rankpct`` are preferable to
    ``z`` in small fields — with six runners the race SD is estimated from
    almost nothing — and that ``norm_spread`` must be computed on a
    *truncated* margin or one detached horse compresses the rest toward zero.
    ``pos_col`` names the finishing-position column used to identify the
    runner-up; without it the second-smallest value is used instead.
    """
    out = d.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce"); g = x.groupby(out[race_col])
        if "z" in suffixes: out[f"{c}_z"] = (x - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        if "rankpct" in suffixes:
            n = g.transform("count"); out[f"{c}_rankpct"] = (g.rank(ascending=False, method="average") - 1) / (n - 1).replace(0, np.nan)
        if "vs_max" in suffixes: out[f"{c}_vs_max"] = x - g.transform("max")
        if "vs_median" in suffixes: out[f"{c}_vs_median"] = x - g.transform("median")
        if "share" in suffixes: out[f"{c}_share"] = x / g.transform("sum").replace(0, np.nan)
        if "norm_spread" in suffixes:
            lo = g.transform("min"); spread = (g.transform("max") - lo).replace(0, np.nan)
            out[f"{c}_norm_spread"] = (x - lo) / spread
        if "vs_2nd" in suffixes:
            if pos_col is not None and pos_col in out.columns:
                p = pd.to_numeric(out[pos_col], errors="coerce")
                second = x.where(p == 2).groupby(out[race_col]).transform("max")
            else:
                second = x.where(g.rank(method="min") == 2).groupby(out[race_col]).transform("max")
            out[f"{c}_vs_2nd"] = x - second
    return out


# ---------------------------------------------------------------------------
# Margin normalisation, aggregation and distributional hygiene (§1.5.d/g, 8.2)
# ---------------------------------------------------------------------------

#: The six §1.5.d names, in the spec's order.
MARGIN_NORMALISATIONS = ["bl_z", "bl_vs_median", "bl_rankpct", "bl_share", "bl_norm_spread", "bl_vs_2nd"]

#: The §1.5.g margin aggregates that carry most weight. All of them are lagged
#: summaries of the horse's PREVIOUS runs, which is what makes the post-race
#: geometry below legitimate as an input.
MARGIN_AGGREGATE_FEATURES = ["bl_vs_par_iqm_L5", "bl_vs_par_max_career", "bl_vs_par_min_L10", "bl_vs_par_slope_L5",
                             "bl_pct_wintime_iqm_L3", "bl_isolation_mean_L3", "won_by_vs_par_max"]


def add_margin_normalisations(df: pd.DataFrame, col: str = "dabl", race_col: str = "raceid",
                              pos_col: str = "placing_numerical") -> pd.DataFrame:
    """§1.5.d, D20: the six within-race margin normalisations, under the
    spec's names, computed from the truncated distance-adjusted margin.

    POST-RACE. Every one of these is a statement about the finish of the race
    it is computed from. They belong in a lagged aggregate, never in the
    feature vector for the race being predicted.
    """
    out = within_race_transforms(df, [col], race_col=race_col, pos_col=pos_col,
                                 suffixes=("z", "rankpct", "vs_median", "share", "norm_spread", "vs_2nd"))
    for name in MARGIN_NORMALISATIONS:
        out[name] = out[f"{col}_{name[3:]}"]
    return out


#: Which aggregators each margin primitive earns, as (windows, aggs,
#: iqm_min_window). §1.5.g names seven features, and computing the full
#: aggregator cross-product for every margin column instead would spend most of
#: its time on IQMs nobody reads.
MARGIN_AGGREGATION_PLAN = {
    "bl_vs_par": ((5, 10), ("mean", "iqm", "max", "min", "slope"), 5),
    "bl_pct_wintime": ((3,), ("mean", "iqm"), 3),
    "bl_isolation": ((3,), ("mean",), 3),
    "bl_signed": ((3,), ("mean", "max"), 5),
    "won_by_vs_par_win": ((), ("max",), 5),
}


def add_margin_aggregates(df: pd.DataFrame, horse_col: str = "horse_name", date_col: str = "race_date",
                          time_col: str = "race_time", plan: dict | None = None) -> pd.DataFrame:
    """§1.5.g: run the margin primitives through the Part 3 machinery.

    ``bl_vs_par_slope_L5`` is a trajectory feature and pairs with the Part 7
    ratings-change suite: a horse whose margins are shrinking while its
    official rating stands still is exactly the profile the handicapper has not
    caught up with. ``won_by_vs_par_max`` is taken over the horse's *wins*
    only — carried race-wide the winning margin would credit beaten horses
    with the win.

    Every feature here is a summary of the horse's PREVIOUS runs, which is what
    makes the post-race geometry underneath it legitimate.
    """
    out = df
    for col, (windows, aggs, iqm_min) in (plan or MARGIN_AGGREGATION_PLAN).items():
        if col not in out.columns:
            continue
        out = aggregate_history(out, [col], windows=windows, horse_col=horse_col, date_col=date_col,
                                time_col=time_col, weight_col=None, aggs=aggs, iqm_min_window=iqm_min)
    if "won_by_vs_par_win_max_career" in out.columns:
        out["won_by_vs_par_max"] = out["won_by_vs_par_win_max_career"]
    return out


def skew_aware_transform(x, name: str = "x", skew_threshold: float = 1.0) -> tuple[np.ndarray, str]:
    """§1.5.c / 8.2: test the shape before transforming, then z-score.

    Margins are right-skewed even after truncation and distance adjustment, and
    z-scoring a skewed variable buys a mean of zero and nothing else. Returns
    the transformed, re-z-scored values and the name of the transform applied,
    so the feature dictionary can record which one it was.
    """
    v = pd.to_numeric(pd.Series(x), errors="coerce")
    ok = v.notna()
    if ok.sum() < 8:
        return v.values, "none"
    sk = float(v[ok].skew())
    if sk > skew_threshold and (v[ok] >= 0).all():
        t = np.sqrt(v) if sk < 2 * skew_threshold else np.log1p(v)
        kind = "sqrt" if sk < 2 * skew_threshold else "log1p"
    elif sk < -skew_threshold:
        t = v ** 2
        kind = "square"
    else:
        t, kind = v, "none"
    sd = t.std()
    return ((t - t.mean()) / (sd if sd else np.nan)).values, kind


# ---------------------------------------------------------------------------
# §2.4 PvS — performance relative to strength of schedule
# ---------------------------------------------------------------------------

class PerformanceVsSchedule:
    """§2.4 / E10: the residual of finishing position on strength of schedule.

    A horse with a mediocre record against strong opposition scores well; a
    horse with a flattering record against weak fields scores badly. The spec
    calls this "the single most important correction to naive form aggregates",
    and it is the piece a career mean of NMFP cannot express: two horses with
    identical form figures are not equal if one of them earned them in
    class-2 handicaps.

    Fitted the same way as the par surfaces — one OLS of ``nmfp`` on
    ``fss_qual`` over a training window, applied forward — so that the
    coefficients never see the races they are scoring.
    """

    def __init__(self, y_col: str = "nmfp", x_cols=("fss_qual",)):
        self.y_col = y_col; self.x_cols = list(x_cols); self.coef_ = None; self.n_ = 0

    def fit(self, df: pd.DataFrame, mask: pd.Series | None = None):
        d = df if mask is None else df[mask]
        y = pd.to_numeric(d[self.y_col], errors="coerce")
        X = np.column_stack([np.ones(len(d))] + [pd.to_numeric(d[c], errors="coerce").values for c in self.x_cols])
        ok = ~np.isnan(X).any(axis=1) & y.notna().values
        if ok.sum() <= len(self.x_cols) + 1:
            self.coef_ = np.zeros(len(self.x_cols) + 1); self.n_ = 0
            return self
        self.coef_ = np.linalg.lstsq(X[ok], y.values[ok], rcond=None)[0]
        self.n_ = int(ok.sum())
        return self

    def expected(self, df: pd.DataFrame) -> pd.Series:
        X = np.column_stack([np.ones(len(df))] + [pd.to_numeric(df[c], errors="coerce").values for c in self.x_cols])
        return pd.Series(X @ self.coef_, index=df.index)

    def residual(self, df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(df[self.y_col], errors="coerce") - self.expected(df)


def add_pvs_features(df: pd.DataFrame, y_col: str = "nmfp", x_cols=("fss_qual",), windows=(3, 5, 10),
                     horse_col: str = "horse_name", date_col: str = "race_date", time_col: str = "race_time",
                     fit_end: str | pd.Timestamp | None = None, model: PerformanceVsSchedule | None = None
                     ) -> tuple[pd.DataFrame, PerformanceVsSchedule]:
    """PvS as a feature block (E10) plus §3.3's ``lto_was_flattered`` (F14).

        {y}_expected      the fitted value given the strength of the schedule
        pvs_run           the residual for THIS run — POST-RACE
        pvs_mean_career   the horse's mean residual over its prior runs, which
                          is the feature §2.4 actually asks for
        pvs_mean_L{w}     the same over recent windows
        lto_was_flattered last run's residual: the horse ran above or below
                          what the strength of that race predicted

    ``fit_end`` restricts the regression to rows before that date, so a
    walk-forward can fit on the training side of the fold boundary and score
    forward.
    """
    if model is None:
        mask = None
        if fit_end is not None:
            mask = pd.to_datetime(df[date_col], errors="coerce") < pd.Timestamp(fit_end)
        model = PerformanceVsSchedule(y_col=y_col, x_cols=x_cols).fit(df, mask)
    out = df.copy()
    out[f"{y_col}_expected"] = model.expected(out)
    out["pvs_run"] = model.residual(out)
    out = aggregate_history(out, ["pvs_run"], windows=windows, horse_col=horse_col, date_col=date_col,
                            time_col=time_col, weight_col=None, aggs=("mean", "iqm", "min"))
    out["lto_was_flattered"] = out["pvs_run_last"]
    out = out.rename(columns={c: "pvs_" + c[len("pvs_run_"):] for c in out.columns
                              if c.startswith("pvs_run_") and c != "pvs_run_last"})
    return out, model


# ---------------------------------------------------------------------------
# §2.1c race strength, §2.2 effective field size, §2.6 placed flags
# ---------------------------------------------------------------------------

DEBUTANT_PRB2_PAR = 0.3050          # §2.1c: the internal research's debutant figure


def race_strength(df: pd.DataFrame, race_col: str = "raceid", horse_col: str = "horse_name",
                  date_col: str = "race_date", time_col: str = "race_time", window: int = 3,
                  debut_par: float = DEBUTANT_PRB2_PAR) -> pd.DataFrame:
    """§2.1c / E4: RaceStrength = √(LR_%RB²) · √(CR_%RB²), matched windows.

    The geometric mean of the horse's recent and career convex form, averaged
    over the field to describe the race. Debutants take the research's 30.50%
    par rather than dropping out, which is also §3.4's padding rule for thin
    records. Both inputs are the horse's own prior runs, so the race-level mean
    contains nothing from the race itself.

    The Hunter min-max iterated variant named in §2.1c is not implemented: the
    spec names it without giving the iteration.
    """
    d = df.sort_values([horse_col, date_col, time_col]).copy()
    if "prb2" not in d.columns:
        d["prb2"] = prb(d["number_of_runners"].values, d["placing_numerical"].values) ** 2
    prior = d.groupby(horse_col)["prb2"].shift(1)
    lr = prior.groupby(d[horse_col]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    cr = prior.groupby(d[horse_col]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    d["prb2_lr"] = lr.fillna(debut_par); d["prb2_cr"] = cr.fillna(debut_par)
    d["race_strength_runner"] = np.sqrt(d["prb2_lr"].clip(lower=0) * d["prb2_cr"].clip(lower=0))
    d = d.sort_index()
    key = ensure_race_key(d, race_col)
    d["race_strength"] = d["race_strength_runner"].groupby(key).transform("mean")
    d["race_strength_max"] = d["race_strength_runner"].groupby(key).transform("max")
    return d


def n_eff_from_probs(p, race_key) -> pd.Series:
    """§2.2 / E5: effective field size, exp(−Σ π ln π).

    STAGE C ONLY when ``p`` comes from market prices. A sixteen-runner race
    where one horse is 1/3 has an effective field size near two, which is a far
    better description of competitiveness than N — and a market-derived one, so
    it never enters Stage F. ``n_eff_from_ratings`` is the market-free version.
    """
    s = pd.to_numeric(pd.Series(p), errors="coerce")
    key = pd.Series(np.asarray(race_key), index=s.index)
    norm = s / s.groupby(key).transform("sum")
    ent = -(norm * np.log(norm.clip(lower=1e-12))).groupby(key).transform("sum")
    return np.exp(ent)


def connection_plc_fsa(df: pd.DataFrame, plc_col: str = "plc_fsa", trainer_col: str = "trainer",
                       jockey_col: str = "jockey_name", race_col: str = "raceid", min_races: int = 3) -> pd.DataFrame:
    """§2.6 / E12: the jockey's and trainer's field-size-adjusted placed rate.

    "Placed last time out" is field-size contaminated — places paid are 2 for
    N ≤ 4, 3 for 5–7 and 4 for big handicaps — so the flag means something
    different at every field size; ``plc_fsa`` centres it on the random-chance
    baseline. §2.6 asks for the rolling, recency-weighted version rather than a
    literal single last runner, because one observation is noise.

    Trainer and jockey repeat *inside* a race, so the rolling mean goes through
    ``race_lagged_expanding_mean``: a plain ``shift(1).expanding()`` on a
    trainer-sorted frame steps back to a stablemate in the same race, and since
    the feature is otherwise constant across the race that leak is the only
    within-race variation the model would see.
    """
    out = df.copy()
    for col, name in ((trainer_col, "trainer_lto_plc_fsa"), (jockey_col, "jockey_lto_plc_fsa")):
        if col in out.columns and plc_col in out.columns:
            out[name] = race_lagged_expanding_mean(out, col, plc_col, race_col=race_col, min_races=min_races)
    return out


def add_lto_context_features(df: pd.DataFrame, rating_col: str | None = "official_rating",
                             horse_col: str = "horse_name", date_col: str = "race_date",
                             time_col: str = "race_time") -> pd.DataFrame:
    """§3.3's remaining last-run context: ``lto_field_size``, ``lto_FSS_qual``,
    ``lto_n_eff_rating``, ``lto_wide_flag`` and ``lto_slow_start_flag``.

    Last run is both the most predictive single observation and the most
    overbet one, so features that say *when LTO is misleading* are where the
    edge is. The flags come from the previous race's in-running comment, which
    is available by the time today's race is priced.
    """
    d = df.sort_values([horse_col, date_col, time_col]).copy()
    g = d.groupby(horse_col)
    src = {"lto_field_size": "number_of_runners", "lto_FSS_qual": "fss_qual", "lto_n_eff_rating": "n_eff_rating",
           "lto_wide_flag": "wide_flag", "lto_slow_start_flag": "slow_start_flag", "lto_trouble_flag": "hampered_flag",
           "lto_going": "going_description", "lto_dist": "dist_furlongs", "lto_class": "race_class"}
    for name, col in src.items():
        if col in d.columns:
            d[name] = g[col].shift(1)
    if "lto_dist" in d.columns and "dist_furlongs" in d.columns:
        d["lto_dist_delta"] = pd.to_numeric(d["dist_furlongs"], errors="coerce") - pd.to_numeric(d["lto_dist"], errors="coerce")
    if rating_col and rating_col in d.columns:
        d["lto_or"] = g[rating_col].shift(1)
    return d.sort_index()


# ---------------------------------------------------------------------------
# Post-race registry
# ---------------------------------------------------------------------------

#: Columns this module computes from the CURRENT race's result. They are
#: legitimate inputs to a lagged feature and must never be features
#: themselves — the within-race geometry in particular reads like an ordinary
#: numeric feature and would carry enormous gain in training before collapsing
#: to nothing on a live card, which has no finishing positions yet.
#:
#: ``model.custom_metrics.POST_RACE_ONLY`` is the set ``train_bfsp`` asserts
#: against; union this into it before any of these names reaches a feature
#: list.
POST_RACE_PRIMITIVES = frozenset({
    # position and margin primitives for the race being predicted
    "prb", "prb2", "prb2_fsa", "nmfp", "plc_fsa", "bl_winner", "bl_trunc", "dabl",
    "bl_seconds", "bl_pct_wintime", "bl_prop_figure", "perf_lbs",
    # §1.5.a cluster geometry
    "bl_signed", "bl_won_by", "bl_next", "bl_behind", "bl_gap_ahead", "bl_gap_behind",
    "bl_isolation", "bl_to_last", "bl_spread", "finished", "is_dead_heat",
    # §1.5.e par comparisons for this run
    "bl_par", "bl_vs_par", "bl_par_ratio", "bl_per_position", "won_by_par",
    "won_by_vs_par", "won_by_vs_par_win",
    # §1.5.d within-race normalisations
    *MARGIN_NORMALISATIONS,
    # §1.5.f comment flags and §2.4's residual for this run
    "bl_censored", "eased_flag", "pulled_up_flag", "tailed_off_flag", "hampered_flag",
    "slipped_flag", "wide_flag", "slow_start_flag", "pvs_run", "nmfp_expected",
})
