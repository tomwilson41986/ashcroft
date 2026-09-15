"""
Odds-derived metrics — Section III Part 4 of the racing² master framework,
plus the Stage C plumbing of Section IV. **Everything here is Stage C.**

Every quantity in this module is a function of the market's prices. Feeding
any of it to the fundamental model double-counts the market and destroys the
ΔR² measurement, which is the only number the framework treats as predictive
of profit (principle P5). `model.feature_registry.stage_of` classifies every
column produced here as "C", so `stage_f_columns` can never hand one to
Stage F by accident.

Base transformations (§4.1)
    pi_raw   = 1 / decimal odds
    pi       = pi_raw / Σ pi_raw          proportional overround removal, which
                                          is adequate on a liquid Betfair book
                                          running at 101-102%
    pi_shin  = Shin (1993) recovery       for *bookmaker* prices, where the
                                          margin is loaded onto the longshots,
                                          so proportional normalisation
                                          distorts exactly the tail we price

OTRR — odds to runner ratio (§4.2)
    OTRR = pi·N is a runner's implied probability as a multiple of an equal
    share of the race: 1.0 is exactly average, 4.0 in a 12-runner race means
    the market makes it four times the field average. The raw ratio is
    heavily right-skewed, so `ln_OTRR` is the form the Stage C logit wants.
    Variants: OTRR_rank, OTRR_norm (relative to the favourite), OTRR_gap_fav,
    OTRR_gap_next, is_fav / is_jt_fav / fav_rank.

Effective field size and OFSR (§2.2, §4.3)
    n_eff = exp(−Σ pi ln pi). Field size counts bodies; effective field size
    counts contenders. A twenty-runner handicap with one 1/3 shot is not a
    twenty-horse puzzle, and `number_of_runners` says it is.
    OFSR_eff  = n_eff / N    what fraction of the field is genuinely live
    OFSR_conc = pi_fav · N   the favourite's own OTRR: a concentration measure
    Both are race-level constants — they cancel in the softmax, and the spec
    names them as its primary *bet-selection* variables: model edge is
    systematically easier to find in some ranges of OFSR_eff than others.

Stage C plumbing (§IV.1, §IV.2)
    unratable_runner_rule   Benter's rule: a runner the fundamental model
                            cannot rate takes the market's probability and the
                            ratable runners share what is left; races with too
                            few ratable runners are skipped outright.
    benter_reliability      reliability diagrams in Benter's format (band, n,
    reliability_comparison  expected, actual, Z) for market / fundamental /
                            combined — the three-way table §IV.2 asks for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from model.lagsafe import ensure_race_key

EPS = 1e-12

RUNNER_ODDS_FEATURES = [
    "pi_raw", "pi_market", "OTRR", "ln_OTRR", "OTRR_rank", "OTRR_norm",
    "OTRR_gap_fav", "OTRR_gap_next", "is_fav", "is_jt_fav", "fav_rank",
]
RACE_ODDS_FEATURES = ["overround", "n_priced", "n_eff_market", "OFSR_eff", "OFSR_conc"]
ODDS_FEATURES = RUNNER_ODDS_FEATURES + RACE_ODDS_FEATURES

# Benter's reliability bands (Table 3 / 6 of "Computer Based Horse Race
# Handicapping and Wagering Systems"): narrow at the short end, where almost
# all of the staked money sits.
DEFAULT_BANDS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def _arr(x) -> np.ndarray:
    return pd.to_numeric(pd.Series(np.asarray(x).ravel()), errors="coerce").to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# §4.1 Base transformations
# ---------------------------------------------------------------------------

def implied_probs(odds) -> np.ndarray:
    """pi_raw = 1 / decimal odds. Prices at or below 1.0 are impossible -> NaN."""
    o = _arr(odds)
    return np.where(o > 1.0, 1.0 / np.where(o > 1.0, o, np.nan), np.nan)


def normalise(p) -> np.ndarray:
    """Proportional overround removal over one race's probability vector."""
    v = _arr(p)
    tot = np.nansum(v)
    return v / tot if tot > 0 else np.full(v.shape, np.nan)


def shin_book(p, z: float) -> np.ndarray:
    """Shin's *forward* map: the book quoted against true probabilities ``p`` by
    a bookmaker facing a fraction ``z`` of insider money.

        pi_i = sqrt(B · p_i · (z + (1−z)·p_i)),   sqrt(B) = Σ_i sqrt(p_i(z + (1−z)p_i))

    Useful for testing the recovery below and for reading off the margin the
    model implies: the booksum B exceeds 1 by more when z is larger, and the
    excess is loaded onto the longshots because the map is concave in p.
    """
    q = normalise(p)
    r = np.sqrt(q * (z + (1.0 - z) * q))
    return r * np.nansum(r)


def _shin_p(pi: np.ndarray, book: float, z: float) -> np.ndarray:
    z = min(max(float(z), 0.0), 1.0 - 1e-12)
    return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / book) - z) / (2.0 * (1.0 - z))


def shin_z(pi_raw) -> float:
    """The insider fraction z that makes Shin's recovered probabilities sum to 1.

    Zero for a fair or under-round book (a Betfair market after commission can
    sum below 1), in which case Shin degenerates to proportional normalisation.
    """
    pi = _arr(pi_raw)
    pi = pi[np.isfinite(pi) & (pi > 0)]
    book = float(pi.sum())
    if len(pi) < 2 or not np.isfinite(book) or book <= 1.0 + 1e-12:
        return 0.0
    f = lambda z: float(_shin_p(pi, book, z).sum() - 1.0)  # noqa: E731
    lo, hi = 1e-12, 1.0 - 1e-9
    if f(lo) <= 0:
        return 0.0
    if f(hi) >= 0:
        return float(hi)
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-13, maxiter=200))


def shin_probabilities(pi_raw) -> np.ndarray:
    """Shin (1993) recovery of true win probabilities from a bookmaker's book.

    **When to use it.** For *bookmaker* prices only. Shin models the margin as
    the bookmaker's defence against insider money, which makes the implied
    overround a convex function of the quoted probability — the longshot is
    loaded much harder than the favourite. Proportional normalisation assumes
    the margin is spread evenly, so on a bookmaker book it systematically
    over-states longshots, which is the tail a value model lives in. The
    evidence (Štrumbelj 2014; Clarke et al. 2017) favours Shin there.

    For liquid Betfair markets near the off, the overround runs at 101-102%
    and proportional normalisation is adequate; the framework lists bookmaker
    prices as a later fallback source, so this is here for that day and for
    any SP-based backtest. Shin also degenerates gracefully: at a booksum of
    1.0 the recovered z is 0 and the answer is the proportional one.
    """
    pi = _arr(pi_raw)
    out = np.full(pi.shape, np.nan)
    m = np.isfinite(pi) & (pi > 0)
    if m.sum() == 0:
        return out
    v = pi[m]
    book = float(v.sum())
    z = shin_z(v)
    p = v / book if z <= 0 else _shin_p(v, book, z)
    out[m] = p / p.sum()
    return out


def _shin_group(s: pd.Series) -> np.ndarray:
    v = s.to_numpy(dtype=float)
    out = np.full(v.shape, np.nan)
    m = np.isfinite(v) & (v > 0)
    if m.sum() >= 2:
        out[m] = shin_probabilities(v[m])
    elif m.sum() == 1:
        out[m] = 1.0
    return out


# ---------------------------------------------------------------------------
# §2.2 Effective field size
# ---------------------------------------------------------------------------

def effective_field_size(p) -> float:
    """N_eff = exp(−Σ p ln p) for one probability vector (perplexity of the book).

    Equals N for a uniform book and collapses toward 1 as one runner takes the
    mass: (0.5, 0.25, 0.25) gives exactly 2√2 ≈ 2.828, not 3.
    """
    q = normalise(p)
    q = q[np.isfinite(q) & (q > 0)]
    if q.size == 0:
        return float("nan")
    return float(np.exp(-np.sum(q * np.log(q))))


def n_eff_by_race(df: pd.DataFrame, p_col: str, race_col: str = "raceid") -> pd.Series:
    """Per-row effective field size of the race's probability vector.

    Works for any probability column: pass the market vector for `N_eff`
    (Stage C) or the fundamental model's for `N_eff_fund`.
    """
    race = _race_key(df, race_col)
    p = pd.to_numeric(df[p_col], errors="coerce")
    tot = p.groupby(race).transform("sum")
    q = p / tot.replace(0, np.nan)
    ent = -(q * np.log(q.clip(lower=EPS))).groupby(race).transform("sum")
    return np.exp(ent)


def _race_key(df: pd.DataFrame, race_col: str) -> pd.Series:
    if race_col in df.columns:
        return df[race_col].astype(str)
    return ensure_race_key(df, race_col)


# ---------------------------------------------------------------------------
# §4.2 / §4.3 The runner and race blocks
# ---------------------------------------------------------------------------

def add_odds_metrics(df: pd.DataFrame, price_col: str = "bfsp", prob_col: str | None = None,
                     race_col: str = "raceid", method: str = "proportional", prefix: str = "") -> pd.DataFrame:
    """Attach the whole Part 4 odds block to a runner-level frame. Stage C only.

    ``price_col`` is decimal odds; pass ``prob_col`` instead to start from an
    implied-probability column. ``method`` is "proportional" (Betfair) or
    "shin" (bookmaker prices). ``prefix`` namespaces the output columns so the
    block can be computed on more than one price snapshot, e.g.
    ``prefix="am_"`` on the morning WAP.

    N is the number of *priced* runners, so that the vector OTRR is measured
    against actually sums to one and OTRR = 1 really does mean average.
    """
    d = df.copy()
    race = _race_key(d, race_col)
    if prob_col is not None:
        pi_raw = pd.to_numeric(d[prob_col], errors="coerce").where(lambda s: s > 0)
    else:
        price = pd.to_numeric(d[price_col], errors="coerce").where(lambda s: s > 1.0)
        pi_raw = 1.0 / price
    book = pi_raw.groupby(race).transform("sum")
    if method == "shin":
        pi = pi_raw.groupby(race, group_keys=False).transform(_shin_group)
    elif method == "proportional":
        pi = pi_raw / book.replace(0, np.nan)
    else:
        raise ValueError(f"method must be 'proportional' or 'shin', got {method!r}")

    n = pi.notna().groupby(race).transform("sum").astype(float).replace(0, np.nan)
    otrr = pi * n
    pi_max = pi.groupby(race).transform("max")
    rank = otrr.groupby(race).rank(method="min", ascending=False)
    is_fav = (rank == 1).astype(float).where(rank.notna())
    n_top = (rank == 1).astype(float).groupby(race).transform("sum")

    p = prefix
    d[f"{p}pi_raw"] = pi_raw
    d[f"{p}pi_market"] = pi
    d[f"{p}overround"] = book
    d[f"{p}n_priced"] = n
    d[f"{p}OTRR"] = otrr
    d[f"{p}ln_OTRR"] = np.log(otrr.where(otrr > 0))
    d[f"{p}OTRR_rank"] = rank
    d[f"{p}fav_rank"] = rank
    d[f"{p}OTRR_norm"] = otrr / otrr.groupby(race).transform("max").replace(0, np.nan)
    d[f"{p}OTRR_gap_fav"] = np.log(pi_max.where(pi_max > 0)) - np.log(pi.where(pi > 0))
    d[f"{p}OTRR_gap_next"] = _gap_to_next_shorter(pi, race)
    d[f"{p}is_fav"] = is_fav
    d[f"{p}is_jt_fav"] = ((rank == 1) & (n_top > 1)).astype(float).where(rank.notna())
    d[f"{p}n_eff_market"] = np.exp(-(pi * np.log(pi.clip(lower=EPS))).groupby(race).transform("sum")).where(pi.notna().groupby(race).transform("any"))
    d[f"{p}OFSR_eff"] = d[f"{p}n_eff_market"] / n
    d[f"{p}OFSR_conc"] = pi_max * n
    return d


def _gap_to_next_shorter(pi: pd.Series, race: pd.Series) -> pd.Series:
    """ln(pi_i) − ln(pi of the next shorter-priced runner in the race).

    Non-positive by construction, and NaN for the favourite, which has nothing
    shorter than it. It measures how isolated a runner is in the betting: a
    second favourite a whisker behind reads very differently from one at half
    the favourite's implied chance.
    """
    tmp = pd.DataFrame({"_race": race.to_numpy(), "_pi": pi.to_numpy(dtype=float)})   # positional: the caller's index may repeat
    tmp["_o"] = tmp.groupby("_race")["_pi"].rank(method="first", ascending=False)
    srt = tmp.sort_values(["_race", "_o"], kind="stable")
    prev = srt.groupby("_race")["_pi"].shift(1).reindex(tmp.index).to_numpy(dtype=float)
    cur = tmp["_pi"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log(np.where(cur > 0, cur, np.nan)) - np.log(np.where(prev > 0, prev, np.nan))
    return pd.Series(out, index=pi.index)


def race_odds_summary(df: pd.DataFrame, price_col: str = "bfsp", race_col: str = "raceid",
                      method: str = "proportional") -> pd.DataFrame:
    """One row per race: the §4.3 selection variables (N, N_eff, OFSR_eff, OFSR_conc).

    These are the filters, not features: §4.3 is explicit that OFSR_eff is a
    bet-selection variable because edge is systematically easier to find in
    some ranges of it than in others.
    """
    d = add_odds_metrics(df, price_col=price_col, race_col=race_col, method=method)
    race = _race_key(d, race_col)
    d = d.assign(_race=race)
    out = d.groupby("_race").agg(n_priced=("n_priced", "first"), overround=("overround", "first"),
                                 n_eff_market=("n_eff_market", "first"), OFSR_eff=("OFSR_eff", "first"),
                                 OFSR_conc=("OFSR_conc", "first"), fav_pi=("pi_market", "max")).reset_index()
    return out.rename(columns={"_race": race_col})


# ---------------------------------------------------------------------------
# §IV.1 Unratable runners (Benter's rule)
# ---------------------------------------------------------------------------

def unratable_runner_rule(f, pi, ratable, groups, min_ratable: int = 2) -> dict:
    """Benter's rule: do not guess at a horse the fundamental model cannot rate.

    An unratable runner is assigned the market's implied probability; the
    ratable runners then share whatever probability mass is left, in
    proportion to their fundamental estimates. A race with fewer than
    ``min_ratable`` ratable runners is flagged for skipping — Benter skipped
    roughly 5% of races on exactly this rule, and a race made of guesses is
    not a race you can measure an edge in.

    Returns {"p": adjusted probability vector, "skip": per-row race-skip mask,
    "n_ratable": ratable runners in the race}.
    """
    codes, uniq = pd.factorize(pd.Series(np.asarray(groups).ravel()), sort=False)
    G = len(uniq)
    fv = np.nan_to_num(_arr(f), nan=0.0)
    pv = _arr(pi)
    r = np.asarray(ratable).ravel().astype(bool)
    if not (len(fv) == len(pv) == len(r) == len(codes)):
        raise ValueError("f, pi, ratable and groups must be the same length")

    pi_tot = np.bincount(codes, weights=np.nan_to_num(pv), minlength=G)[codes]
    pin = np.where(pi_tot > 0, pv / np.where(pi_tot > 0, pi_tot, 1.0), np.nan)
    mass_un = np.bincount(codes, weights=np.where(~r, np.nan_to_num(pin), 0.0), minlength=G)[codes]
    f_ok = np.where(r, np.clip(fv, 0.0, None), 0.0)
    f_tot = np.bincount(codes, weights=f_ok, minlength=G)[codes]
    share = np.where(f_tot > 0, f_ok / np.where(f_tot > 0, f_tot, 1.0), 0.0)
    p = np.where(r & (f_tot > 0), np.clip(1.0 - mass_un, 0.0, None) * share, pin)
    tot = np.bincount(codes, weights=np.nan_to_num(p), minlength=G)[codes]
    p = np.where(tot > 0, p / np.where(tot > 0, tot, 1.0), pin)
    n_ratable = np.bincount(codes, weights=r.astype(float), minlength=G)[codes]
    return {"p": p, "skip": n_ratable < min_ratable, "n_ratable": n_ratable}


# ---------------------------------------------------------------------------
# §IV.2 Calibration in Benter's format
# ---------------------------------------------------------------------------

def benter_reliability(y, p, bands=DEFAULT_BANDS) -> pd.DataFrame:
    """Reliability diagram in Benter's exact format: band, n, expected, actual, Z.

    Z = (actual − expected) / sqrt(Σ p(1−p)) is the standardised discrepancy
    of the Poisson-binomial count, so |Z| > 2 in a band is a real miscalibration
    rather than the eye reading noise off a plot.
    """
    yv, pv = _arr(y), _arr(p)
    m = np.isfinite(yv) & np.isfinite(pv)
    yv, pv = yv[m], pv[m]
    edges = np.asarray(bands, float)
    idx = np.clip(np.searchsorted(edges, pv, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        sel = idx == k
        if not sel.any():
            continue
        exp = float(pv[sel].sum())
        act = float(yv[sel].sum())
        var = float(np.sum(pv[sel] * (1 - pv[sel])))
        rows.append({"band_lo": float(edges[k]), "band_hi": float(edges[k + 1]), "n": int(sel.sum()),
                     "expected": exp, "actual": act, "exp_rate": exp / sel.sum(), "act_rate": act / sel.sum(),
                     "z": (act - exp) / np.sqrt(var) if var > 0 else np.nan})
    return pd.DataFrame(rows)


def reliability_comparison(y, forecasts: dict, bands=DEFAULT_BANDS) -> pd.DataFrame:
    """The three-way table §IV.2 asks for: the same bands for market, fundamental
    and combined, stacked with a ``forecast`` column so they can be read against
    each other. Pass e.g. {"market": pi, "fundamental": f, "combined": c}."""
    out = []
    for name, p in forecasts.items():
        t = benter_reliability(y, p, bands=bands)
        t.insert(0, "forecast", name)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
