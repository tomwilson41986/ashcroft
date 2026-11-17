"""The league table refitted (the feature review of 17 Nov 2026, reports/greyhound_feature_review.md, "league lg_":
"Fixed half-life and K; weekly refit misses this week's results. Fit constants; within-week update").

greyhound/pit/league.py (Hunter's MM, track-and-grade strengths, strength of schedule, market strength faced, the
trap) with two changes, every column the league table's under ``lgd_``:

**1. The constants fitted on an early slice.** ``HALF_LIFE``, ``KD`` and ``KL`` (league.py: 180 days, 5 and 20
pseudo-runs, set by hand) are chosen by a coordinate search over ``GRID`` (one pass: the half-life, then KD, then
KL, each with the others at their best so far) on the runs of the early slice only: the days before ``SLICE_END``
(1 Jul 2016, about the first 20% of a store from 2014), or, when the store holds fewer than ``MIN_SLICE_MONDAYS``
scoring Mondays there, the first ``SLICE_SHARE`` of its days. The score: on every ``EVERY``-th Monday M of the
slice, the league fit on the window before M (league.py ``league_mm``), and each run of M's week whose race has two
dogs or more rated by that fit, its race-centred PRB against the race-centred rating (the dogs' ratings carry their
leagues' strengths, so only the cross-league scale is scored); the mean squared error over the slice. The set of
rows scored does not depend on the constants. The test period (1 Jun 2026 on) is never read: the slice ends
before it, and a fallback slice that reaches it is cut there. ``ctx["league2_constants"]`` (a dict half_life, kd,
kl: the nightly table's ``notes["constants"]``) replaces the fit, which a card build on a short store must pass
(its early slice is not the nightly build's).

**2. The within-week update.** league.py refits on each Monday, so a Thursday race reads nothing of Monday to
Wednesday. Here every fit (the leagues, the schedule, the market, the trap) is refitted EACH DAY from the runs of
the days before it, warm-started from the day before (league.py ``build(step=1)``): a race reads every result up
to the day before its day, never its own day's.

Card-safe: every column, given the same constants (a card row reads the fits of the days before its day, as its
run would; the constants come from the slice or from ``ctx``). The rows inside the slice are in-sample for the
constants: score from after ``notes["constants"]["slice_end"]``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from greyhound.pit import league

DEPENDS = ("league",)
PREFIX = "lgd_"
SOURCE = ("GBGB results (sources/gbgb.py) as the league table (greyhound/pit/league.py), its half-life and "
          "pseudo-runs fitted on the early slice, refitted each day from the runs of the days before it")
COLUMNS = {f"lgd_{c[3:]}": f"{d} (refitted daily, constants fitted on the early slice)"
           for c, d in league.COLUMNS.items()}

SLICE_END = "2016-07-01"
TEST_START = "2026-06-01"
SLICE_SHARE = 0.2
MIN_SLICE_MONDAYS = 8
EVERY = 4
GRID = {"half_life": (45.0, 90.0, 180.0, 365.0, 730.0), "kd": (1.0, 2.0, 5.0, 10.0, 20.0),
        "kl": (5.0, 20.0, 50.0, 100.0, 200.0)}


def _prepare(runs: pd.DataFrame):
    """The league fit's inputs as league.build makes them: time order, day numbers, dog and league codes, the
    race-centred PRB (results only; card and trial rows excluded)."""
    n = len(runs)
    card = runs.is_card.to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    trial = runs.is_trial.to_numpy(bool) if "is_trial" in runs else np.zeros(n, bool)
    dn = (pd.to_datetime(runs.race_date).to_numpy("datetime64[D]") - np.datetime64("1970-01-01")).astype(np.int64)
    tt = runs.t.to_numpy("datetime64[s]").astype(np.int64) if "t" in runs else dn
    o = np.lexsort((runs.race_id.to_numpy(), tt, dn))
    r = runs.iloc[o]
    dn = dn[o]
    ran = ~card[o] & ~trial[o]
    dog, nd = league._codes(r.dog_id.to_numpy())
    rid = r.race_id.to_numpy()
    race = np.cumsum(np.r_[True, rid[1:] != rid[:-1]]) - 1 if n else np.zeros(0, np.int64)
    nr = int(race.max()) + 1 if n else 0
    lg, nl = league._codes(r.track.to_numpy(), r.race_class.to_numpy())
    pos = pd.to_numeric(r.position, errors="coerce").to_numpy(float)
    fin = ran & np.isfinite(pos)
    nf = np.bincount(race[fin], minlength=nr)[race].astype(float)
    rank = np.full(n, np.nan)
    rank[fin] = pd.Series(pos[fin]).groupby(race[fin]).rank(method="average").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        prb = np.where(fin & (nf >= 2), (nf - rank) / (nf - 1), np.nan)
    y = np.where(nf >= 3, prb - 0.5, np.nan)
    ok = np.isfinite(y) & (lg >= 0)
    return {"dn": dn[ok], "dog": dog[ok], "nd": nd, "lg": lg[ok], "nl": nl, "race": race[ok], "y": y[ok],
            "all_days": np.unique(dn[ran])}


def _mondays(p, end: int) -> np.ndarray:
    """The scoring Mondays: every ``EVERY``-th Monday before ``end`` with at least two earlier days of runs."""
    if not len(p["dn"]):
        return np.zeros(0, np.int64)
    m = np.unique(league.monday(p["dn"]))
    m = m[(m < end) & (m > p["dn"].min() + 1)]
    return m[::EVERY]


def score(p, mondays, half_life: float, kd: float, kl: float) -> tuple[float, int]:
    """The early slice's mean squared error of the race-centred PRB on the race-centred rating (the docstring)."""
    dn, dog, lg, y, race = p["dn"], p["dog"], p["lg"], p["y"], p["race"]
    sse, cnt = 0.0, 0
    for M in mondays:
        lo, hi = np.searchsorted(dn, M - league.WINDOW, "left"), np.searchsorted(dn, M, "left")
        a1 = np.searchsorted(dn, M + 7, "left")
        if hi - lo < 2 or a1 <= hi:
            continue
        i = np.arange(lo, hi)
        dl, ids = league._compact(dog[i], p["nd"])
        w = 0.5 ** ((M - dn[i]) / half_life)
        rc, _, _, _ = league.league_mm(dl, lg[i], y[i], w, p["nl"], None, kd, kl)
        rg = np.full(p["nd"], np.nan)
        rg[ids] = rc
        j = np.arange(hi, a1)
        v = rg[dog[j]]
        has = np.isfinite(v)
        rq = race[j][has]
        if not len(rq):
            continue
        code, _ = pd.factorize(rq)
        k = np.bincount(code)
        two = k[code] >= 2
        pred = v[has] - (np.bincount(code, v[has]) / k)[code]
        obs = y[j][has] - (np.bincount(code, y[j][has]) / k)[code]
        sse += float(((obs - pred)[two] ** 2).sum())
        cnt += int(two.sum())
    return (sse / cnt if cnt else np.nan), cnt


def fit_constants(runs: pd.DataFrame) -> dict:
    """The half-life and pseudo-runs by a coordinate search on the early slice (the docstring)."""
    p = _prepare(runs)
    epoch = np.datetime64("1970-01-01")
    test = int((np.datetime64(TEST_START) - epoch).astype(np.int64))
    end = int((np.datetime64(SLICE_END) - epoch).astype(np.int64))
    mondays, how = _mondays(p, end), "fixed"
    if len(mondays) < MIN_SLICE_MONDAYS and len(p["all_days"]):
        days = p["all_days"]
        end = min(int(days[max(int(np.ceil(SLICE_SHARE * len(days))) - 1, 0)]) + 1, test)
        mondays, how = _mondays(p, end), "first_share"
    best = {"half_life": league.HALF_LIFE, "kd": league.KD, "kl": league.KL}
    if not len(mondays):
        return {**best, "fitted": False, "slice": how, "note": "no data found: no scoring Monday in the slice"}
    base, n = score(p, mondays, **best)
    trace = {"default": round(base, 6)}
    for name in ("half_life", "kd", "kl"):
        res = {v: score(p, mondays, **{**best, name: v})[0] for v in GRID[name]}
        res = {v: s for v, s in res.items() if np.isfinite(s)}
        if res:
            best[name] = min(res, key=res.get)
            trace[name] = {str(v): round(s, 6) for v, s in res.items()}
    return {**best, "fitted": True, "slice": how, "slice_end": str(np.datetime64(int(end), "D")),
            "scoring_mondays": int(len(mondays)), "rows_scored": int(n),
            "mse_default": round(base, 6), "mse_fitted": round(score(p, mondays, **best)[0], 6), "trace": trace}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None):
    t0 = time.time()
    ctx = ctx or {}
    given = ctx.get("league2_constants")
    if given:
        const = {"half_life": float(given["half_life"]), "kd": float(given["kd"]), "kl": float(given["kl"]),
                 "fitted": False, "from_ctx": True}
        const.update({k: v for k, v in given.items() if k not in const})
    else:
        const = fit_constants(runs)
    tab, notes = league.build(runs, figs, ctx, half_life=const["half_life"], kd=const["kd"], kl=const["kl"], step=1)
    tab = tab.rename(columns={c: f"lgd_{c[3:]}" for c in league.COLUMNS})
    assert list(tab.columns) == league.KEY + list(COLUMNS)
    notes = {**notes, "constants": const, "card_safe": list(COLUMNS),
             "seconds_total": round(time.time() - t0, 1)}
    return tab, notes
