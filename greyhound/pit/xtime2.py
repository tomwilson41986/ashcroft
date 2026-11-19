"""The expected-time table, second version (reports/greyhound_feature_review.md, "xtime" row; plan item 2, xtime2).

The same model of the time as greyhound/pit/xtime.py (``xt_``, the second family by importance), built by its code with
its keywords (``run_filter``, ``fit_constants``, ``build``), with the review's five changes:

- (a) the trap and grade effects are exponentially forgetting means, each with a half-life in days fitted on the early
  slice (xtime.py keeps cumulative means that never forget);
- (b) the constants ``K_TRAP``, ``K_GRADE``, ``K_PREF``, ``STD_CAP`` and ``NOISY`` are fitted on the early slice (the
  first 20% of days: before ``FIT_BEFORE``) by the error of each run's forecast from the state before its day, one
  coordinate pass after the filter's noise and drift (``xtime.fit_constants``); the trap, preference and noise
  constants on the error against the race's mean (the going is not the dog), the grade and standard ones, which are
  the same for every runner of a race, on the error itself. The fitted values are in the notes (``<time>.tuned``);
- (c) the preferences are updated on the residual the level's update left, not the same residual (the double count);
- (d) ``xu_tot_pfast``: the chance of the fastest whole-race time when each dog's time is normal about its expected
  time with ``xu_tot_sd`` (times a spread multiplier fitted on the early races' winners), by Gauss-Hermite
  integration over the race's dogs;
- (e) a vacant box beside the dog is room (read as the race's slowest expected sectional, as bend.py reads it), not
  NaN; NaN is left only where there is no box (inside trap 1, outside trap 6).

Point in time: every column is read from the state after earlier days (xtime.py), and every fitted constant from the
early slice only. Card-safe: every column (a card row has the expected times its run would have; the vacant boxes are
the card's, so a non-runner after the card is a vacant box in the nightly build, as for ext.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, xtime

PREFIX = "xu_"
SOURCE = xtime.SOURCE
DEPENDS = ("xtime",)
#: the early slice: about the first 20% of the days from 2014 to the test (1 Jun 2026) - a fixed date, so the card's
#: build and the nightly build fit the same constants. A frame with under 5% of its rows before it (the unit tests)
#: uses the first 20% of its days
FIT_BEFORE = "2016-07-01"
EARLY_SHARE = 0.2
INF = float("inf")
#: the coordinate search (one pass, in this order): the run_filter keyword, the values, scored within the race or not
SEARCH = (("k_trap", (12.5, 25.0, 50.0, 100.0, 200.0), True),
          ("hl_trap", (60.0, 120.0, 240.0, 480.0, 960.0, INF), True),
          ("k_grade", (7.5, 15.0, 30.0, 60.0, 120.0), False),
          ("hl_grade", (60.0, 120.0, 240.0, 480.0, 960.0, INF), False),
          ("k_pref", (1.25, 2.5, 5.0, 10.0, 20.0), True),
          ("noisy", (1.0, 1.5, 2.25, 3.4), True),
          ("std_cap", (200.0, 400.0, 800.0, 1600.0, 3200.0), False))
MULTS = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)      # the spread multiplier's grid for P(fastest)
NODES = 32
CHUNK = 5000


def early_slice(day: pd.Series) -> pd.Series:
    early = day < pd.Timestamp(FIT_BEFORE)
    if early.sum() < 0.05 * len(day) and len(day):
        days = np.sort(day.dropna().unique())
        early = day <= days[int(len(days) * EARLY_SHARE)]
    return early


def _kw(c: dict) -> dict:
    return {"consts": {k: c[k] for k in ("k_trap", "k_grade", "k_pref", "std_cap", "noisy")},
            "forget": {"trap": c["hl_trap"], "grade": c["hl_grade"]}, "pref_post": True}


def tune(P: dict, y: np.ndarray, early_to: int, name: str) -> tuple[dict, dict, dict]:
    """Review (b): the filter's noise and drift (xtime's grid), then each constant in turn on the early slice."""
    c = {"k_trap": xtime.K_TRAP, "k_grade": xtime.K_GRADE, "k_pref": xtime.K_PREF, "std_cap": xtime.STD_CAP,
         "noisy": xtime.NOISY, "hl_trap": INF, "hl_grade": INF}
    prm, grid = xtime.fit_constants(P, y, early_to, filter_kw=_kw(c))
    score = xtime.score_rows(P, early_to)
    search = {}
    if score is not None:
        for key, values, within in SEARCH:
            res = {}
            for v in values:
                o = xtime.run_filter(P, y, prm, upto=early_to, fit=False, **_kw({**c, key: v}))
                res[v] = xtime.forecast_mse(P, o, score, within)
            ok = {v: m for v, m in res.items() if np.isfinite(m)}
            if ok:
                c[key] = min(ok, key=lambda v: (ok[v], v != c[key]))       # a tie keeps xtime's value
            search[key] = {("inf" if v == INF else v): (round(m, 5) if np.isfinite(m) else None) for v, m in res.items()}
    fitted = {k: ("inf" if v == INF else v) for k, v in c.items()}
    return prm, _kw(c), {"grid": grid, "constants": fitted, "search": search, "pref_post": True,
                         "early_to_day": int(early_to)}


def p_fastest(f: np.ndarray, sd: np.ndarray, race: np.ndarray, mult: float = 1.0) -> np.ndarray:
    """Review (d): P(dog i is the fastest) when each dog's figure (higher faster) is normal about ``f`` with spread
    ``mult x sd``, independently: the integral of its density times the others' distribution functions, by
    Gauss-Hermite quadrature, vectorised over races (padded to the largest field). A missing ``f`` is the race's mean,
    a missing or non-positive ``sd`` the race's median (else the frame's, else 1)."""
    from scipy.special import log_ndtr
    n = len(f)
    out = np.full(n, np.nan)
    if not n:
        return out
    f = pd.Series(np.asarray(f, float))
    sd = pd.Series(np.asarray(sd, float)).where(lambda s: s > 0)
    rc = pd.factorize(pd.Series(race))[0]
    f = f.fillna(f.groupby(rc).transform("mean")).fillna(0.0).to_numpy()
    glob = float(np.nanmedian(sd)) if sd.notna().any() else 1.0
    sd = (sd.fillna(sd.groupby(rc).transform("median")).fillna(glob).to_numpy()) * mult
    order = np.argsort(rc, kind="stable")
    r_sorted = rc[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r_sorted)) + 1]
    sizes = np.diff(np.r_[starts, n])
    pos = np.arange(n) - np.repeat(starts, sizes)
    F = int(sizes.max())
    R = len(starts)
    mu = np.zeros((R, F))
    s = np.ones((R, F))
    have = np.zeros((R, F), bool)
    ri = np.repeat(np.arange(R), sizes)
    mu[ri, pos], s[ri, pos], have[ri, pos] = f[order], sd[order], True
    z, w = np.polynomial.hermite_e.hermegauss(NODES)
    w = w / np.sqrt(2.0 * np.pi)
    p = np.zeros((R, F))
    eye = np.eye(F, dtype=bool)
    for lo in range(0, R, CHUNK):
        m, ss, hv = mu[lo:lo + CHUNK], s[lo:lo + CHUNK], have[lo:lo + CHUNK]
        x = m[:, :, None] + ss[:, :, None] * z[None, None, :]                        # (r, i, k)
        lg = log_ndtr((x[:, :, None, :] - m[:, None, :, None]) / ss[:, None, :, None])   # (r, i, j, k)
        drop = eye[None, :, :, None] | ~hv[:, None, :, None]
        lg = np.where(drop, 0.0, lg).sum(axis=2)                                      # (r, i, k)
        p[lo:lo + CHUNK] = np.where(hv, (np.exp(lg) * w).sum(axis=2), 0.0)
    p = p / p.sum(axis=1, keepdims=True)
    out[order] = p[ri, pos]
    return out


def fit_mult(f: pd.Series, sd: pd.Series, race: pd.Series, won: pd.Series, early: pd.Series) -> tuple[float, dict]:
    """The spread multiplier with the smallest log loss of the early races' winners (races with every expected time
    and spread known and one winner); 1 with fewer than 200 such races."""
    e = early & f.notna() & sd.notna()
    full = e.groupby(race).transform("all") & e
    k = full & (won.where(full).groupby(race).transform("sum") == 1)
    if race[k].nunique() < 200:
        return 1.0, {}
    grid = {}
    for m in MULTS:
        p = p_fastest(f[k].to_numpy(), sd[k].to_numpy(), race[k].to_numpy(), m)
        grid[m] = round(float(-np.log(np.clip(p[(won[k] == 1).to_numpy()], 1e-9, None)).mean()), 5)
    return min(grid, key=grid.get), grid


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    figs = figs if figs is not None else core.per_run_figures(runs)
    tab, notes = xtime.build(runs, figs, ctx, prefix=PREFIX, early_fn=early_slice, tune=tune, vacant_room=True)
    df = figs.sort_index()                                         # xtime.build's rows, in its order
    race = tab.race_id
    day = core.day(df).reset_index(drop=True)
    early = early_slice(day)
    f = tab[f"{PREFIX}tot"].astype(float)
    sd = tab[f"{PREFIX}tot_sd"].astype(float)
    won = pd.to_numeric(df.won, errors="coerce").reset_index(drop=True)
    mult, grid = fit_mult(f, sd, race, won, early)
    p = p_fastest(f.to_numpy(), sd.to_numpy(), race.to_numpy(), mult)
    tab[f"{PREFIX}tot_pfast"] = p.astype(np.float32)
    tab[f"{PREFIX}tot_lpfast"] = np.log(np.clip(p, 1e-9, None)).astype(np.float32)
    notes.update({"pfast_mult": mult, "pfast_grid": grid, "fit_before": FIT_BEFORE,
                  "card_safe": "every column", "variant_of": "xtime"})
    return tab, notes


_SEC = {"sec_vs_in": "expected sectional against the dog drawn one trap inside (lengths; above 0 it leads it); a "
                     "vacant box there is room, read as the race's slowest expected sectional; NaN inside trap 1",
        "sec_vs_out": "expected sectional against the dog drawn one trap outside; a vacant box there is room; NaN "
                      "outside trap 6"}
COLUMNS = {
    **{PREFIX + c[len(xtime.PREFIX):]: _SEC.get(c[len(xtime.PREFIX):], d) + " (xtime2: forgetting trap and grade, "
       "fitted constants, preference on the post-level residual)" for c, d in xtime.COLUMNS.items()},
    f"{PREFIX}tot_pfast": "the chance of the fastest whole-race time: each dog normal about its expected time with "
                          "its spread (xu_tot_sd times a multiplier fitted on the early winners)",
    f"{PREFIX}tot_lpfast": "the log of that chance",
}
