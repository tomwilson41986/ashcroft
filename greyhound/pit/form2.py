"""Form rebuilt in two places (the feature review of 17 Nov 2026, the form row; reports/greyhound_feature_review.md):
the form table's (``fm_``) trend and age curve.

- **A damped trend.** The form table carries the level forward by the slope over the whole gap since the last run,
  so a dog improving a length a month comes back from 90 days off three lengths better on paper. Here the slope decays
  with days off at a fitted rate (a damped local linear trend in continuous time): over a gap of g days the level moves
  by S x tau x (1 - exp(-g / tau)) and the slope becomes S x exp(-g / tau). tau, with the level's and the slope's drift
  and the noise, is fitted by the one-step predictive likelihood on the early days (before ``form.FIT_BEFORE``, or
  the first 20% of days).
- **An age curve within dog.** The form table's curve averages the change between consecutive runs at most 60 days
  apart, so it is read on the dogs still racing (the survivors) and on short gaps only. Here the curve g(age) (a
  piecewise-linear spline in months, knots every 6 months from 12 to 72, one a sex) is fitted with dog fixed effects:
  each dog's figures less its own mean, on its age less its mean age (every run of the early days, any gap). The
  filter reads the age-free figure (figure less g(age at the run)) and the level today adds g(today's age): the
  expected change to today's age is g(age today) - g(age at the last figure).

The figures, the noise of a trial or troubled run and the Huber down-weighting are the form table's.
Every column reads days strictly before the row's day and the card's birth month and sex: card-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, figs as fg, form

PREFIX = "fq_"
DEPENDS = ("form", "figs")
KNOTS = np.arange(12.0, 73.0, 6.0)
REF_AGE = 30.0
SMOOTH = 1e-3                                          # the roughness penalty a run (lengths^2 a unit of curvature)
TAUS = (7.0, 21.0, 45.0, 90.0, 180.0, 1e9)            # 1e9: undamped, the form table's trend
SOURCE = "GBGB results (sources/gbgb.py): times, sectionals, grade, born, sex, comment"


def hat(age: np.ndarray) -> np.ndarray:
    """The piecewise-linear spline's basis at ``age`` (months, clipped to the knots): each row sums to 1."""
    a = np.clip(np.asarray(age, float), KNOTS[0], KNOTS[-1])
    j = np.clip(np.searchsorted(KNOTS, a, side="right") - 1, 0, len(KNOTS) - 2)
    w = (a - KNOTS[j]) / (KNOTS[j + 1] - KNOTS[j])
    B = np.zeros((len(a), len(KNOTS)))
    B[np.arange(len(a)), j] = 1 - w
    B[np.arange(len(a)), j + 1] = w
    return B


def age_curve_fe(dog, age, bitch, f, early, min_runs: int = 2) -> dict:
    """g(age) at the knots, one a sex, by least squares with dog fixed effects (each dog's figures and basis less its
    own means) on the early runs with a figure and an age; g(REF_AGE) = 0. A sex with too few runs gets a flat curve."""
    d = pd.DataFrame({"dog": np.asarray(dog), "age": np.asarray(age, float), "b": np.asarray(bitch, float),
                      "f": np.asarray(f, float), "e": np.asarray(early, bool)})
    d = d[d.e & np.isfinite(d.f) & np.isfinite(d.age)]
    d = d[d.groupby("dog").f.transform("size") >= min_runs]
    curves = {}
    for b in (0, 1):
        s = d[d.b.round() == b]
        if len(s) < 5 * len(KNOTS):
            curves[b] = np.zeros(len(KNOTS))
            continue
        B = pd.DataFrame(hat(s.age.to_numpy()), index=s.index)
        Bd = (B - B.groupby(s.dog.to_numpy()).transform("mean")).to_numpy()
        yd = (s.f - s.groupby("dog").f.transform("mean")).to_numpy()
        ref = hat(np.array([REF_AGE]))[0]
        keep = np.flatnonzero(np.abs(ref) < 1e-12) if (np.abs(ref) < 1e-12).any() else np.arange(1, len(KNOTS))
        # a light roughness penalty (second differences at the knots) and ridge: knots the early runs barely span
        # stay smooth and near 0 instead of fitting noise
        D2 = np.diff(np.eye(len(KNOTS)), 2, axis=0)
        pen = SMOOTH * len(s) * (D2.T @ D2 + 0.1 * np.eye(len(KNOTS)))
        Xf = np.zeros((len(s), len(KNOTS)))
        Xf[:, keep] = Bd[:, keep]
        A = Xf.T @ Xf + pen
        A[np.ix_(np.setdiff1d(np.arange(len(KNOTS)), keep), np.setdiff1d(np.arange(len(KNOTS)), keep))] += 1e12
        g = np.linalg.solve(A, Xf.T @ yd)
        curves[b] = g - hat(np.array([REF_AGE]))[0] @ g
    return curves


def curve_at(curves: dict, age: np.ndarray, bitch: np.ndarray) -> np.ndarray:
    B = hat(np.nan_to_num(np.asarray(age, float), nan=REF_AGE))
    v = np.where(np.asarray(bitch, float) >= 0.5, B @ curves[1], B @ curves[0])
    return np.where(np.isfinite(np.asarray(age, float)), v, 0.0)


def damped(g: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """(D, phi): the level's move a unit of slope over ``g`` days, and the slope's survival."""
    phi = np.exp(-g / tau)
    return tau * -np.expm1(-g / tau), phi


def damped_filter(dog_id, dn, z, noisy, prm: dict, upto: int | None = None) -> tuple[pd.DataFrame, float]:
    """The damped local-linear-trend filter on the age-free figures ``z``: before each row's day, the level carried by
    the damped slope, the slope today, the trend's share of the move, the level's spread, the days since the last
    reading; and the mean one-step predictive log likelihood (the fit's score). One reading a dog a day."""
    dog = pd.factorize(pd.Series(dog_id))[0]
    dn = np.asarray(dn, float)
    z = np.asarray(z, float)
    noisy = np.asarray(noisy, bool)
    tau = prm["tau"]
    nd = dog.max() + 1 if len(dog) else 0
    L, S = np.full(nd, prm["m0"]), np.zeros(nd)
    P00, P01, P11 = np.full(nd, prm["p0"]), np.zeros(nd), np.full(nd, prm["s0"])
    N, LAST = np.zeros(nd), np.full(nd, -1.0)
    n = len(z)
    lev, slo, ext, sd, gap = (np.full(n, np.nan) for _ in range(5))
    order = np.argsort(dn, kind="stable")
    bounds = np.flatnonzero(np.diff(dn[order])) + 1
    ll_sum, ll_n = 0.0, 0
    for rows in np.split(order, bounds):
        if not len(rows):
            continue
        if upto is not None and dn[rows[0]] >= upto:
            break
        dg = dog[rows]
        g = np.where(LAST[dg] >= 0, dn[rows] - LAST[dg], 0.0)
        D, phi = damped(g, tau)
        seen = N[dg] > 0
        lev[rows] = np.where(seen, L[dg] + D * S[dg], np.nan)
        slo[rows] = np.where(seen, phi * S[dg] * 30.44, np.nan)
        ext[rows] = np.where(seen, D * S[dg], np.nan)
        sd[rows] = np.sqrt(P00[dg] + 2 * D * P01[dg] + D * D * P11[dg] + prm["ql"] * g)
        gap[rows] = np.where(seen, g, np.nan)
        have = rows[np.isfinite(z[rows])]
        if not len(have):
            continue
        ud, inv = np.unique(dog[have], return_inverse=True)
        cnt = np.bincount(inv)
        zm = np.bincount(inv, weights=z[have]) / cnt
        nz = np.bincount(inv, weights=noisy[have].astype(float)) / cnt > 0.5
        day = dn[have][0]
        gg = np.where(LAST[ud] >= 0, day - LAST[ud], 0.0)
        D, phi = damped(gg, tau)
        l0 = L[ud] + D * S[ud]
        s0 = phi * S[ud]
        q00 = P00[ud] + 2 * D * P01[ud] + D * D * P11[ud] + prm["ql"] * gg
        q01 = phi * (P01[ud] + D * P11[ud])
        q11 = phi * phi * P11[ud] + prm["qs"] * gg
        r = prm["r"] * np.where(nz, 1.5, 1.0)
        e = zm - l0
        s2 = q00 + r
        ll_sum += float((-0.5 * (np.log(2 * np.pi * s2) + e * e / s2)).sum())
        ll_n += len(ud)
        big = np.abs(e) > form.HUBER * np.sqrt(s2)
        r = np.where(big, r * (np.abs(e) / (form.HUBER * np.sqrt(s2))) ** 2, r)
        s2 = q00 + r
        k0, k1 = q00 / s2, q01 / s2
        L[ud] = l0 + k0 * e
        S[ud] = s0 + k1 * e
        P00[ud], P01[ud], P11[ud] = (1 - k0) * q00, (1 - k0) * q01, q11 - k1 * q01
        N[ud] += 1
        LAST[ud] = day
    out = pd.DataFrame({"level": lev, "slope": slo, "ext": ext, "sd": sd, "gap": gap})
    return out, (ll_sum / ll_n if ll_n else np.nan)


def fit_damped(dog_id, dn, z, noisy, early: np.ndarray) -> dict:
    """The filter's constants by the one-step predictive likelihood on the early days, in two passes: the level's
    drift and the noise at a middling tau, then tau and the slope's drift."""
    dn = np.asarray(dn, float)
    z = np.asarray(z, float)
    cut = int(dn[early].max()) + 1 if early.any() else int(np.nanmax(dn)) + 1
    e = pd.DataFrame({"dog": np.asarray(dog_id), "dn": dn, "z": z})[early & np.isfinite(z)].sort_values(["dog", "dn"])
    first = e.groupby("dog").z.first()
    m0 = float(first.mean()) if len(first) else 0.0
    p0 = float(first.var()) if len(first) > 1 else 25.0
    tot = float(e.z.var()) if len(e) > 1 else 25.0
    base = {"m0": m0, "p0": p0 if np.isfinite(p0) else 25.0, "s0": 1e-4, "qs": 1e-6, "tau": 45.0}
    grid, best = [], None

    def score(prm):
        _, ll = damped_filter(dog_id, dn, z, noisy, prm, upto=cut)
        grid.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in prm.items() if k in
                     ("ql", "qs", "r", "tau")} | {"ll": round(ll, 5) if np.isfinite(ll) else None})
        return ll

    for ql in (0.01, 0.05, 0.2):
        for share in (0.4, 0.6, 0.8):
            prm = {**base, "ql": ql, "r": max(tot * share, 0.5)}
            ll = score(prm)
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, prm)
    stage1 = best[1] if best else {**base, "ql": 0.05, "r": max(tot * 0.6, 0.5)}
    for tau in TAUS:
        for qs in (1e-7, 1e-6, 1e-5, 1e-4):
            prm = {**stage1, "tau": tau, "qs": qs}
            ll = score(prm)
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, prm)
    prm = best[1] if best else stage1
    return {**prm, "loglik": round(best[0], 5) if best else None, "grid": grid}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    race = df.race_id
    early = form._early(df).to_numpy()
    f = fg.figures(df).f_tot.to_numpy(float)
    trouble = pd.to_numeric(df.get("trouble"), errors="coerce").fillna(0.0).to_numpy() if "trouble" in df else \
        np.zeros(len(df))
    trial = df.is_trial.fillna(False).astype(bool).to_numpy() if "is_trial" in df else np.zeros(len(df), bool)
    noisy = trial | (trouble > 0)
    age = form._age_months(df).to_numpy(float)
    bitch = form._bitch(df).to_numpy(float)
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float)
    curves = age_curve_fe(df.dog_id.to_numpy(), age, bitch, f, early)
    g_now = curve_at(curves, age, bitch)
    z = f - g_now                                                        # the age-free figure
    prm = fit_damped(df.dog_id.to_numpy(), dn, z, noisy, early)
    st, ll = damped_filter(df.dog_id.to_numpy(), dn, z, noisy, prm)
    # the age at the dog's last figure before today: today's age less the days since it
    age_last = age - st.gap.to_numpy() / 30.44
    g_last = curve_at(curves, age_last, bitch)
    out = pd.DataFrame({"race_id": race.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    seen = np.isfinite(st.level.to_numpy())
    out[f"{PREFIX}level"] = np.where(seen, st.level.to_numpy() + g_now, np.nan)
    out[f"{PREFIX}slope"] = st.slope.to_numpy()
    out[f"{PREFIX}trend_ext"] = st.ext.to_numpy()
    out[f"{PREFIX}sd"] = st.sd.to_numpy()
    out[f"{PREFIX}age_change"] = np.where(seen & np.isfinite(age), g_now - g_last, np.nan)
    h = 0.5
    out[f"{PREFIX}age_rate"] = np.where(np.isfinite(age), (curve_at(curves, age + h, bitch)
                                                           - curve_at(curves, age - h, bitch)) / (2 * h), np.nan)
    for c in ("level", "slope"):
        v = out[f"{PREFIX}{c}"]
        g = v.groupby(race)
        others = (g.transform("sum") - v.fillna(0)) / (g.transform("count") - v.notna()).replace(0, np.nan)
        out[f"{PREFIX}{c}_vs_field"] = v - others
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    notes = {"fit_before": form.FIT_BEFORE, "card_safe": cols,
             "trend": {k: v for k, v in prm.items() if k != "grid"}, "grid": prm["grid"],
             "loglik_all": round(ll, 5) if np.isfinite(ll) else None,
             "age_curve_lengths": {("bitch" if b else "dog"): {int(a): round(float(v), 3) for a, v in zip(KNOTS, c)}
                                   for b, c in curves.items()}}
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}level": "the damped-trend filter's whole-race level carried to today, plus the within-dog age curve at "
                      "today's age (lengths)",
    f"{PREFIX}slope": "the damped slope today, age excluded (lengths a month): the slope after the days off",
    f"{PREFIX}trend_ext": "the lengths the damped trend adds over the gap since the last figure",
    f"{PREFIX}sd": "the level's spread today",
    f"{PREFIX}age_change": "the within-dog age curve's expected change from the age at the last figure to today's "
                           "(lengths)",
    f"{PREFIX}age_rate": "the within-dog age curve's slope at today's age and sex (lengths a month)",
    f"{PREFIX}level_vs_field": "the level against the mean of the race's other runners",
    f"{PREFIX}slope_vs_field": "the slope against the mean of the race's other runners",
}
