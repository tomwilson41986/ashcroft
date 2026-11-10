"""Form: a rating that knows about trend and age, levels on fitted decay and shrinkage, and the market's memory (the
BFSP plan, reports/greyhound_bfsp_plan.md, steps 2.6, 2.2 and 2.5; the owner, 10 Nov 2026: "work on the remaining
plan steps").

**Fitted decay and shrinkage (2.6).** The figs table decays a dog's figures with a 90-day half-life and shrinks them
toward today's grade par by 2 runs of weight, both chosen by hand. Here both are chosen for each figure (whole race,
sectional, run-in) by how well the level before a run predicts that run's figure (mean squared error), on the runs
before ``FIT_BEFORE`` only: ``fm_tot``, ``fm_sec``, ``fm_run``.

**Trend and age (2.2).** A local-linear-trend Kalman filter on each dog's whole-race figure: a level and a slope
(lengths a day), the level carried forward by the slope plus the population's age curve between runs (the mean change
in figure a day at that age and sex, from consecutive runs before ``FIT_BEFORE``). A trial or a troubled run reads with
1.5 times the noise, and a reading more than 3 standard deviations from the prediction is down-weighted (its noise
inflated in proportion: Huber), so one bad run does not move the rating. The level's drift, the slope's drift and the
noise are fitted on the runs before ``FIT_BEFORE`` by the one-step predictive likelihood. Read at today's date, before
the run: ``fm_level``, ``fm_slope`` (lengths a month, the age curve included), ``fm_sd``, ``fm_age_drift``.

**The market's memory (2.5).** The market-implied rating's innovations (greyhound/pit/figs.py ``market_rating``: each
race's race-relative log share less the rating before it, on the Betfair SP where the race is wholly priced by it, the
SP elsewhere), decayed over the dog's earlier days: how the market has been moving on the dog lately, beyond what its
rating already said (``fm_mr_innov``).

Every column reads earlier days only (tests/test_greyhound_pit.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, figs as fg

PREFIX = "fm_"
FIT_BEFORE = "2018-01-01"
HALF_LIVES = (30.0, 60.0, 90.0, 180.0, 365.0)
PRIORS = (0.5, 1.0, 2.0, 4.0, 8.0)
INNOV_HALF_LIFE = 60.0
HUBER = 3.0
SOURCE = "GBGB results (sources/gbgb.py): times, sectionals, grade, born, sex, comment, SP; Betfair's price files (BSP)"


def _early(df: pd.DataFrame) -> pd.Series:
    e = core.day(df) < pd.Timestamp(FIT_BEFORE)
    if e.sum() < 0.05 * len(df) and len(df):
        e = core.day(df) <= core.day(df).sort_values().iloc[int(len(df) * 0.2)]
    return e


def fit_level(df: pd.DataFrame, f: pd.Series, prior: pd.Series, weight: pd.Series, early: pd.Series) -> tuple:
    """The half-life and prior weight that best predict each run's figure from the dog's earlier days (mean squared
    error on the early runs with a figure). Returns (half_life, k, grid, the decayed sums at the chosen half-life)."""
    ok = (early & f.notna() & (weight >= 1)).to_numpy()
    grid, best, sums = {}, None, {}
    for hl in HALF_LIVES:
        s = fg.decayed(df, f, weight, hl)
        sums[hl] = s
        for k in PRIORS:
            level = (s.swx + k * prior) / (s.sw + k)
            mse = float(((level - f).to_numpy()[ok] ** 2).mean()) if ok.any() else np.inf
            grid[f"{hl:g}|{k:g}"] = round(mse, 4)
            if best is None or mse < best[0]:
                best = (mse, hl, k)
    return best[1], best[2], grid, sums[best[1]]


def age_curve(df: pd.DataFrame, f: pd.Series, early: pd.Series) -> dict:
    """The population's mean change in whole-race figure a day, by age in months (12 to 72) and sex, from consecutive
    runs at most 60 days apart before FIT_BEFORE; empty buckets take the nearest filled one."""
    d = pd.DataFrame({"dog": df.dog_id.to_numpy(), "dn": (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy(),
                      "f": f.to_numpy(), "age": _age_months(df).to_numpy(), "bitch": _bitch(df).to_numpy(),
                      "early": early.to_numpy()})
    d = d[d.f.notna()].sort_values(["dog", "dn"])
    g = d.groupby("dog")
    gap = g.dn.diff()
    rate = g.f.diff() / gap
    ok = d.early & (gap > 0) & (gap <= 60) & d.age.notna()
    t = pd.DataFrame({"age": (d.age - gap / 30.44).clip(12, 72).round(), "bitch": d.bitch, "rate": rate})[ok]
    curve = {}
    for b in (0.0, 1.0):
        s = t[t.bitch == b].groupby("age").rate.agg(["mean", "size"])
        s = s[s["size"] >= 200]["mean"]
        full = pd.Series(np.nan, index=np.arange(12, 73, dtype=float))
        full.loc[s.index] = s.values
        curve[int(b)] = full.interpolate(limit_direction="both").fillna(0.0).to_dict() if s.notna().any() else \
            {a: 0.0 for a in full.index}
    return curve


def _age_months(df: pd.DataFrame) -> pd.Series:
    born = pd.to_datetime("15-" + df.born.astype(str), format="%d-%b-%Y", errors="coerce") if "born" in df else \
        pd.Series(pd.NaT, index=df.index)
    return (core.day(df) - born).dt.days / 30.44


def _bitch(df: pd.DataFrame) -> pd.Series:
    return (df.sex.astype(str).str.lower() == "b").astype(float) if "sex" in df else pd.Series(0.0, index=df.index)


def trend_filter(df: pd.DataFrame, f: pd.Series, noisy: np.ndarray, curve: dict, prm: dict,
                 upto: int | None = None) -> tuple[pd.DataFrame, float]:
    """The local-linear-trend filter's state before each row's day (level and slope carried to that day, the level's
    spread, the age drift a day used), and the mean one-step predictive log likelihood of the figures read (the fit's
    score). Days in order; a day's rows read the state after the dog's earlier days."""
    dog = pd.factorize(df.dog_id)[0]
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    z = f.to_numpy(float)
    age = _age_months(df).clip(12, 72).round().fillna(36).to_numpy()
    bitch = _bitch(df).to_numpy() >= 0.5
    drift = np.where(bitch, np.vectorize(curve[1].get)(age), np.vectorize(curve[0].get)(age)).astype(float)
    nd = dog.max() + 1
    L, S = np.full(nd, prm["m0"]), np.zeros(nd)
    P00, P01, P11 = np.full(nd, prm["p0"]), np.zeros(nd), np.full(nd, prm["s0"])
    N, LAST = np.zeros(nd), np.full(nd, -1)
    lev, slo, sd = np.full(len(df), np.nan), np.full(len(df), np.nan), np.full(len(df), np.nan)
    order = np.argsort(dn, kind="stable")
    bounds = np.flatnonzero(np.diff(dn[order])) + 1
    ll_sum, ll_n = 0.0, 0
    for rows in np.split(order, bounds):
        if upto is not None and dn[rows[0]] >= upto:
            break
        dg = dog[rows]
        g = np.where(LAST[dg] >= 0, dn[rows] - LAST[dg], 0).astype(float)
        a = drift[rows]
        lp = L[dg] + (S[dg] + a) * g
        p00 = P00[dg] + 2 * g * P01[dg] + g * g * P11[dg] + prm["ql"] * g
        lev[rows] = np.where(N[dg] > 0, lp, np.nan)
        slo[rows] = np.where(N[dg] > 0, (S[dg] + a) * 30.44, np.nan)
        sd[rows] = np.sqrt(p00)
        have = rows[np.isfinite(z[rows])]
        if not len(have):
            continue
        # one reading a dog a day (the day's figures averaged), the state carried to the day first
        ud, inv = np.unique(dog[have], return_inverse=True)
        zm = np.bincount(inv, weights=z[have]) / np.bincount(inv)
        nz = np.bincount(inv, weights=noisy[have].astype(float)) / np.bincount(inv) > 0.5
        day = dn[have][0]
        gg = np.where(LAST[ud] >= 0, day - LAST[ud], 0).astype(float)
        au = np.bincount(inv, weights=drift[have]) / np.bincount(inv)
        l0 = L[ud] + (S[ud] + au) * gg
        q00 = P00[ud] + 2 * gg * P01[ud] + gg * gg * P11[ud] + prm["ql"] * gg
        q01 = P01[ud] + gg * P11[ud]
        q11 = P11[ud] + prm["qs"] * gg
        r = prm["r"] * np.where(nz, 1.5, 1.0)
        e = zm - l0
        s2 = q00 + r
        ll_sum += float((-0.5 * (np.log(2 * np.pi * s2) + e * e / s2)).sum())
        ll_n += len(ud)
        big = np.abs(e) > HUBER * np.sqrt(s2)
        r = np.where(big, r * (np.abs(e) / (HUBER * np.sqrt(s2))) ** 2, r)       # Huber: a wild run reads as noise
        s2 = q00 + r
        k0, k1 = q00 / s2, q01 / s2
        L[ud] = l0 + k0 * e
        S[ud] = S[ud] + k1 * e
        P00[ud], P01[ud], P11[ud] = (1 - k0) * q00, (1 - k0) * q01, q11 - k1 * q01
        N[ud] += 1
        LAST[ud] = day
    out = pd.DataFrame({"level": lev, "slope": slo, "sd": sd, "drift": drift * 30.44}, index=df.index)
    return out, (ll_sum / ll_n if ll_n else np.nan)


def fit_trend(df, f, noisy, curve, early) -> dict:
    """The filter's constants by the one-step predictive likelihood on the early days: the level's drift ``ql`` and
    the slope's ``qs`` (a day), the noise ``r``; the debut level and spread from the early first figures."""
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    cut = int(dn[early.to_numpy()].max()) + 1 if early.any() else int(dn.max())
    e = df.assign(_f=f)[early & f.notna()].sort_values(["dog_id", "t"])
    first = e.groupby("dog_id")._f.first()
    m0 = float(first.mean()) if len(first) else 0.0
    p0 = float(first.var()) if len(first) > 1 else 25.0
    tot = float(e._f.var()) if len(e) > 1 else 25.0
    grid, best = [], None
    for ql in (0.01, 0.05, 0.2):
        for qs in (1e-6, 1e-5, 1e-4):
            for share in (0.4, 0.6, 0.8):
                prm = {"ql": ql, "qs": qs, "r": max(tot * share, 0.5), "m0": m0, "p0": p0, "s0": 1e-4}
                _, ll = trend_filter(df, f, noisy, curve, prm, upto=cut)
                grid.append((ql, qs, round(prm["r"], 2), round(ll, 5) if np.isfinite(ll) else None))
                if np.isfinite(ll) and (best is None or ll > best[0]):
                    best = (ll, prm)
    prm = best[1] if best else {"ql": 0.05, "qs": 1e-5, "r": max(tot * 0.6, 0.5), "m0": m0, "p0": p0, "s0": 1e-4}
    return {**prm, "loglik": round(best[0], 5) if best else None, "grid": grid}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    race = df.race_id
    early = _early(df)
    fgs = fg.figures(df)
    trouble = pd.to_numeric(df.get("trouble"), errors="coerce").fillna(0.0) if "trouble" in df else \
        pd.Series(0.0, index=df.index)
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    noisy = (trial | (trouble > 0)).to_numpy()
    weight = pd.Series(np.where(noisy, 0.5, 1.0), index=df.index)
    tmp = df[["race_date", "race_time", "track", "distance_m", "race_class"]].assign(raceid=race.astype(str))
    out = pd.DataFrame({"race_id": race.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    notes = {"fit_before": FIT_BEFORE, "levels": {}}
    # 2.6: each figure's level on its fitted half-life and prior weight
    for n in fg.FIGS:
        f = fgs[f"f_{n}"]
        par = core.race_lagged_expanding_mean(tmp.assign(_f=f), ["track", "distance_m", "race_class"], "_f")
        pop = core.race_lagged_expanding_mean(tmp.assign(_f=f, _p=0), "_p", "_f")
        prior = par.fillna(pop).fillna(0.0)
        hl, k, grid, s = fit_level(df, f, prior, weight, early)
        out[f"{PREFIX}{n}"] = (s.swx + k * prior) / (s.sw + k)
        notes["levels"][n] = {"half_life": hl, "k": k, "grid_mse": grid}
    # 2.2: the trend filter on the whole-race figure, with the age curve
    f = fgs.f_tot
    curve = age_curve(df, f, early)
    prm = fit_trend(df, f, noisy, curve, early)
    st, ll = trend_filter(df, f, noisy, curve, prm)
    out[f"{PREFIX}level"] = st.level
    out[f"{PREFIX}slope"] = st.slope
    out[f"{PREFIX}sd"] = st.sd
    out[f"{PREFIX}age_drift"] = st.drift
    for c in ("level", "slope"):
        v = out[f"{PREFIX}{c}"]
        g = v.groupby(race)
        others = (g.transform("sum") - v.fillna(0)) / (g.transform("count") - v.notna()).replace(0, np.nan)
        out[f"{PREFIX}{c}_vs_field"] = v - others
    notes["trend"] = {**{k_: v_ for k_, v_ in prm.items() if k_ != "grid"}, "grid": prm["grid"],
                      "loglik_all": round(ll, 5) if np.isfinite(ll) else None,
                      "age_curve_lengths_a_month": {s_: {int(a): round(v_ * 30.44, 4) for a, v_ in c.items()
                                                         if int(a) % 6 == 0} for s_, c in curve.items()}}
    # 2.5: the market's memory, the rating's innovations on the BSP where the race has it, the SP elsewhere
    prices = (ctx or {}).get("prices")
    d2 = df
    if prices is not None and len(prices) and "bsp_p_norm" not in df:
        from greyhound.parity import attach_bsp
        b = attach_bsp(df[["race_date", "race_time", "track", "trap", "race_id"]], prices)
        d2 = df.assign(bsp_p_norm=b.bsp_p_norm.to_numpy())
    sp = pd.to_numeric(d2.sp_p_norm, errors="coerce")
    if "bsp_p_norm" in d2:
        bq = pd.to_numeric(d2.bsp_p_norm, errors="coerce")
        q = bq.where(bq.notna().groupby(race).transform("all"), sp)
    else:
        q = sp
    mprm = fg.fit_market_rating(d2)
    mr, _ = fg.market_rating(d2.assign(_q=q), mprm, share_col="_q")
    s = fg.decayed(df, mr.innov, pd.Series(1.0, index=df.index), INNOV_HALF_LIFE)
    out[f"{PREFIX}mr_innov"] = s.swx / s.sw.replace(0, np.nan)
    out[f"{PREFIX}mr_innov_n"] = s.sw
    notes["market_memory"] = {"half_life": INNOV_HALF_LIFE, **{k_: v_ for k_, v_ in mprm.items() if k_ != "grid"}}
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    **{f"{PREFIX}{n}": f"the dog's {lab} level on a half-life and prior weight fitted to predict the next run "
                       "(before 2018), shrunk toward today's grade par"
       for n, lab in (("tot", "whole-race"), ("sec", "sectional"), ("run", "run-in"))},
    f"{PREFIX}level": "the trend filter's whole-race level carried to today by its slope and the age curve (lengths)",
    f"{PREFIX}slope": "its slope, the age curve included (lengths a month)",
    f"{PREFIX}sd": "the level's spread today",
    f"{PREFIX}age_drift": "the population's expected change at the dog's age and sex (lengths a month)",
    f"{PREFIX}level_vs_field": "the level against the mean of the race's other runners",
    f"{PREFIX}slope_vs_field": "the slope against the mean of the race's other runners",
    f"{PREFIX}mr_innov": "the market rating's recent innovations: how the market has moved on the dog beyond its rating",
    f"{PREFIX}mr_innov_n": "the decayed count of readings behind it",
}
