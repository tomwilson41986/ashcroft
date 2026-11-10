"""Corrected speed figures, form that knows how much evidence it rests on, class in lengths, and a market-implied
rating (the BFSP plan of 10 Nov 2026, reports/greyhound_bfsp_plan.md: step 0.4, the feature chapter's flaws 1 to 4,
its points 6, 7 and 9, and step 2.7).

**The figures.** Each run gets three figures in lengths (higher is faster), each on its own standard and its own
meeting variant:
- ``tot``: GBGB's calculated time (its going allowance applied) against the track and distance's standard, the median
  winning time over the ``STD_DAYS`` before the day;
- ``sec``: the raw sectional against the median sectional of every runner there over the same days;
- ``run``: the raw run-in (run time less sectional) against the median winning run-in.
The meeting variant of each is measured against **the same standard the figure uses** (the engine's track-normalised
times measured it against the all-history standard and applied it to a rolling one, counting a track's drift twice:
flaw 1), on the same runners the standard is made of (winners; every runner for the sectional), each one's deviation first taken against what its own grade
usually runs there relative to its meeting (so a night of top grades is not read as fast going, while a track that is
quicker on every grade still is), and shrunk to zero by n / (n + k), k the
within-meeting over between-meeting variance from earlier days (empirical Bayes). The sectional and the run-in get
their own variant (the engine gave the sectional none, flaw 2, and charged the run-in the whole going allowance,
flaw 3).

**The dog's level** on each figure is a mean over its earlier days, each run weighted by 0.5 to the power of its age
over ``HALF_LIFE`` days (decay in days, not runs), a trial or a troubled run at half weight, then shrunk toward the
par of today's grade at today's track by ``K_PRIOR`` runs of weight: a debutant reads the grade's par, a dog with
twenty recent runs its own figures (flaw: no window allowed for how many runs it rested on). Its spread is shrunk
toward the population's within-dog spread the same way.

**Against the field** (point 9): lengths against the mean of the other runners, and each runner's share of the race
from a softmax of its level over a temperature fitted on the races before ``FIT_BEFORE`` by the winners' log loss.

**Class in lengths** (flaw 4): the par of a grade at a track and distance is the mean winning figure there on earlier
days, so a move from an A3 at one track to an A2 at another is measured in lengths, not in grade numbers.

**The market-implied rating** (step 2.7): a rating a dog, updated after every race by how the SP rated it against
that field: the race-relative log of its SP share, re-centred on the field's average rating before the race, read by
a Kalman filter (drift a day between runs, noise a race, a debut prior), its constants fitted on the races before
``FIT_BEFORE`` by how well the ratings before each race predicted its SP shares. It learns from every runner in
every race, opponent-adjusted, with its own uncertainty.

Every column is read from earlier days only (tests/test_greyhound_pit.py rewrites the last day's results).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "fg_"
STD_DAYS = 365
HALF_LIFE = 90.0
K_PRIOR = 2.0
OUTLIER = 40.0
FIT_BEFORE = "2018-01-01"
TAUS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0)
FIGS = ("tot", "sec", "run")
SOURCE = "GBGB results (sources/gbgb.py): calculated time, run time, sectional, grade, comment, SP"


def _key(df: pd.DataFrame) -> pd.Series:
    return df.track.astype(str) + "|" + pd.to_numeric(df.distance_m, errors="coerce").round().astype("Int64").astype(str)


def _dayn(df: pd.DataFrame) -> np.ndarray:
    return (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()


def rolling_median(df: pd.DataFrame, value: pd.Series, days: int = STD_DAYS) -> pd.Series:
    """Each row's track-and-distance median of ``value`` (NaN rows left out) over the ``days`` before its day."""
    key, day = _key(df), core.day(df)
    pts = pd.DataFrame({"key": key, "day": day, "v": value})
    pts = pd.concat([pts[pts.v.notna()], pts[["key", "day"]].drop_duplicates().assign(v=np.nan)], ignore_index=True)
    pts = pts.sort_values(["key", "day"], kind="stable")
    out = []
    for k, g in pts.groupby("key", sort=False):
        s = g.set_index("day").v
        med = s.rolling(f"{days}D", closed="left", min_periods=1).median()
        out.append(pd.DataFrame({"key": k, "day": s.index, "m": med.to_numpy()}))
    tab = pd.concat(out, ignore_index=True).drop_duplicates(["key", "day"], keep="last")
    m = pd.DataFrame({"key": key, "day": day}).merge(tab, on=["key", "day"], how="left")
    return pd.Series(m.m.to_numpy(), index=df.index)


def eb_variant(df: pd.DataFrame, dev: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Each meeting's variant (seconds; above 0 the night ran slow): the mean of ``dev`` over the meeting's rows that
    have one, shrunk to 0 by n / (n + k), k the within / between variance ratio from the meetings of EARLIER days
    (greyhound/times.py's method, on any deviation). Also the shrinkage n / (n + k)."""
    day = core.day(df)
    mt = pd.DataFrame({"meeting": df.meeting_id.to_numpy(), "day": day.to_numpy(), "dev": dev.to_numpy()})
    g = mt.dropna(subset=["dev"]).groupby("meeting")
    st = pd.DataFrame({"day": g.day.first(), "n": g.dev.size(), "mean": g.dev.mean(),
                       "ss": (g.dev.var(ddof=1) * (g.dev.size() - 1)).fillna(0.0)})
    if not len(st):
        z = pd.Series(0.0, index=df.index)
        return z, z * np.nan
    by_day = st.groupby("day").agg(m=("mean", "size"), s1=("mean", "sum"), s2=("mean", lambda x: (x ** 2).sum()),
                                   ss=("ss", "sum"), dfw=("n", lambda x: (x - 1).clip(lower=0).sum()),
                                   inv_n=("n", lambda x: (1.0 / x).sum())).sort_index()
    cum = by_day.cumsum().shift(1)
    s2_w = cum.ss / cum.dfw.replace(0, np.nan)
    mm = cum.s1 / cum.m
    s2_b = (cum.s2 / cum.m - mm ** 2 - s2_w * cum.inv_n / cum.m).clip(lower=1e-6)
    st = st.join((s2_w / s2_b).rename("k"), on="day")
    rel = st.n / (st.n + st.k)
    st["variant"] = (st["mean"] * rel).where(rel.notna(), 0.0)
    st["rel"] = rel
    m = mt[["meeting"]].join(st[["variant", "rel"]], on="meeting")
    return (pd.Series(m.variant.fillna(0.0).to_numpy(), index=df.index),
            pd.Series(m.rel.to_numpy(), index=df.index))


def figures(df: pd.DataFrame) -> pd.DataFrame:
    """The three figures of each run (lengths, higher faster), their standards and variants (seconds)."""
    L = core.LENGTH_S
    bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in df else \
        pd.Series(False, index=df.index)
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    won = df.won.to_numpy(float) == 1
    t_tot = pd.to_numeric(df.adjusted_time, errors="coerce")
    sec = pd.to_numeric(df.sectional, errors="coerce")
    t_run = pd.to_numeric(df.run_time, errors="coerce") - sec
    times = {"tot": (t_tot, t_tot.where(won)), "sec": (sec, sec), "run": (t_run, t_run.where(won))}
    tmp = df[["race_date", "race_time", "track", "distance_m", "race_class"]].assign(raceid=df.race_id.astype(str))
    out = {}
    for name, (t, std_from) in times.items():
        std = rolling_median(df, std_from.where(~bad & ~trial))
        # the variant is read on the same runners the standard is made of (winners for the whole race and the run-in,
        # every runner for the sectional): losers' times measured against a winners' standard would read as slow going
        dev = (std_from - std).where(~bad & ~trial)
        dev = dev.where(dev.abs() <= OUTLIER * L)
        # each runner against what its own grade usually runs there RELATIVE TO ITS MEETING (earlier days), so a
        # night of top grades is not read as fast going, while a track that is quicker on every grade still is
        rel_dev = dev - dev.groupby(df.meeting_id).transform("mean")
        cpar = core.race_lagged_expanding_mean(tmp.assign(_d=rel_dev), ["track", "distance_m", "race_class"], "_d")
        var, rel = eb_variant(df, dev - cpar.fillna(0.0))
        f = (std - (t - var)) / L
        out[f"f_{name}"] = f.where((f.abs() <= OUTLIER) & ~bad)
        out[f"var_{name}"] = var
        out[f"rel_{name}"] = rel
    return pd.DataFrame(out, index=df.index)


def decayed(df: pd.DataFrame, value: pd.Series, weight: pd.Series, half_life: float = HALF_LIFE) -> pd.DataFrame:
    """Per dog, over its EARLIER days: the decayed sums of weight, weight x value and weight x value^2 (each run
    weighted by ``weight`` x 0.5 ** (its age in days / ``half_life``)). Aligned to ``df.index``; 0 with no history."""
    lam = np.log(2.0) / half_life
    v = value.to_numpy(float)
    w = np.where(np.isfinite(v), weight.to_numpy(float), 0.0)
    v = np.nan_to_num(v)
    d = pd.DataFrame({"dog": df.dog_id.to_numpy(), "dn": _dayn(df), "w": w, "wx": w * v, "wx2": w * v * v})
    per = d.groupby(["dog", "dn"], sort=False)[["w", "wx", "wx2"]].sum().reset_index().sort_values(["dog", "dn"])
    d0 = per.groupby("dog").dn.transform("min")
    up = np.exp(lam * (per.dn - d0).to_numpy(float))
    cum = per[["w", "wx", "wx2"]].mul(up, axis=0).groupby(per.dog).cumsum()
    prev = cum.groupby(per.dog).shift(1).fillna(0.0)
    down = np.exp(-lam * (per.dn - d0).to_numpy(float))
    prev = prev.mul(down, axis=0)
    prev[["dog", "dn"]] = per[["dog", "dn"]].to_numpy()
    m = d[["dog", "dn"]].merge(prev, on=["dog", "dn"], how="left")
    return pd.DataFrame({"sw": m.w.to_numpy(), "swx": m.wx.to_numpy(), "swx2": m.wx2.to_numpy()}, index=df.index)


def race_softmax(x: pd.Series, race: pd.Series, tau: float) -> pd.Series:
    z = x / tau
    z = z - z.groupby(race).transform("max")
    e = np.exp(z)
    return e / e.groupby(race).transform("sum")


def fit_tau(level: pd.Series, race: pd.Series, won: pd.Series, early: pd.Series) -> tuple[float, dict]:
    """The softmax temperature (lengths) that best predicts the winners of the early races."""
    e = early & level.notna()
    full = e.groupby(race).transform("all") & e
    one = won.where(full).groupby(race).transform("sum") == 1
    k = full & one
    if k.sum() < 200:
        return 3.0, {}
    grid = {}
    for tau in TAUS:
        p = race_softmax(level[k], race[k], tau)
        grid[tau] = round(float(-np.log(p[won[k] == 1].clip(1e-9)).mean()), 5)
    return min(grid, key=grid.get), grid


# ---- the market-implied rating -------------------------------------------------------------------------------------

def market_rating(df: pd.DataFrame, prm: dict, upto: int | None = None) -> tuple[pd.DataFrame, float]:
    """The ratings before each row's day (mean, sd, races behind it) and the mean cross-entropy of the observed SP
    shares under the softmax of the ratings before each race (the fit's score; races with a full SP only). Days in
    order; each day's rows read the state after the dog's EARLIER days, then the day's races update it."""
    dog = pd.factorize(df.dog_id)[0]
    dn = _dayn(df)
    race = pd.factorize(df.race_id)[0]
    q = pd.to_numeric(df.sp_p_norm, errors="coerce").to_numpy(float)
    ok_race = np.isfinite(q) & (q > 0)
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(len(df), bool)
    use = ok_race & ~trial
    full = pd.Series(use).groupby(race).transform("all").to_numpy() & (df.field.to_numpy(float) >= 2)
    nd = dog.max() + 1
    R, V, N, LAST = np.zeros(nd), np.full(nd, prm["v0"]), np.zeros(nd), np.full(nd, -1)
    m_out, sd_out, n_out = np.full(len(df), np.nan), np.full(len(df), np.nan), np.zeros(len(df))
    order = np.argsort(dn, kind="stable")
    bounds = np.flatnonzero(np.diff(dn[order])) + 1
    ce_sum, ce_n = 0.0, 0
    for rows in np.split(order, bounds):
        if upto is not None and dn[rows[0]] >= upto:
            break
        dg = dog[rows]
        gap = np.where(LAST[dg] >= 0, dn[rows] - LAST[dg], 0)
        v_pre = V[dg] + prm["q"] * gap
        r_pre = R[dg]
        m_out[rows] = np.where(N[dg] > 0, r_pre, np.nan)
        sd_out[rows] = np.sqrt(v_pre)
        n_out[rows] = N[dg]
        f = rows[full[rows]]
        if not len(f):
            continue
        fr = race[f]
        u, inv = np.unique(fr, return_inverse=True)
        cnt = np.bincount(inv)
        lq = np.log(q[f])
        rp = R[dog[f]]
        mean_lq = np.bincount(inv, weights=lq) / cnt
        mean_r = np.bincount(inv, weights=rp) / cnt
        # the fit's score: the observed shares under the softmax of the ratings before the race
        mx = np.full(len(u), -np.inf)
        np.maximum.at(mx, inv, rp)
        ez = np.exp(rp - mx[inv])
        p = ez / np.bincount(inv, weights=ez)[inv]
        ce_sum += float(-(q[f] * np.log(np.clip(p, 1e-12, None))).sum())
        ce_n += len(u)
        x = lq - mean_lq[inv] + mean_r[inv]                  # its log share against the field, on the field's scale
        dgf = dog[f]
        # a dog twice in a day: its readings averaged
        ud, inv2 = np.unique(dgf, return_inverse=True)
        xm = np.bincount(inv2, weights=x) / np.bincount(inv2)
        vpu = V[ud] + prm["q"] * np.where(LAST[ud] >= 0, dn[f][0] - LAST[ud], 0)
        k = vpu / (vpu + prm["r"])
        R[ud] = R[ud] + k * (xm - R[ud])
        V[ud] = (1 - k) * vpu
        N[ud] += 1
        LAST[ud] = dn[f][0]
    return pd.DataFrame({"m": m_out, "sd": sd_out, "n": n_out}, index=df.index), (ce_sum / ce_n if ce_n else np.nan)


def fit_market_rating(df: pd.DataFrame, before: str = FIT_BEFORE) -> dict:
    """The rating's noise ``r`` (a race), drift ``q`` (a day) and debut variance ``v0`` by the cross-entropy of the
    SP shares on the days before ``before`` (or the first fifth of the days when there are too few)."""
    dn = _dayn(df)
    cut = int((pd.Timestamp(before) - pd.Timestamp("2000-01-01")).days)
    if (dn < cut).sum() < 50_000:
        cut = int(np.sort(dn)[int(len(dn) * 0.2)]) if len(dn) else cut
    grid, best = [], None
    # the first full build (10 Nov 2026, run 42792867097) chose r 0.4, q 0.008, v0 1.0 with q at the grid's edge:
    # the grid reaches further now
    for r in (0.1, 0.2, 0.4, 0.8):
        for q in (0.0005, 0.002, 0.008, 0.02, 0.05):
            for v0 in (0.3, 1.0, 2.0):
                prm = {"r": r, "q": q, "v0": v0}
                _, ce = market_rating(df, prm, upto=cut)
                grid.append((r, q, v0, round(ce, 5) if np.isfinite(ce) else None))
                if np.isfinite(ce) and (best is None or ce < best[0]):
                    best = (ce, prm)
    prm = best[1] if best else {"r": 0.3, "q": 0.002, "v0": 1.0}
    return {**prm, "ce": round(best[0], 5) if best else None, "grid": grid}


# ---- the table -----------------------------------------------------------------------------------------------------

def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    race = df.race_id
    won = pd.to_numeric(df.won, errors="coerce")
    early = core.day(df) < pd.Timestamp(FIT_BEFORE)
    if early.sum() < 0.05 * len(df):
        early = core.day(df) <= core.day(df).sort_values().iloc[int(len(df) * 0.2)] if len(df) else early
    fg = figures(df)
    trouble = pd.to_numeric(df.get("trouble"), errors="coerce").fillna(0.0) if "trouble" in df else 0.0
    trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
    weight = pd.Series(np.where(trial | (trouble > 0), 0.5, 1.0), index=df.index)
    tmp = df[["race_date", "race_time", "track", "distance_m", "race_class"]].assign(raceid=race.astype(str))
    out = pd.DataFrame({"race_id": race.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    notes = {"std_days": STD_DAYS, "half_life_days": HALF_LIFE, "k_prior": K_PRIOR,
             "variant_shrinkage_median": {n: round(float(fg[f"rel_{n}"].median()), 3) for n in FIGS
                                          if fg[f"rel_{n}"].notna().any()}}
    for n in FIGS:
        f = fg[f"f_{n}"]
        # the par of today's grade at today's track and trip, from every runner's figure on earlier days
        par = core.race_lagged_expanding_mean(tmp.assign(_f=f), ["track", "distance_m", "race_class"], "_f")
        pop = core.race_lagged_expanding_mean(tmp.assign(_f=f, _p=0), "_p", "_f")
        prior = par.fillna(pop).fillna(0.0)
        s = decayed(df, f, weight)
        level = (s.swx + K_PRIOR * prior) / (s.sw + K_PRIOR)
        out[f"{PREFIX}{n}"] = level
        out[f"{PREFIX}{n}_n"] = s.sw
        # the spread: the dog's decayed variance, shrunk toward the population's within-dog variance (earlier days)
        mean = s.swx / s.sw.replace(0, np.nan)
        var = (s.swx2 / s.sw.replace(0, np.nan) - mean ** 2).clip(lower=0)
        resid2 = (f - mean) ** 2
        pop_w = core.race_lagged_expanding_mean(tmp.assign(_r=resid2, _p=0), "_p", "_r")
        pop_w = pop_w.fillna(float(resid2[early].mean()) if resid2[early].notna().any() else 9.0)
        out[f"{PREFIX}{n}_sd"] = np.sqrt((s.sw * var.fillna(0.0) + K_PRIOR * pop_w) / (s.sw + K_PRIOR))
        g = level.groupby(race)
        others = (g.transform("sum") - level) / (g.transform("size") - 1).replace(0, np.nan)
        out[f"{PREFIX}{n}_vs_field"] = level - others
        out[f"{PREFIX}{n}_gap"] = level - g.transform("max")
    tau, grid = fit_tau(out[f"{PREFIX}tot"], race, won, early)
    sh = race_softmax(out[f"{PREFIX}tot"], race, tau)
    out[f"{PREFIX}tot_share"] = sh
    out[f"{PREFIX}tot_lshare"] = np.log(sh.clip(1e-9))
    notes.update({"tau": tau, "tau_grid": grid})
    # class in lengths: the par of a grade at a track and trip is its winners' mean figure on earlier days
    cls = core.race_lagged_expanding_mean(tmp.assign(_w=fg.f_tot.where(won == 1)),
                                          ["track", "distance_m", "race_class"], "_w")
    cls = cls.fillna(core.race_lagged_expanding_mean(tmp.assign(_w=fg.f_tot.where(won == 1), _p=0), "_p", "_w"))
    df2 = df.assign(_cls=cls)
    prev_cls = core.prev_day_values(df2, "dog_id", ["_cls"])["_cls"]
    out[f"{PREFIX}class_par"] = cls
    out[f"{PREFIX}class_move"] = cls - prev_cls
    out[f"{PREFIX}class_vs_level"] = out[f"{PREFIX}tot"] - cls
    # the market-implied rating
    prm = fit_market_rating(df)
    mr, ce = market_rating(df, prm)
    out[f"{PREFIX}mr"] = mr.m
    out[f"{PREFIX}mr_sd"] = mr.sd
    out[f"{PREFIX}mr_n"] = mr.n
    filled = mr.m.fillna(mr.m.groupby(race).transform("mean")).fillna(0.0)
    g = filled.groupby(race)
    out[f"{PREFIX}mr_vs_field"] = (filled - (g.transform("sum") - filled) /
                                   (g.transform("size") - 1).replace(0, np.nan)).where(mr.m.notna())
    msh = race_softmax(filled, race, 1.0)
    out[f"{PREFIX}mr_share"] = msh
    out[f"{PREFIX}mr_lshare"] = np.log(msh.clip(1e-9))
    notes["market_rating"] = {**prm, "ce_all": round(ce, 5) if np.isfinite(ce) else None}
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    **{f"{PREFIX}{n}": f"the dog's {lab} level (lengths, higher faster): its figures on earlier days decayed by a "
                       f"{int(HALF_LIFE)}-day half-life, shrunk toward today's grade par by {int(K_PRIOR)} runs"
       for n, lab in (("tot", "whole-race"), ("sec", "early-pace (sectional)"), ("run", "run-in"))},
    **{f"{PREFIX}{n}_n": "the decayed weight of evidence behind it (effective runs)" for n in FIGS},
    **{f"{PREFIX}{n}_sd": "the dog's spread on the figure, shrunk toward the population's within-dog spread"
       for n in FIGS},
    **{f"{PREFIX}{n}_vs_field": "lengths against the mean of the race's other runners" for n in FIGS},
    **{f"{PREFIX}{n}_gap": "lengths behind the race's best" for n in FIGS},
    f"{PREFIX}tot_share": "its share of the race from a softmax of the whole-race levels (temperature fitted early)",
    f"{PREFIX}tot_lshare": "the log of that share",
    f"{PREFIX}class_par": "today's grade's par at the track and trip (winners' mean figure, earlier days; lengths)",
    f"{PREFIX}class_move": "today's grade par less the par of the dog's last race (lengths; above 0 a step up)",
    f"{PREFIX}class_vs_level": "the dog's whole-race level less today's grade par (lengths)",
    f"{PREFIX}mr": "the market-implied rating: its race-relative log SP share, opponent-adjusted, Kalman-filtered",
    f"{PREFIX}mr_sd": "the rating's spread",
    f"{PREFIX}mr_n": "races behind the rating",
    f"{PREFIX}mr_vs_field": "the rating against the mean of the race's other runners",
    f"{PREFIX}mr_share": "its share of the race from a softmax of the field's ratings (the rating's forecast SP share)",
    f"{PREFIX}mr_lshare": "the log of that share",
}
