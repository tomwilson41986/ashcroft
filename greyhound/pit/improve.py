"""Improvement: where a dog is on its own curve, how much of today's conditions it has met, what the population
learns on the nth outing, and form the bare result hid.

Screened on 12 Nov 2026 (wave 4) against what the stack with the trials and optics tables still misses (the screen
of the discovery scripts, 16 months of GBGB to 5 Nov 2026: the set took the residual's RMSE from 0.2701 to 0.2688 on
49,612 runners, z -8.2, and 0.2667 to 0.2659 beyond the wave-3 tables). The screen's ``im_slope_age`` (the slope a
month of age) is not here: its reviewer found it a duplicate of ``im_slope_run`` (correlation 0.95 on the screen's
rows).

**Per-run measures** (each read by LATER days only):

- the figure f = (standard - calculated time) / 0.08, lengths, higher faster, capped at +-25; likewise the sectional
  (+-10) and the run home (calculated time less sectional, +-20). The standard is the mean at the track and trip
  over the MONTHS before the run's month (``K_STD`` runs or more); where those months are too few (a store's first
  months, a new trip), the mean over the EARLIER DAYS (``K_STD`` or more). The screen's script fitted its first two
  months on themselves; a table cannot, so those months read earlier days here;
- the level before the day L = (sum 0.8^(i-1) f_i + 2 par) / (sum 0.8^(i-1) + 2) over the last 6 race days, par the
  shrunk mean figure of today's (track, class) in the months before; the innovation nu = f - L;
- the market surprise: the share of rivals beaten less its mean for the SP rank and field size in the months before.

**The learned sequences.** Eight cells of today's card and the dog's earlier days (nth outing at the track; nth at
the trip by step up or down; nth in the trap; the class move by times in the class; days off by a trial in the gap;
nth run back from a 21-day break; the field's change; age in months by sex). A cell's value is the mean innovation
of the runs that were in it in the months BEFORE the race's month, shrunk to the pooled mean (``K_CURVE`` runs; the
market surprise ``K_MKT``), less the pooled mean; refitted every month from earlier months only; an unseen cell is
0, the store's first month has none. A count cut by the store's start (the dog's first row is within
``COMPLETE_DAYS`` days of the store's first day) is trusted only once it reaches the open bin.

**The own curve**: the ridge slope of the last six race days' figures a run, and their curvature.

**Hidden form**: the last two race days against the four before (sectional, run home, rivals beaten); the run home
improving beyond the result (each over its sd on the rows of the months before); and the excuse score: a ridge of
the NEXT run's innovation on this run's trouble, wide path, break, finish, lead and sectional against the dog's own
norm (this run's innovation a control, not scored), refitted for each month on the pairs whose next run is in an
earlier month (``MIN_PAIRS`` or more).

**The rule.** A dog's past arrives through a one-day shift of its dog-day table; a count subtracts the day's own
rows; every standard, par, curve, market map, sd and weight is fitted on earlier months (or earlier days). The field
of a race is every row of ``runs`` with its race_id. A card row (``is_card``) has its result columns ignored and
gets its features from history and the card (tests/test_greyhound_improve.py). Trials (``ctx["trial_runs"]`` and
any T-class row of ``runs``) are read only as outings: a date, a track, a trip, a trap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "im_"
SOURCE = "GBGB results (calculated times, sectionals, positions, SP ranks, comments) and the trial rows as outings"
LENGTH_S = core.LENGTH_S
K_STD = 30            # runs needed for a track-and-trip standard
K_PAR = 20.0          # runs of prior on a (track, class) par
K_LEVEL = 2.0         # runs of prior (the class par) on the dog's level
DECAY = 0.8           # weight of a run one further back in the level
K_CURVE = 50.0        # runs of prior on a cell of an exposure curve (figure currency)
K_MKT = 150.0         # ... in the market currency
K_MAP = 5.0           # runs of prior on a cell of the market-to-finish map
RIDGE_SLOPE = 8.0     # added to Sxx in the slope over runs
RIDGE_H = 200.0       # ridge on the excuse weights
MIN_PAIRS = 2000      # pairs of consecutive runs needed to fit the excuse weights
MIN_SD = 200          # rows needed for the sd of a last-two-against-four difference
COMPLETE_DAYS = 45    # a dog first seen this long after the store's first day has its whole career in the store
TRIAL_RE = r"^T\d*$"
Z = ["z_trouble", "z_wide", "z_slow", "z_quick", "z_finup", "z_findn", "z_led", "secdev"]

COLUMNS = {
    f"{PREFIX}nth_track": "the population's mean innovation (lengths) on the nth outing at the track, races and "
                          "trials counted, fitted on the months before",
    f"{PREFIX}nth_trip": "the same on the nth outing at the track and trip, by a step up, down or level in distance",
    f"{PREFIX}class_exp": "the same for today's class move (grades up or down, or the change of family) by the times "
                          "the dog has raced in the class",
    f"{PREFIX}break_exp": "the same for the days since the last race day, by whether a trial was run in the gap",
    f"{PREFIX}back_exp": "the same for the nth run back from a break of 21 days or more",
    f"{PREFIX}age_exp": "the same for the dog's age in months, by sex (the population's age curve)",
    f"{PREFIX}exp_sum": "the sum of the eight cells' mean innovations (the six above, the nth time in the trap and "
                        "the field's change)",
    f"{PREFIX}mkt_sum": "the same eight cells in the market's currency: the share of rivals beaten less what the SP "
                        "rank and field size expected, summed",
    f"{PREFIX}ev_var": "the mean over the eight cells of log(the cell's mean squared innovation / the pooled): how "
                       "unpredictable runs in today's cells have been",
    f"{PREFIX}cover": "the share of today's track, trip, trap, class and field size the dog has met before",
    f"{PREFIX}runs": "log(1 + the dog's outings before today, races and trials), for a dog whose whole career is in "
                     "the store",
    f"{PREFIX}slope_run": "the ridge slope of the dog's figure a run over its last six race days (3 or more), "
                          "lengths a run",
    f"{PREFIX}curve": "the curvature of those six figures: mean of the last two - 2 x the middle two + the first two",
    f"{PREFIX}sec_up": "the sectional figure of the last two race days less the four before (lengths)",
    f"{PREFIX}home_up": "the run-home figure (calculated time less sectional) of the last two less the four before",
    f"{PREFIX}beat_up": "the share of rivals beaten in the last two less the four before",
    f"{PREFIX}home_hid": "home_up over its sd less beat_up over its sd (sds of the rows in the months before): the "
                         "run home improving beyond the results",
    f"{PREFIX}hidden": "the excuse score of the last three race days (weights 1, 0.6, 0.36): what their trouble, "
                       "path, break, finish and lead said of the NEXT run's innovation, fitted on earlier months",
    f"{PREFIX}uncashed": "the sum of the excuse scores since the dog last ran above its level",
}
CARD_UNSAFE: tuple[str, ...] = ()


def lagged_cell(code: np.ndarray, mo: np.ndarray, val: np.ndarray):
    """For each row: the sum and count of ``val`` over the rows of its cell in the months BEFORE its month, and the
    pooled mean of every row in the months before. A code below 0 is no cell (0, 0)."""
    code, mo, val = np.asarray(code, np.int64), np.asarray(mo, np.int64), np.asarray(val, float)
    n = len(code)
    ok = np.isfinite(val) & (code >= 0)
    if not ok.any():
        return np.zeros(n), np.zeros(n), np.full(n, np.nan)
    a = pd.DataFrame({"c": code[ok], "m": mo[ok], "v": val[ok]})
    g = a.groupby(["c", "m"]).v.agg(["sum", "count"]).reset_index().sort_values(["c", "m"])
    g[["S", "N"]] = g.groupby("c")[["sum", "count"]].cumsum()
    left = pd.DataFrame({"c": code, "m": mo, "i": np.arange(n)}).sort_values("m", kind="stable")
    r = pd.merge_asof(left, g[["c", "m", "S", "N"]].sort_values("m", kind="stable"), on="m", by="c",
                      allow_exact_matches=False).sort_values("i")
    p = a.groupby("m").v.agg(["sum", "count"]).sort_index().cumsum()
    p = p.reindex(np.arange(int(mo.min()), int(mo.max()) + 2)).ffill().shift(1)
    pool = (p["sum"] / p["count"]).reindex(mo).to_numpy()
    return r.S.fillna(0.0).to_numpy(), r.N.fillna(0.0).to_numpy(), pool


def shrunk(code, mo, val, k: float, relative: bool = True):
    """The cell's mean over the months before, shrunk to the pooled mean by ``k`` rows (less it when ``relative``)."""
    s, n, pool = lagged_cell(code, mo, val)
    f = (s + k * np.nan_to_num(pool)) / (n + k)
    f = f - np.nan_to_num(pool) if relative else f
    return np.where((np.asarray(code) >= 0) & np.isfinite(pool), f, np.nan), n


def lagged_sd(mo: np.ndarray, val: np.ndarray) -> np.ndarray:
    """The sd of ``val`` over every row in the months before the row's month (``MIN_SD`` rows or more)."""
    mo, val = np.asarray(mo, np.int64), np.asarray(val, float)
    ok = np.isfinite(val)
    if not ok.any():
        return np.full(len(mo), np.nan)
    a = pd.DataFrame({"m": mo[ok], "v": val[ok], "v2": val[ok] ** 2, "n": 1.0}).groupby("m").sum().sort_index().cumsum()
    a = a.reindex(np.arange(int(mo.min()), int(mo.max()) + 2)).ffill().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = a.v2 / a.n - (a.v / a.n) ** 2
        sd = np.sqrt(var.where((a.n >= MIN_SD) & (var > 0)))
    return sd.reindex(mo).to_numpy()


def codes(*cols) -> np.ndarray:
    """One integer a cell; -1 where any part is missing."""
    d = pd.concat([pd.Series(np.asarray(c)).reset_index(drop=True) for c in cols], axis=1)
    d.columns = [f"k{i}" for i in range(d.shape[1])]
    bad = d.isna().any(axis=1).to_numpy()
    out = d.groupby(list(d.columns), dropna=True, sort=False).ngroup().to_numpy(float)
    return np.where(bad | ~np.isfinite(out) | (out < 0), -1, out).astype(np.int64)


def _flag(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].fillna(False).to_numpy(bool) if col in df else np.zeros(len(df), bool)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float) if col in df else pd.Series(np.nan, index=df.index)


def _dn(dates: pd.Series) -> pd.Series:
    return (pd.to_datetime(dates, errors="coerce").dt.normalize() - pd.Timestamp("2000-01-01")).dt.days


def _frame(runs: pd.DataFrame) -> pd.DataFrame:
    """The columns the table reads, one row a row of ``runs`` (``row``: its place in ``runs``); a card row's result
    columns emptied."""
    card = _flag(runs, "is_card")
    r = pd.DataFrame({"row": np.arange(len(runs)), "race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    r["card"] = card
    cls = runs.race_class.astype(object).where(runs.race_class.notna(), "?").astype(str).str.strip().to_numpy() \
        if "race_class" in runs else "?"
    r["race_class"] = cls
    r["is_trial"] = _flag(runs, "is_trial") | r.race_class.str.upper().str.match(TRIAL_RE).to_numpy()
    r["race_date"] = runs.race_date.astype(str).to_numpy()
    d = pd.to_datetime(r.race_date, errors="coerce")
    r["d"] = d
    r["t"] = pd.to_datetime(runs.t).to_numpy() if "t" in runs else d
    r["dn"] = _dn(r.race_date)
    r["mo"] = d.dt.year * 12 + d.dt.month
    r["track"] = runs.track.astype(object).where(runs.track.notna(), "?").astype(str).to_numpy() if "track" in runs else "?"
    for c in ("distance_m", "trap", "grade", "position", "adjusted_time", "sectional", "market_pos"):
        r[c] = _num(runs, c).to_numpy()
    r["grade_family"] = runs.grade_family.astype(object).to_numpy() if "grade_family" in runs else None
    r["dnf"] = _flag(runs, "dnf") & ~card
    r["comment"] = runs.comment.fillna("").astype(str).to_numpy() if "comment" in runs else ""
    r["born"] = runs.born.astype(str).to_numpy() if "born" in runs else ""
    r["sex"] = runs.sex.astype(str).str.lower().to_numpy() if "sex" in runs else ""
    # a card has no result: whatever a card row carries in a result column is not read
    for c in ("position", "adjusted_time", "sectional", "market_pos"):
        r.loc[r.card, c] = np.nan
    r.loc[r.card, "comment"] = ""
    r.loc[~(r.position >= 1), "position"] = np.nan
    for c in ("adjusted_time", "sectional"):
        r.loc[~(r[c] > 0), c] = np.nan
    return r


def _trial_rows(r: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    """The trials as outings: dog, day number, track, trip, trap (the T-class rows of ``runs`` and ctx's)."""
    cols = ["dog_id", "dn", "track", "distance_m", "trap", "race_id"]
    parts = [r.loc[r.is_trial & ~r.card, cols]]
    if extra is not None and len(extra):
        e = extra[extra.dog_id.notna() & extra.race_date.notna()]
        t = pd.DataFrame({"dog_id": e.dog_id.to_numpy(), "dn": _dn(e.race_date).to_numpy(),
                          "track": e.track.astype(str).to_numpy() if "track" in e else "?",
                          "distance_m": _num(e, "distance_m").to_numpy(), "trap": _num(e, "trap").to_numpy(),
                          "race_id": e.race_id.to_numpy() if "race_id" in e else np.arange(len(e))})
        parts.append(t)
    tr = pd.concat(parts, ignore_index=True).dropna(subset=["dog_id", "dn"])
    tr = tr.drop_duplicates(["race_id", "dog_id"], keep="last")
    tr["dog_id"] = tr.dog_id.astype(r.dog_id.dtype)
    tr["dn"] = tr.dn.astype(np.int64)
    return tr


def _standards(r: pd.DataFrame, col: str) -> np.ndarray:
    """The mean of ``col`` at the track and trip over the months before the row's month; with fewer than ``K_STD``
    runs there, over the earlier days."""
    s, n, _ = lagged_cell(r.td, r.mo, r[col])
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.where(n >= K_STD, s / np.maximum(n, 1), np.nan)
        need = ~np.isfinite(std) & (r.td.to_numpy() >= 0)
        if need.any():
            v = r[col]
            w = core.prior_window(pd.DataFrame({"td": r.td.to_numpy(), "race_date": r.race_date.to_numpy(),
                                                "v": v.fillna(0.0).to_numpy(), "k": v.notna().to_numpy(float)}),
                                  ["td"], ["v", "k"], None)
            early = np.where(w.k.to_numpy() >= K_STD, w.v.to_numpy() / np.maximum(w.k.to_numpy(), 1), np.nan)
            std = np.where(need, early, std)
    return np.where(r.td.to_numpy() >= 0, std, np.nan)


def _excuse(dd: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """For each dog-day (``dd`` in dog, day order): the excuse score of its last three earlier days and the sum of the
    scores since the dog last ran above its level, under the weights fitted for the day's month on pairs whose next
    run is in an earlier month."""
    n = len(dd)
    dog = dd.dog_id.to_numpy()
    mo = dd.mo.to_numpy(np.int64)
    nu = dd.nu.to_numpy(float)
    X = dd[Z + ["nu"]].fillna(0.0).to_numpy(float)
    k = X.shape[1]
    same_next = np.r_[dog[1:] == dog[:-1], False]
    nu_next = np.where(same_next, np.r_[nu[1:], np.nan], np.nan)
    mo_next = np.where(same_next, np.r_[mo[1:], -1], -1)
    fit = np.isfinite(nu_next) & np.isfinite(nu)
    months = np.unique(mo)
    OM = np.full((len(months), k - 1), np.nan)
    C0 = np.full(len(months), np.nan)
    weights = {}
    if fit.any():
        fm = np.unique(mo_next[fit])
        Xf, yf, gi = X[fit], nu_next[fit], np.searchsorted(fm, mo_next[fit])
        cn = np.cumsum(np.bincount(gi, minlength=len(fm)).astype(float))
        sx = np.zeros((len(fm), k))
        np.add.at(sx, gi, Xf)
        sx = np.cumsum(sx, 0)
        sy = np.cumsum(np.bincount(gi, weights=yf, minlength=len(fm)))
        xy = np.zeros((len(fm), k))
        np.add.at(xy, gi, Xf * yf[:, None])
        xy = np.cumsum(xy, 0)
        order = np.argsort(gi, kind="stable")
        bounds = np.searchsorted(gi[order], np.arange(len(fm) + 1))
        xx = np.zeros((k, k))
        XX = []
        for j in range(len(fm)):
            b = Xf[order[bounds[j]:bounds[j + 1]]]
            xx = xx + b.T @ b
            XX.append(xx)
        for i, m in enumerate(months):
            j = int(np.searchsorted(fm, m)) - 1                      # the last fitting month before m
            if j < 0 or cn[j] < MIN_PAIRS:
                continue
            mu, my = sx[j] / cn[j], sy[j] / cn[j]
            A = XX[j] - cn[j] * np.outer(mu, mu) + RIDGE_H * np.eye(k)
            om = np.linalg.solve(A, xy[j] - cn[j] * mu * my)
            OM[i], C0[i] = om[:-1], float(mu[:-1] @ om[:-1])
            weights[int(m)] = dict(zip(Z + ["nu"], np.round(om, 4).tolist()), n=int(cn[j]))
    mi = np.searchsorted(months, mo)
    om_r, c0_r = OM[mi], C0[mi]                                       # the reader's month's weights
    Zm = X[:, :-1]
    idx = np.arange(n)
    num, den = np.zeros(n), np.zeros(n)
    has1 = np.zeros(n, bool)
    for lag, w in ((1, 1.0), (2, 0.6), (3, 0.36)):
        src = idx - lag
        ok = src >= 0
        ok[ok] = dog[src[ok]] == dog[ok]
        h = np.einsum("ij,ij->i", Zm[np.where(ok, src, 0)], om_r) - c0_r
        num += np.where(ok, w * h, 0.0)
        den += np.where(ok, w, 0.0)
        if lag == 1:
            has1 = ok
    with np.errstate(invalid="ignore", divide="ignore"):
        h3 = np.where(has1, num / den, np.nan)
    # the scores since the last run above the level: rows c + 1 .. j - 1, c the last earlier dog-day with nu > 0
    start = np.r_[True, dog[1:] != dog[:-1]]
    first = np.maximum.accumulate(np.where(start, idx, 0))
    lastc = np.maximum.accumulate(np.where(nu > 0, idx, -1))
    c = np.maximum(np.r_[-1, lastc[:-1]], first - 1)                  # as of the day before
    CZ = np.vstack([np.zeros((1, k - 1)), np.cumsum(Zm, 0)])
    u = np.einsum("ij,ij->i", CZ[idx] - CZ[c + 1], om_r) - (idx - 1 - c) * c0_r
    u = np.where(has1, u, np.nan)
    return h3, u, weights


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    from greyhound.metrics import parse_comment
    cols = list(COLUMNS)
    table = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    allr = _frame(runs)
    tr = _trial_rows(allr, (ctx or {}).get("trial_runs"))
    r = allr[~allr.is_trial & allr.d.notna()].copy()
    notes = {"rows": int(len(runs)), "race_rows": int(len(r)), "trial_rows": int(len(tr)),
             "card_rows": int(allr.card.sum())}
    if not len(r):
        for c in cols:
            table[c] = np.float32(np.nan)
        return table, notes
    r = r.sort_values(["dog_id", "t"], kind="stable").reset_index(drop=True)
    r["dn"] = r.dn.astype(np.int64)
    r["mo"] = r.mo.astype(np.int64)
    r["field"] = r.groupby("race_id").dog_id.transform("size").astype(float)       # every row of the race

    # ---- per-run figures against the standard for the track and trip (earlier months, else earlier days)
    r["home_t"] = r.adjusted_time - r.sectional
    r["td"] = codes(r.track, r.distance_m)
    for col, name, cap in (("adjusted_time", "fig", 25.0), ("sectional", "sec", 10.0), ("home_t", "home", 20.0)):
        r[name] = ((_standards(r, col) - r[col]) / LENGTH_S).clip(-cap, cap)
    r["beat"] = ((r.field - r.position) / (r.field - 1)).where(r.field > 1)
    r.loc[r.dnf & (r.field > 1), "beat"] = 0.0
    notes["with_a_figure"] = round(float(r.fig.notna().mean()), 4)

    # ---- the market-to-finish map (the share of rivals beaten by SP rank and field), months before only
    exp_beat, _ = shrunk(codes(r.market_pos, r.field), r.mo, r.beat, K_MAP, relative=False)
    r["msur"] = r.beat - exp_beat

    # ---- each run's comment codes (read later only as the past)
    memo = {c: parse_comment(c) for c in pd.unique(r.comment)}
    pc = pd.DataFrame([memo[c] for c in r.comment], index=r.index, columns=["brk", "epace", "led", "trouble", "fin", "path"])
    r["z_trouble"] = pc.trouble.fillna(0).clip(0, 4)
    r["z_wide"] = (pc.path >= 2.5).astype(float)
    r["z_slow"] = (pc.brk < 0).astype(float)
    r["z_quick"] = (pc.brk > 0).astype(float)
    r["z_finup"] = (pc.fin > 0).astype(float)
    r["z_findn"] = (pc.fin < 0).astype(float)
    r["z_led"] = pc.led.fillna(0)

    # ---- dog-days: one row a dog a day; all a later run reads of the past comes through a shift of these
    means = ["fig", "sec", "home", "beat", "msur", "z_trouble", "z_wide", "z_slow", "z_quick", "z_finup", "z_findn", "z_led"]
    by = r.groupby(["dog_id", "dn"], sort=True)
    dd = by[means].mean().join(by[["track", "distance_m", "race_class", "grade", "grade_family", "field", "mo"]].last())
    dd = dd.reset_index()
    g = dd.groupby("dog_id", sort=False)
    for c in ("fig", "sec", "home", "beat"):
        for i in range(1, 7):
            dd[f"{c}_l{i}"] = g[c].shift(i)
    dd["dn_l1"] = g.dn.shift(1)
    for c in ("distance_m", "race_class", "grade", "grade_family", "field"):
        dd[f"p_{c}"] = g[c].shift(1)
    dd["n_days"] = g.cumcount()                                       # race days before this one, in the store
    dd["gap"] = dd.dn - dd.dn_l1
    brk = (dd.gap >= 21) | dd.gap.isna()
    seg = brk.groupby(dd.dog_id).cumsum()
    dd["n_back"] = dd.groupby([dd.dog_id, seg]).cumcount()            # 0: today is the first run back
    dd["seen_break"] = (dd.gap >= 21).groupby(dd.dog_id).cumsum() > 0
    keep = ["dog_id", "dn", "n_days", "gap", "n_back", "seen_break"] + [c for c in dd.columns if "_l" in c or c.startswith("p_")]
    r = r.merge(dd[keep], on=["dog_id", "dn"], how="left")

    # ---- complete careers: the dog's first row of any kind against the store's first day
    seen = pd.concat([r[["dog_id", "dn"]], tr[["dog_id", "dn"]]], ignore_index=True)
    first = seen.groupby("dog_id").dn.min()
    r["complete"] = r.dog_id.map(first).to_numpy() >= int(seen.dn.min()) + COMPLETE_DAYS
    notes["complete_careers"] = round(float(r.complete.mean()), 4)

    # ---- exposures counted on earlier days, races and trials together
    o = pd.concat([r[["dog_id", "dn", "track", "distance_m", "trap"]].assign(race=1, i=np.arange(len(r))),
                   tr[["dog_id", "dn", "track", "distance_m", "trap"]].assign(race=0, i=-1)], ignore_index=True)
    o[["distance_m", "trap"]] = o[["distance_m", "trap"]].fillna(-1.0)
    o = o.sort_values(["dog_id", "dn"], kind="stable").reset_index(drop=True)
    is_race = (o.race == 1).to_numpy()

    def before(keys: list[str]) -> np.ndarray:
        n = (o.groupby(["dog_id"] + keys).cumcount() - o.groupby(["dog_id", "dn"] + keys).cumcount()).to_numpy()
        out = np.zeros(len(r))
        out[o.i.to_numpy()[is_race]] = n[is_race]
        return out
    r["n_track"] = before(["track"])
    r["n_trip"] = before(["track", "distance_m"])
    r["n_trap"] = before(["trap"])
    r["n_any"] = before([])
    if len(tr):                                                       # the last trial on an earlier day
        td_ = tr[["dog_id", "dn"]].drop_duplicates().sort_values("dn").rename(columns={"dn": "tdn"})
        td_["tdn2"] = td_.tdn
        q = pd.merge_asof(r[["dog_id", "dn"]].assign(i=r.index).sort_values("dn", kind="stable"), td_, left_on="dn",
                          right_on="tdn", by="dog_id", allow_exact_matches=False).set_index("i").tdn2.reindex(r.index)
        r["trial_in_gap"] = (q > (r.dn - r.gap.fillna(1e9))).astype(float)
    else:
        r["trial_in_gap"] = 0.0
    r["n_class"] = (r.groupby(["dog_id", "race_class"]).cumcount() - r.groupby(["dog_id", "dn", "race_class"]).cumcount()).to_numpy()
    r["n_field"] = (r.groupby(["dog_id", "field"]).cumcount() - r.groupby(["dog_id", "dn", "field"]).cumcount()).to_numpy()

    # ---- the level before the day, shrunk to today's class par (months before); each run's innovation
    par, _ = shrunk(codes(r.track, r.race_class), r.mo, r.fig, K_PAR, relative=False)
    w = DECAY ** np.arange(6)
    F = r[[f"fig_l{i}" for i in range(1, 7)]].to_numpy(float)
    ok = np.isfinite(F)
    sw = (ok * w).sum(1)
    swx = np.where(ok, F, 0.0) @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.where(np.isfinite(par), (swx + K_LEVEL * np.nan_to_num(par)) / (sw + K_LEVEL),
                         np.where(sw > 0, swx / np.maximum(sw, 1e-9), np.nan))
    r["nu"] = r.fig - level
    r["nu2"] = r.nu ** 2

    # ---- today's cells (card facts and earlier days only)
    def bins(n: pd.Series, edges: list, valid_from: int) -> pd.Series:
        """The exposure bin; unknown where the store's start cuts the career and the count is below the open bin."""
        b = pd.Series(np.searchsorted(edges, n.to_numpy(), side="right").astype(float), index=n.index)
        return b.where(r.complete | (n >= valid_from))
    age_m = (r.d - pd.to_datetime("15-" + r.born, format="%d-%b-%Y", errors="coerce")).dt.days / 30.44
    bitch = (r.sex == "b").astype(int)
    dstep = r.distance_m - r.p_distance_m
    step = np.sign(dstep.where(dstep.abs() > 30, 0.0)).where(r.p_distance_m.notna() & r.distance_m.notna())
    gd = (r.grade - r.p_grade).clip(-2, 2)
    same = (r.grade_family == r.p_grade_family) & gd.notna()
    move = pd.Series(np.where(same, "g" + gd.round().fillna(0).astype(int).astype(str),
                              r.p_grade_family.astype(str) + ">" + r.grade_family.astype(str)), index=r.index)
    move = move.where(r.p_race_class.notna())
    gapb = pd.Series(np.searchsorted([5, 8, 11, 15, 22, 36, 61, 121], r.gap.fillna(-1).to_numpy(), side="right").astype(float),
                     index=r.index).where(r.gap.notna())
    fch = np.sign(r.field - r.p_field).where(r.p_field.notna())
    back = r.n_back.clip(upper=4).astype(float).where(r.seen_break | r.complete | (r.n_back >= 4))
    cells = {
        "track": codes(bins(r.n_track, [1, 2, 3, 4, 7, 13], 13)),
        "trip": codes(bins(r.n_trip, [1, 2, 3, 4, 7, 13], 13), step),
        "trap": codes(bins(r.n_trap, [1, 2, 3, 4, 7], 7)),
        "class": codes(move, bins(r.n_class, [1, 3], 3)),
        "break": codes(gapb, r.trial_in_gap),
        "back": codes(back, r.n_days.clip(upper=3).where(r.complete, 9)),
        "field": codes(fch, r.field.clip(lower=4), bins(r.n_field, [1, 4], 4)),
        "age": codes(age_m.clip(15, 66).round(), bitch),
    }
    exp = {}
    for e, c in cells.items():
        f, _ = shrunk(c, r.mo, r.nu, K_CURVE)
        m, _ = shrunk(c, r.mo, r.msur, K_MKT)
        v, _ = shrunk(c, r.mo, r.nu2, K_CURVE, relative=False)
        _, _, pool2 = lagged_cell(c, r.mo, r.nu2)
        with np.errstate(invalid="ignore", divide="ignore"):
            lv = np.log(v / pool2)
        exp[e] = (f, m, np.where(np.isfinite(lv), lv, np.nan))
    notes["cells"] = {e: int(c.max()) + 1 for e, c in cells.items()}

    out = pd.DataFrame(index=r.index)
    out[f"{PREFIX}nth_track"] = exp["track"][0]
    out[f"{PREFIX}nth_trip"] = exp["trip"][0]
    out[f"{PREFIX}class_exp"] = exp["class"][0]
    out[f"{PREFIX}break_exp"] = exp["break"][0]
    out[f"{PREFIX}back_exp"] = exp["back"][0]
    out[f"{PREFIX}age_exp"] = exp["age"][0]
    A, M, V = (np.column_stack([exp[e][i] for e in cells]) for i in range(3))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"{PREFIX}exp_sum"] = np.where(np.isfinite(A).any(1), np.nansum(A, 1), np.nan)
            out[f"{PREFIX}mkt_sum"] = np.where(np.isfinite(M).any(1), np.nansum(M, 1), np.nan)
            out[f"{PREFIX}ev_var"] = np.where(np.isfinite(V).any(1), np.nanmean(V, 1), np.nan)

            # ---- unexposed: how little of today's conditions the dog has met
            out[f"{PREFIX}cover"] = np.column_stack([(r[c] > 0).to_numpy() for c in
                                                     ("n_track", "n_trip", "n_trap", "n_class", "n_field")]).mean(1)
            out[f"{PREFIX}runs"] = np.log1p(r.n_any).where(r.complete)

            # ---- the dog's own curve over its last six race days
            F6 = F[:, ::-1]                                           # oldest first
            ok6 = np.isfinite(F6)
            n6 = ok6.sum(1)
            Xr = np.where(ok6, np.tile(np.arange(6.0), (len(r), 1)), np.nan)
            Fn = np.where(ok6, F6, np.nan)
            sxy = np.nansum((Xr - np.nanmean(Xr, 1, keepdims=True)) * (Fn - np.nanmean(Fn, 1, keepdims=True)), 1)
            sxx = np.nansum((Xr - np.nanmean(Xr, 1, keepdims=True)) ** 2, 1)
            out[f"{PREFIX}slope_run"] = np.where(n6 >= 3, sxy / (sxx + RIDGE_SLOPE), np.nan)
            out[f"{PREFIX}curve"] = np.where(n6 == 6, F6[:, 4:].mean(1) - 2 * F6[:, 2:4].mean(1) + F6[:, :2].mean(1), np.nan)

            # ---- hidden form: the last two against the four before
            def recent(c: str) -> np.ndarray:
                x = r[[f"{c}_l{i}" for i in range(1, 7)]].to_numpy(float)
                a, b = np.nanmean(x[:, :2], 1), np.nanmean(x[:, 2:], 1)
                return np.where(np.isfinite(x[:, :2]).any(1) & (np.isfinite(x[:, 2:]).sum(1) >= 2), a - b, np.nan)
            dsec, dhome, dbeat = recent("sec"), recent("home"), recent("beat")
    out[f"{PREFIX}sec_up"] = dsec
    out[f"{PREFIX}home_up"] = dhome
    out[f"{PREFIX}beat_up"] = dbeat
    # the sds from the rows of earlier months (the screen's script took them from the rows before 1 Jun 2026)
    out[f"{PREFIX}home_hid"] = dhome / lagged_sd(r.mo, dhome) - dbeat / lagged_sd(r.mo, dbeat)

    # ---- the excuse score: what a past run's trouble, path, break and finish said of the NEXT run's innovation
    dd = dd.merge(r.groupby(["dog_id", "dn"]).agg(nu=("nu", "mean"), secdev=("sec", "mean")).reset_index(),
                  on=["dog_id", "dn"], how="left")
    dd["secdev"] = (dd.secdev - dd[[f"sec_l{i}" for i in range(1, 7)]].mean(axis=1)).fillna(0.0).clip(-6, 6)
    h3, unc, weights = _excuse(dd)
    got = r[["dog_id", "dn"]].merge(dd[["dog_id", "dn"]].assign(h3=h3, u=unc), on=["dog_id", "dn"], how="left")
    out[f"{PREFIX}hidden"] = got.h3.to_numpy()
    out[f"{PREFIX}uncashed"] = got.u.to_numpy()
    if weights:
        notes["excuse_weights_last_month"] = weights[max(weights)]
        notes["excuse_months_fitted"] = len(weights)

    vals = np.full((len(runs), len(cols)), np.nan, np.float32)
    vals[r.row.to_numpy()] = out[cols].to_numpy(np.float32)
    for j, c in enumerate(cols):
        table[c] = vals[:, j]
    notes["non_null"] = {c: round(float(np.isfinite(vals[:, j]).mean()), 4) for j, c in enumerate(cols)}
    return table, notes
