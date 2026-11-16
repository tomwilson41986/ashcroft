"""Debutants and first-time conditions: the prior for a dog with little race form.

Screened on 12 Nov 2026 (wave 4) against what the stack with the trials and optics tables still misses: the set took
the stack's miss from 0.2701 to 0.2689 (RMSE, 49,612 runners, z -5.6) and 0.2667 to 0.2657 beyond wave 3, most of it
from the time gap (``db_tt_gap``) and the price map (``db_prior_gap``); the pedigree, track and trap priors were
close to null alone on 16 months and are kept as the map's inputs and for the longer history.

The trials table reads a debutant's trial times; breeding reads a sire's progeny over ALL their runs. Nothing read
how the MARKET prices first runs, how the dog was seeded, or let a prior fade as the dog's own form arrives.

- **Experience**: ``n`` = the dog's races on earlier days. Its start is "seen" when its first race or timed trial in
  the data is at least ``SEEN_AFTER`` days after the data's first day, or it was born no more than ``BORN_BEFORE``
  days before that day; otherwise ``CENSOR_ADD`` runs are assumed before the data. ``db_w = 1 / (1 + n)`` multiplies
  the priors (1, 1/2, 1/3 ...). "First" is n = 0, "early" n < 3, both for a start that was seen.
- **Per earlier run**: lsp = the log SP share less its race's mean (how short the market made it); won and the SP
  share (actual and expected); the share of the field beaten less a half.
- **Pedigree**: the sire's earlier first runs' lsp, the dam's progeny's earlier early runs' lsp, their early winners
  over expected and early field beaten, the dog's own runs taken out, shrunk to the population's value so far.
- **Track and seeding**: how the market has priced debutants in this grade family at this track (cell shrunk to the
  track, the track to GB), debutants' price by trap, the cell's usual debut grade against today's.
- **First time** at a track, a trip (up or down, by distance band) or a grade family: the re-rating (lsp less the
  dog's own mean earlier lsp) earlier first-timers of that cell got.
- **The time gap**: the other runners' mean earlier calculated time at today's track and trip less the dog's own best
  there (trials included), in lengths.
- **The debut price map**: a ridge of lsp on the above for a seen dog's first ten runs.

Every prior, population value and the price map is refitted from earlier days only: each sum is over days strictly
before the race's day (``_prior``), and the map is refitted at each month's start on the rows of earlier months
(``MIN_FIT`` rows at least; none before). A card row (no result) gets the same values: nothing of the day is read
but the card (track, distance, class, trap, the field's dogs, and the dog's sire, dam, birth month and sex). The
field is every row of ``runs`` with the race's id. The trainer is never read.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from greyhound.pit import core, trials as trials_mod

PREFIX = "db_"
SOURCE = ("GBGB results (the SP, the result, the class, the trap, sire, dam, birth month, sex) and the trial rows "
          "(classes T, T1-T4) that data.clean drops")
EPOCH = pd.Timestamp("2000-01-01")
SEEN_AFTER = 92            # a first outing this many days after the data's first day is a career start we saw
BORN_BEFORE = 487          # ... or a dog born no more than this many days before the data's first day (~16 months)
CENSOR_ADD = 20.0          # runs assumed before the data for a dog whose start was not seen
K_SIRE_LSP, K_SIRE_AE, K_SIRE_FLD = 8.0, 4.0, 10.0
K_DAM_LSP, K_DAM_AE, K_DAM_FLD = 5.0, 3.0, 6.0
K_CELL, K_TRACK, K_TRACK_AE, K_TRAP, K_FT = 20.0, 50.0, 10.0, 50.0, 30.0
MIN_DEBUTS = 5             # debuts in a (track, grade family) cell before its usual debut grade is read
RIDGE = 20.0
MIN_FIT = 800              # rows of earlier months before the price map is fitted
MAP_RUNS = 10              # the price map covers a seen dog's first ten runs
GAP_CLIP = 40.0

COLUMNS = {
    f"{PREFIX}w": "the prior's weight: 1 / (1 + races before today), 20 more assumed when the dog's start is before "
                  "the data",
    f"{PREFIX}sire_first_lsp_w": "the sire's earlier first runs' race-centred log SP share against the population's "
                                 "debutants so far (shrunk, k 8), x w",
    f"{PREFIX}sire_early_ae_w": "log of the sire's progeny's winners over expected in their first three runs, earlier "
                                "days (shrunk to the population's, k 4), x w",
    f"{PREFIX}sire_early_fld_w": "the sire's progeny's share of the field beaten less a half in their first three "
                                 "runs, earlier days (shrunk to 0, k 10), x w",
    f"{PREFIX}dam_early_lsp_w": "the dam's progeny's log SP share in their first three runs against the population's "
                                "(shrunk, k 5), x w",
    f"{PREFIX}dam_early_ae_w": "log of the dam's progeny's winners over expected in their first three runs (shrunk, "
                               "k 3), x w",
    f"{PREFIX}dam_early_fld_w": "the dam's progeny's share of the field beaten less a half in their first three runs "
                                "(shrunk to 0, k 6), x w",
    f"{PREFIX}age_debut": "the dog's age in days at its first race (born taken as the 15th); none when its start is "
                          "before the data",
    f"{PREFIX}bitch_w": "1 for a bitch, x w",
    f"{PREFIX}trk_deb_lsp_w": "the log SP share earlier debutants got in this grade family at this track (cell shrunk "
                              "to the track, k 20; the track to GB, k 50), x w",
    f"{PREFIX}trap_deb_lsp_w": "the log SP share earlier debutants got from today's trap against all debutants (GB, "
                               "k 50), x w",
    f"{PREFIX}grade_seed_w": "the cell's mean debut grade number on earlier days (5 debuts at least) less today's "
                             "grade number (above 0: seeded into a better grade than usual), x w",
    f"{PREFIX}field_inexp": "the race's other runners' summed w (how inexperienced the opposition is)",
    f"{PREFIX}ft_track": "first run at this track (not a debut): the re-rating earlier first-timers here got (their "
                         "log SP share less their own earlier mean; shrunk to GB's, k 30); 0 otherwise",
    f"{PREFIX}ft_trip": "first run at this distance: the same for earlier first-timers stepping up or down into this "
                        "distance band; 0 otherwise",
    f"{PREFIX}ft_fam": "first run in this grade family: the same for earlier first-timers in the family; 0 otherwise",
    f"{PREFIX}prior_lsp": "the debut price map: the log SP share forecast for a seen dog in its first ten runs (ridge "
                          "on the priors, the time gap and the run count, fitted on earlier months only)",
    f"{PREFIX}prior_gap": "the map's forecast less the dog's own mean log SP share on earlier days (from the second "
                          "run)",
    f"{PREFIX}trial_gap_w": "the time gap (db_tt_gap) x w for a seen dog in its first three runs",
    f"{PREFIX}tt_gap": "the other runners' mean earlier calculated time at today's track and trip less the dog's own "
                       "best there on earlier days, trials included (lengths; above 0: faster than the field)",
}
EV = ["d1", "d_lsp", "e1", "e_lsp", "e_w", "e_e", "e_fn", "e_fld"]
MAP_INPUTS = ("n1", "n2", "age", "bitch", "gap", "gap_na", "ntr", "sire_lsp", "sire_ae", "sire_fld", "dam_lsp",
              "dam_fld", "cell", "trk_ae", "trapl", "seed", "inexp", "own", "own_na", "trap1", "trap2", "trap3",
              "trap4", "trap5", "trap6", "gap_n0", "own_n1", "n39", "ln", "gap_n39", "own_n39", "own_w")


def _dn(dates: pd.Series) -> pd.Series:
    return (pd.to_datetime(dates, errors="coerce").dt.normalize() - EPOCH).dt.days


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "").astype(str).str.strip().str.lower()


def _prior(E: pd.DataFrame, Q: pd.DataFrame, key: str, vals: list[str], how: str = "sum") -> pd.DataFrame:
    """For each row of Q: the sum (or min) of E[vals] over E's rows with the same ``key`` on days STRICTLY before the
    row's day. 0 (sum) or NaN (min) with none."""
    if not len(E):
        return pd.DataFrame(0.0 if how == "sum" else np.nan, index=Q.index, columns=vals)
    g = E.groupby([key, "day"], sort=True)[vals].agg(how)
    g = (g.groupby(level=0).cumsum() if how == "sum" else g.groupby(level=0).cummin()).reset_index()
    g = g.sort_values("day", kind="stable")
    q = Q[[key, "day"]].reset_index().sort_values("day", kind="stable")
    m = pd.merge_asof(q, g, on="day", by=key, allow_exact_matches=False).set_index("index").reindex(Q.index)
    return m[vals].fillna(0.0) if how == "sum" else m[vals]


def _frame(runs: pd.DataFrame) -> pd.DataFrame:
    """The columns the table reads, one row a row of ``runs`` (a column a frame lacks is unknown)."""
    def col(name, numeric=True):
        if name not in runs:
            return pd.Series(np.nan, index=runs.index, dtype=float if numeric else object)
        return pd.to_numeric(runs[name], errors="coerce").astype(float) if numeric else runs[name]

    Q = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    Q["day"] = _dn(runs.race_date).to_numpy()
    Q["tie"] = pd.to_datetime(runs["t"], errors="coerce").to_numpy() if "t" in runs else np.arange(len(runs))
    for c in ("distance_m", "grade", "trap", "sp_p_norm", "position", "adjusted_time"):
        Q[c] = col(c).to_numpy()
    won = col("won") if "won" in runs else (col("position") == 1).astype(float).where(col("position").notna())
    Q["won"] = won.to_numpy()
    Q["tk"] = _norm(col("track", False)).to_numpy()
    Q["sire_k"], Q["dam_k"] = _norm(col("sire", False)).to_numpy(), _norm(col("dam", False)).to_numpy()
    born = col("born", False)
    born = born.astype(object).where(born.notna(), "").astype(str)
    memo = {b: pd.to_datetime("15-" + b, format="%d-%b-%Y", errors="coerce") for b in pd.unique(born)}
    Q["born_dn"] = (pd.to_datetime(born.map(memo)) - EPOCH).dt.days.to_numpy()
    sex = col("sex", False)
    Q["bitch"] = np.where(sex.isna(), np.nan, (_norm(sex) == "b").astype(float))
    fam = col("grade_family", False)
    Q["fam"] = fam.astype(object).where(fam.notna(), "?").astype(str).to_numpy()
    cls = col("race_class", False)
    Q["race"] = ~trials_mod.is_trial_class(cls).fillna(False).to_numpy(bool)       # a T-class row is no race
    Q["one"] = 0
    Q["ev1"] = 1.0
    return Q


def _trials(runs: pd.DataFrame, ctx: dict | None) -> pd.DataFrame:
    """The timed trial rows: those handed over (``ctx["trial_runs"]``) and any T-class row of ``runs``."""
    parts = []
    in_runs = trials_mod.is_trial_class(runs.race_class).fillna(False) if "race_class" in runs else None
    if in_runs is not None and in_runs.any():
        parts.append(trials_mod.select(runs[in_runs]))
    extra = (ctx or {}).get("trial_runs")
    if extra is not None and len(extra):
        parts.append(trials_mod.select(extra))
    if not parts:
        return pd.DataFrame(columns=["dog_id", "day", "tk", "distance_m", "at", "ev1"])
    tr = pd.concat(parts, ignore_index=True).drop_duplicates(["race_id", "dog_id"], keep="last")
    tt = pd.DataFrame({"dog_id": tr.dog_id.to_numpy(), "day": _dn(tr.race_date).to_numpy(),
                       "tk": _norm(tr.track).to_numpy(),
                       "distance_m": pd.to_numeric(tr.distance_m, errors="coerce").astype(float).to_numpy(),
                       "at": pd.to_numeric(tr.adjusted_time, errors="coerce").astype(float).to_numpy(), "ev1": 1.0})
    tt = tt[tt.day.notna()].copy()
    tt["day"] = tt.day.astype("int64")
    return tt


def _ridge_map(Z: pd.DataFrame, y: np.ndarray, day: np.ndarray, month_start: np.ndarray, mapped: np.ndarray,
               fit_ok: np.ndarray) -> tuple[np.ndarray, dict]:
    """The price map: for each month, a ridge of ``y`` on the standardised ``Z`` (a missing value at the mean) fitted
    on the mapped rows dated before the month's first day, applied to the month's mapped rows."""
    pred = np.full(len(Z), np.nan)
    fits = {}
    idx = np.flatnonzero(mapped)
    if not len(idx):
        return pred, fits
    idx = idx[np.argsort(day[idx], kind="stable")]
    Zs, ys, ds, ms, ok = Z.to_numpy(float)[idx], y[idx], day[idx], month_start[idx], fit_ok[idx]
    for m in np.unique(ms):
        k = np.searchsorted(ds, m)                                   # rows dated strictly before the month's start
        trn = ok[:k]
        app = ms == m
        if trn.sum() < MIN_FIT or not app.any():
            continue
        A = Zs[:k][trn]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # a column with no value yet: left at 0
            mu, sd = np.nanmean(A, axis=0), np.nanstd(A, axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        A = np.nan_to_num((A - mu) / sd, nan=0.0)
        yt = ys[:k][trn]
        ym = yt.mean()
        b = np.linalg.solve(A.T @ A + RIDGE * np.eye(A.shape[1]), A.T @ (yt - ym))
        pred[idx[app]] = np.nan_to_num((Zs[app] - mu) / sd, nan=0.0) @ b + ym
        fits[str((EPOCH + pd.Timedelta(days=int(m))).to_period("M"))] = {"train_rows": int(trn.sum()),
                                                                        "applied": int(app.sum())}
    return pred, fits


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    cols = list(COLUMNS)
    out = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    if not len(runs):
        for c in cols:
            out[c] = np.float32(np.nan)
        return out, {"rows": 0}
    Q = _frame(runs)
    Q["day"] = Q.day.astype("int64")
    tt = _trials(runs, ctx)
    R = Q[Q.race]
    start = int(Q.day.min())                                         # the data's first day: fixed as days are added
    burn, born_obs = start + SEEN_AFTER, start - BORN_BEFORE

    # the outcomes of a row: read only through _prior (strictly earlier days), never for the row's own features
    lp = np.log(Q.sp_p_norm.where(Q.sp_p_norm > 0))
    Q["lsp"] = lp - lp.groupby(Q.race_id).transform("mean")           # race-centred log SP share
    Q["e"] = Q.sp_p_norm
    Q["w_"] = Q.won.where(Q.sp_p_norm.notna())
    rk = Q.position.groupby(Q.race_id).rank(ascending=False, method="average")
    npos = Q.position.notna().groupby(Q.race_id).transform("sum")
    Q["fld"] = (rk - 1) / (npos - 1).where(npos > 1) - 0.5           # share of the field beaten, centred

    # experience: the races before today, and whether the start was seen
    Q["n"] = _prior(R, Q, "dog_id", ["ev1"])["ev1"].to_numpy()
    first_seen = pd.concat([Q[["dog_id", "day"]], tt[["dog_id", "day"]]]).groupby("dog_id").day.min()
    Q["obs"] = (Q.dog_id.map(first_seen) >= burn) | (Q.born_dn >= born_obs)
    Q["w"] = 1.0 / (1.0 + Q.n + np.where(Q.obs, 0.0, CENSOR_ADD))
    Q["deb"] = Q.obs & (Q.n == 0)
    Q["early"] = Q.obs & (Q.n < 3)
    Q["mapped"] = Q.obs & (Q.n < MAP_RUNS)

    # the events: race rows with an SP
    E = Q[Q.lsp.notna() & Q.race].copy()
    E["d1"] = E.deb.astype(float)
    E["d_lsp"] = E.lsp.where(E.deb, 0.0)
    E["e1"] = E.early.astype(float)
    E["e_lsp"] = E.lsp.where(E.early, 0.0)
    E["e_w"] = E.w_.where(E.early, 0.0)
    E["e_e"] = E.e.where(E.early, 0.0)
    E["e_fn"] = (E.early & E.fld.notna()).astype(float)
    E["e_fld"] = E.fld.where(E.early).fillna(0.0)
    E["grade_n"] = (E.deb & E.grade.notna()).astype(float)
    E["grade_s"] = E.grade.where(E.deb).fillna(0.0)
    E["l1"] = 1.0
    E["l_s"] = E.lsp

    # the population (all GB) on earlier days: the cold-start value everything shrinks to
    G = _prior(E, Q, "one", EV)
    mu_d = (G.d_lsp / G.d1.where(G.d1 > 0)).fillna(0.0)               # mean lsp of debutants so far
    mu_e = (G.e_lsp / G.e1.where(G.e1 > 0)).fillna(0.0)               # ... of dogs in their first three runs
    # their winners over expected; 1 until the first early winner (a rate of 0 has no log: the cold start)
    ae_e = (G.e_w / G.e_e.where(G.e_e > 0)).where(G.e_w > 0).fillna(1.0)
    own = _prior(E, Q, "dog_id", EV)                                  # the dog's own early runs: taken out

    f = pd.DataFrame(index=Q.index)
    f[f"{PREFIX}w"] = Q.w
    ped = {}
    for who, kl, ka, kf in (("sire", K_SIRE_LSP, K_SIRE_AE, K_SIRE_FLD), ("dam", K_DAM_LSP, K_DAM_AE, K_DAM_FLD)):
        key = f"{who}_k"
        P = _prior(E[E[key] != ""], Q, key, EV) - own
        known = (Q[key] != "").to_numpy()
        if who == "sire":        # its first runs against the market, as a deviation from the debutants' mean
            v1 = (P.d_lsp + kl * mu_d) / (P.d1 + kl) - mu_d
        else:                    # a dam has few first runs: her progeny's first three runs
            v1 = (P.e_lsp + kl * mu_e) / (P.e1 + kl) - mu_e
        v2 = np.log((P.e_w + ka * ae_e) / (P.e_e + ka))
        v3 = P.e_fld / (P.e_fn + kf)
        for nm, v in (("first_lsp" if who == "sire" else "early_lsp", v1), ("early_ae", v2), ("early_fld", v3)):
            v = pd.Series(np.where(known, v, np.nan), index=Q.index)
            ped[f"{who}_{nm}"] = v
            f[f"{PREFIX}{who}_{nm}_w"] = v * Q.w

    # the age at the first race (today's, for a debutant); a row before the dog's first race reads none
    first_race = Q.dog_id.map(R.groupby("dog_id").day.min())
    f[f"{PREFIX}age_debut"] = (first_race - Q.born_dn).where(Q.obs & (first_race <= Q.day))
    f[f"{PREFIX}bitch_w"] = Q.bitch * Q.w

    # the track's debutant record and the seeding
    Q["cell"] = Q.tk + "|" + Q.fam
    E["cell"] = E.tk + "|" + E.fam
    Tk = _prior(E, Q, "tk", EV + ["grade_n", "grade_s"])
    Ce = _prior(E, Q, "cell", EV + ["grade_n", "grade_s"])
    tk_lsp = (Tk.d_lsp + K_TRACK * mu_d) / (Tk.d1 + K_TRACK)
    cell_lsp = (Ce.d_lsp + K_CELL * tk_lsp) / (Ce.d1 + K_CELL)
    trk_ae = np.log((Tk.e_w + K_TRACK_AE * ae_e) / (Tk.e_e + K_TRACK_AE))    # an input of the price map only
    f[f"{PREFIX}trk_deb_lsp_w"] = cell_lsp * Q.w
    Q["trapk"] = Q.trap.fillna(-1).astype(int)
    E["trapk"] = E.trap.fillna(-1).astype(int)
    Tp = _prior(E, Q, "trapk", EV)
    trap_lsp = (Tp.d_lsp + K_TRAP * mu_d) / (Tp.d1 + K_TRAP) - mu_d
    f[f"{PREFIX}trap_deb_lsp_w"] = trap_lsp * Q.w
    seed = Ce.grade_s / Ce.grade_n.where(Ce.grade_n >= MIN_DEBUTS) - Q.grade
    f[f"{PREFIX}grade_seed_w"] = seed * Q.w
    f[f"{PREFIX}field_inexp"] = Q.w.groupby(Q.race_id).transform("sum") - Q.w

    # first time at a track, a trip, a grade family
    dog = Q.dog_id.astype(str)
    dist = Q.distance_m.astype(str)
    Q["dt"], Q["dd"], Q["df"] = dog + "|" + Q.tk, dog + "|" + dist, dog + "|" + Q.fam
    R = Q[Q.race]
    seen_ok = (Q.obs | (Q.day >= burn)) & (Q.n >= 1)                  # not a debut; the data old enough to tell
    ft_track = seen_ok & (_prior(R, Q, "dt", ["ev1"])["ev1"] == 0)
    ft_trip = seen_ok & Q.distance_m.notna() & (_prior(R, Q, "dd", ["ev1"])["ev1"] == 0)
    ft_fam = seen_ok & (Q.fam != "?") & (_prior(R, Q, "df", ["ev1"])["ev1"] == 0)
    ol = _prior(E, Q, "dog_id", ["l1", "l_s"])
    own_lsp = ol.l_s / ol.l1.where(ol.l1 > 0)                         # the dog's mean lsp on earlier days
    Q["resid"] = Q.lsp - own_lsp                                      # how the market re-rated it on the day
    order = Q.sort_values(["dog_id", "day", "tie"], kind="stable")
    last_d = order.groupby("dog_id").distance_m.shift(1).reindex(Q.index)    # the trip of the run before (card fact)
    longer = np.where(Q.distance_m > last_d, "up", np.where(Q.distance_m < last_d, "dn", "eq"))
    band = pd.cut(Q.distance_m, [0, 399, 549, 10_000], labels=["sprint", "standard", "stay"]).astype(str)
    ft_n = {}
    for nm, flag, cell in (("track", ft_track, Q.tk), ("trip", ft_trip, pd.Series(longer, index=Q.index) + "|" + band),
                           ("fam", ft_fam, Q.fam)):
        Q["_c"], Q["_f"] = cell.astype(str), flag.to_numpy()
        Ev = Q[Q._f & Q.resid.notna() & Q.race].assign(r1=1.0, r_s=lambda d: d.resid)
        C, A = _prior(Ev, Q, "_c", ["r1", "r_s"]), _prior(Ev, Q, "one", ["r1", "r_s"])
        mu = (A.r_s / A.r1.where(A.r1 > 0)).fillna(0.0)
        f[f"{PREFIX}ft_{nm}"] = np.where(Q._f, (C.r_s + K_FT * mu) / (C.r1 + K_FT), 0.0)
        ft_n[nm] = int(flag.sum())

    # the time gap: own best calculated time at today's track and trip (trials and races on earlier days) against the
    # other runners' mean calculated time there in earlier races, in lengths
    Q["kt"] = dog + "|" + Q.tk + "|" + dist
    tt = tt[tt["at"] > 0].copy()
    tt["kt"] = tt.dog_id.astype(str) + "|" + tt.tk + "|" + tt.distance_m.astype(str)
    rr = Q[Q.race & (Q.adjusted_time > 0)].assign(at=lambda d: d.adjusted_time, a1=1.0)
    allt = pd.concat([tt[["kt", "day", "at"]], rr[["kt", "day", "at"]]], ignore_index=True)
    best = _prior(allt, Q, "kt", ["at"], how="min")["at"]
    rs = _prior(rr, Q, "kt", ["at", "a1"])
    mean_t = rs["at"] / rs.a1.where(rs.a1 > 0)
    cnt = mean_t.notna().groupby(Q.race_id).transform("sum") - mean_t.notna()
    others = (mean_t.groupby(Q.race_id).transform("sum") - mean_t.fillna(0.0)) / cnt.where(cnt >= 2)
    gap = ((others - best) / core.LENGTH_S).clip(-GAP_CLIP, GAP_CLIP)
    n_tr = _prior(tt, Q, "dog_id", ["ev1"])["ev1"]

    # the debut price map, refitted each month on earlier months only
    own_n = own_lsp.where(Q.n > 0)
    Z = pd.DataFrame({"n1": (Q.n == 1).astype(float), "n2": (Q.n == 2).astype(float),
                      "age": (Q.day - Q.born_dn) / 365.0, "bitch": Q.bitch, "gap": gap,
                      "gap_na": gap.isna().astype(float), "ntr": np.log1p(n_tr), "sire_lsp": ped["sire_first_lsp"],
                      "sire_ae": ped["sire_early_ae"], "sire_fld": ped["sire_early_fld"],
                      "dam_lsp": ped["dam_early_lsp"], "dam_fld": ped["dam_early_fld"], "cell": cell_lsp,
                      "trk_ae": trk_ae, "trapl": trap_lsp, "seed": seed, "inexp": f[f"{PREFIX}field_inexp"],
                      "own": own_n, "own_na": (own_lsp.isna() | (Q.n == 0)).astype(float)})
    for t_ in range(1, 7):
        Z[f"trap{t_}"] = (Q.trap == t_).astype(float)
    Z["gap_n0"] = Z.gap * (Q.n == 0)
    Z["own_n1"] = Z.own * (Q.n == 1)
    Z["n39"] = (Q.n >= 3).astype(float)
    Z["ln"] = np.log1p(Q.n)
    Z["gap_n39"] = Z.gap * (Q.n >= 3)
    Z["own_n39"] = Z.own * (Q.n >= 3)
    Z["own_w"] = Z.own * Q.w
    month_start = _dn((EPOCH + pd.to_timedelta(Q.day, unit="D")).dt.to_period("M").dt.to_timestamp()).to_numpy()
    pred, fits = _ridge_map(Z[list(MAP_INPUTS)], Q.lsp.to_numpy(float), Q.day.to_numpy(), month_start,
                            Q.mapped.to_numpy(bool), (Q.lsp.notna() & Q.race).to_numpy(bool))
    pred = pd.Series(pred, index=Q.index)
    f[f"{PREFIX}prior_lsp"] = pred                                    # none beyond the tenth run or an unseen start
    f[f"{PREFIX}prior_gap"] = (pred - own_n).where(Q.n > 0)
    f[f"{PREFIX}trial_gap_w"] = gap.where(Q.early) * Q.w
    f[f"{PREFIX}tt_gap"] = gap

    for c in cols:
        out[c] = f[c].to_numpy(float).astype(np.float32)
    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(len(runs), bool)
    notes = {"rows": int(len(out)), "first_day": str((EPOCH + pd.Timedelta(days=start)).date()),
             "starts_seen_from": str((EPOCH + pd.Timedelta(days=burn)).date()),
             "trial_rows": int(len(tt)), "debut_rows": int(Q.deb.sum()), "early_rows": int(Q.early.sum()),
             "unseen_start_rows": int((~Q.obs).sum()), "debut_events_with_sp": int(E.deb.sum()),
             "early_events_with_sp": int(E.early.sum()), "first_time_rows": ft_n, "ridge_fits": len(fits),
             "ridge_first_fit": min(fits) if fits else None,
             "ridge_last_fit": {max(fits): fits[max(fits)]} if fits else None,
             "card_rows": int(card.sum()),
             "null_share": {c: round(float(out[c].isna().mean()), 4) for c in cols}}
    return out, notes
