"""The last race: what the dog did last time, how it was bet, what the market usually does next, what changes today.

Screened on 12 Nov 2026 (wave 4) against what the stack with the trials and optics tables still misses (the miss of
the stacked forecast of the Betfair SP on 110,788 priced runners, 1 Jun to 10 Nov 2026). The printed card's last
line is read by every table already (the position, the time, the comment); this table reads the last RACE as an
event with a market: how fancied the opposition was, whether the dog outran its market rank, what the SP usually
does after that (finish, market rank) cell, and what is different today.

**The last run** of a runner is its run on its last EARLIER day (the last run of that day; a run earlier the same
day is never read). A card row (``is_card``: no result yet) is never a last run and gets its features from history.
The field of a race is every row of ``runs`` with its ``race_id``.

- How it ran: ``lr_pos_gain`` (rank at the first sectional less the finish, only where every runner has a
  sectional), ``lr_fin_speed`` (the run home, ``run_time`` less ``sectional``, against the race's mean, lengths at
  0.08 s, clipped at 15; at least 3 runners timed), ``lr_gain_x_trip`` (``lr_pos_gain`` x ``lr_dist_chg``).
- How it was bet. The SP share is lsp = log(``sp_p_norm`` x field): 0 an even chance in any field size.
  ``lr_mkt_beat`` (market rank less finish), ``lr_ae_last`` (won less its SP chance), ``lr_sp_vs_prev`` (lsp of the
  last run less the mean lsp of the three runs before it), ``lr_react_resid`` (the move in lsp from the run before
  last to the last run, less the usual move after such a run before last: the map below).
- The market maps, by the last run's (finish, market rank) cell, both clipped at 6: ``lr_exp_react`` (the usual
  move in lsp at the next run), ``lr_exp_ae`` (the usual won less SP chance at the next run); by (comment category,
  finish bucket): ``lr_exp_ae_cmt``.
- The run before last against the last: ``lr_pos_contrast``, ``lr_time_contrast``.
- Today against the last run: ``lr_trap_chg``, ``lr_grade_chg`` (same track and grade family only),
  ``lr_dist_chg`` (log ratio), ``lr_track_chg``, ``lr_field_chg``.
- The race itself: ``lr_race_speed`` (its winner against the course standard, lengths), ``lr_opp_fancied`` (the
  mean, over the other runners, of each one's usual lsp before that race).

**Everything fitted is refitted from earlier days only.**

- The course standard of a race on day d: the winners' calculated times on days before d (a time outside ``PACE``
  seconds a metre is a keying error and is left out), std = (n_g x m_g + ``K_STD`` x m_c) / (n_g + ``K_STD``), m_c
  the track-and-distance mean (none under ``MIN_STD`` winners), m_g and n_g the mean and count of the race's class
  there; a class with no winner yet starts at the course mean.
- The maps for a runner in calendar month M are fitted on pairs (a run -> the dog's next run) whose NEXT run is
  dated before the first of M: cell = (n x cell mean + ``K_CELL`` x the finish position's mean) / (n + ``K_CELL``);
  the comment map = (n x cell mean + ``K_CMT`` x the mean of all pairs) / (n + ``K_CMT``). Under ``MIN_PAIRS``
  pairs there is no map (the first weeks of a store). So a day's results never reach that day's rows, nor any row of
  its month, and later days change nothing (tests/test_greyhound_lastrace.py).
- A runner's usual lsp before a race: the mean over its earlier days with an SP (at least ``MIN_PRIOR``); the
  opposition needs at least 2 such runners.

No imputation: no earlier run -> every column NaN; no run before last -> the contrast and reaction columns NaN; a
track without sectionals -> the shape columns NaN.

Against the screen's script (disc2/lastrace/build.py): the same definitions; the map sums are kept by month and
accumulated (the same numbers, one pass); a usual lsp is over the days WITH an SP (the script divided by every
earlier day; 48 runs of 350,818 have no SP); an unseen comment cell takes the mean of all pairs (the script left it
NaN); ``MIN_PAIRS`` is new.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "lr_"
SOURCE = "GBGB results (positions, sectionals, times, comments, SPs) and the card (track, trip, class, trap, field)"
KEY = ["race_id", "dog_id"]

LEN = core.LENGTH_S
PACE = (0.050, 0.080)        # seconds a metre a greyhound's time lies in (the optics table's bounds); fixed, not fitted
K_STD, MIN_STD = 20.0, 20    # the class standard shrunk to the course's; the course needs 20 earlier winners
K_CELL = 50.0                # a (finish, market rank) cell shrunk to its finish position's mean
K_CMT = 200.0                # a (comment category, finish bucket) cell shrunk to the mean of all pairs
MIN_PRIOR = 3                # earlier days with an SP for a "usual SP share"
MIN_PAIRS = 1000             # pairs before a month for that month to have maps
N_CAT, N_PB = 6, 3

TROUBLE = re.compile(r"crd|crowd|bmp|bump|blk|baulk|chk|check|fcdt|imp|stumb|hmp")
SLOW = re.compile(r"saw|slowaw|slaw|msdbrk|missedbr|mbrk|stbst|slow")
LED = re.compile(r"led|\bld\d|aled|\bald\b|ldto|ld-")
RANON = re.compile(r"ranon|rnon|finwell|fin well|ranwell|styd|stayed")
CLEAR = re.compile(r"clearrun|clrrun|clear run|evch|everych")

COLUMNS = {
    f"{PREFIX}pos_gain": "last run: its rank at the first sectional less its finishing position (above 0 it gained "
                         "places; only where every runner has a sectional)",
    f"{PREFIX}fin_speed": "last run: its run home (time less sectional) against the race's mean, lengths, above 0 "
                          "faster home",
    f"{PREFIX}gain_x_trip": "lr_pos_gain x lr_dist_chg (a dog that gained places, going up in trip today)",
    f"{PREFIX}mkt_beat": "last run: its SP rank less its finishing position (above 0 it outran its market rank)",
    f"{PREFIX}ae_last": "last run: won (1/0) less its SP chance",
    f"{PREFIX}sp_vs_prev": "last run: its log SP share less the mean of the three runs before it (above 0 it was "
                           "bet shorter than its usual)",
    f"{PREFIX}react_resid": "the move in log SP share from the run before last to the last run, less the usual move "
                            "after the run before last's (finish, market rank) cell",
    f"{PREFIX}exp_react": "the usual move in log SP share at the next run after the last run's (finish, market rank) "
                          "cell, fitted on pairs dated before this month",
    f"{PREFIX}exp_ae": "the usual won less SP chance at the next run after that cell, fitted on pairs dated before "
                       "this month",
    f"{PREFIX}exp_ae_cmt": "the usual won less SP chance at the next run after the last run's (comment category, "
                           "finish bucket) cell, fitted on pairs dated before this month",
    f"{PREFIX}pos_contrast": "the finishing position of the run before last less the last run's (above 0 it "
                             "finished nearer the front last time)",
    f"{PREFIX}time_contrast": "the last run's calculated time against the course standard less the run before "
                              "last's, lengths (above 0 it ran faster last time)",
    f"{PREFIX}trap_chg": "today's trap less the last run's trap",
    f"{PREFIX}grade_chg": "today's grade number less the last run's, same track and grade family only (above 0 a "
                          "lower grade today)",
    f"{PREFIX}dist_chg": "log(today's distance / the last run's distance)",
    f"{PREFIX}track_chg": "1 when today's track is not the last run's track",
    f"{PREFIX}field_chg": "today's field size less the last run's",
    f"{PREFIX}race_speed": "the last race's winning calculated time against the course standard before that day, "
                           "lengths, above 0 a fast race",
    f"{PREFIX}opp_fancied": "the last race's other runners: the mean of each one's usual log SP share before that "
                            "race (above 0 a field of dogs usually fancied)",
}


def category(comment: pd.Series) -> np.ndarray:
    """One code a comment (the first that applies): 0 trouble, 1 slow away, 2 led, 3 ran on, 4 clear run, 5 other."""
    c = comment.fillna("").astype(str).str.lower()
    uniq = pd.Series(pd.unique(c))
    code = np.full(len(uniq), 5)
    for k, rx in ((4, CLEAR), (3, RANON), (2, LED), (1, SLOW), (0, TROUBLE)):      # the later, the higher priority
        code[uniq.str.contains(rx).to_numpy()] = k
    return c.map(dict(zip(uniq, code))).to_numpy()


def standards(h: pd.DataFrame) -> pd.Series:
    """The course standard of each race of ``h`` (results only), from winners on EARLIER days: by race_id."""
    w = h[(h.position == 1) & h.atime.notna()].drop_duplicates("race_id")
    races = h.drop_duplicates("race_id")[["race_id", "d", "track", "distance_m", "race_class"]].copy()
    races["track"] = races.track.astype(object).where(races.track.notna(), "?").astype(str)
    races["race_class"] = races.race_class.astype(object).where(races.race_class.notna(), "?").astype(str)
    races["distance_m"] = races.distance_m.fillna(-1.0)
    races = races.merge(w[["race_id", "atime"]], on="race_id", how="left")
    races["n"] = races.atime.notna().astype(float)
    races["s"] = races.atime.fillna(0.0)
    out = {}
    for keys, sfx in ((["track", "distance_m"], "c"), (["track", "distance_m", "race_class"], "g")):
        a = races.groupby(keys + ["d"], as_index=False)[["s", "n"]].sum().sort_values(keys + ["d"])
        g = a.groupby(keys)
        a["S" + sfx] = g.s.cumsum() - a.s                       # the days strictly before
        a["N" + sfx] = g.n.cumsum() - a.n
        j = races[keys + ["d"]].merge(a[keys + ["d", "S" + sfx, "N" + sfx]], on=keys + ["d"], how="left")
        out["S" + sfx], out["N" + sfx] = j["S" + sfx].to_numpy(), j["N" + sfx].to_numpy()
    nc, ng = out["Nc"], out["Ng"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mc = np.where(nc >= MIN_STD, out["Sc"] / nc, np.nan)
        mg = np.where(ng > 0, out["Sg"] / ng, mc)
        std = (ng * mg + K_STD * mc) / (ng + K_STD)
    return pd.Series(std, index=races.race_id.to_numpy())


def _lookup(arr: np.ndarray, *ix) -> np.ndarray:
    """``arr[ix]`` a row, NaN where any index is missing (indices are floats, NaN for none)."""
    ok = np.ones(len(ix[0]), bool)
    for i in ix:
        ok &= ~np.isnan(i)
    out = np.full(len(ix[0]), np.nan)
    out[ok] = arr[tuple(i[ok].astype(int) for i in ix)]
    return out


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    n = len(runs)
    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    cols = ["race_id", "dog_id", "t", "track", "distance_m", "race_class", "grade", "grade_family", "trap", "field",
            "position", "market_pos", "sp_p_norm", "won", "sectional", "run_time", "adjusted_time", "comment"]
    df = pd.DataFrame({c: (runs[c].to_numpy() if c in runs else np.nan) for c in cols})
    for c in ("distance_m", "grade", "trap", "field", "position", "market_pos", "sp_p_norm", "won", "sectional",
              "run_time", "adjusted_time"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["d"] = core.day(runs).to_numpy()
    df["_i"] = np.arange(n)

    # ---- the results (never a card row): each run's own quantities, read below only from a LATER day ----
    h = df[~card].copy()
    g = h.groupby("race_id")
    nrow = g.dog_id.transform("size")
    h["erank"] = g.sectional.rank(method="min").where(g.sectional.transform("count") == nrow)
    h["pos_gain"] = h.erank - h.position
    ok_t = (h.run_time / h.distance_m).between(*PACE) & h.sectional.notna() & (h.sectional < h.run_time)
    h["home"] = (h.run_time - h.sectional).where(ok_t)
    gh = h.groupby("race_id").home
    h["fin_speed"] = ((gh.transform("mean") - h.home) / LEN).where(gh.transform("count") >= 3)
    h["mkt_beat"] = h.market_pos - h.position
    h["ae"] = h.won - h.sp_p_norm
    h["lsp"] = np.log((h.sp_p_norm * nrow).where(h.sp_p_norm > 0))
    h["cat"] = category(h.comment).astype(float)
    h["atime"] = h.adjusted_time.where((h.adjusted_time / h.distance_m).between(*PACE))
    std = standards(h)
    h["std"] = h.race_id.map(std)
    win = h[(h.position == 1) & h.atime.notna()].drop_duplicates("race_id").set_index("race_id").atime
    h["race_speed"] = (h["std"] - h.race_id.map(win)) / LEN
    h["fig"] = (h["std"] - h.atime) / LEN

    # ---- dog-days: the day's last run; every lag is over dog-days, so over strictly earlier days ----
    h = h.sort_values(["dog_id", "t"], kind="stable")
    P = h.drop_duplicates(["dog_id", "d"], keep="last").copy()
    gd = P.groupby("dog_id")
    has = P.lsp.notna().astype(float)
    cnt = has.groupby(P.dog_id).cumsum() - has                         # earlier days with an SP
    cs = P.lsp.fillna(0.0).groupby(P.dog_id).cumsum() - P.lsp.fillna(0.0)
    P["lsp_prior_mean"] = cs / cnt.where(cnt >= MIN_PRIOR)
    for c in ("lsp", "position", "fig"):
        P[c + "_p1"] = gd[c].shift(1)
    P["sp_vs_prev"] = P.lsp - pd.concat([gd.lsp.shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1)
    P["dlsp"] = P.lsp - P.lsp_p1
    P["pos_contrast"] = P.position_p1 - P.position
    P["time_contrast"] = P.fig - P.fig_p1
    # the opposition of this run: the other runners' usual SP share before it
    h["lsp_prior_mean"] = P.lsp_prior_mean.reindex(h.index)            # NaN on a run that is not its day's last
    gg = h.groupby("race_id").lsp_prior_mean
    tot, k = gg.transform("sum"), gg.transform("count")
    others = k - h.lsp_prior_mean.notna()
    P["opp_fancied"] = ((tot - h.lsp_prior_mean.fillna(0.0)) / others.where(others >= 2)).reindex(P.index)
    # the cells of a run
    P["cp"] = P.position.clip(1, 6)
    P["cm"] = P.market_pos.clip(1, 6)
    P["pb"] = pd.Series(np.select([P.position == 1, P.position <= 3], [0.0, 1.0], 2.0), index=P.index).where(P.position.notna())
    for c in ("cp", "cm", "pb", "cat"):
        P[c + "_p1"] = gd[c].shift(1)

    # ---- the maps: sums of the pairs (run j -> next run i) by i's month, accumulated over the months BEFORE ----
    mon = lambda d: (d.dt.year * 12 + d.dt.month).to_numpy()            # noqa: E731
    m0 = int(mon(df.d).min()) if n else 0
    nm = (int(mon(df.d).max()) - m0 + 1) if n else 1
    pr = P.dropna(subset=["cp_p1", "cm_p1", "dlsp", "ae"])
    pi = mon(pr.d) - m0
    a, b = pr.cp_p1.to_numpy(int) - 1, pr.cm_p1.to_numpy(int) - 1
    ct, pb = pr.cat_p1.to_numpy(int), pr.pb_p1.to_numpy(int)

    def before(shape, ix, v):
        """Sums of ``v`` by (month, cell), accumulated over the months strictly before each month."""
        s = np.zeros((nm + 1,) + shape)
        np.add.at(s, (pi + 1,) + ix, v)
        return np.cumsum(s, axis=0)[:nm]

    one = np.ones(len(pr))
    N = before((6, 6), (a, b), one)
    Sr = before((6, 6), (a, b), pr.dlsp.to_numpy())
    Sa = before((6, 6), (a, b), pr.ae.to_numpy())
    Nc = before((N_CAT, N_PB), (ct, pb), one)
    Sc = before((N_CAT, N_PB), (ct, pb), pr.ae.to_numpy())
    total = N.sum(axis=(1, 2))
    fitted = total >= MIN_PAIRS
    with np.errstate(divide="ignore", invalid="ignore"):
        npos = N.sum(axis=2)
        pm_r = np.where(fitted[:, None], Sr.sum(axis=2) / npos, np.nan)          # a finish position's mean
        pm_a = np.where(fitted[:, None], Sa.sum(axis=2) / npos, np.nan)
        map_r = (Sr + K_CELL * pm_r[:, :, None]) / (N + K_CELL)
        map_a = (Sa + K_CELL * pm_a[:, :, None]) / (N + K_CELL)
        mu = np.where(fitted, Sa.sum(axis=(1, 2)) / total, np.nan)
        map_c = (Sc + K_CMT * mu[:, None, None]) / (Nc + K_CMT)

    # ---- every row of runs: its dog's last run on an EARLIER day ----
    keep = ["track", "distance_m", "grade", "grade_family", "trap", "field", "pos_gain", "fin_speed", "mkt_beat",
            "ae", "sp_vs_prev", "dlsp", "pos_contrast", "time_contrast", "race_speed", "opp_fancied", "cp", "cm",
            "pb", "cat", "cp_p1", "cm_p1"]
    L = P[["dog_id", "d"] + keep].rename(columns={c: "L_" + c for c in keep}).sort_values("d", kind="stable")
    L["L_d"] = L.d
    t = df[["_i", "dog_id", "d", "track", "distance_m", "grade", "grade_family", "trap", "field"]]
    t = t.sort_values("d", kind="stable")
    m = pd.merge_asof(t, L, on="d", by="dog_id", allow_exact_matches=False).sort_values("_i").reset_index(drop=True)
    assert len(m) == n and (m.L_d.dropna() < m.d[m.L_d.notna()]).all()

    mi = (mon(m.d) - m0).astype(float)
    cp, cm = m.L_cp.to_numpy() - 1, m.L_cm.to_numpy() - 1
    cp1, cm1 = m.L_cp_p1.to_numpy() - 1, m.L_cm_p1.to_numpy() - 1

    def cell(mp, pmean, p, q):
        v = _lookup(mp, mi, p, q)
        return np.where(np.isnan(v), _lookup(pmean, mi, p), v)          # no market rank: the finish position's mean

    exp_react = cell(map_r, pm_r, cp, cm)
    exp_ae = cell(map_a, pm_a, cp, cm)
    exp_react_p = cell(map_r, pm_r, cp1, cm1)
    exp_ae_cmt = _lookup(map_c, mi, m.L_cat.to_numpy(), m.L_pb.to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.log((m.distance_m / m.L_distance_m).where((m.distance_m > 0) & (m.L_distance_m > 0)))
    seen = m.L_d.notna()
    same = (m.track == m.L_track) & (m.grade_family == m.L_grade_family)
    o = {
        "pos_gain": m.L_pos_gain,
        "fin_speed": m.L_fin_speed.clip(-15, 15),
        "gain_x_trip": m.L_pos_gain * dlog,
        "mkt_beat": m.L_mkt_beat,
        "ae_last": m.L_ae,
        "sp_vs_prev": m.L_sp_vs_prev,
        "react_resid": m.L_dlsp - exp_react_p,
        "exp_react": exp_react,
        "exp_ae": exp_ae,
        "exp_ae_cmt": exp_ae_cmt,
        "pos_contrast": m.L_pos_contrast,
        "time_contrast": m.L_time_contrast.clip(-25, 25),
        "trap_chg": m.trap - m.L_trap,
        "grade_chg": (m.grade - m.L_grade).where(same),
        "dist_chg": dlog,
        "track_chg": (m.track != m.L_track).astype(float).where(seen),
        "field_chg": m.field - m.L_field,
        "race_speed": m.L_race_speed.clip(-25, 25),
        "opp_fancied": m.L_opp_fancied,
    }
    out = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    for c, v in o.items():
        out[PREFIX + c] = np.asarray(v, dtype=float).astype(np.float32)
    assert list(out.columns) == KEY + list(COLUMNS)

    def share(x) -> float | None:
        return round(float(np.mean(x)), 4) if len(x) else None

    months = np.flatnonzero(fitted)
    notes = {"rows": int(n), "card_rows": int(card.sum()), "dog_days": int(len(P)), "pairs": int(len(pr)),
             "rows_with_a_last_run": share(seen.to_numpy()),
             "card_rows_with_a_last_run": share(seen.to_numpy()[card]),
             "rows_with_a_map": share(~np.isnan(exp_react)),
             "rows_with_sectional_shape": share(m.L_pos_gain.notna().to_numpy()),
             "rows_with_a_race_speed": share(m.L_race_speed.notna().to_numpy()),
             "months": int(nm), "months_with_a_map": int(fitted.sum()),
             "first_month_with_a_map": (f"{(m0 + months[0] - 1) // 12}-{(m0 + months[0] - 1) % 12 + 1:02d}"
                                        if len(months) else None),
             "pairs_behind_the_last_month": int(total[-1]),
             "refit": "the maps each calendar month on pairs whose next run is dated before the month; the course "
                      "standard a race from winners on earlier days"}
    return out, notes
