"""Trials on the race scale (the feature review of 17 Nov 2026, "trials" row and B2; a research variant of trials.py).

trials.py reads a trial against the graded standard as if it were a race. A trial is not a race: many are run alone,
in a session with no race meeting's going, and its margins in company are never read. This table:

- **(a) the trial-to-race offset**: from dogs that trial and then race within ``PAIR_DAYS`` days, the race figure (the
  engine's gsr: standard, going and class par taken out) less the trial figure (the same standard, no going), by the
  trial's track and trip and solo or in company, shrunk to the solo/company mean by ``K_OFF`` pairs. Walk-forward: a
  trial reads the pairs whose race was on a day before the trial's day.
- **(b) the session's going**: a trial session (track and day) has no meeting variant. Its trials run by dogs with an
  established race level (``MIN_RACES`` races, decayed level before the trial's day) say how fast it ran: their
  offset-corrected figure less their level, averaged leaving the dog itself out, shrunk by k = within / between
  variance (fitted on the first fifth of the trial days), and taken off.
- **(c) the trial in company**: the finishing position, the field, the margin (lengths behind the winner by the clock;
  the winner's is minus its winning distance), and a going-free figure from the rivals: each rival with an
  established level says the dog's level is that level plus the rival's margin less the dog's.
- **(d) levels**: the dog's trial level on the race scale (decayed by ``HALF_LIFE``), its race level, the gap, days
  since the last trial and the trials since the last race.

Every column of a race reads trials and races of days before the race's day only; the card row gets the values the
result row gets (all columns card-safe, given ``ctx["trial_runs"]`` holds the trials to the day before).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core
from greyhound.pit import trials as tl

PREFIX = "tq_"
SOURCE = ("GBGB results: the trial rows (classes T, T1-T4) data.clean drops (ctx['trial_runs']) and the race rows "
          "through the metrics engine's figures (greyhound/metrics.py)")
DEPENDS = ("trials",)
PAIR_DAYS = 21
K_OFF = 20.0
MIN_RACES = 3
HALF_LIFE = 60.0
OUTLIER = 40.0
PAIR_CLIP = 20.0
MARGIN_CAP = 15.0
FIT_SHARE = 0.2
MIN_SESSIONS = 20
BIG = tl.BIG
BAD = r"\(Handslip\)|Fell|Fll|BrkLine"
FIGS = ("gsr", "esr", "fsr")


def _dn(dates) -> np.ndarray:
    return tl._dn(pd.Series(dates)).astype(float)


def trial_rows(df: pd.DataFrame, ctx: dict | None) -> pd.DataFrame:
    """The trial rows: T-class rows of ``df`` and ``ctx["trial_runs"]`` (trials.select), one a trial and dog."""
    in_runs = tl.is_trial_class(df.race_class)
    tr = [tl.select(df[in_runs])] if in_runs.any() else []
    extra = (ctx or {}).get("trial_runs")
    if extra is not None and len(extra):
        tr.append(tl.select(extra))
    if not tr:
        return pd.DataFrame(columns=tl.KEEP)
    return pd.concat(tr, ignore_index=True).drop_duplicates(["race_id", "dog_id"], keep="last").reset_index(drop=True)


def raw_figures(tr: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Each trial's gsr, esr and fsr against the engine's standards (graded winners' calculated time, fair sectionals
    and run-ins at the track and trip, the mean over every day before the trial's), no going: lengths, higher faster."""
    L = core.LENGTH_S
    bad = races.comment.fillna("").str.contains(BAD, case=False)
    trial = races.is_trial.fillna(False).astype(bool) if "is_trial" in races else pd.Series(False, index=races.index)
    fair = ~bad & ~trial
    adj = pd.to_numeric(races.adjusted_time, errors="coerce")
    sec = pd.to_numeric(races.sectional, errors="coerce")
    won = pd.to_numeric(races.won, errors="coerce") == 1
    vals = {"t": adj.where(won & fair), "s": sec.where(fair), "r": (adj - sec).where(fair)}
    st = pd.DataFrame({"key": tl._key(races).to_numpy(), "dn": _dn(races.race_date)})
    for k, v in vals.items():
        st[f"s_{k}"] = v.fillna(0.0).to_numpy()
        st[f"n_{k}"] = v.notna().to_numpy(float)
    st = st.dropna(subset=["dn"]).groupby(["key", "dn"], as_index=False).sum().sort_values(["key", "dn"])
    cs = [c for c in st.columns if c[:2] in ("s_", "n_")]
    st[cs] = st.groupby("key")[cs].cumsum()
    st = st.sort_values("dn")
    q = pd.DataFrame({"key": tl._key(tr).to_numpy(), "dn": _dn(tr.race_date), "_i": np.arange(len(tr))})
    q = q.dropna(subset=["dn"]).sort_values("dn")
    m = pd.merge_asof(q, st, on="dn", by="key", allow_exact_matches=False, direction="backward").sort_values("_i")
    std = {k: np.full(len(tr), np.nan) for k in vals}
    for k in vals:
        with np.errstate(invalid="ignore", divide="ignore"):
            std[k][m._i.to_numpy()] = (m[f"s_{k}"] / m[f"n_{k}"].where(m[f"n_{k}"] > 0)).to_numpy()
    tadj = pd.to_numeric(tr.adjusted_time, errors="coerce").to_numpy(float)
    tsec = pd.to_numeric(tr.sectional, errors="coerce").to_numpy(float)
    tbad = (tr.comment.fillna("").str.contains(BAD, case=False).to_numpy() if "comment" in tr
            else np.zeros(len(tr), bool))
    out = pd.DataFrame(index=tr.index)
    for name, v in (("gsr", (std["t"] - tadj) / L), ("esr", (std["s"] - tsec) / L),
                    ("fsr", (std["r"] - (tadj - tsec)) / L)):
        out[name] = np.where(~tbad & (np.abs(v) <= OUTLIER), v, np.nan)
    return out


def level_at(r_dog: np.ndarray, r_dn: np.ndarray, r_v: np.ndarray, q_dog: np.ndarray, q_dn: np.ndarray,
             half_life: float = HALF_LIFE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each query's (dog, day) decayed mean of the dog's values on days strictly before (one weight a day, the day's
    mean), the decayed weight and the count of days. Integer dog codes; NaN level with no earlier day."""
    lam = np.log(2.0) / half_life
    ok = np.isfinite(r_v) & np.isfinite(r_dn)
    d = pd.DataFrame({"dc": r_dog[ok].astype(np.int64), "dn": r_dn[ok].astype(np.int64), "v": r_v[ok]})
    per = d.groupby(["dc", "dn"], as_index=False).v.mean().sort_values(["dc", "dn"])
    nq = len(q_dog)
    if not len(per):
        return np.full(nq, np.nan), np.zeros(nq), np.zeros(nq)
    d0 = per.groupby("dc").dn.transform("min").to_numpy(np.int64)
    up = np.exp(lam * (per.dn.to_numpy() - d0))
    g = per.dc.to_numpy()
    W = pd.Series(up).groupby(g).cumsum().to_numpy()
    C = pd.Series(up * per.v.to_numpy()).groupby(g).cumsum().to_numpy()
    N = pd.Series(np.ones(len(per))).groupby(g).cumsum().to_numpy()
    rkey = g * BIG + per.dn.to_numpy(np.int64)
    qd = np.where(np.isfinite(q_dn), q_dn, -1).astype(np.int64)
    qkey = q_dog.astype(np.int64) * BIG + qd
    i = np.searchsorted(rkey, qkey, side="left") - 1
    hit = (i >= 0) & (g[np.clip(i, 0, None)] == q_dog) & (qd >= 0)
    ii = np.where(hit, i, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lvl = np.where(hit, C[ii] / W[ii], np.nan)
        w = np.where(hit, W[ii] * np.exp(-lam * (qd - d0[ii])), 0.0)
    return lvl, w, np.where(hit, N[ii], 0.0)


def offsets(tr: pd.DataFrame, fig: np.ndarray, r_dog: np.ndarray, r_dn: np.ndarray, r_fig: np.ndarray,
            k: float = K_OFF) -> tuple[np.ndarray, dict]:
    """(a) Each trial's offset (race figure less trial figure) for its track and trip and solo or company, walk-forward:
    from the pairs (a trial, then the dog's next race within PAIR_DAYS days) whose race was before the trial's day."""
    t_dn = tr.dn.to_numpy(float)
    solo = (tr["size"].to_numpy() <= 1).astype(int)
    ok = np.isfinite(r_fig)
    rr = pd.DataFrame({"dc": r_dog[ok].astype(np.int64), "dn": r_dn[ok].astype(np.int64), "v": r_fig[ok]})
    rr = rr.groupby(["dc", "dn"], as_index=False).v.mean().sort_values(["dc", "dn"])
    rkey = rr.dc.to_numpy() * BIG + rr.dn.to_numpy()
    good = np.isfinite(fig) & np.isfinite(t_dn)
    tdc, tdd = tr.dc.to_numpy(np.int64), np.where(good, t_dn, 0).astype(np.int64)
    j = np.searchsorted(rkey, tdc * BIG + tdd + 1)
    jj = np.clip(j, 0, max(len(rr) - 1, 0))
    if len(rr):
        nxt = good & (j < len(rr)) & (rr.dc.to_numpy()[jj] == tdc) & (rr.dn.to_numpy()[jj] - tdd <= PAIR_DAYS)
    else:
        nxt = np.zeros(len(tr), bool)
    diff = np.where(nxt, (rr.v.to_numpy()[jj] if len(rr) else 0.0) - fig, np.nan)
    nxt &= np.abs(diff) <= PAIR_CLIP
    pr = pd.DataFrame({"key": tr.key.to_numpy()[nxt], "solo": solo[nxt], "dn": rr.dn.to_numpy()[jj][nxt].astype(float),
                       "s": diff[nxt], "n": 1.0})
    notes = {"pairs": int(nxt.sum()),
             "pair_mean_solo": round(float(pr.s[pr.solo == 1].mean()), 3) if (pr.solo == 1).any() else None,
             "pair_mean_company": round(float(pr.s[pr.solo == 0].mean()), 3) if (pr.solo == 0).any() else None}
    q = pd.DataFrame({"key": tr.key.to_numpy(), "solo": solo, "dn": t_dn, "_i": np.arange(len(tr))})
    q = q[np.isfinite(t_dn)].sort_values("dn")

    def cum(by: list) -> tuple[np.ndarray, np.ndarray]:
        s, n = np.zeros(len(tr)), np.zeros(len(tr))
        if not len(pr):
            return s, n
        a = pr.groupby(by + ["dn"], as_index=False)[["s", "n"]].sum().sort_values(by + ["dn"])
        a[["s", "n"]] = a.groupby(by)[["s", "n"]].cumsum()
        m = pd.merge_asof(q, a.sort_values("dn"), on="dn", by=by, allow_exact_matches=False, direction="backward")
        s[m._i.to_numpy()] = m.s.fillna(0.0).to_numpy()
        n[m._i.to_numpy()] = m.n.fillna(0.0).to_numpy()
        return s, n

    sg, ng = cum(["key", "solo"])
    ss, ns = cum(["solo"])
    with np.errstate(invalid="ignore", divide="ignore"):
        glob = np.where(ns > 0, ss / np.where(ns > 0, ns, 1.0), 0.0)
    return (sg + k * glob) / (ng + k), notes


def session_variant(tr: pd.DataFrame, res: np.ndarray, est: np.ndarray,
                    fit_days: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """(b) Each trial's session going (lengths, above 0 the session ran fast): the mean of the other established dogs'
    residuals in its session (track and day), shrunk by k = within / between variance, k fitted on ``fit_days``."""
    sess = pd.factorize(tr.track.astype(str) + "|" + tr.dn.astype(str))[0]
    e = est & np.isfinite(res)
    r0 = np.where(e, res, 0.0)
    S = np.bincount(sess, weights=r0)
    N = np.bincount(sess, weights=e.astype(float))
    S2 = np.bincount(sess, weights=r0 * r0)
    early = np.zeros(len(S), bool)
    early[sess[fit_days]] = True
    use = early & (N >= 2)
    k, notes = 3.0, {}
    if use.sum() >= MIN_SESSIONS:
        within = float((S2[use] - S[use] ** 2 / N[use]).sum() / (N[use] - 1).sum())
        means = S[use] / N[use]
        between = float(np.var(means) - np.mean(within / N[use]))
        k = float(np.clip(within / between, 0.5, 50.0)) if between > 0 else 50.0
        notes = {"within_var": round(within, 3), "between_var": round(between, 3)}
    n_loo = N[sess] - e
    s_loo = S[sess] - r0
    v = np.where(n_loo > 0, s_loo / (n_loo + k), 0.0)
    return v, n_loo, {"k": round(k, 3), "sessions_fit": int(use.sum()), **notes}


def race_scale(tr: pd.DataFrame, races: pd.DataFrame, figs_cols=("gsr",),
               dogs: pd.Index | None = None) -> tuple[pd.DataFrame, dict]:
    """The trials' figures on the race scale (raw + offset - session going), one set a figure in ``figs_cols``, with
    the margins in company and the rival-implied figure (gsr). ``races`` carries the engine's figures."""
    tr = tr.copy().reset_index(drop=True)
    dogs = dogs if dogs is not None else pd.Index(pd.unique(np.concatenate([races.dog_id.to_numpy(),
                                                                            tr.dog_id.to_numpy()])))
    tr["dc"] = dogs.get_indexer(tr.dog_id)
    tr["dn"] = _dn(tr.race_date)
    tr["key"] = tl._key(tr).to_numpy()
    tr["size"] = tr.groupby("race_id").dog_id.transform("size").astype(float)
    raw = raw_figures(tr, races)
    r_dog = dogs.get_indexer(races.dog_id).astype(np.int64)
    r_dn = _dn(races.race_date)
    days = np.sort(pd.unique(tr.dn.dropna()))
    cut = days[int(len(days) * FIT_SHARE)] if len(days) else 0
    fit_days = (tr.dn < cut).to_numpy()
    notes = {}
    for f in figs_cols:
        rf = pd.to_numeric(races[f], errors="coerce").to_numpy(float)
        x = raw[f].to_numpy(float)
        off, n_off = offsets(tr, x, r_dog, r_dn, rf)
        lvl, _, n = level_at(r_dog, r_dn, rf, tr.dc.to_numpy(np.int64), tr.dn.to_numpy(float))
        est = (n >= MIN_RACES) & np.isfinite(lvl)
        v, n_s, n_v = session_variant(tr, x + off - lvl, est, fit_days)
        tr[f"{f}_raw"], tr[f"{f}_off"], tr[f"{f}_sess"], tr[f"{f}_sess_n"] = x, off, v, n_s
        tr[f"{f}_rs"] = x + off - v
        if f == "gsr":
            tr["race_lvl"], tr["est"] = lvl, est
        notes[f] = {**n_off, "session": n_v}
    # (c) in company: the margin by the clock, the position, the rivals' levels
    L = core.LENGTH_S
    rt = pd.to_numeric(tr.run_time, errors="coerce")
    g = rt.groupby(tr.race_id)
    first = g.transform("min")
    second = rt.where(g.rank(method="first") == 2).groupby(tr.race_id).transform("max")
    mg = ((rt - first) / L).clip(upper=MARGIN_CAP)
    mg = mg.where(rt > first, -((second - first) / L).clip(upper=MARGIN_CAP))
    comp = tr["size"].to_numpy() >= 2
    tr["margin"] = np.where(comp, mg.to_numpy(float), np.nan)
    tr["pos"] = np.where(comp, pd.to_numeric(tr.position, errors="coerce").to_numpy(float), np.nan)
    if "race_lvl" in tr:
        behind = tr.margin.clip(lower=0.0).to_numpy(float)          # lengths behind the trial's winner
        e = tr.est.to_numpy(bool) & np.isfinite(behind)
        a = np.where(e, tr.race_lvl.to_numpy(float) + behind, 0.0)
        rid = pd.factorize(tr.race_id)[0]
        S, N = np.bincount(rid, weights=a), np.bincount(rid, weights=e.astype(float))
        n_r = N[rid] - e
        with np.errstate(invalid="ignore", divide="ignore"):
            tr["comp_fig"] = np.where(comp & (n_r > 0), (S[rid] - a) / np.where(n_r > 0, n_r, 1) - behind, np.nan)
        tr["comp_n"] = np.where(comp, n_r, np.nan)
    return tr, notes


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    in_runs = tl.is_trial_class(df.race_class)
    races = df[~in_runs]
    tr = trial_rows(df, ctx)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    cols = list(COLUMNS)
    for c in cols:
        out[c] = np.nan
    dogs = pd.Index(pd.unique(np.concatenate([df.dog_id.to_numpy(), tr.dog_id.to_numpy()])))
    dc = dogs.get_indexer(df.dog_id).astype(np.int64)
    dn = tl._dn(df.race_date).astype(np.int64)
    r_dog = dogs.get_indexer(races.dog_id).astype(np.int64)
    r_dn = _dn(races.race_date)
    rlvl, _, rn = level_at(r_dog, r_dn, pd.to_numeric(races.gsr, errors="coerce").to_numpy(float), dc, dn.astype(float))
    out[f"{PREFIX}race_lvl"] = rlvl
    out[f"{PREFIX}race_n"] = rn
    notes = {"trial_rows": int(len(tr))}
    if not len(tr):
        out[f"{PREFIX}n_since_race"] = 0.0
        out[cols] = out[cols].astype(np.float32)
        return out.sort_index().reset_index(drop=True), notes
    tr, fit = race_scale(tr, races, ("gsr",), dogs)
    notes["fit"] = fit
    tr = tr[np.isfinite(tr.dn)]
    per = tr.groupby(["dc", "dn"], as_index=False).agg(
        rs=("gsr_rs", "mean"), raw=("gsr_raw", "mean"), off=("gsr_off", "mean"), sess=("gsr_sess", "mean"),
        sess_n=("gsr_sess_n", "mean"), size=("size", "min")).sort_values(["dc", "dn"]).reset_index(drop=True)
    gp = per.groupby("dc").rs
    per["best3"] = gp.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    per["mean3"] = gp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    tkey = per.dc.to_numpy(np.int64) * BIG + per.dn.to_numpy(np.int64)
    block = np.searchsorted(tkey, dc * BIG)

    def before(day: np.ndarray) -> np.ndarray:
        return np.searchsorted(tkey, dc * BIG + day) - block

    n_all = before(dn)
    has = n_all > 0
    li = np.where(has, block + n_all - 1, 0)

    def take(col: str, tab=per, h=has, i=li) -> np.ndarray:
        return np.where(h, tab[col].to_numpy(float)[i], np.nan)

    out[f"{PREFIX}days"] = np.where(has, dn - per.dn.to_numpy()[li], np.nan)
    for c, src in (("fig_raw_last", "raw"), ("offset_last", "off"), ("sess_last", "sess"), ("sess_n_last", "sess_n"),
                   ("fig_last", "rs"), ("fig_best3", "best3"), ("fig_mean3", "mean3"), ("solo_last", "size")):
        out[f"{PREFIX}{c}"] = take(src)
    out[f"{PREFIX}solo_last"] = np.where(has, (out[f"{PREFIX}solo_last"] <= 1).astype(float), np.nan)
    tlvl, tw, _ = level_at(per.dc.to_numpy(np.int64), per.dn.to_numpy(float), per.rs.to_numpy(float), dc,
                           dn.astype(float))
    out[f"{PREFIX}lvl"], out[f"{PREFIX}lvl_n"] = tlvl, tw
    out[f"{PREFIX}gap"] = tlvl - rlvl
    out[f"{PREFIX}gap_last"] = out[f"{PREFIX}fig_last"].to_numpy() - rlvl
    # trials since the last race (every trial before today when there is none)
    rk = np.unique(r_dog * BIG + np.where(np.isfinite(r_dn), r_dn, 0).astype(np.int64))
    rblock = np.searchsorted(rk, dc * BIG)
    r_before = np.searchsorted(rk, dc * BIG + dn) - rblock
    last_race = np.where(r_before > 0, rk[np.where(r_before > 0, rblock + r_before - 1, 0)] % BIG, -1)
    out[f"{PREFIX}n_since_race"] = (n_all - np.where(r_before > 0, before(last_race + 1), 0)).astype(float)
    # (c) the last trial in company
    cp = tr[tr["size"] >= 2].groupby(["dc", "dn"], as_index=False).agg(
        pos=("pos", "mean"), size=("size", "max"), margin=("margin", "mean"), comp=("comp_fig", "mean"),
        comp_n=("comp_n", "sum")).sort_values(["dc", "dn"]).reset_index(drop=True)
    ckey = cp.dc.to_numpy(np.int64) * BIG + cp.dn.to_numpy(np.int64)
    cblock = np.searchsorted(ckey, dc * BIG)
    cn = np.searchsorted(ckey, dc * BIG + dn) - cblock
    ch = cn > 0
    ci = np.where(ch, cblock + cn - 1, 0)
    if len(cp):
        out[f"{PREFIX}comp_days"] = np.where(ch, dn - cp.dn.to_numpy()[ci], np.nan)
        for c, src in (("pos_last", "pos"), ("field_last", "size"), ("margin_last", "margin"),
                       ("comp_fig_last", "comp"), ("comp_n_last", "comp_n")):
            out[f"{PREFIX}{c}"] = take(src, cp, ch, ci)
    # against the field: the dog's level (its race level once established, else its trial level)
    best = np.where((rn >= MIN_RACES) & np.isfinite(rlvl), rlvl, np.where(np.isfinite(tlvl), tlvl, rlvl))
    lv = pd.Series(best, index=df.index)
    gr = lv.groupby(df.race_id)
    out[f"{PREFIX}lvl_vs_field"] = (lv - (gr.transform("sum") - lv.fillna(0.0)) /
                                    (gr.transform("count") - lv.notna()).replace(0, np.nan)).to_numpy()
    out[cols] = out[cols].astype(np.float32)
    notes.update({"trial_dog_days": int(len(per)), "company_dog_days": int(len(cp)),
                  "rows_with_a_trial_before": round(float(has.mean()), 4)})
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}days": "days since the last trial",
    f"{PREFIX}n_since_race": "trial days since the dog's last race (every trial day when it has not raced)",
    f"{PREFIX}fig_raw_last": "the last trial's calculated time against the engine's graded standard, no going "
                            "(lengths)",
    f"{PREFIX}offset_last": "the trial-to-race offset applied to it: race figure less trial figure for its track, trip "
                            "and solo or company, walk-forward from dogs that raced within 21 days of a trial "
                            "(lengths)",
    f"{PREFIX}sess_last": "its session's going: the established dogs' trials in the session against their race "
                          "levels, the dog itself left out, shrunk (lengths, above 0 fast)",
    f"{PREFIX}sess_n_last": "established dogs behind that session going",
    f"{PREFIX}fig_last": "the last trial on the race scale: raw figure + offset - session going (lengths)",
    f"{PREFIX}fig_best3": "the best of the last three trial days on the race scale",
    f"{PREFIX}fig_mean3": "the mean of the last three",
    f"{PREFIX}solo_last": "1 when the last trial was run alone",
    f"{PREFIX}lvl": f"the dog's trial level on the race scale, decayed by {int(HALF_LIFE)} days",
    f"{PREFIX}lvl_n": "the decayed weight of trial days behind it",
    f"{PREFIX}race_lvl": f"the dog's race level (engine gsr), decayed by {int(HALF_LIFE)} days",
    f"{PREFIX}race_n": "race days behind it",
    f"{PREFIX}gap": "the trial level less the race level",
    f"{PREFIX}gap_last": "the last trial on the race scale less the race level",
    f"{PREFIX}comp_days": "days since the last trial in company",
    f"{PREFIX}pos_last": "the finishing position in the last trial in company",
    f"{PREFIX}field_last": "runners in it",
    f"{PREFIX}margin_last": "lengths behind its winner by the clock (the winner: minus its winning distance)",
    f"{PREFIX}comp_fig_last": "a going-free figure from it: the mean over rivals with an established race level of "
                              "their level plus their margin less the dog's (lengths)",
    f"{PREFIX}comp_n_last": "rivals with an established level in it",
    f"{PREFIX}lvl_vs_field": "the dog's level (race level once established, else trial level) against the mean of the "
                             "race's other runners",
}
