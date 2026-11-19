"""Breeding rebuilt (the feature review of 17 Nov 2026, the breeding row; reports/greyhound_feature_review.md): the
sire's and the dam's progeny measures of the breeding table (``br_``) built the way a prior should be.

- **Leave one dog out.** A dog's own runs never sit inside its sire's or its dam's mean, and for the dam neither do
  its littermates' (same dam, same month of birth): the breeding table counted them, so a dog's own form came back to
  it as "breeding".
- **Each progeny dog first.** A progeny's figures are averaged to one value a dog, then the dogs are averaged: a
  prolific son no longer outweighs ten lightly raced ones (the breeding table weighted by runs).
- **Empirical-Bayes shrinkage** toward the population's dog-first mean, by M / (M + k) with M the progeny dogs and
  k = sigma^2 / tau^2 (the spread of dogs within a sire or dam over the spread between sires or dams), fitted by the
  method of moments on an early slice (days before ``FIT_BEFORE``, or the first 20% of days), in place of the hard cut
  at 20 runs.
- **A/E against the de-biased SP** (``core.debiased_probs``, Shin): a dog-first excess (wins less chances, a run) and a
  run-weighted A/E with a gamma prior whose weight is fitted on the early slice (review B5).
- **Trials.** ``data.clean`` drops every trial, so the breeding table's "trials included" never was (review B2). The
  trial rows (``ctx["trial_runs"]``, ``greyhound/pit/trials.py select``) enter one measure, ``fig``: the calculated
  time against the track and trip's graded standard (the trials table's scale, lengths, higher faster), races and
  trials together, so a sire's unraced and lightly raced progeny count. The engine figures (gsr, esr, fsr) have no
  trial values.

Every column reads days strictly before the row's day and needs only the card's sire, dam and birth month: card-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, figs as figs_mod, trials as trials_mod
from greyhound.pit.breeding import band

PREFIX = "bq_"
DEPENDS = ("breeding", "trials", "figs")
FIT_BEFORE = "2018-01-01"
K_MIN, K_MAX = 0.5, 500.0
SOURCE = ("GBGB results and trial rows (sire, dam, born, times, SP: sources/gbgb.py) through the engine's figures; "
          "the SP de-biased (Shin)")
WHO = ("sire", "dam")


def _dn(dates) -> np.ndarray:
    return (pd.to_datetime(pd.Series(dates), errors="coerce").dt.normalize()
            - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float)


def _norm(s: pd.Series) -> pd.Series:
    k = s.astype(object).where(s.notna(), "").astype(str).str.strip().str.lower()
    return k.where(k != "", "?")


def unit_events(unit, key, lit, dn, v) -> pd.DataFrame:
    """Each unit's (a progeny dog, or a dog in a band) change of its running mean on each of its days with a value:
    ``dS`` (the new mean less the old, the whole new mean on its first day) and ``dM`` (1 on its first day). Summed
    over a group's units and days before D they give the sum of its units' means and their count at D."""
    d = pd.DataFrame({"u": np.asarray(unit), "key": np.asarray(key), "lit": np.asarray(lit), "dn": np.asarray(dn, float),
                      "v": np.asarray(v, float)})
    d = d[np.isfinite(d.v) & np.isfinite(d.dn)]
    if not len(d):
        return pd.DataFrame(columns=["u", "key", "lit", "dn", "dS", "dM"])
    first = d.groupby("u")[["key", "lit"]].first()                       # one sire, dam and litter a unit
    per = d.groupby(["u", "dn"], sort=False).v.agg(["sum", "count"]).reset_index().sort_values(["u", "dn"])
    g = per.groupby("u")
    m = g["sum"].cumsum() / g["count"].cumsum()
    prev = m.groupby(per.u).shift(1)
    per["dS"] = (m - prev.fillna(0.0)).to_numpy()
    per["dM"] = prev.isna().astype(float).to_numpy()
    per["key"] = first.key.reindex(per.u).to_numpy()
    per["lit"] = first.lit.reindex(per.u).to_numpy()
    return per[["u", "key", "lit", "dn", "dS", "dM"]]


def before(ev: pd.DataFrame, by: str, cols: list[str], qkey, qdn) -> np.ndarray:
    """The sums of ``cols`` over ``ev``'s rows of the query's group ``by`` on days strictly before ``qdn``."""
    res = np.zeros((len(qdn), len(cols)))
    if not len(ev):
        return res
    agg = ev.groupby([by, "dn"], sort=False)[cols].sum().reset_index().sort_values([by, "dn"])
    agg[cols] = agg.groupby(by)[cols].cumsum().to_numpy()
    q = pd.DataFrame({by: np.asarray(qkey).astype(str), "dn": np.asarray(qdn, float), "_i": np.arange(len(qdn))})
    q = q[np.isfinite(q.dn)].sort_values("dn")
    agg[by] = agg[by].astype(str)
    m = pd.merge_asof(q, agg.sort_values("dn"), on="dn", by=by, allow_exact_matches=False, direction="backward")
    res[m._i.to_numpy()] = m[cols].fillna(0.0).to_numpy()
    return res


def fit_k(unit_mean: pd.Series, key: pd.Series) -> dict:
    """k = sigma^2 / tau^2 by the method of moments on units' means grouped by sire or dam: sigma^2 the pooled spread
    within a group, tau^2 the spread of group means less their sampling share (groups of 2 or more units)."""
    d = pd.DataFrame({"x": unit_mean.to_numpy(float), "k": key.to_numpy()})
    d = d[np.isfinite(d.x) & (d.k != "?")]
    g = d.groupby("k").x
    m, mean = g.transform("size"), g.transform("mean")
    ok = m >= 2
    if ok.sum() < 10:
        return {"k": None, "sigma2": None, "tau2": None, "groups": 0}
    sigma2 = float(((d.x - mean)[ok] ** 2).sum() / (g.size()[g.size() >= 2] - 1).sum())
    gm = g.agg(["mean", "size"])
    gm = gm[gm["size"] >= 2]
    tau2 = float(gm["mean"].var() - (sigma2 / gm["size"]).mean())
    k = sigma2 / tau2 if tau2 > 0 else K_MAX
    return {"k": float(np.clip(k, K_MIN, K_MAX)), "sigma2": round(sigma2, 5), "tau2": round(tau2, 5),
            "groups": int(len(gm))}


def progeny(ev: pd.DataFrame, rows: pd.DataFrame, who: str, early_dn: float, pop: str = "_all",
            excl: str | None = None) -> tuple[pd.Series, pd.Series, dict]:
    """The rows' shrunk dog-first progeny mean (against the population's dog-first mean before the day) and the
    progeny units behind it, the row's own unit (and for the dam its litter) left out."""
    S = before(ev, "key", ["dS", "dM"], rows.key, rows.dn)
    if who == "dam":
        X = before(ev, "lit", ["dS", "dM"], rows.lit, rows.dn)
    else:
        X = before(ev.assign(uu=ev.u.astype(str)), "uu", ["dS", "dM"], rows.unit.astype(str), rows.dn)
    s, n = S[:, 0] - X[:, 0], np.round(S[:, 1] - X[:, 1])
    P = before(ev.assign(_p=ev[pop] if pop in ev else "all"), "_p", ["dS", "dM"],
               rows[pop] if pop in rows else np.full(len(rows), "all"), rows.dn)
    mu = np.where(P[:, 1] > 0, P[:, 0] / np.where(P[:, 1] > 0, P[:, 1], 1), np.nan)
    # the fit: each unit's mean over the early days, grouped by the parent (pooled over bands for the band measure)
    e = ev[ev.dn < early_dn].sort_values(["u", "dn"])
    um = e.groupby("u").dS.sum()                         # the sum of a unit's mean changes is its last mean
    fit = fit_k(um, e.groupby("u").key.first().reindex(um.index))
    k = fit["k"] if fit["k"] is not None else 5.0
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar = np.where(n > 0, s / np.where(n > 0, n, 1), mu)
        shr = mu + n / (n + k) * (xbar - mu)
    known = (rows.key != "?").to_numpy()
    return (pd.Series(np.where(known, shr - mu, np.nan), index=rows.index),
            pd.Series(np.where(known, np.maximum(n, 0), np.nan), index=rows.index), fit)


def _trial_fig(df: pd.DataFrame, trial_runs: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series, int]:
    """Each race row's and each trial row's calculated time against the track and trip's graded standard (lengths,
    the trials table's scale) and the trial rows themselves (dog, sire, dam, born, day)."""
    in_runs = trials_mod.is_trial_class(df.race_class).fillna(False) if "race_class" in df else \
        pd.Series(False, index=df.index)
    races = df[~in_runs]
    std = trials_mod.standards(races)
    L = core.LENGTH_S
    bad = races.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in races \
        else pd.Series(False, index=races.index)
    fr = ((std.std_t - pd.to_numeric(races.adjusted_time, errors="coerce")) / L).where(~bad)
    fr = fr.where(fr.abs() <= trials_mod.OUTLIER).reindex(df.index)
    parts = [trials_mod.select(df[in_runs])] if in_runs.any() else []
    if trial_runs is not None and len(trial_runs):
        parts.append(trials_mod.select(trial_runs))
    if not parts:
        return pd.DataFrame(columns=["dog_id", "sire", "dam", "born", "dn", "fig"]), fr, 0
    tr = pd.concat(parts, ignore_index=True).drop_duplicates(["race_id", "dog_id"], keep="last")
    key_r = figs_mod._key(races)
    look = pd.DataFrame({"key": key_r.to_numpy(), "dn": _dn(races.race_date), "std_t": std.std_t.to_numpy()})
    look = look.dropna(subset=["dn"]).drop_duplicates(["key", "dn"]).sort_values("dn")
    tr = tr.assign(key=figs_mod._key(tr).to_numpy(), dn=_dn(tr.race_date)).dropna(subset=["dn"]).sort_values("dn")
    tr = pd.merge_asof(tr, look, on="dn", by="key", direction="backward")
    bad_t = tr.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in tr else False
    f = ((tr.std_t - pd.to_numeric(tr.adjusted_time, errors="coerce")) / L).where(~bad_t)
    tr["fig"] = f.where(f.abs() <= trials_mod.OUTLIER)
    for c in ("sire", "dam", "born"):
        if c not in tr:
            tr[c] = np.nan
    return tr[["dog_id", "sire", "dam", "born", "dn", "fig"]], fr, int(tr.fig.notna().sum())


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    out = pd.DataFrame({"race_id": df.race_id, "dog_id": df.dog_id}, index=df.index)
    dn = _dn(df.race_date)
    days = np.sort(np.unique(dn[np.isfinite(dn)]))
    early_dn = float(_dn([FIT_BEFORE])[0])
    if (dn < early_dn).sum() < 0.05 * len(df) and len(days):
        early_dn = float(days[int(len(days) * 0.2)])                     # the first 20% of days
    born = _norm(df.born) if "born" in df else pd.Series("?", index=df.index)
    dog = df.dog_id.astype(str)
    # the de-biased SP chance of each race run (review B5); a card row has no result and enters nothing
    p = core.debiased_probs(pd.to_numeric(df.get("sp_decimal"), errors="coerce"), df.race_id, "shin") \
        if "sp_decimal" in df else df.sp_p_norm.to_numpy(float)
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(len(df), bool)
    racing = ~trial & np.isfinite(p) & df.won.notna().to_numpy()
    won = np.where(racing, df.won.to_numpy(float), np.nan)
    xs = won - np.where(racing, p, np.nan)
    tr, fr, n_trial_figs = _trial_fig(df, (ctx or {}).get("trial_runs"))
    # pandas 3: astype(str) keeps a missing trip as NaN, which breaks the band keys; a run with no trip is band "?"
    bnd = band(df.distance_m).astype(object)
    bnd = bnd.where(bnd.notna() & (bnd.astype(str) != "nan"), "?").map(str)
    notes = {"fit_before_day": str(pd.Timestamp("2000-01-01") + pd.Timedelta(days=early_dn))[:10],
             "trial_figures": n_trial_figs, "debias": "shin" if "sp_decimal" in df else "proportional (no sp_decimal)",
             "k": {}}
    for who in WHO:
        key = _norm(df[who]) if who in df else pd.Series("?", index=df.index)
        lit = np.where(born != "?", key + "|" + born, "dog|" + dog) if who == "dam" else "dog|" + dog
        rows = pd.DataFrame({"unit": dog, "key": key, "lit": lit, "dn": dn}, index=df.index)
        P = f"{PREFIX}{who}_"
        for f in ("gsr", "esr", "fsr"):
            ev = unit_events(dog, key, lit, dn, df[f].to_numpy(float))
            out[f"{P}{f}"], n, fit = progeny(ev, rows, who, early_dn)
            notes["k"][f"{who}_{f}"] = fit
            if f == "gsr":
                out[f"{P}dogs"] = n
        # races and trials on one scale (the trials table's figure): a trial-only progeny counts
        tkey = _norm(tr[who]) if len(tr) else pd.Series(dtype=object)
        tborn = _norm(tr.born) if len(tr) else pd.Series(dtype=object)
        tdog = tr.dog_id.astype("int64").astype(str) if len(tr) else pd.Series(dtype=object)
        tlit = (np.where(tborn != "?", tkey + "|" + tborn, "dog|" + tdog) if who == "dam" else "dog|" + tdog) \
            if len(tr) else np.array([], dtype=object)
        ev = unit_events(np.r_[dog.to_numpy(), tdog.to_numpy()], np.r_[key.to_numpy(), tkey.to_numpy()],
                         np.r_[np.broadcast_to(lit, len(df)), np.asarray(tlit)], np.r_[dn, tr.dn.to_numpy(float)],
                         np.r_[fr.to_numpy(float), tr.fig.to_numpy(float)])
        out[f"{P}fig"], out[f"{P}fig_dogs"], notes["k"][f"{who}_fig"] = progeny(ev, rows, who, early_dn)
        # A/E on the de-biased SP: a dog-first excess (wins less chances a run), and run-weighted with a gamma prior
        ev = unit_events(dog, key, lit, dn, xs)
        out[f"{P}xs"], _, notes["k"][f"{who}_xs"] = progeny(ev, rows, who, early_dn)
        rr = pd.DataFrame({"u": dog, "key": key, "lit": lit, "dn": dn, "w": won, "e": np.where(racing, p, np.nan)})
        rr = rr[racing]
        W = before(rr, "key", ["w", "e"], key, dn)
        X = before(rr, "lit", ["w", "e"], lit, dn) if who == "dam" else before(rr, "u", ["w", "e"], dog, dn)
        w_, e_ = W[:, 0] - X[:, 0], W[:, 1] - X[:, 1]
        early = rr[rr.dn < early_dn].groupby("key")[["w", "e"]].sum()
        early = early[(early.index != "?") & (early.e >= 1.0)]
        if len(early) >= 10:
            v = float((early.w / early.e).var() - (1.0 / early.e).mean())
            kg = float(np.clip(1.0 / v, K_MIN, K_MAX)) if v > 0 else K_MAX
        else:
            kg = 5.0
        notes["k"][f"{who}_ae_gamma"] = round(kg, 3)
        out[f"{P}ae"] = np.where(key != "?", (w_ + kg) / (np.maximum(e_, 0) + kg), np.nan)
        # stamina: the progeny's dog-first gsr in today's distance band against the population's in it, less overall
        ub = (dog + "|" + bnd).to_numpy()
        kb = np.where(key != "?", key + "|" + bnd, "?")
        lb = np.asarray(lit, dtype=object) + "|" + bnd.to_numpy() if who == "dam" else ub
        ev = unit_events(ub, kb, lb, dn, df.gsr.to_numpy(float)).assign(
            _band=lambda e: e.u.astype(str).str.rsplit("|", n=1).str[1])
        rb = pd.DataFrame({"unit": ub, "key": kb, "lit": lb, "dn": dn, "_band": bnd.to_numpy()}, index=df.index)
        in_band, _, notes["k"][f"{who}_band"] = progeny(ev, rb, who, early_dn, pop="_band")
        out[f"{P}band_vs_all"] = in_band - out[f"{P}gsr"]
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    notes["card_safe"] = cols
    return out.reset_index(drop=True), notes


COLUMNS = {
    **{f"{PREFIX}{w}_dogs": f"the {w}'s progeny dogs with a speed rating before today, the dog"
                            f"{' and its littermates' if w == 'dam' else ''} left out" for w in WHO},
    **{f"{PREFIX}{w}_{f}": f"the {w}'s progeny's {lab}, each dog averaged first, the dog"
                           f"{' and its littermates' if w == 'dam' else ''} left out, shrunk by empirical Bayes to the "
                           "population's, against the population's (lengths)"
       for w in WHO for f, lab in (("gsr", "speed rating"), ("esr", "early pace"), ("fsr", "finishing speed"))},
    **{f"{PREFIX}{w}_fig": f"the {w}'s progeny's calculated time against the graded standard, races and trials "
                           "together, dog-first, left out and shrunk as above, against the population's (lengths)"
       for w in WHO},
    **{f"{PREFIX}{w}_fig_dogs": f"the {w}'s progeny dogs behind it (trial-only dogs included)" for w in WHO},
    **{f"{PREFIX}{w}_xs": f"the {w}'s progeny's wins less their de-biased SP chances a run, dog-first, left out and "
                          "shrunk" for w in WHO},
    **{f"{PREFIX}{w}_ae": f"the {w}'s progeny's wins over their de-biased SP chances, run-weighted, left out, with a "
                          "gamma prior fitted on the early slice" for w in WHO},
    **{f"{PREFIX}{w}_band_vs_all": f"the {w}'s progeny's shrunk speed in today's distance band less their overall "
                                   "(each against the population's): stamina" for w in WHO},
}
