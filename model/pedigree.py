"""
Pedigree suites: sire, damsire, dam, siblings, the nick and the female family
(racing² master framework, Section III Part 5.3 - 5.4).

Why this module exists, and why it is not more sire strike rates
----------------------------------------------------------------
``model/custom_metrics.py`` already carries sire and damsire *levels* — shrunk
mean NFP and win rate, overall and conditioned on going or on distance. §5.3 is
explicit that levels are the part of pedigree the market prices well ("Raw sire
strike rate is largely 'is this a good sire'") and that the alpha is in the
*residual*: how a sire's runners do on today's going, at today's trip, on sand,
as two-year-olds, on debut — **relative to that same sire's own overall
level**. Subtracting the sire's own mean is what removes the "good sire"
component; a conditional level keeps it and duplicates the market.

Every aptitude here is therefore

    apt = (Σx_cell + k · m_overall) / (n_cell + k) − m_overall
        ≡ n_cell / (n_cell + k) · (mean_cell − m_overall)

— the cell mean shrunk toward the entity's own overall mean (Part 7
credibility), expressed as a residual. A sire with three runners on heavy
ground gets an aptitude near zero rather than a number.

What is here
------------
    sire_prog_runs / _prog_wins / _sr_shrunk / _nmfp_shrunk / _prb2_fsa_shrunk
    sire_going_apt, sire_dist_apt, sire_surface_apt, sire_course_apt,
    sire_2yo_apt, sire_first_time_out_apt, sire_improvement
    sire_class_rating, sire_elite_share, sire_track_strength,
    sire_median_hcap_debut_or
    damsire_*  — the same structure (§5.4: "mirrors the sire aptitude
                 structure", stamina and temperament rather than speed)
    dam_runners, dam_winners, dam_sr_shrunk, dam_progeny_mean_nmfp,
    dam_black_type_progeny, dam_best_progeny_rating,
    dam_own_runs / _nmfp / _best_or / _black_type
    sib_count, sib_mean_nmfp, sib_dist_profile, sib_going_profile,
    sib_precocity, sib_improvement_curve
    nick_runs, nick_apt, nick_sr_apt
    family_bt_density, family_class_rating

(§5.4 lists ``sib_best_rating`` and ``dam_best_progeny_rating`` as separate
names for the same quantity — the best rating achieved by the horse's
siblings — so it is computed once, as ``dam_best_progeny_rating``.)

Lag safety
----------
A sire or a dam can have several runners in one race and several more on the
same card. ``shift(1)`` on such a key steps back to a rival whose result is not
yet known, and because these figures are otherwise flat across a race the
leaked result is the only thing that varies within it. Every aggregate below is
built with ``model.lagsafe.race_lagged_expanding_*`` keyed on the *day*
(``model.connections.prior_stats``), so a figure can only see progeny runs from
strictly earlier days: not the rest of today's race, and not the rest of
today's card. Only ``horse_name`` is ever row-shifted, because a horse runs
once in a race.

Sibling and dam figures subtract the horse's own contribution from the dam's
totals, both taken on the same day-lagged basis so the subtraction is exact: a
horse is not its own sibling, and its own form already reaches the model
through the horse features. ``dam_best_progeny_rating`` is a maximum, which
does not subtract, so it is built from a running top-two keyed on distinct
progeny.

No column of ``model.custom_metrics.POST_RACE_ONLY`` is used as a feature.
Finishing positions enter only as *prior* runs of other horses, which is what
a pedigree statistic is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.connections import (CARD_KEY, aptitude_residual, card_key, dist_band, first_time_out, going_group,
                               going_number, is_black_type, is_handicap, lr_place_rating, lr_place_score,
                               prior_stats, running_population_mean, shrink, surface_is_aw, _first_col,
                               N_CANDIDATES, POS_CANDIDATES)
from model.primitives import nmfp as _nmfp_primitive, prb, prb2_par

#: §5.3's "elite" runner: one that beat at least 78% of its rivals. Reading the
#: 78th percentile within the race makes the threshold field-size-free (P1) and
#: needs no cross-race quantile, which could not be taken lag-safely.
ELITE_PRB = 0.78


def class_rating(geo_wpmcv, prb2_mean, index=None) -> pd.Series:
    """v1 §5.2 class rating: ``LN(geomean WPMCV) · LN(geomean WPMCV · %RB²/100)``.

    %RB² arrives as a fraction, which is the ``/100`` of the published formula
    (written for %RB² on a 0-100 scale). Undefined where either logarithm has a
    non-positive argument."""
    g = pd.to_numeric(pd.Series(np.asarray(geo_wpmcv, dtype=float)), errors="coerce").values
    p = pd.to_numeric(pd.Series(np.asarray(prb2_mean, dtype=float)), errors="coerce").values
    inner = g * p
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where((g > 0) & (inner > 0), np.log(np.where(g > 0, g, np.nan))
                       * np.log(np.where(inner > 0, inner, np.nan)), np.nan)
    return pd.Series(out, index=index)


def _norm_name(s: pd.Series) -> pd.Series:
    """Names come from two scrapes with inconsistent case and padding."""
    return s.fillna("").astype(str).str.strip().str.lower()


_UNKNOWN = {"", "unknown", "unnamed", "n/a", "na", "none", "nan", "-", "?"}


def _known(s: pd.Series) -> pd.Series:
    return ~s.isin(_UNKNOWN)


def _working_frame(src: pd.DataFrame, sire_col: str, dam_col: str, damsire_col: str, horse_col: str,
                   date_col: str, nmfp_col: str) -> pd.DataFrame:
    """Per-run value columns, entity keys and aptitude cells — and nothing else.

    Every aggregate below groups, sorts and merges this frame, so it is built
    narrow rather than by bolting columns onto the caller's two hundred."""
    pos_c, n_c = _first_col(src, POS_CANDIDATES), _first_col(src, N_CANDIDATES)
    P = pd.to_numeric(src[pos_c], errors="coerce").values if pos_c else np.full(len(src), np.nan)
    N = pd.to_numeric(src[n_c], errors="coerce").values if n_c else np.full(len(src), np.nan)
    idx = src.index
    c: dict = {"race_date": src[date_col], CARD_KEY: card_key(src, date_col),
               "_time": src["race_time"].astype(str) if "race_time" in src.columns else "",
               "_one": pd.Series(1.0, index=idx)}
    if "won" in src.columns:
        c["_won"] = pd.to_numeric(src["won"], errors="coerce")
    else:
        c["_won"] = pd.Series(np.where(np.isnan(P), np.nan, (P == 1).astype(float)), index=idx)
    c["_placed"] = pd.Series(np.where(np.isnan(P), np.nan, (P <= 3).astype(float)), index=idx)
    c["_prb"] = pd.Series(prb(N, P), index=idx)
    c["_prb2"] = c["_prb"] ** 2
    c["_prb2_fsa"] = c["_prb2"] - prb2_par(N)
    c["_elite"] = pd.Series(np.where(np.isnan(c["_prb"].values), np.nan,
                                     (c["_prb"].values >= ELITE_PRB).astype(float)), index=idx)
    if nmfp_col in src.columns:
        c["_nmfp"] = pd.to_numeric(src[nmfp_col], errors="coerce")
    else:
        c["_nmfp"] = pd.Series(_nmfp_primitive(N, P), index=idx)
    c["_or"] = pd.to_numeric(src["official_rating"], errors="coerce") if "official_rating" in src.columns else np.nan
    c["_age"] = pd.to_numeric(src["horse_age"], errors="coerce") if "horse_age" in src.columns else np.nan
    dist_c = _first_col(src, ("dist_furlongs", "distance_f"))
    c["_dist"] = pd.to_numeric(src[dist_c], errors="coerce") if dist_c else np.nan
    going_c = _first_col(src, ("going_description", "going"))
    c["_going_num"] = going_number(src[going_c]) if going_c else np.nan
    c["_ln_prize"] = (np.log(pd.to_numeric(src["prize_money"], errors="coerce").where(lambda s: s > 0))
                      if "prize_money" in src.columns else np.nan)
    bt = is_black_type(src)
    c["_bt_placing"] = pd.Series(np.where(np.isnan(P), np.nan, (bt.values & (P <= 3)).astype(float)), index=idx)
    c["_lr_score"] = lr_place_score(src, pos_col=pos_c)
    fto = first_time_out(src)
    c["_fto"] = fto
    c["_non_fts"] = 1.0 - fto.fillna(0.0)

    for col, name in ((sire_col, "_sire"), (dam_col, "_dam"), (damsire_col, "_damsire"), (horse_col, "_horse")):
        c[name] = _norm_name(src[col]) if col in src.columns else pd.Series("", index=idx)
        c[name + "_known"] = _known(c[name])
    # an unnamed horse must never merge with another unnamed horse
    c["_horse"] = c["_horse"].where(c["_horse_known"], pd.Series("__row_", index=idx) + idx.astype(str))

    # --- aptitude cells (object, NaN where the frame cannot say) -----------
    c["_cell_going"] = going_group(src[going_c]) if going_c else pd.Series(np.nan, index=idx)
    c["_cell_dist"] = dist_band(c["_dist"])
    c["_cell_surface"] = surface_is_aw(src)
    track_c = _first_col(src, ("track", "course", "course_bf"))
    c["_cell_course"] = src[track_c].astype(str).where(src[track_c].notna()) if track_c else pd.Series(np.nan, index=idx)
    for name in ("_cell_going", "_cell_dist", "_cell_surface", "_cell_course"):
        s = pd.Series(c[name], index=idx)
        c[name] = s.astype(object).where(s.notna())

    # --- masked per-run subsets (sire-level, not conditioned on today) -----
    c["_nmfp_2yo"] = c["_nmfp"].where(c["_age"] == 2)                       # precocity
    c["_nmfp_fto"] = c["_nmfp"].where(c["_fto"] > 0)                        # first time out

    w = pd.DataFrame(c, index=idx)

    # --- per-horse career markers (row-shift on the horse is the safe one) --
    order = w.sort_values(["race_date", "_time"], kind="stable").index
    horses = w["_horse"].reindex(order)
    if "career_runs" in src.columns and pd.to_numeric(src["career_runs"], errors="coerce").notna().any():
        run_idx = pd.to_numeric(src["career_runs"], errors="coerce")
    else:
        run_idx = horses.groupby(horses).cumcount().reindex(w.index).astype(float)
    w["_run_idx"] = run_idx
    w["_debut_flag"] = (run_idx == 0).astype(float)
    first_nmfp = w["_nmfp"].where(run_idx == 0).groupby(w["_horse"]).transform("max")
    w["_improvement"] = (w["_nmfp"] - first_nmfp).where(run_idx == 2)       # run 3 − run 1 (§5.3)
    prior_wins = w["_won"].reindex(order).groupby(horses).cumsum().reindex(w.index) - w["_won"].fillna(0.0)
    first_win = (w["_won"] > 0) & (prior_wins.fillna(0.0) <= 0)
    w["_win_first"] = first_win.astype(float)
    w["_first_win_age"] = w["_age"].where(first_win)
    w["_win_dist"] = w["_dist"].where(w["_won"] > 0)
    w["_win_going"] = w["_going_num"].where(w["_won"] > 0)
    prior_bt = (w["_bt_placing"].reindex(order).groupby(horses).cumsum().reindex(w.index)
                - w["_bt_placing"].fillna(0.0))
    w["_bt_first"] = ((w["_bt_placing"] > 0) & (prior_bt.fillna(0.0) <= 0)).astype(float)
    hcap = is_handicap(src).reindex(order).fillna(False).astype(bool)
    h_idx = order[hcap.values]
    hcap_debut = pd.Series(False, index=idx)
    if len(h_idx):
        hh = w["_horse"].reindex(h_idx)
        hcap_debut.loc[h_idx[(hh.groupby(hh).cumcount() == 0).values]] = True
    w["_hcap_debut_or"] = w["_or"].where(hcap_debut)
    return w


# ---------------------------------------------------------------------------
# Sire / damsire: levels, then the aptitude residuals that matter
# ---------------------------------------------------------------------------

def _add_sire_like(w: pd.DataFrame, new: dict, key: str, prefix: str, pop: dict, k_sr: float,
                   k_nmfp: float, k_apt: float, cells=("going", "dist", "surface", "course"),
                   masked=(("2yo", "_nmfp_2yo"), ("first_time_out", "_nmfp_fto"))) -> list[str]:
    known = w[key + "_known"]
    wins, runs = prior_stats(w, key, "_won")
    nm, nm_n = prior_stats(w, key, "_nmfp")
    pr2, pr2_n = prior_stats(w, key, "_prb2_fsa")
    # The *level* is shrunk toward the population; the residual base is the
    # entity's own raw mean. Comparing a raw cell mean against a shrunk overall
    # would leave the shrinkage gap in the residual, and every cell of a strong
    # sire would read as an aptitude.
    base = (nm / nm_n.replace(0, np.nan)).fillna(pop["nmfp"])
    w[f"_{prefix}_nmfp_raw"] = base
    w[f"_{prefix}_sr_raw"] = (wins / runs.replace(0, np.nan)).fillna(pop["sr"])
    new[f"{prefix}_prog_runs"] = runs.where(known)
    new[f"{prefix}_prog_wins"] = wins.where(known)
    new[f"{prefix}_sr_shrunk"] = shrink(wins, runs, pop["sr"], k_sr).where(known)
    new[f"{prefix}_nmfp_shrunk"] = shrink(nm, nm_n, pop["nmfp"], k_nmfp).where(known)
    new[f"{prefix}_prb2_fsa_shrunk"] = shrink(pr2, pr2_n, pop["prb2"], k_nmfp).where(known)
    feats = [f"{prefix}_prog_runs", f"{prefix}_prog_wins", f"{prefix}_sr_shrunk",
             f"{prefix}_nmfp_shrunk", f"{prefix}_prb2_fsa_shrunk"]

    for name in cells:
        col = f"_cell_{name}"
        if col not in w.columns or not w[col].notna().any():
            continue
        here = known & w[col].notna()
        c_tot, c_n = prior_stats(w, [key, col], "_nmfp")
        new[f"{prefix}_{name}_apt"] = aptitude_residual(c_tot, c_n, base, k_apt).where(here)
        new[f"{prefix}_{name}_n"] = c_n.where(here)
        feats += [f"{prefix}_{name}_apt", f"{prefix}_{name}_n"]

    for name, col in masked:
        if col not in w.columns:
            continue
        m_tot, m_n = prior_stats(w, key, col)
        new[f"{prefix}_{name}_apt"] = aptitude_residual(m_tot, m_n, base, k_apt).where(known)
        new[f"{prefix}_{name}_n"] = m_n.where(known)
        feats += [f"{prefix}_{name}_apt", f"{prefix}_{name}_n"]
    return feats


def _add_sire_extras(w: pd.DataFrame, new: dict, key: str, prefix: str, pop: dict, k_sr: float,
                     k_apt: float, k_lr: float, min_hcap_debuts: int) -> list[str]:
    """The §5.3 composites: do the progeny progress, what class are they, how
    many are elite, and the two constructions taken from the internal research."""
    known = w[key + "_known"]
    imp_tot, imp_n = prior_stats(w, key, "_improvement")
    # a mean of (run 3 − run 1) NMFP, shrunk toward 0 — toward "no progression"
    new[f"{prefix}_improvement"] = shrink(imp_tot, imp_n, 0.0, k_apt).where(known)
    el_tot, el_n = prior_stats(w, key, "_elite")
    new[f"{prefix}_elite_share"] = shrink(el_tot, el_n, pop["elite"], k_sr).where(known)
    lp, lp_n = prior_stats(w, key, "_ln_prize")
    geo = np.exp(lp / lp_n.replace(0, np.nan))                          # geomean WPMCV
    p2, p2_n = prior_stats(w, key, "_prb2")
    new[f"{prefix}_class_geo"] = geo.where(known)
    new[f"{prefix}_class_rating"] = class_rating(geo.values, shrink(p2, p2_n, pop["prb2_raw"], k_sr).values,
                                                 index=w.index).where(known)
    # §5.3 Track Strength, built like the §5.2 LR place rating: black-type
    # placings weighted by the win% of each placing, per non-first-time starter.
    # Read as a sire-level scalar; the per-course view is sire_course_apt.
    new[f"{prefix}_track_strength"] = lr_place_rating(w, key, k=k_lr).where(known)
    new[f"{prefix}_median_hcap_debut_or"] = _median_hcap_debut_or(w, key, min_n=min_hcap_debuts).where(known)
    return [f"{prefix}_improvement", f"{prefix}_elite_share", f"{prefix}_class_geo",
            f"{prefix}_class_rating", f"{prefix}_track_strength", f"{prefix}_median_hcap_debut_or"]


def _median_hcap_debut_or(w: pd.DataFrame, key: str, years: float = 3.0, min_n: int = 30) -> pd.Series:
    """§5.3's Median Handicap Debut OR: the median mark the sire's progeny are
    given on their *handicap* debut, over a rolling `years` window, reported
    only once at least `min_n` of them qualify.

    Evaluated at the first day of each month against debutants strictly before
    that day, so it cannot see the card it is used on (and is at most a month
    stale — the metric moves on a three-year window, so that costs nothing).

    §X.10 records that our own 2016 test of this metric did not replicate the
    internal research's win-share claim and that a full rolling backtest is
    still owed. It ships as a feature to be tested, not as one to trust."""
    v = pd.to_numeric(w["_hcap_debut_or"], errors="coerce")
    dates = pd.to_datetime(w["race_date"], errors="coerce").values
    days = dates.astype("datetime64[D]").astype(np.int64)
    months = dates.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    k = w[key].values
    ok = v.notna().values & w[key + "_known"].values
    contrib: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if ok.any():
        c = pd.DataFrame({"k": k[ok], "d": days[ok], "v": v.values[ok]}).sort_values(["k", "d"], kind="stable")
        for name, blk in c.groupby("k", sort=False):
            contrib[name] = (blk["d"].values, blk["v"].values)
    window = int(round(years * 365.25))
    ev = pd.DataFrame({"k": k, "m": months}).drop_duplicates()
    res: dict[tuple, float] = {}
    for name, blk in ev.groupby("k", sort=False):
        arr = contrib.get(name)
        if arr is None:
            continue
        cd, cv = arr
        ms = blk["m"].values
        lo = np.searchsorted(cd, ms - window, side="left")
        hi = np.searchsorted(cd, ms, side="left")
        enough = (hi - lo) >= min_n
        for m_i, a, b in zip(ms[enough], lo[enough], hi[enough]):
            res[(name, int(m_i))] = float(np.median(cv[a:b]))
    if not res:
        return pd.Series(np.nan, index=w.index, dtype=float)
    src = pd.Series(list(res.values()), index=pd.MultiIndex.from_tuples(list(res.keys())))
    return pd.Series(src.reindex(pd.MultiIndex.from_arrays([k, months])).values, index=w.index)


# ---------------------------------------------------------------------------
# Dam, siblings, nick, female family
# ---------------------------------------------------------------------------

def _sibling_stats(w: pd.DataFrame, value_col: str) -> tuple[pd.Series, pd.Series]:
    """(sum, count) over the dam's *other* progeny, on earlier days.

    The horse's own contribution is subtracted on the same day-lagged basis, so
    the subtraction is exact: its runs are a subset of its dam's."""
    dam_tot, dam_n = prior_stats(w, "_dam", value_col)
    own_tot, own_n = prior_stats(w, "_horse", value_col)
    return (dam_tot - own_tot), (dam_n - own_n).clip(lower=0.0)


def _fold_top2(state, val, member):
    v1, m1, v2, m2 = state
    if member == m1:
        return (max(v1, val), m1, v2, m2)
    if val > v1:
        return (val, member, v1, m1) if m1 is not None else (val, member, v2, m2)
    if member == m2:
        return (v1, m1, max(v2, val), m2)
    if val > v2:
        return (v1, m1, val, member)
    return state


def _prior_sibling_max(w: pd.DataFrame, group_col: str, member_col: str, value_col: str) -> pd.Series:
    """Max of `value_col` over the group's *other* members, on earlier days.

    A maximum cannot be de-contaminated by subtraction the way a sum can, so
    this keeps a running top two keyed on distinct members: the answer is the
    group's best unless the member holding it is this horse, in which case it
    is the runner-up. State is emitted before the day is folded in, which is
    what makes it strictly prior."""
    v = pd.to_numeric(w[value_col], errors="coerce")
    g = w[group_col].values
    known = w[group_col + "_known"].values
    ok = v.notna().values & known
    per = (pd.DataFrame({"g": g[ok], "day": w[CARD_KEY].values[ok], "m": w[member_col].values[ok], "v": v.values[ok]})
           .groupby(["g", "day", "m"], sort=False, observed=True)["v"].max().reset_index())
    ev = pd.DataFrame({"g": g, "day": w[CARD_KEY].values}).loc[known].drop_duplicates()
    ev["m"] = None; ev["v"] = np.nan; ev["_kind"] = 0
    per["_kind"] = 1
    stream = pd.concat([ev, per], ignore_index=True).sort_values(["g", "day", "_kind"], kind="stable")
    states, state, prev = [], (-np.inf, None, -np.inf, None), object()
    for grp, day, member, val, kind in zip(stream["g"].values, stream["day"].values, stream["m"].values,
                                           stream["v"].values, stream["_kind"].values):
        if grp != prev:
            state, prev = (-np.inf, None, -np.inf, None), grp
        if kind == 0:
            states.append((grp, day, state[0], state[1], state[2], state[3]))
        else:
            state = _fold_top2(state, val, member)
    if not states:
        return pd.Series(np.nan, index=w.index, dtype=float)
    st = pd.DataFrame(states, columns=["g", "day", "v1", "m1", "v2", "m2"])
    key = pd.DataFrame({"g": g, "day": w[CARD_KEY].values, "m": w[member_col].values})
    merged = key.merge(st, on=["g", "day"], how="left")
    pick = np.where(merged["m1"].values == merged["m"].values, merged["v2"].values, merged["v1"].values)
    pick = pd.to_numeric(pd.Series(pick), errors="coerce").values
    return pd.Series(np.where(np.isfinite(pick), pick, np.nan), index=w.index)


def _dam_own_record(w: pd.DataFrame) -> dict[str, pd.Series]:
    """The dam's own racing record as it stood the day before this race.

    A broodmare's career is over long before her progeny run, but the cut is
    made anyway rather than assumed: the join is an as-of join on her name."""
    per = (w.groupby(["_horse", CARD_KEY], observed=True)
             .agg(_n=("_one", "sum"), _nm=("_nmfp", "sum"), _nmn=("_nmfp", "count"),
                  _or=("_or", "max"), _bt=("_bt_placing", "sum"))
             .reset_index().sort_values(["_horse", CARD_KEY], kind="stable"))
    grp = per.groupby("_horse", observed=True)
    for col in ("_n", "_nm", "_nmn", "_bt"):
        per["c" + col] = grp[col].cumsum()
    per["c_or"] = grp["_or"].cummax()
    per["_date"] = pd.to_datetime(per[CARD_KEY])
    left = pd.DataFrame({"_k": w["_dam"].values, "_date": pd.to_datetime(w["race_date"]).values,
                         "_i": np.arange(len(w))}).sort_values("_date", kind="stable")
    right = (per.rename(columns={"_horse": "_k"})[["_k", "_date", "c_n", "c_nm", "c_nmn", "c_or", "c_bt"]]
             .sort_values("_date", kind="stable"))
    m = pd.merge_asof(left, right, on="_date", by="_k", allow_exact_matches=False).sort_values("_i")
    n = m["c_nmn"].values
    return {"dam_own_runs": pd.Series(m["c_n"].values, index=w.index),
            "dam_own_nmfp": pd.Series(m["c_nm"].values / np.where(n > 0, n, np.nan), index=w.index),
            "dam_own_best_or": pd.Series(m["c_or"].values, index=w.index),
            "dam_own_black_type": pd.Series(m["c_bt"].values, index=w.index)}


def _add_dam_block(w: pd.DataFrame, new: dict, pop: dict, k_dam: float, k_dam_sr: float,
                   k_family: float) -> list[str]:
    """§5.4. Dams have single-digit progeny counts, so shrinkage dominates and
    the sibling view is the practical route into dam quality."""
    known = w["_dam_known"]
    r_t, r_n = _sibling_stats(w, "_one")
    w_t, _ = _sibling_stats(w, "_won")
    wf_t, _ = _sibling_stats(w, "_win_first")
    nm_t, nm_n = _sibling_stats(w, "_nmfp")
    new["dam_runners"] = r_n.where(known)
    new["dam_winners"] = wf_t.where(known)                     # distinct progeny that have won
    new["dam_sr_shrunk"] = shrink(w_t, r_n, pop["sr"], k_dam_sr).where(known)
    new["sib_count"] = _sibling_stats(w, "_debut_flag")[0].where(known)
    new["sib_mean_nmfp"] = shrink(nm_t, nm_n, pop["nmfp"], k_dam).where(known)
    # §5.4: shrink the dam's progeny mean hard toward the damsire / family level
    family_prior = pd.Series(new.get("damsire_nmfp_shrunk", pop["nmfp"]), index=w.index).fillna(pop["nmfp"])
    new["dam_progeny_mean_nmfp"] = shrink(nm_t, nm_n, family_prior, k_family).where(known)
    new["dam_black_type_progeny"] = _sibling_stats(w, "_bt_first")[0].where(known)
    new["dam_best_progeny_rating"] = _prior_sibling_max(w, "_dam", "_horse", "_or").where(known)
    feats = ["dam_runners", "dam_winners", "dam_sr_shrunk", "sib_count", "sib_mean_nmfp",
             "dam_progeny_mean_nmfp", "dam_black_type_progeny", "dam_best_progeny_rating"]

    for name, col, prior in (("sib_dist_profile", "_win_dist", None), ("sib_going_profile", "_win_going", None),
                             ("sib_precocity", "_first_win_age", None), ("sib_improvement_curve", "_improvement", 0.0)):
        t, n = _sibling_stats(w, col)
        new[name] = ((t / n.replace(0, np.nan)) if prior is None else shrink(t, n, prior, 4.0)).where(known)
        feats.append(name)

    for name, series in _dam_own_record(w).items():
        new[name] = series.where(known)
        feats.append(name)
    return feats


def _add_nick_block(w: pd.DataFrame, new: dict, k_nick: float) -> list[str]:
    """§5.4: the sire × damsire cross, shrunk toward the sire's own mean.

    "Sample sizes are brutal here - most crosses have fewer than 20 runners -
    so this needs the hardest shrinkage in the whole suite and should be tested
    for incremental ΔR² before inclusion rather than assumed to work." Hence
    k_nick, the largest constant in the module, and nick_runs, so that the
    thinness stays visible to whatever consumes it."""
    known = w["_sire_known"] & w["_damsire_known"]
    w["_nick"] = w["_sire"] + "|" + w["_damsire"]
    t, n = prior_stats(w, "_nick", "_nmfp")
    won, won_n = prior_stats(w, "_nick", "_won")
    new["nick_runs"] = n.where(known)
    new["nick_apt"] = aptitude_residual(t, n, w["_sire_nmfp_raw"], k_nick).where(known)
    new["nick_sr_apt"] = aptitude_residual(won, won_n, w["_sire_sr_raw"], k_nick).where(known)
    return ["nick_runs", "nick_apt", "nick_sr_apt"]


def _add_family_block(w: pd.DataFrame, new: dict, pop: dict, k_family: float) -> list[str]:
    """§5.4's female family, as far as this schema reaches.

    The spec wants black-type density in the first three dams and a family class
    rating from the Blandford database. The feed carries the dam and the damsire
    and no further up the distaff line, so the observable family is the
    damsire's daughters' produce — one generation, not three. It is built from
    exactly the same constructions (black-type placing density, and the §5.2
    class rating) so that it can be swapped for a true broodmare-line key when
    one exists."""
    known = w["_damsire_known"]
    fam_t, fam_n = prior_stats(w, "_damsire", "_bt_placing")
    own_t, own_n = prior_stats(w, "_horse", "_bt_placing")
    t, n = fam_t - own_t, (fam_n - own_n).clip(lower=0.0)
    new["family_bt_density"] = shrink(t, n, pop["bt"], k_family).where(known)
    lp, lp_n = prior_stats(w, "_damsire", "_ln_prize")
    p2, p2_n = prior_stats(w, "_damsire", "_prb2")
    geo = np.exp(lp / lp_n.replace(0, np.nan))
    new["family_class_rating"] = class_rating(geo.values, shrink(p2, p2_n, pop["prb2_raw"], k_family).values,
                                              index=w.index).where(known)
    return ["family_bt_density", "family_class_rating"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def add_pedigree_features(df: pd.DataFrame, sire_col: str = "stallion", dam_col: str = "dam",
                          damsire_col: str = "dam_stallion", horse_col: str = "horse_name",
                          date_col: str = "race_date", nmfp_col: str = "nmfp",
                          k_sr: float = 40.0, k_nmfp: float = 25.0, k_apt: float = 25.0,
                          k_dam: float = 10.0, k_dam_sr: float = 25.0, k_family: float = 25.0,
                          k_nick: float = 60.0, k_lr: float = 50.0, min_hcap_debuts: int = 30,
                          sire: bool = True, damsire: bool = True, dam: bool = True,
                          nick: bool = True, family: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Attach the Part 5.3 - 5.4 pedigree suites; returns (frame, feature names).

    Shrinkage constants rise as the samples fall: the sire suite rests on
    thousands of progeny runs, the dam suite on single digits, and the nick on
    fewer still, which is why ``k_nick`` is the largest constant here. All of
    them are parameters — the framework fits them per family by out-of-sample
    likelihood — and every figure ships beside its own run count so a model can
    discount thin evidence further."""
    orig_index = df.index
    src = df.reset_index(drop=True)
    w = _working_frame(src, sire_col, dam_col, damsire_col, horse_col, date_col, nmfp_col)

    pop = {"sr": running_population_mean(w, "_won", fallback=0.1),
           "nmfp": running_population_mean(w, "_nmfp", fallback=0.0),
           "prb2": running_population_mean(w, "_prb2_fsa", fallback=0.0),
           "prb2_raw": running_population_mean(w, "_prb2", fallback=1.0 / 3.0),
           "elite": running_population_mean(w, "_elite", fallback=1.0 - ELITE_PRB),
           "bt": running_population_mean(w, "_bt_placing", fallback=0.0)}

    new: dict[str, pd.Series] = {}
    feats: list[str] = []
    if damsire:
        # §5.4: the damsire mirrors the sire's aptitude structure (stamina and
        # temperament rather than precocity), so no 2yo / debut cells here.
        feats += _add_sire_like(w, new, "_damsire", "damsire", pop, k_sr, k_nmfp, k_apt,
                                cells=("going", "dist", "surface"), masked=())
    if sire:
        feats += _add_sire_like(w, new, "_sire", "sire", pop, k_sr, k_nmfp, k_apt)
        feats += _add_sire_extras(w, new, "_sire", "sire", pop, k_sr, k_apt, k_lr, min_hcap_debuts)
    if dam:
        feats += _add_dam_block(w, new, pop, k_dam, k_dam_sr, k_family)
    if nick and sire:
        feats += _add_nick_block(w, new, k_nick)
    if family:
        feats += _add_family_block(w, new, pop, k_family)

    out = pd.concat([src.drop(columns=[c for c in new if c in src.columns]),
                     pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in new.items()}, index=src.index)], axis=1)
    out.index = orig_index
    return out, feats
