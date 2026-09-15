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
from model.primitives import prb, prb2_par

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


def _order(d: pd.DataFrame, date_col: str = "race_date") -> pd.DataFrame:
    cols = [date_col] + (["race_time"] if "race_time" in d.columns else [])
    return d.sort_values(cols, kind="stable")


def _prepare(d: pd.DataFrame, sire_col: str, dam_col: str, damsire_col: str, horse_col: str,
             date_col: str, nmfp_col: str) -> pd.DataFrame:
    """Per-run value columns. Everything downstream is an aggregate of these."""
    pos_c, n_c = _first_col(d, POS_CANDIDATES), _first_col(d, N_CANDIDATES)
    P = pd.to_numeric(d[pos_c], errors="coerce").values if pos_c else np.full(len(d), np.nan)
    N = pd.to_numeric(d[n_c], errors="coerce").values if n_c else np.full(len(d), np.nan)
    d["_one"] = 1.0
    if "won" in d.columns:
        d["_won"] = pd.to_numeric(d["won"], errors="coerce")
    else:
        d["_won"] = np.where(np.isnan(P), np.nan, (P == 1).astype(float))
    d["_placed"] = np.where(np.isnan(P), np.nan, (P <= 3).astype(float))
    d["_prb"] = prb(N, P)
    d["_prb2"] = d["_prb"] ** 2
    d["_prb2_fsa"] = d["_prb2"] - prb2_par(N)
    d["_elite"] = np.where(np.isnan(d["_prb"].values), np.nan, (d["_prb"].values >= ELITE_PRB).astype(float))
    if nmfp_col in d.columns:
        d["_nmfp"] = pd.to_numeric(d[nmfp_col], errors="coerce")
    else:
        with np.errstate(invalid="ignore"):
            d["_nmfp"] = np.where(N > 1, (N + 1 - 2 * P) / np.sqrt(3 * (N * N - 1)), np.nan)
    d["_or"] = pd.to_numeric(d["official_rating"], errors="coerce") if "official_rating" in d.columns else np.nan
    d["_age"] = pd.to_numeric(d["horse_age"], errors="coerce") if "horse_age" in d.columns else np.nan
    dist_c = _first_col(d, ("dist_furlongs", "distance_f"))
    d["_dist"] = pd.to_numeric(d[dist_c], errors="coerce") if dist_c else np.nan
    going_c = _first_col(d, ("going_description", "going"))
    d["_going_num"] = going_number(d[going_c]) if going_c else np.nan
    d["_ln_prize"] = np.log(pd.to_numeric(d["prize_money"], errors="coerce").where(lambda s: s > 0)) \
        if "prize_money" in d.columns else np.nan
    d["_bt"] = is_black_type(d).astype(float)
    d["_bt_placing"] = np.where(np.isnan(P), np.nan, ((d["_bt"].values > 0) & (P <= 3)).astype(float))
    d["_lr_score"] = lr_place_score(d, pos_col=pos_c)
    fto = first_time_out(d)
    d["_fto"] = fto
    d["_non_fts"] = (1.0 - fto.fillna(0.0)).where(d["_one"].notna())

    # --- entity keys -------------------------------------------------------
    for col, name in ((sire_col, "_sire"), (dam_col, "_dam"), (damsire_col, "_damsire"), (horse_col, "_horse")):
        if col in d.columns:
            d[name] = _norm_name(d[col])
        else:
            d[name] = ""
        d[name + "_known"] = _known(d[name])
    d.loc[~d["_horse_known"], "_horse"] = "__row_" + d.index.astype(str)   # never merge unnamed horses

    # --- aptitude cells ----------------------------------------------------
    d["_cell_going"] = going_group(d[going_c]) if going_c else np.nan
    d["_cell_dist"] = dist_band(d["_dist"])
    aw = surface_is_aw(d)
    d["_cell_surface"] = aw
    track_c = _first_col(d, ("track", "course", "course_bf"))
    d["_cell_course"] = d[track_c].astype(str).where(d[track_c].notna()) if track_c else np.nan
    for c in ("_cell_going", "_cell_dist", "_cell_surface", "_cell_course"):
        d[c] = d[c].astype(object).where(d[c].notna())

    # --- masked per-run subsets (sire-level, not conditioned on today) ------
    d["_nmfp_2yo"] = d["_nmfp"].where(d["_age"] == 2)                       # precocity
    d["_nmfp_fto"] = d["_nmfp"].where(d["_fto"] > 0)                        # first time out

    # --- per-horse career markers (row-shift on the horse is the safe one) --
    o = _order(d)
    run_idx = pd.to_numeric(d["career_runs"], errors="coerce") if "career_runs" in d.columns else pd.Series(np.nan, index=d.index)
    if not run_idx.notna().any():
        run_idx = o.groupby("_horse").cumcount().reindex(d.index).astype(float)
    d["_run_idx"] = run_idx
    d["_debut_flag"] = (d["_run_idx"] == 0).astype(float)
    first_nmfp = d["_nmfp"].where(d["_run_idx"] == 0).groupby(d["_horse"]).transform("max")
    d["_improvement"] = (d["_nmfp"] - first_nmfp).where(d["_run_idx"] == 2)  # run 3 − run 1 (§5.3)
    prior_wins = (o.groupby("_horse")["_won"].cumsum().reindex(d.index) - d["_won"].fillna(0.0))
    first_win = (d["_won"] > 0) & (prior_wins.fillna(0.0) <= 0)
    d["_win_first"] = first_win.astype(float)
    d["_first_win_age"] = d["_age"].where(first_win)
    d["_win_dist"] = d["_dist"].where(d["_won"] > 0)
    d["_win_going"] = d["_going_num"].where(d["_won"] > 0)
    prior_bt = (o.groupby("_horse")["_bt_placing"].cumsum().reindex(d.index) - d["_bt_placing"].fillna(0.0))
    d["_bt_first"] = ((d["_bt_placing"] > 0) & (prior_bt.fillna(0.0) <= 0)).astype(float)
    hcap = is_handicap(d)
    h_order = o[hcap.reindex(o.index).fillna(False).astype(bool)]
    hcap_debut = pd.Series(False, index=d.index)
    if len(h_order):
        first_hcap = h_order.groupby("_horse").cumcount() == 0
        hcap_debut.loc[h_order.index[first_hcap.values]] = True
    d["_hcap_debut_or"] = d["_or"].where(hcap_debut)
    return d


# ---------------------------------------------------------------------------
# Sire / damsire: levels, then the aptitude residuals that matter
# ---------------------------------------------------------------------------

def _add_sire_like(d: pd.DataFrame, key: str, prefix: str, pop: dict, k_sr: float, k_nmfp: float,
                   k_apt: float, cells=("going", "dist", "surface", "course"),
                   masked=(("2yo", "_nmfp_2yo"), ("first_time_out", "_nmfp_fto"))) -> list[str]:
    known = d[key + "_known"]
    wins, runs = prior_stats(d, key, "_won")
    nm, nm_n = prior_stats(d, key, "_nmfp")
    pr2, pr2_n = prior_stats(d, key, "_prb2_fsa")
    # The *level* is shrunk toward the population; the residual base is the
    # entity's own raw mean. Comparing a raw cell mean against a shrunk overall
    # would leave the shrinkage gap in the residual and every cell of a strong
    # sire would look like an aptitude.
    overall = shrink(nm, nm_n, pop["nmfp"], k_nmfp)
    base = (nm / nm_n.replace(0, np.nan)).fillna(pd.Series(pop["nmfp"], index=d.index))
    d[f"_{prefix}_nmfp_raw"] = base
    d[f"_{prefix}_sr_raw"] = (wins / runs.replace(0, np.nan)).fillna(pd.Series(pop["sr"], index=d.index))
    d[f"{prefix}_prog_runs"] = runs.where(known)
    d[f"{prefix}_prog_wins"] = wins.where(known)
    d[f"{prefix}_sr_shrunk"] = shrink(wins, runs, pop["sr"], k_sr).where(known)
    d[f"{prefix}_nmfp_shrunk"] = overall.where(known)
    d[f"{prefix}_prb2_fsa_shrunk"] = shrink(pr2, pr2_n, pop["prb2"], k_nmfp).where(known)
    feats = [f"{prefix}_prog_runs", f"{prefix}_prog_wins", f"{prefix}_sr_shrunk",
             f"{prefix}_nmfp_shrunk", f"{prefix}_prb2_fsa_shrunk"]

    for name in cells:
        col = f"_cell_{name}"
        if col not in d.columns or not d[col].notna().any():
            continue
        c_tot, c_n = prior_stats(d, [key, col], "_nmfp")
        d[f"{prefix}_{name}_apt"] = aptitude_residual(c_tot, c_n, base, k_apt).where(known & d[col].notna())
        d[f"{prefix}_{name}_n"] = c_n.where(known & d[col].notna())
        feats += [f"{prefix}_{name}_apt", f"{prefix}_{name}_n"]

    for name, col in masked:
        if col not in d.columns:
            continue
        m_tot, m_n = prior_stats(d, key, col)
        d[f"{prefix}_{name}_apt"] = aptitude_residual(m_tot, m_n, base, k_apt).where(known)
        d[f"{prefix}_{name}_n"] = m_n.where(known)
        feats += [f"{prefix}_{name}_apt", f"{prefix}_{name}_n"]
    return feats


def _add_sire_extras(d: pd.DataFrame, key: str, prefix: str, pop: dict, k_sr: float,
                     k_apt: float, k_lr: float, min_hcap_debuts: int) -> list[str]:
    """The §5.3 composites: do the progeny progress, what class are they, how
    many are elite, and the two constructions taken from the internal research."""
    known = d[key + "_known"]
    imp_tot, imp_n = prior_stats(d, key, "_improvement")
    # a mean of (run 3 − run 1) NMFP: shrunk toward 0, i.e. toward "no progression"
    d[f"{prefix}_improvement"] = shrink(imp_tot, imp_n, 0.0, k_apt).where(known)

    el_tot, el_n = prior_stats(d, key, "_elite")
    d[f"{prefix}_elite_share"] = shrink(el_tot, el_n, pop["elite"], k_sr).where(known)

    lp, lp_n = prior_stats(d, key, "_ln_prize")
    geo = np.exp(lp / lp_n.replace(0, np.nan))                      # geomean WPMCV
    p2, p2_n = prior_stats(d, key, "_prb2")
    d[f"{prefix}_class_geo"] = geo.where(known)
    d[f"{prefix}_class_rating"] = class_rating(geo.values, shrink(p2, p2_n, pop["prb2_raw"], k_sr).values,
                                               index=d.index).where(known)
    # §5.3 Track Strength, built like the §5.2 LR place rating: black-type
    # placings weighted by the win% of each placing, per non-first-time starter.
    # Read as a sire-level scalar; the per-course view is sire_course_apt.
    d[f"{prefix}_track_strength"] = lr_place_rating(d, key, k=k_lr).where(known)
    d[f"{prefix}_median_hcap_debut_or"] = _median_hcap_debut_or(d, key, min_n=min_hcap_debuts).where(known)
    return [f"{prefix}_improvement", f"{prefix}_elite_share", f"{prefix}_class_geo",
            f"{prefix}_class_rating", f"{prefix}_track_strength", f"{prefix}_median_hcap_debut_or"]


def _median_hcap_debut_or(d: pd.DataFrame, key: str, years: float = 3.0, min_n: int = 30,
                          date_col: str = "race_date") -> pd.Series:
    """§5.3's Median Handicap Debut OR: the median mark the sire's progeny are
    given on their *handicap* debut, over a rolling `years` window, reported
    only once at least `min_n` of them qualify.

    Evaluated at the first day of each month against debutants strictly before
    that day, so it cannot see the card it is used on (and is at most a month
    stale — the metric moves on a three-year window, so that costs nothing).

    §X.10 records that our own 2016 test of this metric did not replicate the
    internal research's win-share claim and that a full rolling backtest is
    still owed. It ships as a feature to be tested, not as one to trust."""
    v = pd.to_numeric(d["_hcap_debut_or"], errors="coerce")
    days = pd.to_datetime(d[date_col], errors="coerce").values.astype("datetime64[D]").astype(np.int64)
    months = pd.to_datetime(d[date_col], errors="coerce").values.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)
    k = d[key].values
    ok = v.notna().values & d[key + "_known"].values
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
        return pd.Series(np.nan, index=d.index, dtype=float)
    idx = pd.MultiIndex.from_tuples(list(res.keys()))
    src = pd.Series(list(res.values()), index=idx)
    tgt = pd.MultiIndex.from_arrays([k, months])
    return pd.Series(src.reindex(tgt).values, index=d.index)


# ---------------------------------------------------------------------------
# Dam, siblings, nick, female family
# ---------------------------------------------------------------------------

def _sibling_stats(d: pd.DataFrame, value_col: str) -> tuple[pd.Series, pd.Series]:
    """(sum, count) over the dam's *other* progeny, on earlier days.

    The horse's own contribution is subtracted on the same day-lagged basis, so
    the subtraction is exact: its runs are a subset of its dam's."""
    dam_tot, dam_n = prior_stats(d, "_dam", value_col)
    own_tot, own_n = prior_stats(d, "_horse", value_col)
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


def _prior_sibling_max(d: pd.DataFrame, group_col: str, member_col: str, value_col: str,
                       day_col: str = CARD_KEY) -> pd.Series:
    """Max of `value_col` over the group's *other* members, on earlier days.

    A maximum cannot be de-contaminated by subtraction the way a sum can, so
    this keeps a running top two keyed on distinct members: the answer is the
    group's best unless the member holding it is this horse, in which case it
    is the runner-up. State is emitted before the day is folded in, which is
    what makes it strictly prior."""
    v = pd.to_numeric(d[value_col], errors="coerce")
    g = d[group_col].values
    ok = v.notna().values & d[group_col + "_known"].values
    per = (pd.DataFrame({"g": g[ok], "day": d[day_col].values[ok], "m": d[member_col].values[ok], "v": v.values[ok]})
           .groupby(["g", "day", "m"], sort=False, observed=True)["v"].max().reset_index())
    ev = pd.DataFrame({"g": g, "day": d[day_col].values}).loc[d[group_col + "_known"].values].drop_duplicates()
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
        return pd.Series(np.nan, index=d.index, dtype=float)
    st = pd.DataFrame(states, columns=["g", "day", "v1", "m1", "v2", "m2"])
    key = pd.DataFrame({"g": g, "day": d[day_col].values, "m": d[member_col].values})
    merged = key.merge(st, on=["g", "day"], how="left")
    pick = np.where(merged["m1"].values == merged["m"].values, merged["v2"].values, merged["v1"].values)
    pick = pd.to_numeric(pd.Series(pick), errors="coerce").values
    return pd.Series(np.where(np.isfinite(pick), pick, np.nan), index=d.index)


def _dam_own_record(d: pd.DataFrame, date_col: str = "race_date") -> pd.DataFrame:
    """The dam's own racing record as it stood the day before this race.

    A broodmare's career is over long before her progeny run, but the cut is
    made anyway rather than assumed: the join is an as-of join on her name."""
    per = (d.groupby(["_horse", CARD_KEY], observed=True)
             .agg(_n=("_one", "sum"), _nm=("_nmfp", "sum"), _nmn=("_nmfp", "count"),
                  _or=("_or", "max"), _bt=("_bt_placing", "sum"))
             .reset_index().sort_values(["_horse", CARD_KEY], kind="stable"))
    grp = per.groupby("_horse", observed=True)
    for c in ("_n", "_nm", "_nmn", "_bt"):
        per["c" + c] = grp[c].cumsum()
    per["c_or"] = grp["_or"].cummax()
    per["_date"] = pd.to_datetime(per[CARD_KEY])
    left = pd.DataFrame({"_k": d["_dam"].values, "_date": pd.to_datetime(d[date_col]).values,
                         "_i": np.arange(len(d))}).sort_values("_date", kind="stable")
    right = (per.rename(columns={"_horse": "_k"})[["_k", "_date", "c_n", "c_nm", "c_nmn", "c_or", "c_bt"]]
             .sort_values("_date", kind="stable"))
    m = pd.merge_asof(left, right, on="_date", by="_k", allow_exact_matches=False).sort_values("_i")
    out = pd.DataFrame(index=d.index)
    out["dam_own_runs"] = m["c_n"].values
    out["dam_own_nmfp"] = (m["c_nm"].values / np.where(m["c_nmn"].values > 0, m["c_nmn"].values, np.nan))
    out["dam_own_best_or"] = m["c_or"].values
    out["dam_own_black_type"] = m["c_bt"].values
    return out


def _add_dam_block(d: pd.DataFrame, pop: dict, k_dam: float, k_dam_sr: float, k_family: float) -> list[str]:
    """§5.4. Dams have single-digit progeny counts, so shrinkage dominates and
    the sibling view is the practical route into dam quality."""
    known = d["_dam_known"]
    r_t, r_n = _sibling_stats(d, "_one")
    w_t, _ = _sibling_stats(d, "_won")
    wf_t, _ = _sibling_stats(d, "_win_first")
    nm_t, nm_n = _sibling_stats(d, "_nmfp")
    d["dam_runners"] = r_n.where(known)
    d["dam_winners"] = wf_t.where(known)                    # distinct progeny that have won
    d["dam_sr_shrunk"] = shrink(w_t, r_n, pop["sr"], k_dam_sr).where(known)
    d["sib_count"] = _sibling_stats(d, "_debut_flag")[0].where(known)
    d["sib_mean_nmfp"] = shrink(nm_t, nm_n, pop["nmfp"], k_dam).where(known)
    # §5.4: shrink the dam's progeny mean hard toward the damsire / family level
    family_prior = d["damsire_nmfp_shrunk"] if "damsire_nmfp_shrunk" in d.columns else pop["nmfp"]
    d["dam_progeny_mean_nmfp"] = shrink(nm_t, nm_n, pd.Series(family_prior, index=d.index).fillna(pop["nmfp"]),
                                        k_family).where(known)
    d["dam_black_type_progeny"] = _sibling_stats(d, "_bt_first")[0].where(known)
    d["dam_best_progeny_rating"] = _prior_sibling_max(d, "_dam", "_horse", "_or").where(known)
    feats = ["dam_runners", "dam_winners", "dam_sr_shrunk", "sib_count", "sib_mean_nmfp",
             "dam_progeny_mean_nmfp", "dam_black_type_progeny", "dam_best_progeny_rating"]

    for name, col, prior in (("sib_dist_profile", "_win_dist", None), ("sib_going_profile", "_win_going", None),
                             ("sib_precocity", "_first_win_age", None), ("sib_improvement_curve", "_improvement", 0.0)):
        t, n = _sibling_stats(d, col)
        if prior is None:
            d[name] = (t / n.replace(0, np.nan)).where(known)
        else:
            d[name] = shrink(t, n, prior, 4.0).where(known)
        feats.append(name)

    own = _dam_own_record(d)
    for c in own.columns:
        d[c] = own[c].where(known)
    feats += list(own.columns)
    return feats


def _add_nick_block(d: pd.DataFrame, k_nick: float) -> list[str]:
    """§5.4: the sire × damsire cross, shrunk toward the sire's own mean.

    "Sample sizes are brutal here - most crosses have fewer than 20 runners -
    so this needs the hardest shrinkage in the whole suite and should be tested
    for incremental ΔR² before inclusion rather than assumed to work." Hence
    k_nick, which is the largest constant in the module, and nick_runs, so the
    thinness is visible to whatever consumes it."""
    known = d["_sire_known"] & d["_damsire_known"]
    d["_nick"] = d["_sire"] + "|" + d["_damsire"]
    t, n = prior_stats(d, "_nick", "_nmfp")
    w, wn = prior_stats(d, "_nick", "_won")
    d["nick_runs"] = n.where(known)
    d["nick_apt"] = aptitude_residual(t, n, d["_sire_nmfp_raw"], k_nick).where(known)
    d["nick_sr_apt"] = aptitude_residual(w, wn, d["_sire_sr_raw"], k_nick).where(known)
    return ["nick_runs", "nick_apt", "nick_sr_apt"]


def _add_family_block(d: pd.DataFrame, pop: dict, k_family: float) -> list[str]:
    """§5.4's female family, as far as this schema reaches.

    The spec wants black-type density in the first three dams and a family class
    rating from the Blandford database. The feed carries the dam and the damsire
    and no further up the distaff line, so the observable family is the damsire's
    daughters' produce — one generation, not three. It is built on exactly the
    same constructions (black-type placing density, and the §5.2 class rating)
    so that it can be swapped for a true broodmare-line key when one exists."""
    known = d["_damsire_known"]
    fam_t, fam_n = prior_stats(d, "_damsire", "_bt_placing")
    own_t, own_n = prior_stats(d, "_horse", "_bt_placing")
    t, n = fam_t - own_t, (fam_n - own_n).clip(lower=0.0)
    d["family_bt_density"] = shrink(t, n, pop["bt"], k_family).where(known)
    lp, lp_n = prior_stats(d, "_damsire", "_ln_prize")
    p2, p2_n = prior_stats(d, "_damsire", "_prb2")
    geo = np.exp(lp / lp_n.replace(0, np.nan))
    d["family_class_rating"] = class_rating(geo.values, shrink(p2, p2_n, pop["prb2_raw"], k_family).values,
                                            index=d.index).where(known)
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
    """Attach the Part 5.3 - 5.4 pedigree suites and return (frame, feature names).

    Shrinkage constants rise as the samples fall: the sire suite is fitted on
    thousands of progeny runs, the dam suite on single digits, and the nick on
    fewer still, which is why ``k_nick`` is the largest constant here. All of
    them are parameters — the framework fits them per family by out-of-sample
    likelihood — and every figure ships beside its own run count so a model can
    discount thin evidence further."""
    pre = set(df.columns)
    orig_index = df.index
    d = df.reset_index(drop=True).copy()
    d[CARD_KEY] = card_key(d, date_col)
    d = _prepare(d, sire_col, dam_col, damsire_col, horse_col, date_col, nmfp_col)

    pop = {"sr": running_population_mean(d, "_won", date_col=date_col, fallback=0.1),
           "nmfp": running_population_mean(d, "_nmfp", date_col=date_col, fallback=0.0),
           "prb2": running_population_mean(d, "_prb2_fsa", date_col=date_col, fallback=0.0),
           "prb2_raw": running_population_mean(d, "_prb2", date_col=date_col, fallback=1.0 / 3.0),
           "elite": running_population_mean(d, "_elite", date_col=date_col, fallback=1.0 - ELITE_PRB),
           "bt": running_population_mean(d, "_bt_placing", date_col=date_col, fallback=0.0)}

    feats: list[str] = []
    if damsire:
        # §5.4: the damsire mirrors the sire's aptitude structure (stamina and
        # temperament rather than precocity), so no 2yo / debut cells here.
        feats += _add_sire_like(d, "_damsire", "damsire", pop, k_sr, k_nmfp, k_apt,
                                cells=("going", "dist", "surface"), masked=())
    if sire:
        feats += _add_sire_like(d, "_sire", "sire", pop, k_sr, k_nmfp, k_apt)
        feats += _add_sire_extras(d, "_sire", "sire", pop, k_sr, k_apt, k_lr, min_hcap_debuts)
    if dam:
        feats += _add_dam_block(d, pop, k_dam, k_dam_sr, k_family)
    if nick and sire:
        feats += _add_nick_block(d, k_nick)
    if family:
        feats += _add_family_block(d, pop, k_family)

    d = d.drop(columns=[c for c in d.columns if c.startswith("_") and c not in pre])
    d = d.sort_index()
    d.index = orig_index
    return d, feats
