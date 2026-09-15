"""
Lag-safe, credibility-shrunk connection features (racing² framework Part 5.1 -
5.2, shrinkage per Part 7) — market-free by default, so legal in Stage F.

For each entity column (trainer, jockey, ...) and each run, using only runs of
that entity from *earlier days*:

    {e}_runs, {e}_wins            volume
    {e}_sr_shrunk                 strike rate shrunk toward the population rate
    {e}_place_rate_shrunk         place rate, same treatment
    {e}_nmfp_shrunk               mean NMFP shrunk toward the population mean
    {e}_prb2_fsa_shrunk           mean field-size-adjusted %RB², shrunk
    {e}_form_{w}d                 recency form: mean NMFP over the last w days
    {e}_runner_count_{w}d         runners in that window — what the form is worth
    {e}_form_vs_career            form over the shortest window − career mean
    {e}_runs_season,              this season (calendar year) so far, and the
    {e}_sr_season_shrunk,         completed previous season
    {e}_sr_last_season_shrunk
    {e}_{context}_apt             context splits, as credibility residuals
    {e}_{context}_n               how many runs the split rests on
    {e}_course_sr_shrunk          the course record (§5.2 jockey_course_record)
    {e}_lr_place_rating           class of runner: black-type placings weighted
                                  by the win% of each placing, per non-FTS run
    {e}_sr_expected,              §5.1's key correction: a 22% strike rate in
    {e}_sr_resid,                 Class 6 sellers is not stronger than 12% in
    {e}_class_geo,                Class 2 handicaps, so strike rate is scored
    {e}_sched_strength,           against what the *schedule entered* was worth
    {e}_stable_size, {e}_strength

Trainer × jockey pairing strike rate shrunk toward the trainer's own rate,
``jockey_booking_upgrade`` (today's jockey vs the horse's usual booking),
``jockey_rides_today_at_meeting``, ``jockey_claim_lb`` and ``is_claimer``.

With ``price_col`` (Stage C only — these are market-derived and must not reach
Stage F):

    {e}_ae_ratio                  actual wins / market-expected wins, shrunk to 1
    {e}_roi_bsp, {e}_roi_sp       profit per £1 level stake, shrunk to 0

Why the statistics lag by *day* and not by row
----------------------------------------------
A yard and a jockey can have several runners in one race and many more on one
card. ``shift(1)`` on such a key steps back to a rival in the same race, so the
"previous run" is a result that has not happened yet, and because these
features are otherwise flat across a race the leaked result is the only thing
that varies within it. Every aggregate here is built with
``model.lagsafe.race_lagged_expanding_*`` keyed on the day, which is the
convention the time-window helper below already used: a figure can only see
runs from strictly earlier days. Only ``horse_name`` is ever row-shifted,
because a horse runs once in a race.

k (the shrinkage constant) is fitted per family by out-of-sample likelihood in
the framework; here it is a parameter with sensible defaults (k = 30 for strike
rates, 15 for NMFP), and the run count is always carried so a model can
discount thin evidence further.

The helpers in the first section are the shared Part 5 machinery and are used
by ``model/pedigree.py`` (the sire / dam / damsire suites) as well.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_count, race_lagged_expanding_mean
from model.primitives import nmfp as _nmfp_primitive, prb, prb2_par

CARD_KEY = "_cardkey"

#: Overall win% of a horse whose last run was 1st / 2nd / 3rd. Used as the
#: placing weights of the §5.2 "LR place rating" (same constants as the LRP
#: block of model/custom_metrics.py — one source, one meaning).
LR_PLACE_WEIGHTS = (17.41, 14.77, 12.62)

POS_CANDIDATES = ("placing_numerical", "position", "placing")
N_CANDIDATES = ("number_of_runners", "n_runners", "field_size")


# ---------------------------------------------------------------------------
# Shared Part 5 machinery
# ---------------------------------------------------------------------------

def _first_col(d: pd.DataFrame, names) -> str | None:
    return next((c for c in names if c in d.columns), None)


def card_key(df: pd.DataFrame, date_col: str = "race_date") -> pd.Series:
    """The day a connection or pedigree statistic may not see into.

    Excluding the whole card rather than just the race costs a few hours of
    information and buys immunity from race-time strings that do not sort."""
    return pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("__nodate__")


def prior_stats(d: pd.DataFrame, keys, value_col: str, key_col: str = CARD_KEY,
                date_col: str = "race_date") -> tuple[pd.Series, pd.Series]:
    """(sum, count) of ``value_col`` over the group's runs on *earlier days*.

    Thin wrapper over ``model.lagsafe`` so that every statistic in Part 5 lags
    the same way. NaN values are ignored, so a column masked to a subset (NaN
    elsewhere) gives that subset's sum and count."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    sub = d.loc[:, list(dict.fromkeys(keys + [value_col]))].copy()
    sub["race_date"] = d[date_col]
    sub[key_col] = d[key_col] if key_col in d.columns else card_key(d, date_col)
    n = race_lagged_expanding_count(sub, keys, value_col, race_col=key_col).fillna(0.0)
    m = race_lagged_expanding_mean(sub, keys, value_col, race_col=key_col)
    return (m * n).fillna(0.0), n


def shrink(total, n, prior, k: float):
    """Credibility mean of Part 7: ``(Σx + k·prior) / (n + k)``."""
    return (total + k * prior) / (n + k)


def aptitude_residual(cell_total, cell_n, overall, k: float):
    """The cell's mean shrunk toward the entity's *own* overall level, then
    expressed as a residual against it:

        (Σx_cell + k·overall) / (n_cell + k) − overall
            ≡  n_cell / (n_cell + k) · (mean_cell − overall)

    Zero for an empty cell, which is what "no evidence of aptitude" should look
    like, and free of the entity's general quality — the part the market
    already prices (§5.3)."""
    return shrink(cell_total, cell_n, overall, k) - overall


def running_population_mean(d: pd.DataFrame, value_col: str, key_col: str = CARD_KEY,
                            date_col: str = "race_date", fallback: float | None = None) -> pd.Series:
    """Population mean of ``value_col`` over all earlier days — a lag-safe prior.

    The first day has nothing earlier and falls back to the frame mean (a level
    constant, not a per-row look-ahead)."""
    sub = d.loc[:, [value_col]].copy()
    sub["_pop"] = "all"
    sub["race_date"] = d[date_col]
    sub[key_col] = d[key_col] if key_col in d.columns else card_key(d, date_col)
    m = race_lagged_expanding_mean(sub, "_pop", value_col, race_col=key_col)
    if fallback is None:
        fallback = float(pd.to_numeric(d[value_col], errors="coerce").mean())
    return m.fillna(0.0 if fallback != fallback else fallback)


def running_z(d: pd.DataFrame, values: pd.Series, key_col: str = CARD_KEY,
              date_col: str = "race_date") -> pd.Series:
    """z-score against the population *as it stood on earlier days*.

    Standardising against the whole frame would let the composite below see the
    future; the running mean and SD (from E[x] and E[x²]) do not."""
    x = pd.to_numeric(values, errors="coerce")
    sub = pd.DataFrame({"_x": x, "_x2": x * x, "race_date": d[date_col].values},
                       index=d.index)
    if key_col in d.columns:
        sub[key_col] = d[key_col]
    m = running_population_mean(sub, "_x", key_col)
    m2 = running_population_mean(sub, "_x2", key_col)
    sd = np.sqrt(np.clip(m2 - m ** 2, 1e-12, None))
    return (x - m) / sd


def _text_blob(df: pd.DataFrame, cols) -> pd.Series:
    out = pd.Series("", index=df.index, dtype=object)
    for c in cols:
        if c in df.columns:
            out = out + " " + df[c].fillna("").astype(str)
    return out.str.lower()


BLACK_TYPE_RX = re.compile(r"\b(?:group|grade)\s*[123]\b|\bg[123]\b|\blisted\b|\bclass\s*1\b", re.I)
HANDICAP_RX = re.compile(r"handicap|\bhcap\b|\bh'cap\b", re.I)


def is_black_type(df: pd.DataFrame) -> pd.Series:
    """Group / Grade / Listed — and UK Class 1, which is black type by definition."""
    return _text_blob(df, ("major", "race_name", "race_class")).str.contains(BLACK_TYPE_RX).fillna(False)


def is_handicap(df: pd.DataFrame) -> pd.Series:
    return _text_blob(df, ("race_name", "race_type", "major", "race_class")).str.contains(HANDICAP_RX).fillna(False)


#: Firmness scale, firm high. Kept numeric so sibling/going profiles can be
#: averaged; the *groups* below are what the aptitude cells key on.
_GOING_RX = ((r"hard", 7.0), (r"good\s*(?:to|/|-)\s*firm|gd\s*-?\s*f|g/f", 5.0), (r"firm", 6.0),
             (r"good\s*(?:to|/|-)\s*soft|gd\s*-?\s*s|g/s", 3.0), (r"heavy", 1.0),
             (r"soft|yielding|holding", 2.0), (r"standard\s*(?:to|/|-)\s*slow", 3.5),
             (r"standard\s*(?:to|/|-)\s*fast", 4.5), (r"standard", 4.0), (r"good", 4.0))


def going_number(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for rx, v in _GOING_RX:
        m = out.isna() & t.str.contains(rx, regex=True)
        out[m] = v
    return out


def going_group(s: pd.Series) -> pd.Series:
    """Three cells, because per-description cells are too thin to shrink out of."""
    n = going_number(s)
    g = pd.Series(np.nan, index=s.index, dtype=object)
    g[n >= 5.0] = "firm"
    g[(n >= 3.5) & (n < 5.0)] = "good"
    g[n < 3.5] = "soft"
    return g


def dist_band(s: pd.Series, edges=(0.0, 6.5, 8.5, 10.5, 13.0, 40.0)) -> pd.Series:
    return pd.cut(pd.to_numeric(s, errors="coerce"), list(edges), labels=False)


AW_RX = re.compile(r"all.?weather|\baw\b|poly|tapeta|fibre|dirt|\bsand\b|standard", re.I)


def surface_is_aw(df: pd.DataFrame, surface_col: str = "surface_type",
                  going_col: str = "going_description") -> pd.Series:
    """1 on an artificial surface, 0 on turf, NaN when the frame cannot say."""
    blob = _text_blob(df, (surface_col, going_col, "race_code"))
    known = blob.str.strip().str.len() > 0
    return pd.Series(np.where(known, blob.str.contains(AW_RX).astype(float), np.nan), index=df.index)


def lr_place_score(d: pd.DataFrame, weights=LR_PLACE_WEIGHTS, pos_col: str | None = None) -> pd.Series:
    """Black-type placing weighted by the overall win% of that placing (§5.2)."""
    pos_col = pos_col or _first_col(d, POS_CANDIDATES)
    if pos_col is None:
        return pd.Series(np.nan, index=d.index, dtype=float)
    pos = pd.to_numeric(d[pos_col], errors="coerce")
    bt = is_black_type(d)
    out = pd.Series(0.0, index=d.index, dtype=float)
    out[pos.isna()] = np.nan
    for i, w in enumerate(weights, start=1):
        out[bt & (pos == i)] = float(w)
    return out


def lr_place_rating(d: pd.DataFrame, keys, score_col: str = "_lr_score", denom_col: str = "_non_fts",
                    k: float = 50.0, key_col: str = CARD_KEY) -> pd.Series:
    """§5.2's class-of-runner rating: black-type placings weighted by the win% of
    each placing, divided by runs excluding first-time starters.

    A measure of the class of horse an entity is associated with rather than of
    volume — it separates a jockey who rides good horses from one who rides a
    lot, and it is the construction §5.3 reuses for Sire Track Strength."""
    score, _ = prior_stats(d, keys, score_col, key_col)
    denom, _ = prior_stats(d, keys, denom_col, key_col)
    return score / (denom + k)


def first_time_out(d: pd.DataFrame) -> pd.Series:
    """1 on a horse's first career start. ``career_runs`` when the feed carries
    it (it counts runs before today, including runs before this frame starts),
    otherwise the horse's first appearance here."""
    if "career_runs" in d.columns:
        cr = pd.to_numeric(d["career_runs"], errors="coerce")
        if cr.notna().any():
            return (cr <= 0).astype(float).where(cr.notna())
    if "runs_count" in d.columns:
        rc = pd.to_numeric(d["runs_count"], errors="coerce")
        if rc.notna().any():
            return (rc <= 0).astype(float).where(rc.notna())
    if "horse_name" not in d.columns:
        return pd.Series(np.nan, index=d.index, dtype=float)
    order = d.sort_values(["race_date"] + (["race_time"] if "race_time" in d.columns else []), kind="stable")
    return (order.groupby("horse_name").cumcount() == 0).astype(float).reindex(d.index)


def career_run_index(d: pd.DataFrame) -> pd.Series:
    """0-based career run number (0 = debut), preferring the feed's own count."""
    if "career_runs" in d.columns:
        cr = pd.to_numeric(d["career_runs"], errors="coerce")
        if cr.notna().any():
            return cr
    order = d.sort_values(["race_date"] + (["race_time"] if "race_time" in d.columns else []), kind="stable")
    return order.groupby("horse_name").cumcount().reindex(d.index).astype(float)


def _time_window_prior_mean(df: pd.DataFrame, key: str, val_col: str, date_col: str,
                            window_days: int) -> tuple[pd.Series, pd.Series]:
    """Mean of val over prior rows of the same entity within window_days.

    Rows on the same day are excluded — the card has not been run yet."""
    out_mean = np.full(len(df), np.nan); out_n = np.zeros(len(df))
    d = df[[key, date_col, val_col]].reset_index(drop=True)
    d["_pos"] = np.arange(len(d))
    d = d.sort_values([key, date_col, "_pos"])
    keys = d[key].values
    dates = pd.to_datetime(d[date_col]).values.astype("datetime64[D]").astype(np.int64)
    vals = d[val_col].values.astype(float); pos = d["_pos"].values
    start = 0
    for i in range(len(d)):
        if i == 0 or keys[i] != keys[i - 1]:
            start = i
        lo = start
        while lo < i and dates[lo] < dates[i] - window_days:
            lo += 1
        j = i
        while j > lo and dates[j - 1] == dates[i]:
            j -= 1
        w = vals[lo:j]; w = w[~np.isnan(w)]
        out_n[pos[i]] = len(w)
        if len(w):
            out_mean[pos[i]] = w.mean()
    return pd.Series(out_mean, index=df.index), pd.Series(out_n, index=df.index)


def context_split_keys(d: pd.DataFrame) -> dict[str, pd.Series]:
    """The §5.1 trainer/jockey context splits this frame can support.

    "Each shrunk hard, these get thin fast" — so every split below is consumed
    as a credibility *residual* against the entity's own overall level, never as
    a raw conditional rate. Splits whose columns are absent are skipped rather
    than faked."""
    out: dict[str, pd.Series] = {}
    track_c = _first_col(d, ("track", "course", "course_bf"))
    if track_c:
        out["course"] = d[track_c].astype(str).where(d[track_c].notna())
    dist_c = _first_col(d, ("dist_furlongs", "distance_f"))
    if dist_c:
        out["dist"] = dist_band(d[dist_c])
    going_c = _first_col(d, ("going_description", "going"))
    if going_c:
        out["going"] = going_group(d[going_c])
    if "race_class" in d.columns:
        out["class"] = d["race_class"].astype(str).str.extract(r"(\d+)", expand=False)
    if "race_type" in d.columns:
        out["type"] = d["race_type"].astype(str).str.lower().where(d["race_type"].notna())
    age_c = _first_col(d, ("horse_age", "age"))
    if age_c:
        out["age"] = pd.to_numeric(d[age_c], errors="coerce").clip(upper=8.0)
    n_c = _first_col(d, N_CANDIDATES)
    if n_c:
        out["fieldsize"] = pd.cut(pd.to_numeric(d[n_c], errors="coerce"), [0, 7, 11, 15, 40], labels=False)
    fto = first_time_out(d)
    if fto.notna().any():
        out["fto"] = fto
    dslr_c = _first_col(d, ("days_since_lr", "dslr"))
    if dslr_c and "horse_name" in d.columns:
        days = pd.to_numeric(d[dslr_c], errors="coerce")
        brk = (days >= 90).astype(float).where(days.notna())
        out["off_break"] = brk
        order = d.sort_values(["race_date"] + (["race_time"] if "race_time" in d.columns else []), kind="stable")
        out["second_after_break"] = brk.reindex(order.index).groupby(order["horse_name"]).shift(1).reindex(d.index)
    if "headgear" in d.columns and "horse_name" in d.columns:
        hg = d["headgear"].fillna("").astype(str).str.strip().str.lower()
        order = d.sort_values(["race_date"] + (["race_time"] if "race_time" in d.columns else []), kind="stable")
        prev = hg.reindex(order.index).groupby(order["horse_name"]).shift(1).reindex(d.index)
        out["hg_change"] = (hg != prev).astype(float).where(prev.notna())
    if "jockeys_claim" in d.columns:
        claim = pd.to_numeric(d["jockeys_claim"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
        out["claim"] = (claim.fillna(0.0) > 0).astype(float)
    return out


def add_connection_features(df: pd.DataFrame, entities=("trainer", "jockey_name"), nmfp_col: str = "nmfp",
                            won_col: str = "won", date_col: str = "race_date", time_col: str = "race_time",
                            k_sr: float = 30.0, k_nmfp: float = 15.0, k_split: float = 20.0,
                            form_windows=(14, 30, 90), context_splits: bool = True, seasons: bool = True,
                            strength: bool = True, price_col: str | None = None,
                            sp_col: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    orig_index = df.index
    d = df.reset_index(drop=True).copy()
    d[CARD_KEY] = card_key(d, date_col)
    d["_one"] = 1.0
    pos_c, n_c = _first_col(d, POS_CANDIDATES), _first_col(d, N_CANDIDATES)
    if won_col in d.columns:
        d["_won"] = pd.to_numeric(d[won_col], errors="coerce").fillna(0.0)
    elif pos_c:
        d["_won"] = (pd.to_numeric(d[pos_c], errors="coerce") == 1).astype(float)
    else:
        raise KeyError(f"add_connection_features needs a '{won_col}' column or a finishing position")
    if nmfp_col in d.columns:
        d["_nmfp"] = pd.to_numeric(d[nmfp_col], errors="coerce")
    elif pos_c and n_c:
        d["_nmfp"] = _nmfp_primitive(pd.to_numeric(d[n_c], errors="coerce").values,
                                     pd.to_numeric(d[pos_c], errors="coerce").values)
    else:
        d["_nmfp"] = np.nan
    if pos_c and n_c:
        P = pd.to_numeric(d[pos_c], errors="coerce").values; N = pd.to_numeric(d[n_c], errors="coerce").values
        d["_prb2_fsa"] = prb(N, P) ** 2 - prb2_par(N)
        d["_placed"] = np.where(np.isnan(P), np.nan, (P <= 3).astype(float))
    else:
        d["_prb2_fsa"] = np.nan
        d["_placed"] = pd.to_numeric(d["placed"], errors="coerce") if "placed" in d.columns else np.nan
    d["_lr_score"] = lr_place_score(d, pos_col=pos_c)
    fto = first_time_out(d)
    d["_non_fts"] = (1.0 - fto.fillna(0.0)).where(d["_won"].notna())

    pop_sr = running_population_mean(d, "_won", date_col=date_col)
    pop_plc = running_population_mean(d, "_placed", date_col=date_col)
    pop_nmfp = running_population_mean(d, "_nmfp", date_col=date_col, fallback=0.0)
    pop_prb2 = running_population_mean(d, "_prb2_fsa", date_col=date_col, fallback=0.0)
    splits = context_split_keys(d) if context_splits else {}
    if strength:
        d["_sr_expected"] = _schedule_expected_sr(d, pop_sr, date_col)
        d["_field_strength"] = _field_strength(d)
        d["_ln_prize"] = np.log(pd.to_numeric(d["prize_money"], errors="coerce").where(lambda s: s > 0)) \
            if "prize_money" in d.columns else np.nan

    feats: list[str] = []
    for e in entities:
        if e not in d.columns:
            continue
        d[e] = d[e].fillna("__missing__").astype(str)
        p = e.split("_")[0]
        wins, runs = prior_stats(d, e, "_won", date_col=date_col)
        plc, plc_n = prior_stats(d, e, "_placed", date_col=date_col)
        nm, nm_n = prior_stats(d, e, "_nmfp", date_col=date_col)
        pr2, pr2_n = prior_stats(d, e, "_prb2_fsa", date_col=date_col)
        d[f"{p}_runs"] = runs
        d[f"{p}_wins"] = wins
        d[f"{p}_sr_shrunk"] = shrink(wins, runs, pop_sr, k_sr)
        d[f"{p}_place_rate_shrunk"] = shrink(plc, plc_n, pop_plc, k_sr)
        d[f"{p}_nmfp_shrunk"] = shrink(nm, nm_n, pop_nmfp, k_nmfp)
        d[f"{p}_prb2_fsa_shrunk"] = shrink(pr2, pr2_n, pop_prb2, k_nmfp)
        feats += [f"{p}_runs", f"{p}_sr_shrunk", f"{p}_nmfp_shrunk", f"{p}_place_rate_shrunk", f"{p}_prb2_fsa_shrunk"]

        for w in form_windows:
            m, n = _time_window_prior_mean(d, e, "_nmfp", date_col, w)
            d[f"{p}_form_{w}d"] = (m.fillna(0.0) * n) / (n + 6.0)   # shrink toward 0 by runner count
            d[f"{p}_runner_count_{w}d"] = n
            feats += [f"{p}_form_{w}d", f"{p}_runner_count_{w}d"]
        # §5.1: form_vs_career is the *short* window against the career mean
        d[f"{p}_form_vs_career"] = d[f"{p}_form_{min(form_windows)}d"] - d[f"{p}_nmfp_shrunk"]
        feats.append(f"{p}_form_vs_career")

        d[f"{p}_lr_place_rating"] = lr_place_rating(d, e)
        feats.append(f"{p}_lr_place_rating")

        if seasons:
            d["_season"] = pd.to_datetime(d[date_col], errors="coerce").dt.year
            s_w, s_n = prior_stats(d, [e, "_season"], "_won", date_col=date_col)
            d[f"{p}_runs_season"] = s_n
            d[f"{p}_sr_season_shrunk"] = shrink(s_w, s_n, d[f"{p}_sr_shrunk"], k_sr)
            lw, ln_ = _last_season_totals(d, e, "_season", "_won", "_one")
            d[f"{p}_sr_last_season_shrunk"] = shrink(lw, ln_, d[f"{p}_sr_shrunk"], k_sr)
            feats += [f"{p}_runs_season", f"{p}_sr_season_shrunk", f"{p}_sr_last_season_shrunk"]

        for name, key in splits.items():
            col = f"_split_{name}"
            d[col] = key.astype(str).where(key.notna())
            c_nm, c_n = prior_stats(d, [e, col], "_nmfp", date_col=date_col)
            d[f"{p}_{name}_apt"] = aptitude_residual(c_nm, c_n, d[f"{p}_nmfp_shrunk"], k_split).where(key.notna())
            d[f"{p}_{name}_n"] = c_n.where(key.notna())
            feats += [f"{p}_{name}_apt", f"{p}_{name}_n"]
            if name == "course":
                c_w, c_wn = prior_stats(d, [e, col], "_won", date_col=date_col)
                d[f"{p}_course_sr_shrunk"] = shrink(c_w, c_wn, d[f"{p}_sr_shrunk"], k_sr).where(key.notna())
                feats.append(f"{p}_course_sr_shrunk")
            d = d.drop(columns=[col])

        if strength:
            feats += _add_strength(d, e, p, runs, wins, k_sr, date_col)

        if price_col and price_col in d.columns:
            feats += _add_market_features(d, e, p, price_col, sp_col, date_col)

    if "trainer" in d.columns and "jockey_name" in d.columns:
        d["_tj"] = d["trainer"].astype(str) + "|" + d["jockey_name"].astype(str)
        tj_w, tj_n = prior_stats(d, "_tj", "_won", date_col=date_col)
        d["tj_runs"] = tj_n
        d["tj_sr_shrunk"] = shrink(tj_w, tj_n, d["trainer_sr_shrunk"], 20.0)  # toward the trainer's own rate
        feats += ["tj_runs", "tj_sr_shrunk"]
        if "horse_name" in d.columns:
            # booking upgrade: today's jockey vs the horse's usual jockey quality.
            # Grouping on the horse is the one safe row-shift: a horse runs once in a race.
            order = d.sort_values([date_col] + ([time_col] if time_col in d.columns else []), kind="stable")
            usual = (d["jockey_sr_shrunk"].reindex(order.index).groupby(order["horse_name"])
                     .transform(lambda s: s.shift(1).expanding().mean()).reindex(d.index))
            d["jockey_booking_upgrade"] = d["jockey_sr_shrunk"] - usual
            feats.append("jockey_booking_upgrade")
        d = d.drop(columns=["_tj"])

    track_c = _first_col(d, ("track", "course", "course_bf"))
    if "jockey_name" in d.columns and track_c:
        # declared rides, not results — known when the card comes out
        d["jockey_rides_today_at_meeting"] = (d.groupby(["jockey_name", track_c, CARD_KEY], observed=True)["_one"]
                                              .transform("size").astype(float))
        feats.append("jockey_rides_today_at_meeting")
    if "jockeys_claim" in d.columns:
        claim = pd.to_numeric(d["jockeys_claim"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").fillna(0.0)
        d["jockey_claim_lb"] = claim
        d["is_claimer"] = (claim > 0).astype(float)
        feats += ["jockey_claim_lb", "is_claimer"]

    d = d.drop(columns=[c for c in ("_won", "_nmfp", "_one", "_placed", "_prb2_fsa", "_lr_score", "_non_fts",
                                    "_season", "_sr_expected", "_field_strength", "_ln_prize", CARD_KEY)
                        if c in d.columns])
    d = d.sort_index()
    d.index = orig_index
    return d, feats


def _field_strength(d: pd.DataFrame) -> pd.Series:
    """How strong the opposition entered was: the race's rating level when the
    feed carries one, otherwise the field-size credibility weight √((N−1)/(N+1))."""
    for c in ("median_or", "max_or_in_race"):
        if c in d.columns:
            v = pd.to_numeric(d[c], errors="coerce")
            if v.notna().any():
                return v
    n_c = _first_col(d, N_CANDIDATES)
    if n_c:
        N = pd.to_numeric(d[n_c], errors="coerce")
        return np.sqrt(((N - 1) / (N + 1)).clip(lower=0.0))
    return pd.Series(np.nan, index=d.index, dtype=float)


def _schedule_expected_sr(d: pd.DataFrame, pop_sr: pd.Series, date_col: str = "race_date") -> pd.Series:
    """Population strike rate in the (class × field-size) cell this race sits in.

    §5.1: "a trainer with a 22% strike rate in Class 6 sellers is not stronger
    than one with 12% in Class 2 handicaps". This is what an average yard would
    have won from the same schedule, and it is what the strike rate is scored
    against."""
    cls = (d["race_class"].astype(str).str.extract(r"(\d+)", expand=False).fillna("?")
           if "race_class" in d.columns else pd.Series("?", index=d.index))
    n_c = _first_col(d, N_CANDIDATES)
    band = (pd.cut(pd.to_numeric(d[n_c], errors="coerce"), [0, 7, 11, 15, 40], labels=False).astype(str)
            if n_c else pd.Series("?", index=d.index))
    sub = pd.DataFrame({"_won": d["_won"], "race_date": d[date_col].values,
                        "_cell": cls.astype(str) + "|" + band.astype(str)}, index=d.index)
    sub[CARD_KEY] = d[CARD_KEY]
    tot, n = prior_stats(sub, "_cell", "_won")
    return shrink(tot, n, pop_sr, 200.0)


def _last_season_totals(d: pd.DataFrame, key: str, season_col: str, won_col: str,
                        one_col: str) -> tuple[pd.Series, pd.Series]:
    """(wins, runs) of the *completed* previous season — finished before this one
    began, so no lagging is needed beyond requiring the seasons to be adjacent."""
    agg = (d.groupby([key, season_col], observed=True)
             .agg(_w=(won_col, "sum"), _n=(one_col, "sum")).reset_index()
             .sort_values([key, season_col], kind="stable"))
    g = agg.groupby(key, observed=True)
    agg["_pw"] = g["_w"].shift(1); agg["_pn"] = g["_n"].shift(1); agg["_py"] = g[season_col].shift(1)
    stale = agg["_py"] != agg[season_col] - 1
    agg.loc[stale, ["_pw", "_pn"]] = np.nan
    src = pd.MultiIndex.from_arrays([agg[key], agg[season_col]])
    tgt = pd.MultiIndex.from_arrays([d[key], d[season_col]])
    w = pd.Series(pd.Series(agg["_pw"].values, index=src).reindex(tgt).values, index=d.index)
    n = pd.Series(pd.Series(agg["_pn"].values, index=src).reindex(tgt).values, index=d.index)
    return w.fillna(0.0), n.fillna(0.0)


def _add_strength(d: pd.DataFrame, e: str, p: str, runs: pd.Series, wins: pd.Series,
                  k_sr: float, date_col: str) -> list[str]:
    """§5.1 TrainerStrength, schedule-adjusted, and its components.

    The composite is an equal-ish weighting of *running* z-scores (standardised
    against the population as it stood on earlier days, so it cannot see the
    future). Components the frame cannot supply are dropped from the average
    rather than imputed."""
    exp_tot, _ = prior_stats(d, e, "_sr_expected", date_col=date_col)
    runs_nz = runs.replace(0, np.nan)
    d[f"{p}_sr_expected"] = exp_tot / runs_nz
    # excess wins per run over what the schedule was worth, credibility-shrunk
    d[f"{p}_sr_resid"] = (wins - exp_tot) / (runs + k_sr)
    lp, lp_n = prior_stats(d, e, "_ln_prize", date_col=date_col)
    d[f"{p}_class_geo"] = np.exp(lp / lp_n.replace(0, np.nan))     # geomean win prize money (WPMCV)
    fs, fs_n = prior_stats(d, e, "_field_strength", date_col=date_col)
    d[f"{p}_sched_strength"] = fs / fs_n.replace(0, np.nan)
    _, size_n = _time_window_prior_mean(d, e, "_one", date_col, 365)
    d[f"{p}_stable_size"] = size_n

    parts, wsum = [], 0.0
    for series, w in ((d[f"{p}_nmfp_shrunk"], 1.0), (d[f"{p}_prb2_fsa_shrunk"], 1.0),
                      (d[f"{p}_sr_resid"], 1.0), (np.log(d[f"{p}_class_geo"]), 0.5),
                      (d[f"{p}_sched_strength"], 0.5)):
        s = pd.Series(series, index=d.index)
        if not s.notna().any():
            continue
        parts.append(w * running_z(d, s).fillna(0.0)); wsum += w
    d[f"{p}_strength"] = (sum(parts) / wsum) if parts else np.nan
    return [f"{p}_sr_expected", f"{p}_sr_resid", f"{p}_class_geo", f"{p}_sched_strength",
            f"{p}_stable_size", f"{p}_strength"]


def _add_market_features(d: pd.DataFrame, e: str, p: str, price_col: str, sp_col: str | None,
                         date_col: str, k_ae: float = 20.0, k_roi: float = 50.0) -> list[str]:
    """§5.1 A/E and ROI. Market-derived: Stage C only, never Stage F."""
    feats: list[str] = []
    for col, tag in ((price_col, "bsp"), (sp_col, "sp")):
        if not col or col not in d.columns:
            continue
        price = pd.to_numeric(d[col], errors="coerce").where(lambda s: s > 1.0)
        d["_profit"] = pd.Series(np.where(d["_won"] > 0, price - 1.0, -1.0), index=d.index).where(price.notna())
        ptot, pn = prior_stats(d, e, "_profit", date_col=date_col)
        d[f"{p}_roi_{tag}"] = ptot / (pn + k_roi)          # profit per £1 level stake, shrunk to 0
        feats.append(f"{p}_roi_{tag}")
        if tag == "bsp":
            d["_pimp"] = 1.0 / price
            d["_won_priced"] = d["_won"].where(price.notna())
            exp_tot, _ = prior_stats(d, e, "_pimp", date_col=date_col)
            w_tot, _ = prior_stats(d, e, "_won_priced", date_col=date_col)
            d[f"{p}_ae_ratio"] = (w_tot + k_ae) / (exp_tot + k_ae)   # shrunk toward 1
            feats.append(f"{p}_ae_ratio")
    d.drop(columns=[c for c in ("_profit", "_pimp", "_won_priced") if c in d.columns], inplace=True)
    return feats
