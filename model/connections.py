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
    {e}_lto_plc_fsa_{w}d          recent field-size-adjusted place rate
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


def _narrow(d: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """A small frame built from Series, so a wide caller frame is never copied.

    ``d.loc[:, cols].copy()`` consolidates the whole block manager; on the
    hundred-odd column frames these features are built on that dominates the
    cost of every aggregate."""
    return pd.DataFrame(cols, index=d.index, copy=False)


def prior_stats(d: pd.DataFrame, keys, value_col: str, key_col: str = CARD_KEY,
                date_col: str = "race_date") -> tuple[pd.Series, pd.Series]:
    """(sum, count) of ``value_col`` over the group's runs on *earlier days*.

    Thin wrapper over ``model.lagsafe`` so that every statistic in Part 5 lags
    the same way. NaN values are ignored, so a column masked to a subset (NaN
    elsewhere) gives that subset's sum and count."""
    keys = [keys] if isinstance(keys, str) else list(keys)
    cols = {k: d[k] for k in dict.fromkeys(keys)}
    cols[value_col] = d[value_col]
    cols["race_date"] = d[date_col]
    cols[key_col] = d[key_col] if key_col in d.columns else card_key(d, date_col)
    sub = _narrow(d, cols)
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
    already prices (§5.3).

    ``overall`` must be the entity's *raw* mean, not its shrunk level: against a
    shrunk level the residual would carry the shrinkage gap, and every cell of
    a strong sire or a strong yard would read as an aptitude."""
    return shrink(cell_total, cell_n, overall, k) - overall


def running_population_mean(d: pd.DataFrame, value_col: str, key_col: str = CARD_KEY,
                            date_col: str = "race_date", fallback: float | None = None) -> pd.Series:
    """Population mean of ``value_col`` over all earlier days — a lag-safe prior.

    The first day has nothing earlier and falls back to the frame mean (a level
    constant, not a per-row look-ahead)."""
    sub = _narrow(d, {value_col: d[value_col], "_pop": "all", "race_date": d[date_col],
                      key_col: d[key_col] if key_col in d.columns else card_key(d, date_col)})
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
    sub = _narrow(d, {"_x": x, "_x2": x * x, "race_date": d[date_col],
                      key_col: d[key_col] if key_col in d.columns else card_key(d, date_col)})
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




def _working_frame(src: pd.DataFrame, nmfp_col: str, won_col: str, date_col: str,
                   context_splits, strength: bool, price_col: str | None,
                   sp_col: str | None) -> pd.DataFrame:
    """Everything the aggregates need and nothing else.

    A dozen columns instead of the caller's two hundred: every aggregate below
    groups, sorts and merges this frame, and doing that against a wide frame is
    what makes a feature build take hours instead of minutes."""
    pos_c, n_c = _first_col(src, POS_CANDIDATES), _first_col(src, N_CANDIDATES)
    cols: dict = {"race_date": src[date_col], CARD_KEY: card_key(src, date_col),
                  "_one": pd.Series(1.0, index=src.index)}
    if won_col in src.columns:
        cols["_won"] = pd.to_numeric(src[won_col], errors="coerce").fillna(0.0)
    elif pos_c:
        cols["_won"] = (pd.to_numeric(src[pos_c], errors="coerce") == 1).astype(float)
    else:
        raise KeyError(f"add_connection_features needs a '{won_col}' column or a finishing position")
    P = pd.to_numeric(src[pos_c], errors="coerce").values if pos_c else np.full(len(src), np.nan)
    N = pd.to_numeric(src[n_c], errors="coerce").values if n_c else np.full(len(src), np.nan)
    if nmfp_col in src.columns:
        cols["_nmfp"] = pd.to_numeric(src[nmfp_col], errors="coerce")
    elif pos_c and n_c:
        cols["_nmfp"] = pd.Series(_nmfp_primitive(N, P), index=src.index)
    else:
        cols["_nmfp"] = np.nan
    if pos_c and n_c:
        cols["_prb2_fsa"] = pd.Series(prb(N, P) ** 2 - prb2_par(N), index=src.index)
        placed = pd.Series(np.where(np.isnan(P), np.nan, (P <= 3).astype(float)), index=src.index)
        cols["_placed"] = placed
        # place rate net of what the field size gives away (§5.1 trainer_lto_plc_fsa)
        cols["_plc_fsa"] = placed - 3.0 / N
    else:
        cols["_prb2_fsa"] = np.nan
        cols["_placed"] = pd.to_numeric(src["placed"], errors="coerce") if "placed" in src.columns else np.nan
        cols["_plc_fsa"] = np.nan
    cols["_lr_score"] = lr_place_score(src, pos_col=pos_c)
    cols["_non_fts"] = 1.0 - first_time_out(src).fillna(0.0)
    cols["_season"] = pd.to_datetime(src[date_col], errors="coerce").dt.year
    if strength:
        cols["_field_strength"] = _field_strength(src)
        cols["_ln_prize"] = (np.log(pd.to_numeric(src["prize_money"], errors="coerce").where(lambda s: s > 0))
                             if "prize_money" in src.columns else np.nan)
        cls = (src["race_class"].astype(str).str.extract(r"(\d+)", expand=False).fillna("?")
               if "race_class" in src.columns else pd.Series("?", index=src.index))
        band = (pd.cut(pd.to_numeric(src[n_c], errors="coerce"), [0, 7, 11, 15, 40], labels=False).astype(str)
                if n_c else pd.Series("?", index=src.index))
        cols["_sched_cell"] = cls.astype(str) + "|" + band.astype(str)
    for col, tag in ((price_col, "_price"), (sp_col, "_sp")):
        if col and col in src.columns:
            cols[tag] = pd.to_numeric(src[col], errors="coerce").where(lambda s: s > 1.0)
    if context_splits is not False:
        wanted = None if context_splits is True else set(context_splits)
        for name, key in context_split_keys(src).items():
            if wanted is None or name in wanted:
                cols[f"_split_{name}"] = key.astype(str).where(key.notna())
    return pd.DataFrame(cols, index=src.index)


def add_connection_features(df: pd.DataFrame, entities=("trainer", "jockey_name"), nmfp_col: str = "nmfp",
                            won_col: str = "won", date_col: str = "race_date", time_col: str = "race_time",
                            k_sr: float = 30.0, k_nmfp: float = 15.0, k_split: float = 20.0,
                            form_windows=(14, 30, 90), context_splits=True, seasons: bool = True,
                            strength: bool = True, price_col: str | None = None,
                            sp_col: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Attach the Part 5.1 - 5.2 connection suites; returns (frame, feature names).

    ``price_col`` / ``sp_col`` switch on the A/E and ROI block, which is
    market-derived and therefore Stage C only — leave them unset for Stage F.

    ``context_splits`` takes True (every split the frame supports), False, or an
    explicit list of split names — see ``context_split_keys``. Each split costs
    one lag-safe pass per entity, which is the bulk of the build time on a large
    frame, so a caller that only wants some of them can say so."""
    orig_index = df.index
    src = df.reset_index(drop=True)
    w = _working_frame(src, nmfp_col, won_col, date_col, context_splits, strength, price_col, sp_col)
    if strength:
        w["_sr_expected"] = _schedule_expected_sr(w, running_population_mean(w, "_won"))
    splits = [c[len("_split_"):] for c in w.columns if c.startswith("_split_")]

    pop_sr = running_population_mean(w, "_won")
    pop_plc = running_population_mean(w, "_placed")
    pop_nmfp = running_population_mean(w, "_nmfp", fallback=0.0)
    pop_prb2 = running_population_mean(w, "_prb2_fsa", fallback=0.0)

    new: dict[str, pd.Series] = {}
    feats: list[str] = []
    for e in entities:
        if e not in src.columns:
            continue
        w[e] = src[e].fillna("__missing__").astype(str)
        p = e.split("_")[0]
        wins, runs = prior_stats(w, e, "_won")
        plc, plc_n = prior_stats(w, e, "_placed")
        nm, nm_n = prior_stats(w, e, "_nmfp")
        pr2, pr2_n = prior_stats(w, e, "_prb2_fsa")
        sr = shrink(wins, runs, pop_sr, k_sr)
        nmfp_shrunk = shrink(nm, nm_n, pop_nmfp, k_nmfp)
        base_nmfp = (nm / runs.replace(0, np.nan)).fillna(pop_nmfp)      # the residual base
        new[f"{p}_runs"] = runs
        new[f"{p}_wins"] = wins
        new[f"{p}_sr_shrunk"] = sr
        new[f"{p}_place_rate_shrunk"] = shrink(plc, plc_n, pop_plc, k_sr)
        new[f"{p}_nmfp_shrunk"] = nmfp_shrunk
        new[f"{p}_prb2_fsa_shrunk"] = shrink(pr2, pr2_n, pop_prb2, k_nmfp)
        feats += [f"{p}_runs", f"{p}_wins", f"{p}_sr_shrunk", f"{p}_place_rate_shrunk", f"{p}_nmfp_shrunk",
                  f"{p}_prb2_fsa_shrunk"]

        for win in form_windows:
            m, n = _time_window_prior_mean(w, e, "_nmfp", "race_date", win)
            new[f"{p}_form_{win}d"] = (m.fillna(0.0) * n) / (n + 6.0)    # shrink toward 0 by runner count
            new[f"{p}_runner_count_{win}d"] = n
            feats += [f"{p}_form_{win}d", f"{p}_runner_count_{win}d"]
        short = min(form_windows)
        # §5.1: form_vs_career is the *short* window against the career mean
        new[f"{p}_form_vs_career"] = new[f"{p}_form_{short}d"] - nmfp_shrunk
        m, n = _time_window_prior_mean(w, e, "_plc_fsa", "race_date", short)
        new[f"{p}_lto_plc_fsa_{short}d"] = (m.fillna(0.0) * n) / (n + 6.0)
        new[f"{p}_lr_place_rating"] = lr_place_rating(w, e)
        feats += [f"{p}_form_vs_career", f"{p}_lto_plc_fsa_{short}d", f"{p}_lr_place_rating"]

        if seasons:
            s_w, s_n = prior_stats(w, [e, "_season"], "_won")
            new[f"{p}_runs_season"] = s_n
            new[f"{p}_sr_season_shrunk"] = shrink(s_w, s_n, sr, k_sr)
            lw, ln_ = _last_season_totals(w, e, "_season", "_won", "_one")
            new[f"{p}_sr_last_season_shrunk"] = shrink(lw, ln_, sr, k_sr)
            feats += [f"{p}_runs_season", f"{p}_sr_season_shrunk", f"{p}_sr_last_season_shrunk"]

        for name in splits:
            col = f"_split_{name}"
            here = w[col].notna()
            c_nm, c_n = prior_stats(w, [e, col], "_nmfp")
            new[f"{p}_{name}_apt"] = aptitude_residual(c_nm, c_n, base_nmfp, k_split).where(here)
            new[f"{p}_{name}_n"] = c_n.where(here)
            feats += [f"{p}_{name}_apt", f"{p}_{name}_n"]
            if name == "course":     # §5.2 jockey_course_record, shrunk to the overall rate
                c_w, c_wn = prior_stats(w, [e, col], "_won")
                new[f"{p}_course_sr_shrunk"] = shrink(c_w, c_wn, sr, k_sr).where(here)
                feats.append(f"{p}_course_sr_shrunk")

        if strength:
            feats += _add_strength(w, new, e, p, runs, wins, nmfp_shrunk, k_sr)
        if "_price" in w.columns or "_sp" in w.columns:
            feats += _add_market_features(w, new, e, p)

    if {"trainer", "jockey_name"} <= set(w.columns):
        w["_tj"] = w["trainer"] + "|" + w["jockey_name"]
        tj_w, tj_n = prior_stats(w, "_tj", "_won")
        new["tj_runs"] = tj_n
        new["tj_sr_shrunk"] = shrink(tj_w, tj_n, new["trainer_sr_shrunk"], 20.0)  # toward the trainer's rate
        feats += ["tj_runs", "tj_sr_shrunk"]
        if "horse_name" in src.columns:
            # booking upgrade: today's jockey against the horse's usual booking.
            # Grouping on the horse is the one safe row-shift: it runs once in a race.
            key = src["horse_name"].astype(str)
            order = w.sort_values(["race_date"] + ([time_col] if time_col in w.columns else []), kind="stable").index
            usual = (new["jockey_sr_shrunk"].reindex(order).groupby(key.reindex(order))
                     .transform(lambda s: s.shift(1).expanding().mean()).reindex(w.index))
            new["jockey_booking_upgrade"] = new["jockey_sr_shrunk"] - usual
            feats.append("jockey_booking_upgrade")

    track_c = _first_col(src, ("track", "course", "course_bf"))
    if "jockey_name" in w.columns and track_c:
        # declared rides, not results — known as soon as the card comes out
        rides = _narrow(w, {"_j": w["jockey_name"], "_t": src[track_c].astype(str), "_d": w[CARD_KEY],
                            "_one": w["_one"]})
        new["jockey_rides_today_at_meeting"] = rides.groupby(["_j", "_t", "_d"], observed=True)["_one"].transform("size").astype(float)
        feats.append("jockey_rides_today_at_meeting")
    if "jockeys_claim" in src.columns:
        claim = pd.to_numeric(src["jockeys_claim"].astype(str).str.extract(r"(\d+)", expand=False),
                              errors="coerce").fillna(0.0)
        new["jockey_claim_lb"] = claim
        new["is_claimer"] = (claim > 0).astype(float)
        feats += ["jockey_claim_lb", "is_claimer"]

    out = pd.concat([src.drop(columns=[c for c in new if c in src.columns]),
                     pd.DataFrame({k: np.asarray(v) for k, v in new.items()}, index=src.index)], axis=1)
    out.index = orig_index
    return out, feats


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


def _schedule_expected_sr(w: pd.DataFrame, pop_sr: pd.Series) -> pd.Series:
    """Population strike rate in the (class × field-size) cell this race sits in.

    §5.1: "a trainer with a 22% strike rate in Class 6 sellers is not stronger
    than one with 12% in Class 2 handicaps". This is what an average yard would
    have won from the same schedule, and it is what the strike rate is scored
    against."""
    tot, n = prior_stats(w, "_sched_cell", "_won")
    return shrink(tot, n, pop_sr, 200.0)


def _last_season_totals(w: pd.DataFrame, key: str, season_col: str, won_col: str,
                        one_col: str) -> tuple[pd.Series, pd.Series]:
    """(wins, runs) of the *completed* previous season — finished before this one
    began, so no lagging is needed beyond requiring the seasons to be adjacent."""
    agg = (w.groupby([key, season_col], observed=True)
             .agg(_w=(won_col, "sum"), _n=(one_col, "sum")).reset_index()
             .sort_values([key, season_col], kind="stable"))
    g = agg.groupby(key, observed=True)
    agg["_pw"] = g["_w"].shift(1); agg["_pn"] = g["_n"].shift(1); agg["_py"] = g[season_col].shift(1)
    agg.loc[agg["_py"] != agg[season_col] - 1, ["_pw", "_pn"]] = np.nan
    src = pd.MultiIndex.from_arrays([agg[key], agg[season_col]])
    tgt = pd.MultiIndex.from_arrays([w[key], w[season_col]])
    wins = pd.Series(pd.Series(agg["_pw"].values, index=src).reindex(tgt).values, index=w.index)
    runs = pd.Series(pd.Series(agg["_pn"].values, index=src).reindex(tgt).values, index=w.index)
    return wins.fillna(0.0), runs.fillna(0.0)


def _add_strength(w: pd.DataFrame, new: dict, e: str, p: str, runs: pd.Series, wins: pd.Series,
                  nmfp_shrunk: pd.Series, k_sr: float) -> list[str]:
    """§5.1 TrainerStrength, schedule-adjusted, and its components.

    The composite is an equal-ish weighting of *running* z-scores (standardised
    against the population as it stood on earlier days, so it cannot see the
    future). Components the frame cannot supply are dropped from the average
    rather than imputed."""
    exp_tot, _ = prior_stats(w, e, "_sr_expected")
    runs_nz = runs.replace(0, np.nan)
    new[f"{p}_sr_expected"] = exp_tot / runs_nz
    # excess wins per run over what the schedule entered was worth, shrunk
    new[f"{p}_sr_resid"] = (wins - exp_tot) / (runs + k_sr)
    lp, lp_n = prior_stats(w, e, "_ln_prize")
    new[f"{p}_class_geo"] = np.exp(lp / lp_n.replace(0, np.nan))       # geomean win prize money (WPMCV)
    fs, fs_n = prior_stats(w, e, "_field_strength")
    new[f"{p}_sched_strength"] = fs / fs_n.replace(0, np.nan)
    _, size_n = _time_window_prior_mean(w, e, "_one", "race_date", 365)
    new[f"{p}_stable_size"] = size_n

    parts, wsum = [], 0.0
    for series, weight in ((nmfp_shrunk, 1.0), (new[f"{p}_prb2_fsa_shrunk"], 1.0), (new[f"{p}_sr_resid"], 1.0),
                           (np.log(new[f"{p}_class_geo"]), 0.5), (new[f"{p}_sched_strength"], 0.5)):
        s = pd.Series(series, index=w.index)
        if not s.notna().any():
            continue
        parts.append(weight * running_z(w, s).fillna(0.0)); wsum += weight
    new[f"{p}_strength"] = (sum(parts) / wsum) if parts else pd.Series(np.nan, index=w.index)
    return [f"{p}_sr_expected", f"{p}_sr_resid", f"{p}_class_geo", f"{p}_sched_strength",
            f"{p}_stable_size", f"{p}_strength"]


def _add_market_features(w: pd.DataFrame, new: dict, e: str, p: str, k_ae: float = 20.0,
                         k_roi: float = 50.0) -> list[str]:
    """§5.1 A/E and ROI. Market-derived: Stage C only, never Stage F."""
    feats: list[str] = []
    for col, tag in (("_price", "bsp"), ("_sp", "sp")):
        if col not in w.columns:
            continue
        price = w[col]
        w["_profit"] = pd.Series(np.where(w["_won"] > 0, price - 1.0, -1.0), index=w.index).where(price.notna())
        ptot, pn = prior_stats(w, e, "_profit")
        new[f"{p}_roi_{tag}"] = ptot / (pn + k_roi)          # profit per £1 level stake, shrunk to 0
        feats.append(f"{p}_roi_{tag}")
        if tag == "bsp":
            w["_pimp"] = 1.0 / price
            w["_won_priced"] = w["_won"].where(price.notna())
            exp_tot, _ = prior_stats(w, e, "_pimp")
            w_tot, _ = prior_stats(w, e, "_won_priced")
            new[f"{p}_ae_ratio"] = (w_tot + k_ae) / (exp_tot + k_ae)     # shrunk toward 1
            feats.append(f"{p}_ae_ratio")
    return feats
