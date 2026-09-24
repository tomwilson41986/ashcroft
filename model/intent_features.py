"""Intent: what the connections' choices before the off say, and how the price has taken them.

A trainer's choices for a horse today are on the card:
- a move to a new yard;
- a first handicap;
- a gelding operation;
- first-time headgear;
- a return from a break;
- a drop in class;
- a first run in a code (first over hurdles or fences, first on the all-weather);
- a debut;
- the jockey booked, and whether a claim is taken off.

Some yards' choices have meant more than others'. The market sees the same card, so
the question is not whether an angle wins but whether it wins MORE than its price
says. That is measured per trainer and angle over earlier days, against the race-
normalised BSP probability p:

    it_ae_<angle>   the trainer's decayed, shrunk mean of (won - p) over its earlier
                    runners with this angle, for a runner with the angle today (else 0)

and a positive value means the price has underrated this yard's use of it.

Every statistic uses races on EARLIER DAYS only (model.race_shape.asof_decayed_mean),
and the "previous run" is the horse's last run on an earlier day, so nothing of today's
results -- nor of today's other races -- enters a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.lagsafe import race_minutes
from model.race_shape import _codes, asof_decayed_mean, day_index, race_code, race_key, shrink

#: Half-life in days of a past runner in a trainer x angle record: yards change.
INTENT_HALFLIFE_DAYS = 730.0

#: Shrinkage of a trainer x angle residual toward zero, in effective runners. A yard
#: with ten handicap debutants has told us almost nothing yet.
INTENT_AE_K = 60.0

#: Jockey strength: half-life and shrinkage toward the population's win rate.
JOCKEY_HALFLIFE_DAYS = 365.0
JOCKEY_K = 100.0

#: Days off that make a run a return from a break.
LAYOFF_DAYS = 90

#: career_runs counts the runs BEFORE this one (it equals the runs the database
#: holds on 98% of rows of horses first seen from 2022; research/queries/done/
#: intent_survey.py). A horse's history is complete when the frame holds all of
#: them; without career_runs, when its first row held was its debut or it has this
#: many runs held. "First time" angles are unknown for a horse whose history is not.
KNOWN_RUNS = 5

#: Headgear items as the feed writes them ("CkPc TT", "Blnk Eye"...): a first-time
#: item is one the horse has not worn before, whatever else it has worn.
HEADGEAR_ITEMS = ("ckpc", "tt", "blnk", "hood", "vsor", "eye", "h+b")

ANGLES = ("new_yard", "hcap_debut", "gelded", "ft_headgear", "layoff", "class_drop", "code_switch", "debut")

_HCAP_RX = r"handicap|nursery|\bh'cap\b|\bhcap\b"
_ENTIRE = ("c", "h", "r")          # colt, horse, rig: what a gelding was before

INTENT_FEATURES = (
    [f"it_{a}" for a in ANGLES]
    + ["it_runs_for_yard", "it_seen_runs", "it_jockey_upgrade", "it_claim_change", "it_class_change"]
    + [f"it_ae_{a}" for a in ANGLES]
    + ["it_ae_sum", "it_ae_best", "it_n_angles"]
)


def previous_run(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(position of the horse's previous run, runs before today) for every row.

    The previous run is the horse's last row on an EARLIER DAY (-1 when there is
    none); a second row the same day is not "previous" to the first."""
    horse = _codes(df["horse_name"].astype(str).str.strip().str.lower())
    day = day_index(df)
    mins = race_minutes(df["race_time"]).to_numpy() if "race_time" in df.columns else np.zeros(len(df))
    order = np.lexsort((mins, day, horse))
    hs, ds = horse[order], day[order]
    new_block = np.r_[True, (hs[1:] != hs[:-1]) | (ds[1:] != ds[:-1])]
    blk = np.cumsum(new_block) - 1
    starts = np.flatnonzero(new_block)
    ends = np.r_[starts[1:] - 1, len(order) - 1]
    same_horse = np.r_[False, hs[starts[1:]] == hs[starts[:-1]]]
    prev_pos_blk = np.where(same_horse, order[np.r_[0, ends[:-1]]], -1)
    new_horse = np.r_[True, hs[1:] != hs[:-1]]
    horse_start = np.maximum.accumulate(np.where(new_horse, np.arange(len(order)), 0))
    seen_sorted = starts[blk] - horse_start          # rows on earlier days
    prev = np.empty(len(df), dtype=np.int64)
    seen = np.empty(len(df), dtype=np.int64)
    prev[order] = prev_pos_blk[blk]
    seen[order] = seen_sorted
    return prev, seen


def _prior_count(df: pd.DataFrame, flag: np.ndarray, extra_key=None) -> np.ndarray:
    """How many of the horse's runs on earlier days had `flag` (optionally within a
    further key, e.g. the trainer)."""
    horse = df["horse_name"].astype(str).str.strip().str.lower()
    key = _codes(horse) if extra_key is None else _codes(horse, extra_key)
    day = day_index(df)
    m, n = asof_decayed_mean(key, day, flag.astype(float), key, day, np.inf)
    return np.where(n > 0, np.nan_to_num(m) * n, 0.0)


def _at(values, prev: np.ndarray, fill=""):
    """`values` at each row's previous run (`fill` where there is none)."""
    v = np.asarray(values, dtype=object)
    return np.where(prev >= 0, v[np.clip(prev, 0, None)], fill)


def add_intent_features(df: pd.DataFrame, halflife_days: float = INTENT_HALFLIFE_DAYS,
                        k: float = INTENT_AE_K) -> tuple[pd.DataFrame, list[str]]:
    """The intent block on a frame of history plus (optionally) today's card.

    Needs race_date, race_time, track, horse_name, trainer, race_type (race_name
    helps), and where present horse_sex, headgear, race_class, jockey_name,
    jockeys_claim, career_runs, surface_type, placing_numerical and bfsp. Card rows
    need no result or price: they read the history before their day and add
    nothing to it."""
    n_rows = len(df)
    day = day_index(df)
    rk = race_key(df)
    prev, seen = previous_run(df)
    has_prev = prev >= 0
    df["it_seen_runs"] = np.minimum(seen, 20).astype(float)

    def col(name, default=""):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    # --- is the horse's history in the frame complete? ----------------------------------
    cr = pd.to_numeric(col("career_runs", np.nan), errors="coerce").to_numpy()
    horse = df["horse_name"].astype(str).str.strip().str.lower()
    first_cr = pd.Series(np.where(seen == 0, cr, np.nan), index=df.index).groupby(horse).transform("max")
    complete = np.where(np.isfinite(cr), seen >= cr, (first_cr.to_numpy() <= 1) | (seen >= KNOWN_RUNS))

    # --- the angles ------------------------------------------------------------------------
    tr = col("trainer").fillna("").astype(str).str.strip().str.lower().to_numpy()
    ptr = _at(tr, prev)
    known = has_prev & (tr != "") & (ptr != "")
    df["it_new_yard"] = np.where(known, (tr != ptr).astype(float), np.nan)
    df["it_runs_for_yard"] = np.minimum(_prior_count(df, np.ones(n_rows), extra_key=tr), 20.0)

    sex = col("horse_sex").fillna("").astype(str).str.strip().str.lower().str[:1].to_numpy()
    psex = _at(sex, prev)
    df["it_gelded"] = np.where(has_prev & (psex != ""),
                               ((sex == "g") & np.isin(psex.astype(str), _ENTIRE)).astype(float), np.nan)

    text = (col("race_type").fillna("").astype(str) + " " + col("race_name").fillna("").astype(str)).str.lower()
    hcap = text.str.contains(_HCAP_RX).to_numpy()
    prior_hcaps = _prior_count(df, hcap.astype(float))
    df["it_hcap_debut"] = np.where(complete & has_prev, (hcap & (prior_hcaps == 0)).astype(float), np.nan)

    hg = " " + col("headgear").fillna("").astype(str).str.lower().str.strip() + " "
    new_item = np.zeros(n_rows, dtype=bool)
    for item in HEADGEAR_ITEMS:
        worn = hg.str.contains(f" {item} ", regex=False).to_numpy()
        new_item |= worn & (_prior_count(df, worn.astype(float)) == 0)
    df["it_ft_headgear"] = np.where(complete & has_prev, new_item.astype(float), np.nan)

    pday = np.where(has_prev, day[np.clip(prev, 0, None)], -1)
    gap = np.where(has_prev, day - pday, np.nan)
    df["it_layoff"] = np.where(has_prev, (gap >= LAYOFF_DAYS).astype(float), np.nan)

    cls = pd.to_numeric(col("race_class").astype(str).str.extract(r"(\d)")[0], errors="coerce").to_numpy()
    pcls = np.where(has_prev, cls[np.clip(prev, 0, None)], np.nan)
    df["it_class_change"] = cls - pcls                    # + = down in class (class 1 is the top)
    df["it_class_drop"] = np.where(np.isfinite(df["it_class_change"]), (df["it_class_change"] >= 1).astype(float),
                                   np.nan)

    code = race_code(df).to_numpy()
    prior_in_code = _prior_count(df, np.ones(n_rows), extra_key=code)
    df["it_code_switch"] = np.where(complete & has_prev, (prior_in_code == 0).astype(float), np.nan)

    df["it_debut"] = np.where(seen == 0, np.where(np.isfinite(cr), (cr <= 0).astype(float), np.nan), 0.0)

    # --- the jockey booked, against the horse's last one ------------------------------------
    # A run counts once it has a result or a starting price; a card row has neither.
    jk = col("jockey_name").fillna("").astype(str).str.strip().str.lower().to_numpy()
    place = pd.to_numeric(col("placing_numerical", np.nan), errors="coerce")
    bsp = pd.to_numeric(col("bfsp", np.nan), errors="coerce")
    won = (place == 1).astype(float)
    ran = (place.notna() | (bsp > 1)).to_numpy()
    y = np.where(ran, won.to_numpy(), np.nan)
    pjk = _at(jk, prev).astype(str)
    jcode_all = _codes(np.r_[jk, pjk])
    jcode = np.where(jk != "", jcode_all[:n_rows], -1)
    pjcode = np.where(pjk != "", jcode_all[n_rows:], -1)
    # every rider is shrunk toward the population's win rate on earlier days (a global
    # mean would carry today's results into today's features)
    zero = np.zeros(n_rows, dtype=np.int64)
    pop, _ = asof_decayed_mean(zero, day, y, zero, day, JOCKEY_HALFLIFE_DAYS)
    pop = np.where(np.isfinite(pop), pop, 0.1)
    m_now, n_now = asof_decayed_mean(jcode, day, y, jcode, day, JOCKEY_HALFLIFE_DAYS)
    m_prev, n_prev = asof_decayed_mean(jcode, day, y, pjcode, day, JOCKEY_HALFLIFE_DAYS)
    sr_now = shrink(m_now, n_now, pop, JOCKEY_K)
    sr_prev = shrink(m_prev, n_prev, pop, JOCKEY_K)
    df["it_jockey_upgrade"] = np.where((jcode >= 0) & (pjcode >= 0), sr_now - sr_prev, np.nan)
    claim = pd.to_numeric(col("jockeys_claim", np.nan), errors="coerce").fillna(0.0).to_numpy()
    pclaim = np.where(has_prev, claim[np.clip(prev, 0, None)], np.nan)
    df["it_claim_change"] = claim - pclaim               # + = a bigger claim today

    # --- each trainer's record with each angle, against the price ---------------------------
    # Only races where every runner has a BSP: p is then the price's own chance.
    inv = 1.0 / bsp.where(bsp > 1)
    priced = inv.groupby(rk).transform("count") == inv.groupby(rk).transform("size")
    p = (inv / inv.groupby(rk).transform("sum")).where(priced)
    resid = (won - p).to_numpy()
    tcode = np.where(tr != "", _codes(tr), -1)
    ae_cols = []
    for a in ANGLES:
        flag = np.nan_to_num(df[f"it_{a}"].to_numpy(dtype=float)) > 0
        key = np.where(flag & (tcode >= 0), tcode, -1)
        m, n = asof_decayed_mean(key, day, resid, key, day, halflife_days)
        df[f"it_ae_{a}"] = np.where(flag, shrink(m, n, 0.0, k), 0.0)
        ae_cols.append(f"it_ae_{a}")
    ae = df[ae_cols].to_numpy()
    active = np.nan_to_num(df[[f"it_{a}" for a in ANGLES]].to_numpy(dtype=float)) > 0
    df["it_ae_sum"] = ae.sum(axis=1)
    df["it_n_angles"] = active.sum(axis=1).astype(float)
    df["it_ae_best"] = np.where(active.any(axis=1), np.where(active, ae, -np.inf).max(axis=1), 0.0)
    return df, list(INTENT_FEATURES)
