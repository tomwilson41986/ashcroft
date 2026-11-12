"""Gear: what a change of headgear says, by how severe it is and by the yard that makes it.

The intent block has one flag for first-time headgear and each trainer's record with it against the price. The
owner's ask of 12 Nov 2026: "gear changes, more severe gear, gear first time; uplift, residuals, by trainer, intent".
The card's headgear is read on a ladder of severity (how much it restricts the horse): none 0; hood or eyeshield 1;
cheekpieces 2; visor 3; blinkers (and hood with blinkers) 4. The tongue tie is a breathing aid and is read apart.

    gr_sev               today's severity
    gr_sev_change        today's less the last run's (above zero: more severe; below: taken down or off)
    gr_more  gr_less  gr_off     1 for a step up, a step down, and all headgear off after wearing some last time
    gr_first             1 when an item is worn that the horse has never worn
    gr_first_blnk  gr_first_ckpc  gr_first_vsor  gr_first_hood  gr_first_tt     the same by item
    gr_tt_on  gr_tt_off  tongue tie on after none last time, and off after one
    gr_runs_at           the horse's earlier runs at today's severity
    gr_h_uplift          the horse's mean result against the price (s) at today's severity less its mean over all
                         its runs (NaN until it has run at this severity)

and the yard's record with each move, for a runner making that move today (else 0), over its earlier runners,
decayed over three years and shrunk to zero by twenty runners:

    gr_tr_worn_s                                            the yard's runners in any headgear, against the price
    gr_tr_first_s  gr_tr_more_s  gr_tr_off_s  gr_tr_tt_s     against the price: s = the normalised finishing position
                                                            less the mean of earlier runners at its price
    gr_tr_first_f  gr_tr_more_f                             against the horse: the time figure less the horse's
                                                            own decayed mean figure, in pounds

Earlier days only for every record; the gear itself is the card's. "Last run" is the horse's last earlier racing day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.blocks import time_figure as _tf
from model.blocks._resid import price_residual
from model.freshness_features import _Days, _col
from model.race_shape import asof_decayed_mean, day_index, race_key

ITEMS = {"hood": 1, "eye": 1, "ckpc": 2, "vsor": 3, "blnk": 4, "h+b": 4, "b": 4}
FIRSTS = {"blnk": ("blnk", "h+b", "b"), "ckpc": ("ckpc",), "vsor": ("vsor",), "hood": ("hood", "h+b"), "tt": ("tt",)}
FEATURES = (["gr_sev", "gr_sev_change", "gr_more", "gr_less", "gr_off", "gr_first"]
            + [f"gr_first_{k}" for k in FIRSTS] + ["gr_tt_on", "gr_tt_off", "gr_runs_at", "gr_h_uplift",
                                                   "gr_tr_first_s", "gr_tr_more_s", "gr_tr_off_s", "gr_tr_tt_s",
                                                   "gr_tr_first_f", "gr_tr_more_f", "gr_tr_worn_s"])
POST_RACE: set[str] = set()
READS = sorted(set(_tf.READS) | {"trainer", "headgear", "number_of_runners", "bfsp"})
K = 20.0
HALF = 1095.0


def _codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    return np.where(t.isin(["", "nan", "none"]).to_numpy(), -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    horse, trainer = _codes(df["horse_name"]), _codes(_col(df, "trainer"))
    H = _Days(horse, day)
    hg = " " + _col(df, "headgear").fillna("").astype(str).str.lower().str.strip() + " "
    worn = {k: hg.str.contains(f" {k} ", regex=False).to_numpy() for k in set(ITEMS) | {"tt"}}
    sev = np.zeros(len(df))
    for k, v in ITEMS.items():
        sev = np.where(worn[k], np.maximum(sev, v), sev)
    tt = worn["tt"].astype(float)

    prev = np.clip(np.arange(len(H.u)) - 1, 0, None)
    has_prev_b = H.valid & (H.pos >= 1)

    def last(v):
        b = H.per_block(v, how="max")
        return H.to_rows(np.where(has_prev_b, b[prev], np.nan))

    def before(flag):
        """How many of the horse's earlier racing days the flag held on."""
        b = np.nan_to_num(H.per_block(flag.astype(float), how="max"))
        cs = pd.Series(b).groupby(H.key).cumsum().to_numpy()
        return H.to_rows(np.where(has_prev_b, cs[prev], 0.0))

    psev, ptt = last(sev), last(tt)
    known = np.isfinite(psev)
    cols: dict[str, np.ndarray] = {"gr_sev": sev, "gr_sev_change": np.where(known, sev - psev, np.nan)}
    cols["gr_more"] = np.where(known, (sev > psev).astype(float), np.nan)
    cols["gr_less"] = np.where(known, ((sev < psev) & (sev > 0)).astype(float), np.nan)
    cols["gr_off"] = np.where(known, ((sev == 0) & (psev > 0)).astype(float), np.nan)
    first_any = np.zeros(len(df), dtype=bool)
    for name, items in FIRSTS.items():
        on = np.zeros(len(df), dtype=bool)
        for k in items:
            on |= worn[k]
        f = on & (before(on) == 0)
        first_any |= f
        cols[f"gr_first_{name}"] = np.where(known, f.astype(float), np.nan)
    for k in ("eye",):
        first_any |= worn[k] & (before(worn[k]) == 0)
    cols["gr_first"] = np.where(known, first_any.astype(float), np.nan)
    cols["gr_tt_on"] = np.where(known, ((tt == 1) & (ptt == 0)).astype(float), np.nan)
    cols["gr_tt_off"] = np.where(known, ((tt == 0) & (ptt == 1)).astype(float), np.nan)

    # the residuals: against the price and against the horse's own level
    race = pd.Series(pd.factorize(race_key(df), sort=True)[0], index=df.index)
    s, _ = price_residual(df, day, race)
    fig = np.asarray(_tf.figures(df)["fig"], dtype=float)
    own, _ = asof_decayed_mean(horse, day, fig, horse, day, halflife_days=365.0)
    f = fig - own

    # the horse at today's severity against all its runs
    cols["gr_runs_at"] = np.zeros(len(df))
    at_m, at_n = np.full(len(df), np.nan), np.zeros(len(df))
    for level in sorted(set(ITEMS.values()) | {0}):
        at = sev == level
        if not at.any():
            continue
        cols["gr_runs_at"] = np.where(at, before(at), cols["gr_runs_at"])
        m1, n1 = asof_decayed_mean(horse, day, np.where(at, s, np.nan), horse, day)
        at_m, at_n = np.where(at, m1, at_m), np.where(at, n1, at_n)
    all_m, all_n = asof_decayed_mean(horse, day, s, horse, day)
    cols["gr_h_uplift"] = np.where((at_n > 0) & (all_n > 0), at_m - all_m, np.nan)

    def yard(flag, r):
        m, cnt = asof_decayed_mean(trainer, day, np.where(flag == 1, r, np.nan), trainer, day, halflife_days=HALF)
        return np.where(flag == 1, np.nan_to_num(m) * np.nan_to_num(cnt) / (np.nan_to_num(cnt) + K), 0.0)

    cols["gr_tr_first_s"], cols["gr_tr_first_f"] = yard(cols["gr_first"], s), yard(cols["gr_first"], f)
    cols["gr_tr_more_s"], cols["gr_tr_more_f"] = yard(cols["gr_more"], s), yard(cols["gr_more"], f)
    cols["gr_tr_off_s"], cols["gr_tr_tt_s"] = yard(cols["gr_off"], s), yard(cols["gr_tt_on"], s)
    cols["gr_tr_worn_s"] = yard((sev > 0).astype(float), s)

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
