"""Cell habits of the price: the price's past record in each cell the card shows.

A runner sits in several cells that anyone can read off the card before the off: its trap at this track and
distance, its trap at this track, its trainer at this track, its trainer, its sire, its grade move at this track
(up, down, same, a new track, a first run, another class family) and its age band by sex. For each cell this table
keeps what the starting price has done there on EARLIER days:

- ``_ae``: log((winners + K_AE) / (the SP's expected winners + K_AE)) of the cell's runners: how the cell has done
  against its price (above 0 the cell has won more than the SP said). The expected winners are the SP's chances
  normalised to 1 within the race (``sp_p_norm``, greyhound/data.py);
- ``_sp``: the sum of the race-centred log SP given to the cell's runners over (their number + K_SP): the price
  the cell habitually gets (below 0 shorter than its races' average, pulled to 0 with little evidence). The log SP
  is centred on the mean log SP of the race's field, and the field is every row of ``runs`` with that race_id that
  has an SP (not only the runners with a Betfair price);
- ``_n`` (the trainer only): log(1 + the cell's runners with a result).

``_ae`` and ``_sp`` read the results that have an SP only (a winner with no SP has no expected winners to be set
against); ``_n`` counts every result.

**What is read.** A cell's record on a row is made from the rows of ``runs`` that have a result (not a card row,
not a trial, ``won`` known) on days strictly before the row's day: a race earlier the same day is not read. The
row itself gives only its card fields (track, distance, class, trap, trainer, sire, born, sex). The grade move
compares today's class with the class of the dog's last run on an earlier day (core.prev_day_values), both on the
card. A card row (``is_card``, no result yet) therefore gets the same features its run gets once the result is in
(tests/test_greyhound_cells.py: a planted effect, a middle day's results rewritten, later days cut off or added, a
card row, a row-by-row reading of the specification, no duplicate keys).

A row with a missing cell key (no class, no birth month, no sire ...) has NaN in that cell's columns; a row whose
cell has no earlier result has 0 (the prior: no habit known).

**The trainer.** GBGB serves each dog's CURRENT trainer on every past result (core.trainer_at_the_time). With a
fetch record in ``ctx["fetch_lag"]`` (the nightly build and the live path give one) the two trainer cells read a
past row's trainer only where GBGB was read within core.FRESH_DAYS of the race, as the connections and market
tables do. With no fetch record the trainer is read as served, which is what the screen did; the notes say which.

**Screened** on 12 Nov 2026 against what the stack still misses (the discovery round's ``ope_`` set: the stack's
out-of-sample residual on the Betfair SP, 49,612 runners): the set took the residual's RMSE from 0.2872 to 0.2868
(z -2.7) where a shuffled copy raised it (0.2875), and from 0.2854 to 0.2851 on top of the optics table. A small
effect, carried mostly by the trap and trainer cells' ``_ae``. This module is that set as a table; the one change
of definition is the grade move's previous run (the last EARLIER day's last run here; the screen took the dog's
previous row and called a second run on the same day a first run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "ch_"
SOURCE = ("GBGB results (sources/gbgb.py): winners and starting prices of every earlier day's runners, by the "
          "card's cells (trap, track, distance, class, trainer, sire, born, sex)")
K_AE = 10.0                  # winners added to each side of the A/E (the screen's constant, not fitted)
K_SP = 20.0                  # runners added under the mean centred log SP (the screen's constant, not fitted)
AGE_EDGES = (0, 22, 26, 30, 36, 42, 48, 200)      # months, right-closed bands
MONTH_DAYS = 30.44
BIG = 1 << 21                # day numbers are below this: a cell's block of the (cell, day) key
MOVES = ("first", "newtrack", "other", "up", "down", "same")

#: cell -> (the card fields that make it, the columns it gives)
CELLS = {
    "tdt": (("track", "_dist", "trap"), ("ae", "sp")),
    "tt": (("track", "trap"), ("ae",)),
    "trtrack": (("_trainer", "track"), ("ae", "sp")),
    "tr": (("_trainer",), ("ae", "n")),
    "sire": (("sire",), ("ae", "sp")),
    "move": (("track", "_move"), ("ae", "sp")),
    "agesex": (("_age", "sex"), ("ae", "sp")),
}
_WHAT = {
    "tdt": "its trap at this track and distance",
    "tt": "its trap at this track (every distance)",
    "trtrack": "its trainer at this track",
    "tr": "its trainer (every track)",
    "sire": "its sire's progeny",
    "move": "its grade move at this track (first run, new track, another class family, up, down, same)",
    "agesex": f"its age band (months, edges {', '.join(str(e) for e in AGE_EDGES[1:-1])}) and sex",
}


def grade_move(df: pd.DataFrame) -> np.ndarray:
    """Each row's grade move against the dog's last run on an earlier day, as a code into ``MOVES`` (float; NaN
    where today's row has no class family). Lower grade numbers are higher classes: A3 to A2 is up."""
    one = df.assign(_one=1.0)
    prev = core.prev_day_values(one, "dog_id", ["_one", "track", "grade", "grade_family"])
    has = prev["_one"].notna().to_numpy()
    grade = pd.to_numeric(df.grade, errors="coerce").to_numpy(float)
    pg = pd.to_numeric(prev["grade"], errors="coerce").to_numpy(float)
    fam = df.grade_family.astype(object).to_numpy()
    pfam = prev["grade_family"].astype(object).to_numpy()
    same_track = (prev["track"].astype(object).to_numpy() == df.track.astype(object).to_numpy())
    other = (pfam != fam) | pd.isna(pfam) | np.isnan(pg) | np.isnan(grade)
    with np.errstate(invalid="ignore"):
        mv = np.where(~has, 0, np.where(~same_track, 1, np.where(other, 2, np.where(grade < pg, 3,
                                                                                  np.where(grade > pg, 4, 5)))))
    return np.where(pd.isna(fam), np.nan, mv.astype(float))


def age_band(df: pd.DataFrame, day: pd.Series) -> np.ndarray:
    """The age band's number (float; NaN with no birth month or an age outside the bands)."""
    born = pd.to_datetime(df.born, format="%b-%Y", errors="coerce")
    months = ((day - born).dt.days / MONTH_DAYS).to_numpy(float)
    band = np.searchsorted(np.asarray(AGE_EDGES, float), months, side="left") - 1.0       # right-closed bands
    ok = np.isfinite(months) & (months > AGE_EDGES[0]) & (months <= AGE_EDGES[-1])
    return np.where(ok, band, np.nan)


def cell_codes(df: pd.DataFrame, keys) -> np.ndarray:
    """One integer a distinct combination of ``keys``; -1 where any key is missing."""
    code = np.zeros(len(df), np.int64)
    miss = np.zeros(len(df), bool)
    for k in keys:
        c, u = pd.factorize(df[k], use_na_sentinel=True)
        miss |= c < 0
        code = code * (len(u) + 1) + np.where(c < 0, 0, c)
    code = pd.factorize(code)[0].astype(np.int64)
    return np.where(miss, -1, code)


def before(code: np.ndarray, dn: np.ndarray, res: np.ndarray, vals: np.ndarray, hist_code: np.ndarray | None = None
           ) -> np.ndarray:
    """For each row the sums of ``vals`` (rows x columns) over the rows with a result (``res``) whose cell
    (``hist_code``, by default ``code``) is the row's cell and whose day is strictly earlier. 0 with none."""
    hc = code if hist_code is None else hist_code
    h = res & (hc >= 0)
    hk = hc[h] * BIG + dn[h]
    uk, inv = np.unique(hk, return_inverse=True)
    out = np.zeros((len(code), vals.shape[1]))
    q = np.where(code >= 0, code, 0)
    hi = np.searchsorted(uk, q * BIG + dn, side="left")              # the cell's days before today end here
    lo = np.searchsorted(uk, q * BIG, side="left")                   # and start here
    for j in range(vals.shape[1]):
        c = np.r_[0.0, np.cumsum(np.bincount(inv, weights=vals[h, j], minlength=len(uk)))]
        out[:, j] = c[hi] - c[lo]
    return out


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = runs.drop_duplicates(["race_id", "dog_id"]).reset_index(drop=True)
    n = len(df)
    day = core.day(df)
    dn = (day - pd.Timestamp("2000-01-01")).dt.days.fillna(0).to_numpy().astype(np.int64)
    card = df.is_card.fillna(False).to_numpy(bool) if "is_card" in df else np.zeros(n, bool)
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(n, bool)
    won = pd.to_numeric(df.won, errors="coerce").to_numpy(float)
    res = ~card & ~trial & np.isfinite(won) & day.notna().to_numpy()                # the rows a later day may read
    p = pd.to_numeric(df.sp_p_norm, errors="coerce").to_numpy(float)
    sp = pd.to_numeric(df.sp_decimal, errors="coerce")
    lsp = np.log(sp.where(sp > 0))
    lsp_c = (lsp - lsp.groupby(df.race_id).transform("mean")).to_numpy(float)       # the field: every row of the race
    # a result with no SP (48 of 350,818 on 1 Jul 2025 to 5 Nov 2026, 11 of them winners) has no expected winners
    # to set its win against and no price to average: the A/E and the price habit read the priced results only
    priced = res & np.isfinite(p) & np.isfinite(lsp_c)
    vals = np.column_stack([np.where(priced, won, 0.0), np.where(priced, p, 0.0), np.where(priced, lsp_c, 0.0),
                            priced.astype(float), res.astype(float)])

    lag = (ctx or {}).get("fetch_lag")
    at_the_time = lag is not None and len(lag) > 0
    df["_trainer"] = df.trainer if "trainer" in df else np.nan
    # a past row's trainer, as a later day may read it
    df["_trainer_then"] = core.trainer_at_the_time(df, lag) if at_the_time else df["_trainer"]
    df["_dist"] = pd.to_numeric(df.distance_m, errors="coerce").round()
    df["_move"] = grade_move(df)
    df["_age"] = age_band(df, day)
    for c in ("sire", "sex"):
        if c not in df:
            df[c] = np.nan

    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()})
    notes = {"rows": int(n), "duplicate_keys_dropped": int(len(runs) - n), "rows_with_a_result": int(res.sum()),
             "results_with_no_sp_left_out_of_the_price_columns": int((res & ~priced).sum()),
             "card_rows": int(card.sum()), "k_ae": K_AE, "k_sp": K_SP,
             "trainer_read": "at the time (fetch record)" if at_the_time else "as served (no fetch record)",
             "moves": {m: int((df["_move"] == i).sum()) for i, m in enumerate(MOVES)}, "cells": {}}
    for name, (keys, outs) in CELLS.items():
        code = cell_codes(df, keys)
        hist = None
        if "_trainer" in keys and at_the_time:                       # the same cells, a past row's trainer as known then
            both = pd.concat([df[list(keys)], df[list(keys)].assign(_trainer=df["_trainer_then"])], ignore_index=True)
            cc = cell_codes(both, keys)
            code, hist = cc[:n], cc[n:]
        s = before(code, dn, res, vals, hist)
        ok = code >= 0
        if "ae" in outs:
            out[f"{PREFIX}{name}_ae"] = np.where(ok, np.log((s[:, 0] + K_AE) / (s[:, 1] + K_AE)), np.nan)
        if "sp" in outs:
            out[f"{PREFIX}{name}_sp"] = np.where(ok, s[:, 2] / (s[:, 3] + K_SP), np.nan)
        if "n" in outs:
            out[f"{PREFIX}{name}_n"] = np.where(ok, np.log1p(s[:, 4]), np.nan)
        notes["cells"][name] = {"cells": int(code.max()) + 1 if n else 0,
                                "rows_with_the_cell": round(float(ok.mean()), 4) if n else None,
                                "rows_with_an_earlier_result": round(float((s[:, 4] > 0).mean()), 4) if n else None,
                                "median_earlier_results": float(np.median(s[ok, 4])) if ok.any() else None}
    out = out[["race_id", "dog_id"] + list(COLUMNS)]
    out[list(COLUMNS)] = out[list(COLUMNS)].astype(np.float32)
    return out, notes


def _columns() -> dict:
    cols = {}
    for name, (keys, outs) in CELLS.items():
        what = _WHAT[name]
        if "ae" in outs:
            cols[f"{PREFIX}{name}_ae"] = (f"{what}: log((winners + {K_AE:g}) / (the SP's expected winners + {K_AE:g})) "
                                          "of the cell's priced runners on earlier days (above 0 it has beaten its price)")
        if "sp" in outs:
            cols[f"{PREFIX}{name}_sp"] = (f"{what}: the cell's runners' race-centred log SP on earlier days, summed "
                                          f"over (their number + {K_SP:g}) (below 0 the cell is habitually short)")
        if "n" in outs:
            cols[f"{PREFIX}{name}_n"] = f"{what}: log(1 + its runners with a result on earlier days)"
    return cols


COLUMNS = _columns()
#: the trainer cells read GBGB's trainer as served (the dog's trainer NOW on every past row) unless the build passes a
#: fetch record: not the card's until then (as the fast loop treats cx_tr_ and mk_tr_)
CARD_UNSAFE = tuple(c for c in COLUMNS if "_tr_" in c or "trtrack" in c)
