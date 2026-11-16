"""The litter as a live sensor: common-mode form, the litter level, the runt (the owner, 11 Nov 2026: "identify NEW
features that have not been discovered yet").

Littermates (the same sire, the same dam, the same birth month) share half their genes, all of their rearing and
schooling and often a kennel, and greyhound litters are large and nearly all race. A dog gives one noisy reading of
itself a week; its brothers and sisters give three or four more readings of whatever they share. So:

- what the REST of the litter is doing now (running above or below its expected times) is a sensor of a hidden state
  the dog shares with it: a maturation spurt, a virus in the kennel, a change of regime;
- where the litter's level sits is a prior for a young dog with little form of its own;
- its weight against the littermates of its sex says whether it is the runt.

The litter is fixed at birth and printed on the card, so unlike the trainer (served as the current value on old
results) it is known at the time on every row.

Construction. The litter key is the sire, the dam and ``born``, each lower case with letters and digits only; a row
with any of the three missing has no litter and every column is NaN. Everything is read from EARLIER days only (a
littermate running today, in this race or another, is not read), and the dog's OWN runs are always taken out: each
decayed sum is made once by the litter and once by the dog within the litter (``earlier_by``, the cumulative-sum
trick of ``figs.decayed`` with any key in place of the dog) and the two are subtracted.

- ``lt_size``: distinct dogs of the litter with a run before today, the dog itself included if it has one;
- ``lt_active``: littermates (other dogs) with a run in the ``WINDOW`` days before today;
- ``lt_surprise``: the decayed mean (half-life ``HL_SURPRISE`` days) of the littermates' residuals: each graded run's
  calculated time less its expected time before the race (``dyad.residuals`` on the expected-time table's
  ``xt_tot_time``), less the mean of that over the race's runners (two or more needed: the night's going cancels),
  in lengths, above 0 slower than expected against its field. (the litter's decayed sum - the dog's own) / (the
  litter's decayed weight - the dog's own); NaN when the others' weight is under ``MIN_W``. ``lt_surprise_n`` is
  that weight;
- ``lt_level``: the same on the littermates' ``xt_tot_level`` (their level before each of their runs, lengths, higher
  faster; half-life ``HL_LEVEL`` days), and ``lt_level_gap`` = ``lt_level`` less the dog's own ``xt_tot_level`` today
  (above 0: the litter sits above the dog);
- ``lt_wt``: the dog's last earlier weight less the decayed mean weight (half-life ``HL_WT`` days) of its
  littermates of the same sex, kg (well below 0: the runt). The decayed mean stands for "each littermate's most
  recent weight": it weights a littermate by how often and how lately it has run;
- ``lt_start_gap``: days from the litter's first run to the dog's own first run (0 for the first of the litter to
  run; for a dog with no run before today, the days from the litter's first run to today);
- ``lt_won``: the decayed share of graded races won by the littermates (half-life ``HL_WON`` days).

"A run" is a result row of the input (an intertrack trial is a run; a card row is not; the T-class trials of
``ctx["trial_runs"]`` are not read); a residual and a win are read from graded races only. The counts know only the
dogs and days in the data they are given: in a store that starts in mid-2025 a litter's first run is its first run
in the store.

Every column is read from earlier days only, plus today's card fields and the dog's own expected-time level before
the race (tests/test_greyhound_litter.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, dyad

PREFIX = "lt_"
SOURCE = ("GBGB results (sire, dam, born, sex, weight, position, calculated time: sources/gbgb.py) against the "
          "expected-time table (xt_tot_time, xt_tot_level)")
HL_SURPRISE = 21.0           # days: the litter's form now
HL_LEVEL = 180.0             # days: where the litter sits
HL_WT = 90.0                 # days: the littermates' weights
HL_WON = 90.0
WINDOW = 30                  # days: a littermate is active with a run in the last WINDOW days
MIN_W = 0.5                  # decayed runs of the others below which a mean is not read
BIG = 1_000_000              # day numbers are below this: a key's block of a sorted (key, day) code
EPS = 1e-9


def earlier_by(key: np.ndarray, dn: np.ndarray, cols: list[np.ndarray], half_life: float | None = None) -> list[np.ndarray]:
    """Decayed sums over EARLIER days by an arbitrary key. For rows keyed by ``key`` (integer codes, below 0: no key)
    on day number ``dn``: each column's sum over the key's rows on days before the row's day, each such row weighted
    0.5 ** (its age in days / ``half_life``) (``None``: no decay). The day's own rows, before or after, are left
    out. Aligned to the input order; NaN where there is no key, 0 where the key has no earlier day. A key's span must
    stay under about 1,000 half-lives (the sums are carried scaled up from the key's first day)."""
    key, dn = np.asarray(key, dtype=np.int64), np.asarray(dn, dtype=np.int64)
    out = [np.full(len(key), np.nan) for _ in cols]
    rows = np.flatnonzero(key >= 0)
    if not len(rows):
        return out
    order = rows[np.lexsort((dn[rows], key[rows]))]
    k, d = key[order], dn[order]
    new_key = np.r_[True, k[1:] != k[:-1]]
    new_day = new_key | np.r_[True, d[1:] != d[:-1]]
    start = np.flatnonzero(new_day)                           # the first row of each (key, day) cell
    cell = np.cumsum(new_day) - 1                             # each row's cell
    cell_key = (np.cumsum(new_key) - 1)[start]                # each cell's key, 0, 1, 2 ...
    since = (d[start] - d[np.flatnonzero(new_key)][cell_key]).astype(float)      # days from the key's first day
    up = np.exp(np.log(2.0) / half_life * since) if half_life else np.ones(len(start))
    first_cell = np.r_[True, cell_key[1:] != cell_key[:-1]]   # the key's first day: nothing before it
    for j, c in enumerate(cols):
        s = np.add.reduceat(np.nan_to_num(np.asarray(c, dtype=float)[order]), start) * up
        upto = pd.Series(s).groupby(cell_key).cumsum().to_numpy()
        # the key's earlier cells only: the running sum as it stood after the cell before. The day's own values are
        # never added and taken away again ((before + today) - today is before only to rounding, and is nothing like
        # it when today's value is large), so no row's sum is a function of its own day at all
        before = np.where(first_cell, 0.0, np.r_[0.0, upto[:-1]])
        out[j][order] = (before / up)[cell]
    return out


def _norm_codes(s: pd.Series) -> np.ndarray:
    """Codes of a name column, lower case with letters and digits only; -1 where missing or empty."""
    codes, uniq = pd.factorize(s)
    if not len(uniq):
        return np.full(len(s), -1, dtype=np.int64)
    norm = pd.Series(np.asarray(uniq, dtype=object)).astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    nc = pd.factorize(norm.where(norm != ""))[0]
    return np.where(codes >= 0, nc[np.clip(codes, 0, None)], -1).astype(np.int64)


def _combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compact codes of the pair (a, b); -1 where either is missing."""
    out = np.full(len(a), -1, dtype=np.int64)
    ok = (a >= 0) & (b >= 0)
    if ok.any():
        out[ok] = pd.factorize(a[ok] * (int(b[ok].max()) + 1) + b[ok])[0]
    return out


def litter_codes(df: pd.DataFrame) -> np.ndarray:
    """Each row's litter: the normalised sire, dam and born; -1 where any of the three is missing."""
    if not all(c in df for c in ("sire", "dam", "born")):
        return np.full(len(df), -1, dtype=np.int64)
    return _combine(_combine(_norm_codes(df.sire), _norm_codes(df.dam)), _norm_codes(df.born))


def _others(key, own, dn, value, half_life) -> tuple[np.ndarray, np.ndarray]:
    """The decayed mean of ``value`` (NaN: not read) over the key's earlier days less the dog's own, and its weight."""
    w = np.isfinite(value).astype(float)
    sw, swx = earlier_by(key, dn, [w, np.nan_to_num(value)], half_life)
    ow, owx = earlier_by(own, dn, [w, np.nan_to_num(value)], half_life)
    n = sw - ow
    n = np.where(n < EPS, 0.0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n >= MIN_W - EPS, (swx - owx) / n, np.nan)
    return mean, n


def _last_before(dog: np.ndarray, dn: np.ndarray, t: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Each row's dog's last value (NaN skipped) on an earlier day."""
    o = np.argsort(t, kind="stable")
    d = pd.DataFrame({"k": dog[o], "dn": dn[o], "v": value[o]})
    per = d.groupby(["k", "dn"], sort=True).v.last().reset_index()
    per["p"] = per.groupby("k").v.ffill()
    per["p"] = per.groupby("k").p.shift(1)
    m = pd.DataFrame({"k": dog, "dn": dn}).merge(per[["k", "dn", "p"]], on=["k", "dn"], how="left")
    return m.p.to_numpy(float)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    xt = (ctx or {}).get("xtime")
    if xt is None:
        from greyhound.pit import xtime
        xt, _ = xtime.build(runs, df)
    j = df[["race_id", "dog_id"]].merge(
        xt[["race_id", "dog_id", "xt_tot_time", "xt_tot_level"]].drop_duplicates(["race_id", "dog_id"]),
        on=["race_id", "dog_id"], how="left")
    n = len(df)
    # the residual against the race: less the mean of the race's runners that have one (two or more needed). The
    # night's going at the track is in every runner's residual and littermates run where the dog runs, so the raw
    # residual made lt_surprise partly the track's (the review of 11 Nov 2026: 0.039 with the run's residual, 0.028
    # once centred); a price is made within the race, where the track's share is no use
    r = dyad.residuals(df, pd.Series(j.xt_tot_time.to_numpy(), index=df.index))
    in_race = r.groupby(df.race_id)
    resid = (r - in_race.transform("mean")).where(in_race.transform("count") >= 2).to_numpy(float)
    level = j.xt_tot_level.to_numpy(float)
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(np.int64)
    dn = dn - (int(dn.min()) if n else 0)
    card = df.is_card.fillna(False).to_numpy(bool) if "is_card" in df else np.zeros(n, bool)
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(n, bool)
    ran = ~card                                                       # a result row: the dog ran
    won = pd.to_numeric(df.won, errors="coerce").to_numpy(float) if "won" in df else np.full(n, np.nan)
    won = np.where(ran & ~trial, won, np.nan)
    wt = pd.to_numeric(df.weight_kg, errors="coerce").to_numpy(float) if "weight_kg" in df else np.full(n, np.nan)
    wt = np.where(ran & (wt > 0), wt, np.nan)
    dog = pd.factorize(df.dog_id)[0].astype(np.int64)
    sex = _norm_codes(df.sex.astype("string").str.strip().str[:1]) if "sex" in df else np.full(n, -1, dtype=np.int64)
    lit = litter_codes(df)
    has = lit >= 0
    own = _combine(lit, dog)                                          # the dog within its litter
    lit_sex = _combine(lit, sex)
    own_sex = _combine(lit_sex, dog)

    # the counts: every (dog, day) run of every litter, sorted; the litter's blocks of sorted (litter, day) codes
    size, active, start_gap, mates = (np.full(n, np.nan) for _ in range(4))
    q = has & ran
    dd = np.unique(own[q] * BIG + dn[q])
    if len(dd):
        dd_own, dd_dn = dd // BIG, dd % BIG
        lit_of = np.zeros(int(own.max()) + 1, dtype=np.int64)
        lit_of[own[has]] = lit[has]
        dd_lit = lit_of[dd_own]
        first = np.r_[True, dd_own[1:] != dd_own[:-1]]
        last = np.r_[first[1:], True]
        # a run on day d makes the dog active on the days d+1 .. d+WINDOW, or to its next run if that is sooner
        until = np.where(last, dd_dn + WINDOW, np.minimum(dd_dn + WINDOW, np.r_[dd_dn[1:], 0]))
        A = np.sort(dd_lit * BIG + dd_dn)
        B = np.sort(dd_lit * BIG + until)
        F = np.sort(dd_lit[first] * BIG + dd_dn[first])               # each dog's first run, by litter
        h = np.flatnonzero(has)
        q0, q1 = lit[h] * BIG, lit[h] * BIG + dn[h]

        def before(arr):                                              # the litter's entries on days before today
            return np.searchsorted(arr, q1, "left") - np.searchsorted(arr, q0, "left")

        sz = before(F)
        lo = np.searchsorted(dd, own[h] * BIG, "left")
        i = np.searchsorted(dd, own[h] * BIG + dn[h], "left") - 1     # the dog's last run before today
        ran_before = i >= lo
        prev = dd_dn[np.clip(i, 0, len(dd) - 1)]
        own_first = np.where(ran_before, dd_dn[np.clip(lo, 0, len(dd) - 1)], dn[h])
        lit_first = F[np.clip(np.searchsorted(F, q0, "left"), 0, len(F) - 1)] - q0
        size[h] = sz
        mates[h] = sz - ran_before                                    # the littermates that have run
        active[h] = before(A) - before(B) - (ran_before & (dn[h] - prev <= WINDOW))
        start_gap[h] = np.where(sz > 0, own_first - lit_first, 0.0)
    else:
        size[has], active[has], start_gap[has], mates[has] = 0.0, 0.0, 0.0, 0.0

    surprise, surprise_n = _others(lit, own, dn, np.where(ran, resid, np.nan), HL_SURPRISE)
    lt_level, _ = _others(lit, own, dn, np.where(ran, level, np.nan), HL_LEVEL)
    lt_won, _ = _others(lit, own, dn, won, HL_WON)
    mates_wt, _ = _others(lit_sex, own_sex, dn, wt, HL_WT)
    own_wt = _last_before(dog, dn, df.t.to_numpy(), wt)

    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    out[f"{PREFIX}size"] = size
    out[f"{PREFIX}active"] = active
    out[f"{PREFIX}surprise"] = surprise
    out[f"{PREFIX}surprise_n"] = surprise_n
    out[f"{PREFIX}level"] = lt_level
    out[f"{PREFIX}level_gap"] = lt_level - level
    out[f"{PREFIX}wt"] = np.where(has, own_wt - mates_wt, np.nan)
    out[f"{PREFIX}start_gap"] = start_gap
    out[f"{PREFIX}won"] = np.clip(lt_won, 0.0, 1.0)
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)

    members = np.bincount(lit_of[np.unique(own[q])]) if len(dd) else np.zeros(0, dtype=np.int64)
    dogs_lit = pd.DataFrame({"d": dog[has], "l": lit[has]}).drop_duplicates()
    notes = {"half_life_surprise": HL_SURPRISE, "half_life_level": HL_LEVEL, "half_life_weight": HL_WT,
             "half_life_won": HL_WON, "active_window_days": WINDOW, "min_weight": MIN_W,
             "rows": int(n), "rows_with_a_litter": int(has.sum()),
             "litters": int(lit.max()) + 1 if has.any() else 0,
             "litters_with_a_run": int((members > 0).sum()),
             "litters_with_two_or_more_dogs_run": int((members >= 2).sum()),
             "largest_litter_dogs_run": int(members.max()) if len(members) else 0,
             "dogs_with_a_litter": int(dogs_lit.d.nunique()),
             "dogs_under_more_than_one_litter_key": int((dogs_lit.groupby("d").l.size() > 1).sum()),
             "share_of_litter_rows_with_a_littermate_run_before":
                 round(float((mates[has] >= 1).mean()), 4) if has.any() else None,
             "share_of_rows_with_a_litter_surprise": round(float(np.isfinite(surprise).mean()), 4) if n else None,
             "share_of_rows_with_a_litter_weight": round(float(out[f"{PREFIX}wt"].notna().mean()), 4) if n else None,
             "mean_active_littermates": round(float(np.nanmean(active[has])), 3) if has.any() else None}
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}size": "distinct dogs of the litter (same sire, dam and birth month) with a run before today, the dog "
                     "itself included if it has one",
    f"{PREFIX}active": f"littermates (other dogs) with a run in the {WINDOW} days before today",
    f"{PREFIX}surprise": f"the littermates' runs against their expected times and their fields on earlier days, decayed mean "
                         f"(half-life {HL_SURPRISE:g} days), the dog's own runs left out (lengths; above 0 the litter "
                         f"is running slower than expected; NaN under {MIN_W:g} decayed runs)",
    f"{PREFIX}surprise_n": "the decayed number of littermates' runs behind it",
    f"{PREFIX}level": f"the littermates' expected-time level (xt_tot_level before each of their earlier runs), decayed "
                      f"mean (half-life {HL_LEVEL:g} days), the dog's own left out (lengths, higher faster)",
    f"{PREFIX}level_gap": "that less the dog's own level today (above 0: the litter sits above the dog)",
    f"{PREFIX}wt": f"the dog's last earlier weight less the decayed mean weight (half-life {HL_WT:g} days) of its "
                   "littermates of the same sex (kg; well below 0: the runt)",
    f"{PREFIX}start_gap": "days from the litter's first run to the dog's own first run (0: the first to run; no run "
                          "yet: to today)",
    f"{PREFIX}won": f"the decayed share of graded races won by the littermates (half-life {HL_WON:g} days)",
}
