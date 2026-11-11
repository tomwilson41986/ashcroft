"""Optical surplus: how much better the printed card looks than the dog ran (the owner, 11 Nov 2026: "identify NEW
features that have not been discovered yet").

One target is the Betfair SP, a crowd product, and the crowd reads the printed card: the last six lines, the
finishing positions, the fastest calculated time, the wins. It cannot compute what the expected-time table computes
(the night's variant, the trap, the grade, the dog's level across tracks). Where the card LOOKS better than the dog
ran (a win in a weak race, a fast time on a fast night, a round number just beaten, a run of short heads) the price
should be shorter than the ability, and the reverse. This table is the look, the substance, and the look less the
substance.

**The card.** A dog's lines are its last ``N_CARD`` earlier DAYS with a finishing position (a void race is no line;
a day with two runs is read by its last run; real trials are not in ``runs`` and are not read). Nothing of today.

**The look** (what a reader takes off the card):

- ``op_best``: its fastest calculated time (``adjusted_time``, seconds) among its lines at today's track and
  distance, a time no greyhound runs left out (outside ``PACE`` seconds a metre: GBGB's keying errors, a going
  allowance of 514 s, a 238 m time under a 415 m race; fixed bounds, not fitted); ``op_best_vs``: (the mean of the other runners' ``op_best`` less its own) / 0.08, lengths, above 0 the
  time-finder's pick; ``op_fastest``: 1 when it has the race's fastest ``op_best``;
- ``op_pos_last``, ``op_pos_sum`` (the form figures added up, at the rate of six lines: 6 x the mean position of
  its lines, so a dog with four lines is not flattered by having fewer digits; NaN with fewer than ``MIN_POS``),
  ``op_wins6``, ``op_lto_won``;
- the left-digit illusion (a printed 29.94 reads a class apart from 30.02): ``op_round_under`` = 1 when the
  hundredths of ``op_best`` are .90 to .99, ``op_round_over`` = 1 when they are .00 to .09, ``op_round_dist`` =
  ``op_best`` less the nearest whole second (-0.50 to 0.49: a time on the half second reads -0.50).

**The substance**: the expected-time table's ``xt_tot`` (lengths against the standard, higher faster), a pre-race
quantity.

**The surplus.** Within today's race, z-score (less the race's mean, over the race's sd, 0 when the sd is 0; over
the runners that have the column) each of ``op_best_vs``, minus ``op_pos_sum`` and ``op_wins6``; ``op_look`` = the
mean of those a dog has; ``op_real`` = the z-score of ``xt_tot``; ``op_surplus`` = ``op_look`` - ``op_real``.

**Luck in the digits.** For every earlier race the dog's expected finishing rank before it = the rank of its
``xt_tot`` in that race (1 the fastest expected, ties the mean rank), and its actual position. ``op_luck6`` = the
mean over its lines of (expected rank - position): above 0 it has been finishing better than expected and the
digits flatter it. ``op_photo6`` = how many of its lines were settled by ``PHOTO`` lengths or less: its own beaten
distance (to the dog in front) or that of the dog placed immediately behind it, or a dead heat.

The expected rank is among the race's runners that have an ``xt_tot`` and the position is among all of them: the
two differ where a runner has no expected time (the build's notes count those lines).

The race-relative columns are made from the pre-race columns of today's runners only. Every column is read from
earlier days and today's card (tests/test_greyhound_optics.py: the last day's and a middle day's results rewritten,
the later days cut off, a card row, and the specification read a row at a time).

**What it inherits.** ``op_real``, ``op_surplus`` and ``op_luck6`` are as point-in-time as the expected-time table.
That table chooses its three filter constants on the days before ``xtime.FIT_BEFORE`` (2018) and, where a store
starts later, on the first fifth of the frame it is given: on such a store the rows of that first fifth are priced
with constants chosen on their own period, and every row's value can change when the store grows (the review of
11 Nov 2026: the 16-month store cut at 31 Dec 2025 chose other constants and moved ``op_real`` on 126,003 of
135,511 rows). On the full store (2014 on) the window is fixed at 2014 to 2017 and the rows from 2018 are clean.
The other twelve columns read no fitted quantity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core, figs as figs_mod

PREFIX = "op_"
SOURCE = ("GBGB results (calculated time, position, distance beaten: sources/gbgb.py) of the dog's last six race "
          "days, against the expected-time table (xt_tot)")
N_CARD = 6                   # the lines a card prints
MIN_POS = 3                  # form figures need this many lines
PHOTO = 0.25                 # lengths: a head or less (sources/gbgb.py reads Hd 0.25, SH 0.125, DH 0; a neck 0.375)
PACE = (0.050, 0.080)        # seconds a metre a greyhound's calculated time lies in (20 to 12.5 m/s); the store of
#                              1 Jul 2025 to 5 Nov 2026 has 16 times outside (3 races, keying errors) of 346,695, the
#                              rest 0.0559 to 0.0754
BIG = 1 << 21                # day numbers are below this: a dog's block of the (dog, day) key
EPS = 1e-9


def race_z(v: np.ndarray, rc: np.ndarray, nr: int) -> np.ndarray:
    """``v`` less its race's mean over its race's sd (the runners that have a value; 0 when the sd is 0)."""
    ok = np.isfinite(v)
    n = np.bincount(rc[ok], minlength=nr).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(rc[ok], weights=v[ok], minlength=nr) / n
        dev = v - mean[rc]
        sd = np.sqrt(np.bincount(rc[ok], weights=dev[ok] ** 2, minlength=nr) / n)[rc]
        z = np.where(sd > EPS, dev / np.where(sd > EPS, sd, 1.0), 0.0)
    return np.where(ok, z, np.nan)


def photo_flags(rc: np.ndarray, pos: np.ndarray, beaten: np.ndarray) -> np.ndarray:
    """1 where the run won or lost a place by ``PHOTO`` lengths or less: its own distance to the dog in front, the
    distance of the dog placed immediately behind it, or a dead heat. NaN with no position."""
    has = np.isfinite(pos)
    o = np.lexsort((np.where(has, pos, np.inf), rc))
    r, p, b, h = rc[o], pos[o], beaten[o], has[o]
    nxt = (r[1:] == r[:-1]) & h[1:] & h[:-1]                         # row k + 1 is placed behind (or with) row k
    behind = np.r_[np.where(nxt, b[1:], np.nan), np.nan]
    tie = np.r_[nxt & (p[1:] == p[:-1]), False]
    tie = tie | np.r_[False, tie[:-1]]
    with np.errstate(invalid="ignore"):
        f = ((b <= PHOTO) & (p > 1)) | (behind <= PHOTO) | tie
    out = np.full(len(rc), np.nan)
    out[o] = np.where(h, f.astype(float), np.nan)
    return out


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["t", "race_id", "trap"], kind="stable")
    xt = (ctx or {}).get("xtime")
    if xt is None:
        from greyhound.pit import xtime
        xt, _ = xtime.build(runs, df)
    j = df[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_tot"]].drop_duplicates(["race_id", "dog_id"]),
                                        on=["race_id", "dog_id"], how="left")
    real = j.xt_tot.to_numpy(float)
    n = len(df)
    rc = pd.factorize(df.race_id)[0]
    nr = int(rc.max()) + 1 if n else 0
    dog = pd.factorize(df.dog_id)[0].astype(np.int64)
    dn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(np.int64)
    td = pd.factorize(figs_mod._key(df))[0]
    card = df.is_card.fillna(False).to_numpy(bool) if "is_card" in df else np.zeros(n, bool)
    pos = pd.to_numeric(df.position, errors="coerce").to_numpy(float)
    pos = np.where((pos >= 1) & ~card, pos, np.nan)
    time = pd.to_numeric(df.adjusted_time, errors="coerce").to_numpy(float)
    time = np.where((time > 0) & ~card, time, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pace = time / pd.to_numeric(df.distance_m, errors="coerce").to_numpy(float)
        odd = np.isfinite(pace) & ((pace < PACE[0]) | (pace > PACE[1]))             # not a greyhound's time
    time = np.where(odd, np.nan, time)
    beaten = pd.to_numeric(df.beaten_lengths, errors="coerce").to_numpy(float) if "beaten_lengths" in df else \
        np.full(n, np.nan)
    blank = np.zeros(n, bool)
    if "beaten" in df:                                              # GBGB's text: an empty distance is not a dead heat
        blank = (df.beaten.fillna("").astype(str).str.strip() == "").to_numpy() & (pos > 1)
        beaten = np.where(blank, np.nan, beaten)

    # ---- each run's own result, to be read on LATER days only
    won = (pos == 1).astype(float)
    exp_rank = pd.Series(real).groupby(rc).rank(ascending=False, method="average").to_numpy(float)
    luck = exp_rank - pos                                           # above 0: finished better than expected
    n_x = np.bincount(rc[np.isfinite(real)], minlength=nr)
    n_p = np.bincount(rc[np.isfinite(pos)], minlength=nr)
    n_xp = np.bincount(rc[np.isfinite(real) & np.isfinite(pos)], minlength=nr)
    unranked = np.isfinite(luck) & ((n_x != n_xp) | (n_p != n_xp))[rc]          # a runner with no xt_tot or no position
    photo = photo_flags(rc, pos, beaten)

    # ---- the lines: one a dog-day with a position, the day's last run
    line = np.isfinite(pos)
    li = np.flatnonzero(line)
    key_l = dog[li] * BIG + dn[li]
    o = np.argsort(key_l, kind="stable")                             # rows are in time order: a day's last run is last
    li, key_l = li[o], key_l[o]
    last = np.r_[key_l[1:] != key_l[:-1], True] if len(li) else np.zeros(0, bool)
    two = int((~last).sum())
    li, key_l = li[last], key_l[last]
    l_dog = dog[li]
    first = np.searchsorted(key_l, dog * BIG + dn, side="left")      # the dog's lines before today start below this

    lines = np.zeros(n)
    best = np.full(n, np.inf)
    pos_sum, pos_n, wins, luck_sum, luck_n, photos = (np.zeros(n) for _ in range(6))
    pos_last = lto_won = np.full(n, np.nan)
    for lag in range(1, N_CARD + 1):
        q = first - lag
        ok = q >= 0
        ok[ok] = l_dog[q[ok]] == dog[ok]
        src = li[np.where(ok, q, 0)] if len(li) else np.zeros(n, dtype=np.int64)       # the line's row
        lines += ok
        best = np.fmin(best, np.where(ok & (td[src] == td), time[src], np.nan))
        p = np.where(ok, pos[src], np.nan)
        pos_sum += np.nan_to_num(p)
        pos_n += np.isfinite(p)
        wins += np.where(ok, won[src], 0.0)
        lk = np.where(ok, luck[src], np.nan)
        luck_sum += np.nan_to_num(lk)
        luck_n += np.isfinite(lk)
        photos += np.nan_to_num(np.where(ok, photo[src], np.nan))
        if lag == 1:
            pos_last, lto_won = p, np.where(ok, won[src], np.nan)
    some = lines > 0
    best = np.where(np.isfinite(best), best, np.nan)
    has = np.isfinite(best)

    # ---- the look against today's field
    cnt = np.bincount(rc[has], minlength=nr)
    tot = np.bincount(rc[has], weights=best[has], minlength=nr)
    with np.errstate(invalid="ignore", divide="ignore"):
        others = np.where(cnt[rc] - has > 0, (tot[rc] - np.where(has, best, 0.0)) / (cnt[rc] - has), np.nan)
        best_vs = np.where(has, (others - best) / core.LENGTH_S, np.nan)
        pos6 = np.where(pos_n >= MIN_POS, N_CARD * pos_sum / np.maximum(pos_n, 1), np.nan)
        luck6 = np.where(luck_n > 0, luck_sum / np.maximum(luck_n, 1), np.nan)
    low = np.full(nr, np.inf)
    np.minimum.at(low, rc[has], best[has])
    fastest = np.where(has, (best <= low[rc] + EPS).astype(float), np.nan)
    wins6 = np.where(some, wins, np.nan)
    hund = np.where(has, np.rint(np.where(has, best, 0.0) * 100.0) % 100, np.nan)          # the printed hundredths
    z = np.vstack([race_z(best_vs, rc, nr), race_z(-pos6, rc, nr), race_z(wins6, rc, nr)])
    parts = np.isfinite(z).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        look = np.where(parts > 0, np.nansum(z, axis=0) / np.maximum(parts, 1), np.nan)
    z_real = race_z(real, rc, nr)

    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    out[f"{PREFIX}best"] = best
    out[f"{PREFIX}best_vs"] = best_vs
    out[f"{PREFIX}fastest"] = fastest
    out[f"{PREFIX}pos_last"] = pos_last
    out[f"{PREFIX}pos_sum"] = pos6
    out[f"{PREFIX}wins6"] = wins6
    out[f"{PREFIX}lto_won"] = lto_won
    out[f"{PREFIX}round_under"] = np.where(has, (hund >= 90).astype(float), np.nan)
    out[f"{PREFIX}round_over"] = np.where(has, (hund <= 9).astype(float), np.nan)
    out[f"{PREFIX}round_dist"] = np.where(has, (((hund + 50) % 100) - 50) / 100.0, np.nan)
    out[f"{PREFIX}look"] = look
    out[f"{PREFIX}real"] = z_real
    out[f"{PREFIX}surplus"] = look - z_real
    out[f"{PREFIX}luck6"] = luck6
    out[f"{PREFIX}photo6"] = np.where(some, photos, np.nan)
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)

    def share(m) -> float | None:
        return round(float(np.mean(m)), 4) if n else None

    notes = {"n_card": N_CARD, "min_pos": MIN_POS, "photo_lengths": PHOTO, "rows": int(n),
             "lines": int(len(li)), "dog_days_with_a_second_run_dropped": two,
             "rows_with_a_line": share(some), "rows_with_six_lines": share(lines == N_CARD),
             "rows_with_a_best_time_here": share(has), "rows_with_a_rival_to_compare": share(np.isfinite(best_vs)),
             "rows_with_a_look": share(np.isfinite(look)), "rows_with_a_surplus": share(np.isfinite(look - z_real)),
             "look_parts_mean": round(float(parts.mean()), 3) if n else None,
             "lines_settled_by_a_photo": round(float(np.nanmean(photo[li])), 4) if len(li) else None,
             "lines_mean_luck": round(float(np.nanmean(luck[li])), 4) if len(li) and np.isfinite(luck[li]).any()
             else None,
             "best_just_under_a_second": round(float((hund[has] >= 90).mean()), 4) if has.any() else None,
             "best_just_over_a_second": round(float((hund[has] <= 9).mean()), 4) if has.any() else None,
             "non_winners_with_no_distance": int(blank.sum()),
             "times_no_greyhound_runs_left_out": int(odd.sum()),
             "lines_with_luck_in_a_race_with_a_runner_unranked": int(unranked[li].sum())}
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}best": f"the look: its fastest calculated time (seconds) in its last {N_CARD} race days at today's "
                     f"track and distance (NaN with none there; a time outside {PACE[0]:g} to {PACE[1]:g} s a metre is "
                     "not read)",
    f"{PREFIX}best_vs": "the mean of the other runners' best less its own, in lengths (above 0 the time-finder's pick)",
    f"{PREFIX}fastest": "1 when it has the race's fastest best time (NaN with none)",
    f"{PREFIX}pos_last": "its last finishing position",
    f"{PREFIX}pos_sum": f"the form figures added up at the rate of six lines: {N_CARD} x the mean finishing position "
                        f"of its last {N_CARD} race days (NaN with fewer than {MIN_POS})",
    f"{PREFIX}wins6": f"wins in its last {N_CARD} race days",
    f"{PREFIX}lto_won": "1 if it won last time",
    f"{PREFIX}round_under": "1 when the best time's hundredths are .90 to .99: just under a whole second",
    f"{PREFIX}round_over": "1 when they are .00 to .09: just over a whole second (the left-digit illusion)",
    f"{PREFIX}round_dist": "the best time less the nearest whole second (seconds, -0.50 to 0.49, below 0 just under)",
    f"{PREFIX}look": "the mean of the within-race z-scores of the best time against the field, the form figures "
                     "(negated) and the wins, over those it has",
    f"{PREFIX}real": "the substance: the within-race z-score of its expected whole-race time (xt_tot, higher faster)",
    f"{PREFIX}surplus": "the look less the substance (above 0 the card flatters the dog)",
    f"{PREFIX}luck6": f"the mean over its last {N_CARD} race days of its expected finishing rank before the race (the "
                      "rank of its xt_tot in that race) less its position (above 0 it has been finishing better than "
                      "its expected times said; on the real runs that persists rather than reverts)",
    f"{PREFIX}photo6": f"how many of its last {N_CARD} runs were settled by {PHOTO:g} lengths or less: its own beaten "
                       "distance, that of the dog placed immediately behind it, or a dead heat",
}
