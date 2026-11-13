"""The card readers: the price as a pool of punters, each reading the printed card by one fast-and-frugal rule.

One target is the Betfair SP, a crowd product. A good part of the crowd does not weigh the card: it applies one rule
to it and backs the rule's pick. This table makes six such rules, ranks today's field by each (1 = the rule's pick)
and measures how far the rules agree: a dog every rule picks is the obvious one, a dog that is the best on average
without being any rule's pick is the one the single-rule readers pass over.

- **The rules** (each read off the dog's last six runs, the lines the card prints, or the track's record):
  ``pos`` the best finishing position last time out; ``beat`` the fewest lengths beaten last time out; ``days`` the
  most recent run; ``wins`` the most wins in the last six; ``time`` the fastest calculated time in the last six at
  today's track and trip (a time no greyhound runs, outside ``PACE`` seconds a metre, is a keying error and is not
  read: 5 of the 350,818 local runs of 1 Jul 2025 to 5 Nov 2026); ``trap`` the trap with the best win rate at today's track (at least ``TRAP_MIN`` earlier
  runs from it).
- **The field** of a race is every row of ``runs`` with its race_id, a card's runners included (the screen ranked
  only the runners with a Betfair SP: the ranks of the two differ where a race had an unpriced runner).
- **Point in time**: a rule reads only the dog's runs dated strictly before the race's day (a run earlier the same
  day is not on the card), the trap's record on earlier days only, and the card's track, distance, trap and field.
  A card row (``is_card``: no result yet) is never history and gets its features as a run does. A run with no
  finishing position (a non-finisher) is a line: it dates the last run and gives the position and distance rules
  nothing to read. No column reads the
  weigh-in or the price: none is card-unsafe.

Screened on 12 Nov 2026 against what the stack still misses (the log miss of the stacked BFSP forecast, 1 Jun to
10 Nov 2026, 110,788 priced runners: the scratch study disc2/readers, not in the repository): the set took the held
out RMSE of the miss from 0.2872 to 0.2864 (z -4.7, 49,612 runners; a noise set of the same width +0.00027), and
from 0.2854 to 0.2849 beside the optics table. The single columns' rank correlations with the miss: the time rule's
rank 0.056, the mean rank 0.051, the beaten-distance rule's 0.040, the number of rules picking the dog -0.037, the
trap rule's 0.030; the position, wins and days rules, the spread and the two flags under 0.01 alone (kept: they are
the terms the mean, the count and the spread are made of, and a tree reads them together). A screen, not a
registered trial: nothing here is a result until the fast loop scores the table on both targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "rd_"
SOURCE = "GBGB results (the last six runs before the day, the track's trap record) and the card (track, trip, trap, field)"
LINES = 6                              # the card prints six lines
TRAP_MIN = 100                         # earlier runs from the trap at the track before its win rate is read
RULES = ("pos", "time", "wins", "trap", "days", "beat")
PACE = (0.050, 0.080)                  # seconds a metre a greyhound's calculated time lies in (optics.py's bounds)
BIG = 1_000_000                        # day numbers are below this: a dog's block of the sorted (dog, day) key


def rule_values(runs: pd.DataFrame) -> pd.DataFrame:
    """Each rule's raw reading of each row of ``runs`` (lower = the rule likes the dog more), aligned to
    ``runs.index``; NaN where the rule has nothing to read."""
    n = len(runs)
    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    num = lambda c: (pd.to_numeric(runs[c], errors="coerce").to_numpy(float) if c in runs  # noqa: E731
                     else np.full(n, np.nan))
    dn = (core.day(runs) - pd.Timestamp("2000-01-01")).dt.days.to_numpy().astype(np.int64)
    dog = pd.factorize(runs.dog_id)[0].astype(np.int64)
    track = pd.factorize(runs.track.astype(str))[0]
    dist = num("distance_m")
    # the history: every run with a result (a card row is no line), each dog's in time order
    order = np.lexsort((np.arange(n), runs.t.to_numpy(), dn, dog))          # by day first: the key below is sorted
    order = order[~card[order]]
    hkey = dog[order] * BIG + dn[order]
    block = np.searchsorted(hkey, dog * BIG, side="left")
    before = np.searchsorted(hkey, dog * BIG + dn, side="left") - block       # runs on strictly earlier days
    h_pos, h_beat, h_won = num("position")[order], num("beaten_lengths")[order], num("won")[order]
    h_time, h_dn, h_track, h_dist = num("adjusted_time")[order], dn[order], track[order], dist[order]
    with np.errstate(invalid="ignore", divide="ignore"):
        pace = h_time / h_dist
        h_time = np.where((pace >= PACE[0]) & (pace <= PACE[1]), h_time, np.nan)   # not a greyhound's time: not read
    has = before >= 1
    last = np.where(has, block + before - 1, 0)
    v = {"pos": np.where(has, h_pos[last], np.nan) if len(order) else np.full(n, np.nan),
         "beat": np.where(has, h_beat[last], np.nan) if len(order) else np.full(n, np.nan),
         "days": np.where(has, dn - h_dn[last], np.nan) if len(order) else np.full(n, np.nan)}
    wins = np.zeros(n)
    best = np.full(n, np.inf)
    for j in range(1, LINES + 1):
        ok = before >= j
        if not len(order):
            break
        i = np.where(ok, block + before - j, 0)
        wins += np.where(ok, np.nan_to_num(h_won[i], nan=0.0), 0.0)
        here = ok & (h_track[i] == track) & (h_dist[i] == dist) & np.isfinite(h_time[i])
        best = np.where(here, np.minimum(best, h_time[i]), best)
    v["wins"] = np.where(has, -wins, np.nan)
    v["time"] = np.where(np.isfinite(best), best, np.nan)
    # the trap's win rate at the track on every earlier day
    won = num("won")
    known = np.isfinite(won) & ~card
    d = pd.DataFrame({"race_date": runs.race_date.to_numpy(), "track": runs.track.astype(str).to_numpy(),
                      "trap": num("trap"), "_w": np.where(known, won, 0.0), "_n": known.astype(float)})
    d["trap"] = d.trap.fillna(-1.0)
    w = core.prior_window(d, ["track", "trap"], ["_w", "_n"], None)
    seen = w._n.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        v["trap"] = np.where((seen >= TRAP_MIN) & (d.trap.to_numpy() >= 0), -w._w.to_numpy() / seen, np.nan)
    return pd.DataFrame({k: v[k] for k in RULES}, index=runs.index)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    v = rule_values(runs).reset_index(drop=True)                 # by position: the frame's index may repeat
    race = runs.race_id.to_numpy()
    g = v.groupby(race, sort=False)
    R = g.rank(method="average")
    top = (g.rank(method="min") == 1)
    any_rule = R.notna().any(axis=1)
    out = pd.DataFrame({"race_id": race, "dog_id": runs.dog_id.to_numpy()})
    for k in RULES:
        out[f"{PREFIX}rk_{k}"] = R[k]
    ntop = top.sum(axis=1).astype(float).where(any_rule)
    mean = R.mean(axis=1)
    sd = R.std(axis=1, ddof=0).where(R.notna().sum(axis=1) >= 3)
    out[f"{PREFIX}ntop"], out[f"{PREFIX}mean"], out[f"{PREFIX}sd"] = ntop, mean, sd
    out[f"{PREFIX}mean_rk"] = mean.groupby(race, sort=False).rank(method="average")
    at_max = ntop == ntop.groupby(race, sort=False).transform("max")
    obv = at_max & (at_max.groupby(race, sort=False).transform("sum") == 1)      # most rules' pick, and alone in that
    best = mean.groupby(race, sort=False).rank(method="min") == 1
    ok = mean.notna()
    out[f"{PREFIX}obv_notbest"] = (obv & ~best).astype(float).where(ok)
    out[f"{PREFIX}best_notobv"] = (best & ~obv).astype(float).where(ok)
    out[f"{PREFIX}race_sd"] = sd.groupby(race, sort=False).transform("mean")
    cols = list(COLUMNS)
    out = out[["race_id", "dog_id"] + cols]
    out[cols] = out[cols].astype(np.float32)
    notes = {"rows": int(len(out)), "races": int(pd.unique(race).size),
             "rows_with_a_rule": round(float(any_rule.mean()), 4) if len(out) else None,
             "rule_coverage": {k: round(float(v[k].notna().mean()), 4) for k in RULES} if len(out) else {},
             "obvious_not_best": round(float(np.nanmean(out[f"{PREFIX}obv_notbest"])), 4) if ok.any() else None,
             "best_not_obvious": round(float(np.nanmean(out[f"{PREFIX}best_notobv"])), 4) if ok.any() else None,
             "lines": LINES, "trap_min": TRAP_MIN}
    return out, notes                                             # one row a row of runs, in the runs' order


COLUMNS = {
    f"{PREFIX}rk_pos": "the dog's rank in the field by its last finishing position (1 = the best last time out; ties "
                       "share the mean rank)",
    f"{PREFIX}rk_time": f"its rank by the fastest calculated time in its last {LINES} runs at today's track and trip",
    f"{PREFIX}rk_wins": f"its rank by wins in its last {LINES} runs (1 = the most)",
    f"{PREFIX}rk_trap": f"its rank by its trap's win rate at today's track on every earlier day (at least {TRAP_MIN} "
                        f"runs from the trap)",
    f"{PREFIX}rk_days": "its rank by days since its last run (1 = the most recent)",
    f"{PREFIX}rk_beat": "its rank by the lengths it was beaten last time out (1 = the fewest)",
    f"{PREFIX}ntop": "how many of the six rules pick the dog (rank it first or joint first)",
    f"{PREFIX}mean": "the mean of its ranks over the rules that can read it",
    f"{PREFIX}sd": "the spread of its ranks over the rules (three or more): how far the readers disagree about the dog",
    f"{PREFIX}mean_rk": "its rank in the field by that mean rank",
    f"{PREFIX}obv_notbest": "1 when it alone is picked by the most rules yet is not the best on the mean rank",
    f"{PREFIX}best_notobv": "1 when it is the best on the mean rank yet is not the one dog most rules pick",
    f"{PREFIX}race_sd": "the mean of the field's spreads: how far the readers disagree over the race",
}
