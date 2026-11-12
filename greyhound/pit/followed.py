"""The followed dog: the attention a dog carries into a race from its own starting-price history.

A dog that has been favourite, has gone off short and has let its backers down is a "name": the crowd has a view of
it before the card is read, and that view is priced on top of the form. This table measures the name from the SP
history alone (the SP, the market rank, the finishing position and the win of the dog's EARLIER runs), and sets it
against the other names in today's race.

Screened on 12 Nov 2026 against what the stack still misses (the stack's miss on the Betfair SP, 49,612 runners:
the set took the miss's RMSE from 0.2872 to 0.2865, z -5.5, against a noise floor of +0.0003; beyond the optics
table 0.2854 to 0.2849). The screen's script read the same history; this table differs from it in one thing, the
field (below).

**The history.** A dog's runs on days strictly before today's, each with an SP, a market rank and a finishing
position (a non-finisher, a void race and a trial are no run here; a card row is never history). The runs are in
time order and the windows count RUNS; a dog that ran twice on an earlier day has both runs read.

**The dog** (NaN with no such run, except ``fo_n``, which is 0):

- ``fo_n``: the runs; ``fo_fav_rate``, ``fo_fav_rate_l5``: the share it started favourite (market rank 1), in all
  and in its last five; ``fo_fav_streak``, ``fo_short_streak``: the consecutive runs to its last it was favourite,
  and at ``SHORT`` (2/1) or shorter;
- ``fo_beaten_fav_rate`` (NaN until it has been favourite), ``fo_beaten_fav_l5``, ``fo_last_beaten_fav``: the
  favourites it lost, as a share of its favourites, as a count in its last five runs, and last time;
- ``fo_overbet``, ``fo_overbet_l5``: the mean of (finishing position - market rank), above 0 a dog that finishes
  behind where the money put it; ``fo_ae``: wins less the SP's normalised chances, a run;
- ``fo_logsp_l5``: the mean log SP of its last five runs; ``fo_days_short_win``: days since it last won at 2/1 or
  shorter (capped at 365; NaN if it never has).

**The field** is every row of ``runs`` with today's ``race_id`` (the screen's field was the runners with a Betfair
price only): ``fo_oth_fav_rate_l5`` is the mean ``fo_fav_rate_l5`` of the other runners that have one, and
``fo_name_gap`` the dog's own less the highest of the others' (0 where no other has one): above 0 it is the only
name in the race.

Nothing of today is read but the race's dogs: no column is card-unsafe (tests/test_greyhound_followed.py: a day's
results rewritten, the later days cut off, a card row, a row-at-a-time reading of this specification).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "fo_"
SOURCE = ("GBGB results (starting price, market rank, finishing position: sources/gbgb.py) of the dog's earlier "
          "runs, and of the other dogs in today's race")
SHORT = 3.0                  # a "short price": an SP of 2/1 (3.0 decimal) or shorter
LAST = 5                     # the recent window, in runs
CAP_DAYS = 365.0
STATE = ["n", "fav_rate", "fav_rate_l5", "short_streak", "fav_streak", "beaten_fav_rate", "beaten_fav_l5",
         "last_beaten_fav", "overbet", "overbet_l5", "ae", "logsp_l5"]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def states(runs: pd.DataFrame) -> pd.DataFrame:
    """Each dog's state after its last run of each day it ran (``dog_id``, ``_d``, the ``STATE`` columns and
    ``short_win_d``, the day of its last win at a short price), from the runs with an SP, a market rank and a
    position."""
    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(len(runs), bool)
    sp, mp, pos = _num(runs, "sp_decimal"), _num(runs, "market_pos"), _num(runs, "position")
    ok = (sp > 1.0) & mp.notna() & (pos >= 1) & ~card & core.day(runs).notna()
    r = pd.DataFrame({"dog_id": runs.dog_id, "t": runs.t, "_d": core.day(runs), "sp": sp, "mp": mp, "pos": pos,
                      "exp": _num(runs, "sp_p_norm").fillna(0.0)})[ok.to_numpy()]
    # by day first: a run with no time of day stays on its own day (sorted on the time alone it would fall after
    # every later run of the dog, and its day's state would hold them)
    r = r.sort_values(["dog_id", "_d", "t"], kind="stable").reset_index(drop=True)
    g = r.dog_id
    k = r.groupby("dog_id", sort=False).cumcount().astype(float)
    fav = (r.mp == 1).astype(float)
    short = (r.sp <= SHORT).astype(float)
    won = (r.pos == 1).astype(float)
    over = r.pos - r.mp                                   # above 0: finished behind its market rank

    def cum(v: pd.Series) -> pd.Series:
        return v.groupby(g, sort=False).cumsum()

    def last(v: pd.Series) -> pd.Series:                  # the sum over the dog's last LAST runs to here
        c = cum(v)
        return c - c.groupby(g, sort=False).shift(LAST).fillna(0.0)

    def streak(flag: pd.Series) -> pd.Series:             # the consecutive runs with the flag, ending at this run
        return k - k.where(flag == 0).groupby(g, sort=False).ffill().fillna(-1.0)

    n = k + 1.0
    n5 = np.minimum(n, LAST)
    nf = cum(fav)
    beaten = fav * (1.0 - won)
    h = pd.DataFrame({"dog_id": r.dog_id, "_d": r._d})
    h["n"] = n
    h["fav_rate"] = nf / n
    h["fav_rate_l5"] = last(fav) / n5
    h["short_streak"] = streak(short)
    h["fav_streak"] = streak(fav)
    h["beaten_fav_rate"] = (cum(beaten) / nf).where(nf > 0)
    h["beaten_fav_l5"] = last(beaten)
    h["last_beaten_fav"] = beaten
    h["overbet"] = cum(over) / n
    h["overbet_l5"] = last(over) / n5
    h["ae"] = (cum(won) - cum(r.exp)) / n
    h["logsp_l5"] = last(np.log(r.sp)) / n5
    h["short_win_d"] = r._d.where((short == 1) & (won == 1)).groupby(g, sort=False).ffill()
    return h.drop_duplicates(["dog_id", "_d"], keep="last")


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    n = len(runs)
    h = states(runs).sort_values("_d", kind="stable")
    q = pd.DataFrame({"dog_id": runs.dog_id.to_numpy(), "_q": core.day(runs).to_numpy(), "_i": np.arange(n)})
    m = pd.merge_asof(q.dropna(subset=["_q"]).sort_values("_q", kind="stable"), h, left_on="_q", right_on="_d",
                      by="dog_id", allow_exact_matches=False)          # the dog's state on its last EARLIER day
    assert (m._d.isna() | (m._d < m._q)).all()
    m["days_short_win"] = (m._q - m.short_win_d).dt.days.clip(upper=CAP_DAYS)
    m = m.set_index("_i").reindex(np.arange(n))
    val = {c: m[c].to_numpy(float) for c in STATE + ["days_short_win"]}
    seen = np.isfinite(val["n"])
    val["n"] = np.where(seen, val["n"], 0.0)

    # ---- the field: every row of runs in the race
    rc = pd.factorize(runs.race_id)[0]
    nr = int(rc.max()) + 1 if n else 0
    own = val["fav_rate_l5"]
    has = np.isfinite(own)
    cnt = np.bincount(rc[has], minlength=nr)[rc] - has
    tot = np.bincount(rc[has], weights=own[has], minlength=nr)[rc] - np.where(has, own, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        oth = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    f0 = np.where(has, own, -1.0)
    o = np.lexsort((-f0, rc))                                          # each race's rows, the highest first
    first = np.r_[True, rc[o][1:] != rc[o][:-1]] if n else np.zeros(0, bool)
    start = np.flatnonzero(first)
    size = np.diff(np.r_[start, n])
    top = np.full(nr, np.nan)
    second = np.full(nr, np.nan)
    if n:
        top[rc[o][start]] = f0[o][start]
        two = size > 1
        second[rc[o][start[two]]] = f0[o][start[two] + 1]
    with np.errstate(invalid="ignore"):
        best_other = np.where(f0 >= top[rc], second[rc], top[rc])      # the highest of the others; NaN when alone
        gap = own - np.maximum(best_other, 0.0)                        # np.maximum keeps the NaN of a field of one
    gap = np.where(np.isfinite(best_other), gap, np.nan)

    out = pd.DataFrame({"race_id": runs.race_id.to_numpy(), "dog_id": runs.dog_id.to_numpy()})
    for c in STATE + ["days_short_win"]:
        out[f"{PREFIX}{c}"] = val[c]
    out[f"{PREFIX}oth_fav_rate_l5"] = oth
    out[f"{PREFIX}name_gap"] = gap
    cols = list(COLUMNS)
    out = out[["race_id", "dog_id"] + cols]
    out[cols] = out[cols].astype(np.float32)

    def share(x) -> float | None:
        return round(float(np.mean(x)), 4) if n else None

    card = runs.is_card.fillna(False).to_numpy(bool) if "is_card" in runs else np.zeros(n, bool)
    notes = {"short_price": SHORT, "last_runs": LAST, "rows": int(n), "card_rows": int(card.sum()),
             "history_dog_days": int(len(h)), "rows_with_a_history": share(seen),
             "rows_with_a_favourite_before": share(np.isfinite(val["beaten_fav_rate"])),
             "rows_with_a_short_win_before": share(np.isfinite(val["days_short_win"])),
             "rows_with_another_name_to_compare": share(np.isfinite(oth)),
             "rows_the_only_name": share(np.nan_to_num(gap, nan=-1.0) > 0),
             "has_market_pos": "market_pos" in runs, "has_sp_p_norm": "sp_p_norm" in runs}
    return out, notes


COLUMNS = {
    f"{PREFIX}n": "its earlier runs with an SP, a market rank and a finishing position (0 with none)",
    f"{PREFIX}fav_rate": "the share of those runs it started favourite (market rank 1)",
    f"{PREFIX}fav_rate_l5": f"the same over its last {LAST} runs",
    f"{PREFIX}short_streak": f"the consecutive runs, to its last, it started at an SP of {SHORT:g} or shorter",
    f"{PREFIX}fav_streak": "the consecutive runs, to its last, it started favourite",
    f"{PREFIX}beaten_fav_rate": "the share of its favourites it lost (NaN until it has been favourite)",
    f"{PREFIX}beaten_fav_l5": f"how many of its last {LAST} runs it was a beaten favourite",
    f"{PREFIX}last_beaten_fav": "1 if it was a beaten favourite last time",
    f"{PREFIX}overbet": "the mean of (finishing position - market rank) over its runs: above 0 it finishes behind "
                        "where the money put it",
    f"{PREFIX}overbet_l5": f"the same over its last {LAST} runs",
    f"{PREFIX}ae": "its wins less the SP's normalised chances, a run (below 0 it has won less than its prices said)",
    f"{PREFIX}logsp_l5": f"the mean log SP (decimal) of its last {LAST} runs",
    f"{PREFIX}days_short_win": f"days since it last won at an SP of {SHORT:g} or shorter (capped at {CAP_DAYS:g}; "
                               "NaN if it never has)",
    f"{PREFIX}oth_fav_rate_l5": f"the mean favourite share over the last {LAST} runs of the other dogs in today's "
                                "race (every row of the race; NaN when none has a history)",
    f"{PREFIX}name_gap": f"its favourite share over its last {LAST} runs less the highest of the others' (0 where no "
                         "other has one): above 0 it is the only name in the race",
}
