"""The GBGB run table, cleaned for modelling: one row a dog a race, in time order.

Reads ``gbgb/runs_<year>.parquet`` from the sources store (S3, or a local folder with ``root``). Drops the rows a model
cannot use (vacant traps, no dog, no finishing position, races that were not run), and adds the race key, the off time,
the field, the class on one ladder, the handicap and its start, and the checks on the distances and the allowance.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from sources.common import Store
from sources.gbgb import lengths as gbgb_lengths

#: GBGB's classes on one ladder, lower is better. A, B, D, S, P and H are each a track's own ladder for one trip
#: (A the standard trip, B a second standard trip - Doncaster 450 m, Towcester 460 m - D the sprint, S the stayers',
#: P puppies, H hurdles), numbered from 1, so the number is the grade. The open classes sit above A1: over 1 Jul 2025
#: to 5 Nov 2026 the winners' calculated time against their track and trip's median, in lengths (- faster), was
#: OR1 -8.5 (729 races), OR2 -7.1 (275), OR3 -5.4 (1,437), IV -5.0 (457), A1 -4.6 (2,226), OR -4.4 (4,067),
#: A2 -3.3, A3 -2.1, A4 -1.1, A5 +0.3 - a grade of A is about 1.2 lengths - which puts OR1 at -2, OR2 at -1, OR3 at 0
#: and OR and IV level with A1. B1 (-4.5 at its trips) is level with A1, not two below it. A handicap (HP) and an
#: intertrack race (IT) have no place on the ladder: a handicap mixes classes by design and an intertrack race is
#: made between two tracks. Until 11 Nov 2026 OR1-OR3 were coded as A1-A3, OR, HP and IV as 0 (the top) and B as two
#: grades below A (the audit of 11 Nov 2026, findings 3 and 53)
GRADE_LADDER = {"A": 0.0, "B": 0.0, "D": 0.0, "S": 0.0, "P": 0.0, "H": 0.0}
GRADE_FIXED = {"OR1": -2.0, "OR2": -1.0, "OR3": 0.0, "OR": 1.0, "IV": 1.0}
#: the ladder a class belongs to: a change of grade is read only within one (an A3 to a D2 is a change of trip, not of
#: class); the open classes share the standard trip's ladder
GRADE_SCALE = {"A": "A", "OR": "A", "IV": "A", "B": "B", "D": "D", "S": "S", "P": "P", "H": "H"}


#: one name a track. GBGB's results carry the name of the day, so a rename started every track-keyed history again
#: (the standards, the dog's record at the track, the trap bias, the grade pars) and broke the join to Betfair's
#: files where the two changed name on different days: Pelaw Grange became Star Pelaw in December 2024, inside the
#: live model's fit window; The Valley is Valley; Sittingbourne became Central Park in June 2016; Mildenhall became
#: Suffolk Downs (research/greyhound/reference/README.md, each with its source; the audit of 11 Nov 2026,
#: finding 56). The current name is the key, on the results, the trials, the card and the price files alike
TRACK_NAMES = {"Pelaw Grange": "Star Pelaw", "The Valley": "Valley", "Sittingbourne": "Central Park",
               "Mildenhall": "Suffolk Downs"}


def canonical_track(track: pd.Series) -> pd.Series:
    return track.map(lambda x: TRACK_NAMES.get(x.strip(), x.strip()) if isinstance(x, str) else x)


def _class(c) -> str:
    return str(c or "").strip().upper()


def grade_rank(c) -> float:
    c = _class(c)
    if c in GRADE_FIXED:
        return GRADE_FIXED[c]
    m = re.match(r"([A-Z]+)(\d+)$", c)
    if not m or m.group(1) not in GRADE_LADDER:
        return np.nan
    return GRADE_LADDER[m.group(1)] + float(m.group(2))


def grade_family(c) -> str:
    """The grade's letter: A (standard), D (sprint), S (stayers), H (hurdles), OR, ..."""
    m = re.match(r"([A-Z]+)", _class(c))
    return m.group(1) if m else "?"


def grade_scale(c) -> str | None:
    """The ladder the class is a rung of (``GRADE_SCALE``), or None for a class with no place on one."""
    return GRADE_SCALE.get(grade_family(c)) if np.isfinite(grade_rank(c)) else None


def handicap_start(text) -> float:
    """GBGB's trapHandicap ("R7": receives 7 metres; "Scr": scratch) in metres; NaN when the race is no handicap."""
    t = str(text or "").strip().lower()
    if t in ("scr", "scratch"):
        return 0.0
    m = re.fullmatch(r"r\s*(\d+(?:\.\d+)?)", t)
    return float(m.group(1)) if m else np.nan


def load_runs(store: Store | None = None, years=None, frames=None) -> pd.DataFrame:
    """Every GBGB run (the years held, or ``years``), or ``frames`` given directly (tests)."""
    if frames is None:
        store = store or Store()
        keys = sorted(k for k in store.listing("gbgb/") if re.search(r"gbgb/runs_\d{4}\.parquet$", k))
        if years:
            keys = [k for k in keys if int(re.search(r"(\d{4})", k).group(1)) in set(years)]
        frames = [pd.read_parquet(io.BytesIO(store.get(k))) for k in keys]
    df = pd.concat(frames, ignore_index=True)
    return clean(df)


#: a trial (T, T1..T4): no market prices it, so it is never a target. GBGB serves trials with no off time, so none
#: passes ``clean`` (they are read on their own channel, greyhound/pit/trials.py). An intertrack race (IT) is a race:
#: 327 of them between 1 Jul 2025 and 5 Nov 2026, 93.9% of their runners with an SP. Until 11 Nov 2026 IT was flagged
#: a trial here, and so was never a target, never priced live and (from c2e76fa) out of the time standards; "IT "
#: with a trailing space (180 rows) was treated the other way (the audit of 11 Nov 2026, findings 6 and 47)
TRIAL_RE = r"^T\d*$"
#: a going allowance beyond this many hundredths of a second is a typing error, not going (the largest real
#: allowances in the same months are about +/-1.2 s; the table holds 2020 and 51430)
GOING_MAX = 150.0
#: GBGB's calculated time is the run time plus the going allowance plus 0.08 s (a length) a metre of handicap start;
#: a calculated time further than this from that sum has an allowance that was typed wrong or never applied
ALLOWANCE_TOL = 0.02
LENGTH_S = 0.08


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # a race card (greyhound/card.py: today's runners, before the off) has no result yet and is kept as it is
    card = df["is_card"].fillna(False).astype(bool) if "is_card" in df else pd.Series(False, index=df.index)
    df = df.assign(is_card=card)
    # (the frame may have been cleaned before - the live card is priced on clean(history + card) - so every step
    # below gives the same rows and values the second time: a non-finisher keeps its row and its flag)
    was_dnf = df["dnf"].fillna(False).astype(bool) if "dnf" in df else pd.Series(False, index=df.index)
    df = df[df.dog_id.notna() & (df.position.notna() | card | was_dnf) & df.race_id.notna() & df.race_date.notna()]
    df["race_class"] = df.race_class.map(lambda c: _class(c) if isinstance(c, str) else c)
    df["track"] = canonical_track(df.track)
    df["is_trial"] = df.race_class.astype(str).str.match(TRIAL_RE)
    df["dog_id"] = df.dog_id.astype("int64")
    df["race_id"] = df.race_id.astype("int64")
    df["raceid"] = df.race_id.astype(str)                                   # the key model.lagsafe groups races on
    df["race_time"] = df.race_time.astype(str).str[:5]
    df["t"] = pd.to_datetime(df.race_date + " " + df.race_time, errors="coerce")
    df = df.dropna(subset=["t"])
    # a race with no winner was not run or was voided (GBGB writes position 0 on every dog, the comment "(NoRace)"):
    # it is no run for any dog. Until 11 Nov 2026 its dogs were kept, each "placed" at position 0 and beaten 0 lengths
    # (the audit of 11 Nov 2026, findings 4 and 50: 207 such races in 1 Jul 2025 to 5 Nov 2026)
    run = (df.position == 1).groupby(df.race_id).transform("any") | df.is_card
    df = df[run]
    # a dog with position 0 in a race that was run started (it has an SP) and has no finishing position: it did not
    # finish or its result was annulled. It lost; its position and distance are unknown, not "better than first"
    dnf = ((df.position == 0) | was_dnf.reindex(df.index)) & ~df.is_card
    df["won"] = (df.position == 1).astype(float)
    df["placed2"] = ((df.position >= 1) & (df.position <= 2)).astype(float)
    df["dnf"] = dnf
    df.loc[dnf, "position"] = np.nan
    df.loc[df.is_card, ["won", "placed2"]] = np.nan
    df["field"] = df.groupby("race_id").dog_id.transform("size")
    # the race's number: its order by the off within its meeting, on the results and on a card alike (card.py numbers
    # a card the same way). GBGB's own number counts the trials and void races between (it equals this order on 98.1%
    # of races), and a card cannot know those
    if "meeting_id" in df:
        df["race_number"] = df.groupby("meeting_id").t.rank(method="dense")
    df["grade"] = df.race_class.map(grade_rank)
    df["grade_family"] = df.race_class.map(grade_family)
    df["grade_scale"] = df.race_class.map(grade_scale)
    # a handicap: GBGB's flag and the class HP are the same races (2,082 of 2,082), and a card has only the class
    flag = df["handicap"].fillna(False).astype(bool) if "handicap" in df else pd.Series(False, index=df.index)
    df["handicap"] = flag | (df.grade_family == "HP")
    start = df["trap_handicap"].map(handicap_start) if "trap_handicap" in df else pd.Series(np.nan, index=df.index)
    df["hcp_start"] = start.where(df.handicap)
    df["hurdles"] = (df["race_type"].astype(str).str.strip().str.lower() == "hurdles") if "race_type" in df else False
    if "beaten" in df:
        # the distances again from GBGB's text (the stored column may be from an older reading of the short margins).
        # No text on a beaten dog is no distance (351 runs in 1 Jul 2025 to 5 Nov 2026, stored as 0), not a dead heat
        txt = df.beaten.fillna("").astype(str).str.strip()
        memo = {b: gbgb_lengths(b) for b in pd.unique(txt)}
        stored = pd.to_numeric(df["beaten_lengths"], errors="coerce") if "beaten_lengths" in df else pd.Series(np.nan, index=df.index)
        blank = pd.Series(np.where(df.position == 1, 0.0, np.nan), index=df.index).where(~(stored.notna() & (stored != 0)), stored)
        df["beaten_lengths"] = txt.map(memo).astype(float).where(txt != "", blank)
    # a calculated time whose allowance does not add up is no time: the standards never forget one
    if {"adjusted_time", "run_time", "going"} <= set(df.columns):
        want = df.run_time + df.going / 100.0 + LENGTH_S * df.hcp_start.fillna(0.0)
        wrong = ((df.adjusted_time - want).abs() > ALLOWANCE_TOL) | (df.going.abs() > GOING_MAX)
        before = df["allowance_fault"].fillna(False).astype(bool) if "allowance_fault" in df else False
        df["allowance_fault"] = (wrong.fillna(False) & df.adjusted_time.notna()) | before
        df.loc[df.allowance_fault, "adjusted_time"] = np.nan
    df["sp_p"] = 1.0 / df.sp_decimal
    df.loc[~np.isfinite(df.sp_p), "sp_p"] = np.nan
    df["sp_p_norm"] = df.sp_p / df.groupby("race_id").sp_p.transform("sum")
    return df.sort_values(["t", "race_id", "trap"], kind="stable").reset_index(drop=True)
