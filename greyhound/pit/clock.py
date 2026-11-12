"""Body clock, the wait in the racing kennels, and the calendar of noise (the owner, 11 Nov 2026: "identify NEW
features that have not been discovered yet").

Dogs are kennelled at the track before the first race of a meeting and wait there until their own; they are creatures
of routine, fed, exercised and raced at fixed hours, with an entrained hour of their own; and they are disturbed by
fireworks. None of this is a rating: it is the state of the animal at the off. The off time and the race's number
are in the engine as race-level fields, the same for all six runners; this table sets them against each dog's own
habit and its own record of running after a wait.

Everything of today is read from the card (the off time, the track, the date, the race's number, the other races'
off times at the track) and everything of the dog from its EARLIER days.

- **The meeting and the wait.** ``minutes`` = the off time as minutes after midnight (UK clock time, as GBGB and the
  card give it). A meeting is the track's races of the day, a new one beginning where more than ``GAP`` minutes pass
  between consecutive offs (see the note below on why not ``meeting_id``). ``ck_wait`` = ``minutes`` less the
  meeting's first off: the wait in the racing kennels. ``ck_slot`` = the race's number over the meeting's largest (a
  later meeting of the track's day numbered from its own first race; where there is no race number, the race's rank
  by off time over the number of offs).
- **The habitual hour.** The mean of the dog's ``minutes`` on earlier days, each run weighted by 0.5 to the power of
  its age over ``HALF_LIFE`` days (``figs.decayed``). ``ck_habit_gap`` = (today's minutes - habit) / 60, hours, above
  0 later than its habit; ``ck_habit_abs`` its size; ``ck_habit_n`` the decayed weight of runs behind the habit.
- **Its own sensitivity to the wait.** With ``w`` the wait and ``r`` the residual (``dyad.residuals``: lengths
  slower than the expected time before the race) of its runs on earlier days that have both, the shrunk slope
  ``b = sum((w - mean w) r) / (sum((w - mean w)^2) + K_WAIT)``, ``K_WAIT`` = 3600 x 5 (five runs an hour apart in
  wait). ``ck_wait_slope`` = 60 b, lengths slower an hour of wait; ``ck_wait_pred`` = b x (today's wait - its mean
  earlier wait), the lengths that slope says today's wait costs. Both are missing until the dog has one such run.
- **The session.** 0 before 12:30, 1 from 12:30 to 17:59, 2 from 18:00 (``ck_session``). ``ck_session_new`` = 1 when
  the dog has raced before and none of its last ``LAST_DAYS`` earlier race days was in today's session.
- **The clock change.** ``ck_dst`` = 1 on the day of a UK clock change (the last Sunday of March and of October) and
  the six days after, -1 on the seven days before, else 0: the card moves an hour and the dog's clock does not.
- **Darkness.** Sunset at the track in UK clock time, from the day of the year and the track's latitude and longitude
  (research/greyhound/reference/gb_tracks.csv): the sun's declination -23.44 cos(2 pi (N + 10) / 365) degrees, the
  hour angle at which its centre is 0.833 degrees below the horizon, the equation of time
  9.87 sin 2B - 7.53 cos B - 1.5 sin B minutes with B = 2 pi (N - 81) / 364, four minutes a degree of longitude, an
  hour added in summer time (``sunset_minutes``). ``ck_dark`` = 1 when the off is after it (missing for a track not
  in the file); ``ck_dark_share`` = the share of the dog's earlier race days that were in the dark;
  ``ck_dark_new`` = ck_dark x (1 - ck_dark_share).
- **Fireworks.** ``ck_fireworks`` = 1 for an evening race (session 2) on 2 to 8 November or 30 December to
  1 January; ``ck_days_since_fw`` = days since the last 5 November or 31 December (0 on the night itself).

**The meeting is read from the off times, not from ``meeting_id``.** A card's ``meeting_id`` and race number
(greyhound/card.py) run over the track's whole day, and a track often runs two meetings a day; the meeting's own id
would give a card row of the second meeting a wait counted from the morning's first race, not what its run gets in
the nightly build. The off times are on the card and in the results alike. In the GBGB runs of 1 Jul 2025 to
5 Nov 2026 no two races of one meeting are more than 75 minutes apart and no two meetings of a track's day less
than 190 (the build's notes count the rows whose meeting differs from ``meeting_id``'s).

**What a result row cannot know.** ``ck_wait`` reads the meeting's first off and ``ck_slot`` its largest race number,
and GBGB's results hold only the races that were run. A card lists every race; a result row of a meeting that was
abandoned part-way has a slot counted against the last race RUN (race 3 of a meeting stopped after three reads 1.0
where its card read 3/12), and a meeting whose first race was void a wait counted from its second. No result of the
day is read (tests/test_greyhound_clock.py rewrites a middle day's results and removes a day's later races: only
``ck_slot`` moves), but whether the later races were run is a fact of the day, and an abandoned meeting is often a
slow one. In the runs of 1 Jul 2025 to 5 Nov 2026: 5 of 5,508 meetings have no race numbered 1, 21 a number missing
inside, and 34 a largest number 4 or more under the median of the track's last four meetings on that weekday in that
session (their 1,041 runs a mean 0.41 lengths slower than expected, the rest 0.04) - most of those 34 are short
cards, not abandonments; the results cannot tell the two apart, so the slot is left as defined and the build's notes
count the meetings not numbered from 1 and with a number missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from greyhound.pit import core, dyad, figs as figs_mod

PREFIX = "ck_"
SOURCE = ("GBGB results and the card (off time, track, date, race number: sources/gbgb.py, greyhound/card.py); the "
          "expected-time table (xt_tot_time) for the earlier runs' residuals; track coordinates "
          "(research/greyhound/reference/gb_tracks.csv); the UK clock-change rule and a solar formula")
GAP = 120.0                   # minutes between consecutive offs at a track that begin a new meeting
HALF_LIFE = 60.0              # days: the habitual hour
K_WAIT = 3600.0 * 5           # minutes squared: the slope's prior, five runs an hour apart in wait
LAST_DAYS = 6                 # earlier race days read for a new session
NOON, EVENING = 12 * 60 + 30, 18 * 60
SUN_ALTITUDE = -0.833         # degrees: the sun's centre at sunset (refraction and its half-width)
TRACKS = Path(__file__).resolve().parents[2] / "research" / "greyhound" / "reference" / "gb_tracks.csv"


def minutes_of(race_time: pd.Series) -> np.ndarray:
    """HH:MM (or HH:MM:SS) as minutes after midnight; NaN where it does not parse."""
    codes, uniq = pd.factorize(race_time.astype(str))                 # a day has 1,440 minutes: few distinct values
    p = pd.Series(uniq, dtype=str).str.extract(r"^\s*(\d{1,2}):(\d{2})")
    h, m = pd.to_numeric(p[0], errors="coerce"), pd.to_numeric(p[1], errors="coerce")
    v = (h * 60.0 + m).where((h < 24) & (m < 60)).to_numpy(float)
    return v[codes] if len(v) else np.full(len(codes), np.nan)


def clock_changes(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The UK clock changes of a year: the last Sunday of March (summer time begins) and of October (it ends)."""
    out = []
    for month in (3, 10):
        d = pd.Timestamp(year=year, month=month, day=31)
        out.append(d - pd.Timedelta(days=(d.dayofweek + 1) % 7))
    return out[0], out[1]


def calendar(days: pd.DatetimeIndex) -> pd.DataFrame:
    """For each day: the clock-change flag, whether summer time is on, the fireworks dates and the days since the
    last 5 November or 31 December."""
    days = pd.DatetimeIndex(days)
    dst = np.zeros(len(days))
    bst = np.zeros(len(days), dtype=bool)
    since = np.full(len(days), np.nan)
    yr = days.year.to_numpy()
    for y in np.unique(yr[np.isfinite(yr)]).astype(int):              # a handful of years
        m = yr == y
        d = days[m]
        spring, autumn = clock_changes(y)
        bst[m] = (d >= spring) & (d < autumn)
        f = np.zeros(m.sum())
        for ch in (spring, autumn):
            off = (d - ch).days.to_numpy()
            f = np.where((off >= 0) & (off <= 6), 1.0, np.where((off >= -7) & (off <= -1), -1.0, f))
        dst[m] = f
        dec, nov, eve = pd.Timestamp(year=y - 1, month=12, day=31), pd.Timestamp(year=y, month=11, day=5), \
            pd.Timestamp(year=y, month=12, day=31)
        s = (d - dec).days.to_numpy().astype(float)
        for last in (nov, eve):
            off = (d - last).days.to_numpy().astype(float)
            s = np.where(off >= 0, np.minimum(s, off), s)
        since[m] = s
    mo, dd = days.month.to_numpy(), days.day.to_numpy()
    fw = ((mo == 11) & (dd >= 2) & (dd <= 8)) | ((mo == 12) & (dd >= 30)) | ((mo == 1) & (dd == 1))
    return pd.DataFrame({"dst": dst, "bst": bst, "fw_date": fw, "since_fw": since,
                         "doy": days.dayofyear.to_numpy().astype(float)}, index=days)


def sunset_minutes(doy, lat, lon, bst) -> np.ndarray:
    """Sunset in UK clock time, minutes after midnight, on day ``doy`` of the year at ``lat``, ``lon`` (degrees, east
    positive); ``bst`` true in summer time. NaN where the sun does not set or a coordinate is missing."""
    doy, lat, lon = (np.asarray(x, dtype=float) for x in (doy, lat, lon))
    decl = np.radians(-23.44 * np.cos(2 * np.pi * (doy + 10.0) / 365.0))
    b = 2 * np.pi * (doy - 81.0) / 364.0
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)                # minutes, sundial ahead of clock
    phi = np.radians(lat)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosw = (np.sin(np.radians(SUN_ALTITUDE)) - np.sin(phi) * np.sin(decl)) / (np.cos(phi) * np.cos(decl))
        w = np.degrees(np.arccos(np.where(np.abs(cosw) <= 1, cosw, np.nan)))       # the hour angle at sunset
    return 720.0 + 4.0 * w - 4.0 * lon - eot + 60.0 * np.asarray(bst, dtype=float)


def track_coordinates(path: Path | str | None = None) -> pd.DataFrame:
    """Latitude and longitude by GBGB track name (empty when the reference file is not there)."""
    try:
        t = pd.read_csv(path or TRACKS, usecols=["gbgb_name", "latitude", "longitude"])
    except (OSError, ValueError):
        return pd.DataFrame(columns=["latitude", "longitude"])
    t["gbgb_name"] = t.gbgb_name.astype(str).str.strip().str.lower()
    t = t.dropna(subset=["latitude", "longitude"]).drop_duplicates("gbgb_name")
    return t.set_index("gbgb_name")[["latitude", "longitude"]]


def meetings(track: np.ndarray, dn: np.ndarray, minutes: np.ndarray, race_number: np.ndarray) -> pd.DataFrame:
    """Each row's meeting read from the off times (the track's races of the day, split where more than ``GAP``
    minutes pass between consecutive offs): its code, its first off, the race's slot in it."""
    n = len(track)
    order = np.lexsort((minutes, dn, track))                          # NaN minutes last within the track's day
    tk, d, mi, rn = track[order], dn[order], minutes[order], race_number[order]
    new_day = np.r_[True, (tk[1:] != tk[:-1]) | (d[1:] != d[:-1])] if n else np.zeros(0, bool)
    with np.errstate(invalid="ignore"):
        step = np.r_[0.0, mi[1:] - mi[:-1]] if n else np.zeros(0)
        new = new_day | (step > GAP)
        new_off = new | (step > 0)
    seg = np.cumsum(new) - 1
    g = pd.DataFrame({"seg": seg, "mi": mi, "rn": rn}).groupby("seg", sort=True)
    first = g.mi.transform("min").to_numpy()
    rn_lo, rn_hi = g.rn.transform("min").to_numpy(), g.rn.transform("max").to_numpy()
    seg_start = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    day_start = np.maximum.accumulate(np.where(new_day, np.arange(n), 0))
    later = seg[seg_start] > seg[day_start]                           # not the first meeting of the track's day
    before = np.where(later, rn_lo - 1.0, 0.0)                        # the numbers the earlier meetings used
    with np.errstate(invalid="ignore", divide="ignore"):
        by_number = np.where((rn - before >= 1) & (rn_hi - before >= 1), (rn - before) / (rn_hi - before), np.nan)
        rank = np.cumsum(new_off)
        rank = rank - rank[seg_start] + 1.0                           # the race's place by off time in its meeting
        offs = pd.Series(rank).groupby(seg).transform("max").to_numpy()
        by_rank = np.where(np.isfinite(mi), rank / offs, np.nan)
    out = np.empty((n, 4))
    out[order] = np.column_stack([seg, first, np.where(np.isfinite(by_number), by_number, by_rank),
                                  np.isfinite(by_number)])
    return pd.DataFrame(out, columns=["meeting", "first", "slot", "numbered"])


def earlier_days(dog: np.ndarray, dn: np.ndarray, vals: np.ndarray, last: int | None = None) -> tuple:
    """Per dog, over its EARLIER days (the last ``last`` of them; None: all): each column's sum of the day's means,
    the days that had a value, and the earlier days in the window. One row an input row; the day's own rows, before
    or after, are left out."""
    code = pd.DataFrame({"dog": dog, "dn": dn}).groupby(["dog", "dn"], sort=True).ngroup().to_numpy()
    ng = int(code.max()) + 1 if len(code) else 0
    fin = np.isfinite(vals)
    k = vals.shape[1]
    s = np.column_stack([np.bincount(code, weights=np.where(fin[:, j], vals[:, j], 0.0), minlength=ng) for j in range(k)])
    c = np.column_stack([np.bincount(code, weights=fin[:, j].astype(float), minlength=ng) for j in range(k)])
    has = (c > 0).astype(float)
    mean = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
    gd = np.zeros(ng, dtype=np.int64)
    gd[code] = dog
    idx = np.arange(ng)
    start = np.maximum.accumulate(np.where(np.r_[True, gd[1:] != gd[:-1]], idx, 0)) if ng else idx
    lo = start if last is None else np.maximum(idx - last, start)
    before, nbefore = np.cumsum(mean, axis=0) - mean, np.cumsum(has, axis=0) - has
    return (before - before[lo])[code], (nbefore - nbefore[lo])[code], (idx - lo)[code].astype(float)


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    n = len(df)
    day = core.day(df)
    dn = (day - pd.Timestamp("2000-01-01")).dt.days.fillna(-1).to_numpy(np.int64)
    dog = df.dog_id.to_numpy(np.int64)
    minutes = minutes_of(df.race_time)
    tcode, tname = pd.factorize(df.track.astype(object).where(df.track.notna(), "?"))
    names, tname = pd.factorize(pd.Series(tname, dtype=object).astype(str).str.strip())
    tcode, tname = names[tcode], pd.Series(tname, dtype=object)
    rn = pd.to_numeric(df.race_number, errors="coerce").to_numpy(float) if "race_number" in df else np.full(n, np.nan)
    mt = meetings(tcode, dn, minutes, rn)
    wait = minutes - mt["first"].to_numpy()
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": df.dog_id.to_numpy()}, index=df.index)
    out[f"{PREFIX}wait"] = wait
    out[f"{PREFIX}slot"] = mt.slot.to_numpy()
    # the habitual hour: the decayed mean off time of its earlier days
    dec = figs_mod.decayed(df, pd.Series(minutes, index=df.index), pd.Series(1.0, index=df.index), HALF_LIFE)
    sw = dec.sw.fillna(0.0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        habit = np.where(sw > 0, dec.swx.to_numpy() / sw, np.nan)
    gap = (minutes - habit) / 60.0
    out[f"{PREFIX}habit_gap"] = gap
    out[f"{PREFIX}habit_abs"] = np.abs(gap)
    out[f"{PREFIX}habit_n"] = sw
    # its own sensitivity to the wait: the shrunk slope of the residual on the wait over its earlier days
    xt = (ctx or {}).get("xtime")
    if xt is None:
        from greyhound.pit import xtime
        xt, _ = xtime.build(runs, df)
    j = df[["race_id", "dog_id"]].merge(xt[["race_id", "dog_id", "xt_tot_time"]].drop_duplicates(["race_id", "dog_id"]),
                                        on=["race_id", "dog_id"], how="left")
    r = dyad.residuals(df, pd.Series(j.xt_tot_time.to_numpy(), index=df.index)).to_numpy(float)
    both = np.isfinite(wait) & np.isfinite(r)
    w0, r0 = np.where(both, wait, 0.0), np.where(both, r, 0.0)
    tmp = pd.DataFrame({"dog_id": dog, "race_date": df.race_date.to_numpy(), "_w": w0, "_r": r0, "_wr": w0 * r0,
                        "_ww": w0 * w0, "_k": both.astype(float)}, index=df.index)
    pw = core.prior_window(tmp, ["dog_id"], ["_w", "_r", "_wr", "_ww", "_k"], None)
    k = pw._k.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        w_mean = np.where(k > 0, pw._w.to_numpy() / k, np.nan)
        sxy = pw._wr.to_numpy() - w_mean * pw._r.to_numpy()
        sxx = np.clip(pw._ww.to_numpy() - w_mean * pw._w.to_numpy(), 0.0, None)
        b = np.where(k > 0, sxy / (sxx + K_WAIT), np.nan)
    out[f"{PREFIX}wait_slope"] = b * 60.0
    out[f"{PREFIX}wait_pred"] = b * (wait - w_mean)
    # the session, and whether any of its last few race days was in it
    sess = np.where(minutes < NOON, 0.0, np.where(minutes < EVENING, 1.0, 2.0))
    sess = np.where(np.isfinite(minutes), sess, np.nan)
    flags = np.column_stack([(sess == s).astype(float) for s in (0.0, 1.0, 2.0)])
    seen, _, ndays = earlier_days(dog, dn, flags, LAST_DAYS)
    seen_today = np.where(np.isfinite(sess), seen[np.arange(n), np.nan_to_num(sess).astype(int)], np.nan)
    out[f"{PREFIX}session"] = sess
    out[f"{PREFIX}session_new"] = ((ndays > 0) & (seen_today == 0)).astype(float)
    # the calendar: the clock change, sunset, the fireworks
    udays = pd.DatetimeIndex(pd.unique(day.dropna()))
    cal = calendar(udays).reindex(pd.DatetimeIndex(day))
    out[f"{PREFIX}dst"] = cal.dst.to_numpy()
    xy = track_coordinates((ctx or {}).get("tracks_file")).reindex(tname.str.lower().to_numpy())
    lat, lon = xy.latitude.to_numpy(float)[tcode], xy.longitude.to_numpy(float)[tcode]
    sunset = sunset_minutes(cal.doy.to_numpy(), lat, lon, cal.bst.fillna(False).to_numpy(bool))
    dark = np.where(np.isfinite(sunset) & np.isfinite(minutes), (minutes > sunset).astype(float), np.nan)
    d_sum, d_n, _ = earlier_days(dog, dn, dark[:, None], None)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(d_n[:, 0] > 0, d_sum[:, 0] / d_n[:, 0], np.nan)
    out[f"{PREFIX}dark"] = dark
    out[f"{PREFIX}dark_share"] = share
    out[f"{PREFIX}dark_new"] = np.where(dark == 0, 0.0, dark * (1.0 - share))
    fw = cal.fw_date.fillna(False).to_numpy(bool) & (sess == 2.0)
    out[f"{PREFIX}fireworks"] = fw.astype(float)
    out[f"{PREFIX}days_since_fw"] = cal.since_fw.to_numpy()
    cols = [c for c in out.columns if c.startswith(PREFIX)]
    out[cols] = out[cols].astype(np.float32)
    # the notes: real counts of this build
    races = pd.DataFrame({"race": df.race_id.to_numpy(), "meeting": mt.meeting.to_numpy(), "wait": wait, "sess": sess,
                          "dark": dark, "fw": fw, "dst": cal.dst.to_numpy()}).drop_duplicates("race")
    notes = {"rows": int(n), "races": int(len(races)), "meetings_by_off_time": int(mt.meeting.nunique()),
             "gap_minutes": GAP, "half_life_days": HALF_LIFE, "k_wait": K_WAIT,
             "rows_off_time_unread": int((~np.isfinite(minutes)).sum()),
             "rows_slot_from_race_number": int(mt.numbered.sum()),
             "wait_minutes_quartiles_of_races": [round(float(q), 1) for q in races.wait.quantile([0.25, 0.5, 0.75])]
             if races.wait.notna().any() else None,
             "wait_minutes_max": float(np.nanmax(wait)) if np.isfinite(wait).any() else None,
             "races_by_session": {str(int(s)): int(c) for s, c in races.sess.value_counts().sort_index().items()},
             "rows_with_a_habit": int((sw > 0).sum()),
             "rows_two_hours_or_more_off_habit": int((np.abs(gap) >= 2).sum()),
             "rows_session_new": int(out[f"{PREFIX}session_new"].sum()),
             "runs_with_wait_and_residual": int(both.sum()),
             "rows_with_a_wait_slope": int(np.isfinite(b).sum()),
             "wait_slope_lengths_an_hour_quartiles": [round(float(q), 4) for q in np.nanquantile(b * 60.0, [0.25, 0.5, 0.75])]
             if np.isfinite(b).any() else None,
             "races_clock_change_week_after": int((races.dst == 1).sum()),
             "races_clock_change_week_before": int((races.dst == -1).sum()),
             "tracks_without_coordinates": sorted(tname[np.isnan(xy.latitude.to_numpy(float))]),
             "races_in_the_dark": int((races.dark == 1).sum()), "races_dark_unknown": int(races.dark.isna().sum()),
             "races_on_fireworks_evenings": int(races.fw.sum())}
    # the races the results do not hold (void, abandoned): a meeting not numbered from its first race, a number missing
    num = pd.DataFrame({"m": mt.meeting.to_numpy(), "rn": rn, "d": dn, "t": tcode}).dropna(subset=["rn"])
    if len(num):
        g = num.groupby("m")
        lo, hi, cnt = g.rn.min(), g.rn.max(), g.rn.nunique()
        first_of_day = g.m.first() == num.groupby(["t", "d"]).m.transform("min").groupby(num.m).first()
        notes["meetings_first_of_the_day_not_numbered_from_1"] = int((first_of_day & (lo != 1)).sum())
        notes["meetings_with_a_race_number_missing"] = int((hi - lo + 1 != cnt).sum())
    if "meeting_id" in df:
        # the check on the meeting read from the off times: rows whose first off differs from meeting_id's
        own = pd.Series(minutes).groupby([df.meeting_id.to_numpy(), dn]).transform("min").to_numpy()
        known = df.meeting_id.notna().to_numpy() & np.isfinite(own) & np.isfinite(mt["first"].to_numpy())
        notes["rows_first_off_differs_from_meeting_id"] = int((known & (own != mt["first"].to_numpy())).sum())
        notes["meetings_by_meeting_id"] = int(pd.DataFrame({"m": df.meeting_id.to_numpy(), "d": dn}).drop_duplicates().shape[0])
    return out.sort_index().reset_index(drop=True), notes


COLUMNS = {
    f"{PREFIX}wait": "minutes from the meeting's first off to this race's off: the wait in the racing kennels (the "
                     f"meeting: the track's races of the day, split where over {GAP:g} minutes pass between offs)",
    f"{PREFIX}slot": "the race's number over the meeting's largest (its rank by off time where there is no number)",
    f"{PREFIX}habit_gap": f"hours today's off is later than the dog's habitual hour (the mean off time of its earlier "
                          f"days, decayed by {HALF_LIFE:g} days); below 0 earlier",
    f"{PREFIX}habit_abs": "the size of that gap in hours",
    f"{PREFIX}habit_n": "the decayed weight of earlier runs behind the habitual hour (0: none)",
    f"{PREFIX}wait_slope": "the dog's own lengths slower than its expected time for an hour more of wait: the slope of "
                           f"its earlier runs' residuals on their waits, shrunk by {K_WAIT:g} minutes squared",
    f"{PREFIX}wait_pred": "that slope times (today's wait less its mean earlier wait): the lengths today's wait costs",
    f"{PREFIX}session": "0 an off before 12:30, 1 from 12:30 to 17:59, 2 from 18:00",
    f"{PREFIX}session_new": f"1 when the dog has raced before and none of its last {LAST_DAYS} earlier race days was "
                            "in today's session",
    f"{PREFIX}dst": "1 on the day of a UK clock change and the six days after, -1 on the seven days before, else 0",
    f"{PREFIX}dark": "1 when the off is after sunset at the track (computed from the date and the track's latitude "
                     "and longitude; missing for a track without coordinates)",
    f"{PREFIX}dark_share": "the share of the dog's earlier race days that were in the dark",
    f"{PREFIX}dark_new": "ck_dark x (1 - ck_dark_share): a dark race for a dog not used to them (0 in daylight)",
    f"{PREFIX}fireworks": "1 for an off from 18:00 on 2 to 8 November or 30 December to 1 January",
    f"{PREFIX}days_since_fw": "days since the last 5 November or 31 December (0 on the night)",
}
