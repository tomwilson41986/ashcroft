"""HorseRaceBase's System Builder categories, converted to greyhounds (the owner's ask, 6 Oct 2026: "evaluate these
features and see if any are useful; convert from horse to greyhound racing").

The categories the engine and the parity block did not already have, each from the dogs' earlier days only (one row a
dog a day, lagged a day; group statistics from earlier days through ``model.lagsafe``), all prefixed ``hb_``:

- **H2H** (HRB "H2H Wins / Runs / Draws"): against today's opponents, the past meetings, how often it finished ahead,
  how many of today's field it has met, and the net (ahead less behind).
- **The field** (HRB "No. of CD winners", "% Track winners", "No. H-won LR", "No. H-placed LR", "No. H-ran 30 days",
  "Pos days since run"): how many of the others have a course-and-distance win, a track win, won or placed last time,
  ran in the last 30 days; this dog's rank for freshness.
- **The dog's record by condition** ("H-Win%/Plc% (Class / Trap / Distance / Going)"): at today's grade, trap, trip,
  going band, with today's trainer; shrunk toward its overall record.
- **Class and trip** ("Highest Class Run / Win", "Max Distance Won", "Class Move"): the best grade raced and won, today
  against them, the longest trip won, the grade of the last win.
- **Since a win** ("Days Since Win", "Runs Since Win", "Days Since Track/Dist Win").
- **Best in ten** ("Best in Ten Runs"): the best speed rating of the last 10 days run.
- **Weight** ("Weight v Max / Min"): today's weight against its heaviest and lightest.
- **Trainer** ("T-Runners / Win% (7 / 14 days)", "T-Win% (Track)", "T-Primary Location %"): the last 7 and 14 days,
  at this track, and the share of the trainer's runners at this track (its home track or a raid).
- **Breeding** ("S-Win% (Distance)", "D-Win% (Dam)"): the sire at today's trip band, the dam's progeny.
- **Owner** ("Owner"): strike rate, shrunk.
- **The race** ("Meeting Time", "Ordered Card Number", "Date (Day / Month)", "Prize Money", "Handicap"): the hour, the
  race's number on the card, the weekday and month, the winner's prize, a handicap.
- **Age** ("Age vs Youngest / Oldest", "Foal Month"): against the field, the birth month.
- **Season** (greyhound only): the days since a bitch's last season, GBGB's record of it.

Not converted: jockey, headgear, tongue tie, official ratings, fences and hurdles, surface, the forecast and the day's
odds (the model is market-blind; the market's history is in the parity block).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_count, race_lagged_expanding_mean, race_lagged_expanding_sum

PREFIX = "hb_"
#: the groups left out of the block's features (computed, not served): dropping each made the model better on 2026's
#: three folds (reports/greyhound_hrb_local_1006.json): class, trip and wins (best grade raced and won, longest trip
#: won, days and runs since a win, best of ten), and breeding and owner (sire by trip, dam, owner)
LEFT_OUT = ("hb_best_grade_run", "hb_best_grade_won", "hb_grade_vs_best_run", "hb_grade_vs_best_won",
            "hb_grade_vs_last_win", "hb_max_dist_won", "hb_trip_vs_max_won", "hb_days_since_win", "hb_runs_since_win",
            "hb_days_since_track_win", "hb_days_since_cd_win", "hb_best_gsr10", "hb_sire_band_win", "hb_dam_win",
            "hb_owner_win")
MONTHS = {"ja": 1, "fe": 2, "mr": 3, "ap": 4, "my": 5, "jn": 6, "jy": 7, "au": 8, "sp": 9, "oc": 10, "nv": 11, "de": 12}


def _shrunk_rate(df, keys, col, prior: pd.Series, k: float = 5.0) -> pd.Series:
    """The group's rate of ``col`` over earlier days, shrunk toward ``prior`` (the dog's overall rate) by k runs."""
    m = race_lagged_expanding_mean(df, keys, col)
    n = race_lagged_expanding_count(df, keys, col)
    return (m.fillna(0) * n + prior * k) / (n + k)


def season_date(s) -> pd.Timestamp | None:
    """GBGB's season field ("15.Sp.25") as a date; "Unknown", "Suppressed" or blank are unknown."""
    m = re.match(r"^\s*(\d{1,2})\.([A-Za-z]{2})\.(\d{2})\s*$", str(s or ""))
    if not m or m.group(2).lower() not in MONTHS:
        return None
    try:
        return pd.Timestamp(2000 + int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    except ValueError:
        return None


def h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Head to head against today's field, from earlier days: for each dog, the past meetings with today's opponents,
    how many it finished ahead in, how many of the field it has met."""
    r = df[["race_id", "race_date", "dog_id", "position"]].reset_index()
    pairs = r.merge(r, on=["race_id", "race_date"], suffixes=("", "_o"))
    pairs = pairs[pairs.dog_id != pairs.dog_id_o]
    met = pairs.position.notna() & pairs.position_o.notna()
    pairs = pairs.assign(_met=met.astype(np.int32), _ahead=(met & (pairs.position < pairs.position_o)).astype(np.int32))
    day = pairs.groupby(["dog_id", "dog_id_o", "race_date"], sort=True)[["_met", "_ahead"]].sum()
    g = day.groupby(level=[0, 1])
    prior = (g.cumsum() - day).rename(columns={"_met": "m", "_ahead": "a"})      # before the day
    pairs = pairs.join(prior, on=["dog_id", "dog_id_o", "race_date"])
    pairs["_seen"] = (pairs.m > 0).astype(np.int32)
    agg = pairs.groupby("index").agg(m=("m", "sum"), a=("a", "sum"), met=("_seen", "sum"))
    out = pd.DataFrame(index=df.index)
    out[f"{PREFIX}h2h_meetings"] = agg.m.reindex(df.index).astype(float)
    out[f"{PREFIX}h2h_ahead"] = agg.a.reindex(df.index).astype(float)
    out[f"{PREFIX}h2h_rate"] = (out[f"{PREFIX}h2h_ahead"] + 1) / (out[f"{PREFIX}h2h_meetings"] + 2)
    out[f"{PREFIX}h2h_net"] = 2 * out[f"{PREFIX}h2h_ahead"] - out[f"{PREFIX}h2h_meetings"]
    out[f"{PREFIX}h2h_opponents_met"] = agg.met.reindex(df.index).astype(float)
    return out


def calculate(df: pd.DataFrame, dog_days) -> tuple[pd.DataFrame, list[str]]:
    """The block for the engine's frame (after the base and parity blocks; ``dog_days`` is metrics.DogDays)."""
    cols = {}
    missing = {c: np.nan for c in ("dam", "owner", "race_number", "prizes", "handicap", "season", "going", "born")
               if c not in df.columns}
    if missing:                                         # a source without these fields: each is unknown
        df = df.assign(**missing)
    g = df.race_id
    won = df.won
    placed = df.placed2
    # ---- the dog's record by condition, shrunk toward its overall record
    overall = race_lagged_expanding_mean(df, "dog_id", "won").fillna(1.0 / df.field)
    overall_p = race_lagged_expanding_mean(df, "dog_id", "placed2").fillna(2.0 / df.field)
    band = pd.cut(df.distance_m, [0, 350, 550, 10_000], labels=False).astype(float)
    going_b = np.sign(df.going.where(df.going.abs() <= 300)).fillna(0.0)
    tmp = df.assign(_band=band, _going=going_b)
    for name, keys in (("grade", ["dog_id", "race_class"]), ("trap", ["dog_id", "trap"]),
                       ("trip", ["dog_id", "distance_m"]), ("band", ["dog_id", "_band"]),
                       ("going", ["dog_id", "_going"]), ("trainer", ["dog_id", "trainer"])):
        cols[f"{PREFIX}win_{name}"] = _shrunk_rate(tmp, keys, "won", overall)
        cols[f"{PREFIX}plc_{name}"] = _shrunk_rate(tmp, keys, "placed2", overall_p)
        cols[f"{PREFIX}runs_{name}"] = race_lagged_expanding_count(tmp, keys, "won").astype(float)
    # ---- class, trip and wins over the dog's earlier days
    d = df.assign(_g_won=df.grade.where(won == 1), _dist_won=df.distance_m.where(won == 1),
                  _trk_win=won, _cd_win=won)
    dd = dog_days(d, ["grade", "_g_won", "_dist_won", "won", "gsr", "weight_kg"], [])
    x = dd.dd
    gd = x.groupby("dog_id", sort=False)
    day = {}
    prev_grade = dd.prev("grade")
    day["best_grade_run"] = prev_grade.groupby(x.dog_id).cummin()
    day["best_grade_won"] = dd.prev("_g_won").groupby(x.dog_id).cummin()
    day["last_win_grade"] = dd.prev("_g_won").groupby(x.dog_id).ffill()
    day["max_dist_won"] = dd.prev("_dist_won").groupby(x.dog_id).cummax()
    win_day = x._d.where(x.won > 0)
    last_win = win_day.groupby(x.dog_id).shift(1).groupby(x.dog_id).ffill()
    day["days_since_win"] = (x._d - last_win).dt.days.astype(float)
    runs_cum = gd._runs.cumsum() - x._runs
    win_runs = runs_cum.where(x.won > 0)
    last_win_runs = win_runs.groupby(x.dog_id).shift(1).groupby(x.dog_id).ffill()
    day["runs_since_win"] = (runs_cum - last_win_runs).astype(float)
    day["best_gsr10"] = dd.window("gsr", 10, "max")
    day["weight_max"] = dd.prev("weight_kg").groupby(x.dog_id).cummax()
    day["weight_min"] = dd.prev("weight_kg").groupby(x.dog_id).cummin()
    back = {k: dd.back(v) for k, v in day.items()}
    cols[f"{PREFIX}best_grade_run"] = back["best_grade_run"]
    cols[f"{PREFIX}best_grade_won"] = back["best_grade_won"]
    cols[f"{PREFIX}grade_vs_best_run"] = df.grade.values - back["best_grade_run"]
    cols[f"{PREFIX}grade_vs_best_won"] = df.grade.values - back["best_grade_won"]
    cols[f"{PREFIX}grade_vs_last_win"] = df.grade.values - back["last_win_grade"]
    cols[f"{PREFIX}max_dist_won"] = back["max_dist_won"]
    cols[f"{PREFIX}trip_vs_max_won"] = df.distance_m.values - back["max_dist_won"]
    cols[f"{PREFIX}days_since_win"] = back["days_since_win"]
    cols[f"{PREFIX}runs_since_win"] = back["runs_since_win"]
    cols[f"{PREFIX}best_gsr10"] = back["best_gsr10"]
    cols[f"{PREFIX}weight_vs_max"] = df.weight_kg.values - back["weight_max"]
    cols[f"{PREFIX}weight_vs_min"] = df.weight_kg.values - back["weight_min"]
    # days since a win here and at this course and distance (from the group's last winning day, earlier days only)
    for name, keys in (("track", ["dog_id", "track"]), ("cd", ["dog_id", "track", "distance_m"])):
        w = df.assign(_wd=df.t.dt.normalize().where(won == 1))
        k = w.groupby(keys + ["race_date"], sort=False)._wd.max().reset_index()
        k = k.sort_values(keys + ["race_date"], kind="stable").reset_index(drop=True)
        k["_last"] = k.groupby(keys, sort=False)._wd.shift(1)
        k["_last"] = k.groupby(keys, sort=False)._last.ffill()
        m = w[keys + ["race_date"]].merge(k[keys + ["race_date", "_last"]], on=keys + ["race_date"], how="left")
        cols[f"{PREFIX}days_since_{name}_win"] = (df.t.dt.normalize().values - pd.to_datetime(m._last).values) \
            / np.timedelta64(1, "D")
    # ---- trainer: the last 7 and 14 days, this track, the share of its runners here
    t = df[["trainer", "race_date", "won", "sp_p_norm"]].assign(_d=pd.to_datetime(df.race_date))
    tday = t.groupby(["trainer", "_d"]).agg(w=("won", "sum"), x=("sp_p_norm", "sum"), n=("won", "size")).reset_index()
    tday = tday.sort_values(["trainer", "_d"])
    idx = pd.MultiIndex.from_arrays([t.trainer, t._d])
    for days in (7, 14):
        roll = tday.set_index("_d").groupby("trainer")[["w", "n"]].rolling(f"{days}D", closed="left").sum().reset_index()
        key = pd.MultiIndex.from_frame(roll[["trainer", "_d"]])
        n = pd.Series(roll.n.values, index=key).reindex(idx).values
        w = pd.Series(roll.w.values, index=key).reindex(idx).values
        cols[f"{PREFIX}trainer{days}_runs"] = n
        cols[f"{PREFIX}trainer{days}_win"] = (w + 1.0) / (n + 6.0)
    trainer_all = race_lagged_expanding_mean(df, "trainer", "won").fillna(1.0 / df.field)
    cols[f"{PREFIX}trainer_track_win"] = _shrunk_rate(df, ["trainer", "track"], "won", trainer_all, k=30.0)
    here = race_lagged_expanding_count(df, ["trainer", "track"], "won")
    allr = race_lagged_expanding_count(df, "trainer", "won")
    cols[f"{PREFIX}trainer_track_share"] = (here / allr.replace(0, np.nan))
    # ---- breeding and owner
    pop = race_lagged_expanding_mean(df.assign(_p=0), "_p", "won").fillna(1 / 6)
    cols[f"{PREFIX}sire_band_win"] = _shrunk_rate(tmp, ["sire", "_band"], "won", pop, k=50.0)
    cols[f"{PREFIX}dam_win"] = _shrunk_rate(df, "dam", "won", pop, k=30.0)
    cols[f"{PREFIX}owner_win"] = _shrunk_rate(df, "owner", "won", pop, k=30.0)
    # ---- the race
    cols[f"{PREFIX}hour"] = pd.to_numeric(df.race_time.astype(str).str[:2], errors="coerce").values
    cols[f"{PREFIX}race_number"] = pd.to_numeric(df.race_number, errors="coerce").values
    cols[f"{PREFIX}weekday"] = df.t.dt.weekday.values.astype(float)
    cols[f"{PREFIX}month"] = df.t.dt.month.values.astype(float)
    cols[f"{PREFIX}prize_1st"] = pd.to_numeric(df.prizes.astype(str).str.extract(r"1st\s*£\s*([\d,]+)")[0]
                                               .str.replace(",", ""), errors="coerce").values
    cols[f"{PREFIX}handicap"] = pd.to_numeric(df.handicap, errors="coerce").astype(float).values
    # ---- age and season
    age = df.age_months
    cols[f"{PREFIX}age_vs_youngest"] = (age - age.groupby(g).transform("min")).values
    cols[f"{PREFIX}age_vs_oldest"] = (age - age.groupby(g).transform("max")).values
    born = pd.to_datetime(df.born, format="%b-%Y", errors="coerce")
    cols[f"{PREFIX}birth_month"] = born.dt.month.values.astype(float)
    uniq = {s: season_date(s) for s in pd.unique(df.season.astype(str))}
    sd = pd.to_datetime(df.season.astype(str).map(uniq))
    since = (df.t.dt.normalize() - sd).dt.days
    cols[f"{PREFIX}days_since_season"] = since.where(since >= 0).values.astype(float)
    out = pd.DataFrame(cols, index=df.index)
    # ---- the field, from the others' pre-race records
    has_cd = (df.cd_win.fillna(0) * df.cd_runs.fillna(0) > 0).astype(float)
    has_trk = (df.trk_win.fillna(0) * df.trk_runs.fillna(0) > 0).astype(float)
    won_lr = (df.won_mean1 == 1).astype(float)
    placed_lr = (df.placed2_mean1 == 1).astype(float)
    ran30 = (df.days_since <= 30).astype(float)
    for name, s in (("cd_winners", has_cd), ("track_winners", has_trk), ("won_lr", won_lr),
                    ("placed_lr", placed_lr), ("ran_30d", ran30)):
        out[f"{PREFIX}field_{name}"] = s.groupby(g).transform("sum") - s
    out[f"{PREFIX}days_since_rank"] = df.days_since.groupby(g).rank(method="average")
    out[f"{PREFIX}win_grade_rank"] = out[f"{PREFIX}win_grade"].groupby(g).rank(ascending=False, method="average")
    # ---- head to head
    out = out.join(h2h(df))
    out[f"{PREFIX}h2h_rate_rank"] = out[f"{PREFIX}h2h_rate"].groupby(g).rank(ascending=False, method="average")
    out = out.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return out, [c for c in out.columns if c not in LEFT_OUT]
