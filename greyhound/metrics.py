"""The greyhound metrics engine: the horse engine's ideas (model/custom_metrics.py) for the dogs, lag-safe throughout.

Every feature of a run is built from runs BEFORE it: a dog's own earlier runs by row (a dog runs once a day), and
every group statistic (track standards, grade pars, trap bias, trainer and sire records) from earlier DAYS only, with
``model.lagsafe`` (a group can have several runners in one race and several races in one day).

The metrics, each a per-run figure first, then windowed over the dog's earlier runs:

| Metric | What it is |
|---|---|
| GSR, the speed rating | the run's calculated time (GBGB's going-corrected time) against the track and distance's standard, in lengths (0.08 s a length), plus our own meeting variant (what the going allowance missed) |
| ESR, early speed | the first sectional against the track and distance's standard sectional, in lengths |
| FSR, finishing speed | the run after the sectional (calculated time less sectional) against its standard |
| LBW, lengths beaten | lengths behind the winner (the GBGB distances, summed down the finishing order) |
| BRK, the break | from the comment: very quick away +2 .. missed the break -2 |
| LED | led at or before the first bend |
| TRB, trouble | crowding, bumping, baulking, checking, stumbling at the start: a count |
| PATH | the running line: rails 1, middle 2, wide 3 |
| FIN | ran on, finished well +1; faded -1 |
| RvP, rating against par | the dog's recent GSR against the par (winners' GSR) of today's track and grade |
| TBIAS | today's trap at today's track and distance: the earlier win rate against the field's share |
| STYLE x TRAP | a railer drawn wide, a wide runner drawn on the inside |
| TRN | the trainer's strike rate and A/E against the SP (shrunk to the population) |
| SIRE | the sire's strike rate (shrunk) |
| MKT | the dog's past SP (the bookmakers' view of it) and its A/E against that SP |
| FRESH | days since the last run, runs in the last 28 days, career runs, age |
| CLASS | grade change, rating against the grade's par, distance change |

and each against today's field: rank, the gap to the field's best, and a pace map (the early-speed order across the
traps: who leads, who is crowded).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_count, race_lagged_expanding_mean, race_lagged_expanding_sum

LENGTH_S = 0.08                     # GBGB's time a length

# ----- the comment codes (GBGB's race comments, e.g. "QAw,Rls,Led1,ALd"), as regexes on each comma-separated token
_BREAK = [(r"^v(ery)?q(uick)?aw|^vqaw", 2.0), (r"^q(uick)?aw|^quickaway|^qaway|^quickaw", 1.0),
          (r"^e(arly)?p(ace)?$|^epace|^earlyp|^ep$", 1.0), (r"^modaway|^moderatelyaway", 0.0),
          (r"^lck|^lacked", -1.0), (r"^s(low)?aw|^slowaway|^saway|^slowaw", -1.0),
          (r"^vs(low)?aw|^vsaw|^veryslow", -2.0), (r"^m(i)?s(se)?d ?br|^missedbr|^msdbrk", -2.0)]
_LED = r"^(led|ld|a ?led|ald|alwaysled|snld|snled|soonled)( ?1|to1|runup|rnup|$)|^ledto[1-9]|^led1|^ld1|^ald$|^aled$"
_TROUBLE = r"^(b?crd|crowded|b?bmp|bumped|blk|baulked|fcd|forced|ck|checked|stb|stumbled|fell|fll|imp|hmp|ran ?wide)"
_SLIGHT = r"^slt"
_RAILS = r"^(rls|rails|railed)$"
_MID = r"^(mid|middle)$"
_WIDE = r"^(w|wide|vw|verywide)$"
_TO = r"(rls|rails|mid|middle|w|wide)[- ]?t(o)?[- ]?(rls|rails|mid|middle|w|wide)$|^(rls|rails|mid|middle)[- ]?(mid|w|wide|rls)$"
_FIN_UP = r"^(ranon|rnon|finwell|ranonlate|ranonwell|chl|chal|ldrnin|ledrunin|ledrnin|ldnrln|lednrln|heldon)"
_FIN_DOWN = r"^(fdd|faded|tired|wkn|weakened|outpaced|stayedon ?slowly)"
_PATH = {"rls": 1.0, "rails": 1.0, "mid": 2.0, "middle": 2.0, "w": 3.0, "wide": 3.0}


def _tokens(comment) -> list[str]:
    return [t.strip().lower() for t in str(comment or "").split(",") if t.strip()]


def parse_comment(comment) -> dict:
    """One run's comment -> break, led, trouble, path, finish."""
    brk, led, trouble, fin, paths = [], 0.0, 0.0, 0.0, []
    for t in _tokens(comment):
        t2 = t.replace(" ", "")
        for pat, v in _BREAK:
            if re.search(pat, t2):
                brk.append(v)
                break
        if re.search(_LED, t2):
            led = 1.0
        if re.search(_TROUBLE, t2):
            trouble += 0.5 if re.search(_SLIGHT, t2) else 1.0
        elif re.search(_SLIGHT, t2) and re.search(r"crd|bmp|blk|ck", t2):
            trouble += 0.5
        if re.search(_FIN_UP, t2):
            fin = max(fin, 1.0)
        if re.search(_FIN_DOWN, t2):
            fin = min(fin, -1.0)
        if re.search(_RAILS, t2):
            paths.append(1.0)
        elif re.search(_MID, t2):
            paths.append(2.0)
        elif re.search(_WIDE, t2):
            paths.append(3.0)
        else:
            m = re.search(_TO, t2.replace("-", ""))
            if m:
                ends = [v for k, v in _PATH.items() if k in t2]
                if ends:
                    paths.append(float(np.mean(ends[:2])))
    return {"brk": max(brk) if brk else np.nan, "led": led, "trouble": trouble, "fin": fin,
            "path": float(np.mean(paths)) if paths else np.nan}


# ----- per-dog windows over earlier DAYS: one row a dog a day (a dog runs twice in a day only rarely: a trial and a race,
# or heats), lagged a whole day, so nothing from the day being priced enters a dog's form (model.lagsafe's rule)

class DogDays:
    """The runs folded to one row a dog a day (means of the day's values, its last grade, distance and weight), in
    date order, with windows over earlier days and a way back onto the runs."""

    def __init__(self, df: pd.DataFrame, mean_cols: list[str], last_cols: list[str]):
        self.key = ["dog_id", "race_date"]
        agg = {c: "mean" for c in mean_cols}
        agg.update({c: "last" for c in last_cols})
        dd = df.groupby(self.key, sort=False).agg(agg)
        dd["_runs"] = df.groupby(self.key, sort=False).size()
        dd = dd.reset_index()
        dd["_d"] = pd.to_datetime(dd.race_date)
        self.dd = dd.sort_values(["dog_id", "_d"], kind="stable").reset_index(drop=True)
        self.g = self.dd.groupby("dog_id", sort=False)
        self._index = pd.MultiIndex.from_frame(df[self.key])

    def prev(self, col: str, k: int = 1) -> pd.Series:
        return self.g[col].shift(k)

    def window(self, col: str, n: int, how: str = "mean") -> pd.Series:
        r = self.prev(col).groupby(self.dd.dog_id, sort=False).rolling(n, min_periods=1)
        v = r.max() if how == "max" else r.min() if how == "min" else r.std() if how == "std" else r.mean()
        return v.reset_index(level=0, drop=True).reindex(self.dd.index)

    def ewm(self, col: str, halflife: float = 3.0) -> pd.Series:
        return self.prev(col).groupby(self.dd.dog_id, sort=False).transform(
            lambda x: x.ewm(halflife=halflife, ignore_na=True).mean())

    def back(self, series: pd.Series) -> np.ndarray:
        """A dog-day series onto the runs."""
        s = pd.Series(series.values, index=pd.MultiIndex.from_frame(self.dd[self.key]))
        return s.reindex(self._index).values


def _shrunk(df: pd.DataFrame, group: str | list[str], col: str, k: float = 30.0) -> pd.Series:
    """The group's mean of ``col`` over earlier days, shrunk toward the population's earlier mean by k runs."""
    m = race_lagged_expanding_mean(df, group, col)
    n = race_lagged_expanding_count(df, group, col)
    d = df.assign(_pop=0)
    pop = race_lagged_expanding_mean(d, "_pop", col).fillna(df[col].mean())
    return (m.fillna(pop) * n + pop * k) / (n + k)


class GreyhoundMetricsEngine:
    """``calculate_all(runs)`` -> the runs with every metric added; ``features`` lists the model's columns."""

    def __init__(self, windows=(1, 3, 6), blocks=("parity",)):
        self.windows = windows
        self.blocks = tuple(blocks)
        self.features: list[str] = []
        self.block_features: dict[str, list[str]] = {}

    # ------------------------------------------------------------------------------------------------ per-run figures
    def _per_run(self, df: pd.DataFrame) -> pd.DataFrame:
        uniq = {c: parse_comment(c) for c in pd.unique(df.comment.fillna(""))}       # far fewer comments than runs
        parsed = pd.DataFrame([uniq[c] for c in df.comment.fillna("").values], index=df.index)
        df = df.join(parsed)
        # lengths behind the winner: the GBGB distances are to the dog in front, summed down the finishing order
        o = df.sort_values(["race_id", "position"])
        step = o.beaten_lengths.where(o.position > 1, 0.0).fillna(0.0)
        df["lbw"] = step.groupby(o.race_id).cumsum().reindex(df.index)
        by_time = (df.run_time - df.groupby("race_id").run_time.transform("min")) / LENGTH_S
        df["lbw"] = df.lbw.where(df.lbw.notna() & (df.lbw < 40), by_time).clip(upper=40)
        # standards from earlier days: the track and distance's mean winning calculated time, sectional, run-in
        df["_win_time"] = df.adjusted_time.where(df.won == 1)
        df["runin"] = df.adjusted_time - df.sectional
        key = ["track", "distance_m"]
        df["std_time"] = race_lagged_expanding_mean(df, key, "_win_time")
        df["std_sec"] = race_lagged_expanding_mean(df, key, "sectional")
        df["std_runin"] = race_lagged_expanding_mean(df, key, "runin")
        # the meeting's own variant: what the going allowance missed, from its winners (the run is in the past when
        # a later race reads it, so the whole meeting is known)
        dev = (df._win_time - df.std_time)
        df["variant"] = dev.groupby(df.meeting_id).transform("median").fillna(0.0)
        df["gsr"] = (df.std_time - (df.adjusted_time - df.variant)) / LENGTH_S
        df["esr"] = (df.std_sec - df.sectional) / LENGTH_S
        df["fsr"] = (df.std_runin - (df.runin - df.variant)) / LENGTH_S
        # a figure more than 40 lengths from the standard is a bad time or a wrong distance in the source, not a run
        for c in ("gsr", "esr", "fsr"):
            df[c] = df[c].where(df[c].abs() <= 40)
        # a run spoiled at the start (hand-slipped, fell) or a trial is not a fair rating
        bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False)
        for c in ("gsr", "esr", "fsr"):
            df[c] = df[c].where(~bad)
        df["gsr_clean"] = df.gsr.where(df.trouble == 0)              # rating from a clear run only
        df["gsr_trouble_adj"] = df.gsr + 1.5 * df.trouble             # a troubled run, credited ~1.5 lengths a bump
        return df

    # ----------------------------------------------------------------------------------------------- the dog's form
    def _dog_form(self, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
        df["_ae"] = df.won - df.sp_p_norm
        df["_trial"] = df.is_trial.astype(float)
        means = ["gsr", "gsr_clean", "gsr_trouble_adj", "esr", "fsr", "position", "lbw", "won", "placed2", "brk", "led",
                 "trouble", "path", "fin", "sp_p_norm", "_ae", "_trial"]
        dd = DogDays(df, means, ["grade", "distance_m", "weight_kg", "track", "trap"])
        day = {}
        day["runs_before"] = dd.g._runs.cumsum() - dd.dd._runs
        day["trials_before"] = (dd.g._trial.cumsum() - dd.dd._trial).astype(float)
        day["days_since"] = (dd.dd._d - dd.prev("_d")).dt.days.astype(float)
        day["days_since2"] = (dd.dd._d - dd.prev("_d", 2)).dt.days.astype(float)
        day["last_was_trial"] = dd.prev("_trial")
        for col, hows in (("gsr", ("mean", "max")), ("gsr_clean", ("mean",)), ("gsr_trouble_adj", ("mean", "max")),
                          ("esr", ("mean", "max")), ("fsr", ("mean",)), ("position", ("mean",)), ("lbw", ("mean",)),
                          ("won", ("mean",)), ("placed2", ("mean",)), ("brk", ("mean",)), ("led", ("mean",)),
                          ("trouble", ("mean",)), ("path", ("mean",)), ("fin", ("mean",))):
            for how in hows:
                for n in self.windows:
                    day[f"{col}_{how}{n}"] = dd.window(col, n, how)
        for col in ("gsr", "esr", "position"):
            day[f"{col}_ewm"] = dd.ewm(col)
        day["gsr_trend"] = day["gsr_mean1"] - day["gsr_mean6"]
        day["gsr_std6"] = dd.window("gsr", 6, "std")
        day["prev_grade"] = dd.prev("grade")
        day["prev_distance"] = dd.prev("distance_m")
        day["prev_weight"] = dd.prev("weight_kg")
        day["sp_p_mean3"] = dd.window("sp_p_norm", 3)
        day["dog_ae6"] = dd.window("_ae", 6)
        cols = {k: dd.back(v) for k, v in day.items()}
        # the same track, the same track and distance, today's trap: the dog's earlier days there (model.lagsafe)
        for key, name in ((["dog_id", "track"], "trk"), (["dog_id", "track", "distance_m"], "cd"),
                          (["dog_id", "trap"], "trap")):
            cols[f"{name}_runs"] = race_lagged_expanding_count(df, key, "gsr").values
            cols[f"{name}_gsr"] = race_lagged_expanding_mean(df, key, "gsr").values
            if name != "trap":
                cols[f"{name}_win"] = race_lagged_expanding_mean(df, key, "won").values
        cols["trap_gsr_vs_all"] = cols.pop("trap_gsr") - race_lagged_expanding_mean(df, "dog_id", "gsr").values
        # class and condition against the last day the dog ran
        cols["grade_change"] = df.grade.values - cols.pop("prev_grade")
        cols["dist_change"] = df.distance_m.values - cols.pop("prev_distance")
        cols["weight_change"] = df.weight_kg.values - cols.pop("prev_weight")
        cols["weight_vs_mean"] = df.weight_kg.values - race_lagged_expanding_mean(df, "dog_id", "weight_kg").values
        born = pd.to_datetime(df.born, format="%b-%Y", errors="coerce")
        cols["age_months"] = ((df.t - born).dt.days / 30.44).values
        cols["bitch"] = (df.sex.astype(str).str.lower() == "b").astype(float).values
        out = pd.DataFrame(cols, index=df.index)
        feats += list(out.columns)
        return out

    # ------------------------------------------------------------------------------------------ groups, earlier days
    def _groups(self, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
        cols = {}
        # the par of today's track and grade: the earlier winners' GSR
        df["_win_gsr"] = df.gsr.where(df.won == 1)
        par = race_lagged_expanding_mean(df, ["track", "race_class"], "_win_gsr")
        cols["grade_par"] = par
        # trap bias at today's track and distance: the trap's earlier win rate against its fair share
        df["_fair"] = 1.0 / df.field
        tw = race_lagged_expanding_sum(df, ["track", "distance_m", "trap"], "won")
        tf = race_lagged_expanding_sum(df, ["track", "distance_m", "trap"], "_fair")
        cols["trap_bias"] = (tw + 5.0) / (tf + 5.0)               # shrunk: 5 fair wins of prior
        # trainer and sire, shrunk; the trainer's A/E against the SP
        cols["trainer_win"] = _shrunk(df, "trainer", "won")
        df["_spp"] = df.sp_p_norm
        aw = race_lagged_expanding_sum(df, "trainer", "won")
        ae = race_lagged_expanding_sum(df, "trainer", "_spp")
        cols["trainer_ae"] = (aw.fillna(0) + 20.0) / (ae.fillna(0) + 20.0)
        cols["sire_win"] = _shrunk(df, "sire", "won", k=50.0)
        out = pd.DataFrame(cols, index=df.index)
        feats += list(out.columns)
        return out

    # ------------------------------------------------------------------------------------------- against the field
    def _race_relative(self, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
        cols = {}
        best_recent = df[["gsr_max3", "gsr_trouble_adj_max3"]].max(axis=1)
        cols["rating_vs_par"] = best_recent - df.grade_par
        for col in ("gsr_mean3", "gsr_max6", "gsr_ewm", "gsr_trouble_adj_max3", "esr_mean3", "esr_max3", "fsr_mean3",
                    "position_mean3", "won_mean6", "trainer_win", "rating_vs_par", "brk_mean6", "led_mean6"):
            x = df[col] if col in df else pd.Series(cols.get(col), index=df.index)
            grp = x.groupby(df.race_id)
            cols[f"{col}_rank"] = grp.rank(ascending=False, method="average")
            cols[f"{col}_gap"] = x - grp.transform("max")
            cols[f"{col}_z"] = (x - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
        # the pace map: early speed order across the traps; a dog with faster dogs drawn either side is squeezed
        esr = df.esr_mean3.fillna(df.esr_mean3.median())
        cols["early_rank"] = esr.groupby(df.race_id).rank(ascending=False)
        o = df.assign(_e=esr).sort_values(["race_id", "trap"])
        left = o.groupby("race_id")._e.shift(1)
        right = o.groupby("race_id")._e.shift(-1)
        squeeze = ((left > o._e).astype(float) + (right > o._e).astype(float))
        cols["squeeze"] = squeeze.reindex(df.index)
        inside_best = o.groupby("race_id")._e.transform(lambda s: s.shift(1).cummax())
        cols["faster_inside"] = (inside_best > o._e).astype(float).reindex(df.index)
        # running line against the trap: a railer drawn wide, a wide runner drawn on the inside
        path = df.path_mean6
        cols["path_vs_trap"] = (path - 2.0) - (df.trap - 3.5) / 1.25
        cols["path_conflict"] = (path - 2.0) * (df.trap - 3.5) * -1.0
        cols["open_race"] = (df.grade_family == "OR").astype(float)
        cols["sprint"] = (df.grade_family == "D").astype(float)
        cols["stayer"] = df.grade_family.isin(["S"]).astype(float)
        cols["hurdle"] = df.grade_family.isin(["H", "HP"]).astype(float)
        out = pd.DataFrame(cols, index=df.index)
        feats += list(out.columns) + ["trap", "field", "distance_m", "grade"]      # today's race, as it is
        return out

    def calculate_all(self, runs: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
        """Every metric. With the ``parity`` block (``greyhound.parity``) the horse model's families too, the Betfair
        SP history among them when ``prices`` (the greyhound price-file tables) is given."""
        if "parity" in self.blocks:
            from greyhound.parity import attach_bsp
            runs = attach_bsp(runs, prices)
        df = runs.sort_values(["t", "race_id", "trap"], kind="stable").reset_index(drop=True)
        import time
        clock = time.monotonic()
        df = self._per_run(df)
        self.timings = {"per_run": round(time.monotonic() - clock, 1)}
        feats: list[str] = []
        for name, step in (("dog_form", self._dog_form), ("groups", self._groups), ("race_relative", self._race_relative)):
            clock = time.monotonic()
            df = df.join(step(df, feats))
            self.timings[name] = round(time.monotonic() - clock, 1)
        self.block_features = {"base": list(dict.fromkeys(feats))}
        if "parity" in self.blocks:
            from greyhound import parity
            clock = time.monotonic()
            cols, names = parity.calculate(df, DogDays)
            df = df.join(cols)
            self.block_features["parity"] = names
            feats += names
            self.timings["parity"] = round(time.monotonic() - clock, 1)
        if "hrb" in self.blocks:
            from greyhound import hrb
            clock = time.monotonic()
            cols, names = hrb.calculate(df, DogDays)
            df = df.join(cols)
            self.block_features["hrb"] = names
            feats += names
            self.timings["hrb"] = round(time.monotonic() - clock, 1)
        if "lib" in self.blocks:
            from greyhound import lib
            clock = time.monotonic()
            cols, names = lib.calculate(df, DogDays)
            df = df.join(cols)
            self.block_features["lib"] = names
            feats += names
            self.timings["lib"] = round(time.monotonic() - clock, 1)
        self.features = list(dict.fromkeys(feats))
        return df
