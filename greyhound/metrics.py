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
LBW_CAP = 15.0                      # lengths behind the winner: the margin's cap (its 99th percentile)
K_CLASS = 10.0                      # winners of prior on a class's par at a track and trip (meeting variant)
K_MEET = 2.0                        # races of prior on a meeting's variant

# ----- the comment codes (GBGB's race comments, e.g. "QAw,Rls,Led1,ALd"), as regexes on each comma-separated token
# with its spaces removed, and on each "&"-joined part of it ("Crd&Led1": crowded, and led at the first bend). Read
# against the tokens GBGB wrote between 1 Jul 2025 and 5 Nov 2026 (18,229 distinct, 901,364 in all; the audit of
# 11 Nov 2026, finding 9, found the forms the first parser misread):
#   the break is the exit from the traps (QAw, SAw, MsdBrk); early pace (EP, LckEP) is the speed to the bend and is
#   its own measure - "SAw,EP" (a slow start, then pace) was read as a fast break;
#   led is the lead at the first bend: "Led1", "Ld-1", "LedFr1", "LdT1", always or soon led, led on the run-up, and
#   "LedTo3" (led until the third bend, so from the start) - but not "Led 1/2" (led at halfway), "Led2", "LedRnIn";
#   trouble is interference from another dog: crowded, bumped, baulked, forced to check, hampered, impeded, brought
#   down, a stumble; "badly" counts 2, "slightly" a half, and each place named counts ("Crd1&2" is two). The dog's
#   own line ("RanWide") is not trouble
_BREAK = [(r"^v(ery)?q(uick)?aw", 2.0), (r"^q(uick)?aw", 1.0), (r"^mod(erately)?aw", 0.0), (r"^s(low)?aw", -1.0),
          (r"^v(ery)?s(low)?aw", -2.0), (r"^m(i)?s(se)?dbr", -2.0), (r"^(stb|stumbled)(stt|start|st$)", -2.0)]
_EP = [(r"^e(arly)?p(ace)?$|^earlyp$", 1.0), (r"^(lck|lacked)(e(arly)?p(ace)?)?$", -1.0)]
_LED = (r"^(a(lways)?le?d$|s(oo)?nle?d|le?d(rn|run)up|le?d-?to|le?dt-?1(?![/\d])|le?d-?(fr|from)?-?1(?![/\d]))")
_TROUBLE = (r"^(v?b(adly)?|slt|slightly)?(crd|crowded|bmp|bumped|blk|baulked|balked|hmp|hampered|imp|impeded|stb|stumbled"
            r"|(fcd|forced)-?(to|t)?-?(b|badly)?-?(c(hec)?k|w(ide)?(\d|$))|ck|checked|brtdn|broughtdown"
            r"|knockedover|fell|fll)")
_BADLY = r"^(v?b|v?badly)(crd|crowded|bmp|bumped|blk|baulked|balked|hmp|hampered)"
_SLIGHT = r"^(slt|slightly)"
_PLACE = r"^(\d(/\d)?|rnup|runup|rnin|runin|stt|start|nrln|nearline)$"
_PATHS = r"(vw|vwide|verywide|wide|w|rls|rails|railed|middle|mid)"
_PATH_RE = rf"^{_PATHS}(?:-|to|t)?{_PATHS}?$"
_FIN_UP = r"^(ranon|rnon|fin(ished)?well|strfin|chl|chal|ldrnin|ledrunin|ledrnin|ldrunin|ldnrln|lednrln|lednear|heldon|hldon)"
_FIN_DOWN = r"^(fdd|faded|tired|wkn|weakened|outpaced|stayedonslowly)"
_PATH = {"rls": 1.0, "rails": 1.0, "railed": 1.0, "mid": 2.0, "middle": 2.0, "w": 3.0, "wide": 3.0, "vw": 3.0,
         "vwide": 3.0, "verywide": 3.0}


def _tokens(comment) -> list[str]:
    return [t.strip().lower() for t in str(comment or "").split(",") if t.strip()]


def parse_comment(comment) -> dict:
    """One run's comment -> break, early pace, led, trouble, path, finish."""
    brk, epace, led, trouble, fin, paths = [], [], 0.0, 0.0, 0.0, []
    for t in _tokens(comment):
        weight = None                                     # the weight of the token's last interference, for "&2"
        for part in t.replace(" ", "").split("&"):
            if not part:
                continue
            for pat, v in _BREAK:
                if re.search(pat, part):
                    brk.append(v)
                    break
            for pat, v in _EP:
                if re.search(pat, part):
                    epace.append(v)
                    break
            if re.search(_LED, part):
                led = 1.0
            if re.search(_TROUBLE, part):
                weight = 0.5 if re.search(_SLIGHT, part) else 2.0 if re.search(_BADLY, part) else 1.0
                trouble += weight
            elif weight is not None and re.search(_PLACE, part):
                trouble += weight                         # "Crd1&2": the same again at the second bend
            else:
                weight = None
            if re.search(_FIN_UP, part):
                fin = max(fin, 1.0)
            if re.search(_FIN_DOWN, part):
                fin = min(fin, -1.0)
            m = re.match(_PATH_RE, part)
            if m:
                paths.append(float(np.mean([_PATH[g] for g in m.groups() if g])))
    mean = lambda v: float(np.mean(v)) if v else np.nan
    return {"brk": mean(brk), "epace": mean(epace), "led": led, "trouble": trouble, "fin": fin, "path": mean(paths)}


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
        # lengths behind the winner: the GBGB distances are to the dog in front, summed down the finishing order. A
        # distance GBGB did not give as a number ("Dis", "DNF") is unknown for that dog and for every dog behind it:
        # those are read from the times, and with no time they stay unknown. The margin is capped at LBW_CAP: past
        # the 99th percentile of the margins (15.5 lengths in 1 Jul 2025 to 5 Nov 2026) it measures the incident,
        # not the dog. Until 11 Nov 2026 a missing distance was summed as 0, so a dog beaten a distance took the
        # lengths of the dog in front of it (4,210 runs in those months, engine lbw a median 7.5 lengths; the 121
        # with a time were a median 32 behind: the audit of 11 Nov 2026, findings 8 and 54). Read on the same
        # runs, the 6-run mean of the margin lines up with the SP at -0.198 under the old rule, -0.176 if the
        # unknown are given the old 40-length cap (one such dog moves its whole race's mean), -0.204 as here
        o = df.sort_values(["race_id", "position"])
        step = o.beaten_lengths.where(o.position > 1, 0.0).where(o.position.notna())
        unknown = step.isna().groupby(o.race_id).cummax()
        df["lbw"] = step.fillna(0.0).groupby(o.race_id).cumsum().where(~unknown).reindex(df.index)
        by_time = (df.run_time - df.groupby("race_id").run_time.transform("min")) / LENGTH_S
        df["lbw"] = df.lbw.where(df.lbw.notna() & (df.lbw < LBW_CAP), by_time).clip(lower=0.0, upper=LBW_CAP)
        # standards from earlier days: the track and distance's mean winning calculated time, sectional, run-in, of
        # GRADED races only. Until 11 Nov 2026 a trial's first dog (often running alone) counted as a winner in the
        # standard and in the meeting's variant, and a fall or a hand-slipped start counted in the sectional and
        # run-in standards (the review of 11 Nov, reports/greyhound_model.md section 41.1; the owner: "fix this")
        trial = df.is_trial.fillna(False).astype(bool) if "is_trial" in df else pd.Series(False, index=df.index)
        bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False)
        fair = ~trial & ~bad
        df["_win_time"] = df.adjusted_time.where(df.won == 1)
        df["runin"] = df.adjusted_time - df.sectional
        df["_std_win"] = df._win_time.where(fair)
        df["_std_sec"] = df.sectional.where(fair)
        df["_std_runin"] = df.runin.where(fair)
        key = ["track", "distance_m"]
        df["std_time"] = race_lagged_expanding_mean(df, key, "_std_win")
        df["std_sec"] = race_lagged_expanding_mean(df, key, "_std_sec")
        df["std_runin"] = race_lagged_expanding_mean(df, key, "_std_runin")
        # the meeting's own variant: what the going allowance missed, from its graded winners (the run is in the
        # past when a later race reads it, so the whole meeting is known); a meeting of trials alone has none.
        # Each winner is read against the standard PLUS its class's par (the class's earlier winners against the
        # standard at the track and trip, shrunk by K_CLASS winners), and the median is shrunk by K_MEET races.
        # Until 11 Nov 2026 the winners were read against the all-class standard alone, so a card of low grades
        # had slow winners and a positive variant that was then credited to every dog on it: over 3,429 meetings
        # the variant's correlation with the card's mean grade was 0.41 and the best and worst quarters of cards
        # were 1.9 lengths apart (the audit of 11 Nov 2026, finding 2)
        dev = (df._std_win - df.std_time)
        df["_dev"] = dev.where(dev.abs() <= 40 * LENGTH_S)            # a winner 40 lengths out is a bad time
        ck = key + ["race_class"]
        cn = race_lagged_expanding_count(df, ck, "_dev")
        df["class_par_s"] = race_lagged_expanding_mean(df, ck, "_dev").fillna(0.0) * cn / (cn + K_CLASS)
        resid = (df._dev - df.class_par_s).groupby(df.meeting_id)
        n = resid.transform("count")
        df["variant"] = (resid.transform("median") * n / (n + K_MEET)).fillna(0.0)
        df["gsr"] = (df.std_time - (df.adjusted_time - df.variant)) / LENGTH_S
        df["esr"] = (df.std_sec - df.sectional) / LENGTH_S
        df["fsr"] = (df.std_runin - (df.runin - df.variant)) / LENGTH_S
        # a figure more than 40 lengths from the standard is a bad time or a wrong distance in the source, not a run
        for c in ("gsr", "esr", "fsr"):
            df[c] = df[c].where(df[c].abs() <= 40)
        # a run spoiled at the start (hand-slipped, fell) is not a fair rating. A trial keeps its figure (it is the
        # only form a debutant or a dog back from a break has; the model is told it was a trial), read against the
        # graded standard
        for c in ("gsr", "esr", "fsr"):
            df[c] = df[c].where(~bad)
        df = df.drop(columns=["_std_win", "_std_sec", "_std_runin", "_dev"])
        df["gsr_clean"] = df.gsr.where(df.trouble == 0)              # rating from a clear run only
        # a troubled run, credited ~1.5 lengths a bump, two bumps at most (a judgement, not a fit: the fitted price
        # of trouble is the expected-time table's, greyhound/pit/xtime.py)
        df["gsr_trouble_adj"] = df.gsr + 1.5 * df.trouble.clip(upper=2.0)
        return df

    # ----------------------------------------------------------------------------------------------- the dog's form
    def _dog_form(self, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
        df["_ae"] = df.won - df.sp_p_norm
        # no trial reaches this frame (GBGB serves trials with no off time: data.clean), so the engine keeps no trial
        # counts: until 11 Nov 2026 trials_before and last_was_trial counted the intertrack races, which are races.
        # A dog's trials are read on their own channel (greyhound/pit/trials.py)
        means = ["gsr", "gsr_clean", "gsr_trouble_adj", "esr", "fsr", "position", "lbw", "won", "placed2", "brk", "epace",
                 "led", "trouble", "path", "fin", "sp_p_norm", "_ae"]
        if "grade_scale" not in df:                       # a frame not made by data.clean (a test's)
            df["grade_scale"] = df.grade_family if "grade_family" in df else "A"
        dd = DogDays(df, means, ["grade", "grade_scale", "distance_m", "weight_kg", "track", "trap"])
        day = {}
        day["runs_before"] = dd.g._runs.cumsum() - dd.dd._runs
        day["days_since"] = (dd.dd._d - dd.prev("_d")).dt.days.astype(float)
        day["days_since2"] = (dd.dd._d - dd.prev("_d", 2)).dt.days.astype(float)
        for col, hows in (("gsr", ("mean", "max")), ("gsr_clean", ("mean",)), ("gsr_trouble_adj", ("mean", "max")),
                          ("esr", ("mean", "max")), ("fsr", ("mean",)), ("position", ("mean",)), ("lbw", ("mean",)),
                          ("won", ("mean",)), ("placed2", ("mean",)), ("brk", ("mean",)), ("epace", ("mean",)),
                          ("led", ("mean",)),
                          ("trouble", ("mean",)), ("path", ("mean",)), ("fin", ("mean",))):
            for how in hows:
                for n in self.windows:
                    day[f"{col}_{how}{n}"] = dd.window(col, n, how)
        for col in ("gsr", "esr", "position"):
            day[f"{col}_ewm"] = dd.ewm(col)
        day["gsr_trend"] = day["gsr_mean1"] - day["gsr_mean6"]
        day["gsr_std6"] = dd.window("gsr", 6, "std")
        day["prev_grade"] = dd.prev("grade")
        day["prev_scale"] = dd.prev("grade_scale")
        day["prev_distance"] = dd.prev("distance_m")
        day["prev_weight"] = dd.prev("weight_kg")
        day["sp_p_mean3"] = dd.window("sp_p_norm", 3)
        day["dog_ae6"] = dd.window("_ae", 6)
        prev_scale = dd.back(day.pop("prev_scale"))
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
        # ... on one ladder only: an A3 after a D2 is a change of trip, not of class (data.GRADE_SCALE)
        same = pd.Series(prev_scale, index=df.index).eq(df.grade_scale) & df.grade_scale.notna()
        cols["grade_change"] = np.where(same, df.grade.values - cols.pop("prev_grade"), np.nan)
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
        df["_won_sp"] = df.won.where(df.sp_p_norm.notna())            # wins the SP priced: the same runs as the SP's
        aw = race_lagged_expanding_sum(df, "trainer", "_won_sp")
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
        # the pace map: early speed order across the traps; a dog with a faster dog in the box beside it is squeezed.
        # A dog with no sectional behind it has no early speed to compare (NaN, not the frame's median: 27.6% of
        # runners were given one, and the median read later days), and the boxes beside are trap - 1 and trap + 1:
        # an empty box is room, not the next dog along (28.9% of races have one between two runners; the audit of
        # 11 Nov 2026, finding 5)
        esr = df.esr_mean3
        mine = esr.to_numpy(float)
        cols["early_rank"] = esr.groupby(df.race_id).rank(ascending=False)
        box = esr.groupby([df.race_id, df.trap]).first()
        beside = lambda step: box.reindex(pd.MultiIndex.from_arrays([df.race_id, df.trap + step])).to_numpy(float)
        with np.errstate(invalid="ignore"):
            squeeze = (beside(-1) > mine).astype(float) + (beside(1) > mine).astype(float)
        cols["squeeze"] = np.where(np.isfinite(mine), squeeze, np.nan)
        o = df.assign(_e=esr.fillna(-np.inf)).sort_values(["race_id", "trap"])
        inside_best = o.groupby("race_id")._e.transform(lambda x: x.shift(1, fill_value=-np.inf).cummax()).reindex(df.index)
        cols["faster_inside"] = np.where(np.isfinite(mine), (inside_best.to_numpy(float) > mine).astype(float), np.nan)
        # running line against the trap: a railer drawn wide (-), a wide runner drawn on the inside (+); 0 when the
        # line fits the box (rails in 1, wide in 6). Until 11 Nov 2026 the trap term ran -2..2 against the path's
        # -1..1, so a railer in trap 1 scored +1 (finding 10)
        path = df.path_mean6
        cols["path_vs_trap"] = (path - 2.0) - (df.trap - 3.5) / 2.5
        cols["path_conflict"] = (path - 2.0) * (df.trap - 3.5) * -1.0
        cols["open_race"] = (df.grade_family == "OR").astype(float)
        cols["sprint"] = (df.grade_family == "D").astype(float)
        cols["stayer"] = df.grade_family.isin(["S"]).astype(float)
        # a handicap (class HP): until 11 Nov 2026 this column was named "hurdle" - HP was read as hurdles, and the
        # real hurdle races (race_type, 22 of them in 1 Jul 2025 to 5 Nov 2026, classed OR) were not flagged
        cols["is_handicap"] = (df.grade_family == "HP").astype(float)
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
        # the roadmap's F3, F5 and F6 (reports/greyhound_roadmap.md 3.1): track-normalised times with a mixed-model
        # meeting variant, trainer intent, ability with its uncertainty; and the expected final time (the owner, 3 Nov);
        # each its own block, earlier days only
        for name in ("times", "intent", "ability", "etime"):
            if name in self.blocks:
                import importlib
                mod = importlib.import_module(f"greyhound.{name}")
                clock = time.monotonic()
                cols, names = mod.calculate(df, DogDays)
                df = df.join(cols)
                self.block_features[name] = names
                feats += names
                self.timings[name] = round(time.monotonic() - clock, 1)
        self.features = list(dict.fromkeys(feats))
        return df
