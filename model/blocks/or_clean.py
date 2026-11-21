"""The official rating with 0 read as unrated (the owner's review of 20 Nov 2026, item F2 / C1).

The results table holds 0 for a horse with no official rating: 538,146 rows, 27% of the history
(outputs/horse_v0_20261118/data_audit.md). The newer code maps 0 to missing (model/form_windows.py,
model/primitives.py, model/blocks/kalman.py, handicap_angles.py); the older families read it as a rating of 0:

    or_num, median_or_num, or_vs_median, or_vs_max   the race's median and today's place against it count the
                                                     unrated as 0 (model/bfsp_features.py)
    or_change, or_change_3, or_vs_best, or_off_peak, or_vs_last_win   a horse unrated last time and rated 70 today
                                                     reads "+70 lb" (model/custom_metrics.py _calc_or_trajectory)
    or_momentum                                      the same, two runs apart (model/financial_features.py)
    FCS (and rFCS, rw_FCS_*)                         the race's mean OR with the unrated at 0
    rr_official_rating_*, rw_field_*_official_rating, rw_or_change_*   the race-relative readings of the above

Here each is rebuilt with 0 as missing, under its own name, so a loop arm can withhold the old ones and read these
(research/loop.json "drop"). Today's rating is the card's, known before the off; every history reading is from the
horse's rated runs on earlier days only:

    orc_or            today's rating (NaN unrated)        orc_rated        1 rated today, 0 not
    orc_race_mean     the rated runners' mean             orc_race_median  their median
    orc_rated_share   share of the field rated            orc_vs_median    rating less the rated median
    orc_vs_max        rating less the race's top rating   orc_z            (rating - mean) / sd over the rated, 3+
    orc_gap           rating less the best rival's        orc_field_sd     sd of the rated field's ratings
    orc_field_top2    mean of the two top ratings
    orc_change        today's rating less its last rated run's on an earlier day
    orc_change_3      today's less its rating three rated runs back
    orc_best          the best rating of its earlier rated runs
    orc_vs_best       today's less that best              orc_off_peak     1 when 5 lb or more below it
    orc_vs_last_win   today's less its rating when it last won (a rated win, an earlier day)
    orc_momentum      its last rated run's rating less the one three rated runs back
    orc_change_z      orc_change standardised within today's race (3+ readings)
    orc_runs_rated    rated runs on earlier days
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import day_index, race_key

FEATURES = ["orc_or", "orc_rated", "orc_race_mean", "orc_race_median", "orc_rated_share", "orc_vs_median",
            "orc_vs_max", "orc_z", "orc_gap", "orc_field_sd", "orc_field_top2", "orc_change", "orc_change_3",
            "orc_best", "orc_vs_best", "orc_off_peak", "orc_vs_last_win", "orc_momentum", "orc_change_z",
            "orc_runs_rated"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "official_rating", "placing_numerical"]
OFF_PEAK_LBS = 5.0
#: the history only: the race readings use today's card alone, so the block reads results through orc_vs_last_win
READS_RESULTS = True


def _names(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def _race_readings(orr: np.ndarray, race: np.ndarray) -> dict[str, np.ndarray]:
    v = pd.Series(orr)
    g = v.groupby(race)
    n_rated = g.transform("count").to_numpy(float)
    rows = pd.Series(1.0, index=v.index).groupby(race).transform("sum").to_numpy(float)
    mean, sd = g.transform("mean").to_numpy(float), g.transform("std").to_numpy(float)
    top = g.transform("max").to_numpy(float)
    second = v.where(v < top).groupby(race).transform("max").to_numpy(float)
    ties = (v == top).groupby(race).transform("sum").to_numpy(float)
    rival = np.where((orr == top) & (ties < 2), second, top)
    top2 = v.groupby(race).transform(lambda s: s.nlargest(2).mean() if s.notna().any() else np.nan).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where((n_rated >= 3) & (sd > 0), (orr - mean) / sd,
                     np.where((n_rated >= 3) & (sd == 0) & np.isfinite(orr), 0.0, np.nan))
    return {"orc_race_mean": np.where(n_rated > 0, mean, np.nan),
            "orc_race_median": g.transform("median").to_numpy(float),
            "orc_rated_share": n_rated / rows,
            "orc_vs_median": orr - g.transform("median").to_numpy(float),
            "orc_vs_max": orr - top,
            "orc_z": z,
            "orc_gap": np.where(n_rated >= 2, orr - rival, np.nan),
            "orc_field_sd": np.where(n_rated >= 2, sd, np.nan),
            "orc_field_top2": top2}


def _history(h: np.ndarray, day: np.ndarray, orr: np.ndarray, won: np.ndarray) -> dict[str, np.ndarray]:
    """Per row, from the horse's rated runs on earlier days: the last rating, the one three rated runs back, the
    best, the rating at its last rated win, and the count."""
    n = len(h)
    d = pd.DataFrame({"h": h, "day": day, "orr": orr, "won_or": np.where(won == 1, orr, np.nan)})
    d = d[(d["h"] >= 0) & (d["day"] >= 0)]
    # one record a horse a day (a second run the same day reads nothing of the first)
    days = d.groupby(["h", "day"], sort=True).agg(orr=("orr", "max"), won_or=("won_or", "max")).reset_index()
    rated = days[days["orr"].notna()].copy()
    rated["k"] = rated.groupby("h").cumcount()                       # 0, 1, 2 ... the horse's rated days in order
    # for each horse-day, how many rated days come strictly before it
    days["before"] = days.groupby("h")["orr"].transform(lambda s: s.notna().cumsum().shift(1, fill_value=0))
    look = rated.set_index(["h", "k"])["orr"]

    def kth_back(back: int) -> np.ndarray:
        k = days["before"].to_numpy() - back
        idx = pd.MultiIndex.from_arrays([days["h"].to_numpy(), k])
        return np.where(k >= 0, look.reindex(idx).to_numpy(float), np.nan)

    days["last"] = kth_back(1)
    days["third"] = kth_back(3)
    best = days.groupby("h")["orr"].cummax()
    days["best"] = best.groupby(days["h"]).ffill().groupby(days["h"]).shift(1).to_numpy(float)
    lw = days.groupby("h")["won_or"].ffill()
    days["last_win"] = lw.groupby(days["h"]).shift(1).to_numpy(float)
    days["n"] = days["before"].astype(float)
    m = pd.DataFrame({"h": h, "day": day}).merge(days[["h", "day", "last", "third", "best", "last_win", "n"]],
                                                    on=["h", "day"], how="left")
    out = {c: m[c].to_numpy(float) for c in ("last", "third", "best", "last_win", "n")}
    if len(m) != n:
        raise ValueError("or_clean: the history join changed the row count")
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    race = pd.factorize(race_key(df), sort=True)[0]
    h = _names(df["horse_name"])
    orr = pd.to_numeric(df["official_rating"], errors="coerce").where(lambda s: s > 0).to_numpy(float)
    pos = pd.to_numeric(df["placing_numerical"], errors="coerce").to_numpy(float)
    won = np.where(np.isfinite(pos), (pos == 1).astype(float), np.nan)

    cols: dict[str, np.ndarray] = {"orc_or": orr, "orc_rated": np.isfinite(orr).astype(float)}
    cols.update(_race_readings(orr, race))
    hist = _history(h, day, orr, won)
    cols["orc_change"] = orr - hist["last"]
    cols["orc_change_3"] = orr - hist["third"]
    cols["orc_best"] = hist["best"]
    cols["orc_vs_best"] = orr - hist["best"]
    cols["orc_off_peak"] = np.where(np.isfinite(orr) & np.isfinite(hist["best"]),
                                    (hist["best"] - orr >= OFF_PEAK_LBS).astype(float), np.nan)
    cols["orc_vs_last_win"] = orr - hist["last_win"]
    cols["orc_momentum"] = hist["last"] - hist["third"]
    ch = pd.Series(cols["orc_change"])
    g = ch.groupby(race)
    cnt, mean, sd = g.transform("count").to_numpy(float), g.transform("mean").to_numpy(float), g.transform("std").to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["orc_change_z"] = np.where((cnt >= 3) & (sd > 0), (cols["orc_change"] - mean) / sd,
                                        np.where((cnt >= 3) & (sd == 0) & np.isfinite(cols["orc_change"]), 0.0, np.nan))
    cols["orc_runs_rated"] = hist["n"]
    new = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()}, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
