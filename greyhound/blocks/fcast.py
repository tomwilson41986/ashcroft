"""Betfair's forecast price as a prior (reports/greyhound_roadmap.md L2; greyhound/forecast_price.py): the tissue
published before the market forms, joined to each run by track, date, off and trap.

Recorded only from 6 Oct 2026 (the first recorded greyhound catalogue), so the columns are NaN on every earlier run;
the gradient-boosted models take a missing value as its own branch. Features: the forecast's implied chance
normalised over the race's runners that have one (``fc_p``), its log against the race's mean (``fc_logp_z``), the
dog's rank on it (``fc_rank``, 1 the forecast favourite) and the share of the race with a forecast (``fc_cover``).

The forecast is published with the card, before the off, so it is pre-race; nothing of the race's result is read.

The table: ``TABLE`` when a caller has set it (``set_table``, the tests and the live path), else read once from the
recorded catalogues for the days the frame covers (``greyhound.forecast_price.load``; no recorded day, no read).

Live (not wired here): ``greyhound/live.py predict`` prices the day's cards from the same catalogue rows the recorder
keeps, so ``forecast_price.from_catalogue`` of that day's ``markets_greyhound.csv`` would give the card's forecasts,
joined to the card rows on the same key before the model is applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREFIX = "fc_"
TABLE: pd.DataFrame | None = None


def set_table(t: pd.DataFrame | None) -> None:
    global TABLE
    TABLE = t


def _table(df: pd.DataFrame) -> pd.DataFrame:
    if TABLE is not None:
        return TABLE
    from greyhound import forecast_price as fp
    days = pd.to_datetime(df.race_date, errors="coerce")
    last = days.max()
    if pd.isna(last) or last.date() < fp.FIRST_DAY:
        return pd.DataFrame(columns=fp.KEY + ["fc_price"])        # nothing recorded in the frame's span
    return fp.load(max(days.min().date(), fp.FIRST_DAY), last.date())


def build(df: pd.DataFrame) -> list[str]:
    t = _table(df)
    key = pd.DataFrame({"race_date": df.race_date.astype(str), "track": df.track.astype(str).str.strip(),
                        "race_time": df.race_time.astype(str).str[:5],
                        "trap": pd.to_numeric(df.trap, errors="coerce")}, index=df.index)
    price = pd.Series(np.nan, index=df.index)
    if len(t):
        tt = t.assign(race_date=t.race_date.astype(str), track=t.track.astype(str).str.strip(),
                      race_time=t.race_time.astype(str).str[:5], trap=pd.to_numeric(t.trap, errors="coerce"))
        tt = tt.drop_duplicates(["race_date", "track", "race_time", "trap"], keep="last")
        price = key.merge(tt[["race_date", "track", "race_time", "trap", "fc_price"]], how="left",
                          on=["race_date", "track", "race_time", "trap"]).fc_price.set_axis(df.index)
    race = df.race_id
    raw = 1.0 / price
    p = raw / raw.groupby(race).transform("sum")
    logp = np.log(p)
    z = logp - logp.groupby(race).transform("mean")
    cover = price.notna().groupby(race).transform("mean")
    df[f"{PREFIX}p"] = p.astype(np.float32)
    df[f"{PREFIX}logp_z"] = z.astype(np.float32)
    df[f"{PREFIX}rank"] = price.groupby(race).rank(method="average").astype(np.float32)
    df[f"{PREFIX}cover"] = cover.astype(np.float32)
    return [f"{PREFIX}{n}" for n in ("p", "logp_z", "rank", "cover")]
