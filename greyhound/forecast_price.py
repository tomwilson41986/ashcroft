"""Betfair's forecast price for each greyhound (reports/greyhound_roadmap.md L2): the tissue price published with the
runner's metadata before the market forms (``FORECASTPRICE_NUMERATOR`` / ``_DENOMINATOR``), a prior where the
exchange book is still hollow.

The recorder keeps it in its catalogue rows (``betfair_recorder.catalogue_rows``: ``forecast_num``, ``forecast_den``)
in the day files ``betfair_live/<day>/markets_greyhound.csv`` (the recorder's) and ``markets_greyhound_live.csv``
(greyhound/book_record.py's), from the first recorded greyhound day (6 Oct 2026). Nothing before that day exists:
a race without a recorded forecast has none (NaN), never an invented one.

Each WIN runner is keyed as the GBGB rows are: the venue as the track, the UK date and the off as ``HH:MM``, the
trap from the runner's name ("1. Dog Name"). The decimal forecast price is ``1 + numerator / denominator``.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FIRST_DAY = date(2026, 10, 6)
FILES = ("markets_greyhound.csv", "markets_greyhound_live.csv")
KEY = ["race_date", "track", "race_time", "trap"]


def from_catalogue(markets: pd.DataFrame) -> pd.DataFrame:
    """The catalogue rows' WIN runners with a forecast price, keyed as the GBGB rows (``KEY``) with ``fc_price``."""
    if markets is None or not len(markets) or "forecast_num" not in markets:
        return pd.DataFrame(columns=KEY + ["market_id", "selection_id", "fc_price"])
    m = markets.copy()
    if "market_type" in m:
        m = m[m.market_type.fillna("WIN").astype(str).isin(["WIN", ""])]
    num = pd.to_numeric(m.forecast_num, errors="coerce")
    den = pd.to_numeric(m.forecast_den, errors="coerce")
    m = m.assign(fc_price=1.0 + num / den.where(den > 0))
    m = m[np.isfinite(m.fc_price) & (m.fc_price > 1.0)]
    if not len(m):
        return pd.DataFrame(columns=KEY + ["market_id", "selection_id", "fc_price"])
    if "recorded_utc" in m:
        m = m.sort_values("recorded_utc", kind="stable")
    m = m.drop_duplicates(["market_id", "selection_id"], keep="last")
    uk = pd.to_datetime(m.market_start_utc, utc=True, errors="coerce").dt.tz_convert("Europe/London")
    out = pd.DataFrame({
        "race_date": uk.dt.strftime("%Y-%m-%d"), "track": m.venue.astype(str).str.strip(),
        "race_time": uk.dt.strftime("%H:%M"),
        "trap": pd.to_numeric(m.runner_name.astype(str).str.extract(r"^(\d+)\.")[0], errors="coerce"),
        "market_id": m.market_id.astype(str), "selection_id": pd.to_numeric(m.selection_id, errors="coerce"),
        "fc_price": m.fc_price.astype(float)})
    return out.dropna(subset=["trap", "race_date"]).drop_duplicates(KEY, keep="last").reset_index(drop=True)


def load(first: date, last: date, read=None) -> pd.DataFrame:
    """Every recorded day's forecast prices from ``first`` to ``last`` (both files a day). ``read(day, name)``: the
    day-file reader (default ``betfair_recorder._read_day_csv``: the server's copy, else S3's)."""
    if read is None:
        from betfair_recorder import _read_day_csv as read
    first = max(first, FIRST_DAY)
    frames, d = [], first
    while d <= last:
        for name in FILES:
            try:
                m = read(d, name)
            except Exception as exc:                     # one unreadable file: noted, the rest still read
                log.warning("%s %s not read (%s)", d, name, exc)
                m = None
            if m is not None and len(m):
                frames.append(from_catalogue(m))
        d += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=KEY + ["market_id", "selection_id", "fc_price"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(KEY, keep="last").reset_index(drop=True)
