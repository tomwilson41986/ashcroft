"""The model's runners on Betfair's markets.

The 06:00 record (predictions/<date>.csv) already carries each runner's market_id and
selection_id. A predict_bfsp_today.py CSV does not: its runners are found by course, UK start
time and name, with the helpers betfair_prices.py matches the historic price files with.
"""

from __future__ import annotations

import pandas as pd

from trading.exchange import Market


def _norm(s: pd.Series, fn) -> pd.Series:
    return s.astype(str).map(fn)


def attach_ids(pred: pd.DataFrame, markets: list[Market]) -> pd.DataFrame:
    """Predictions with market_id and selection_id, found by course, start time and name where
    the file lacks them. Unmatched runners keep empty ids and are not traded."""
    import betfair_prices as bp
    p = pred.copy()
    if {"market_id", "selection_id"} <= set(p.columns) and p["market_id"].notna().all():
        p["market_id"] = p["market_id"].astype(str)
        p["selection_id"] = p["selection_id"].astype("int64")
        return p
    venue = p["venue"] if "venue" in p else p["track"]
    name = p["runner_name"] if "runner_name" in p else p["horse_name"]
    p["_k_track"] = _norm(venue, bp.normalise_track)
    p["_k_time"] = p["race_time"].astype(str).map(bp.race_time_to_24h)
    p["_k_horse"] = _norm(name, bp.normalise_horse)
    rows = []
    for m in markets:
        hhmm = bp.utc_iso_to_uk_hhmm(m.start.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        for sid, rname in m.runners.items():
            rows.append({"market_id": m.market_id, "selection_id": int(sid), "_k_track": bp.normalise_track(m.venue),
                         "_k_time": hhmm, "_k_horse": bp.normalise_horse(rname)})
    bf = pd.DataFrame(rows)
    if bf.empty:
        p["market_id"], p["selection_id"] = None, None
        return p.drop(columns=["_k_track", "_k_time", "_k_horse"])
    p = p.drop(columns=[c for c in ("market_id", "selection_id") if c in p.columns])
    p = p.merge(bf, on=["_k_track", "_k_time", "_k_horse"], how="left")
    return p.drop(columns=["_k_track", "_k_time", "_k_horse"])
