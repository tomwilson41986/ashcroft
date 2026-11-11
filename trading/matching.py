"""The model's runners on Betfair's markets.

The 06:00 record (predictions/<date>.csv) already carries each runner's market_id and
selection_id. A predict_bfsp_today.py CSV does not: its runners are found by course, UK start
time and name, with the helpers betfair_prices.py matches the historic price files with.
"""

from __future__ import annotations

import logging

import pandas as pd

from trading.exchange import Market

log = logging.getLogger(__name__)


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
    bf = bf.drop_duplicates(["_k_track", "_k_time", "_k_horse"])
    p = p.merge(bf, on=["_k_track", "_k_time", "_k_horse"], how="left")
    # a course under two names (Betfair's "Bangor-on-Dee", the card's "Bangor": no Bangor runner was matched, so none
    # traded, until 11 Nov 2026): what the course name leaves unmatched is found by start time and name, then by
    # name alone, each only where it points to one runner of the day
    for keys in (["_k_time", "_k_horse"], ["_k_horse"]):
        miss = p["market_id"].isna()
        if not miss.any():
            break
        one = bf.drop_duplicates(keys, keep=False).set_index(keys)[["market_id", "selection_id"]]
        idx = pd.MultiIndex.from_frame(p.loc[miss, keys]) if len(keys) > 1 else pd.Index(p.loc[miss, keys[0]])
        found = one.reindex(idx)
        p.loc[miss, "market_id"] = found["market_id"].to_numpy()
        p.loc[miss, "selection_id"] = found["selection_id"].to_numpy()
        got = miss & p["market_id"].notna()
        if got.any():
            log.warning("Matched without the course name (%s): %d runners at %s", "+".join(k[3:] for k in keys),
                        int(got.sum()), sorted(set(venue[got.to_numpy()].astype(str))))
    left = p["market_id"].isna()
    if left.any():
        log.warning("No Betfair market for %d priced runners, by course: %s", int(left.sum()),
                    venue[left.to_numpy()].astype(str).value_counts().to_dict())
    return p.drop(columns=["_k_track", "_k_time", "_k_horse"])
