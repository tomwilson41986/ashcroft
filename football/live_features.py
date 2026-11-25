"""The historic price path as the live feed would have seen it: Betfair's ticks sampled at the feed's polling times.

racing² football, 25 Nov 2026 (the owner: the back-to-lay strategy live, GBP 2 a bet, GBP 50 a day). The move model
was trained on the warehouse's price path, built from Betfair's BASIC files: a runner's last traded price about once
a minute while it trades. The live football feed (``football/exchange_feed.py``) reads a market every 30 minutes
until six hours before the off, then every 10, 5, 2 and 1 minutes (``CADENCE``), from 24 hours out at most (its
catalogue looks a day ahead). A model trained on the finer path would be fed a coarser one live.

So the model the live strategy uses is trained on the history thinned to the feed's times: for every runner, the last
traded price at each polling time (counted back from the scheduled kick-off with the feed's own cadence), kept as a
"tick" only when it changed since the previous poll. ``gold_path.build`` then computes the same path features from
those ticks as from the live feed's polls (``ticks_from_polls``): one definition, two sources, the same numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.exchange_feed import cadence_seconds

LOOKBACK_MIN = 1440.0                 # the feed's catalogue looks 24 hours ahead: no poll before that


def poll_times(ko_ms: int, first_ms: int, lookback_min: float = LOOKBACK_MIN) -> np.ndarray:
    """The feed's polling times for a market kicking off at ``ko_ms`` (ms), oldest first: counted back from the
    kick-off with the feed's cadence, no earlier than ``first_ms`` or ``lookback_min`` before the off."""
    start = max(int(first_ms), int(ko_ms - lookback_min * 60_000))
    out, t = [], int(ko_ms) - 1                       # the last poll before the off
    while t >= start:
        out.append(t)
        mins = (ko_ms - t) / 60_000.0
        t -= int(cadence_seconds(mins) * 1000)
    return np.array(out[::-1], dtype=np.int64)


def sample_runner(pt: np.ndarray, ltp: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(times, prices) of the polls at which the runner's last traded price had changed since the previous poll.
    ``pt`` sorted ascending (ms)."""
    idx = np.searchsorted(pt, grid, side="right") - 1
    ok = idx >= 0
    g, p = grid[ok], ltp[idx[ok]]
    if not len(g):
        return g, p
    keep = np.ones(len(p), dtype=bool)
    keep[1:] = p[1:] != p[:-1]
    return g[keep], p[keep]


def thin_ticks(tick: pd.DataFrame, kickoff_ms: dict) -> pd.DataFrame:
    """silver/tick rows (market_id, market_type, selection_id, handicap, pt in ms, seq, ltp, pre_off) thinned to the
    feed's polling times; the same columns back. ``kickoff_ms``: market_id -> scheduled kick-off (ms). Markets with no
    kick-off are left out."""
    t = tick[tick["pre_off"].astype(bool) & tick["market_id"].isin(kickoff_ms.keys())]
    t = t.sort_values(["market_id", "selection_id", "handicap", "pt", "seq"], kind="stable")
    out = []
    for (mid, sid, hc), g in t.groupby(["market_id", "selection_id", "handicap"], sort=False):
        pt, ltp = g["pt"].to_numpy(np.int64), g["ltp"].to_numpy(float)
        ko = kickoff_ms[mid]
        times, prices = sample_runner(pt, ltp, poll_times(ko, pt[0]))
        if not len(times):
            continue
        out.append(pd.DataFrame({"market_id": mid, "market_type": g["market_type"].iloc[0], "selection_id": sid,
                                 "handicap": hc, "pt": times, "seq": 0, "ltp": prices, "pre_off": True}))
    cols = ["market_id", "market_type", "selection_id", "handicap", "pt", "seq", "ltp", "pre_off"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def ticks_from_polls(books: pd.DataFrame, market_types: dict) -> pd.DataFrame:
    """The live feed's recorded polls (football/exchange_feed.py book rows: polled_utc, market_id, selection_id,
    last_traded, inplay, status) as ticks: one row a poll at which a runner's last traded price changed, before the
    market turned in play. ``market_types``: market_id -> market type."""
    b = books[(pd.to_numeric(books["inplay"], errors="coerce").fillna(0) == 0)
              & books["market_id"].astype(str).isin(market_types.keys())].copy()
    t = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
    b["pt"] = (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)        # ms, whatever the resolution
    b["ltp"] = pd.to_numeric(b["last_traded"], errors="coerce")
    b = b[b["ltp"] > 1.0].sort_values(["market_id", "selection_id", "pt"])
    prev = b.groupby(["market_id", "selection_id"])["ltp"].shift(1)
    b = b[prev.isna() | (b["ltp"] != prev)]
    return pd.DataFrame({"market_id": b["market_id"].astype(str), "market_type": b["market_id"].astype(str).map(market_types),
                         "selection_id": b["selection_id"].astype("int64"), "handicap": 0.0, "pt": b["pt"],
                         "seq": 0, "ltp": b["ltp"], "pre_off": True}).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------
# the back-to-lay model's other features, from the live polls (football/b2l.py's FEATURES)
# --------------------------------------------------------------------------------------------------------------------

def book_at(ticks: pd.DataFrame, t_ms: int) -> pd.DataFrame:
    """Each selection's last traded price as of ``t_ms`` and its age in seconds (ticks of one market)."""
    t = ticks[ticks["pt"] <= t_ms].sort_values("pt")
    last = t.groupby("selection_id").tail(1)
    return pd.DataFrame({"selection_id": last["selection_id"].to_numpy(), "ltp": last["ltp"].to_numpy(float),
                         "age_s": (t_ms - last["pt"].to_numpy(np.int64)) / 1000.0})


def implied(p_home: float, p_draw: float, p_away: float, p_over: float) -> dict:
    """gold/implied's total, supremacy and draw premium from the normalised 1X2 and over 2.5 chances."""
    from football.warehouse.gold_implied import rho_from_draw, supremacy_from_1x2, total_from_over
    total = total_from_over(np.array([p_over], float))
    sup = supremacy_from_1x2(np.array([p_home], float), np.array([p_away], float), total)
    lam_h, lam_a = 0.5 * (total + sup), 0.5 * (total - sup)
    rho = rho_from_draw(lam_h, lam_a, np.array([p_draw], float))
    return {"total": float(total[0]), "supremacy": float(sup[0]), "rho": float(rho[0]), "p_over": float(p_over),
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away}


def implied_at(mo: pd.DataFrame, ou: pd.DataFrame | None, roles: dict, over_id, t_ms: int) -> dict | None:
    """The implied columns as gold/implied builds them, from the match-odds and over/under ticks as of ``t_ms``.
    ``roles``: selection id -> home, draw or away; ``over_id``: the over 2.5 runner. None without a complete 1X2."""
    b = book_at(mo, t_ms)
    b = b[b["selection_id"].isin(roles.keys())]
    if len(b) != 3 or (b["ltp"] <= 1.0).any():
        return None
    raw = 1.0 / b["ltp"].to_numpy(float)
    p = dict(zip(b["selection_id"].map(roles), raw / raw.sum()))
    out = {"mo_book_sum": float(raw.sum()), "mo_age_s": float(b["age_s"].max())}
    p_over, ou_age, ou_sum = np.nan, np.nan, np.nan
    if ou is not None and len(ou):
        o = book_at(ou, t_ms)
        if len(o) == 2 and (o["ltp"] > 1.0).all() and over_id in set(o["selection_id"]):
            r = 1.0 / o["ltp"].to_numpy(float)
            p_over = float(r[(o["selection_id"] == over_id).to_numpy()][0] / r.sum())
            ou_age, ou_sum = float(o["age_s"].max()), float(r.sum())
    out.update(implied(p["home"], p["draw"], p["away"], p_over))
    out["ou_age_s"], out["ou_book_sum"] = ou_age, ou_sum
    return out


def checkpoint_rows(mo: pd.DataFrame, ou: pd.DataFrame | None, roles: dict, over_id, market_id: str, event_id: int,
                    kickoff_ms: int, cp_min: int, prev_min: int) -> pd.DataFrame | None:
    """One row a match-odds runner at the checkpoint ``cp_min`` minutes before ``kickoff_ms``: level, implied, implied
    changes from ``prev_min`` and the price path from the thinned ticks (``pp_``). Memory is added by the caller."""
    import pyarrow as pa
    from football.warehouse import gold_path as gp
    from football.warehouse.schema import TS
    t_ms = int(kickoff_ms - cp_min * 60_000)
    now, before = implied_at(mo, ou, roles, over_id, t_ms), implied_at(mo, ou, roles, over_id,
                                                                       int(kickoff_ms - prev_min * 60_000))
    if now is None:
        return None
    b = book_at(mo, t_ms)
    b = b[b["selection_id"].isin(roles.keys())].copy()
    b["role"] = b["selection_id"].map(roles)
    p = 1.0 / b["ltp"]
    b["p_cp"] = p / p.sum()
    ko = pd.Timestamp(kickoff_ms, unit="ms", tz="UTC")
    rows = pd.DataFrame({"market_id": market_id, "selection_id": b["selection_id"].astype("int64"),
                         "role": b["role"], "cp_min": cp_min, "event_id": event_id,
                         "x_cp": np.log(b["p_cp"] / (1 - b["p_cp"])), "cp_book_sum": now["mo_book_sum"],
                         "ko_hour": ko.hour + ko.minute / 60.0, "ko_dow": ko.dayofweek, "p_cp": b["p_cp"],
                         "ltp_cp": b["ltp"]})
    for c in ("total", "supremacy", "rho", "p_over", "ou_age_s"):
        rows[f"imp_{c}"] = now[c]
    for c in ("p_home", "p_draw", "p_away", "p_over", "mo_book_sum", "ou_book_sum", "total", "supremacy", "rho"):
        rows[f"idl_d_{c}"] = (now[c] - before[c]) if before is not None else np.nan
    rows["idl_own_dp"] = [rows[f"idl_d_p_{r}"].iloc[0] for r in rows["role"]]
    # the price path, gold_path's own builder on this market's thinned ticks
    tk = mo.assign(handicap=0.0, seq=0, pre_off=True, market_type="MATCH_ODDS", market_id=market_id)
    tk = tk[tk["pt"] <= t_ms]
    tt = pa.Table.from_pandas(tk[gp.COLUMNS["silver/tick"]].assign(pt=tk["pt"].astype("int64")), preserve_index=False)
    tt = tt.set_column(tt.schema.get_field_index("pt"), "pt", tt.column("pt").cast(pa.timestamp("ms", "UTC")).cast(TS))
    cp = pd.DataFrame({"market_id": market_id, "selection_id": rows["selection_id"], "handicap": 0.0,
                       "cp_min": cp_min, "event_id": event_id, "market_type": "MATCH_ODDS", "role": rows["role"],
                       "cp_time": pd.Timestamp(t_ms, unit="ms", tz="UTC")})
    ct = pa.Table.from_pandas(cp, preserve_index=False)
    ct = ct.set_column(ct.schema.get_field_index("cp_time"), "cp_time", ct.column("cp_time").cast(TS))
    path = gp.build({"silver/tick": tt, "gold/checkpoint": ct})[gp.SPEC.name].to_pandas()
    keep = [c for c in path.columns if c not in gp.COLUMNS["gold/checkpoint"]]
    path = path[["selection_id"] + keep].rename(columns={c: f"pp_{c}" for c in keep})
    return rows.merge(path, on="selection_id", how="left").reset_index(drop=True)
