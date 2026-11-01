"""Panels in the standard open-close shape from the data this repository reaches, until the data agent's panel
exists: football-data.co.uk's gold table (football.data), Betfair's price files for UK/IE racing (betfair_prices in
horse_racing.db) and Betfair's Historic Data tables (betfair_historic, the greyhounds).

Each builder returns one row a selection with features.PANEL_COLUMNS (and ``model_price`` where a forecast is
joined). Times are UTC.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .features import PANEL_COLUMNS, coerce_panel

LONDON = ZoneInfo("Europe/London")


# ---------------------------------------------------------------------------
# football: football-data.co.uk's benchmark open and close (bookmakers' lines, a proxy) or Betfair's own
# ---------------------------------------------------------------------------

def football_panel(gold: pd.DataFrame, venue: str = "benchmark") -> pd.DataFrame:
    """Each 1X2 outcome, each O/U 2.5 side and each AH side of a finished match as a selection.

    ``venue`` "benchmark": the gold table's open (Pinnacle or Betfair Exchange) and close (Pinnacle to 2024/25,
    Betfair Exchange from 2025/26), football.data.benchmarks; "bfe": Betfair Exchange's own open and close
    (football-data's BFE/BFEC columns, 1X2 and O/U 2.5 only, the seasons that carry them). The Asian handicap is
    kept only where the closing line is the opening line (a moved line is a different market). Results: 1X2 from
    the score, O/U from the goals, the handicap left blank (the hedged trade needs none). The open here is a
    bookmaker's line (Pinnacle) or Betfair's, football-data's; a proxy for the exchange's early price."""
    g = gold[gold["status"].eq("FINISHED")].copy()
    kick = pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce")
    # football-data carries no kick-off time before 2019/20: the match day at 15:00 UTC stands in (the day and the
    # day of the week are what the features read; hours_to_off is unknown here anyway)
    if "match_date" in g.columns:
        day = pd.to_datetime(g["match_date"], errors="coerce").dt.tz_localize("UTC") + pd.Timedelta(hours=15)
        kick = kick.fillna(day)
    base = pd.DataFrame({"sport": "football", "country": g["country"].astype(str), "event_id": g["match_id"],
                         "event_name": g["home_team"].astype(str) + " v " + g["away_team"].astype(str),
                         "off_time_utc": kick, "open_time_utc": pd.NaT, "open_matched": np.nan,
                         "pre_play_matched": np.nan, "bsp": np.nan, "p_t24h": np.nan, "p_t60": np.nan,
                         "p_t15": np.nan, "p_t1": np.nan, "season": g["season"], "competition_id": g["competition_id"]})
    goals = g["ft_home"].to_numpy(float) + g["ft_away"].to_numpy(float)
    res_1x2 = np.sign(g["ft_home"].to_numpy(float) - g["ft_away"].to_numpy(float))
    parts = []
    if venue == "benchmark":
        src_1x2 = g["open_1x2_src"].astype(str) + ">" + g["close_1x2_src"].astype(str)
        legs = [("MATCH_ODDS", "H", "open_h", "close_h", (res_1x2 == 1).astype(float), src_1x2),
                ("MATCH_ODDS", "D", "open_d", "close_d", (res_1x2 == 0).astype(float), src_1x2),
                ("MATCH_ODDS", "A", "open_a", "close_a", (res_1x2 == -1).astype(float), src_1x2)]
        src_ou = g["open_ou_src"].astype(str) + ">" + g["close_ou_src"].astype(str)
        legs += [("OVER_UNDER_25", "Over 2.5", "open_o25", "close_o25", (goals > 2.5).astype(float), src_ou),
                 ("OVER_UNDER_25", "Under 2.5", "open_u25", "close_u25", (goals < 2.5).astype(float), src_ou)]
        same_line = g["open_ah_line"].notna() & (g["open_ah_line"] == g["close_ah_line"])
        src_ah = g["open_ah_src"].astype(str) + ">" + g["close_ah_src"].astype(str)
        for name, oc, cc in (("Home", "open_ah_h", "close_ah_h"), ("Away", "open_ah_a", "close_ah_a")):
            legs.append(("ASIAN_HANDICAP", name, oc, cc, np.full(len(g), np.nan), src_ah.where(same_line, None)))
    elif venue == "bfe":
        src = pd.Series("BFE>BFEC", index=g.index)
        legs = [("MATCH_ODDS", "H", "bfe_open_h", "bfe_close_h", (res_1x2 == 1).astype(float), src),
                ("MATCH_ODDS", "D", "bfe_open_d", "bfe_close_d", (res_1x2 == 0).astype(float), src),
                ("MATCH_ODDS", "A", "bfe_open_a", "bfe_close_a", (res_1x2 == -1).astype(float), src),
                ("OVER_UNDER_25", "Over 2.5", "bfe_open_o25", "bfe_close_o25", (goals > 2.5).astype(float), src),
                ("OVER_UNDER_25", "Under 2.5", "bfe_open_u25", "bfe_close_u25", (goals < 2.5).astype(float), src)]
    else:
        raise ValueError(f"venue {venue!r}: benchmark or bfe")
    for market_type, sel, oc, cc, result, src in legs:
        if oc not in g.columns or cc not in g.columns:
            continue
        p = base.copy()
        p["market_type"] = market_type
        p["market_id"] = g["match_id"].astype(str) + ":" + market_type
        p["selection_id"] = sel
        p["selection_name"] = sel
        p["open_price"] = pd.to_numeric(g[oc], errors="coerce").to_numpy()
        p["close_price"] = pd.to_numeric(g[cc], errors="coerce").to_numpy()
        p["result"] = result
        p["source"] = src.to_numpy()
        p["n_runners"] = 3 if market_type == "MATCH_ODDS" else 2
        parts.append(p[p["source"].notna()])
    out = pd.concat(parts, ignore_index=True)
    # a market needs every leg priced at both ends, else its book is not a book
    both = out["open_price"].gt(1) & out["close_price"].gt(1)
    whole = both.groupby(out["market_id"]).transform("all")
    out = out[whole].copy()
    return coerce_panel(out)[PANEL_COLUMNS + ["season", "competition_id"]]


# ---------------------------------------------------------------------------
# horses: Betfair's price files, the morning WAP as the open and the BSP as the close
# ---------------------------------------------------------------------------

def horse_off_utc(race_date: pd.Series, race_time: pd.Series) -> pd.Series:
    """race_date 'YYYY-MM-DD' and race_time 'HH:MM' (UK local, Betfair's EVENT_DT) -> UTC."""
    local = pd.to_datetime(race_date.astype(str) + " " + race_time.astype(str), format="%Y-%m-%d %H:%M",
                           errors="coerce")
    return local.dt.tz_localize(LONDON, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def horses_panel(prices: pd.DataFrame, forecasts: pd.DataFrame | None = None) -> pd.DataFrame:
    """betfair_prices rows (country, market_type, race_date, race_time, event_id, event_name, selection_id,
    selection_name, win_lose, bsp, ppwap, morningwap, morning_vol, pp_vol, and race_results_id, track, rr_time and
    horse_name from race_results where matched) -> the panel. ``forecasts``: the model's out-of-sample
    predicted_bfsp keyed by race_date, track, race_time ('1.20.' form) and horse_name, joined as ``model_price``.

    Betfair's morning price is the volume-weighted average of what traded in its morning window (the files give no
    time stamp, so open_time_utc is blank and hours_to_off unknown); the morning volume is open_matched; the
    pre-play WAP stands in for p_t1 (no mark nearer the off exists in these files)."""
    d = prices.copy()
    n = d.groupby("event_id")["bsp"].transform(lambda s: int((pd.to_numeric(s, errors="coerce") > 1).sum()))
    out = pd.DataFrame({
        "sport": "horse", "country": d["country"].astype(str).str.lower(),
        "market_type": d["market_type"].astype(str).str.upper(),
        "market_id": d["event_id"].astype(str) + ":" + d["market_type"].astype(str),
        "event_id": d["event_id"], "event_name": d["event_name"], "selection_id": d["selection_id"],
        "selection_name": d["selection_name"], "off_time_utc": horse_off_utc(d["race_date"], d["race_time"]),
        "n_runners": n, "open_price": pd.to_numeric(d["morningwap"], errors="coerce"), "open_time_utc": pd.NaT,
        "open_matched": pd.to_numeric(d["morning_vol"], errors="coerce"), "p_t24h": np.nan, "p_t60": np.nan,
        "p_t15": np.nan, "p_t1": pd.to_numeric(d.get("ppwap"), errors="coerce"),
        "close_price": pd.to_numeric(d["bsp"], errors="coerce"), "bsp": pd.to_numeric(d["bsp"], errors="coerce"),
        "pre_play_matched": pd.to_numeric(d.get("pp_vol"), errors="coerce"),
        "result": pd.to_numeric(d["win_lose"], errors="coerce"), "source": "betfair_prices:" + d["country"].astype(str),
    })
    if forecasts is not None and {"track", "rr_time", "horse_name"} <= set(d.columns):
        f = forecasts.rename(columns={"race_time": "rr_time"})[["race_date", "track", "rr_time", "horse_name",
                                                                  "predicted_bfsp"]]
        f = f.drop_duplicates(["race_date", "track", "rr_time", "horse_name"])
        key = d[["race_date", "track", "rr_time", "horse_name"]].astype(str)
        j = key.merge(f.astype({"race_date": str, "track": str, "rr_time": str, "horse_name": str}), how="left",
                      on=["race_date", "track", "rr_time", "horse_name"])
        out["model_price"] = pd.to_numeric(j["predicted_bfsp"], errors="coerce").to_numpy()
    return coerce_panel(out)


# ---------------------------------------------------------------------------
# greyhounds: Betfair's Historic Data (betfair_historic.py): first traded price as the open, the last before in play
# ---------------------------------------------------------------------------

def greyhound_panel(bf: pd.DataFrame, market_types=("WIN",), close: str = "ltp_close") -> pd.DataFrame:
    """markets_greyhound_racing_<year>.parquet rows -> the panel: the first traded price as the open (the BASIC
    files carry no time stamp and no volume at the first trade), the close ``close``: "ltp_close" (the last
    pre-play traded price, the data agent's reading) or "bsp" (the position closed at the SP, as the horses'), the
    BSP beside it either way, the marks at T-24h, T-60, T-15 and T-1 minutes, the result from the runner's status."""
    d = bf[bf["market_type"].isin(market_types) & bf["runner_status"].isin(["WINNER", "LOSER", "PLACED"])].copy()
    off = pd.to_datetime(d["market_time"], utc=True, errors="coerce")
    out = pd.DataFrame({
        "sport": "greyhound", "country": d["country"].astype(str).str.lower(), "market_type": d["market_type"],
        "market_id": d["market_id"].astype(str), "event_id": d["event_id"], "event_name": d["event_name"],
        "selection_id": d["selection_id"], "selection_name": d["runner_name"], "off_time_utc": off,
        "n_runners": d.groupby("market_id")["selection_id"].transform("size"),
        "open_price": pd.to_numeric(d["ltp_first"], errors="coerce"), "open_time_utc": pd.NaT, "open_matched": np.nan,
        "p_t24h": pd.to_numeric(d.get("ltp_t_1440"), errors="coerce"), "p_t60": pd.to_numeric(d.get("ltp_t_60"), errors="coerce"),
        "p_t15": pd.to_numeric(d.get("ltp_t_15"), errors="coerce"), "p_t1": pd.to_numeric(d.get("ltp_t_1"), errors="coerce"),
        "close_price": pd.to_numeric(d[close], errors="coerce"), "bsp": pd.to_numeric(d["bsp"], errors="coerce"),
        "pre_play_matched": np.nan, "result": pd.to_numeric(d["won"], errors="coerce"),
        "source": f"betfair_historic:first>{close}",
    })
    return coerce_panel(out)
