"""How much of our Betfair football does the API-Football feed cover, and with what data?

racing² football research, 6 Nov 2026 (the owner: "we only want data that is a regular feed, gets updated and has
broad coverage across our matches"). API-Football (the Pro plan we hold) is a regular, daily-updated feed; the
question is its breadth over the matches we trade and record.

1. **Our matches:** the match-odds markets of the live Betfair record over the last ``days`` days (with the money
   matched on each, from the recorded books), and the owner's Historic Data bundle (30 Jul - 30 Sep 2026).
2. **Linked:** each market to an API-Football fixture kicking off within 30 minutes whose two sides match
   (``football.reference.pair_event``).
3. **What the feed gives for it:** the coverage flags API-Football publishes for each league and season (``/leagues``:
   events, line-ups, team statistics, player statistics, injuries, predictions), and whether its team statistics
   carry expected goals: one finished fixture a league is read (``/fixtures?ids=``, 20 a request) to see.
4. **What we hold today:** the leagues our history tables already cover (the 22 main divisions).

Read-only on S3; at most ``1 + MAX_XG_CALLS`` API-Football requests (the plan allows 7,500 a day).

    API_FOOTBALL_KEY=... python -m research.open_close.football_feed_coverage --days 7 --out out/feed_coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://v3.football.api-sports.io"
LINK_MINUTES = 30
MAX_XG_CALLS = 25
FLAGS = ["events", "lineups", "statistics_fixtures", "statistics_players", "injuries", "predictions"]


def live_markets(store, days: int, today: date | None = None) -> pd.DataFrame:
    """Match-odds markets of the live record, one row a market, with the most money matched seen on it."""
    from football.exchange_feed import read_day
    today = today or date.today()
    rows = []
    for i in range(days):
        d = (today - timedelta(days=i)).isoformat()
        mk = read_day(store, d, "markets")
        if mk is None or mk.empty:
            continue
        mk = mk[mk["market_type"] == "MATCH_ODDS"].drop_duplicates("market_id")
        bk = read_day(store, d, "books")
        if bk is not None and not bk.empty and "market_total_matched" in bk:
            vol = bk.groupby("market_id")["market_total_matched"].max()
            mk = mk.assign(matched=mk["market_id"].map(vol))
        rows.append(mk.assign(day=d))
    if not rows:
        return pd.DataFrame()
    m = pd.concat(rows, ignore_index=True)
    m["kickoff"] = pd.to_datetime(m["market_start_utc"], utc=True, errors="coerce")
    parts = m["event_name"].astype(str).str.split(" v ", n=1, regex=False)
    m["ev_home"], m["ev_away"] = parts.str[0].str.strip(), parts.str[1].str.strip()
    return m.dropna(subset=["kickoff", "ev_away"])


def bundle_markets(store) -> pd.DataFrame:
    """Match-odds markets of the Historic Data tables, one row a market."""
    keys = sorted(k for k in store.listing("betfair_historic/") if "markets_soccer" in k and k.endswith(".parquet"))
    frames = []
    for k in keys:
        f = store.get_parquet(k)
        if f is not None and len(f):
            frames.append(f[f["market_type"] == "MATCH_ODDS"].drop_duplicates("market_id")[
                ["market_id", "event_name", "market_time", "country"]])
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    m["kickoff"] = pd.to_datetime(m["market_time"], utc=True, errors="coerce")
    parts = m["event_name"].astype(str).str.split(" v ", n=1, regex=False)
    m["ev_home"], m["ev_away"] = parts.str[0].str.strip(), parts.str[1].str.strip()
    return m.dropna(subset=["kickoff", "ev_away"])


def link(m: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Each market -> the fixture within LINK_MINUTES whose two sides both match; unique or nothing."""
    from football.reference import pair_event
    f = fx.copy()
    f["kick"] = pd.to_datetime(f["kickoff_utc"], utc=True, errors="coerce")
    f = f.dropna(subset=["kick"])
    f = f[(f["kick"] >= m["kickoff"].min() - pd.Timedelta(days=1)) & (f["kick"] <= m["kickoff"].max()
                                                                      + pd.Timedelta(days=1))]
    f = f.sort_values("kick").reset_index(drop=True)
    ks = f["kick"].dt.tz_convert(None).to_numpy()
    tol = np.timedelta64(LINK_MINUTES, "m")
    out = []
    for r in m.itertuples(index=False):
        t = np.datetime64(r.kickoff.tz_convert("UTC").tz_localize(None))
        lo, hi = np.searchsorted(ks, t - tol), np.searchsorted(ks, t + tol, side="right")
        cand = f.iloc[lo:hi]
        if cand.empty:
            out.append(None)
            continue
        opts = [(i, h, a) for i, h, a in zip(cand.index, cand["home"].astype(str), cand["away"].astype(str))]
        j, _ = pair_event(r.ev_home, r.ev_away, opts, "", "betfair")
        out.append(opts[j][0] if j is not None else None)
    m = m.copy()
    m["fx_row"] = out
    cols = ["apif_fixture_id", "apif_league_id", "league", "league_country", "apif_season", "status"]
    j = m.join(f[cols], on="fx_row")
    return j


def league_coverage(key: str, http=None) -> pd.DataFrame:
    """API-Football's coverage flags, one row a league and season."""
    import requests
    http = http or requests.Session()
    r = http.get(f"{API}/leagues", headers={"x-apisports-key": key}, timeout=60)
    r.raise_for_status()
    rows = []
    for item in r.json().get("response", []):
        lg = item.get("league") or {}
        for s in item.get("seasons") or []:
            cov = s.get("coverage") or {}
            fxc = cov.get("fixtures") or {}
            rows.append({"apif_league_id": lg.get("id"), "apif_season": s.get("year"),
                         "events": bool(fxc.get("events")), "lineups": bool(fxc.get("lineups")),
                         "statistics_fixtures": bool(fxc.get("statistics_fixtures")),
                         "statistics_players": bool(fxc.get("statistics_players")),
                         "injuries": bool(cov.get("injuries")), "predictions": bool(cov.get("predictions"))})
    return pd.DataFrame(rows)


def xg_sample(key: str, fixture_ids: dict[int, int], max_calls: int = MAX_XG_CALLS, http=None) -> dict[int, bool]:
    """league -> whether a finished fixture's team statistics carry expected goals (one fixture a league)."""
    import requests
    http = http or requests.Session()
    out = {}
    items = list(fixture_ids.items())
    for i in range(0, min(len(items), max_calls * 20), 20):
        chunk = items[i:i + 20]
        r = http.get(f"{API}/fixtures", params={"ids": "-".join(str(f) for _, f in chunk)},
                     headers={"x-apisports-key": key}, timeout=60)
        if r.status_code != 200:
            break
        by_fx = {}
        for f in r.json().get("response", []):
            fid = (f.get("fixture") or {}).get("id")
            has = False
            for side in f.get("statistics") or []:
                for st in side.get("statistics") or []:
                    if st.get("type") == "expected_goals" and st.get("value") not in (None, "", "0", 0):
                        has = True
            by_fx[fid] = has
        for lg, fid in chunk:
            out[lg] = by_fx.get(fid, False)
    return out


def summarise(j: pd.DataFrame, cov: pd.DataFrame | None, xg: dict[int, bool] | None, held: set[int]) -> dict:
    w = j["matched"].fillna(0) if "matched" in j else pd.Series(0.0, index=j.index)
    tot_n, tot_w = len(j), float(w.sum())
    linked = j["apif_fixture_id"].notna()
    share = lambda mask: {"matches": round(float(mask.mean()), 4) if tot_n else None,          # noqa: E731
                          "matched_money": round(float(w[mask].sum() / tot_w), 4) if tot_w else None}
    out = {"markets": tot_n, "matched_money_gbp": round(tot_w, 0), "linked_to_api_football": share(linked),
           "in_the_leagues_we_hold_today": share(j["apif_league_id"].isin(held))}
    if cov is not None and len(cov):
        jj = j.merge(cov, on=["apif_league_id", "apif_season"], how="left")
        out["feed_offers"] = {fl: share(jj[fl].fillna(False).astype(bool)) for fl in FLAGS}
    if xg:
        out["expected_goals_in_team_statistics"] = share(j["apif_league_id"].map(xg).fillna(False).astype(bool))
        out["leagues_sampled_for_xg"] = len(xg)
    top = (j.assign(w=w, linked=linked).groupby(j["league_country"].fillna("?") + " | " + j["league"].fillna("unlinked"))
           .agg(markets=("market_id", "size"), matched=("w", "sum")).sort_values(["matched", "markets"], ascending=False))
    out["top_competitions"] = {k: {"markets": int(r.markets), "matched_gbp": round(float(r.matched), 0)}
                               for k, r in top.head(30).iterrows()}
    unl = j[~linked]
    if len(unl):
        lab = (unl["country"].fillna("?").astype(str) + " | " + unl.get("competition", pd.Series("", index=unl.index))
               .fillna("").astype(str))
        out["unlinked_top"] = lab.value_counts().head(15).to_dict()
    return out


def leagues_to_hold(joined: list[pd.DataFrame], cov: pd.DataFrame | None, fx: pd.DataFrame) -> list[dict]:
    """Every league our Betfair matches were linked into whose latest season the feed covers with team statistics or
    line-ups: id, name, country, our markets and money, the flags, and its finished fixtures a season (the quota a
    season's backfill needs is about that / 20 + 1 requests)."""
    j = pd.concat(joined, ignore_index=True)
    j = j[j["apif_league_id"].notna()]
    if j.empty:
        return []
    j["apif_league_id"] = j["apif_league_id"].astype(int)
    w = j["matched"].fillna(0) if "matched" in j else pd.Series(0.0, index=j.index)
    g = j.assign(w=w).groupby("apif_league_id").agg(markets=("market_id", "size"), matched=("w", "sum"),
                                                    league=("league", "first"), country=("league_country", "first"))
    out = []
    latest = cov.sort_values("apif_season").groupby("apif_league_id").last() if cov is not None and len(cov) else None
    fin = fx[fx["status"] == "FINISHED"].groupby(["apif_league_id", "apif_season"]).size()
    for lg, r in g.sort_values(["matched", "markets"], ascending=False).iterrows():
        flags = latest.loc[lg, FLAGS].to_dict() if latest is not None and lg in latest.index else {}
        if not (flags.get("statistics_fixtures") or flags.get("lineups")):
            continue
        per = fin.loc[lg] if lg in fin.index.get_level_values(0) else pd.Series(dtype=float)
        out.append({"apif_league_id": int(lg), "league": r.league, "country": r.country, "markets": int(r.markets),
                    "matched_gbp": round(float(r.matched), 0), **{k: bool(v) for k, v in flags.items()},
                    "finished_per_season": {int(k): int(v) for k, v in per.tail(3).items()}})
    return out


def run(store, days: int, key: str | None) -> dict:
    from sources.api_football_lineups import main_league_ids
    fx = store.get_parquet("api_football/fixtures.parquet")
    if fx is None or not len(fx):
        return {"status": "no data found (api_football/fixtures.parquet)"}
    cov = league_coverage(key) if key else None
    held = main_league_ids()
    out = {"api_football_key": bool(key), "leagues_held_today": len(held)}
    joined = []
    for name, m in (("live_record", live_markets(store, days)), ("historic_bundle", bundle_markets(store))):
        if m.empty:
            out[name] = {"status": "no data found"}
            continue
        j = link(m, fx)
        joined.append(j)
        xg = None
        if key and name == "live_record":
            fin = fx[fx["status"] == "FINISHED"].sort_values("kickoff_utc")
            lg_ids = j["apif_league_id"].dropna().astype(int)
            order = lg_ids.value_counts().index
            last = fin.groupby("apif_league_id")["apif_fixture_id"].last()
            pick = {int(lg): int(last[lg]) for lg in order if lg in last.index}
            xg = xg_sample(key, pick)
        out[name] = summarise(j, cov, xg, held) | {"days": sorted(m["day"].unique()) if "day" in m else
                                                     [str(m["kickoff"].min()), str(m["kickoff"].max())]}
    if joined:
        lt = leagues_to_hold(joined, cov, fx)
        out["leagues_to_hold"] = {"count": len(lt), "beyond_the_22": sum(1 for x in lt if x["apif_league_id"]
                                                                          not in held),
                                  "requests_per_season_estimate": int(sum(
                                      (max(x["finished_per_season"].values()) if x["finished_per_season"] else 0) / 20
                                      + 1 for x in lt)),
                                  "leagues": lt}
    return out


def main(argv=None) -> int:
    from sources.common import Store, env
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Store(root=a.root), a.days, env("API_FOOTBALL_KEY"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
