"""What we hold for cross-league strength: cup and European matches, and the Betfair record of them.

racing² football research, 6 Nov 2026 (the owner: "record cup markets, and games in Europe"; "work on cross-league
strength, divisions in country for cups, leagues across countries for Europe"). Read-only. Reports:

- **API-Football fixtures** (``api_football/fixtures.parquet``, every competition since 2016): finished matches by
  kind (domestic league, domestic cup, UEFA club competition, other) and season, and the links that tie the pools
  together: cup ties between clubs of different divisions of one country, and UEFA ties between clubs of different
  countries (a strength rating can only compare pools the graph connects).
- **The Betfair record** (``football/exchange/<match day>/markets.csv``, the live-key feed): markets by kind and by
  market type over the last ``days`` days, so the cup and European records can be seen.

    python -m research.open_close.football_cross_league_data --days 3 --out out/cross_league_data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

UEFA_WORDS = ("UEFA", "Champions League", "Europa League", "Conference League", "Super Cup")
CUP_WORDS = ("Cup", "Copa", "Coppa", "Coupe", "Pokal", "Beker", "Taça", "Taca", "Kupa", "Kupası", "Trophy", "Shield")


def kind(league: str, country: str) -> str:
    """'uefa', 'cup' (a domestic cup), 'league' or 'other' from the competition's name and country."""
    name = str(league or "")
    if any(w in name for w in UEFA_WORDS) or str(country) in ("World", "Europe") and "League" in name:
        return "uefa" if any(w in name for w in UEFA_WORDS) else "other"
    if any(w in name for w in CUP_WORDS):
        return "cup"
    return "league" if str(country) not in ("World", "Europe") else "other"


def fixtures_report(fx: pd.DataFrame) -> dict:
    f = fx[fx["status"] == "FINISHED"].copy()
    f["kind"] = [kind(lg, c) for lg, c in zip(f["league"], f["league_country"])]
    f["season"] = f["apif_season"]
    out = {"finished": int(len(f)), "by_kind": f["kind"].value_counts().to_dict(),
           "by_kind_season": {k: g.groupby("season").size().to_dict() for k, g in f.groupby("kind")},
           "top_competitions": {k: g["league"].value_counts().head(25).to_dict() for k, g in f.groupby("kind")}}
    # each club's home league a season: the domestic league it played most in
    lg = f[f["kind"] == "league"]
    sides = pd.concat([lg[["season", "apif_home_id", "league", "league_country"]].rename(columns={"apif_home_id": "team"}),
                       lg[["season", "apif_away_id", "league", "league_country"]].rename(columns={"apif_away_id": "team"})])
    home = sides.groupby(["team", "season"]).agg(home_league=("league", lambda x: x.mode().iloc[0]),
                                                  home_country=("league_country", "first"))
    links = {}
    for k in ("cup", "uefa"):
        t = f[f["kind"] == k][["season", "apif_home_id", "apif_away_id"]]
        a = t.join(home, on=["apif_home_id", "season"]).join(home, on=["apif_away_id", "season"], rsuffix="_a")
        known = a["home_league"].notna() & a["home_league_a"].notna()
        cross_div = known & (a["home_league"] != a["home_league_a"]) & (a["home_country"] == a["home_country_a"])
        cross_cty = known & (a["home_country"] != a["home_country_a"])
        links[k] = {"ties": int(len(a)), "both_clubs_placed": int(known.sum()),
                    "across_divisions_one_country": int(cross_div.sum()), "across_countries": int(cross_cty.sum())}
    out["links"] = links
    return out


def record_report(store, days: int, today: date | None = None) -> dict:
    from football.exchange_feed import read_day
    today = today or date.today()
    rows = []
    for i in range(days):
        d = (today - timedelta(days=i)).isoformat()
        mk = read_day(store, d, "markets")
        if mk is None or mk.empty:
            continue
        m = mk.drop_duplicates("market_id").assign(day=d)
        m["kind"] = [kind(c, k) for c, k in zip(m["competition"], m["country"].fillna(""))]
        rows.append(m)
    if not rows:
        return {"status": "no football exchange record found"}
    m = pd.concat(rows, ignore_index=True)
    return {"markets": int(m["market_id"].nunique()), "days": sorted(m["day"].unique().tolist()),
            "by_kind": m["kind"].value_counts().to_dict(), "by_market_type": m["market_type"].value_counts().to_dict(),
            "uefa_and_cup_competitions": m[m["kind"].isin(["uefa", "cup"])]["competition"].value_counts().head(30)
            .to_dict()}


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    fx = store.get_parquet("api_football/fixtures.parquet")
    res = {"fixtures": fixtures_report(fx) if fx is not None and len(fx) else "no data found",
           "betfair_record": record_report(store, a.days)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
