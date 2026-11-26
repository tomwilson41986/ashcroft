"""Team news for the upcoming matches: who is out, who plays instead, and what it does to the win probabilities.

racing² football, 26 Nov 2026 (the owner: "we just want injuries for the upcoming match and how that impacts team
selection and win probabilities"). For each match of a UK day that API-Football lists injuries for:

1. **The expected eleven** of each side: every player's share of the side's starts in its last ``N_PREV`` matches
   (``api_football/history_players``, with the collector's confirmed elevens, ``api_football/lineups``), with his
   position (G, D, M, F).
2. **The absentees**: API-Football's injuries and suspensions for the fixture (``/injuries?date=<day>``, collected
   every 3 hours by football-trade.yml), ``Missing Fixture`` counted as out, ``Questionable`` as half out.
3. **The replacement**: the same-position players of the squad (seen in its last ``N_SQUAD`` matches, not out
   themselves, not already regulars), weighted by their appearances; none known, the average player (rating 0).
4. **Strength lost**: for each absentee, his share of starts times (his rating less the replacement's), summed.
   Ratings are goals plus-minus (``football.player_ratings.fit_gpm``: goal difference a 90 minutes added, fitted on
   matches before this month), the rating whose line-up surprise predicted the market's late move (report, 8 Nov).
5. **Win probabilities**: the market's (the football feed's latest pre-off Betfair book, mids normalised; the over
   2.5 market where recorded), turned into goal rates (``team_features.implied_rates``). The side's lost goal
   difference is split evenly between its attack and the opponent's (each rate moved by half). Reported: the market's
   probabilities as they are, and with each side at full strength (its losses added back), so the gap is what the
   absences are worth in the model's terms.

The market already knows most announced injuries, so "full strength" is a counterfactual, not a value call: this
is information for team selection and the price, not a trading signal. Nothing is traded.

    python -m football.team_news --day 2026-11-26 --out out/team_news.json --md out/team_news.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

N_PREV = 5                      # matches the expected eleven is read from
N_SQUAD = 10                    # matches a replacement may come from
REGULAR = 0.6                   # share of starts that makes a regular (not a replacement)
OUT_WEIGHT = {"Missing Fixture": 1.0, "Questionable": 0.5}
PREFIX = "api_football"


# ---------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------

def _docs(store, kind: str, day: str) -> list[dict]:
    out = []
    for k in sorted(store.listing(f"{PREFIX}/raw/{kind}/{day}/")):
        if k.endswith(".json.gz"):
            try:
                out.append(json.loads(gzip.decompress(store.get(k))))
            except (OSError, ValueError):
                log.warning("%s unreadable", k)
    return out


def fixtures(docs: list[dict]) -> pd.DataFrame:
    """The day's fixtures from the latest /fixtures?date reply: id, kick-off, league, the two sides."""
    rows = {}
    for d in docs:                                         # oldest first: the latest reply wins
        for f in (d.get("body") or {}).get("response") or []:
            fx, lg, tm = f.get("fixture") or {}, f.get("league") or {}, f.get("teams") or {}
            if fx.get("id") is None or not fx.get("timestamp"):
                continue
            rows[fx["id"]] = {"apif_fixture_id": fx["id"], "kickoff_utc": pd.Timestamp(fx["timestamp"], unit="s", tz="UTC"),
                              "status": (fx.get("status") or {}).get("short"), "apif_league_id": lg.get("id"),
                              "league": lg.get("name"), "country": lg.get("country"),
                              "home_id": (tm.get("home") or {}).get("id"), "home": (tm.get("home") or {}).get("name"),
                              "away_id": (tm.get("away") or {}).get("id"), "away": (tm.get("away") or {}).get("name")}
    return pd.DataFrame(list(rows.values()))


def injuries(docs: list[dict]) -> pd.DataFrame:
    """One row a (fixture, player): the latest type and reason, the team, when first seen."""
    rows = {}
    for d in docs:
        t = d.get("fetched_utc")
        for r in (d.get("body") or {}).get("response") or []:
            pl, fx, tm = r.get("player") or {}, r.get("fixture") or {}, r.get("team") or {}
            if pl.get("id") is None or fx.get("id") is None:
                continue
            k = (fx["id"], pl["id"])
            first = rows[k]["first_seen_utc"] if k in rows else t
            rows[k] = {"apif_fixture_id": fx["id"], "apif_team_id": tm.get("id"), "apif_player_id": pl["id"],
                       "player": pl.get("name"), "type": pl.get("type"), "reason": pl.get("reason"),
                       "first_seen_utc": first, "last_seen_utc": t}
    return pd.DataFrame(list(rows.values()))


def appearances(hp: pd.DataFrame, hdates: pd.DataFrame, lineups: pd.DataFrame | None) -> pd.DataFrame:
    """Every player a match a side: fixture, team, date, player, position, starter or not (history + collector)."""
    cols = ["apif_fixture_id", "apif_team_id", "apif_player_id", "player", "pos", "role"]
    a = hp[[c for c in cols if c in hp.columns]].copy()
    if lineups is not None and len(lineups):
        lu = lineups.copy()
        if "apif_team_id" not in lu.columns:
            lu["apif_team_id"] = np.nan
        a = pd.concat([a, lu[[c for c in cols if c in lu.columns]]], ignore_index=True)
    a = a[a["apif_player_id"].notna() & a["apif_team_id"].notna()]
    a = a.drop_duplicates(["apif_fixture_id", "apif_player_id"], keep="last")
    a = a.merge(hdates[["apif_fixture_id", "date"]], on="apif_fixture_id", how="inner")
    a["apif_team_id"] = a["apif_team_id"].astype("int64")
    a["apif_player_id"] = a["apif_player_id"].astype("int64")
    return a


# ---------------------------------------------------------------------------------------------------------------
# the eleven, the absentees, the replacements
# ---------------------------------------------------------------------------------------------------------------

def side_news(app: pd.DataFrame, team_id: int, before: pd.Timestamp, out: pd.DataFrame, ratings: dict) -> dict:
    """One side: its expected eleven, each absentee with his replacement, and the goal difference a 90 lost."""
    g = app[(app["apif_team_id"] == int(team_id)) & (app["date"] < before)]
    fids = g.drop_duplicates("apif_fixture_id").sort_values("date")["apif_fixture_id"]
    last, squad = fids.tail(N_PREV), fids.tail(N_SQUAD)
    res = {"matches_read": int(len(last)), "last_match": str(g["date"].max().date()) if len(g) else None,
           "absentees": [], "lost_gd90": 0.0, "expected_xi_rating": None}
    if len(last) < 3:
        res["note"] = "fewer than 3 earlier matches with line-ups: no data found"
        return res
    st = g[g["apif_fixture_id"].isin(last) & (g["role"] == "starter")]
    share = st.groupby("apif_player_id").size() / len(last)
    pos = g.sort_values("date").groupby("apif_player_id")["pos"].last()
    name = g.sort_values("date").groupby("apif_player_id")["player"].last()
    res["expected_xi_rating"] = round(float(sum(ratings.get(p, 0.0) * w for p, w in share.items())), 4)
    w_out = {int(r.apif_player_id): OUT_WEIGHT.get(str(r.type), 1.0) for r in out.itertuples(index=False)}
    sq = g[g["apif_fixture_id"].isin(squad)]
    apps = sq.groupby("apif_player_id").size()
    lost = 0.0
    for r in out.itertuples(index=False):
        p = int(r.apif_player_id)
        s = float(share.get(p, 0.0))
        cand = [q for q in apps.index if q != p and pos.get(q) == pos.get(p) and w_out.get(int(q), 0.0) < 1.0
                and share.get(q, 0.0) < REGULAR]
        if cand:
            wt = apps[cand].astype(float)
            repl = float(np.average([ratings.get(q, 0.0) for q in cand], weights=wt))
            repl_names = [str(name.get(q)) for q in wt.sort_values(ascending=False).index[:2]]
        else:
            repl, repl_names = 0.0, []
        d = w_out[p] * s * (ratings.get(p, 0.0) - repl)
        lost += d
        res["absentees"].append({"player": r.player, "type": r.type, "reason": r.reason, "position": pos.get(p),
                                 "start_share": round(s, 2), "rating": round(float(ratings.get(p, 0.0)), 4),
                                 "rated": p in ratings, "replacement": repl_names,
                                 "replacement_rating": round(repl, 4), "gd90_lost": round(float(d), 4)})
    res["absentees"].sort(key=lambda x: -abs(x["gd90_lost"]))
    res["lost_gd90"] = round(float(lost), 4)
    return res


# ---------------------------------------------------------------------------------------------------------------
# win probabilities
# ---------------------------------------------------------------------------------------------------------------

def probs(lh: float, la: float) -> dict:
    from football.ratings import markets, score_matrix
    m = markets(score_matrix(max(lh, 0.05), max(la, 0.05), 0.0))
    return {k: round(m[k], 4) for k in ("p_h", "p_d", "p_a")}


def with_strength(p_h: float, p_a: float, p_over: float | None, lost_home: float, lost_away: float) -> dict:
    """The market's chances, its goal rates, and the chances with each side's losses added back."""
    from football.team_features import implied_rates
    lh, la = implied_rates([p_h], [p_a], None if p_over is None else [p_over])
    lh, la = float(lh[0]), float(la[0])
    base = probs(lh, la)
    full = probs(lh + 0.5 * lost_home - 0.5 * lost_away, la + 0.5 * lost_away - 0.5 * lost_home)
    home_full = probs(lh + 0.5 * lost_home, la - 0.5 * lost_home)
    away_full = probs(lh - 0.5 * lost_away, la + 0.5 * lost_away)
    return {"rates": {"home": round(lh, 3), "away": round(la, 3)}, "model_at_market_rates": base,
            "both_full_strength": full, "home_full_strength": home_full, "away_full_strength": away_full,
            "home_win_pp_lost_to_absences": round(100 * (home_full["p_h"] - base["p_h"]), 2),
            "away_win_pp_lost_to_absences": round(100 * (away_full["p_a"] - base["p_a"]), 2)}


def market_now(store, day: str, fx: pd.DataFrame) -> pd.DataFrame:
    """The latest pre-off Betfair match-odds mid of each fixture (and over 2.5 where recorded), linked by names."""
    from research.open_close.football_feed_coverage import link
    from research.open_close.football_lineup_release import books_for, price_series
    m, b = books_for(store, [day])
    if m.empty:
        return pd.DataFrame()
    mk = m.drop_duplicates("market_id").copy()
    mk["kickoff"] = pd.to_datetime(mk["market_start_utc"], utc=True, errors="coerce")
    parts = mk["event_name"].astype(str).str.split(" v ", n=1, regex=False)
    mk["ev_home"], mk["ev_away"] = parts.str[0].str.strip(), parts.str[1].str.strip()
    j = link(mk.dropna(subset=["kickoff"]), fx.assign(apif_season=0, league_country=fx.get("country")))
    j = j.dropna(subset=["apif_fixture_id"])
    ps = price_series(m, b)
    ps = ps[ps["inplay"] == 0]
    last = ps.groupby("market_id").tail(1)
    out = j[["market_id", "apif_fixture_id", "event_name"]].merge(last, on="market_id")
    inv = 1 / out[["mid_h", "mid_d", "mid_a"]]
    q = inv.div(inv.sum(axis=1), axis=0)
    out["p_h"], out["p_d"], out["p_a"] = q["mid_h"], q["mid_d"], q["mid_a"]
    out["book_utc"] = out["t"].astype(str)
    out["p_over"] = np.nan
    return out[["apif_fixture_id", "market_id", "event_name", "p_h", "p_d", "p_a", "p_over", "book_utc"]]


# ---------------------------------------------------------------------------------------------------------------
# the day
# ---------------------------------------------------------------------------------------------------------------

def run(store, day: str) -> dict:
    from research.open_close.football_lineup_release import gpm_ratings
    fx = fixtures(_docs(store, "day_fixtures", day))
    inj = injuries(_docs(store, "injuries", day))
    res = {"day": day, "fixtures_listed": int(len(fx)), "injury_rows": int(len(inj))}
    if fx.empty or inj.empty:
        res["status"] = "no data found (API-Football fixtures or injuries for the day)"
        return res
    fx = fx[fx["apif_fixture_id"].isin(inj["apif_fixture_id"])]
    res["fixtures_with_injuries"] = int(len(fx))
    ratings, hp, hdates = gpm_ratings(store, pd.Timestamp(day).replace(day=1))
    res["players_rated"] = len(ratings)
    try:
        lu = store.get_parquet(f"{PREFIX}/lineups.parquet")
    except Exception:                                          # the collector's table may not be built yet
        lu = None
    app = appearances(hp, hdates, lu)
    res["history_last_match"] = str(app["date"].max().date()) if len(app) else None
    try:
        mkt = market_now(store, day, fx)
    except Exception as exc:                                   # the price is optional: the strength stands alone
        log.warning("market not read (%s)", exc)
        mkt = pd.DataFrame()
    res["with_betfair_price"] = int(mkt["apif_fixture_id"].nunique()) if len(mkt) else 0
    matches = []
    for r in fx.sort_values("kickoff_utc").itertuples(index=False):
        ko = r.kickoff_utc.tz_convert(None)
        sides = {}
        for side, tid in (("home", r.home_id), ("away", r.away_id)):
            out = inj[(inj["apif_fixture_id"] == r.apif_fixture_id) & (inj["apif_team_id"] == tid)]
            sides[side] = side_news(app, tid, ko, out, ratings)
        rec = {"apif_fixture_id": int(r.apif_fixture_id), "kickoff_utc": str(r.kickoff_utc), "league": r.league,
               "country": r.country, "home_team": r.home, "away_team": r.away, "home": sides["home"],
               "away": sides["away"]}
        p = mkt[mkt["apif_fixture_id"] == r.apif_fixture_id] if len(mkt) else mkt
        if len(p):
            q = p.iloc[0]
            rec["betfair"] = {"market_id": q["market_id"], "book_utc": q["book_utc"],
                              "p_h": round(float(q["p_h"]), 4), "p_d": round(float(q["p_d"]), 4),
                              "p_a": round(float(q["p_a"]), 4)}
            rec["impact"] = with_strength(float(q["p_h"]), float(q["p_a"]),
                                          None if not np.isfinite(q["p_over"]) else float(q["p_over"]),
                                          sides["home"]["lost_gd90"], sides["away"]["lost_gd90"])
        else:
            rec["betfair"] = "no data found (no recorded book linked)"
        matches.append(rec)
    res["matches"] = matches
    return res


def _who(side: dict, n: int = 4) -> str:
    """The absentees who cost something, each with his likely replacement."""
    out = []
    for x in side.get("absentees", []):
        if x["start_share"] <= 0:
            continue
        tag = "?" if x["type"] == "Questionable" else ""
        repl = f" -> {x['replacement'][0]}" if x["replacement"] else " -> (none known)"
        out.append(f"{x['player']}{tag} ({x['position']}, {x['start_share']:.0%} starts){repl}")
    return "; ".join(out[:n]) or "none who start"


def markdown(res: dict) -> str:
    lines = [f"# racing² football - team news, {res['day']}", "",
             f"Fixtures with injuries: {res.get('fixtures_with_injuries', 0)}; with a Betfair price: "
             f"{res.get('with_betfair_price', 0)}; players rated: {res.get('players_rated', 0)}; line-up history to "
             f"{res.get('history_last_match')}.", "",
             "Source: API-Football injuries and line-ups, goals plus-minus ratings (goal difference a 90 minutes "
             "added), the football feed's Betfair books (`football/team_news.py`). `?` = doubtful (counted half). "
             "Win points are the model's: the market's goal rates with the side's losses added back. The market "
             "already prices most announced injuries, so this sizes the absences; it is not a value call.", ""]
    if "matches" not in res:
        return "\n".join(lines + [res.get("status", "no data found")]) + "\n"
    lines += ["| Kick-off (UTC) | Match | Home out -> in | Away out -> in | GD/90 lost H / A | Betfair H/D/A | "
              "Win pts lost H / A |", "|---|---|---|---|---|---|---|"]
    for m in res["matches"]:
        h, a = m["home"], m["away"]
        lh = h["lost_gd90"] if h.get("matches_read", 0) >= 3 else "no data found"
        la = a["lost_gd90"] if a.get("matches_read", 0) >= 3 else "no data found"
        bf = m["betfair"]
        bfs = f"{bf['p_h']:.2f} / {bf['p_d']:.2f} / {bf['p_a']:.2f}" if isinstance(bf, dict) else "no data found"
        imp = m.get("impact") or {}
        lines.append(f"| {m['kickoff_utc'][11:16]} | {m['home_team']} v {m['away_team']} ({m['league']}) | "
                     f"{_who(h)} | {_who(a)} | {lh} / {la} | {bfs} | "
                     f"{imp.get('home_win_pp_lost_to_absences', '-')} / {imp.get('away_win_pp_lost_to_absences', '-')} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", default=date.today().isoformat())
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    res = run(Store(root=a.root) if a.root else Store(), a.day)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    if a.md:
        Path(a.md).write_text(markdown(res))
    print(json.dumps({k: v for k, v in res.items() if k != "matches"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
