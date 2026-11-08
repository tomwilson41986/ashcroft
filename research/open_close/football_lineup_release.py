"""Line-up release at the recorded books: how far does the price move from the moment the elevens appear, and can a
trader take it?

racing² football research, 8 Nov 2026 (the closing-line model, item 1). Betfair's minute marks show line-up news
predicts the move from 60 minutes to the close (``football_lineup_late_move``: slope 0.10, +1.4% at traded prices at
60 minutes, short of the spread). The elevens appear 60-75 minutes before the off; this measures the move from the
release itself, at the books the feed records (football/exchange/<day>/books*), with their real spreads:

- the release: ``api_football/lineup_times.parquet`` (the first poll holding both elevens, ``first_seen_utc``);
- the surprise: the confirmed elevens (``api_football/lineups.parquet``) against the expected ones (each player's
  share of his side's starts in its previous five matches, ``api_football/history_players``), each player rated by
  goals plus-minus (``football.player_ratings.fit_gpm``) fitted on matches before the release's month; home less away;
- the price: match odds' mids normalised to one, as the home side's log chance ratio, read from the last book before
  the release (``pre``), the first book at least ``DELAY_MIN`` minutes after it (``entry``: when a trader could act)
  and the last open book before the off (``close``).

Reported: matches; the correlation of the surprise with the move pre->close and entry->close; and the trade 'back the
side the surprise favours at entry when |surprise| > 0.1 / 0.2, lay it back at the close', at the books' own best
prices both ends (the spread paid), after 2% commission on a market's net win.

    python -m research.open_close.football_lineup_release --days 14 --out out/lineup_release.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DELAY_MIN = 2.0
THRESH = (0.1, 0.2)
COMMISSION = 0.02
N_PREV = 5


class Overlay:
    """Reads the store; keeps writes in memory (the line-up tables rebuilt here never touch S3)."""

    def __init__(self, store):
        self.store, self.mem = store, {}

    def listing(self, prefix):
        return self.store.listing(prefix)

    def get(self, key):
        return self.mem[key] if key in self.mem else self.store.get(key)

    def put(self, key, data):
        self.mem[key] = data

    def put_parquet(self, key, df):
        import io
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.mem[key] = buf.getvalue()

    def get_parquet(self, key):
        import io
        raw = self.get(key)
        return None if raw is None else pd.read_parquet(io.BytesIO(raw))


def lineup_tables(store) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """The line-up tables rebuilt from the collector's raw replies (the stored tables are only rebuilt at the end of
    a full football-trade.yml day, so a cancelled day leaves them behind)."""
    from sources.api_football_lineups import build
    ov = Overlay(store)
    rep = build(ov)
    return ov.get_parquet("api_football/lineup_times.parquet"), ov.get_parquet("api_football/lineups.parquet"), rep


def gpm_ratings(store, as_of: pd.Timestamp) -> dict:
    from football import player_ratings as pr
    hp = store.get_parquet("api_football/history_players.parquet")
    he = store.get_parquet("api_football/history_events.parquet")
    hm = store.get_parquet("api_football/history_matches.parquet")
    m = hm[["apif_fixture_id", "kickoff_utc"]].copy()
    m["date"] = pd.to_datetime(m["kickoff_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    sti = pr.stints(hp, he).merge(m[["apif_fixture_id", "date"]], on="apif_fixture_id")
    return pr.fit_gpm(sti, as_of=as_of), hp, m


def surprise_now(lineups: pd.DataFrame, times: pd.DataFrame, hp: pd.DataFrame, hdates: pd.DataFrame,
                 ratings: dict) -> pd.DataFrame:
    """Home less away of (confirmed eleven's rating - expected eleven's) for each collected match."""
    st = hp[(hp["role"] == "starter") & hp["apif_player_id"].notna()].merge(hdates[["apif_fixture_id", "date"]],
                                                                             on="apif_fixture_id")
    xi = lineups[(lineups["role"] == "starter") & lineups["apif_player_id"].notna()]
    rows = []
    for t in times.dropna(subset=["first_seen_utc"]).itertuples(index=False):
        ko = pd.Timestamp(t.kickoff_utc).tz_convert(None) if pd.Timestamp(t.kickoff_utc).tzinfo else pd.Timestamp(
            t.kickoff_utc)
        val = {}
        for side, team in (("home", t.apif_home_id), ("away", t.apif_away_id)):
            conf = xi[(xi["apif_fixture_id"] == t.apif_fixture_id) & (xi["side"] == side)]["apif_player_id"]
            prev = st[(st["apif_team_id"] == team) & (st["date"] < ko)]
            fids = prev.drop_duplicates("apif_fixture_id").sort_values("date")["apif_fixture_id"].tail(N_PREV)
            if len(conf) < 9 or len(fids) < 3:
                val = None
                break
            share = prev[prev["apif_fixture_id"].isin(fids)].groupby("apif_player_id").size() / len(fids)
            exp = float(sum(ratings.get(p, 0.0) * w for p, w in share.items()))
            got = float(sum(ratings.get(p, 0.0) for p in conf))
            val[side] = got - exp
        if val:
            rows.append({"apif_fixture_id": t.apif_fixture_id, "surprise": val["home"] - val["away"],
                         "first_seen_utc": pd.Timestamp(t.first_seen_utc), "kickoff": pd.Timestamp(t.kickoff_utc),
                         "home": t.home, "away": t.away})
    return pd.DataFrame(rows)


def books_for(store, days: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from football.exchange_feed import read_day
    mk, bk = [], []
    for d in days:
        m = read_day(store, d, "markets")
        b = read_day(store, d, "books")
        if m is None or b is None or m.empty or b.empty:
            continue
        m = m[m["market_type"] == "MATCH_ODDS"]
        mk.append(m)
        bk.append(b[b["market_id"].isin(m["market_id"])])
    if not mk:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(mk, ignore_index=True), pd.concat(bk, ignore_index=True)


def price_series(m: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One row a (market, poll): home/draw/away best back and lay, normalised mid log ratio, in play or not."""
    lab = m.drop_duplicates(["market_id", "selection_id"]).copy()
    parts = lab["event_name"].astype(str).str.split(" v ", n=1, regex=False)
    lab["leg"] = np.where(lab["runner_name"] == "The Draw", "d",
                          np.where(lab["runner_name"] == parts.str[0].str.strip(), "h",
                                   np.where(lab["runner_name"] == parts.str[1].str.strip(), "a", None)))
    b = b.merge(lab[["market_id", "selection_id", "leg"]].astype({"selection_id": "int64"}),
                on=["market_id", "selection_id"])
    b = b[b["leg"].notna()].copy()
    for c in ("back1", "lay1"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    w = b.pivot_table(index=["market_id", "polled_utc", "inplay"], columns="leg", values=["back1", "lay1"],
                      aggfunc="first")
    w.columns = [f"{a}_{c}" for a, c in w.columns]
    w = w.reset_index().dropna()
    for o in "hda":
        w[f"mid_{o}"] = (w[f"back1_{o}"] + w[f"lay1_{o}"]) / 2
    inv = 1 / w[[f"mid_{o}" for o in "hda"]]
    q = inv.div(inv.sum(axis=1), axis=0)
    w["lr"] = np.log(q["mid_h"] / q["mid_a"])
    w["t"] = pd.to_datetime(w["polled_utc"], utc=True, errors="coerce")
    return w.dropna(subset=["t"]).sort_values(["market_id", "t"])


def run(store, days: int, today: date | None = None) -> dict:
    from research.open_close.football_feed_coverage import link
    today = today or date.today()
    times, lineups, rebuild = lineup_tables(store)
    if times is None or lineups is None or times.empty:
        return {"status": "no data found (api_football line-up tables)"}
    dlist = [(today - timedelta(days=i)).isoformat() for i in range(days)]
    times = times[pd.to_datetime(times["kickoff_utc"], utc=True).dt.strftime("%Y-%m-%d").isin(dlist)]
    if times.empty:
        return {"status": "no collected line-ups in the window"}
    ratings, hp, hdates = gpm_ratings(store, pd.Timestamp(min(dlist)).replace(day=1))
    s = surprise_now(lineups, times, hp, hdates, ratings)
    m, b = books_for(store, dlist)
    out = {"rebuild": rebuild, "days": dlist[::-1], "lineups_collected": int(times["first_seen_utc"].notna().sum()),
           "with_surprise": int(len(s)), "players_rated": len(ratings),
           "release_minutes_before_off": {q: round(float(times["minutes_before_ko"].dropna().quantile(q)), 1)
                                          for q in (0.1, 0.5, 0.9)} if times["minutes_before_ko"].notna().any()
           else None}
    if s.empty or m.empty:
        out["status"] = "no surprise or no recorded match odds books"
        return out
    mk = m.drop_duplicates("market_id").copy()
    mk["kickoff"] = pd.to_datetime(mk["market_start_utc"], utc=True, errors="coerce")
    parts = mk["event_name"].astype(str).str.split(" v ", n=1, regex=False)
    mk["ev_home"], mk["ev_away"] = parts.str[0].str.strip(), parts.str[1].str.strip()
    fx = s.rename(columns={"kickoff": "kickoff_utc"}).assign(status="SCHEDULED", apif_league_id=0, league="",
                                                             league_country="", apif_season=0)
    j = link(mk.dropna(subset=["kickoff"]), fx)
    j = j.dropna(subset=["apif_fixture_id"]).merge(s[["apif_fixture_id", "surprise", "first_seen_utc"]],
                                                   on="apif_fixture_id")
    ps = price_series(m, b)
    rows = []
    for r in j.itertuples(index=False):
        p = ps[(ps["market_id"] == r.market_id) & (ps["inplay"] == 0)]
        rel = pd.Timestamp(r.first_seen_utc)
        rel = rel.tz_localize("UTC") if rel.tzinfo is None else rel
        pre = p[p["t"] < rel]
        ent = p[p["t"] >= rel + pd.Timedelta(minutes=DELAY_MIN)]
        if pre.empty or ent.empty:
            continue
        e, c = ent.iloc[0], p.iloc[-1]
        if c["t"] <= e["t"]:
            continue
        rows.append({"market_id": r.market_id, "surprise": r.surprise, "pre": pre.iloc[-1]["lr"], "entry": e["lr"],
                     "close": c["lr"], "entry_wait_min": (e["t"] - rel).total_seconds() / 60,
                     "back_h": e["back1_h"], "back_a": e["back1_a"], "lay_close_h": c["lay1_h"],
                     "lay_close_a": c["lay1_a"]})
    d = pd.DataFrame(rows)
    out["matches_with_books"] = int(len(d))
    if len(d) < 10:
        out["status"] = "too few matches with books around the release yet"
        return out
    d["move_release"] = d["close"] - d["pre"]
    d["move_after_entry"] = d["close"] - d["entry"]
    d["move_to_entry"] = d["entry"] - d["pre"]
    out["corr_surprise_with"] = {k: round(float(np.corrcoef(d["surprise"], d[k])[0, 1]), 4)
                                 for k in ("move_release", "move_to_entry", "move_after_entry")}
    out["median_entry_wait_min"] = round(float(d["entry_wait_min"].median()), 1)
    trades = {}
    for t in THRESH:
        sel = d[d["surprise"].abs() > t]
        prof = np.where(sel["surprise"] > 0, sel["back_h"] / sel["lay_close_h"] - 1, sel["back_a"] / sel["lay_close_a"] - 1)
        prof = np.where(prof > 0, prof * (1 - COMMISSION), prof)
        trades[f"surprise_over_{t}"] = {"bets": int(len(sel)), "roi": round(float(np.mean(prof)), 4) if len(sel) else None,
                                        "right_way": round(float(np.mean(prof > 0)), 4) if len(sel) else None}
    out["trades_at_the_books"] = trades
    return out


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Store(root=a.root), a.days)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
