"""Line-up news and Betfair's late move: is the confirmed eleven priced in by an hour before the off?

racing² football research, 8 Nov 2026 (the owner: "improving our closing line model ... a profitable, automated,
trading solution"). Line-ups come out about an hour before kick-off; it is the biggest known driver of late football
moves, and the one an automated trader can act on fastest. Data, all from regular feeds:

- the line-ups and events of every match API-Football holds for our 361 leagues (``api_football/history_*``);
- each player's goals plus-minus (``football.player_ratings.fit_gpm``: goal difference per 90 while on the pitch,
  ridge-shrunk), refitted monthly on earlier matches only;
- the surprise (``football.player_ratings.surprise``): the confirmed eleven's summed rating less the expected eleven's
  (each player's share of his side's starts in its previous five matches), home less away;
- Betfair's match odds from the owner's Historic Data tables: the last traded price 24 hours, 60, 15 and 1 minute
  before the off and at the close, each match linked to its API-Football fixture.

Fixed before the run:

- the price as the home side's log chance ratio, log(q_home / q_away), the three prices normalised to one;
- the moves: early (24 h to 60 min), late (60 min to the close), last (15 min to the close);
- does the surprise point the way of each move (correlation; the slope of the late move on the surprise with the
  early move beside it, 95% interval by match); and is the late move explained beyond the early one;
- the trade: at 60 (and 15) minutes back the side the surprise favours when |surprise| exceeds 0.1 or 0.2, closed at
  the close; at the last traded prices and a tick worse each end, 2% commission on a market's net win; August and
  September apart (does it hold in both months?).

    python -m research.open_close.football_lineup_late_move --out out/lineup_late_move.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARKS = ["ltp_t_1440", "ltp_t_60", "ltp_t_15", "ltp_t_1", "ltp_close"]
FROM = "2025-06-01"                 # matches older than this only feed the ratings' fits
THRESH = (0.1, 0.2)
COMMISSION = 0.02


def match_odds_paths(store) -> pd.DataFrame:
    """One row a match-odds market: home/draw/away prices at each mark, normalised log chance ratio, kick-off."""
    keys = sorted(k for k in store.listing("betfair_historic/") if "markets_soccer" in k and k.endswith(".parquet"))
    frames = []
    for k in keys:
        f = store.get_parquet(k)
        if f is not None and len(f):
            frames.append(f[f["market_type"] == "MATCH_ODDS"])
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    parts = m["event_name"].astype(str).str.split(" v ", n=1, regex=False)
    m["ev_home"], m["ev_away"] = parts.str[0].str.strip(), parts.str[1].str.strip()
    m["leg"] = np.where(m["runner_name"] == "The Draw", "d", np.where(m["runner_name"] == m["ev_home"], "h",
                                                                         np.where(m["runner_name"] == m["ev_away"], "a",
                                                                                  None)))
    m = m[m["leg"].notna()]
    for c in MARKS:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    w = m.pivot_table(index="market_id", columns="leg", values=MARKS, aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    meta = m.drop_duplicates("market_id").set_index("market_id")[["event_name", "ev_home", "ev_away", "market_time"]]
    w = w.join(meta).reset_index()
    w["kickoff"] = pd.to_datetime(w["market_time"], utc=True, errors="coerce")
    for c in MARKS:
        cols = [f"{c}_{o}" for o in "hda"]
        if not set(cols) <= set(w.columns):
            continue
        inv = 1 / w[cols]
        q = inv.div(inv.sum(axis=1), axis=0)
        w[f"lr_{c}"] = np.log(q[f"{c}_h"] / q[f"{c}_a"])
    return w.dropna(subset=["kickoff"])


def surprises(hm: pd.DataFrame, hp: pd.DataFrame, he: pd.DataFrame, months_from: str) -> pd.DataFrame:
    from football import player_ratings as pr
    m = hm[["apif_fixture_id", "kickoff_utc"]].copy()
    m["date"] = pd.to_datetime(m["kickoff_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    m = m.dropna(subset=["date"])
    st = pr.starters(hp)
    sti = pr.stints(hp, he).merge(m[["apif_fixture_id", "date"]], on="apif_fixture_id")
    gpm = {}
    for mo in pd.period_range(months_from, m["date"].max(), freq="M"):
        gpm[str(mo)] = pr.fit_gpm(sti, as_of=mo.start_time)
    recent = m[m["date"] >= pd.Timestamp(FROM)]
    s = pr.surprise(recent, st[st["apif_fixture_id"].isin(recent["apif_fixture_id"])], gpm, "gpm")
    s.attrs["players_rated"] = {k: len(v) for k, v in gpm.items()}
    s.attrs["stints"] = int(len(sti))
    return s


def _boot_slope(x: np.ndarray, y: np.ndarray, c: np.ndarray | None, n: int = 1000, seed: int = 0) -> tuple:
    X = np.c_[np.ones(len(x)), x] if c is None else np.c_[np.ones(len(x)), x, c]
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    return round(float(b), 4), round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)


def trades(p: pd.DataFrame, mark: str, t: float) -> dict:
    """Back the side the surprise favours at ``mark`` when |surprise| > t, closed at the close."""
    from research.open_close.backtest import tick_over, tick_under
    out = {}
    sel = p[p["gpm_surprise"].abs() > t]
    for cost in ("ltp", "tick"):
        prof, mk, month = [], [], []
        for r in sel.itertuples(index=False):
            leg = "h" if r.gpm_surprise > 0 else "a"
            o, c = getattr(r, f"{mark}_{leg}"), getattr(r, f"ltp_close_{leg}")
            if not (o == o and c == c):
                continue
            if cost == "tick":
                o, c = tick_under(o), tick_over(c)
            prof.append(o / c - 1)
            mk.append(r.market_id)
            month.append(r.kickoff.strftime("%Y-%m"))
        if not prof:
            out[cost] = {"bets": 0}
            continue
        d = pd.DataFrame({"p": prof, "m": mk, "mo": month})
        d["p"] = np.where(d["p"] > 0, d["p"] * (1 - COMMISSION), d["p"])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(d), (1000, len(d)))
        b = d["p"].to_numpy()[idx].mean(1)
        out[cost] = {"bets": int(len(d)), "roi": round(float(d["p"].mean()), 4),
                     "lo95": round(float(np.quantile(b, 0.025)), 4), "hi95": round(float(np.quantile(b, 0.975)), 4),
                     "right_way": round(float((d["p"] > 0).mean()), 4),
                     "by_month": d.groupby("mo")["p"].agg(["size", "mean"]).round(4).to_dict("index")}
    return out


def run(store) -> dict:
    from research.open_close.football_feed_coverage import link
    hm = store.get_parquet("api_football/history_matches.parquet")
    hp = store.get_parquet("api_football/history_players.parquet")
    he = store.get_parquet("api_football/history_events.parquet")
    fx = store.get_parquet("api_football/fixtures.parquet")
    if any(x is None for x in (hm, hp, he, fx)):
        return {"status": "no data found (api_football history tables)"}
    w = match_odds_paths(store)
    if w.empty:
        return {"status": "no Betfair match odds found"}
    first = w["kickoff"].min().strftime("%Y-%m")
    s = surprises(hm, hp, he, first)
    j = link(w, fx)
    j = j.merge(s, on="apif_fixture_id", how="inner")
    need = ["lr_ltp_t_1440", "lr_ltp_t_60", "lr_ltp_t_15", "lr_ltp_close", "gpm_surprise"]
    p = j.dropna(subset=[c for c in need if c in j.columns]).copy()
    out = {"markets": int(len(w)), "linked": int(j["apif_fixture_id"].notna().sum()) if len(j) else 0,
           "with_surprise_and_path": int(len(p)), "stints": s.attrs.get("stints"),
           "players_rated_by_month": s.attrs.get("players_rated"),
           "first_kickoff": str(w["kickoff"].min()), "last_kickoff": str(w["kickoff"].max())}
    if len(p) < 200:
        out["status"] = "too few matches with a surprise and a full price path"
        return out
    p["early"] = p["lr_ltp_t_60"] - p["lr_ltp_t_1440"]
    p["late"] = p["lr_ltp_close"] - p["lr_ltp_t_60"]
    p["last"] = p["lr_ltp_close"] - p["lr_ltp_t_15"]
    x = p["gpm_surprise"].to_numpy(float)
    out["corr_surprise_with"] = {k: round(float(np.corrcoef(x, p[k])[0, 1]), 4) for k in ("early", "late", "last")}
    out["late_on_surprise_slope"] = _boot_slope(x, p["late"].to_numpy(float), None)
    out["late_on_surprise_given_early_slope"] = _boot_slope(x, p["late"].to_numpy(float), p["early"].to_numpy(float))
    out["early_on_surprise_slope"] = _boot_slope(x, p["early"].to_numpy(float), None)
    out["surprise_quantiles"] = {q: round(float(np.quantile(np.abs(x), q)), 4) for q in (0.5, 0.75, 0.9)}
    out["trades"] = {f"at_{m.split('_')[-1]}min_surprise_over_{t}": trades(p, m, t)
                     for m in ("ltp_t_60", "ltp_t_15") for t in THRESH}
    return out


def main(argv=None) -> int:
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Store(root=a.root))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
