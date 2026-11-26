"""When does team news arrive, and when does Betfair's price take it in? The best times to pull injuries and line-ups.

racing² football research, 26 Nov 2026 (the owner: "Review the optimal time to pull team and lineup news").
Descriptive; nothing is traded and no rule is scored. Four reads:

1. **The price's clock** (``gold/checkpoint``, Betfair's own price files, every match-odds market the warehouse
   holds): the home side's normalised chance as a logit at each checkpoint (72 h, 24 h, 12 h, 6 h, 3 h, 2 h, 90, 60,
   45, 30, 15, 5, 1 minutes, the off). For each interval, the mean squared move: its share of the whole move from
   72 hours to the off, and its rate a minute. Information arrives where the rate jumps. Split by how busy the market
   is (price updates up to the off, terciles).
2. **The same, minute by minute near the off** (the football feed's recorded books, the last ``days`` days): the
   squared move of the normalised mid between polls, by 5-minute band from 120 minutes to the off, a minute.
3. **Line-ups** (the collector's raw polls, rebuilt in memory): when API-Football first holds both elevens, minutes
   before kick-off, overall and by league; how often an eleven changed after it first appeared.
4. **Injuries** (the collector's raw ``/injuries?date=`` replies): for each fixture and player, when first seen
   against the kick-off; how much of a fixture's final list the first poll of the day already held; how many
   players were added or changed type in later polls, and when.

    python -m research.open_close.football_news_timing --days 21 --out out/news_timing.json --md out/news_timing.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BANDS = list(range(120, -1, -5))          # minutes before the off, 5-minute bands for the recorded books


# ---------------------------------------------------------------------------------------------------------------
# 1. the price's clock, from Betfair's files
# ---------------------------------------------------------------------------------------------------------------

def checkpoint_moves(store) -> dict:
    from football.warehouse.gold_checkpoint import CHECKPOINTS
    from football.warehouse.store_io import list_parts, read_part
    cols = ["market_id", "cp_min", "market_type", "role", "schedule_known", "book_complete", "p_norm",
            "n_ticks_before"]
    parts = []
    for y, m in list_parts(store, "gold/checkpoint"):
        t = read_part(store, "gold/checkpoint", y, m, cols)
        if t is None:
            continue
        d = t.to_pandas()
        d = d[(d["market_type"] == "MATCH_ODDS") & (d["role"] == "home") & d["schedule_known"] & d["book_complete"]]
        parts.append(d[["market_id", "cp_min", "p_norm", "n_ticks_before"]])
    if not parts:
        return {"status": "no data found (gold/checkpoint)"}
    d = pd.concat(parts, ignore_index=True)
    rows_in = int(len(d))
    p = d["p_norm"].clip(1e-4, 1 - 1e-4)
    d["x"] = np.log(p / (1 - p))
    w = d.pivot_table(index="market_id", columns="cp_min", values="x", aggfunc="first")
    act = d[d["cp_min"] == 0].set_index("market_id")["n_ticks_before"]
    grid = [c for c in sorted(CHECKPOINTS, reverse=True) if c in w.columns]
    w = w[grid].dropna()                                      # one sample: every checkpoint present
    out = {"rows_in": rows_in, "markets_all_checkpoints": int(len(w))}
    if w.empty:
        out["status"] = "no market holds every checkpoint"
        return out
    tiers = pd.qcut(act.reindex(w.index).rank(method="first"), 3, labels=["quiet", "middle", "busy"])

    def table(ww: pd.DataFrame) -> list[dict]:
        total = float(((ww[grid[-1]] - ww[grid[0]]) ** 2).mean())
        res = []
        for a, b in zip(grid[:-1], grid[1:]):
            msq = float(((ww[b] - ww[a]) ** 2).mean())
            res.append({"from_min": int(a), "to_min": int(b), "mean_sq_move": round(msq, 6),
                        "share_of_sum": None, "per_minute": round(msq / max(a - b, 1), 7),
                        "abs_move_median": round(float((ww[b] - ww[a]).abs().median()), 4)})
        s = sum(r["mean_sq_move"] for r in res)
        for r in res:
            r["share_of_sum"] = round(r["mean_sq_move"] / s, 4) if s else None
        return res + [{"total_72h_to_off_mean_sq": round(total, 6)}]

    out["all"] = table(w)
    out["by_activity"] = {str(k): table(w[tiers == k]) for k in ["quiet", "middle", "busy"] if (tiers == k).sum() > 50}
    out["activity_cut_ticks"] = [float(x) for x in act.reindex(w.index).quantile([1 / 3, 2 / 3])]
    return out


# ---------------------------------------------------------------------------------------------------------------
# 2. minute by minute near the off, from the recorded books
# ---------------------------------------------------------------------------------------------------------------

def book_moves(store, days: list[str]) -> dict:
    from research.open_close.football_lineup_release import books_for, price_series
    m, b = books_for(store, days)
    if m.empty:
        return {"status": "no data found (recorded books)"}
    ps = price_series(m, b)
    ps = ps[ps["inplay"] == 0]
    st = m.drop_duplicates("market_id").set_index("market_id")["market_start_utc"]
    ps["ko"] = pd.to_datetime(ps["market_id"].map(st), utc=True, errors="coerce")
    ps = ps.dropna(subset=["ko"]).sort_values(["market_id", "t"])
    ps["mins"] = (ps["ko"] - ps["t"]).dt.total_seconds() / 60
    ps = ps[(ps["mins"] >= 0) & (ps["mins"] <= 125)]
    ps["d"] = ps.groupby("market_id")["lr"].diff()
    ps["dt"] = ps.groupby("market_id")["t"].diff().dt.total_seconds() / 60
    ps = ps.dropna(subset=["d", "dt"])
    ps = ps[(ps["dt"] > 0) & (ps["dt"] <= 6)]
    ps["band"] = (np.floor(ps["mins"] / 5) * 5).astype(int)
    rows = []
    for bnd, g in ps.groupby("band"):
        if bnd > 115:
            continue
        rows.append({"band_min": f"{bnd + 5}-{bnd}", "polls": int(len(g)), "markets": int(g["market_id"].nunique()),
                     "sq_move_per_minute": round(float((g["d"] ** 2).sum() / g["dt"].sum()), 7)})
    rows.sort(key=lambda r: -int(r["band_min"].split("-")[1]))
    return {"days": days, "markets": int(ps["market_id"].nunique()), "bands": rows}


# ---------------------------------------------------------------------------------------------------------------
# 3. line-ups, 4. injuries
# ---------------------------------------------------------------------------------------------------------------

def lineup_timing(store) -> dict:
    from research.open_close.football_lineup_release import lineup_tables
    times, _, rep = lineup_tables(store)
    if times is None or times.empty:
        return {"status": "no data found (line-up polls)"}
    t = times.dropna(subset=["minutes_before_ko"])
    out = {"matches": int(len(times)), "with_both_elevens": int(len(t)),
           "minutes_before_ko": {q: round(float(t["minutes_before_ko"].quantile(q)), 1) for q in (0.1, 0.5, 0.9)},
           "first_poll_already_had_them": int((t["last_unseen_utc"].isna()).sum()) if "last_unseen_utc" in t else None,
           "xi_changed_after_first_seen": int(t["xi_changed_after_first"].sum()) if "xi_changed_after_first" in t else None}
    if "league" in t.columns:
        g = t.groupby("league")["minutes_before_ko"].agg(["size", "median"]).sort_values("size", ascending=False)
        out["by_league"] = [{"league": k, "matches": int(r["size"]), "median_min": round(float(r["median"]), 1)}
                            for k, r in g.head(15).iterrows()]
    return out


def injury_timing(store) -> dict:
    keys = [k for k in sorted(store.listing("api_football/raw/injuries/")) if k.endswith(".json.gz")]
    if not keys:
        return {"status": "no data found (injury polls)"}
    seen, polls = {}, []
    for k in keys:
        try:
            doc = json.loads(gzip.decompress(store.get(k)))
        except (OSError, ValueError):
            continue
        t = pd.Timestamp(doc.get("fetched_utc"))
        t = t.tz_localize("UTC") if t.tzinfo is None else t
        polls.append(t)
        for r in (doc.get("body") or {}).get("response") or []:
            pl, fx = r.get("player") or {}, r.get("fixture") or {}
            if pl.get("id") is None or fx.get("id") is None or not fx.get("timestamp"):
                continue
            ko = pd.Timestamp(fx["timestamp"], unit="s", tz="UTC")
            key = (fx["id"], pl["id"])
            rec = seen.setdefault(key, {"fixture": fx["id"], "ko": ko, "first": t, "types": []})
            rec["first"] = min(rec["first"], t)
            if not rec["types"] or rec["types"][-1][1] != pl.get("type"):
                rec["types"].append((t, pl.get("type")))
    d = pd.DataFrame([{"fixture": v["fixture"], "ko": v["ko"], "first": v["first"],
                       "changes": len(v["types"]) - 1,
                       "last_change": v["types"][-1][0] if len(v["types"]) > 1 else pd.NaT} for v in seen.values()])
    if d.empty:
        return {"status": "no injury rows", "polls": len(polls)}
    d["first_h_before_ko"] = (d["ko"] - d["first"]).dt.total_seconds() / 3600
    first_poll = d.groupby("fixture")["first"].transform("min")
    d["added_later"] = d["first"] > first_poll + pd.Timedelta(minutes=30)
    d["added_h_before_ko"] = np.where(d["added_later"], d["first_h_before_ko"], np.nan)
    ph = pd.Series(polls).dt.hour.value_counts().sort_index()
    return {"polls": len(polls), "days": len({p.date() for p in polls}), "poll_hours_utc": ph.to_dict(),
            "fixture_players": int(len(d)), "fixtures": int(d["fixture"].nunique()),
            "first_seen_hours_before_ko": {q: round(float(d["first_h_before_ko"].quantile(q)), 2) for q in (0.1, 0.5, 0.9)},
            "share_on_first_poll_of_fixture": round(float(1 - d["added_later"].mean()), 4),
            "added_later": int(d["added_later"].sum()),
            "added_later_hours_before_ko": {q: round(float(np.nanquantile(d["added_h_before_ko"], q)), 2)
                                            for q in (0.1, 0.5, 0.9)} if d["added_later"].any() else None,
            "type_changed": int((d["changes"] > 0).sum()),
            "note": "the collector asks /injuries?date=<UK day> only, so nothing is seen before the match day"}


# ---------------------------------------------------------------------------------------------------------------

def markdown(res: dict) -> str:
    L = [f"# racing² football - when team news arrives and when the price takes it in ({res['run_day']})", "",
         "Source: `research/open_close/football_news_timing.py` (gold/checkpoint, the feed's recorded books, the "
         "API-Football collector's raw polls). Descriptive; nothing traded.", ""]
    c = res["price_clock"]
    if "all" in c:
        L += [f"## 1. Betfair's price, 72 h to the off ({c['markets_all_checkpoints']:,} match-odds markets with "
              "every checkpoint)", "", "| Interval (min before off) | Share of the move | Rate a minute | Median abs move |",
              "|---|---|---|---|"]
        for r in c["all"][:-1]:
            L.append(f"| {r['from_min']} to {r['to_min']} | {r['share_of_sum']:.1%} | {r['per_minute']:.2e} | "
                     f"{r['abs_move_median']} |")
        L.append("")
        for k, tab in c.get("by_activity", {}).items():
            top = sorted(tab[:-1], key=lambda r: -r["per_minute"])[:4]
            L.append(f"- {k} markets, fastest intervals: " + ", ".join(f"{r['from_min']}-{r['to_min']}" for r in top))
    else:
        L.append(f"Price clock: {c.get('status')}")
    bm = res["books_near_off"]
    L += ["", "## 2. The recorded books near the off", ""]
    if "bands" in bm:
        L += [f"{bm['markets']} markets, {len(bm['days'])} days.", "", "| Minutes before off | Polls | Squared move a minute |",
              "|---|---|---|"]
        L += [f"| {r['band_min']} | {r['polls']} | {r['sq_move_per_minute']:.2e} |" for r in bm["bands"]]
    else:
        L.append(bm.get("status", "no data found"))
    lu, ij = res["lineups"], res["injuries"]
    L += ["", "## 3. Line-ups from API-Football", "", json.dumps({k: v for k, v in lu.items() if k != "by_league"}),
          "", "## 4. Injuries from API-Football", "", json.dumps(ij, default=str)]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    s = Store()
    today = date.today()
    days = [(today - timedelta(days=i)).isoformat() for i in range(a.days, 0, -1)]
    res = {"run_day": today.isoformat()}
    for name, fn in (("price_clock", lambda: checkpoint_moves(s)), ("books_near_off", lambda: book_moves(s, days)),
                     ("lineups", lambda: lineup_timing(s)), ("injuries", lambda: injury_timing(s))):
        try:
            res[name] = fn()
        except Exception as exc:                                  # each read stands alone
            log.exception("%s failed", name)
            res[name] = {"status": f"failed: {str(exc)[:300]}"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    if a.md:
        Path(a.md).write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
