"""The place market against the win market's close, on the recorded books (read-only; research/greyhound/place_value.py
is the price files' version). The price files showed the place SP mispriced against the win SP (backs at the place SP
where Harville on the win SP clears it by 20%: +25.8% on 18,315 bets, 2023-26), but the win SP is known only at the
off. Here the win market's close is forecast from the book a few seconds before it, as a bet would be:

- at each decision time (``SECONDS`` before the scheduled off), each runner's last pre-off book in the win and the
  place market (greyhound/book_record.py's live-key books, else the recorder's);
- the win price: Betfair's projected SP (``sp_near``) where the book shows one, else the mid; the win chances are its
  inverses normalised in the race, Harville's chance of placing from them (the places the market pays: its
  ``number_of_winners``, else the place market's winners);
- the place price at the decision: the place market's projected SP, else its mid;
- back at the place SP where p x (price - 1) x (1 - c) - (1 - p) clears a threshold, at the decision's place price, then
  settled at the place SP and the result (the runner's WINNER/LOSER in the settled place book).

Every bet at the SP is matched at the SP. Commission 2%. A 90% interval by a bootstrap over races (few days yet).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from research.greyhound import place_value as pv

log = logging.getLogger(__name__)

SECONDS = (120, 60, 30, 10)
THRESHOLDS = (0.0, 0.05, 0.10, 0.20)
COMM = 0.02
FEEDS = (("books_greyhound_live.csv", "markets_greyhound_live.csv", "live key"),
         ("books_greyhound.csv", "markets_greyhound.csv", "recorder"))


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def day_frames(day: date, root=None):
    """The day's books and markets, the live key's where it has both win and place markets, else the recorder's."""
    from betfair_recorder import _read_day_csv
    for bname, mname, label in FEEDS:
        b, m = _read_day_csv(day, bname, root=root), _read_day_csv(day, mname, root=root)
        if b is None or m is None or not len(b):
            continue
        types = set(m.market_type.astype(str))
        if {"WIN", "PLACE"} <= types:
            return b, m, label
    return None, None, None


def settle(b: pd.DataFrame) -> pd.DataFrame:
    """Each runner's SP and result from the settled books."""
    x = b.assign(sp=_num(b.sp_actual), status=b.runner_status.astype(str))
    sp = x[x.sp > 1].groupby(["market_id", "selection_id"]).sp.last()
    res = x[x.status.isin(["WINNER", "LOSER"])].groupby(["market_id", "selection_id"]).status.last()
    nw = _num(x.get("number_of_winners", pd.Series(np.nan, index=x.index))).groupby(x.market_id).max()
    out = pd.concat([sp, res], axis=1).reset_index()
    out["places_paid"] = out.market_id.map(nw)
    return out


def snapshot(b: pd.DataFrame, offs: pd.Series, secs: int) -> pd.DataFrame:
    """Each runner's last open, pre-play book at least ``secs`` before the scheduled off."""
    x = b.copy()
    x["polled"] = pd.to_datetime(x.polled_utc, utc=True, errors="coerce")
    x["off"] = pd.to_datetime(x.market_id.map(offs), utc=True, errors="coerce")
    ok = ((x.source.astype(str) != "final") & (x.status.astype(str) == "OPEN")
          & ~x.inplay.astype(str).str.lower().isin(["true", "1"])
          & (x.runner_status.fillna("ACTIVE").astype(str) == "ACTIVE")
          & (x.polled <= x.off - pd.Timedelta(seconds=secs)))
    x = x[ok].sort_values("polled").drop_duplicates(["market_id", "selection_id"], keep="last")
    back1, lay1 = _num(x.back1), _num(x.lay1)
    mid = np.where((lay1 > 1) & (lay1 / back1 < 1.25), np.sqrt(back1 * lay1), back1)
    near = _num(x.sp_near)
    x["price"] = np.where((near > 1) & (near < 1000), near, mid)
    x["from_sp_near"] = (near > 1) & (near < 1000)
    x["age_s"] = (x.off - x.polled).dt.total_seconds()
    return x[["market_id", "selection_id", "price", "from_sp_near", "age_s"]]


def day_rows(day: date, root=None) -> pd.DataFrame:
    b, m, feed = day_frames(day, root)
    if b is None:
        return pd.DataFrame()
    b = b.assign(market_id=b.market_id.astype(str), selection_id=_num(b.selection_id))
    m = m.assign(market_id=m.market_id.astype(str), selection_id=_num(m.selection_id))
    mk = m.drop_duplicates(["market_id", "selection_id"])
    offs = mk.drop_duplicates("market_id").set_index("market_id").market_start_utc
    # a race's win and place markets: the same event and start; a runner by its name
    mk = mk.assign(race=mk.event_id.astype(str) + "|" + mk.market_start_utc.astype(str),
                   name=mk.runner_name.astype(str).str.replace(r"^\d+\.\s*", "", regex=True).str.strip().str.lower())
    win = mk[mk.market_type.astype(str) == "WIN"][["race", "name", "market_id", "selection_id", "venue"]]
    plc = mk[mk.market_type.astype(str) == "PLACE"][["race", "name", "market_id", "selection_id"]]
    pair = win.merge(plc, on=["race", "name"], suffixes=("_w", "_p"))
    st = settle(b)
    pair = pair.merge(st.rename(columns={"market_id": "market_id_p", "selection_id": "selection_id_p", "sp": "sp_p",
                                         "status": "res_p", "places_paid": "places_paid"}),
                      on=["market_id_p", "selection_id_p"], how="left")
    out = []
    for secs in SECONDS:
        s = snapshot(b, offs, secs)
        x = pair.merge(s.rename(columns={"market_id": "market_id_w", "selection_id": "selection_id_w",
                                         "price": "win_now", "from_sp_near": "w_near", "age_s": "w_age"}),
                       on=["market_id_w", "selection_id_w"], how="inner")
        x = x.merge(s.rename(columns={"market_id": "market_id_p", "selection_id": "selection_id_p",
                                      "price": "place_now", "from_sp_near": "p_near", "age_s": "p_age"}),
                     on=["market_id_p", "selection_id_p"], how="inner")
        out.append(x.assign(secs=secs, day=f"{day}", feed=feed))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score(rows: pd.DataFrame) -> dict:
    d = rows.copy()
    d = d[(d.sp_p > 1) & d.res_p.isin(["WINNER", "LOSER"]) & (d.win_now > 1) & (d.place_now > 1)]
    d["placed"] = (d.res_p == "WINNER").astype(int)
    key = ["race", "secs"]
    g = d.groupby(key)
    n = g.name.transform("size")
    k = d.places_paid.where(d.places_paid.isin([2, 3]), g.placed.transform("sum"))
    d = d[(n >= 4) & (n <= pv.MAXN) & k.isin([2, 3]) & (k < n)].assign(k=k)
    d = d.sort_values(key + ["name"])
    out = {"runners": len(d), "races": int(d.race.nunique()), "days": sorted(d.day.unique()),
           "feeds": d.drop_duplicates("race").feed.value_counts().to_dict(), "by_secs": []}
    for secs, x in d.groupby("secs"):
        inv = 1.0 / x.win_now
        x = x.assign(q=inv / inv.groupby(x.race).transform("sum"))
        codes, idx = np.unique(x.race.to_numpy(), return_inverse=True)
        pos = x.groupby("race").cumcount().to_numpy()
        Q = np.zeros((len(codes), pv.MAXN))
        Q[idx, pos] = x.q.to_numpy()
        K = x.groupby("race").k.first().reindex(codes).to_numpy()
        x["p"] = pv.harville(Q, K)[idx, pos]
        ev = x.p * (x.place_now - 1) * (1 - COMM) - (1 - x.p)
        pnl = np.where(x.placed == 1, (x.sp_p - 1) * (1 - COMM), -1.0)
        row = {"secs": int(secs), "runners": len(x), "win_from_sp_near": round(float(x.w_near.mean()), 3),
               "place_from_sp_near": round(float(x.p_near.mean()), 3),
               "median_book_age_s": float(x.w_age.median()),
               "rmse_log_place_now_vs_sp": round(float(np.sqrt(np.mean(np.log(x.place_now / x.sp_p) ** 2))), 4),
               "rules": []}
        for th in THRESHOLDS:
            sel = (ev > th).to_numpy()
            if sel.sum() < 20:
                row["rules"].append({"edge>": th, "bets": int(sel.sum())})
                continue
            r = pd.DataFrame({"race": x.race.to_numpy()[sel], "pnl": pnl[sel]})
            byr = r.groupby("race").agg(pnl=("pnl", "sum"), n=("pnl", "size"))
            lo, hi = pv._boot(byr.pnl, byr.n)
            row["rules"].append({"edge>": th, "bets": int(sel.sum()), "races": len(byr),
                                 "mean_place_sp": round(float(x.sp_p.to_numpy()[sel].mean()), 2),
                                 "roi%": round(100 * r.pnl.mean(), 2), "ci90%": [round(100 * lo, 2), round(100 * hi, 2)],
                                 "pnl_units": round(float(r.pnl.sum()), 2)})
        out["by_secs"].append(row)
    return out


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="first", default="2026-10-06")
    ap.add_argument("--to", dest="last", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    first = date.fromisoformat(a.first)
    last = date.fromisoformat(a.last) if a.last else date.today() - timedelta(days=1)
    parts, seen = [], []
    d = first
    while d <= last:
        try:
            r = day_rows(d, a.root)
        except Exception as exc:
            log.warning("%s: %s", d, exc)
            r = pd.DataFrame()
        if len(r):
            parts.append(r)
            seen.append(f"{d} ({r.feed.iloc[0]}, {r.race.nunique()} races)")
        d += timedelta(days=1)
    log.info("days with win and place books: %s", seen)
    if not parts:
        print(json.dumps({"note": "no day with both win and place books"}))
        return 0
    rep = score(pd.concat(parts, ignore_index=True))
    text = json.dumps(rep, indent=1, default=str)
    if a.out:
        open(a.out, "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
