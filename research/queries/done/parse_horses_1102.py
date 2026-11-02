"""Oddschecker's bookmaker prices as a signal for the Betfair books and the BSP (the owner's ask, 1 Nov 2026; read-only).

Reads the Parse capture committed beside the code (data/parse/oddschecker_prices.parquet: every GB/IE runner's price at
about 25 bookmakers, the first snapshot of the day and T-120/T-60/T-15) and horse_racing.db (betfair_live_markets,
betfair_live_marks: the recorder's books and the settled BSP). For each runner:

  - the bookmakers' morning price: the best and the median across the books at the first snapshot;
  - the Betfair 08:00 UK mid (the recorder's open) and the T-15 mid; the BSP; the result.

It asks three things, on the days held (one day at first: descriptive, not a test):

  1. which morning price is nearer the BSP: the bookmakers' median, their best, or the Betfair 08:00 mid (median and
     spread of log(BSP / price), and the log loss of each book's normalised chances against the result);
  2. does the bookmakers' morning price add to the Betfair mid in forecasting the BSP (OLS of log BSP on both);
  3. the exchange trade it would signal (exchange legs only, the owner's rule, 1 Nov 2026: no bookmaker bets). The
     bookmakers' morning median is information, not a venue. Where it sits shorter than Betfair's 08:00 book, back
     at the 08:00 best back on Betfair and lay at the BSP; where it sits longer, lay at the 08:00 best lay and back
     at the BSP. Greened, per GBP1 backed: B/BSP - 1; per GBP1 laid: 1 - L/BSP (return on the liability L - 1);
     2% commission on each market's net profit. Reported by how far the bookmakers' median sat from the 08:00 book.
  3b. across exchanges: back at Matchbook's morning price (Oddschecker's code MA: its price sits above every
     bookmaker's, as an exchange's would; the name is unverified), lay at the Betfair BSP, where
     Matchbook's price is over Betfair's 08:00 best back. Matchbook's own commission rate is not applied (to
     re-check before any money); Betfair's 2% is.
"""
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.getcwd())
sys.path.insert(0, str(ROOT))
DB = os.getenv("OC_DB", "horse_racing.db")
CAPTURE = ROOT / "data" / "parse" / "oddschecker_prices.parquet"
COMM = 0.02


def norm_name(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", str(s or "").lower())
    return re.sub(r"[^a-z]", "", s)


def exchange_trade(j, morning, days, mark, tag, note, window=None):
    """Section 3 at one Betfair mark: the trade the bookmakers' morning price would signal, every leg on an exchange."""
    # 3. the exchange trade the bookmakers' morning price would signal: every leg on Betfair
    print(f"\n-- 3{tag}. exchange only: the bookmakers' morning median against Betfair's {mark} book, traded on "
          f"Betfair at that book's best back or best lay, closed at the BSP. {note}")
    conn = sqlite3.connect(DB)
    m8 = pd.read_sql_query("SELECT market_id, selection_id, back1, lay1 FROM betfair_live_marks "
                           "WHERE mark = ? AND race_date >= ? AND race_date <= ?", conn,
                           params=(mark, days[0], days[-1]))
    conn.close()
    m8["selection_id"] = pd.to_numeric(m8["selection_id"], errors="coerce").astype("Int64").astype(str)
    j["selection_id"] = j["selection_id"].astype(str)
    d = j.merge(m8.drop_duplicates(["market_id", "selection_id"], keep="last"), on=["market_id", "selection_id"],
                how="inner")
    if window is not None:                            # the bookmakers' snapshot near the mark, the race after it
        at = pd.Timestamp(f"{days[0]} {mark.split('_')[1]}", tz="Europe/London").tz_convert("UTC")
        first = d["off"] - pd.to_timedelta(d["mins_first"], unit="m")
        d = d[((first - at).abs() <= pd.Timedelta(minutes=window)) & (d["off"] > at + pd.Timedelta(minutes=window))]
        print(f"runners whose bookmaker snapshot fell within {window} min of {mark} and ran after it: {len(d)}")
    for c in ("back1", "lay1"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["bk_am_median", "back1", "lay1", "bsp"])
    d = d[(d["back1"] > 1) & (d["lay1"] > 1)]
    rows = []
    if not d.empty:
        short = d["bk_am_median"] < d["back1"]            # the bookmakers shorter than Betfair's best back: back
        long_ = d["bk_am_median"] > d["lay1"]             # longer than Betfair's best lay: lay
        d["gap"] = np.where(short, np.log(d["back1"] / d["bk_am_median"]),
                            np.where(long_, np.log(d["bk_am_median"] / d["lay1"]), 0.0))
        d["side"] = np.where(short, "back", np.where(long_, "lay", "none"))
        d["pnl"] = np.where(short, d["back1"] / d["bsp"] - 1, np.where(long_, 1 - d["lay1"] / d["bsp"], 0.0))
        d["risk"] = np.where(short, 1.0, np.where(long_, d["lay1"] - 1, 0.0))
        d["net"] = d["pnl"] - COMM * d["pnl"].clip(lower=0)
        t = d[d["side"] != "none"]
        bins = [0, 0.05, 0.10, 0.20, np.inf]
        t["gap_band"] = pd.cut(t["gap"], bins, labels=["0-5%", "5-10%", "10-20%", "20%+"])
        tab = t.groupby(["side", "gap_band"], observed=True).agg(bets=("net", "size"), net=("net", "sum"),
                                                                    risk=("risk", "sum"),
                                                                    right_way=("pnl", lambda x: (x > 0).mean()))
        tab["roi_on_risk"] = (tab["net"] / tab["risk"]).round(4)
        print(tab.round(3).to_string())
        print(f"runners with a two-sided book at the mark and a morning bookmaker price: {len(d)}; signalled "
              f"{len(t)} ({(t['side'] == 'back').sum()} backs, {(t['side'] == 'lay').sum()} lays); the rest sat "
              f"inside Betfair's spread")
        rows = [{"bets": int(len(t)), "net": round(float(t["net"].sum()), 3), "risk": round(float(t["risk"].sum()), 3)}]
    else:
        print("no data found: no runner with an 08:00 two-sided book and a morning bookmaker price")

    # b. across exchanges: Matchbook's morning price against the Betfair BSP
    print(f"\n-- 3{tag}b. across exchanges: back at Matchbook's morning price (Oddschecker MA), lay at the Betfair BSP, "
          "where it is over Betfair's best back at the mark (Betfair's 2% only; Matchbook's commission not applied)")
    ma = morning[morning["bookmaker"] == "MA"].groupby(["race_url", "key_horse"])["price"].max().rename("ma_am")
    x = d.join(ma, on=["race_url", "key_horse"]) if not d.empty else d
    if not x.empty and x["ma_am"].notna().any():
        x = x.dropna(subset=["ma_am"])
        over = x[x["ma_am"] > x["back1"]]
        pnl = over["ma_am"] / over["bsp"] - 1
        net = pnl - COMM * pnl.clip(lower=0)
        print(f"runners with a Matchbook morning price {len(x)}; over Betfair's best back at the mark {len(over)} "
              f"({len(over) / len(x):.1%}); their mean per GBP1 backed {net.mean():+.4f}, right way {(pnl > 0).mean():.1%}")
    else:
        print("no data found: no Matchbook (MA) morning price joined")
    return rows


def main() -> int:
    print("== Parse x Betfair: the bookmakers' morning prices against the BSP")
    if not CAPTURE.exists():
        print("no data found: data/parse/oddschecker_prices.parquet is not in the checkout")
        return 0
    oc = pd.read_parquet(CAPTURE)
    oc = oc.dropna(subset=["price", "off_utc"])
    oc["fetched"] = pd.to_datetime(oc["fetched_utc"], utc=True)
    oc["off"] = pd.to_datetime(oc["off_utc"], utc=True)
    oc = oc[oc["fetched"] < oc["off"]]
    oc["key_course"] = oc["course"].map(norm_name)
    oc["key_horse"] = oc["horse"].map(norm_name)
    print(f"oddschecker rows {len(oc):,}; days {sorted(oc['race_date'].unique())}; races "
          f"{oc['race_url'].nunique()}; runners {oc.groupby('race_url')['horse'].nunique().sum()}; "
          f"bookmakers {oc['bookmaker'].nunique()}")

    # the first snapshot of each race (the morning book) and the one nearest T-15
    snap = oc.groupby(["race_url", "fetched"]).size().reset_index()[["race_url", "fetched"]]
    first = snap.groupby("race_url")["fetched"].min().rename("first_fetched")
    oc = oc.join(first, on="race_url")
    morning = oc[oc["fetched"] == oc["first_fetched"]]
    oc["mins"] = (oc["off"] - oc["fetched"]).dt.total_seconds() / 60
    t15 = oc[(oc["mins"] > 5) & (oc["mins"] <= 25)]
    t15 = t15[t15["fetched"] == t15.groupby("race_url")["fetched"].transform("max")]

    def agg(df, tag):
        g = df.groupby(["race_url", "race_date", "key_course", "off", "key_horse", "horse"])["price"]
        return pd.DataFrame({f"{tag}_best": g.max(), f"{tag}_median": g.median(), f"{tag}_books": g.size()})

    runners = agg(morning, "bk_am").join(agg(t15, "bk_t15"), how="left").reset_index()
    runners["mins_first"] = runners["race_url"].map(
        (oc.drop_duplicates("race_url").set_index("race_url")["off"]
         - oc.drop_duplicates("race_url").set_index("race_url")["first_fetched"]).dt.total_seconds() / 60)
    print(f"runners with a morning book {len(runners):,}; median minutes before the off of the first snapshot "
          f"{runners['mins_first'].median():.0f}")

    if not Path(DB).exists():
        print(f"no data found: {DB}")
        return 0
    from sources.open_close import read_horse_marks
    days = sorted(runners["race_date"].unique())
    bf = read_horse_marks(DB, date_from=days[0], date_to=days[-1])
    bf = bf[bf["market_type"] == "WIN"]
    conn = sqlite3.connect(DB)
    cat = pd.read_sql_query("SELECT DISTINCT market_id, venue, market_start_utc FROM betfair_live_markets "
                            "WHERE market_type = 'WIN'", conn)
    conn.close()
    bf = bf.merge(cat, on="market_id", how="left")
    bf["key_course"] = bf["venue"].map(norm_name)
    bf["key_horse"] = bf["selection_name"].map(norm_name)
    bf["off"] = pd.to_datetime(bf["market_start_utc"], utc=True)
    print(f"betfair WIN runners on those days {len(bf):,}, with a BSP {bf['bsp'].notna().sum():,}, with an 08:00 mid "
          f"{bf['open_price'].notna().sum():,}")
    j = runners.merge(bf[["key_course", "off", "key_horse", "open_price", "p_t15", "close_price", "bsp", "result",
                          "market_id", "selection_id"]], on=["key_course", "off", "key_horse"], how="left")
    print(f"joined: {j['market_id'].notna().sum():,} of {len(j):,} runners matched a Betfair WIN runner "
          f"(by course, off time and name); with a BSP {j['bsp'].notna().sum():,}")
    j = j.dropna(subset=["bsp"])
    j = j[j["bsp"] > 1.0]
    if j.empty:
        print("no data found: no runner with both a morning book and a BSP (the night's load may not hold the day)")
        return 0

    # 1. which morning price sits nearest the BSP
    print("\n-- 1. log(BSP / morning price): median, IQR and sd, by price")
    rows = []
    for col, lab in (("bk_am_median", "bookmakers' median, morning"), ("bk_am_best", "bookmakers' best, morning"),
                     ("open_price", "Betfair 08:00 mid"), ("bk_t15_median", "bookmakers' median, T-15"),
                     ("p_t15", "Betfair T-15 mid")):
        d = j.dropna(subset=[col])
        if d.empty:
            continue
        x = np.log(d["bsp"] / d[col])
        rows.append({"price": lab, "n": len(d), "median": round(x.median(), 3), "q25": round(x.quantile(.25), 3),
                     "q75": round(x.quantile(.75), 3), "sd": round(x.std(), 3),
                     "abs_mean": round(x.abs().mean(), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n-- the log loss of each book's normalised chances against the result (races with every runner priced)")
    ll = []
    for col, lab in (("bk_am_median", "bookmakers' median, morning"), ("open_price", "Betfair 08:00 mid"),
                     ("bsp", "BSP")):
        d = j.dropna(subset=[col, "result"]).copy()
        full = d.groupby("race_url")[col].transform("size") == d.groupby("race_url")["horse"].transform("size")
        d = d[full]
        d["p"] = (1 / d[col]) / d.groupby("race_url")[col].transform(lambda s: (1 / s).sum())
        w = d[d["result"] == 1]
        if len(w):
            ll.append({"book": lab, "races": d["race_url"].nunique(), "log_loss_winner": round(-np.log(w["p"]).mean(), 4)})
    print(pd.DataFrame(ll).to_string(index=False) if ll else "no data found")

    # 2. does the bookmakers' morning price add to the Betfair mid
    print("\n-- 2. OLS of log BSP on the morning prices (in sample; one day is descriptive only)")
    d = j.dropna(subset=["bk_am_median", "open_price"])
    out = {}
    if len(d) >= 30:
        y = np.log(d["bsp"].to_numpy())
        for lab, cols in (("Betfair 08:00 mid", ["open_price"]), ("bookmakers' median", ["bk_am_median"]),
                          ("both", ["open_price", "bk_am_median"])):
            X = np.column_stack([np.ones(len(d))] + [np.log(d[c].to_numpy()) for c in cols])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            out[lab] = {"r2": round(float(r2), 4), "coef": [round(float(v), 3) for v in b]}
        print(json.dumps(out, indent=1))
    else:
        print(f"no data found: {len(d)} runners with both prices")

    rows = exchange_trade(j, morning, days, "clock_08:00", "a",
                          "LOOK-AHEAD: the bookmakers' snapshot came hours after 08:00; for reference only.")
    rows_12 = exchange_trade(j, morning, days, "clock_12:00", "b",
                             "The tradable version: the bookmakers' snapshot within 30 minutes of the mark.", window=30)
    print("\n=== JSON ===")
    print(json.dumps({"runners_joined": int(len(j)), "days": days, "ols": out, "exchange_trade_0800_lookahead": rows, "exchange_trade_1200": rows_12}))
    print("=== JSON ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
