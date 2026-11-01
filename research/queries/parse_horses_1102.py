"""Oddschecker's bookmaker prices against the Betfair books and the BSP (the owner's ask, 1 Nov 2026; read-only).

Reads the Parse capture committed beside the code (data/parse/oddschecker_prices.parquet: every GB/IE runner's price at
about 25 bookmakers, the first snapshot of the day and T-120/T-60/T-15) and horse_racing.db (betfair_live_markets,
betfair_live_marks: the recorder's books and the settled BSP). For each runner:

  - the bookmakers' morning price: the best and the median across the books at the first snapshot;
  - the Betfair 08:00 UK mid (the recorder's open) and the T-15 mid; the BSP; the result.

It asks three things, on the days held (one day at first: descriptive, not a test):

  1. which morning price is nearer the BSP: the bookmakers' median, their best, or the Betfair 08:00 mid (median and
     spread of log(BSP / price), and the log loss of each book's normalised chances against the result);
  2. does the bookmakers' morning price add to the Betfair mid in forecasting the BSP (OLS of log BSP on both);
  3. the trade that needs no winner: back with the best bookmaker in the morning, lay at the BSP on Betfair. Per
     GBP1 backed at B and B/BSP laid: B/BSP - 1 if the horse wins, (B/BSP)(1 - c) - 1 if it loses (c the 2%
     commission on the lay's winnings). Reported by how far the morning best price sat above the 08:00 mid. It
     ignores what bookmakers do to accounts that take early prices, which is the binding limit in practice.
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
                          "market_id"]], on=["key_course", "off", "key_horse"], how="left")
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

    # 3. back with the best bookmaker in the morning, lay at the BSP
    print("\n-- 3. back at the bookmakers' best morning price, lay at the BSP: per GBP1 backed, after 2% on the lay")
    d = j.dropna(subset=["bk_am_best", "result"]).copy()
    r = d["bk_am_best"] / d["bsp"]
    d["pnl"] = np.where(d["result"] == 1, r - 1, r * (1 - COMM) - 1)
    d["gap"] = np.where(d["open_price"].notna(), d["bk_am_best"] / d["open_price"] - 1, np.nan)
    bins = [-np.inf, 0.0, 0.05, 0.10, 0.20, np.inf]
    labs = ["best <= 08:00 mid", "0-5% over", "5-10% over", "10-20% over", "20%+ over"]
    d["band"] = pd.cut(d["gap"], bins, labels=labs)
    tab = d.groupby("band", observed=True)["pnl"].agg(["size", "mean", "sum"]).round(3)
    print(tab.to_string())
    print(f"every runner: n {len(d)}, mean {d['pnl'].mean():+.3f} a GBP1, share with the best price over the BSP "
          f"{(r > 1).mean():.1%}")
    print("\n=== JSON ===")
    print(json.dumps({"runners_joined": int(len(j)), "days": days, "ols": out,
                      "trade_mean": round(float(d["pnl"].mean()), 4) if len(d) else None}))
    print("=== JSON ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
