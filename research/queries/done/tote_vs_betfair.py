"""The Tote's prices against Betfair's, race by race (read-only; the owner's ask of 6 Oct 2026: "access the tote
prices and compare to betfair"). TOTE_DAY picks the day (first run 6 Oct, the recorder's first, from 12:56 UTC;
the first whole day is 7 Oct).

The Tote side is tote_recorder.py's record (s3 tote/live/<day>/): each GB/IE race's win and place pools at T-60
to T+5 (each runner's approximate dividend from the pool, base, and the figure the site shows with the Tote
Guarantee, shown) and the declared dividends after the race (win, place, exacta, trifecta, swinger). The Betfair
side is the day's market record (s3 betfair_live/<day>/: the recorder's and the trader's books and catalogues):
each runner's best lay at the minute of each Tote mark, and the BSP. Matched by course, off time and cloth number
(else the horse's name).

  1. Before the off, the pair a trader could take: back on the Tote, lay on Betfair at the same minute. The Tote's
     return over the lay's break-even, edge = D / b(L) - 1 with b(L) = 1 + (L - 1)/0.98 (tote_compare.edge), by
     mark: with the pool's dividend as it stands (base) and with the guarantee the site shows (shown: direct bets
     only). A Tote bet is paid at the dividend the pool declares after the off, not the one shown at the mark, so:
  2. the move from each mark to the off: the runner's base dividend at T+5 (the pool closed) over its dividend at
     the mark; and the pairs taken where the edge at a mark cleared 5%, settled at the declared dividend.
  3. At the close: the near-final dividend (T+5) against the BSP for every runner, and the winners' declared
     dividends against the BSP (the pool's and the guaranteed).
  4. The exotics: each declared dividend D against the dividend fair under the BSP's chances (Harville at 1, 1:
     tote_compare.fair_dividends), as P x D, the return of a unit spread over every combination in proportion to
     its chance (1 the fair price; the deductions are 25% to 30%). The swinger's three pairs are read in the order
     1st+2nd, 1st+3rd, 2nd+3rd, checked against the other order by how the dividends follow the chances.
Pool totals: the answers before the race are in pence, the declared results in pounds.
"""

from __future__ import annotations

import gzip
import io
import os
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import tote_compare as tc  # noqa: E402
from model.ordering import exacta_matrix  # noqa: E402
from tote_recorder import read_lines, runner_rows  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 150)
DAY = os.environ.get("TOTE_DAY") or "2026-10-06"
C = 0.02
PRE = (-60, -30, -15, -10, -5, -3, -2, -1)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")


def tote_file(kind):
    tmp = Path(tempfile.mkdtemp()) / f"{kind}.jsonl.gz"
    try:
        s3.download_file(bucket, f"tote/live/{DAY}/{kind}.jsonl.gz", str(tmp))
    except Exception as exc:
        print(f"tote {kind}: not read ({type(exc).__name__})")
        return []
    return read_lines(tmp)


def betfair_csv(name):
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"betfair_live/{DAY}/{name}.gz")
    except Exception:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(gzip.decompress(obj["Body"].read())), dtype={"market_id": str}, on_bad_lines="skip")


# ------------------------------------------------------------------------------------------------------- the data
detail, results = tote_file("detail"), tote_file("results")
rows = runner_rows(detail)
declared_runners, declared_pools = tc.declared(results)
cat = pd.concat([betfair_csv("markets.csv"), betfair_csv("markets_trader.csv")], ignore_index=True)
books = pd.concat([betfair_csv("books.csv"), betfair_csv("books_trader.csv")], ignore_index=True)
print(f"== {DAY}: Tote: {len(detail):,} pool answers ({rows.race_id.nunique() if len(rows) else 0} races), "
      f"declared dividends for {declared_runners.race_id.nunique() if len(declared_runners) else 0} races; "
      f"Betfair: {cat.market_id.nunique() if len(cat) else 0} markets catalogued, {len(books):,} book rows")
if not len(rows) or not len(cat) or not len(books):
    print("nothing to compare: a side of the record is missing")
    raise SystemExit(0)
cat["market_type"] = cat["market_type"].fillna("").replace("", "WIN")       # the trader's catalogue reads win
cat = cat.drop_duplicates(["market_id", "selection_id"])
books["t"] = pd.to_datetime(books["polled_utc"], utc=True, errors="coerce")
books["selection_id"] = pd.to_numeric(books["selection_id"], errors="coerce")
books = books.dropna(subset=["t", "selection_id"])
books["selection_id"] = books["selection_id"].astype(int)
for c in ("lay1", "back1", "sp_actual", "lay1_size"):
    books[c] = pd.to_numeric(books[c], errors="coerce")
bsp = books.dropna(subset=["sp_actual"]).groupby(["market_id", "selection_id"])["sp_actual"].last()
# the record's BSP is read as each market reconciles at the off (from 7 Oct); before, and wherever it is missing, the
# price files' (betfair_prices, loaded nightly the day after), by the runner's selection id and the market type
bsp_files = {}
try:
    import sqlite3
    with sqlite3.connect(os.environ.get("DB_PATH") or "horse_racing.db") as conn:
        pf = pd.read_sql_query("SELECT selection_id, market_type, bsp FROM betfair_prices WHERE race_date = ? AND bsp > 1",
                               conn, params=[DAY])
    bsp_files = {(str(t).lower(), int(sid)): float(v) for sid, t, v in zip(pf.selection_id, pf.market_type, pf.bsp)}
except Exception as exc:
    print(f"   the price files' BSPs not read ({type(exc).__name__})")
mtype = cat.drop_duplicates("market_id").set_index("market_id")["market_type"].astype(str).str.lower().to_dict()
print(f"   BSPs: {len(bsp):,} runners in the day record, {len(bsp_files):,} in the price files for {DAY}")


def bsp_of(market_id, selection_id):
    v = bsp.get((market_id, selection_id))
    if v is None or not np.isfinite(v):
        v = bsp_files.get((mtype.get(market_id, "win"), int(selection_id)))
    return v

rows["race_id"] = rows["race_id"].astype(str)
rows["t"] = pd.to_datetime(rows["polled_utc"], utc=True, errors="coerce")
races = rows.rename(columns={"meeting": "track"})[["race_id", "track", "post_utc"]].drop_duplicates("race_id")
win_map = tc.match_races(races, cat, "WIN")
place_map = tc.match_races(races, cat, "PLACE")
print(f"   Tote races matched to a Betfair win market {len(win_map)} of {len(races)}, to a place market "
      f"{len(place_map)}")


def attach(frame, race_map):
    """The Betfair market and selection of each Tote runner row, its best lay at the minute of the row (the
    nearest book within 90 seconds) and its BSP."""
    f = frame.copy()
    f["market_id"] = f["race_id"].map(race_map)
    f["selection_id"] = tc.selection_ids(f, cat, race_map)
    f = f.dropna(subset=["market_id", "selection_id", "t"])
    f["selection_id"] = f["selection_id"].astype(int)
    b = books[books["market_id"].isin(f["market_id"].unique())][["market_id", "selection_id", "t", "lay1", "back1",
                                                                 "lay1_size"]]
    b = b.sort_values("t")
    f = pd.merge_asof(f.sort_values("t"), b, on="t", by=["market_id", "selection_id"], direction="nearest",
                      tolerance=pd.Timedelta(seconds=90))
    f["bsp"] = [bsp_of(m, s) for m, s in zip(f["market_id"], f["selection_id"])]
    return f


win = attach(rows[rows["pool"].eq("WIN") & ~rows["scratched"]], win_map)
plc = attach(rows[rows["pool"].eq("PLACE") & ~rows["scratched"]], place_map)
# the Tote record restarts with the trader's job (seven times on 7 Oct), so a mark can be answered twice: the last
win = win.sort_values("t").drop_duplicates(["race_id", "cloth", "mark"], keep="last")
plc = plc.sort_values("t").drop_duplicates(["race_id", "cloth", "mark"], keep="last")
# a place comparison needs the same terms: the Tote's places (the pool's numPositions) against the places the Betfair
# market pays (numberOfWinners on its book, recorded from 7 Oct; before, taken to be the same)
if "number_of_winners" in books.columns and len(plc):
    paid = pd.to_numeric(books["number_of_winners"], errors="coerce")
    bf_places = books.assign(_n=paid).dropna(subset=["_n"]).groupby("market_id")["_n"].last()
    known = plc["market_id"].map(bf_places)
    clash = known.notna() & pd.to_numeric(plc["places"], errors="coerce").ne(known)
    print(f"   place pools with Betfair's terms recorded: {known.notna().mean():.0%} of rows; terms differing, left "
          f"out: {int(clash.sum())} rows")
    plc = plc[~clash]
else:
    print("   place terms: Betfair's not recorded for this day; taken to be the Tote's")
print(f"   win-pool rows joined {win.lay1.notna().sum():,} with a lay at the minute, {win.bsp.notna().sum():,} with a "
      f"BSP; place-pool rows {plc.lay1.notna().sum():,} and {plc.bsp.notna().sum():,}")

# the declared dividends, by race and cloth
dec = declared_runners.copy()
if len(dec):
    dec["race_id"] = dec["race_id"].astype(str)
    won = dec[dec["finish"].eq(1)].set_index(["race_id", "cloth"])


def summary(f, col):
    ok = f.dropna(subset=[col, "lay1"])
    ok = ok[(ok[col] > 1) & (ok["lay1"] > 1)]
    e = pd.Series(tc.edge(ok[col], ok["lay1"], C), index=ok.index)
    return pd.Series({"runners": len(ok), "edge_median": e.median(), "over_0": (e > 0).mean(),
                      "over_5pc": (e > 0.05).mean(), "lay_size_median": ok["lay1_size"].median()})


# --------------------------------------------------------------------------- 1. before the off: the pair at each mark
for name, f in (("win", win), ("place", plc)):
    if not len(f):
        continue
    print(f"\n== 1. {name} pool before the off: the Tote's return over the Betfair lay's break-even at the same minute")
    tab = []
    for m in PRE:
        g = f[f["mark"].eq(m)]
        if len(g):
            tab.append(pd.concat([summary(g, "base").add_prefix("base_"), summary(g, "shown").add_prefix("shown_")])
                       .rename(m))
    if tab:
        print(pd.DataFrame(tab).round(3).to_string())

# ------------------------------------------------------------------- 2. the move to the off, and the pairs settled
final = win[win["mark"].eq(5)].set_index(["race_id", "cloth"])["base"]
if len(final):
    print("\n== 2. the win pool's dividend at the off (T+5, the pool closed) over its dividend at each mark")
    mv = []
    for m in PRE:
        g = win[win["mark"].eq(m)].set_index(["race_id", "cloth"])["base"]
        r = (final.reindex(g.index) / g).dropna()
        r = r[np.isfinite(r)]
        if len(r):
            mv.append({"mark": m, "runners": len(r), "median": r.median(), "q10": r.quantile(0.1),
                       "q90": r.quantile(0.9), "down_10pc_or_more": (r <= 0.9).mean()})
    print(pd.DataFrame(mv).round(3).to_string(index=False))

if len(dec) and len(win):
    print("\n== 2b. pairs taken where the edge at the mark cleared 5% (back on the Tote, lay at the same minute), "
          "settled at the declared dividend: per unit backed")
    out = []
    for m in PRE:
        g = win[win["mark"].eq(m) & (win["base"] > 1) & (win["lay1"] > 1)].copy()
        g = g[tc.edge(g["base"], g["lay1"], C) > 0.05]
        if not len(g):
            continue
        keys = list(zip(g["race_id"], g["cloth"]))
        g["won"] = [k in won.index for k in keys]
        g["d_pool"] = [won["win_base"].get(k, np.nan) for k in keys]
        g["d_paid"] = [won["win"].get(k, np.nan) for k in keys]
        settled = g[g["race_id"].isin(dec["race_id"])]
        for how in ("d_pool", "d_paid"):
            r = tc.hedged_return(settled[how].fillna(0.0), settled["lay1"], settled["won"], C)
            out.append({"mark": m, "settled_at": "pool's dividend" if how == "d_pool" else "with the guarantee",
                        "pairs": len(settled), "winners": int(settled["won"].sum()),
                        "per_unit": float(np.mean(r)) if len(r) else np.nan})
    print(pd.DataFrame(out).round(3).to_string(index=False) if out else "   none cleared 5% at any mark")

# ----------------------------------------------------------------------------------------------- 3. at the close
close = win[win["mark"].eq(5) & (win["base"] > 1) & (win["bsp"] > 1)].copy()
if len(close):
    close["edge_bsp"] = tc.edge(close["base"], close["bsp"], C)
    close["p_bsp"] = close.groupby("race_id")["bsp"].transform(lambda s: (1 / s) / (1 / s).sum())
    close["ev"] = close["p_bsp"] * (close["base"] - tc.breakeven(close["bsp"], C))
    bands = pd.cut(close["bsp"], [1, 3, 6, 11, 21, 1001])
    print("\n== 3. the near-final win dividend (T+5) against the BSP, every runner: the Tote's return over the "
          "break-even of a lay at the BSP")
    print(close.groupby(bands, observed=True).agg(runners=("edge_bsp", "size"), edge_median=("edge_bsp", "median"),
                                                  over_0=("edge_bsp", lambda s: (s > 0).mean()),
                                                  tote_over_bsp=("base", "median")).round(3).to_string())
    print(f"   over all: median edge {close.edge_bsp.median():+.3f}, share over 0 {(close.edge_bsp > 0).mean():.1%}; "
          f"expected per unit under the BSP's chances where the edge is over 0: "
          f"{close.loc[close.edge_bsp > 0, 'ev'].mean():+.4f} ({(close.edge_bsp > 0).sum()} runners)")
if len(dec) and len(win):
    w = dec[dec["finish"].eq(1)].copy()
    w["market_id"] = w["race_id"].map(win_map)
    w["selection_id"] = tc.selection_ids(w, cat, win_map)
    w["bsp"] = [bsp_of(m, int(s)) if isinstance(m, str) and s is not None else None
                for m, s in zip(w["market_id"], w["selection_id"])]
    w = w.dropna(subset=["bsp", "win_base"])
    if len(w):
        w["pool_over_bsp"] = w["win_base"] / w["bsp"]
        w["paid_over_bsp"] = w["win"] / w["bsp"]
        w["sp_over_bsp"] = w["sp"] / w["bsp"]
        print(f"\n== 3b. the {len(w)} winners: the declared win dividend against the BSP (medians): the pool's "
              f"{w.pool_over_bsp.median():.3f}, paid with the guarantee {w.paid_over_bsp.median():.3f}, the industry "
              f"SP {w.sp_over_bsp.median():.3f}; the pool's dividend above the BSP in {(w.pool_over_bsp > 1).mean():.0%},"
              f" the guaranteed in {(w.paid_over_bsp > 1).mean():.0%}")
        print(w[["track", "post_utc", "name", "sp", "bsp", "win_base", "win"]].sort_values("post_utc")
              .to_string(index=False))
    wp = declared_pools[declared_pools["pool"].eq("WIN")]
    if len(wp) and len(dec):
        chk = wp.assign(race_id=wp["race_id"].astype(str)).set_index("race_id")
        ww = dec[dec["finish"].eq(1)].drop_duplicates("race_id").set_index("race_id")
        j = ww.join(chk[["dividends", "base_dividends"]], how="inner")
        first = j["dividends"].map(lambda v: v[0] if v else np.nan)
        print(f"   the win pool's listed dividend equals the winner's paid figure in {(first == j['win']).mean():.0%} "
              f"of {len(j)} races and its pool figure in {(first == j['win_base']).mean():.0%}")

# ----------------------------------------------------------------------------------------------- 4. the exotics
if len(dec) and len(declared_pools):
    print("\n== 4. the exotics: the declared dividend x its chance under the BSP (1 = the fair price)")
    out, sw_check = [], []
    for rid, g in dec.groupby("race_id"):
        mid = win_map.get(str(rid))
        if mid is None:
            continue
        sel = cat[cat["market_id"].eq(mid) & cat["market_type"].eq("WIN")]
        prices = np.array([bsp_of(mid, int(s)) or np.nan for s in sel["selection_id"]], float)
        if not len(prices) or not np.isfinite(prices).all():
            continue
        p = tc.market_probabilities(prices)
        idx = {int(s): i for i, s in enumerate(sel["selection_id"])}
        g = g.sort_values("finish")
        sids = tc.selection_ids(g, cat, {str(rid): mid})
        pos = [idx.get(int(s)) if s is not None else None for s in sids]
        if len(pos) < 3 or any(x is None for x in pos[:3]):
            continue
        a, b, c3 = pos[:3]
        pools = declared_pools[declared_pools["race_id"].astype(str).eq(str(rid))].set_index("pool")
        # a pool with no winning ticket declares no dividend (None): no fair-value reading for that race (9 Oct)
        if "EXACTA" in pools.index and pools.loc["EXACTA", "dividends"] and \
                pools.loc["EXACTA", "dividends"][0] is not None:
            d = pools.loc["EXACTA", "dividends"][0]
            out.append({"pool": "exacta", "race": rid, "pd": exacta_matrix(p)[a, b] * d})
        if "TRIFECTA" in pools.index and pools.loc["TRIFECTA", "dividends"] and \
                pools.loc["TRIFECTA", "dividends"][0] is not None:
            d = pools.loc["TRIFECTA", "dividends"][0]
            out.append({"pool": "trifecta", "race": rid, "pd": tc.trifecta_tensor(p)[a, b, c3] * d})
        if "SWINGER" in pools.index and len(pools.loc["SWINGER", "dividends"] or []) == 3 and \
                all(x is not None for x in pools.loc["SWINGER", "dividends"]):
            m = tc.swinger_matrix(p)
            d12, dx, dy = pools.loc["SWINGER", "dividends"]
            for order, pairs in (("12,13,23", [(a, b), (a, c3), (b, c3)]), ("12,23,13", [(a, b), (b, c3), (a, c3)])):
                ds = [d12, dx, dy]
                for (i, j), d in zip(pairs, ds):
                    sw_check.append({"order": order, "race": rid, "p": m[i, j], "d": d})
    if out:
        o = pd.DataFrame(out)
        print(o.groupby("pool")["pd"].agg(races="size", mean="mean", median="median").round(3).to_string())
    if sw_check:
        s = pd.DataFrame(sw_check)
        corr = s.groupby("order").apply(lambda x: x["p"].rank().corr((1 / x["d"]).rank()))
        best = corr.idxmax()
        print(f"   swinger pairs: the dividends follow the chances best read as {best} "
              f"(rank correlation of P with 1/D: {corr.round(3).to_dict()})")
        sb = s[s["order"].eq(best)]
        print(f"   swinger P x D over {sb.race.nunique()} races: mean {float((sb.p * sb.d).mean()):.3f}, "
              f"median {float((sb.p * sb.d).median()):.3f}")
