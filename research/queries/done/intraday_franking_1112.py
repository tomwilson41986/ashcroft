"""Intraday franking, the first test (reports/discovery_1111.md 4.9; read-only; an event study).

A rival's result earlier the same day is scheduled news. If horse A met horse B in either's last three runs and B runs
at least 20 minutes before A today, does A's price move after B's race the way B's result says, and does the move go
on to the BSP after the first five minutes?

For each live day from 1 Oct: today's runners (race_results joined to the live record by race_results_id, the off
from Betfair's catalogue); the links (A, B) where both ran in one past race among either's last three runs;
w = exp(-|beaten-lengths gap| / 3) x 0.85^(runs ago, the smaller); B's surprise today s = (BSP rank - finishing rank)
/ (ranked runners - 1); x_A = sum of w s over A's links whose race is off at least 20 minutes before A's. A's implied
chance (the middle of back and lay) from the recorded books: the last read at or before B's off, the first read 5
and 15 minutes after it, and the BSP. With several linked rivals the last one's off is used.
Arms: the links; placebo links (each B replaced by a runner of another earlier race that never met A, seeded).
Readings, days clustered by bootstrap: the slope of ln p(BSP) - ln p(+5 min) on x (the registered pass: positive, 90%
interval above zero, the placebo's not); and beside it the slopes of the +5 and +15 minute moves. 1 trial.
"""
import gzip
import io
import os
import sqlite3
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
FIRST, LAST = date(2026, 10, 1), date(2026, 11, 10)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
rng = np.random.default_rng(1112)


def books(D):
    out = []
    for n in ("books_trader", "books"):
        try:
            raw = s3.get_object(Bucket=bucket, Key=f"betfair_live/{D}/{n}.csv.gz")["Body"].read()
        except Exception:  # noqa: BLE001
            continue
        out.append(pd.read_csv(io.BytesIO(gzip.decompress(raw)), low_memory=False, on_bad_lines="skip",
                               dtype={"market_id": str},
                               usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "status", "inplay",
                                                       "runner_status", "back1", "lay1")))
    if not out:
        return pd.DataFrame()
    b = pd.concat(out, ignore_index=True)
    b["t"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
    for c in ("back1", "lay1", "selection_id"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b[(b.status == "OPEN") & b.inplay.astype(str).str.lower().isin(["0", "0.0", "false"])
          & (b.runner_status == "ACTIVE") & (b.back1 > 1) & (b.lay1 > 1) & b.t.notna()]
    b["p"] = (1 / b.back1 + 1 / b.lay1) / 2
    return b[["market_id", "selection_id", "t", "p", "back1"]]


hist = pd.read_sql_query(
    "SELECT id, race_date, race_time, track, horse_name, placing_numerical, total_dst_bt, bfsp FROM race_results "
    "WHERE race_date >= '2025-06-01'", con)
hist["d"] = hist.race_date.str[:10]
hist["race"] = hist.d + "|" + hist.track.astype(str) + "|" + hist.race_time.astype(str)
for c in ("placing_numerical", "total_dst_bt", "bfsp"):
    hist[c] = pd.to_numeric(hist[c], errors="coerce")
cat = pd.read_sql_query("SELECT race_date, market_id, selection_id, market_type, market_start_utc, race_results_id "
                        "FROM betfair_live_markets WHERE race_results_id IS NOT NULL", con)
cat = cat[cat.market_type.fillna("WIN").str.upper().eq("WIN")].drop_duplicates("race_results_id")
cat["off"] = pd.to_datetime(cat.market_start_utc, utc=True, errors="coerce")
fin = pd.read_sql_query("SELECT market_id, selection_id, bsp FROM betfair_live_marks WHERE mark = 'final'", con)
fin["bsp"] = pd.to_numeric(fin.bsp, errors="coerce")

rows, d = [], FIRST
while d <= LAST:
    D = d.isoformat()
    d += timedelta(days=1)
    today = hist[hist.d == D].merge(cat[["race_results_id", "market_id", "selection_id", "off"]], left_on="id",
                                    right_on="race_results_id")
    if today.empty:
        continue
    r = today[(today.bfsp > 1) & today.placing_numerical.notna()].copy()
    r["rk_m"] = r.groupby("race").bfsp.rank()
    r["rk_f"] = r.groupby("race").placing_numerical.rank()
    r["n"] = r.groupby("race").bfsp.transform("size")
    today = today.merge(r.assign(s=(r.rk_m - r.rk_f) / (r.n - 1).clip(lower=1))[["id", "s"]], on="id", how="left")
    past = hist[(hist.d < D) & hist.horse_name.isin(set(today.horse_name))].sort_values("d")
    past["ago"] = past.groupby("horse_name").cumcount(ascending=False)
    past = past[past.ago < 3]
    pr = past.merge(past, on="race", suffixes=("_a", "_b"))
    pr = pr[pr.horse_name_a != pr.horse_name_b]
    pr["w"] = np.exp(-(pr.total_dst_bt_a - pr.total_dst_bt_b).abs().fillna(3.0) / 3.0) * 0.85 ** pr[["ago_a", "ago_b"]].min(axis=1)
    pr = pr.groupby(["horse_name_a", "horse_name_b"]).w.max().reset_index()
    t = today[["horse_name", "race", "off", "market_id", "selection_id", "s"]]
    lk = pr.merge(t.add_suffix("_a"), on="horse_name_a").merge(t.add_suffix("_b"), on="horse_name_b")
    lk = lk[(lk.race_a != lk.race_b) & (lk.off_a - lk.off_b >= pd.Timedelta(minutes=20)) & lk.s_b.notna()]
    if lk.empty:
        continue
    lk["arm"] = "links"
    linked = set(zip(pr.horse_name_a, pr.horse_name_b))
    pool = t[t.s.notna()]
    pl = []
    for a in lk.itertuples():
        c = pool[(pool.race != a.race_a) & (pool.race != a.race_b) & (a.off_a - pool.off >= pd.Timedelta(minutes=20))]
        c = c[[(a.horse_name_a, h) not in linked for h in c.horse_name]]
        if len(c):
            b = c.iloc[int(rng.integers(len(c)))]
            pl.append({**a._asdict(), "horse_name_b": b.horse_name, "race_b": b.race, "off_b": b.off, "s_b": b.s})
    both = pd.concat([lk, pd.DataFrame(pl).assign(arm="placebo").drop(columns=["Index"], errors="ignore")], ignore_index=True)
    bk = books(D)
    if bk.empty:
        continue
    for arm, g in both.groupby("arm"):
        ev = g.assign(ws=g.w * g.s_b).groupby(["horse_name_a", "market_id_a", "selection_id_a", "off_a"]).agg(
            x=("ws", "sum"), links=("w", "size"), t0=("off_b", "max")).reset_index()
        for e in ev.itertuples():
            q = bk[(bk.market_id == e.market_id_a) & (bk.selection_id == e.selection_id_a)].sort_values("t")
            pre = q[q.t <= e.t0]
            p5 = q[(q.t >= e.t0 + pd.Timedelta(minutes=5)) & (q.t < e.off_a)]
            p15 = q[(q.t >= e.t0 + pd.Timedelta(minutes=15)) & (q.t < e.off_a)]
            if pre.empty or p5.empty or (e.t0 - pre.t.iloc[-1]) > pd.Timedelta(minutes=10):
                continue
            rows.append({"day": D, "arm": arm, "x": e.x, "links": e.links, "p0": pre.p.iloc[-1], "p5": p5.p.iloc[0],
                         "p15": p15.p.iloc[0] if len(p15) else np.nan, "back5": p5.back1.iloc[0],
                         "market_id": e.market_id_a, "selection_id": e.selection_id_a})
E = pd.DataFrame(rows)
if E.empty:
    raise SystemExit("no data found: no linked runner with a price either side of a rival's race")
E = E.merge(fin, on=["market_id", "selection_id"], how="left")
E = E[E.bsp > 1].copy()
E["m5"], E["m15"], E["rest"] = np.log(E.p5 / E.p0), np.log(E.p15 / E.p0), np.log((1 / E.bsp) / E.p5)
print(f"{E.day.min()} to {E.day.max()}: events by arm {E.arm.value_counts().to_dict()}; days {E.day.nunique()}; "
      f"x: sd {E[E.arm == 'links'].x.std():.3f}, links a runner median {E[E.arm == 'links'].links.median():.0f}")


def slope(g, y):
    g = g.dropna(subset=[y, "x"])
    days = sorted(g.day.unique())
    by = {k: v for k, v in g.groupby("day")}
    f = lambda z: float(np.polyfit(z.x, z[y], 1)[0])  # noqa: E731
    bs = []
    for _ in range(2000):
        z = pd.concat([by[k] for k in rng.choice(days, len(days))])
        bs.append(f(z))
    lo, hi = np.percentile(bs, [5, 95])
    return f(g), lo, hi, len(g)


ok = {}
for arm in ("links", "placebo"):
    g = E[E.arm == arm]
    print(f"\n{arm}:")
    for y, label in (("m5", "the move to +5 min"), ("m15", "the move to +15 min"), ("rest", "from +5 min to the BSP")):
        b, lo, hi, n = slope(g, y)
        print(f"  {label:24s} slope on x {b:+.4f} (90% {lo:+.4f} to {hi:+.4f}), {n:,} runners")
        ok[(arm, y)] = (lo, hi)
top = E[(E.arm == "links") & (E.x >= E[E.arm == "links"].x.quantile(.9))]
clv = top.back5 / top.bsp - 1
print(f"\npaper: backing the top tenth by x at the +5 min back price, closed at the BSP: {len(top)} bets, mean CLV "
      f"{clv.mean():+.2%} before commission, {0.98 * clv.mean() if clv.mean() > 0 else clv.mean():+.2%} after 2%")
p = ok[("links", "rest")][0] > 0 and not ok[("placebo", "rest")][0] > 0
print("verdict:", "PASS" if p else "does not pass (the idea closes)")
