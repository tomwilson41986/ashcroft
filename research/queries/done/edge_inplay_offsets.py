"""Market edges, 1: in-play offsets on UK and Irish win and place markets, 2018 to 2026 Q1 (read-only).

The owner's research programme (2 Oct): evaluate each market we can trade for edges, before any model. This one
needs no model and no in-play feed: a back at the Betfair SP with a lay order resting into the race at a shorter
price (persistence into in-play, placed before the off), or a lay at the SP with a back resting at a longer price.
Whether the resting order would have matched is in Betfair's own price files: IPMIN and IPMAX, the shortest and
longest prices traded in running. A resting lay at L is taken once the market trades shorter than L (IPMIN < L);
a resting back at H once it trades longer than H (IPMAX > H).

If in-running prices were fair the chance of reaching a price would be what the odds say (from the SP's 1/B to 1/L
with chance L/B, by optional stopping on the price as a probability) and every offset would earn nothing before
commission. Where the market overshoots in running (a horse going too short, or too long, too often) an offset
earns; the empirical rate against the fair rate says where.

Back-to-lay (per GBP1 backed at the SP B): lay B/L at L; matched, it locks B/L - 1 whatever the result; unmatched,
the back stands. Lay-to-back (per GBP1 laid at B): back B/H at H; matched, it locks 1 - B/H; unmatched, the lay
stands. 2% commission on each runner's net winnings (the account's rate; a race with several runners pays on the
race's net, so per runner slightly overstates it). Offsets are set on the odds: L = 1 + (B - 1) f and
H = 1 + (B - 1) / f, rounded to Betfair's ladder away from the SP.

Cells (market, strategy, f, SP band, code, field) are chosen on 2018-22 and read on 2023 to March 2026; the holdout
(from 1 Apr 2026) is not read. Standard errors are by race.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 400)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
D_FROM, D_TO, SPLIT = "2018-01-01", "2026-03-31", "2023-01-01"
COMM = 0.02
FS = (0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2)
KEEP = ["country", "market_type", "race_date", "event_id", "menu_hint", "event_name", "selection_name", "win_lose",
        "bsp", "ipmin", "ipmax", "pp_vol", "ip_vol"]

# --- the files -------------------------------------------------------------------------------------------------
names = []
for name in bp.archive_index(s3, bucket):
    meta = bp.parse_file_name(name)
    if meta and date(2018, 1, 1) <= meta[2] <= date(2026, 4, 1):
        names.append(name)


def fetch(name):
    raw = s3.get_object(Bucket=bucket, Key=f"{bp.S3_PREFIX}/{name}")["Body"].read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        return bp.parse_file_text(text, name)[KEEP]
    except Exception as exc:                                   # a malformed day file is skipped and counted
        print(f"  {name}: {type(exc).__name__}")
        return None


with ThreadPoolExecutor(32) as ex:
    parts = [p for p in ex.map(fetch, names) if p is not None]
d = pd.concat(parts, ignore_index=True)
d = d[(d.race_date >= D_FROM) & (d.race_date <= D_TO) & (d.bsp > 1)].copy()
print(f"{len(names):,} UK/IE files read, {len(d):,} runners with an SP, {d.race_date.min()} to {d.race_date.max()} "
      f"({time.time() - t0:.0f}s)")
print(d.groupby(["country", "market_type"]).size().to_string())
print("in-play traded (IPMIN > 1):", d.groupby("market_type").ipmin.apply(lambda s: f"{(s > 1).mean():.1%}").to_dict())

d["mkt"] = d.country + d.market_type
d["won"] = (d.win_lose == 1).astype(float)
d["field"] = d.groupby(["mkt", "event_id"]).bsp.transform("size")
name_l = d.event_name.astype(str).str.lower()
d["code"] = np.where(name_l.str.contains(r"\bchs\b|\bhrd\b|nhf|chase|hurdle|bumper"), "jumps", "flat")
d["period"] = np.where(d.race_date < SPLIT, "train", "test")
d["sp_band"] = pd.cut(d.bsp, [1, 2, 3, 5, 8, 13, 21, 50, 1001], right=False,
                      labels=["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-50", "50+"])
d["field_band"] = pd.cut(d.field, [0, 7, 11, 15, 99], labels=["2-7", "8-11", "12-15", "16+"])

# Betfair's ladder: (from, to, increment)
LADDER = [(1.01, 2, 0.01), (2, 3, 0.02), (3, 4, 0.05), (4, 6, 0.1), (6, 10, 0.2), (10, 20, 0.5), (20, 30, 1.0),
          (30, 50, 2.0), (50, 100, 5.0), (100, 1000, 10.0)]


def to_tick(x: np.ndarray, up: bool) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1.01, 1000.0)
    out = x.copy()
    for lo, hi, inc in LADDER:
        m = (x >= lo) & (x < hi)
        k = (x[m] - lo) / inc
        k = np.ceil(k - 1e-9) if up else np.floor(k + 1e-9)
        out[m] = np.minimum(lo + k * inc, 1000.0)
    return np.round(out, 2)


B = d.bsp.to_numpy()
W = d.won.to_numpy()
ipmin = d.ipmin.fillna(0).to_numpy()
ipmax = d.ipmax.fillna(0).to_numpy()


def after_comm(pnl):
    return np.where(pnl > 0, pnl * (1 - COMM), pnl)


rows = []
cols = {}
for f in FS:
    L = to_tick(1 + (B - 1) * f, up=False)
    ok = L < B
    hit = ok & (ipmin > 1) & (ipmin < L)
    pnl = np.where(hit, B / L - 1, np.where(W == 1, B - 1, -1.0))
    cols[("back_lay", f)] = (after_comm(pnl), hit, np.where(ok, L / B, np.nan))
    H = to_tick(1 + (B - 1) / f, up=True)
    ok2 = (H > B) & (H < 1000)
    hit2 = ok2 & (ipmax > H)
    pnl2 = np.where(hit2, 1 - B / H, np.where(W == 1, -(B - 1), 1.0))
    fair2 = np.where(ok2, (1 - 1 / B) / (1 - 1 / H), np.nan)
    cols[("lay_back", f)] = (after_comm(pnl2), hit2, fair2)


def stats(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """n, ROI, match rate, fair rate and a by-race standard error for each group of `frame` (pnl, hit, fair, race)."""
    base = frame.groupby(by, observed=True).agg(n=("pnl", "size"), roi=("pnl", "mean"), match=("hit", "mean"),
                                                fair=("fair", "mean"))
    rs = frame.groupby(by + ["race"], observed=True).pnl.sum().groupby(level=list(range(len(by)))).agg(["std", "count"])
    out = base.join(rs)
    out["se"] = out["std"] * np.sqrt(out["count"]) / out["n"]
    out["t"] = out.roi / out.se
    return out.drop(columns=["std", "count"])


frames = {}
for (strat, f), (pnl, hit, fair) in cols.items():
    ok = ~np.isnan(fair)
    frames[(strat, f)] = pd.DataFrame({
        "mkt": d.mkt.to_numpy()[ok], "period": d.period.to_numpy()[ok], "sp_band": d.sp_band.to_numpy()[ok],
        "code": d.code.to_numpy()[ok], "field_band": d.field_band.to_numpy()[ok],
        "race": (d.mkt + "|" + d.event_id.astype(str)).to_numpy()[ok],
        "pnl": pnl[ok], "hit": hit[ok].astype(float), "fair": fair[ok]})

# --- 1. overall, by market x strategy x f ------------------------------------------------------------------
print("\n== 1. every runner, by market, strategy and offset (ROI per GBP1 backed / laid at the SP, after 2%)")
o = pd.concat({k: stats(v, ["mkt", "period"]) for k, v in frames.items()}, names=["strategy", "f"])
print(o[["n", "match", "fair", "roi", "t"]].unstack("period").round(4).to_string())

# --- 2. by SP band, the most informative offsets ---------------------------------------------------------
print("\n== 2. by SP band, match rate against the fair rate and ROI after 2%")
o2 = pd.concat({k: stats(v, ["mkt", "sp_band", "period"]) for k, v in frames.items() if k[1] in (0.75, 0.5, 0.3)},
               names=["strategy", "f"])
o2 = o2[o2.n >= 200]
print(o2[["n", "match", "fair", "roi", "t"]].unstack("period").round(4).to_string())

# --- 3. cells chosen on 2018-22, read on 2023-26Q1 ---------------------------------------------------------
print("\n== 3. cells (market x SP band x code x field) chosen on 2018-22 (n >= 2,000, t >= 2.5), read on 2023-26Q1")
cell = ["mkt", "sp_band", "code", "field_band"]
o3 = pd.concat({k: stats(v, cell + ["period"]) for k, v in frames.items()}, names=["strategy", "f"])
tr = o3.xs("train", level="period")
te = o3.xs("test", level="period")
chosen = tr[(tr.n >= 2000) & (tr.t >= 2.5)]
res = chosen[["n", "roi", "t"]].add_prefix("train_").join(te[["n", "roi", "t", "match", "fair"]].add_prefix("test_"))
if res.empty:
    print("no cell passed on 2018-22")
else:
    res = res.sort_values("train_t", ascending=False)
    print(res.round(4).to_string())
    print(f"\n{len(res)} cells chosen; {int((res.test_roi > 0).sum())} positive on 2023-26Q1; pooled test ROI "
          f"(weighted by bets) {np.average(res.test_roi.fillna(0), weights=res.test_n.fillna(0) + 1e-9):+.4f}")
# the negative side, for lays of the same: cells losing most on 2018-22 (a loss for one side is not a profit for the
# other here, since the offset's two strategies differ), for reference only
worst = tr[(tr.n >= 2000)].sort_values("t").head(15)
print("\nmost negative cells on 2018-22 (for reference):")
print(worst[["n", "roi", "t", "match", "fair"]].round(4).to_string())
print(f"\ndone in {time.time() - t0:.0f}s")
