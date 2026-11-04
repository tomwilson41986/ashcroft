"""The owner's day report (the standard approach, 22 Oct: "never give me inflated expected value, i need real figures
that i can expect and forecast"; read-only). REPORT_DAY picks the day (default: today in the UK). Three figures only:

1. Settled: the races run and settled, as Betfair settled them (the ledger's settled P&L less commission; where the
   ledger holds no commission for a winning market, Betfair's 2% of its net winnings).
2. The mark on the races to run: every position valued now at the mid (the geometric mean of the best back and best
   lay) and at the best lay (Betfair's Cash Out), each fill's price first cut by the reduction factors of the runners
   withdrawn after it was matched, a non-runner's own stakes void; a race that has run but is not settled at its BSP.
3. The forecast (the owner, 4 Nov: "always forecast on today's markets, not on historic days results ... based on
   historic days mid price and cash outs"): each race to run is today's mark plus the drift from that mark to the
   settled result, a pound staked, on the past live days' races marked the same way at the same time before the off
   (5 min to 10 h, or at the BSP once run): the later non-runners' cuts and voids, the move to the BSP, the commission.
   Read from the mid and, as a check, from Cash Out. The range is the 10th-90th percentile of the sum of each race's
   miss drawn from the past races' misses at that time. The check: the same forecast made on every past day at today's
   clock time, with that day left out of the drift, against what those races then settled at.
"""
import io
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAY = os.environ.get("REPORT_DAY") or datetime.now(ZoneInfo("Europe/London")).date().isoformat()
FIRST = date(2026, 10, 10)
RATE = 0.02
K = ["market_id", "selection_id"]
GRID = [5, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 600]   # minutes before the off; 0 is the BSP
MIN_POOL = 30
DRAWS = 20000
rng = np.random.default_rng(1104)


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def ledger(day):
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        return None
    led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "edge", "selection_id", "pnl", "commission", "bsp"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    return led


def settled_by_market(led):
    """Each settled market's result after commission (the ledger's, else 2% of a winning market's net)."""
    st = led[led.event == "settle"].drop_duplicates(K, keep="last")
    if st.empty:
        return pd.DataFrame(columns=["market_id", "pnl", "commission", "result"])
    p = st.groupby("market_id").pnl.sum().rename("pnl").reset_index()
    c = led[led.event == "commission"].groupby("market_id").commission.sum().rename("commission").reset_index()
    p = p.merge(c, on="market_id", how="left")
    p["commission"] = np.where(p.commission.fillna(0) > 0, p.commission.fillna(0), RATE * p.pnl.clip(lower=0))
    p["result"] = p.pnl - p.commission
    return p


def books(day):
    parts = []
    for n in ("books_trader", "books"):
        raw = get(f"betfair_live/{day}/{n}.csv.gz")
        if raw is not None:
            parts.append(pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False, dtype={"market_id": str},
                                     usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "status",
                                                             "inplay", "runner_status", "back1", "lay1", "sp_actual",
                                                             "adjustment_factor", "removal_utc")))
    if not parts:
        return None
    bk = pd.concat(parts, ignore_index=True)
    for c in ("selection_id", "back1", "lay1", "sp_actual", "adjustment_factor"):
        bk[c] = pd.to_numeric(bk[c], errors="coerce")
    bk = bk.dropna(subset=["selection_id"]).sort_values("polled_utc")
    bk["selection_id"] = bk.selection_id.astype("int64")
    bk["pt"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    return bk


def removals(bk):
    rem = bk[bk.runner_status.eq("REMOVED")]
    first_seen = rem.groupby(K).pt.min()
    rem = rem.drop_duplicates(K, keep="last")[K + ["adjustment_factor", "removal_utc"]].copy()
    rem["removal_t"] = pd.to_datetime(rem.removal_utc, utc=True, errors="coerce")
    rem["removal_t"] = rem.removal_t.fillna(rem.set_index(K).index.map(first_seen).to_series(index=rem.index))
    return rem


def pre_off(bk):
    ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    pre = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE")
             & (bk.back1 > 1) & (bk.lay1 > 1)]
    pre = pre[K + ["pt", "back1", "lay1"]].dropna(subset=["pt"]).copy()
    pre["pt"] = utc_ns(pre.pt)
    return pre.sort_values("pt")


def fills(led, day):
    b = led[(led.event == "back") & (led.matched > 0)].copy()
    b["selection_id"] = b.selection_id.astype("int64")
    hm = pd.to_datetime(b.off, format="mixed", errors="coerce").dt.strftime("%H:%M")
    b["off_uk"] = hm
    b["off_t"] = (pd.to_datetime(day + " " + hm, errors="coerce")
                  .dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"))
    return b.dropna(subset=["t", "off_t"]).reset_index(drop=True)


def utc_ns(s):
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def cut_and_void(f, rem, T=None):
    """Each fill's price cut by the runners withdrawn after it was matched and by T; void if its own horse is out by T
    (T a time, or None to read each row's own T)."""
    f = f.reset_index(drop=True).copy()
    f["row"] = f.index
    if T is not None:
        f["T"] = T
    f["T"] = utc_ns(f["T"])
    x = f[["row", "market_id", "selection_id", "t", "T"]].merge(
        rem.rename(columns={"selection_id": "rsel"}), on="market_id")
    cut = x[(x.rsel != x.selection_id) & (x.removal_t > x.t) & (x.removal_t <= x["T"])]
    lf = np.log(np.clip(1 - cut.adjustment_factor.fillna(0) / 100.0, 1e-9, None)).groupby(cut.row).sum()
    f["P_red"] = np.maximum(1.01, f.avg_price * np.exp(f.row.map(lf).fillna(0.0)))
    own = x[(x.rsel == x.selection_id) & (x.removal_t <= x["T"])].row
    f["void"] = f.row.isin(own)
    return f


def marks_at(b, pre, rem, pts):
    """The mark of each market's position at its points pts [market_id, T, g]: positions complete by T only."""
    n_all = b.groupby("market_id").size()
    f = b.merge(pts, on="market_id")
    f = f[f.t <= f["T"]]
    if f.empty:
        return pd.DataFrame()
    f = cut_and_void(f, rem).sort_values("T")
    f = pd.merge_asof(f, pre, left_on="T", right_on="pt", by=K, direction="backward",
                      tolerance=pd.Timedelta("60min"))
    lv = ~f.void
    mid = np.sqrt(f.back1 * f.lay1)
    f["mark_mid"] = np.where(lv, f.matched * (f.P_red / mid - 1), 0.0)
    f["cash_out"] = np.where(lv, f.matched * (f.P_red / f.lay1 - 1), 0.0)
    f["staked"] = np.where(lv, f.matched, 0.0)
    f["priced"] = ~lv | (f.back1 > 1)
    g = f.groupby(["market_id", "g"]).agg(n=("row", "size"), priced=("priced", "all"), staked=("staked", "sum"),
                                          mark_mid=("mark_mid", "sum"), cash_out=("cash_out", "sum")).reset_index()
    g = g[g.priced & (g.n == g.market_id.map(n_all)) & (g.staked > 0)]
    return g.drop(columns=["n", "priced"])


def marks_bsp(b, rem, sp):
    """The mark of each market's whole position at its BSP, every cut and void applied (point g = 0)."""
    f = cut_and_void(b, rem, pd.Timestamp("2100-01-01", tz="UTC"))
    f = f.merge(sp, on=K, how="left")
    f["bsp_x"] = f.bsp_now.where(f.bsp_now > 1, f.bsp)
    lv = ~f.void
    f["mark"] = np.where(lv, f.matched * (f.P_red / f.bsp_x - 1), 0.0)
    f["staked"] = np.where(lv, f.matched, 0.0)
    f["priced"] = ~lv | (f.bsp_x > 1)
    g = f.groupby("market_id").agg(priced=("priced", "all"), staked=("staked", "sum"), mark=("mark", "sum"))
    g = g[g.priced & (g.staked > 0)].reset_index()
    return g.assign(g=0, mark_mid=g.mark, cash_out=g.mark).drop(columns=["mark", "priced"])


def nearest_g(minutes, have):
    have = np.array(sorted(have))
    pos = have[have > 0]
    return np.array([0 if m is None or (isinstance(m, float) and np.isnan(m)) else
                     int(pos[np.abs(np.log(pos) - np.log(max(m, 1.0))).argmin()]) for m in minutes])


def calibrate(obs):
    """A pound staked: the drift from the mark to the result at each point, and the pools of misses about it."""
    cal = {}
    for g, o in obs.groupby("g"):
        if len(o) < MIN_POOL:
            continue
        row = {}
        for col in ("mark_mid", "cash_out"):
            d = (o.result - o[col]).sum() / o.staked.sum()
            row[col] = (d, ((o.result - o[col] - d * o.staked) / o.staked).values)
        cal[g] = row
    return cal


def forecast(races, cal, col, draws=DRAWS):
    """races [staked, mark col, g]: the forecast and draws of its sum."""
    gs = nearest_g(races.minutes.values, cal.keys())
    gs = np.where(races.ran.values, 0 if 0 in cal else gs, gs)
    f = np.array([races[col].values[i] + cal[g][col][0] * races.staked.values[i] for i, g in enumerate(gs)])
    sims = np.zeros(draws)
    for i, g in enumerate(gs):
        sims += rng.choice(cal[g][col][1], size=draws) * races.staked.values[i]
    return f, f.sum() + sims


# ------------------------------------------------------------------ the past days: marks at each point, the result
now_uk = None
obs, backtest_days, rec = [], {}, []
d = FIRST
today = date.fromisoformat(DAY)
past = []
while d < today:
    day_s = d.isoformat()
    d += timedelta(days=1)
    led = ledger(day_s)
    if led is None:
        continue
    sm = settled_by_market(led)
    rec.append({"day": day_s, "result": float(sm.result.sum())})
    bk = books(day_s)
    if bk is None or sm.empty:
        continue
    b = fills(led, day_s)
    b = b[b.market_id.isin(sm.market_id)]
    if b.empty:
        continue
    rem, pre = removals(bk), pre_off(bk)
    sp = bk[bk.sp_actual > 1].groupby(K).sp_actual.last().rename("bsp_now").reset_index()
    offs = b.groupby("market_id").off_t.first()
    pts = pd.concat([pd.DataFrame({"market_id": offs.index, "T": (offs - pd.Timedelta(minutes=g)).reset_index(drop=True),
                                   "g": g})
                     for g in GRID], ignore_index=True)
    o = pd.concat([marks_at(b, pre, rem, pts), marks_bsp(b, rem, sp)], ignore_index=True)
    o = o.merge(sm[["market_id", "result"]], on="market_id").assign(day=day_s)
    obs.append(o)
    past.append((day_s, b, pre, rem, sp, sm, offs))
obs = pd.concat(obs, ignore_index=True)
rec = pd.DataFrame(rec).set_index("day")
cal = calibrate(obs)

# ------------------------------------------------------------------ the day
led = ledger(DAY)
if led is None:
    raise SystemExit(f"no live ledger for {DAY}")
bk = books(DAY)
rem = removals(bk)
pre = pre_off(bk)
nowpx = pre.groupby(K).agg(back_now=("back1", "last"), lay_now=("lay1", "last")).reset_index()
sp = bk[bk.sp_actual > 1].groupby(K).sp_actual.last().rename("bsp_now").reset_index()
now = bk.pt.max()

b = fills(led, DAY)
f = cut_and_void(b, rem, now)
f = f.merge(nowpx, on=K, how="left").merge(sp, on=K, how="left")
sm = settled_by_market(led)
f["settled"] = f.market_id.isin(sm.market_id)
live = f[~f.void].copy()
ran = live.bsp_now > 1
mid = np.sqrt(live.back_now * live.lay_now)
live["mark_mid"] = live.matched * (live.P_red / np.where(ran, live.bsp_now, mid) - 1)
live["cash_out"] = live.matched * (live.P_red / np.where(ran, live.bsp_now, live.lay_now) - 1)
live["ran"] = ran
m = live.groupby(["off_uk", "venue", "market_id"]).agg(
    horses=("selection_id", "nunique"), staked=("matched", "sum"), mark_mid=("mark_mid", "sum"),
    cash_out=("cash_out", "sum"), settled=("settled", "any"), ran=("ran", "any"), off_t=("off_t", "first")).reset_index()
m = m.merge(sm[["market_id", "result"]], on="market_id", how="left")
m["minutes"] = (m.off_t - now).dt.total_seconds() / 60.0
m["ran"] = m.ran | (m.minutes <= 0)
for c in ("mark_mid", "cash_out"):
    m.loc[m.settled, c] = np.nan
left = m[~m.settled].copy()
done_res = float(sm.result.sum())
if len(left):
    left["forecast"], sims = forecast(left, cal, "mark_mid")
    _, sims_co = forecast(left, cal, "cash_out")
    m = m.merge(left[["market_id", "forecast"]], on="market_id", how="left")
else:
    sims = sims_co = np.zeros(1)
    m["forecast"] = np.nan
f_left, f_left_co = float(np.mean(sims)), float(np.mean(sims_co))
r_lo, r_hi = np.percentile(sims, [10, 90])

# ------------------------------------------------------------------ the check: the same forecast on every past day
clock = now.tz_convert("Europe/London")
bt = []
for day_s, pb, ppre, prem, psp, psm, poffs in past:
    T = pd.Timestamp(f"{day_s} {clock:%H:%M:%S}").tz_localize("Europe/London").tz_convert("UTC")
    run = poffs[poffs > T]
    if run.empty:
        continue
    o = marks_at(pb, ppre, prem, pd.DataFrame({"market_id": run.index, "T": T, "g": -1}))
    if o.empty:
        continue
    o = o.merge(psm[["market_id", "result"]], on="market_id")
    o["minutes"] = (o.market_id.map(poffs) - T).dt.total_seconds() / 60.0
    o["ran"] = False
    cal_loo = calibrate(obs[obs.day != day_s])
    fc, sm_draws = forecast(o, cal_loo, "mark_mid", draws=5000)
    lo_, hi_ = np.percentile(sm_draws, [10, 90])
    bt.append({"day": day_s, "races": len(o), "mark_mid": o.mark_mid.sum(), "cash_out": o.cash_out.sum(),
               "forecast": fc.sum(), "lo": lo_, "hi": hi_, "result": o.result.sum()})
bt = pd.DataFrame(bt)

print(f"== the day report, {DAY}: books read to {bk.polled_utc.max()} UTC")
print(f"   staked GBP{b.matched.sum():,.2f} on {b.groupby(K).ngroups} horses in {b.market_id.nunique()} races; void "
      f"(our non-runners) GBP{f[f.void].matched.sum():,.2f}; prices cut by other non-runners on "
      f"GBP{live[live.P_red < live.avg_price - 1e-9].matched.sum():,.2f}")
print(f"\n1. settled: {len(sm)} races, GBP{done_res:+,.2f} after commission")
print(f"2. the mark on the {len(left)} races to run (after the cuts): at the mid GBP{left.mark_mid.sum():+,.2f}; "
      f"Cash Out (the lays) GBP{left.cash_out.sum():+,.2f}")
print(f"3. the forecast on today's markets: the races to run GBP{f_left:+,.2f} from the mid (GBP{f_left_co:+,.2f} from "
      f"Cash Out), range GBP{r_lo:+,.2f} to GBP{r_hi:+,.2f} (10th-90th); the day GBP{done_res + f_left:+,.2f}, range "
      f"GBP{done_res + r_lo:+,.2f} to GBP{done_res + r_hi:+,.2f}")
print(f"   the drift a pound staked, mark to result, on {obs.day.nunique()} past days "
      f"({len(obs[obs.g > 0].market_id.unique())} races):")
dr = pd.DataFrame([{"min_to_off": g, "races": len(cal[g]["mark_mid"][1]), "from_mid": cal[g]["mark_mid"][0],
                    "from_cash_out": cal[g]["cash_out"][0]} for g in sorted(cal)])
print(dr.round(4).to_string(index=False))
if len(bt):
    inside = int(((bt.result >= bt.lo) & (bt.result <= bt.hi)).sum())
    print(f"\n   the check, at {clock:%H:%M} UK on each past day (that day left out): forecast GBP{bt.forecast.sum():+,.2f}"
          f", settled GBP{bt.result.sum():+,.2f} over {len(bt)} days; mean miss GBP{(bt.result - bt.forecast).mean():+,.2f}"
          f" a day; inside the range on {inside} of {len(bt)} (80% expected)")
    print(bt.round(2).to_string(index=False))
print(f"\n   the record (context, not in the forecast): {len(rec)} days GBP{rec.result.sum():+,.2f}, a day mean "
      f"GBP{rec.result.mean():+,.2f}, {int((rec.result > 0).sum())} up")
print("\n== every race (GBP; mark and Cash Out on the races to run, the result on the races settled)")
out = m[["off_uk", "venue", "horses", "staked", "mark_mid", "cash_out", "forecast", "result"]].sort_values("off_uk")
print(out.round(2).to_string(index=False))
