"""The owner's day-report forecast for the horse traders on the dashboard (read-only).

The method is the day report's (``research/queries/done/day_report.py``; the owner, 22 Oct and 4 Nov 2026: "never give
me inflated expected value", "always forecast on today's markets, not on historic days results"), the same functions
read from the dashboard's source (S3 or a local copy) rather than run as a script:

1. Settled: the races run and settled (the ledger's settled P&L less commission; else 2% of a winning market's net).
2. The mark on the races to run: every position valued now at the mid (the geometric mean of the best back and best
   lay) and at the best lay (Betfair's Cash Out), each fill's price first cut by the reduction factors of the runners
   withdrawn after it was matched, a non-runner's own stakes void; a race that has run but is not settled at its BSP.
3. The forecast: each race to run is today's mark plus the drift from that mark to the settled result, a pound staked,
   on the trader's past days' races marked the same way at the same time before the off (5 min to 10 h, or at the BSP
   once run). The range is the 10th-90th percentile of the sum of each race's miss drawn from the past races' misses
   at that time. Never the expected value at the fill, never a ratio of the record's result to it.

The day report's check (the same forecast on every past day at today's clock time) is left to the day report itself:
it needs every past day's books on each run. Each past day's marks are cached once every race of that day is settled.
"""
from __future__ import annotations

import io
import math
from datetime import date

import numpy as np
import pandas as pd

RATE = 0.02
K = ["market_id", "selection_id"]
GRID = [5, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 600]   # minutes before the off; 0 is the BSP
MIN_POOL = 30                    # past races at a point before its drift is used (as the day report)
DRAWS = 20000
SEED = 1104
CACHE = "dashboard/cache/v2/day_forecast"
OBS_COLS = ["market_id", "g", "staked", "mark_mid", "cash_out", "result", "day"]


# --------------------------------------------------------------------------------------------------------------------
# Reading (the day report's functions, from a dashboard source)
# --------------------------------------------------------------------------------------------------------------------

def _read(raw: bytes | None, gz: bool = False, **kw) -> pd.DataFrame | None:
    if not raw or not raw.strip():
        return None
    try:
        return pd.read_csv(io.BytesIO(raw), compression="gzip" if gz else None, low_memory=False,
                           dtype={"market_id": str}, **kw)
    except pd.errors.EmptyDataError:
        return None


def ledger(src, kind: str, base: str, day: str) -> pd.DataFrame | None:
    led = _read(src.get(kind, f"{base}/{day}/ledger.csv"))
    if led is None or "event" not in led.columns:
        return None
    for c in ("matched", "avg_price", "edge", "selection_id", "pnl", "commission", "bsp"):
        led[c] = pd.to_numeric(led[c], errors="coerce") if c in led.columns else np.nan
    for c in ("off", "venue", "ts"):
        if c not in led.columns:
            led[c] = None
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    return led


def settled_by_market(led: pd.DataFrame) -> pd.DataFrame:
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


BOOK_COLS = ("polled_utc", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "lay1",
             "sp_actual", "adjustment_factor", "removal_utc")


def books(src, day: str) -> pd.DataFrame | None:
    parts = []
    for n in ("books_trader", "books"):
        f = _read(src.get("capture", f"betfair_live/{day}/{n}.csv.gz"), gz=True, usecols=lambda c: c in BOOK_COLS)
        if f is not None:
            parts.append(f)
    if not parts:
        return None
    bk = pd.concat(parts, ignore_index=True)
    for c in BOOK_COLS:
        if c not in bk.columns:
            bk[c] = np.nan
    for c in ("selection_id", "back1", "lay1", "sp_actual", "adjustment_factor"):
        bk[c] = pd.to_numeric(bk[c], errors="coerce")
    bk = bk.dropna(subset=["selection_id"]).sort_values("polled_utc")
    bk["selection_id"] = bk.selection_id.astype("int64")
    bk["pt"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    return bk


def utc_ns(s):
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def removals(bk: pd.DataFrame) -> pd.DataFrame:
    rem = bk[bk.runner_status.eq("REMOVED")]
    if rem.empty:                                   # no non-runner: nothing to cut or void
        return pd.DataFrame({"market_id": pd.Series(dtype=str), "selection_id": pd.Series(dtype="int64"),
                             "adjustment_factor": pd.Series(dtype=float), "removal_utc": pd.Series(dtype=object),
                             "removal_t": pd.Series(dtype="datetime64[ns, UTC]")})
    first_seen = rem.groupby(K).pt.min()
    rem = rem.drop_duplicates(K, keep="last")[K + ["adjustment_factor", "removal_utc"]].copy()
    rem["removal_t"] = pd.to_datetime(rem.removal_utc, utc=True, errors="coerce")
    rem["removal_t"] = rem.removal_t.fillna(rem.set_index(K).index.map(first_seen).to_series(index=rem.index))
    rem["removal_t"] = utc_ns(rem.removal_t)
    return rem


def pre_off(bk: pd.DataFrame) -> pd.DataFrame:
    ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    pre = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE")
             & (bk.back1 > 1) & (bk.lay1 > 1)]
    pre = pre[K + ["pt", "back1", "lay1"]].dropna(subset=["pt"]).copy()
    pre["pt"] = utc_ns(pre.pt)
    return pre.sort_values("pt")


def fills(led: pd.DataFrame, day: str) -> pd.DataFrame:
    b = led[(led.event == "back") & (led.matched > 0)].copy()
    b = b.dropna(subset=["selection_id"])
    b["selection_id"] = b.selection_id.astype("int64")
    hm = pd.to_datetime(b.off, format="mixed", errors="coerce").dt.strftime("%H:%M")
    b["off_uk"] = hm
    b["off_t"] = (pd.to_datetime(day + " " + hm, errors="coerce")
                  .dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"))
    return b.dropna(subset=["t", "off_t"]).reset_index(drop=True)


def cut_and_void(f: pd.DataFrame, rem: pd.DataFrame, T=None) -> pd.DataFrame:
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


def marks_at(b, pre, rem, pts) -> pd.DataFrame:
    """The mark of each market's position at its points pts [market_id, T, g]: positions complete by T only."""
    n_all = b.groupby("market_id").size()
    f = b.merge(pts, on="market_id")
    f = f[f.t <= f["T"]]
    if f.empty:
        return pd.DataFrame(columns=["market_id", "g", "staked", "mark_mid", "cash_out"])
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


def marks_bsp(b, rem, sp) -> pd.DataFrame:
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


def nearest_g(minutes, have) -> np.ndarray:
    have = np.array(sorted(have))
    pos = have[have > 0]
    return np.array([0 if m is None or (isinstance(m, float) and np.isnan(m)) or not len(pos) else
                     int(pos[np.abs(np.log(pos) - np.log(max(m, 1.0))).argmin()]) for m in minutes])


def calibrate(obs: pd.DataFrame, min_pool: int | None = None) -> dict:
    """A pound staked: the drift from the mark to the result at each point, and the pools of misses about it."""
    min_pool = MIN_POOL if min_pool is None else min_pool
    cal = {}
    for g, o in obs.groupby("g"):
        if len(o) < min_pool or o.staked.sum() <= 0:
            continue
        row = {}
        for col in ("mark_mid", "cash_out"):
            d = (o.result - o[col]).sum() / o.staked.sum()
            row[col] = (d, ((o.result - o[col] - d * o.staked) / o.staked).values)
        cal[int(g)] = row
    return cal


def forecast(races: pd.DataFrame, cal: dict, col: str, rng, draws: int = DRAWS):
    """races [staked, mark col, minutes, ran]: the forecast of each race and draws of their sum."""
    gs = nearest_g(races.minutes.values, cal.keys())
    gs = np.where(races.ran.values, 0 if 0 in cal else gs, gs)
    gs = np.array([g if g in cal else min(cal, key=lambda h: abs(h - g)) for g in gs])
    f = np.array([races[col].values[i] + cal[g][col][0] * races.staked.values[i] for i, g in enumerate(gs)])
    sims = np.zeros(draws)
    for i, g in enumerate(gs):
        sims += rng.choice(cal[g][col][1], size=draws) * races.staked.values[i]
    return f, f.sum() + sims


# --------------------------------------------------------------------------------------------------------------------
# The past days (cached once settled) and today
# --------------------------------------------------------------------------------------------------------------------

def day_obs(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, bool]:
    """One past day's races marked at each point before the off and at the BSP, with the settled result; and whether
    every race of the day is settled (the day is then final)."""
    led = ledger(src, kind, base, day)
    if led is None:
        return None, False
    sm = settled_by_market(led)
    backed = set(led[(led.event == "back") & (led.matched > 0)].market_id)
    complete = backed <= set(sm.market_id)
    bk = books(src, day)
    if bk is None or sm.empty:
        return pd.DataFrame(columns=OBS_COLS), complete and bk is not None
    b = fills(led, day)
    b = b[b.market_id.isin(sm.market_id)]
    if b.empty:
        return pd.DataFrame(columns=OBS_COLS), complete
    rem, pre = removals(bk), pre_off(bk)
    sp = bk[bk.sp_actual > 1].groupby(K).sp_actual.last().rename("bsp_now").reset_index()
    offs = b.groupby("market_id").off_t.first()
    pts = pd.concat([pd.DataFrame({"market_id": offs.index, "T": (offs - pd.Timedelta(minutes=g)).reset_index(drop=True),
                                   "g": g}) for g in GRID], ignore_index=True)
    o = pd.concat([marks_at(b, pre, rem, pts), marks_bsp(b, rem, sp)], ignore_index=True)
    o = o.merge(sm[["market_id", "result"]], on="market_id").assign(day=day)
    return o[OBS_COLS], complete


def past_obs(src, trader_id: str, kind: str, base: str, days: list[str], cache: bool = True) -> pd.DataFrame:
    """Every past day's observations, a final day read from the cache (``dashboard/cache/v2/day_forecast``)."""
    out = []
    for day in days:
        key = f"{CACHE}/{trader_id}/{day}.csv"
        if cache:
            raw = src.get("pred", key)
            if raw is not None:                                    # a final day: header-only when it marked nothing
                c = _read(raw)
                if c is not None and len(c):
                    out.append(c)
                continue
        o, final = day_obs(src, kind, base, day)
        if o is None:
            continue
        if len(o):
            out.append(o)
        if cache and final:
            try:
                src.put("pred", key, o.reindex(columns=OBS_COLS).to_csv(index=False).encode(), "text/csv")
            except Exception:  # noqa: BLE001  the cache is a convenience: the forecast stands without it
                pass
    if not out:
        return pd.DataFrame(columns=OBS_COLS)
    o = pd.concat(out, ignore_index=True)
    for c in ("g", "staked", "mark_mid", "cash_out", "result"):
        o[c] = pd.to_numeric(o[c], errors="coerce")
    return o.dropna(subset=["g", "staked", "mark_mid", "cash_out", "result"])


def today_forecast(src, trader_id: str, kind: str, base: str, today: date, record_days: list[str],
                   cache: bool = True, min_pool: int | None = None) -> dict:
    """Today's settled result, the mark on the races to run (mid and Cash Out) and the forecast with its range."""
    ds = today.isoformat()
    obs = past_obs(src, trader_id, kind, base, [d for d in record_days if d < ds], cache=cache)
    cal = calibrate(obs, min_pool)
    out = {"ok": False, "past_days": int(obs.day.nunique()) if len(obs) else 0,
           "past_races": int(obs[obs.g > 0].market_id.nunique()) if len(obs) else 0, "points": sorted(cal)}
    led = ledger(src, kind, base, ds)
    if led is None:
        return {**out, "why": "no ledger today"}
    sm = settled_by_market(led)
    out["settled"] = float(sm.result.sum())
    out["settled_races"] = int(len(sm))
    b = fills(led, ds)
    left_ids = set(b.market_id) - set(sm.market_id)
    if not left_ids:
        return {**out, "ok": True, "races_left": 0, "mark_mid": 0.0, "cash_out": 0.0, "forecast_left": 0.0,
                "forecast_left_cash_out": 0.0, "lo_left": 0.0, "hi_left": 0.0, "unpriced": 0, "races": []}
    if not cal:
        return {**out, "why": f"too few past races marked ({out['past_races']} on {out['past_days']} days; "
                              f"{MIN_POOL if min_pool is None else min_pool} a point)"}
    bk = books(src, ds)
    if bk is None:
        return {**out, "why": "no books recorded today yet"}
    rem, pre = removals(bk), pre_off(bk)
    nowpx = pre.groupby(K).agg(back_now=("back1", "last"), lay_now=("lay1", "last")).reset_index()
    sp = bk[bk.sp_actual > 1].groupby(K).sp_actual.last().rename("bsp_now").reset_index()
    now = bk.pt.max()
    f = cut_and_void(b, rem, now).merge(nowpx, on=K, how="left").merge(sp, on=K, how="left")
    f = f[f.market_id.isin(left_ids)]
    live = f[~f.void].copy()
    ran = live.bsp_now > 1
    mid = np.sqrt(live.back_now * live.lay_now)
    live["mark_mid"] = live.matched * (live.P_red / np.where(ran, live.bsp_now, mid) - 1)
    live["cash_out"] = live.matched * (live.P_red / np.where(ran, live.bsp_now, live.lay_now) - 1)
    live["ran"] = ran
    # grouped as the day report groups them (off, course, market): the draws are taken race by race in that order
    m = live.groupby(["off_uk", "venue", "market_id"], dropna=False).agg(
                                      horses=("selection_id", "nunique"), staked=("matched", "sum"),
                                      mark_mid=("mark_mid", "sum"), cash_out=("cash_out", "sum"),
                                      unpriced=("mark_mid", lambda s: int(s.isna().sum())),
                                      ran=("ran", "any"), off_t=("off_t", "first")).reset_index()
    m["minutes"] = (m.off_t - now).dt.total_seconds() / 60.0
    m["ran"] = m.ran | (m.minutes <= 0)
    priced = m.copy()                     # as the day report: a horse with no price yet adds nothing to its race's mark
    unpriced = int(m.unpriced.sum())
    rng = np.random.default_rng(SEED)
    if len(priced):
        priced["forecast"], sims = forecast(priced, cal, "mark_mid", rng)
        _, sims_co = forecast(priced, cal, "cash_out", rng)
    else:
        sims = sims_co = np.zeros(1)
        priced["forecast"] = np.nan
    lo, hi = np.percentile(sims, [10, 90])
    races = [{"market_id": r.market_id, "off": r.off_uk, "course": None if pd.isna(r.venue) else str(r.venue),
              "horses": int(r.horses), "staked": _r(r.staked), "mark_mid": _r(r.mark_mid), "cash_out": _r(r.cash_out),
              "forecast": _r(r.forecast)} for r in priced.sort_values("off_uk").itertuples()]
    return {**out, "ok": True, "books_to": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "races_left": int(len(m)),
            "unpriced": unpriced, "mark_mid": float(priced.mark_mid.sum()), "cash_out": float(priced.cash_out.sum()),
            "forecast_left": float(np.mean(sims)), "forecast_left_cash_out": float(np.mean(sims_co)),
            "lo_left": float(lo), "hi_left": float(hi), "races": races}


def _r(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, 2)


def record_days(listed: list[str], today: date, record_from: str | None, start_from: str | None) -> list[str]:
    """The past days the drift is read from: the trader's own days before today, from its record's start."""
    lo = max(x for x in (record_from, start_from, "0000-00-00") if x)
    return [d for d in listed if lo <= d < today.isoformat()]

