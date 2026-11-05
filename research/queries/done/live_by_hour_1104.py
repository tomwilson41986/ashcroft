"""The live record by the time of the bet (the owner, 4 Nov: "what is our historic record betting after 10am, 11am? do a
full historic time breakdown"; read-only). Every live back from 10 Oct (the owner's rule of 9 Oct) to 4 Nov on a
settled market: by the UK hour it was matched and by the time to the off, the stake, the CLV against the BSP at the
fill and after the non-runners' cuts (the value locked), and the result. Each back is laid at the SP for its winnings,
so a fill's own result is s(P_cut - P) when the horse wins and s((P - 1)/(B - 1) - 1) when it loses (void when its horse
is out); summed and less Betfair's 2% of each market's net winnings, shared out by the fills' share of it, the days
add up to the ledger's settled results (checked below)."""
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
K = ["market_id", "selection_id"]
FIRST, LAST = date(2026, 10, 10), date(2026, 11, 4)
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
    for c in ("matched", "avg_price", "selection_id", "pnl", "commission", "bsp"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    return led


def removals(day):
    parts = []
    for n in ("books_trader", "books"):
        raw = get(f"betfair_live/{day}/{n}.csv.gz")
        if raw is not None:
            parts.append(pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False, dtype={"market_id": str},
                                     usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "runner_status",
                                                             "adjustment_factor", "removal_utc", "sp_actual")))
    if not parts:
        return None, None
    bk = pd.concat(parts, ignore_index=True)
    for c in ("selection_id", "adjustment_factor", "sp_actual"):
        bk[c] = pd.to_numeric(bk[c], errors="coerce")
    bk = bk.dropna(subset=["selection_id"])
    bk["selection_id"] = bk.selection_id.astype("int64")
    bk["pt"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    rem = bk[bk.runner_status.eq("REMOVED")]
    first = rem.groupby(K).pt.min()
    rem = rem.drop_duplicates(K, keep="last")[K + ["adjustment_factor", "removal_utc"]].copy()
    rem["removal_t"] = pd.to_datetime(rem.removal_utc, utc=True, errors="coerce")
    rem["removal_t"] = rem.removal_t.fillna(rem.set_index(K).index.map(first).to_series(index=rem.index))
    sp = bk[bk.sp_actual > 1].groupby(K).sp_actual.last().rename("bsp_book").reset_index()
    return rem, sp


rows, checks = [], []
d = FIRST
while d <= LAST:
    day = d.isoformat()
    d += timedelta(days=1)
    led = ledger(day)
    if led is None:
        continue
    st = led[led.event == "settle"].drop_duplicates(K, keep="last")
    if st.empty:
        continue
    b = led[(led.event == "back") & (led.matched > 0) & led.market_id.isin(st.market_id)].copy()
    b["selection_id"] = b.selection_id.astype("int64")
    st = st.assign(selection_id=st.selection_id.astype("int64"))
    rem, sp = removals(day)
    if rem is None:
        rem = pd.DataFrame(columns=K + ["adjustment_factor", "removal_t"])
        sp = pd.DataFrame(columns=K + ["bsp_book"])
    b = b.reset_index(drop=True)
    b["row"] = b.index
    x = b[["row", "market_id", "selection_id", "t"]].merge(rem.rename(columns={"selection_id": "rsel"}), on="market_id")
    cut = x[(x.rsel != x.selection_id) & (x.removal_t > x.t)]
    lf = np.log(np.clip(1 - cut.adjustment_factor.fillna(0) / 100.0, 1e-9, None)).groupby(cut.row).sum()
    b["P"] = b.avg_price
    b["P_cut"] = np.maximum(1.01, b.P * np.exp(b.row.map(lf).fillna(0.0)))
    for c in ("pnl_lay", "hedge_liability"):
        st[c] = pd.to_numeric(st[c], errors="coerce")
    st["bsp_paid"] = np.where((st.result == "LOSER") & (st.pnl_lay > 0) & (st.hedge_liability > 0),
                              1 + st.hedge_liability / st.pnl_lay, np.nan)
    b = b.merge(st[K + ["result", "bsp", "bsp_paid"]].rename(columns={"result": "res", "bsp": "bsp_led"}), on=K,
                how="left")
    b = b.merge(sp, on=K, how="left")
    # the BSP: the recorded books' SP, else the ledger's read. A non-runner's reduction cuts the SP lay's liability with
    # the back's price, so the lay still covers the cut winnings (live_by_hour_check_1104 and the check below)
    b["B"] = b.bsp_book.where(b.bsp_book > 1, b.bsp_led)
    b["led_off"] = (b.bsp_led > 1) & (b.B > 1) & ((b.bsp_led / b.B - 1).abs() > 0.02)
    b["void"] = b.res.astype(str).str.upper().eq("REMOVED")
    win = b.res.astype(str).str.upper().eq("WINNER")
    ok = ~b.void & (b.B > 1)
    b["pnl"] = np.where(b.void, 0.0, np.where(win, 0.0, b.matched * ((b.P_cut - 1) / (b.B - 1) - 1)))
    b.loc[~ok & ~b.void, "pnl"] = np.nan
    net = b.groupby("market_id").pnl.transform("sum")
    pos = b.groupby("market_id").pnl.transform(lambda s: s.clip(lower=0).sum())
    b["comm"] = np.where(net > 0, 0.02 * net * b.pnl.clip(lower=0) / pos.replace(0, np.nan), 0.0)
    b["result"] = b.pnl - np.nan_to_num(b.comm)
    b["locked"] = np.where(ok, b.matched * (b.P_cut / b.B - 1), 0.0)
    b["clv_fill_w"] = np.where(ok, b.matched * (b.P / b.B - 1), 0.0)
    b["staked_live"] = np.where(ok, b.matched, 0.0)
    b["day"] = day
    uk = b.t.dt.tz_convert("Europe/London")
    b["hour"] = uk.dt.hour
    off = pd.to_datetime(day + " " + pd.to_datetime(b.off, format="mixed", errors="coerce").dt.strftime("%H:%M"),
                         errors="coerce").dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    b["mins"] = (off - uk).dt.total_seconds() / 60
    b["clv_led_w"] = np.where(ok & (b.bsp_led > 1), b.matched * (b.P / b.bsp_led - 1), 0.0)
    if day in ("2026-11-03", "2026-10-29"):
        per = b.groupby(K).agg(fills=("matched", "size"), staked=("matched", "sum"), mine=("pnl", "sum"),
                               B=("B", "first"), bsp_led=("bsp_led", "first"), bsp_paid=("bsp_paid", "first"),
                               bsp_book=("bsp_book", "first"), res=("res", "first"),
                               won=("P", lambda x: 0)).reset_index()
        per = per.merge(st[K + ["pnl", "pnl_back", "pnl_lay", "hedge_liability", "matched", "avg_price"]], on=K, how="left")
        per["wins_at_fill"] = per.selection_id.map(b.assign(w=b.matched * (b.P - 1)).groupby("selection_id").w.sum())
        per["diff"] = per.mine - per.pnl
        print(f"-- {day}: mine {per.mine.sum():+.2f} against the ledger {per.pnl.sum():+.2f}")
        print(per.reindex(per["diff"].abs().sort_values(ascending=False).index).head(8).drop(columns="won")
              .round(2).to_string(index=False))
    rows.append(b[["day", "hour", "mins", "market_id", "selection_id", "matched", "staked_live", "clv_fill_w",
                   "locked", "result", "void", "led_off", "clv_led_w"]])
    ledg = st.groupby("market_id").pnl.sum()
    c = led[led.event == "commission"].groupby("market_id").commission.sum()
    ledg = (ledg - c.reindex(ledg.index).fillna(0.02 * ledg.clip(lower=0))).sum()
    checks.append({"day": day, "ours": round(b.result.sum(), 2), "ledger": round(float(ledg), 2)})
f = pd.concat(rows, ignore_index=True)
ck = pd.DataFrame(checks)
print(f"== the live record by the time of the bet, {f.day.nunique()} days {f.day.min()} to {f.day.max()}, "
      f"{len(f)} fills, GBP{f.matched.sum():,.0f} staked")
print(f"   the books' BSP and the ledger's differ by more than 2% on {f.led_off.mean():.1%} of the fills")
print(f"   the check: fills' results GBP{ck.ours.sum():+,.2f} against the ledger's settled GBP{ck.ledger.sum():+,.2f}; "
      f"days off by more than GBP5: {int(((ck.ours - ck.ledger).abs() > 5).sum())}")


def table(by):
    out = []
    for k, g in f.groupby(by, observed=True):
        per_day = g.groupby("day").result.sum()
        boot = [per_day.sample(len(per_day), replace=True, random_state=int(rng.integers(1e9))).sum()
                for _ in range(1000)]
        s = g.staked_live.sum()
        out.append({by: k, "days": per_day.size, "fills": len(g), "staked": round(g.matched.sum(), 0),
                    "clv_fill": g.clv_fill_w.sum() / s if s else np.nan,
                    "clv_after_cuts": g.locked.sum() / s if s else np.nan,
                    "locked": round(g.locked.sum(), 0), "result": round(g.result.sum(), 0),
                    "result_per_100": 100 * g.result.sum() / g.matched.sum(),
                    "days_up": int((per_day > 0).sum()),
                    "result_lo": round(np.percentile(boot, 5), 0), "result_hi": round(np.percentile(boot, 95), 0)})
    return pd.DataFrame(out).round({"clv_fill": 4, "clv_after_cuts": 4, "result_per_100": 2})


print("\n-- by the UK hour of the bet (result after commission; lo-hi: 90% range over the days, by bootstrap)")
print(table("hour").to_string(index=False))
f["after"] = np.where(f.hour >= 11, "11:00 on", np.where(f.hour >= 10, "10:00-10:59", "before 10:00"))
print("\n-- before and after 10 and 11")
print(table("after").to_string(index=False))
f["to_off"] = pd.cut(f.mins, [-5, 15, 30, 60, 120, 240, 360, 2000],
                     labels=["15-30m"[:0] + "<15m", "15-30m", "30-60m", "1-2h", "2-4h", "4-6h", "6h+"])
print("\n-- by the time to the off")
print(table("to_off").to_string(index=False))
f["week"] = pd.to_datetime(f.day).dt.strftime("%d %b")
w = f.assign(band=np.where(f.hour >= 10, "10:00 on", "before 10")).pivot_table(
    index="day", columns="band", values="result", aggfunc="sum").round(0)
s = f.assign(band=np.where(f.hour >= 10, "10:00 on", "before 10")).pivot_table(
    index="day", columns="band", values="matched", aggfunc="sum").round(0)
print("\n-- day by day: the result (and the stake) before 10:00 and from 10:00")
print(w.join(s.add_suffix("_staked")).to_string())
