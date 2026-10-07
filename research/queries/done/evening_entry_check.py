"""Betting the evening before (the owner's question of 4 Oct: "even betting at 7pm the evening before?"; read-only).

Tomorrow's GB/IE win markets are recorded the evening before (books_evening, s3 betfair_live/<day>/): by the
recorder's evening run after the last race from 4 Oct (5 Oct's markets on), and by the trader's job from 17:00 to
21:30 UK every 15 minutes from 5 Oct (6 Oct's markets on). For each racing day so recorded, at each evening mark
(17:00, 18:00, 19:00, 20:00, 21:00 UK and the late look):
1. every runner: the best back against its BSP by price band, the size on offer, and how many were taken out
   before the off (a back matched the evening before is cut by each later non-runner's reduction factor);
2. the live rule at that mark: the trader's own plan (trading.strategy.plan_race, the live config, the day's
   model prices) on the evening's book, what the size on offer would have matched, and the CLV of those backs
   at the evening's price against the BSP, before and after the reductions of later non-runners;
3. the day's morning as it was traded (the ledger's first fills), by the same measures, and the horses both chose.
First, a check of the replication itself: the plan run on the books the trader read at its first step of each live
day (books_trader), against the horses it chose then.
The model's prices are the 06:00 run's, made after the evening (an evening session would price the evening's card),
so a race with a runner the morning's card no longer lists is not planned. A few evenings decide nothing; CLV is
before commission.
"""

from __future__ import annotations

import gzip
import io
import os
import sys
from dataclasses import replace
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402
from trading.config import load_config  # noqa: E402
from trading.exchange import Market  # noqa: E402
from trading.matching import attach_ids  # noqa: E402
from trading.strategy import RunnerView, _market_price, plan_race  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
UK = "Europe/London"
FIRST = date.fromisoformat(os.environ.get("EVENING_FROM") or "2026-10-05")   # the first day recorded the evening before
LAST = date.fromisoformat(os.environ.get("EVENING_TO") or "2026-10-05")   # 5 Oct, recorded 17:40-19:15 UK on 4 Oct
LIVE_FROM = date.fromisoformat(os.environ.get("CHECK_FROM") or "2026-10-01")  # the trader's first full day
MARKS = [("17:00", 17 * 60), ("18:00", 18 * 60), ("19:00", 19 * 60), ("20:00", 20 * 60), ("21:00", 21 * 60)]
LATE = (21 * 60 + 35, 24 * 60 + 6 * 60)        # the recorder's look after the trader: 21:35 UK to 06:00 on the day
TOL = 10                                       # minutes either side of a mark
BANDS, BAND_NAMES = [1, 3, 5, 8, 13, 21, 1e9], ["<3", "3-5", "5-8", "8-13", "13-21", "21+"]
KEY = ["market_id", "selection_id"]
cfg = load_config("trading/config_live.json")
lim = cfg.limits
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")


def _get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return None


def _csv(key, **kw):
    raw = _get(key)
    if raw is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(gzip.decompress(raw) if key.endswith(".gz") else raw), low_memory=False, **kw)


def _num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _hm(t) -> str:
    return "-" if pd.isna(t) else f"{t:%d %b %H:%M}"


def books(key):
    """A record's books, each with its poll time and its UK clock and date."""
    b = _csv(key, dtype={"market_id": str, "polled_utc": str})
    if b.empty:
        return b
    b = _num(b, ["selection_id", "back1", "back1_size", "lay1", "last_traded", "runner_total_matched", "sp_actual",
                 "adjustment_factor"]).dropna(subset=["selection_id"]).copy()
    b["selection_id"] = b.selection_id.astype("int64")
    b["t"] = pd.to_datetime(b.polled_utc, utc=True)
    u = b.t.dt.tz_convert(UK)
    b["mins"], b["uk_date"] = u.dt.hour * 60 + u.dt.minute, u.dt.date
    return b


def snapshot(b, on: date, mark: int | None):
    """Each market's whole book at its poll nearest the mark (UK minutes after midnight on ``on``, within TOL);
    with no mark, its last poll of the late look."""
    if b.empty:
        return b
    em = b.mins + 1440 * pd.Series([(d - on).days for d in b.uk_date], index=b.index)
    x = b[(em - mark).abs() <= TOL] if mark is not None else b[(em >= LATE[0]) & (em < LATE[1])]
    if x.empty:
        return x
    polls = x.groupby(["market_id", "polled_utc"], as_index=False).agg(t=("t", "first"))
    if mark is not None:
        target = pd.Timestamp(on, tz=UK) + pd.Timedelta(minutes=mark)
        polls["key"] = (polls.t - target).abs()
        polls = polls.sort_values("key")
    else:
        polls = polls.sort_values("t", ascending=False)
    pick = polls.drop_duplicates("market_id")[["market_id", "polled_utc"]]
    return x.merge(pick, on=["market_id", "polled_utc"])


def settled(day):
    """The day's BSPs and its non-runners (when taken out, and each one's reduction factor) from Betfair's books: the
    BSP read as each market reconciled it at the off (source "bsp", from 7 Oct; a closed market's final book carries
    none), else a final book's, else the price files'."""
    f = books(f"betfair_live/{day}/books.csv.gz")
    bsp = pd.DataFrame(columns=KEY + ["bsp"])
    rem = pd.DataFrame(columns=KEY + ["removed_at", "rf"])
    if not f.empty:
        at_off = f[(f.source == "bsp") & (f.sp_actual > 1)]
        f = f[f.source == "final"].drop_duplicates(KEY, keep="last").copy()
        bsp = (pd.concat([f.loc[f.sp_actual > 1, KEY + ["sp_actual"]], at_off[KEY + ["sp_actual"]]])
               .drop_duplicates(KEY, keep="last").rename(columns={"sp_actual": "bsp"}))
        rem = f.loc[f.runner_status == "REMOVED", KEY + ["adjustment_factor", "removal_utc"]].copy()
        rem["removed_at"] = pd.to_datetime(rem.removal_utc, utc=True, errors="coerce")
        rem["rf"] = rem.adjustment_factor.fillna(0.0)
    parts = []
    for d in (day + timedelta(days=1), day):
        for c in ("uk", "ire"):
            name = f"dwbfprices{c}win{d:%d%m%Y}.csv"
            raw = _get(f"{bp.S3_PREFIX}/{name}")
            if raw is None:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            t = bp.parse_file_text(text, name)
            parts.append(t[t.race_date == day.isoformat()])
    if parts:
        pf = pd.concat(parts, ignore_index=True).dropna(subset=["bsp", "selection_id"])
        pf = pd.DataFrame({"market_id": "1." + pf.event_id.astype("Int64").astype(str),
                           "selection_id": pf.selection_id.astype("int64"), "bsp": pf.bsp.astype(float)})
        bsp = pd.concat([bsp, pf]).drop_duplicates(KEY)
    return bsp, rem


def cut(rows, rem, t_col="t"):
    """The reduction on each back (market_id, matched at t): every non-runner taken out after it whose factor is
    2.5% or more, summed, at most 90% (Betfair's rules; a price of 10.0 cut by 20% settles at 8.0)."""
    if rem is None or rem.empty or rows.empty:
        return np.zeros(len(rows))
    r = rem[rem.rf >= 2.5]
    return np.array([min(float(r.rf[(r.market_id == m) & (r.removed_at > t)].sum()), 90.0) / 100.0
                     for m, t in zip(rows.market_id, rows[t_col])])


def _views(mid, g, prices, blind):
    """One market's runners as the trader's plan reads them."""
    return [RunnerView(selection_id=int(r.selection_id), name=str(r.selection_id),
                       model_price=prices.get((mid, int(r.selection_id))),
                       back=float(r.back1) if r.back1 > 1 else None,
                       back_size=float(np.nan_to_num(r.back1_size)),
                       lay=float(r.lay1) if r.lay1 > 1 else None,
                       last_traded=float(r.last_traded) if r.last_traded > 1 else None,
                       status=str(r.runner_status),
                       traded=0.0 if blind else float(np.nan_to_num(r.runner_total_matched)))
            for r in g.itertuples()]


def reach(snap, prices):
    """Why the rule chose what it did on a book, read blind as the trader reads it: of the races it reads (every
    runner priced by the model), those it skips because the market cannot price a runner (no back, lay or last
    traded price), and on the rest each runner in the price range with its expected CLV at the best back (the
    trader's own plan with the bar set aside: cfg.clv_bar -1), beside the book's back-lay spread and back size."""
    open_bar = replace(cfg, clv_bar=-1.0)
    snap = snap[snap.market_id.isin({m for m, _ in prices})]
    races, unmarketed, evs, best = 0, 0, [], []
    for mid, g in snap.groupby("market_id"):
        views = _views(mid, g, prices, True)
        active = [v for v in views if v.status == "ACTIVE"]
        if len(active) < 2 or any(not (v.model_price and v.model_price > 1) for v in active):
            continue
        races += 1
        if any(_market_price(v) is None for v in active):
            unmarketed += 1
            continue
        e = np.array([d.edge for d in plan_race(views, open_bar, lim.bank)], float)
        e = e[np.isfinite(e)]
        evs.append(e)
        if len(e):
            best.append(e.max())
    e = np.concatenate(evs) if evs else np.array([])
    act = snap[snap.runner_status.eq("ACTIVE")]
    two = act[(act.back1 > 1) & (act.lay1 > 1)]
    return (f"{races} races read, {unmarketed} skipped for a runner the market cannot price; in the rest "
            f"{len(e)} runners in the price range, {int((e >= cfg.clv_bar).sum())} at the {cfg.clv_bar:.0%} bar "
            f"(expected CLV median {np.median(e) if len(e) else np.nan:+.3f}, each race's best: median "
            f"{np.median(best) if best else np.nan:+.3f}, highest {max(best) if best else np.nan:+.3f}); "
            f"lay/back median {(two.lay1 / two.back1).median():.2f} on {len(two)} runners with both, "
            f"size at the best back median GBP{act.back1_size[act.back1 > 1].median():.0f}")


def plan(snap, prices, blind=False):
    """The live rule on each market's book: the trader's own plan, and what the best back's size would match. Like
    the trader, only the win markets the day's model prices carry are read. blind: read the book as the trader's
    delayed feed does, with no matched money (no GBP100 floor per runner, the no-volume closing model); the
    evening recorder's books carry matched money the trader would not see."""
    rows, n_races, unpriced = [], 0, 0
    snap = snap[snap.market_id.isin({m for m, _ in prices})]
    for mid, g in snap.groupby("market_id"):
        views = _views(mid, g, prices, blind)
        active = [v for v in views if v.status == "ACTIVE"]
        if len(active) < 2:
            continue
        n_races += 1
        if any(not (v.model_price and v.model_price > 1) for v in active):
            unpriced += 1
            continue
        for d in plan_race(views, cfg, lim.bank):
            stake = float(np.floor(min(d.target, lim.max_stake, lim.liquidity_share * d.back_size) * 100) / 100)
            ok = stake >= lim.min_stake and stake * (d.price - 1.0) >= lim.min_bsp_liability
            rows.append(dict(market_id=mid, selection_id=int(d.selection_id), price=d.price, target=d.target,
                             size=d.back_size, matched=stake if ok else 0.0, edge=d.edge, t=g.t.iloc[0]))
    return pd.DataFrame(rows), n_races, unpriced


def clv(df, price="price", w="matched", red="reduction"):
    """Stake-weighted CLV against the BSP, as matched and after the later non-runners' reductions, and the
    expected pounds of the hedged backs (stake x CLV after the reductions)."""
    d = df[(df[w] > 0) & (df.bsp > 1)]
    if d.empty:
        return dict(n=0, staked=0.0, clv=np.nan, clv_cut=np.nan, exp_gbp=0.0)
    s = d[w]
    eff = d[price] * (1.0 - d[red]) / d.bsp - 1.0
    return dict(n=len(d), staked=float(s.sum()), clv=float((s * (d[price] / d.bsp - 1.0)).sum() / s.sum()),
                clv_cut=float((s * eff).sum() / s.sum()), exp_gbp=float((s * eff).sum()))


def predictions(day, *catalogues):
    """The day's model prices (the 06:00 run's file, the one the trader reads) keyed by Betfair's ids, matched to
    the record's win markets as the trader matches them (trading.matching.attach_ids: course, off, name)."""
    p = _csv(f"predictions/{day}.csv", dtype={"race_time": str, "market_id": str})
    cat = pd.concat([c for c in catalogues if not c.empty]) if any(not c.empty for c in catalogues) else pd.DataFrame()
    if p.empty or "predicted_bfsp" not in p or cat.empty:
        return {}
    cat = cat[cat.market_type.eq("WIN")] if "market_type" in cat else cat
    cat = _num(cat.copy(), ["selection_id"]).dropna(subset=["selection_id"])
    markets = [Market(market_id=str(mid), venue=str(g.venue.iloc[0]), country=str(g.country.iloc[0]),
                      start=pd.Timestamp(g.market_start_utc.iloc[0]).to_pydatetime(),
                      runners={int(a): str(b) for a, b in zip(g.selection_id, g.runner_name)})
               for mid, g in cat.groupby("market_id")]
    p = attach_ids(p, markets)
    p = _num(p, ["selection_id", "predicted_bfsp"]).dropna(subset=["market_id", "selection_id", "predicted_bfsp"])
    return {(str(r.market_id), int(r.selection_id)): float(r.predicted_bfsp) for r in p.itertuples()
            if r.predicted_bfsp > 1}


def catalogue(day, *tags):
    parts = [_csv(f"betfair_live/{day}/markets{'_' + t if t else ''}.csv.gz", dtype={"market_id": str}) for t in tags]
    return [x for x in parts if not x.empty]


def ledger(day):
    """The live ledger, and each horse's first matched back."""
    led = _csv(f"trading/live/{day}/ledger.csv", dtype={"market_id": str})
    if led.empty:
        return led, led
    led = _num(led, ["selection_id", "matched", "avg_price", "edge"]).dropna(subset=["selection_id"]).copy()
    led["selection_id"] = led.selection_id.astype("int64")
    led["t"] = pd.to_datetime(led.ts, utc=True)
    bk = led[(led.event == "back") & (led.matched > 0)]
    return led, bk.sort_values("t").drop_duplicates(KEY)


# ----------------------------------------------------------------------------------------------------- the check
print("== the replication: the plan on the books the trader read at its first step, against the horses it chose then")
day = LIVE_FROM
while day <= LAST:
    bt = books(f"betfair_live/{day}/books_trader.csv.gz")
    led, _ = ledger(day)
    prices = predictions(day, *catalogue(day, ""))
    if bt.empty or led.empty or not prices:
        print(f"{day}: the trader's books {len(bt)}, ledger {len(led)}, model prices {len(prices)}: not checked")
        day += timedelta(days=1)
        continue
    acts = led[led.event.isin(["back", "skip"])]
    start = acts.t.min()
    first = acts[acts.t <= start + pd.Timedelta(seconds=30)]           # the first step's orders and skips
    theirs = set(zip(first.market_id, first.selection_id))
    seen = bt[bt.t <= start + pd.Timedelta(seconds=5)]
    polls = seen.groupby(["market_id", "polled_utc"], as_index=False).agg(t=("t", "first"))
    pick = polls.sort_values("t", ascending=False).drop_duplicates("market_id")[["market_id", "polled_utc"]]
    snap = seen.merge(pick, on=["market_id", "polled_utc"])
    ours, n_races, unpriced = plan(snap, prices)
    mine = set(zip(ours.market_id, ours.selection_id)) if len(ours) else set()
    both = mine & theirs
    print(f"{day}: the trader's first step {start.tz_convert(UK):%H:%M:%S} UK on books read {_hm(snap.t.min())[-5:]}-"
          f"{_hm(snap.t.max())[-5:]} UTC ({n_races} win races priced, {unpriced} with a runner unpriced): chosen by "
          f"the replication {len(mine)}, by the trader {len(theirs)}, by both {len(both)}")
    if both:
        e = ours.drop_duplicates(KEY).set_index(KEY).edge
        tr = first.drop_duplicates(KEY).set_index(KEY).edge
        k = sorted(both)
        print(f"   the expected CLV of the horses both chose: replication {e.loc[k].mean():+.4f}, "
              f"trader {tr.loc[k].mean():+.4f} (largest gap {float((e.loc[k] - tr.loc[k]).abs().max()):.4f})")
    for label, ks in (("only the replication", sorted(mine - theirs)), ("only the trader", sorted(theirs - mine))):
        if ks:
            print(f"   {label}: {ks[:6]}")
    if n_races:
        print(f"   the book then: {reach(snap, prices)}")
    day += timedelta(days=1)

# ---------------------------------------------------------------------------------------------- the evenings
every, rule, morning, pairs = [], [], [], []
day = FIRST
while day <= LAST:
    ev = books(f"betfair_live/{day}/books_evening.csv.gz")
    if ev.empty:
        print(f"{day}: no evening record")
        day += timedelta(days=1)
        continue
    eve = day - timedelta(days=1)
    bsp, rem = settled(day)
    prices = predictions(day, *catalogue(day, "evening", ""))
    _, fills = ledger(day)
    rem_keys = set(zip(rem.market_id, rem.selection_id))
    print(f"{day}: evening record {_hm(ev.t.min())} to {_hm(ev.t.max())} UTC, {ev.market_id.nunique()} markets; "
          f"BSPs {len(bsp)}, non-runners {len(rem)} ({int(rem.removed_at.isna().sum())} without a time); "
          f"model prices {len(prices)}; morning fills {len(fills)}")
    for name, m in MARKS + [("late", None)]:
        snap = snapshot(ev, eve, m)
        if snap.empty:
            continue
        a = snap[snap.runner_status.eq("ACTIVE") & (snap.back1 > 1)].merge(bsp, on=KEY, how="left")
        a["removed_later"] = [k in rem_keys for k in zip(a.market_id, a.selection_id)]
        a["reduction"] = cut(a, rem)
        every.append(a.assign(mark=name, day=str(day)))
        ours, n_races, unpriced = plan(snap, prices, blind=True)      # as the trader's delayed feed reads it
        seen, _, _ = plan(snap, prices)                                # with the recorder's matched money
        print(f"   {name}: {snap.market_id.nunique()} markets polled {_hm(snap.t.min())[-5:]}-{_hm(snap.t.max())[-5:]}"
              f" UTC, {len(a)} runners priced; the rule read {n_races} races ({unpriced} with a runner unpriced) "
              f"and chose {len(ours)} as the trader would read the book (no matched money), {len(seen)} with the "
              f"recorded matched money (the GBP100 floor a runner)")
        print(f"      why: {reach(snap, prices)}")
        if len(ours):
            ours = ours.merge(bsp, on=KEY, how="left")
            ours["reduction"] = cut(ours, rem)
            rule.append(ours.assign(mark=name, day=str(day), races=n_races, unpriced=unpriced))
            if len(fills):
                pr = ours.merge(fills[KEY + ["avg_price", "matched", "t"]].rename(
                    columns={"avg_price": "m_price", "matched": "m_matched", "t": "m_t"}), on=KEY)
                pr["m_reduction"] = cut(pr, rem, t_col="m_t")
                pairs.append(pr.assign(mark=name, day=str(day)))
    if len(fills):
        # the settled ledger carries its own bsp and clv: the BSP here is the record's, as for the evening's rows
        f = fills.drop(columns=["bsp", "clv"], errors="ignore").merge(bsp, on=KEY, how="left").rename(
            columns={"avg_price": "price"})
        f["reduction"] = cut(f, rem)
        morning.append(f.assign(day=str(day)))
    day += timedelta(days=1)

if every:
    E = pd.concat(every)
    ran = E[E.bsp > 1].copy()
    ran["band"] = pd.cut(ran.bsp, BANDS, right=False, labels=BAND_NAMES)
    ran["ratio"] = ran.back1 / ran.bsp
    ran["clv_cut"] = ran.back1 * (1 - ran.reduction) / ran.bsp - 1
    print("\n== every runner at each evening mark: the best back against the BSP (above 1: a price that shortened)")
    print(ran.groupby(["mark", "band"], observed=True).agg(
        runners=("ratio", "size"), back_bsp_median=("ratio", "median"), clv_level=("ratio", lambda v: (v - 1).mean()),
        clv_level_cut=("clv_cut", "mean"), size_median=("back1_size", "median")).round(3).to_string())
    print("\ntaken out after the mark (share of the runners priced then):")
    print(E.groupby("mark").removed_later.mean().round(3).to_string())
if rule:
    R = pd.concat(rule)
    print("\n== the live rule at each evening mark (the day's model prices, the evening's book)")
    out = []
    for (name, d), g in R.groupby(["mark", "day"]):
        c = clv(g)
        out.append(dict(mark=name, day=d, races=int(g.races.iloc[0]), unpriced=int(g.unpriced.iloc[0]),
                        chosen=len(g), exp_clv=float(g.edge.mean()), target=float(g.target.sum()),
                        matched=c["staked"], fill_share=c["staked"] / float(g.target.sum()), clv=c["clv"],
                        clv_cut=c["clv_cut"], exp_gbp=c["exp_gbp"]))
    print(pd.DataFrame(out).round(4).to_string(index=False))
    print("\npooled over the days:")
    print(pd.DataFrame([dict(mark=name, chosen=len(g), **clv(g)) for name, g in R.groupby("mark")])
          .round(4).to_string(index=False))
if morning:
    M = pd.concat(morning)
    print("\n== the morning as traded (first fills):")
    print(pd.DataFrame([dict(day=d, **clv(g)) for d, g in M.groupby("day")] + [dict(day="all", **clv(M))])
          .round(4).to_string(index=False))
P = pd.concat(pairs) if pairs else pd.DataFrame(columns=["matched", "bsp"])
P = P[(P.matched > 0) & (P.bsp > 1)]
print("\n== horses chosen at the mark and backed in the morning: the evening's price against the morning's fill")
if P.empty:
    print("none yet")
else:
    rows = []
    for name, g in P.groupby("mark"):
        mo = clv(g, price="m_price", w="m_matched", red="m_reduction")
        rows.append(dict(mark=name, horses=len(g), clv_evening=clv(g)["clv_cut"], clv_morning=mo["clv_cut"],
                         evening_above_morning=float((g.price > g.m_price).mean())))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
