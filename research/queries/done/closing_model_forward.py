"""The closing model forward (read-only): on every runner the live trader read, does its expected CLV come true?

The trader backs a runner when its expected CLV at the best back price is at least +3% under the closing model
(trading/strategy.py _plan_closing_clv; the delayed key's feed has no volume, so the model fitted without it). For
each live day with the trader's own books (books_trader, from 2 Oct) and each hour mark from 08:00 to 20:00 UK,
this takes each market's book as the trader read it nearest the mark, recomputes every runner's expected CLV
exactly as the trader does (the same inputs, model, draws and seed; checked against the trader's own choices in
evening_entry_check, 2-3 Oct: identical), and sets it against the CLV that came true at that back price: back /
BSP - 1, and after the reductions of later non-runners. Markets within 15 minutes of the off are left out, as the
trader leaves them. Read: is expected CLV calibrated (realised against expected, by band), does the +3% bar pick
runners that close in our favour, and how does that change with the hour and the time to the off. A runner is
counted at every mark it was read at, so marks are not independent; the days are few.
"""

from __future__ import annotations

import gzip
import io
import os
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402
from model import race_book as rb  # noqa: E402
from trading.config import load_config  # noqa: E402
from trading.exchange import Market  # noqa: E402
from trading.matching import attach_ids  # noqa: E402
from trading.strategy import RunnerView, _closing_model, _market_price  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
UK = "Europe/London"
FIRST = date.fromisoformat(os.environ.get("FORWARD_FROM") or "2026-10-02")   # the first day with the trader's books
LAST = date.fromisoformat(os.environ["FORWARD_TO"]) if os.environ.get("FORWARD_TO") else date.today() - timedelta(days=1)
MARKS = list(range(8, 21))                                                    # 08:00 ... 20:00 UK
TOL = 5                                                                       # minutes either side of a mark
KEY = ["market_id", "selection_id"]
EV_EDGES = [-1.0, -0.10, -0.05, -0.02, 0.0, 0.03, 0.06, 0.10, 0.20, 10.0]
EV_NAMES = ["<-10%", "-10..-5", "-5..-2", "-2..0", "0..3", "3..6", "6..10", "10..20", "20%+"]
PRICE_EDGES, PRICE_NAMES = [1, 3, 5, 8, 13, 21, 1e9], ["<3", "3-5", "5-8", "8-13", "13-21", "21+"]
TO_OFF_EDGES = [15, 60, 180, 360, 1e9]
TO_OFF_NAMES = ["15-60m", "1-3h", "3-6h", "6h+"]
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


def books(key):
    b = _csv(key, dtype={"market_id": str, "polled_utc": str})
    if b.empty:
        return b
    b = _num(b, ["selection_id", "back1", "back1_size", "lay1", "last_traded", "runner_total_matched", "sp_actual",
                 "adjustment_factor"]).dropna(subset=["selection_id"]).copy()
    b["selection_id"] = b.selection_id.astype("int64")
    b["t"] = pd.to_datetime(b.polled_utc, utc=True)
    return b


def catalogue(day):
    parts = [_csv(f"betfair_live/{day}/markets{t}.csv.gz", dtype={"market_id": str}) for t in ("", "_trader")]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts) if parts else pd.DataFrame()


def predictions(day, cat):
    """The 06:00 run's model prices keyed by Betfair's ids, matched as the trader matches them."""
    p = _csv(f"predictions/{day}.csv", dtype={"race_time": str, "market_id": str})
    if p.empty or "predicted_bfsp" not in p or cat.empty:
        return {}
    c = cat[cat.market_type.eq("WIN")] if "market_type" in cat else cat
    c = _num(c.copy(), ["selection_id"]).dropna(subset=["selection_id", "runner_name"])
    markets = [Market(market_id=str(mid), venue=str(g.venue.iloc[0]), country=str(g.country.iloc[0]),
                      start=pd.Timestamp(g.market_start_utc.iloc[0]).to_pydatetime(),
                      runners={int(a): str(b) for a, b in zip(g.selection_id, g.runner_name)})
               for mid, g in c.groupby("market_id")]
    p = _num(attach_ids(p, markets), ["selection_id", "predicted_bfsp"])
    p = p.dropna(subset=["market_id", "selection_id", "predicted_bfsp"])
    return {(str(r.market_id), int(r.selection_id)): float(r.predicted_bfsp) for r in p.itertuples()
            if r.predicted_bfsp > 1}


def settled(day):
    """BSPs and non-runners (removal time, reduction factor) from Betfair's final books; price files for BSPs the
    final books lack."""
    f = books(f"betfair_live/{day}/books.csv.gz")
    bsp = pd.DataFrame(columns=KEY + ["bsp"])
    rem = pd.DataFrame(columns=KEY + ["removed_at", "rf"])
    if not f.empty:
        f = f[f.source == "final"].drop_duplicates(KEY, keep="last").copy()
        bsp = f.loc[f.sp_actual > 1, KEY + ["sp_actual"]].rename(columns={"sp_actual": "bsp"})
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


def cut(rows, rem):
    """The reduction on a back matched at t: later non-runners' factors of 2.5% or more, summed, at most 90%."""
    if rem.empty or rows.empty:
        return np.zeros(len(rows))
    r = rem[rem.rf >= 2.5]
    return np.array([min(float(r.rf[(r.market_id == m) & (r.removed_at > t)].sum()), 90.0) / 100.0
                     for m, t in zip(rows.market_id, rows.t)])


def expected(g, prices):
    """Every active runner's expected CLV at its best back, exactly as the trader computes it for the race."""
    live = [RunnerView(selection_id=int(r.selection_id), name=str(r.selection_id),
                       model_price=prices.get((r.market_id, int(r.selection_id))),
                       back=float(r.back1) if r.back1 > 1 else None, back_size=float(np.nan_to_num(r.back1_size)),
                       lay=float(r.lay1) if r.lay1 > 1 else None,
                       last_traded=float(r.last_traded) if r.last_traded > 1 else None,
                       status=str(r.runner_status), traded=float(np.nan_to_num(r.runner_total_matched)))
            for r in g.itertuples()]
    live = [v for v in live if v.status == "ACTIVE"]
    if len(live) < 2 or any(not (v.model_price and v.model_price > 1) for v in live):
        return None
    fallback = [_market_price(v) for v in live]
    if any(p is None for p in fallback):
        return None
    back = np.array([v.back if (v.back and v.back > 1) else np.nan for v in live], float)
    lay = np.array([v.lay if (v.lay and v.lay > 1) else np.nan for v in live], float)
    traded = np.array([max(float(v.traded or 0.0), 0.0) for v in live], float)
    blind = not (traded > 0).any()
    model = _closing_model(cfg.closing_model_novol if blind else cfg.closing_model)
    with np.errstate(invalid="ignore", divide="ignore"):
        market = np.where(np.isfinite(back), rb.market_now(back, lay), np.array(fallback, float))
    rng = np.random.default_rng(sum(int(v.selection_id) for v in live) % (2 ** 32))
    ev, exp_bsp = rb.race_expected_clv(back, lay, np.array([v.model_price for v in live], float), traded, model,
                                       n_draws=cfg.clv_draws, rng=rng, market=market)
    return pd.DataFrame({"selection_id": [v.selection_id for v in live], "back": back,
                         "back_size": [v.back_size for v in live], "model_price": [v.model_price for v in live],
                         "ev": ev, "exp_bsp": exp_bsp, "blind": blind})


rows = []
day = FIRST
while day <= LAST:
    bt = books(f"betfair_live/{day}/books_trader.csv.gz")
    cat = catalogue(day)
    prices = predictions(day, cat)
    bsp, rem = settled(day)
    if bt.empty or not prices or bsp.empty:
        print(f"{day}: the trader's books {len(bt)}, model prices {len(prices)}, BSPs {len(bsp)}: not read")
        day += timedelta(days=1)
        continue
    offs = (cat.drop_duplicates("market_id").set_index("market_id").market_start_utc
            .map(lambda s: pd.Timestamp(s)).to_dict())
    bt = bt[bt.market_id.isin({m for m, _ in prices})]
    n0 = len(rows)
    for h in MARKS:
        target = pd.Timestamp(day, tz=UK) + pd.Timedelta(hours=h)
        x = bt[(bt.t - target).abs() <= pd.Timedelta(minutes=TOL)]
        if x.empty:
            continue
        polls = x.groupby(["market_id", "polled_utc"], as_index=False).agg(t=("t", "first"))
        polls["gap"] = (polls.t - target).abs()
        pick = polls.sort_values("gap").drop_duplicates("market_id")[["market_id", "polled_utc"]]
        snap = x.merge(pick, on=["market_id", "polled_utc"])
        for mid, g in snap.groupby("market_id"):
            off = offs.get(mid)
            t = g.t.iloc[0]
            if off is None or (off - t) <= pd.Timedelta(minutes=15):
                continue
            e = expected(g, prices)
            if e is None:
                continue
            e["market_id"], e["t"], e["mark"], e["day"] = mid, t, f"{h:02d}:00", str(day)
            e["to_off"] = (off - t).total_seconds() / 60.0
            rows.append(e)
    print(f"{day}: {len(rows) - n0} race-marks read; BSPs {len(bsp)}, non-runners {len(rem)}")
    if len(rows) > n0:
        part = pd.concat(rows[n0:])
        part = part.merge(bsp, on=KEY, how="left")
        part["reduction"] = cut(part, rem)
        rows[n0:] = [part]
    day += timedelta(days=1)

if not rows:
    raise SystemExit("nothing to read")
D = pd.concat(rows, ignore_index=True)
D = D[np.isfinite(D.back) & (D.back > 1) & np.isfinite(D.ev) & (D.bsp > 1)].copy()
D["clv"] = D.back / D.bsp - 1.0
D["clv_cut"] = D.back * (1.0 - D.reduction) / D.bsp - 1.0
D["ev_band"] = pd.cut(D.ev, EV_EDGES, labels=EV_NAMES)
D["price_band"] = pd.cut(D.back, PRICE_EDGES, right=False, labels=PRICE_NAMES)
D["off_band"] = pd.cut(D.to_off, TO_OFF_EDGES, labels=TO_OFF_NAMES)
D["chosen"] = (D.ev >= cfg.clv_bar) & (D.back >= lim.min_price) & (D.back <= lim.max_price)
D["stake"] = np.where(D.chosen, np.minimum(cfg.target / (D.back - 1.0), lim.max_stake), 0.0)
print(f"\n{len(D):,} runner-marks with a back price, a BSP and an expected CLV: {D.selection_id.nunique():,} runners in "
      f"{D.market_id.nunique()} races over {D.day.nunique()} days; feed without volume in {D.blind.mean():.0%}")


def table(g):
    return pd.Series({"n": len(g), "ev_mean": g.ev.mean(), "clv_mean": g.clv.mean(), "clv_cut_mean": g.clv_cut.mean(),
                      "clv_median": g.clv.median(), "closed_shorter": (g.clv > 0).mean()})


print("\n== calibration: expected CLV against the CLV that came true, by expected-CLV band (level, every runner)")
print(D.groupby("ev_band", observed=True).apply(table).round(4).to_string())
fit = np.polyfit(D.ev.clip(-0.5, 1.0), D.clv.clip(-0.9, 3.0), 1)
print(f"realised on expected (clipped), slope {fit[0]:.3f}, intercept {fit[1]:+.4f}")

C = D[D.chosen]
print(f"\n== the live rule's runners (expected CLV >= {cfg.clv_bar:.0%}): {len(C):,} runner-marks, "
      f"{C[KEY].drop_duplicates().shape[0]} runners")
w = C.stake
print(f"   level: expected {C.ev.mean():+.2%}, came true {C.clv.mean():+.2%}, after reductions {C.clv_cut.mean():+.2%}; "
      f"staked to win GBP{cfg.target:.0f}: expected {(w * C.ev).sum() / w.sum():+.2%}, came true "
      f"{(w * C.clv_cut).sum() / w.sum():+.2%}")
for col, title in (("mark", "by hour"), ("off_band", "by time to the off"), ("price_band", "by back price"),
                   ("day", "by day")):
    print(f"\n   {title}:")
    print(C.groupby(col, observed=True).apply(lambda g: pd.Series({
        "n": len(g), "ev_mean": g.ev.mean(), "clv_cut_mean": g.clv_cut.mean(),
        "clv_cut_staked": (g.stake * g.clv_cut).sum() / g.stake.sum(),
        "closed_shorter": (g.clv > 0).mean(), "size_median": g.back_size.median()})).round(4).to_string())

F = C.sort_values("t").drop_duplicates(KEY)                      # each runner at the first mark it crossed the bar
w = F.stake
print(f"\n== each runner at the first mark it crossed the bar (the trader's first back, at the book's price): "
      f"{len(F)} runners; level expected {F.ev.mean():+.2%}, came true {F.clv_cut.mean():+.2%}; staked expected "
      f"{(w * F.ev).sum() / w.sum():+.2%}, came true {(w * F.clv_cut).sum() / w.sum():+.2%}")
print(F.groupby("mark").apply(lambda g: pd.Series({
    "n": len(g), "ev_mean": g.ev.mean(), "clv_cut_mean": g.clv_cut.mean(),
    "clv_cut_staked": (g.stake * g.clv_cut).sum() / g.stake.sum()})).round(4).to_string())
L = C.merge(F[KEY + ["t"]].rename(columns={"t": "t_first"}), on=KEY)
L = L[L.t > L.t_first]
if len(L):
    print(f"   the same runners at later marks (where top-ups go): {len(L)} runner-marks, expected {L.ev.mean():+.2%}, "
          f"came true {L.clv_cut.mean():+.2%}")
print("   (the book's price is the delayed feed's: a back sent at it fills at a better price on a drifter and misses a "
      "steamer; the ledger's fills, not this, are the trade's result)")

print("\n== every runner by hour: expected and realised CLV of the whole field (does the market's drift change?)")
print(D.groupby("mark").apply(lambda g: pd.Series({"n": len(g), "ev_mean": g.ev.mean(), "clv_mean": g.clv.mean(),
                                                    "chosen_share": g.chosen.mean()})).round(4).to_string())
