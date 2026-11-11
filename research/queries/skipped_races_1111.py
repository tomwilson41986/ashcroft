"""How many races does the live trader leave alone because the race is not wholly priced? Read-only; a count.

The edge investigation of 11 Nov (reports/edge_investigation_1111.md, lever 1; findings strategy_horse-00,
trading_code-09, serving_code-08): the owner's rule reads a race only when every runner still running has a model
price and a market price (trading/strategy.py _plan_closing_clv returns nothing otherwise). A race left that way
writes no ledger line, so it has never been counted.

For each live day from 1 Oct: the trader's own book reads (betfair_live/<day>/books_trader.csv.gz), kept where the
market is open, not in play, read by 10:00 UK and more than 15 minutes before the off. Each read is rebuilt as the
trader saw it: the runners ACTIVE, each with the 06:00 file's model price (predictions/<day>.csv by market and
selection, as auto_trade.py takes it) and a market price (a back, a lay or a last trade). A read is whole when at
least two runners are ACTIVE and each has both prices. By race: whole at every read, at some, at none; the runners
that broke it and why. Beside it the WIN markets on the day's catalogue that the trader never read, and the trader's
own ledger (the races it bet in, its settled result).

Limits, said here once: the window is taken as "by 10:00 UK" on every day (the owner's window has ended at 10:00
since 1 Oct by TRADING.md; a day with a later window is undercounted, never overcounted); the trader's reads of a
race after its first bet include reads made only to lay, which this cannot tell apart; the money figure at the end is
the ledger's own settled result spread over the races read, an indication of scale and not a forecast.
"""
import gzip
import io
import os
import sys
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import betfair_prices as bp  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
FIRST, LAST = date(2026, 10, 1), date(2026, 11, 10)
UNTIL_UK, BEFORE_OFF = 10 * 60, 15.0
UK = ZoneInfo("Europe/London")
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"


def get(key):
    """The object, or None when S3 has none under the key (any other failure is raised and names the day)."""
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("NoSuchKey", "ClientError"):
            return None
        raise


def csv(key, **kw):
    raw = get(key)
    if raw is None:
        return pd.DataFrame()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), low_memory=False, **kw)


def utc(s):
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t


def norm(s, fn):
    return s.astype(str).map(fn)


def model_prices(p, cat):
    """(market_id, selection_id) -> the model's price, as the trader took it: the file's own ids when every row has
    one, else every row matched by course, UK start time and name (trading/matching.py as it stood until 11 Nov)."""
    p = p.copy()
    p["predicted_bfsp"] = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    if {"market_id", "selection_id"} <= set(p.columns) and p["market_id"].notna().all():
        how = "the file's own ids"
        q = p[["market_id", "selection_id", "predicted_bfsp"]].copy()
    else:
        how = "by course, time and name"
        venue = p["venue"] if "venue" in p else p["track"]
        name = p["runner_name"] if "runner_name" in p else p["horse_name"]
        k = pd.DataFrame({"_t": norm(venue, bp.normalise_track),
                          "_h": p["race_time"].astype(str).map(bp.race_time_to_24h),
                          "_n": norm(name, bp.normalise_horse), "predicted_bfsp": p["predicted_bfsp"]})
        bf = cat[["market_id", "selection_id", "_t", "_h", "_n"]].drop_duplicates(["_t", "_h", "_n"])
        q = k.merge(bf, on=["_t", "_h", "_n"], how="left")
    q = q[q["market_id"].notna() & q["selection_id"].notna()]
    q["market_id"] = q["market_id"].astype(str)
    q["selection_id"] = pd.to_numeric(q["selection_id"], errors="coerce").astype("int64")
    return {(m, int(s)): float(v) for m, s, v in zip(q.market_id, q.selection_id, q.predicted_bfsp)}, how


def one_day(d):
    D = d.isoformat()
    bt = csv(f"betfair_live/{D}/books_trader.csv.gz", dtype={"market_id": str})
    if bt.empty or "polled_utc" not in bt.columns:
        return None, f"{D}: no trader books"
    cat = pd.concat([csv(f"betfair_live/{D}/{n}.csv.gz", dtype={"market_id": str})
                     for n in ("markets_trader", "markets")], ignore_index=True)
    p = csv(f"predictions/{D}.csv", dtype={"market_id": str, "race_time": str})
    if cat.empty or p.empty or "predicted_bfsp" not in p.columns:
        return None, f"{D}: no catalogue or no 06:00 file"
    if "market_type" in cat.columns:
        cat = cat[cat["market_type"].astype(str).str.upper().eq("WIN")]
    cat = cat[cat["country"].isin(["GB", "IE"])].drop_duplicates(["market_id", "selection_id"]).copy()
    cat["selection_id"] = pd.to_numeric(cat["selection_id"], errors="coerce").astype("int64")
    cat["off"] = utc(cat["market_start_utc"])
    cat["_t"] = norm(cat["venue"], bp.normalise_track)
    cat["_h"] = cat["off"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z").map(bp.utc_iso_to_uk_hhmm)
    cat["_n"] = norm(cat["runner_name"], bp.normalise_horse)
    price, how = model_prices(p, cat)
    off = cat.drop_duplicates("market_id").set_index("market_id")["off"]
    venue = cat.drop_duplicates("market_id").set_index("market_id")["venue"]
    country = cat.drop_duplicates("market_id").set_index("market_id")["country"]
    rname = {(m, int(s)): n for m, s, n in zip(cat.market_id, cat.selection_id, cat.runner_name)}

    # every name in the 06:00 file, with whether it carries a price there
    pname = p["runner_name"] if "runner_name" in p else p["horse_name"]
    pp = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    in_file = dict(zip(norm(pname, bp.normalise_horse), (pp > 1).to_numpy()))

    bt = bt[bt["market_id"].isin(set(off.index))].copy()
    bt["t"] = utc(bt["polled_utc"])
    bt["mins"] = (bt["market_id"].map(off) - bt["t"]).dt.total_seconds() / 60.0
    ukt = bt["t"].dt.tz_convert(UK)
    bt["ukmin"] = ukt.dt.hour * 60 + ukt.dt.minute
    keep = (bt["status"].astype(str).eq("OPEN") & bt["inplay"].astype(str).isin(["0", "False", "false"])
            & (bt["ukmin"] <= UNTIL_UK) & (bt["mins"] > BEFORE_OFF) & bt["t"].notna())
    r = bt[keep & bt["runner_status"].astype(str).eq("ACTIVE")].copy()
    if r.empty:
        return None, f"{D}: no trader read in the window"
    r["selection_id"] = pd.to_numeric(r["selection_id"], errors="coerce").astype("int64")
    for c in ("back1", "lay1", "last_traded"):
        r[c] = pd.to_numeric(r[c], errors="coerce") if c in r.columns else np.nan
    mp = np.array([price.get((m, int(s)), np.nan) for m, s in zip(r.market_id, r.selection_id)], float)
    r["has_model"] = mp > 1
    r["has_market"] = ((r.back1 > 1) | (r.lay1 > 1) | (r.last_traded > 1)).to_numpy()
    r["no_model"], r["no_market"] = ~r.has_model, ~r.has_market
    g = r.groupby(["market_id", "t"]).agg(active=("selection_id", "size"), no_model=("no_model", "sum"),
                                          no_market=("no_market", "sum")).reset_index()
    g["whole"] = (g.active >= 2) & (g.no_model == 0) & (g.no_market == 0)
    g["model_only"] = (g.no_model > 0) & (g.no_market == 0)
    g["market_only"] = (g.no_model == 0) & (g.no_market > 0)
    m = g.groupby("market_id").agg(reads=("whole", "size"), whole=("whole", "sum"), first=("t", "min"),
                                   last=("t", "max"), active=("active", "max"),
                                   bad_model=("no_model", lambda x: int((x > 0).sum())),
                                   bad_market=("no_market", lambda x: int((x > 0).sum()))).reset_index()
    m["day"], m["venue"], m["country"] = D, m.market_id.map(venue), m.market_id.map(country)
    m["off_uk"] = m.market_id.map(off).dt.tz_convert(UK).dt.strftime("%H:%M")
    m["cls"] = np.where(m.whole == m.reads, "every read", np.where(m.whole == 0, "no read", "some reads"))
    m["why"] = np.where(m.cls == "every read", "",
                        np.where((m.bad_model > 0) & (m.bad_market > 0), "model and market",
                                 np.where(m.bad_model > 0, "model price", np.where(m.bad_market > 0, "market price",
                                                                                 "under two runners"))))
    # the runners without a model price, once each a race, and why
    miss = r[r.no_model].groupby(["market_id", "selection_id"]).agg(reads=("t", "nunique")).reset_index()
    miss["runner"] = [rname.get((a, int(b)), "") for a, b in zip(miss.market_id, miss.selection_id)]
    key = miss["runner"].astype(str).map(bp.normalise_horse)
    miss["reason"] = np.where(~key.isin(set(in_file)), "not in the 06:00 file",
                              np.where(key.map(in_file).fillna(False).astype(bool),
                                       "in the file with a price, not joined to this runner",
                                       "in the file without a price"))
    miss["day"], miss["venue"] = D, miss.market_id.map(venue)
    miss["off_uk"] = miss.market_id.map(off).dt.tz_convert(UK).dt.strftime("%H:%M")
    miss["race_cls"] = miss.market_id.map(m.set_index("market_id")["cls"])

    # the WIN markets on the catalogue the trader never read in the window
    first_read = r["t"].min()
    could = off[off > first_read + pd.Timedelta(minutes=BEFORE_OFF)]
    never = sorted(set(could.index) - set(m.market_id))
    priced_markets = {a for a, _ in price}
    never_unpriced = [x for x in never if x not in priced_markets]

    # the trader's ledger: the races bet, the settled result
    led = csv(f"trading/live/{D}/ledger.csv", dtype={"market_id": str})
    bet, pnl = set(), np.nan
    if not led.empty and "event" in led.columns:
        b = led[(led.event == "back") & (pd.to_numeric(led.get("matched"), errors="coerce") > 0)]
        bet = set(b.market_id.astype(str))
        s = led[led.event.isin(["settle", "commission"])]
        pnl = float(pd.to_numeric(s.get("pnl"), errors="coerce").fillna(0).sum()) if len(s) else np.nan
    m["bet"] = m.market_id.isin(bet)
    row = {"day": D, "ids": how, "catalogue": int(off.index.nunique()), "read": len(m),
           "every": int((m.cls == "every read").sum()), "some": int((m.cls == "some reads").sum()),
           "none": int((m.cls == "no read").sum()), "never_read": len(never), "never_read_unpriced": len(never_unpriced),
           "bet": int(m.bet.sum()), "runners_no_model": len(miss), "ledger_pnl": pnl,
           "reads": int(g.shape[0]), "reads_whole": int(g.whole.sum())}
    return (row, m, miss), None


rows, races, missing, notes = [], [], [], []
d = FIRST
while d <= LAST:
    try:
        out, note = one_day(d)
    except Exception as e:  # noqa: BLE001 - a day that cannot be read is named, never guessed
        out, note = None, f"{d}: {type(e).__name__}: {str(e)[:160]}"
    if out is None:
        notes.append(note)
    else:
        rows.append(out[0])
        races.append(out[1])
        missing.append(out[2])
    d += timedelta(days=1)

print(f"days asked {(LAST - FIRST).days + 1}, days read {len(rows)}; not read:")
for n in notes:
    print("  ", n)
if not rows:
    raise SystemExit("no day could be read")
t = pd.DataFrame(rows)
R = pd.concat(races, ignore_index=True)
M = pd.concat(missing, ignore_index=True)
print("\nby day (races the trader read in its window: whole at every read, at some, at none):")
print(t[["day", "catalogue", "read", "every", "some", "none", "never_read", "never_read_unpriced", "bet",
         "runners_no_model", "ledger_pnl", "ids"]].to_string(index=False))

n = len(R)
print(f"\nall days: {n} races read; whole at every read {int((R.cls == 'every read').sum())} "
      f"({(R.cls == 'every read').mean():.1%}), at some {int((R.cls == 'some reads').sum())} "
      f"({(R.cls == 'some reads').mean():.1%}), at none {int((R.cls == 'no read').sum())} "
      f"({(R.cls == 'no read').mean():.1%})")
print(f"reads: {int(t.reads.sum()):,}; whole {int(t.reads_whole.sum()):,} ({t.reads_whole.sum() / t.reads.sum():.1%})")
print(f"catalogue WIN markets the trader never read in the window: {int(t.never_read.sum())}, of which with no "
      f"model price on any runner: {int(t.never_read_unpriced.sum())}")
print("\nraces not whole at every read, by what broke them:")
print(R[R.cls != "every read"].groupby(["cls", "why"]).size().to_string())
print("\nby country:")
print(R.groupby(["country", "cls"]).size().unstack(fill_value=0).to_string())
print("\nby the hour of the off (UK):")
print(R.assign(hour=R.off_uk.str.slice(0, 2)).groupby(["hour", "cls"]).size().unstack(fill_value=0).to_string())
part = R[R.cls == "some reads"]
if len(part):
    print(f"\nraces whole at some reads: {len(part)}; share of their reads that were whole: median "
          f"{(part.whole / part.reads).median():.1%}, mean {(part.whole / part.reads).mean():.1%}")
print(f"\nrunners ACTIVE with no model price (once a race): {len(M)}")
if len(M):
    print(M.groupby("reason").size().to_string())
    print("\nthe same by the race's class:")
    print(M.groupby(["race_cls", "reason"]).size().to_string())
    print("\nexamples (the most reads first):")
    print(M.sort_values("reads", ascending=False).head(60)[["day", "venue", "off_uk", "runner", "reason", "reads",
                                                             "race_cls"]].to_string(index=False))

# an indication of scale only: the ledger's settled result over the races the trader could read at some time
readable = int((R.cls != "no read").sum())
lost = int((R.cls == "no read").sum())
pnl = float(t.ledger_pnl.sum(skipna=True))
print(f"\nledger result over the days read: GBP{pnl:,.2f} over {readable} races whole at some read "
      f"(GBP{pnl / max(readable, 1):.2f} a race); bet in {int(R.bet.sum())} of them")
print(f"races never whole: {lost} ({lost / max(len(t), 1):.1f} a day). At the same result a race that would be "
      f"GBP{lost * pnl / max(readable, 1):,.0f} over the period: an indication of scale, not a forecast "
      f"(untested: the races left alone may differ from the races read)")
print("\nthe races never whole (the first 80):")
print(R[R.cls == "no read"].sort_values(["day", "off_uk"]).head(80)[["day", "venue", "off_uk", "country", "active",
                                                                     "reads", "why", "bet"]].to_string(index=False))
