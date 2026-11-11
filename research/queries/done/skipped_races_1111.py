"""How many races does the live trader leave alone because the race is not wholly priced? Read-only; a count.

The edge investigation of 11 Nov (reports/edge_investigation_1111.md, lever 1; findings strategy_horse-00,
trading_code-09, serving_code-08): the owner's rule reads a race only when every runner still running has a model
price and a market price (trading/strategy.py _plan_closing_clv returns nothing otherwise). A race left that way
writes no ledger line, so it has never been counted.

For each live day from 1 Oct: the trader's own book reads (betfair_live/<day>/books_trader.csv.gz; on 1 Oct the
morning's are in books.csv.gz under source "trader"), kept where the market is open, not in play and more than 15
minutes from the off (that test also leaves out the reads made only to lay or to settle). Each read is rebuilt as the
trader saw it: the runners ACTIVE, each with the 06:00 file's model price (predictions/<day>.csv, joined as
auto_trade.py joined it on the day: by course, UK start time and name, with the course names as they were matched
until 11 Nov) and a market price (a back, a lay or a last trade). A read is whole when at least two runners are ACTIVE
and each has both prices. Whole means only that the rule priced the race at that read: it then still backs nothing
unless a runner clears its bar, its guard and its limits.

Two windows, because the owner's window changed (commit 0ecb3db; TRADING.md): new bets ran to 15 minutes before each
off (to 21:30 UK) until 4 Nov and stop at 10:00 UK from 5 Nov. "By 10:00" is every day read under today's rule;
"the day's rule" is each day under its own. By race and window: whole at every read, at some, at none. For a race
never whole by 10:00: when it first became whole and when the trader first bet in it. For each runner that broke a
race: why (not in the 06:00 file; in it without a price; in it but not joined), what the catalogue says of it (a
jockey or none), what the morning's card as fetched says of it (S3 racecards/, the last card before the 06:00 file
was written: a runner, a reserve, no jockey named, or not on it), its share of the book while it stood, and what
became of it (betfair_live_marks: removed before the off and by which mark, or ran). Beside it the WIN markets the
trader never loaded and why, and the trader's ledger.

Checks printed with the counts (the review of the first version, 11 Nov): every matched back in the ledger proves the
trader saw that race whole at that poll, so each is matched to the rebuilt read within 30 seconds and must be whole;
the 06:00 file in S3 is the day's last version, so a day whose file was written after the trader's first read is
named; a race classed never whole yet bet in is a contradiction and is counted.

Limits: reads made while the session was stopped or held cannot be told from planning reads; on 7 Oct a session
started late may have stopped at 14:00 (not told apart here). The money printed is the ledger's own settled result:
an indication of scale, not a forecast. The day's stake limit binds on some days, and on such a day one more whole
race adds no stake.

The first run (43104279398, by 10:00 only, before the review) found 85 races never whole, 84 of them Irish, every
runner at fault absent from the 06:00 file: the card leaves out reserves not yet in
(daily_predictions.declared_runners) while Betfair lists them as runners.
"""
import gzip
import io
import os
import re
import sqlite3
import sys
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import betfair_prices as bp  # noqa: E402
from daily_predictions import card_status  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
FIRST, LAST = date(2026, 10, 1), date(2026, 11, 10)
TEN, BEFORE_OFF, DAY_LIMIT = 10 * 60, 15.0, 4000.0
UK = ZoneInfo("Europe/London")
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
CLOCK = ["08:00", "09:00", "10:00", "11:00", "12:00"]
PRE = ["off_120", "off_60", "off_30", "off_15", "off_10", "off_5", "off_3", "off_1", "last", "final"]


def until_of(d):
    """The last UK minute for a new bet on day d (trading/config_live.json: 21:30 until 4 Nov, 10:00 from 5 Nov)."""
    return TEN if d >= date(2026, 11, 5) else 21 * 60 + 30


def track_then(name):
    """betfair_prices.normalise_track as the trader had it until 11 Nov (commit 88f1b21 added the Bangor alias)."""
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", str(name or "").lower()))


def get(key):
    """The object, or None when S3 has none under the key; any other failure is raised with the key."""
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:  # noqa: BLE001
        code = (getattr(e, "response", None) or {}).get("Error", {}).get("Code", "")
        if type(e).__name__ == "NoSuchKey" or code in ("NoSuchKey", "404"):
            return None
        raise RuntimeError(f"{key}: {type(e).__name__} {code}") from e


def csv(key, **kw):
    raw = get(key)
    if raw is None:
        return pd.DataFrame()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), low_memory=False, on_bad_lines="skip", **kw)   # a torn line: as the loader


def written(key):
    try:
        return pd.Timestamp(s3.head_object(Bucket=bucket, Key=key)["LastModified"]).tz_convert("UTC")
    except Exception:  # noqa: BLE001 - the check is named as not made, never guessed
        return pd.NaT


def norm(s, fn):
    return s.astype(str).map(fn)


def hhmm(t):
    return t.dt.tz_convert(UK).dt.strftime("%H:%M")


def ukmin(t):
    u = t.dt.tz_convert(UK)
    return u.dt.hour * 60 + u.dt.minute


def model_prices(p, cat, track):
    """(market_id, selection_id) -> the model's price above 1, as the trader took it (trading/session.py): the file's
    own ids when every row has one, else every row matched by course, UK start time and name (trading/matching.py as
    it stood until 11 Nov). ``track`` normalises the course name."""
    p = p.copy()
    p["predicted_bfsp"] = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    if {"market_id", "selection_id"} <= set(p.columns) and p["market_id"].notna().all():
        how = "the file's own ids"
        q = p[["market_id", "selection_id", "predicted_bfsp"]].copy()
    else:
        how = "by course, time and name"
        venue = p["venue"] if "venue" in p else p["track"]
        name = p["runner_name"] if "runner_name" in p else p["horse_name"]
        k = pd.DataFrame({"_t": norm(venue, track), "_h": p["race_time"].astype(str).map(bp.race_time_to_24h),
                          "_n": norm(name, bp.normalise_horse), "predicted_bfsp": p["predicted_bfsp"]})
        bf = cat.assign(_t=norm(cat["venue"], track))[["market_id", "selection_id", "_t", "_h", "_n"]]
        q = k.merge(bf.drop_duplicates(["_t", "_h", "_n"]), on=["_t", "_h", "_n"], how="left")
    q = q[q["market_id"].notna() & q["selection_id"].notna() & (q["predicted_bfsp"] > 1)].copy()
    q["market_id"] = q["market_id"].astype(str)
    q["selection_id"] = pd.to_numeric(q["selection_id"], errors="coerce").astype("int64")
    return {(m, int(s)): float(v) for m, s, v in zip(q.market_id, q.selection_id, q.predicted_bfsp)}, how


def classify(g):
    """By race over the reads given: reads, whole reads, the class."""
    if g.empty:
        return pd.DataFrame(columns=["market_id", "reads", "whole", "bad_model", "bad_market", "cls"])
    m = g.groupby("market_id").agg(reads=("whole", "size"), whole=("whole", "sum"),
                                   bad_model=("no_model", lambda x: int((x > 0).sum())),
                                   bad_market=("no_market", lambda x: int((x > 0).sum()))).reset_index()
    m["cls"] = np.where(m.whole == m.reads, "every", np.where(m.whole == 0, "none", "some"))
    return m


def one_day(d):
    D = d.isoformat()
    bt = csv(f"betfair_live/{D}/books_trader.csv.gz", dtype={"market_id": str})
    extra = 0
    if d == FIRST or bt.empty:                               # 1 Oct's morning was written before the trader's tag
        b0 = csv(f"betfair_live/{D}/books.csv.gz", dtype={"market_id": str})
        if not b0.empty and "source" in b0.columns:
            b0 = b0[b0["source"].astype(str).eq("trader")]
            extra = len(b0)
            bt = pd.concat([bt, b0], ignore_index=True)
    if bt.empty or "polled_utc" not in bt.columns:
        return None, f"{D}: no trader books"
    parts = []
    for n in ("markets_trader", "markets"):
        c = csv(f"betfair_live/{D}/{n}.csv.gz", dtype={"market_id": str})
        if n == "markets_trader" and not c.empty:
            c["market_type"] = "WIN"                         # the trader asks for WIN only and no description
        parts.append(c)
    cat = pd.concat(parts, ignore_index=True)
    p = csv(f"predictions/{D}.csv", dtype={"market_id": str, "race_time": str})
    if cat.empty or p.empty or "predicted_bfsp" not in p.columns:
        return None, f"{D}: no catalogue or no 06:00 file"
    if "market_type" in cat.columns:
        cat = cat[cat["market_type"].astype(str).str.upper().eq("WIN")]
    cat = cat[cat["country"].isin(["GB", "IE"])].copy()
    cat["selection_id"] = pd.to_numeric(cat["selection_id"], errors="coerce")
    cat["off"] = pd.to_datetime(cat["market_start_utc"], utc=True, errors="coerce")
    cat = cat[cat["selection_id"].notna() & cat["off"].notna()].copy()
    cat["selection_id"] = cat["selection_id"].astype("int64")
    cat = cat.drop_duplicates(["market_id", "selection_id"])
    cat["_h"] = cat["off"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z").map(bp.utc_iso_to_uk_hhmm)
    cat["_n"] = norm(cat["runner_name"], bp.normalise_horse)
    price, how = model_prices(p, cat, track_then)
    price_now, _ = model_prices(p, cat, bp.normalise_track)   # with the course names as matched from 11 Nov
    one = cat.drop_duplicates("market_id").set_index("market_id")
    off, venue, country = one["off"], one["venue"], one["country"]
    ck = list(zip(cat.market_id, cat.selection_id))
    rname = dict(zip(ck, cat.runner_name))
    jockey = dict(zip(ck, cat["jockey"])) if "jockey" in cat.columns else {}
    cloth = dict(zip(ck, cat["cloth_number"])) if "cloth_number" in cat.columns else {}

    pname = p["runner_name"] if "runner_name" in p else p["horse_name"]
    pp = pd.to_numeric(p["predicted_bfsp"], errors="coerce")
    in_file = {}
    for k, v in zip(norm(pname, bp.normalise_horse), (pp > 1).to_numpy()):
        in_file[k] = bool(v) or in_file.get(k, False)

    read_ids = set(bt["market_id"].astype(str))
    not_in_cat = len(read_ids - set(off.index))
    bt = bt[bt["market_id"].isin(set(off.index))].copy()
    bt["t"] = pd.to_datetime(bt["polled_utc"], utc=True, errors="coerce")
    bt = bt[bt["t"].notna()].copy()
    bt["mins"] = (bt["market_id"].map(off) - bt["t"]).dt.total_seconds() / 60.0
    bt["ukmin"] = ukmin(bt["t"])
    inplay = bt["inplay"].astype(str).str.strip().str.lower().isin(["0", "0.0", "false"])
    keep = bt["status"].astype(str).eq("OPEN") & inplay & (bt["mins"] > BEFORE_OFF)
    bt["selection_id"] = pd.to_numeric(bt["selection_id"], errors="coerce")
    r = bt[keep & bt["runner_status"].astype(str).eq("ACTIVE") & bt["selection_id"].notna()].copy()
    if r.empty:
        return None, f"{D}: no trader read before the offs"
    r["selection_id"] = r["selection_id"].astype("int64")
    r = r.drop_duplicates(["market_id", "t", "selection_id"])
    for c in ("back1", "lay1", "last_traded"):
        r[c] = pd.to_numeric(r[c], errors="coerce") if c in r.columns else np.nan
    mp = np.array([price.get((m, int(s)), np.nan) for m, s in zip(r.market_id, r.selection_id)], float)
    r["no_model"] = ~(mp > 1)
    b, la, lt = r.back1.to_numpy(float), r.lay1.to_numpy(float), r.last_traded.to_numpy(float)
    with np.errstate(invalid="ignore"):
        mid = np.where((b > 1) & (la > 1), np.sqrt(b * la),
                       np.where(b > 1, b, np.where(lt > 1, lt, np.where(la > 1, la, np.nan))))
    r["mkt"] = mid                                            # trading/strategy.py _market_price
    r["no_market"] = ~(r.mkt > 1)
    r["q"] = np.where(r.mkt > 1, 1.0 / r.mkt, 0.0)
    r["q_unpriced"] = np.where(r.no_model, r.q, 0.0)
    g = r.groupby(["market_id", "t", "ukmin"]).agg(active=("selection_id", "size"), no_model=("no_model", "sum"),
                                                    no_market=("no_market", "sum"), book=("q", "sum"),
                                                    unpriced=("q_unpriced", "sum")).reset_index()
    g["whole"] = (g.active >= 2) & (g.no_model == 0) & (g.no_market == 0)
    g["share"] = np.where(g.book > 0, g.unpriced / g.book, np.nan)
    until = until_of(d)
    m10, mday = classify(g[g.ukmin <= TEN]), classify(g[g.ukmin <= until])
    m = mday.rename(columns={"cls": "cls_day", "reads": "reads_day", "whole": "whole_day"})[
        ["market_id", "reads_day", "whole_day", "cls_day"]].merge(
        m10.rename(columns={"cls": "cls_10", "reads": "reads_10", "whole": "whole_10"})[
            ["market_id", "reads_10", "whole_10", "cls_10", "bad_model", "bad_market"]], on="market_id", how="outer")
    m["cls_10"] = m["cls_10"].fillna("not read by 10:00")
    m["cls_day"] = m["cls_day"].fillna("not read in the window")
    m["first_whole"] = m.market_id.map(g[g.whole & (g.ukmin <= until)].groupby("market_id")["t"].min())
    m["active"] = m.market_id.map(g.groupby("market_id")["active"].max())
    m["share_10"] = m.market_id.map(g[(g.ukmin <= TEN) & ~g.whole].groupby("market_id")["share"].median())
    m["day"], m["venue"], m["country"] = D, m.market_id.map(venue), m.market_id.map(country)
    m["off"] = m.market_id.map(off)

    # the runners without a model price by 10:00, once each a race, and why
    miss = r[r.no_model & (r.ukmin <= TEN)].groupby(["market_id", "selection_id"]).agg(
        reads=("t", "nunique"), price=("mkt", "median")).reset_index()
    mk = list(zip(miss.market_id, miss.selection_id))
    miss["runner"] = [str(rname.get(k, "")) for k in mk]
    known = miss["runner"].map(bp.normalise_horse).map(lambda k: in_file.get(k))
    miss["reason"] = np.where(known.isna(), "not in the 06:00 file",
                              np.where(known.eq(True), "in the file with a price, not joined to this runner",
                                       "in the file without a price"))
    miss["joined_now"] = [k in price_now for k in mk]         # priced once the course names match (from 11 Nov)
    miss["jockey"] = [str(jockey.get(k, "") if pd.notna(jockey.get(k, "")) else "") for k in mk]
    miss["cloth"] = [str(cloth.get(k, "") if pd.notna(cloth.get(k, "")) else "") for k in mk]
    miss["day"], miss["venue"], miss["country"] = D, miss.market_id.map(venue), miss.market_id.map(country)
    miss["off_uk"] = hhmm(miss.market_id.map(off)) if len(miss) else ""
    # what the morning's card, as fetched, says of each: the last card file before the 06:00 file was written
    wr = written(f"predictions/{D}.csv")
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=bucket, Prefix=f"racecards/{D}_").get("Contents", []))
    stamp = wr.strftime("%H%M") if pd.notna(wr) else "9999"
    before = [k for k in keys if k.rsplit("_", 1)[-1][:4] <= stamp]
    card_key = (before[-1] if before else keys[0]) if keys else ""
    on_card = {}
    if card_key and len(miss):
        card = csv(card_key)
        if not card.empty and "horse_name" in card.columns:
            has_j = "jockey_name" in card.columns
            for nm, j in zip(norm(card["horse_name"], bp.normalise_horse), card["jockey_name"] if has_j else [None] * len(card)):
                on_card.setdefault(nm, {"runner": "on the card as a runner", "reserve": "on the card as a reserve",
                                        "withdrawn": "on the card with no jockey named"}[card_status(j)]
                                   if has_j else "on the card (no jockey column)")
    miss["card"] = [on_card.get(bp.normalise_horse(x), "not on the card" if card_key else "no card file")
                    for x in miss["runner"]]

    # the WIN markets on the catalogue the trader never read before their offs
    first_read = r["t"].min()
    could = off[off > first_read + pd.Timedelta(minutes=BEFORE_OFF)]
    never = sorted(set(could.index) - set(g.market_id))
    then, now = {a for a, _ in price}, {a for a, _ in price_now}
    nv = pd.DataFrame({"day": D, "market_id": never, "venue": [venue.get(x) for x in never],
                       "country": [country.get(x) for x in never],
                       "why": ["a model price on the day, yet never read" if x in then else
                               "no model price: the course name was not matched (fixed 11 Nov)" if x in now else
                               "no model price on any runner" for x in never]})

    # the trader's ledger: each race's first matched back, its stake, the settled result; and the rebuild's check
    led = csv(f"trading/live/{D}/ledger.csv", dtype={"market_id": str})
    first_bet, pnl, held = pd.Series(dtype="datetime64[ns, UTC]"), pd.Series(dtype=float), set()
    staked = staked_10 = np.nan
    no_read = not_whole = backs = 0
    skips = {}
    if not led.empty and "event" in led.columns:
        led["matched_"] = pd.to_numeric(led["matched"], errors="coerce") if "matched" in led.columns else np.nan
        led["t"] = pd.to_datetime(led["ts"], utc=True, errors="coerce")
        bk = led[(led.event == "back") & (led.matched_ > 0) & led.t.notna()].copy()
        held = set(led[led.event.isin(["back", "sync", "settle"]) & (led.matched_ > 0)].market_id.astype(str))
        first_bet = bk.groupby("market_id")["t"].min()
        staked = float(bk.matched_.sum())
        staked_10 = float(bk[ukmin(bk.t) <= TEN].matched_.sum()) if len(bk) else 0.0
        s = led[led.event.isin(["settle", "commission"])]
        if len(s) and "pnl" in s.columns:
            pnl = pd.to_numeric(s["pnl"], errors="coerce").fillna(0.0).groupby(s["market_id"].astype(str)).sum()
        if "error" in led.columns:
            skips = led[led.event == "skip"]["error"].astype(str).str.slice(0, 60).value_counts().to_dict()
        backs = len(bk)
        if backs:
            gg = g[["market_id", "t", "whole"]].rename(columns={"t": "t_read"}).sort_values("t_read")
            chk = pd.merge_asof(bk[["market_id", "t"]].sort_values("t"), gg, left_on="t", right_on="t_read",
                                by="market_id", direction="nearest", tolerance=pd.Timedelta("30s"))
            no_read = int(chk.t_read.isna().sum())
            not_whole = int((chk.t_read.notna() & chk.whole.eq(False)).sum())
    m["first_bet"] = m.market_id.map(first_bet)
    m["bet"] = m.market_id.isin(held)
    m["pnl"] = m.market_id.map(pnl)
    row = {"day": D, "ids": how, "until": f"{until // 60:02d}:{until % 60:02d}", "catalogue": int(off.index.nunique()),
           "read": int(m.shape[0]),
           "every_10": int((m.cls_10 == "every").sum()), "some_10": int((m.cls_10 == "some").sum()),
           "none_10": int((m.cls_10 == "none").sum()), "unread_10": int((m.cls_10 == "not read by 10:00").sum()),
           "every_day": int((m.cls_day == "every").sum()), "some_day": int((m.cls_day == "some").sum()),
           "none_day": int((m.cls_day == "none").sum()), "never_loaded": len(never), "bet": int(m.bet.sum()),
           "staked": staked, "staked_10": staked_10, "ledger_pnl": float(pnl.sum()) if len(pnl) else np.nan,
           "backs": backs, "backs_no_read": no_read, "backs_not_whole": not_whole, "not_in_cat": not_in_cat,
           "from_books": extra, "first_read": first_read, "file_written": wr,
           "file_after_first_read": bool(pd.notna(wr) and wr > first_read), "skips": skips,
           "card_key": card_key}
    return (row, m, miss, nv), None


rows, races, missing, nevers, notes = [], [], [], [], []
d = FIRST
while d <= LAST:
    try:
        out, note = one_day(d)
    except Exception as e:  # noqa: BLE001 - a day that cannot be read is named, never guessed
        out, note = None, f"{d}: {type(e).__name__}: {str(e)[:200]}"
    if out is None:
        notes.append(note)
    else:
        rows.append(out[0])
        races.append(out[1])
        missing.append(out[2])
        nevers.append(out[3])
    d += timedelta(days=1)

print(f"days asked {(LAST - FIRST).days + 1}, days read {len(rows)}; not read:")
for n in notes:
    print("  ", n)
if not rows:
    raise SystemExit("no day could be read")
t = pd.DataFrame(rows)
R = pd.concat(races, ignore_index=True)
M = pd.concat(missing, ignore_index=True)
N = pd.concat(nevers, ignore_index=True)
for c in ("first_whole", "first_bet", "off"):                # a day with none of them leaves the column untyped
    R[c] = pd.to_datetime(R[c], utc=True, errors="coerce")
for c in ("first_read", "file_written"):
    t[c] = pd.to_datetime(t[c], utc=True, errors="coerce")
R["off_uk"] = hhmm(R["off"])
R["new_rule"] = R.day >= "2026-11-05"
R["bet_by_10"] = (ukmin(R["first_bet"]) <= TEN).fillna(False)
t_new = t[t.day >= "2026-11-05"]

print("\nby day. _10: the reads by 10:00 UK (today's rule); _day: the reads in the day's own window (until):")
print(t[["day", "until", "catalogue", "read", "every_10", "some_10", "none_10", "unread_10", "every_day", "some_day",
         "none_day", "never_loaded", "bet", "staked", "staked_10", "ledger_pnl"]].round(0).to_string(index=False))
print("ids:", t.ids.value_counts().to_dict(), "| 1 Oct rows taken from books.csv.gz:", int(t.from_books.sum()),
      "| markets read but not on the catalogue:", int(t.not_in_cat.sum()))

print("\nCHECKS of the rebuild")
print(f"  matched backs in the ledgers: {int(t.backs.sum()):,}; with no rebuilt read within 30 s: "
      f"{int(t.backs_no_read.sum()):,}; on a read rebuilt as NOT whole (should be 0): {int(t.backs_not_whole.sum()):,}")
if t.backs_not_whole.sum() or t.backs_no_read.sum():
    print(t[(t.backs_not_whole > 0) | (t.backs_no_read > 0)][["day", "backs", "backs_no_read", "backs_not_whole"]]
          .to_string(index=False))
late = t[t.file_after_first_read]
print(f"  days whose 06:00 file in S3 was written after the trader's first read (the trader may have held an earlier "
      f"version): {len(late)} of {len(t)}; file time not read on {int(t.file_written.isna().sum())}")
if len(late):
    print(late.assign(first_read_uk=hhmm(late.first_read), file_written_uk=hhmm(late.file_written))[
        ["day", "first_read_uk", "file_written_uk", "none_10", "backs_not_whole"]].to_string(index=False))
c1 = R[(R.cls_day == "none") & R.bet]
c2 = R[(R.cls_10 == "none") & R.bet_by_10]
print(f"  races never whole in the day's window yet bet in (should be 0): {len(c1)}; never whole by 10:00 yet first "
      f"bet by 10:00 (should be 0): {len(c2)}")
if len(c1):
    print(c1[["day", "venue", "off_uk", "country", "reads_day", "whole_day"]].head(30).to_string(index=False))


def line(label, x, col):
    n = len(x)
    c = x[col].value_counts()
    print(f"  {label:34s} races {n:5d}  every read {int(c.get('every', 0)):5d} ({c.get('every', 0) / max(n, 1):.1%})  "
          f"some {int(c.get('some', 0)):4d} ({c.get('some', 0) / max(n, 1):.1%})  "
          f"none {int(c.get('none', 0)):4d} ({c.get('none', 0) / max(n, 1):.1%})")


r10 = R[R.cls_10 != "not read by 10:00"]
print("\nunder today's rule (reads by 10:00 UK), races read by then:")
line("all days", r10, "cls_10")
line("GB", r10[r10.country == "GB"], "cls_10")
line("IE", r10[r10.country == "IE"], "cls_10")
line("5-10 Nov (the rule in force)", r10[r10.new_rule], "cls_10")
line("5-10 Nov, IE", r10[r10.new_rule & (r10.country == "IE")], "cls_10")
print("under each day's own rule (to 21:30 UK until 4 Nov, to 10:00 from 5 Nov):")
rd = R[R.cls_day != "not read in the window"]
line("all days", rd, "cls_day")
line("GB", rd[rd.country == "GB"], "cls_day")
line("IE", rd[rd.country == "IE"], "cls_day")
line("1 Oct - 4 Nov", rd[~rd.new_rule], "cls_day")
print(f"races read in the window but not by 10:00 UK: {int((R.cls_10 == 'not read by 10:00').sum())}")

bad = R[R.cls_10 == "none"].copy()
ndays = max(len(t), 1)
print(f"\nraces never whole by 10:00 UK: {len(bad)} ({len(bad) / ndays:.2f} a day); by what broke them: "
      f"model price {int((bad.bad_model > 0).sum())}, market price {int((bad.bad_market > 0).sum())}")
old = bad[~bad.new_rule]
later = old[old.first_whole.notna()]
print(f"  of the {len(old)} to 4 Nov (window to 21:30): became whole later that day {len(later)}, "
      f"never whole before the off {int(old.first_whole.isna().sum())}; bet later that day {int(old.bet.sum())}")
if len(later):
    fw = ukmin(later.first_whole)
    print(f"  first whole at (UK): median {int(fw.median()) // 60:02d}:{int(fw.median()) % 60:02d}; by hour: "
          f"{(fw // 60).astype(int).value_counts().sort_index().to_dict()}")
    lead = (later.off - later.first_whole).dt.total_seconds() / 60.0
    print(f"  minutes from first whole to the off: median {lead.median():.0f}, 25th-75th pct "
          f"{lead.quantile(.25):.0f}-{lead.quantile(.75):.0f}")
new = bad[bad.new_rule]
print(f"  of the {len(new)} from 5 Nov (window to 10:00): {len(new) / max(len(t_new), 1):.2f} a day; bet "
      f"{int(new.bet.sum())}")
if len(bad):
    print(f"  the unpriced runners' share of the book in those races (median of the reads): median "
          f"{bad.share_10.median():.1%}, 75th pct {bad.share_10.quantile(.75):.1%}, 90th pct "
          f"{bad.share_10.quantile(.9):.1%}, max {bad.share_10.max():.1%}")
    print("  by the hour of the off (UK):", bad.off_uk.str.slice(0, 2).value_counts().sort_index().to_dict())
    print("  by course:", bad.venue.value_counts().to_dict())

print(f"\nrunners ACTIVE with no model price by 10:00 UK (once a race): {len(M)}")
if len(M):
    try:
        con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
        mk = pd.read_sql_query(
            "SELECT market_id, selection_id, mark, runner_status, bsp FROM betfair_live_marks "
            f"WHERE race_date >= '{FIRST}' AND race_date <= '{LAST}'", con)
    except Exception as e:  # noqa: BLE001
        print(f"  [betfair_live_marks not read: {type(e).__name__}: {str(e)[:120]}]")
        mk = pd.DataFrame(columns=["market_id", "selection_id", "mark", "runner_status", "bsp"])
    mk["market_id"] = mk.market_id.astype(str)
    mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
    mk = mk[mk.selection_id.notna()].copy()
    mk["selection_id"] = mk.selection_id.astype("int64")
    st = mk.pivot_table(index=["market_id", "selection_id"], columns="mark", values="runner_status", aggfunc="first")
    bsp = pd.to_numeric(mk[mk.mark == "final"].set_index(["market_id", "selection_id"])["bsp"], errors="coerce")
    bsp = bsp[~bsp.index.duplicated(keep="first")]
    idx = pd.MultiIndex.from_frame(M[["market_id", "selection_id"]])
    order = [c for c in CLOCK + PRE if c in st.columns]
    S = st.reindex(idx)[order] if order else pd.DataFrame(index=idx)
    fin = S["final"].to_numpy() if "final" in S.columns else np.array([None] * len(M), dtype=object)
    M["final"] = pd.Series(fin, index=M.index).fillna("no final mark")
    M["bsp"] = bsp.reindex(idx).to_numpy() if len(bsp) else np.nan
    gone = S.eq("REMOVED") if len(order) else pd.DataFrame(index=idx)
    M["removed_by"] = ([next((c for c in order if row[c]), "") for row in gone.to_dict("records")]
                       if len(order) else [""] * len(M))
    M["ran"] = M.final.isin(["WINNER", "LOSER", "PLACED"]) | (M.bsp > 1)
    M["no_jockey"] = M.jockey.astype(str).str.strip().eq("")
    print("why:", M.reason.value_counts().to_dict())
    print("on the morning's card as fetched, by what became of them:")
    print(pd.crosstab(M.card, np.where(M.ran, "ran", "did not run"), margins=True).to_string())
    print("the card files read (the first three days):", list(t.card_key.head(3)))
    print(f"of these, priced once the course names match (from 11 Nov): {int(M.joined_now.sum())}")
    print("what became of them (the final mark):", M.final.value_counts().to_dict())
    print(f"ran in the race: {int(M.ran.sum())}; removed before the off: {int((M.final == 'REMOVED').sum())}")
    print("for those removed, the first mark at which they were out (UK clock marks, then minutes to the off):",
          M[M.final == "REMOVED"].removed_by.value_counts().reindex(order).dropna().astype(int).to_dict())
    print(f"no jockey on Betfair's catalogue: {int(M.no_jockey.sum())} of {len(M)}; "
          f"by what became of them: {M.groupby('final').no_jockey.mean().round(2).to_dict()}")
    print(f"their market price while they stood: median {M.price.median():.0f}, 25th pct {M.price.quantile(.25):.0f}, "
          f"10th pct {M.price.quantile(.1):.0f}, min {M.price.min():.1f}; with no market price {int(M.price.isna().sum())}")
    print("by country:", M.country.value_counts().to_dict())
    ran = M[M.ran]
    if len(ran):
        print(f"\nthe {len(ran)} that RAN with no model price (a gap in the card or the join, not a reserve), the first 60:")
        print(ran.sort_values(["day", "off_uk"]).head(60)[["day", "venue", "off_uk", "runner", "card", "price", "bsp",
                                                             "final"]].to_string(index=False))
    print("\nexamples of those that did not run (the shortest prices first), 30:")
    print(M[~M.ran].sort_values("price").head(30)[["day", "venue", "off_uk", "runner", "card", "price", "removed_by",
                                                    "final"]].to_string(index=False))

print(f"\nWIN markets on the catalogue the trader never read before their offs: {len(N)}")
if len(N):
    print(N.groupby(["why", "day", "venue"], dropna=False).size().to_string())

# the ledger's own result: scale only
print("\nthe day's stake and its limit:")
print(f"  days with GBP{DAY_LIMIT - 50:,.0f} or more matched (the day limit bound, or nearly): "
      f"{int((t.staked >= DAY_LIMIT - 50).sum())} of {len(t)}; from 5 Nov: "
      f"{int((t_new.staked >= DAY_LIMIT - 50).sum())} of {len(t_new)}")
print(f"  matched a day: median GBP{t.staked.median():,.0f}; from 5 Nov median GBP{t_new.staked.median():,.0f}")
sk = {}
for dct in t.skips:
    for k, v in (dct or {}).items():
        sk[k] = sk.get(k, 0) + int(v)
print("  ledger skip lines by reason (once a runner and reason a day):", dict(sorted(sk.items(), key=lambda kv: -kv[1])[:8]))
led = R[R.pnl.notna()]
print(f"\nthe ledger's settled result by race (settle and commission rows); days with no settled ledger: "
      f"{int(t.ledger_pnl.isna().sum())}")
if len(led):
    print(led.groupby("country").agg(races=("pnl", "size"), pnl=("pnl", "sum"), a_race=("pnl", "mean")).round(2).to_string())
    print("the same by whether the race's first bet was by 10:00 UK:")
    print(led.groupby(["country", "bet_by_10"]).agg(races=("pnl", "size"), pnl=("pnl", "sum"),
                                                    a_race=("pnl", "mean")).round(2).to_string())
print(f"all days read: GBP{t.ledger_pnl.sum():,.2f} over {int(R.bet.sum())} races bet. Not a forecast for the races "
      "left alone: they may differ from the races read, and on a day the limit binds they add no stake (untested).")

print("\nthe races never whole by 10:00 UK from 5 Nov (the rule in force):")
print(new.sort_values(["day", "off_uk"])[["day", "venue", "off_uk", "country", "active", "reads_10", "share_10"]]
      .round(4).to_string(index=False))
