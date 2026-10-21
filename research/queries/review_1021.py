"""The owner's asks (21 Oct): "do a full review of all data we have since we started. Are there certain race types we
should avoid? Is it better to wait for BFSP or lay off earlier? e.g. 10:30 or 12:00" (read-only). Every live position
of 30 Sep - 20 Oct settled with a BSP.

A. Race types: the race from Betfair's own catalogue (betfair_live_markets: the race type Betfair gives, the market's
   name for the handicap and the race's kind, the venue for turf against all-weather, the country, the field, the
   trip). By each: races, horses, money, the expected CLV at entry, the CLV at the BSP with a 90% interval by races
   (bootstrap), CLV x stake and the result; 30 Sep - 20 Oct and the rule of 9 Oct (10-20 Oct) apart.
B. Lay off earlier: each position laid at the exchange at a clock time (10:00, 10:30, 11:00, 12:00, 13:00 UK) instead of
   at the SP, for the positions wholly backed by then whose race was still 15 minutes or more away. The lay as now
   (liability = the back's winnings), at the best lays the trader read at that minute, down the three levels as their
   money allows, the rest at the SP. Scored by the expected value it locks, S(P/X - 1) with X the price laid at (the BSP
   for the rule), and by the result as the horses ran, before and after 2% commission.
"""
import io
import os
import re
import sqlite3
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
K = ["day", "market_id", "selection_id"]
DAYS = [(date(2026, 9, 30) + timedelta(n)).isoformat() for n in range(21)]
AW = {"kempton", "wolverhampton", "lingfield", "chelmsford", "chelmsfordcity", "southwell", "newcastle", "dundalk"}


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


leds, books = [], []
BOOK = ["polled_utc", "market_id", "selection_id", "status", "inplay", "lay1", "lay1_size", "lay2", "lay2_size", "lay3",
        "lay3_size"]
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        continue
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = day
    leds.append(d)
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is not None:
        bk = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in BOOK, dtype={"market_id": str},
                         low_memory=False)
        bk["day"] = day
        books.append(bk)
led = pd.concat(leds, ignore_index=True)
for c in ("matched", "avg_price", "bsp", "selection_id", "edge", "pnl", "commission"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
b["sxp"], b["exs"] = b.matched * b.avg_price, b.matched * b.edge
pos = b.groupby(K).agg(S=("matched", "sum"), sxp=("sxp", "sum"), exs=("exs", "sum"), last_fill=("t", "max"),
                       venue=("venue", "first"), off=("off", "first")).reset_index()
pos["Pm"], pos["ev"] = pos.sxp / pos.S, pos.exs / pos.S
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price", "pnl"]] \
    .rename(columns={"avg_price": "P", "bsp": "B"})
v = pos.merge(st, on=K)
v = v[v.result.isin(["WINNER", "LOSER"]) & (v.B > 1) & (v.P > 1)].copy()
v["clv"] = v.P / v.B - 1
v["exp"] = v.S * v.clv
v["selection_id"] = v.selection_id.astype("int64")
print(f"positions settled with a BSP, 30 Sep - 20 Oct: {len(v):,} in {v.market_id.nunique():,} races, GBP{v.S.sum():,.2f}; "
      f"CLV {v.exp.sum() / v.S.sum():+.2%}; result GBP{v.pnl.sum():,.2f} before commission")

# ------------------------------------------------------------------ A. the race types
mk = pd.read_sql_query("""SELECT market_id, market_name, race_type, venue AS mvenue, country, COUNT(*) AS runners
                          FROM betfair_live_markets WHERE race_date >= '2026-09-30'
                          AND UPPER(COALESCE(market_type, 'WIN')) = 'WIN' GROUP BY market_id""", conn)
fin = pd.read_sql_query("""SELECT market_id, SUM(runner_status IN ('WINNER', 'LOSER')) AS ran FROM betfair_live_marks
                           WHERE mark = 'final' AND race_date >= '2026-09-30' GROUP BY market_id""", conn)
mk = mk.merge(fin, on="market_id", how="left")
v = v.merge(mk, on="market_id", how="left")
print(f"   with Betfair's catalogue: {v.market_name.notna().mean():.1%} of positions")
nm = v.market_name.fillna("").str.lower()
rt = v.race_type.fillna("").str.lower()
jumps = rt.str.contains("hurdle|chase|bumper|nh") | nm.str.contains(r"hrd|chs|nhf|hurdle|chase|bumper|inhf")
v["code"] = np.where(jumps, np.where(nm.str.contains("chs|chase") | rt.str.contains("chase"), "jumps: chase",
                                     np.where(nm.str.contains("nhf|bumper|inhf") | rt.str.contains("bumper"),
                                              "jumps: NH flat", "jumps: hurdle")), "flat")
v["handicap"] = np.where(nm.str.contains("hcap|handicap|nursery"), "handicap", "not a handicap")
kinds = [("maiden", r"mdn|maiden"), ("novice", r"nov"), ("nursery", r"nursery"), ("claimer/seller", r"claim|clm|sell|slr"),
         ("beginners", r"beg"), ("group/listed", r"grp|group|listed|list\b|stks? \(l\)"), ("conditions/stakes", r"cond|stks|stakes")]
v["kind"] = "other"
for name, pat in kinds[::-1]:
    v.loc[nm.str.contains(pat) & v.kind.eq("other"), "kind"] = name
v.loc[v.kind.eq("other") & v.handicap.eq("handicap"), "kind"] = "handicap (plain)"
venue = v.mvenue.fillna(v.venue).fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
v["surface"] = np.where(v.code.eq("flat") & venue.isin(AW), "flat all-weather",
                        np.where(v.code.eq("flat"), "flat turf", "jumps (turf)"))
v["country_"] = v.country.fillna("?")
field = v.ran.fillna(v.runners)
v["field"] = pd.cut(field, [0, 6, 9, 12, 16, 100], labels=["2-6", "7-9", "10-12", "13-16", "17+"])


def furlongs(name):
    m = re.search(r"(\d+)m(?:\s*(\d+)f)?|(\d+)f", name)
    if not m:
        return np.nan
    if m.group(3):
        return float(m.group(3))
    return 8.0 * float(m.group(1)) + (float(m.group(2)) if m.group(2) else 0.0)


v["furlongs"] = nm.map(furlongs)
v["trip"] = np.select([v.code.eq("flat") & (v.furlongs <= 7), v.code.eq("flat") & (v.furlongs <= 10),
                       v.code.eq("flat") & (v.furlongs > 10), ~v.code.eq("flat") & (v.furlongs <= 17),
                       ~v.code.eq("flat") & (v.furlongs <= 22), ~v.code.eq("flat") & (v.furlongs > 22)],
                      ["flat 5-7f", "flat 1m-1m2f", "flat 1m3f+", "jumps to 2m1f", "jumps 2m2f-2m6f", "jumps 2m7f+"], "?")
v["band"] = pd.cut(v.Pm, [1, 3, 5, 10, 21, 50, 201], right=False, labels=["<3", "3-5", "5-10", "10-21", "21-50", "50-200"])
rng = np.random.default_rng(0)


def group(g):
    races = g.groupby("market_id").agg(s=("S", "sum"), e=("exp", "sum"), r=("pnl", "sum"))
    boots = []
    for _ in range(400):
        x = races.iloc[rng.integers(0, len(races), len(races))]
        boots.append(x.e.sum() / x.s.sum())
    return pd.Series({"races": len(races), "horses": len(g), "staked": g.S.sum(), "share": g.S.sum() / v.S.sum(),
                      "ev_entry": (g.ev * g.S).sum() / g.S.sum(), "clv": g.exp.sum() / g.S.sum(),
                      "clv_lo90": np.quantile(boots, 0.05), "clv_hi90": np.quantile(boots, 0.95),
                      "clv_x_stake": g.exp.sum(), "result": g.pnl.sum()})


for col, title in (("code", "code"), ("handicap", "handicap"), ("kind", "the race's kind"), ("surface", "surface"),
                   ("country_", "country"), ("field", "runners"), ("trip", "trip"), ("band", "the price backed at")):
    print(f"\n== A. by {title}: 30 Sep - 20 Oct")
    print(v.groupby(col, observed=True).apply(group).round(4).to_string())
    w = v[v.day >= "2026-10-10"]
    t = w.groupby(col, observed=True).apply(group)[["races", "staked", "clv", "clv_lo90", "clv_hi90", "result"]]
    print(f"   the rule of 9 Oct (10-20 Oct):")
    print(t.round(4).to_string())
print("\n== A. code x handicap, and code x the price backed at (CLV, money), 30 Sep - 20 Oct")
for a, b_ in (("code", "handicap"), ("code", "band")):
    t = v.groupby([a, b_], observed=True).apply(lambda g: pd.Series({"staked": g.S.sum(), "clv": g.exp.sum() / g.S.sum()}))
    print(t.unstack(b_).round(3).to_string())

# ------------------------------------------------------------------ B. lay off earlier
print("\n== B. lay off at the exchange at a clock time instead of at the SP")
bk = pd.concat(books, ignore_index=True)
for c in ("selection_id", "lay1", "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size"):
    bk[c] = pd.to_numeric(bk[c], errors="coerce")
bk = bk.dropna(subset=["selection_id"])
bk["selection_id"] = bk.selection_id.astype("int64")
bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
bk = bk[(bk.status == "OPEN") & ~ip & (bk.lay1 > 1)].sort_values("t")
v["off_t"] = pd.to_datetime(v.day + " " + pd.to_datetime(v.off, errors="coerce").dt.strftime("%H:%M"),
                            errors="coerce").dt.tz_localize("Europe/London").dt.tz_convert("UTC")
rows = []
for hhmm in ("10:00", "10:30", "11:00", "12:00", "13:00"):
    sub = []
    for day, g in v.groupby("day"):
        T = pd.Timestamp(f"{day} {hhmm}").tz_localize("Europe/London").tz_convert("UTC")
        g = g[(g.last_fill <= T) & (g.off_t >= T + pd.Timedelta("15min"))].copy()
        if g.empty:
            continue
        rd = bk[(bk.day == day) & (bk.t <= T) & (bk.t >= T - pd.Timedelta("3min"))]
        rd = rd.drop_duplicates(["market_id", "selection_id"], keep="last")
        g = g.merge(rd[["market_id", "selection_id", "lay1", "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size"]],
                    on=["market_id", "selection_id"], how="left")
        g["T"] = hhmm
        sub.append(g)
    if not sub:
        continue
    x = pd.concat(sub, ignore_index=True)
    have = x.lay1.notna()
    W = (x.S * (x.P - 1)).to_numpy()                                    # the winnings the lay must cover
    laid_liab = np.zeros(len(x))
    laid_stake = np.zeros(len(x))
    for lv in (1, 2, 3):
        p_ = x[f"lay{lv}"].to_numpy(float)
        s_ = x[f"lay{lv}_size"].fillna(0).to_numpy(float)
        okp = np.isfinite(p_) & (p_ > 1)
        room = np.maximum(W - laid_liab, 0)
        take = np.where(okp, np.minimum(s_, room / np.where(okp, p_ - 1, 1)), 0)
        laid_stake += take
        laid_liab += take * np.where(okp, p_ - 1, 0)
    rest = np.maximum(W - laid_liab, 0)                                 # the rest laid at the SP, as now
    B = x.B.to_numpy(float)
    S = x.S.to_numpy(float)
    win = x.result.eq("WINNER").to_numpy()
    early = np.where(win, 0.0, -S + laid_stake + rest / (B - 1))       # a winner nets nothing, as now
    now = np.where(win, 0.0, -S + W / (B - 1))
    # the expected value under the BSP's chances: the winner nets nothing, so the loser's profit times (1 - 1/B)
    e_early = (-S + laid_stake + rest / (B - 1)) * (1 - 1 / B)
    e_now = (-S + W / (B - 1)) * (1 - 1 / B)
    x["early"], x["now"], x["e_early"], x["e_now"] = early, now, e_early, e_now
    x["share_at_exchange"] = laid_liab / np.where(W > 0, W, 1)
    mkt = x.groupby(["day", "market_id"])[["early", "now"]].sum()
    comm = lambda s: (0.02 * s.clip(lower=0)).sum()                    # noqa: E731
    rows.append(pd.Series({"positions": len(x), "with_a_lay_book": int(have.sum()), "staked": S.sum(),
                           "laid_at_the_exchange_share": float(np.average(x.share_at_exchange, weights=W)),
                           "median_lay_over_bsp": float(np.nanmedian(x.lay1 / x.B)),
                           "share_lay_under_bsp": float(np.nanmean((x.lay1 < x.B)[have])),
                           "expected_at_sp_now": e_now.sum(), "expected_lay_early": e_early.sum(),
                           "result_at_sp_now": now.sum(), "result_lay_early": early.sum(),
                           "after_comm_now": mkt.now.sum() - comm(mkt.now),
                           "after_comm_early": mkt.early.sum() - comm(mkt.early)}, name=hhmm))
    print(f"   {hhmm} UK: by day (result laid early against at the SP)")
    print(x.groupby("day")[["early", "now"]].sum().round(2).T.to_string())
print("\n   the lay times together (expected: under the BSP's chances; result: as the horses ran)")
print(pd.DataFrame(rows).round(3).to_string())
