"""Every Betfair market on a race against the chances our model, the win market and the BSP imply (read-only; the
owner's ask of 6 Oct 2026: "check every betfair market: Each Way, Top 2, Top 3, Top 4 ... always look at all
markets for edges ... work on probability based on our betfair SP").

The record (s3 betfair_live/<day>/): the 2/3/4 TBP and each-way books (books_other_place, recorded beside the trader
from 6 Oct, every 5 minutes and each minute in the last hour), the win and place books (books.csv.gz, the day
record) and their catalogues; our model's prices (predictions/<day>.csv, the 06:00 run's, matched to Betfair's ids
as the trader matches them). A horse has the same selection id in every market of its race.

A runner's chance of finishing in the first k (1 for win, 2/3/4 for the TBP markets, the race's place terms for the
place market and the each-way bet's place part) comes from a race's win probabilities through model.ordering
(Benter's discounted Harville, the exponents fitted on 2021-22 by field size: ledger place-market-vs-win), from
three sources: our model's predicted BSP (known before racing: what a rule would trade on), the win market's mid at
the same minute (the market's own view), and the win BSP (fixed at the off: the closing benchmark, the CLV of
markets with no SP of their own). The places a market pays (numberOfWinners on its book) and an each-way market's
divisor (eachWayDivisor in its catalogue) are recorded from 7 Oct and used where present; before, the standard terms
are taken: place 2 at 5-7 runners, 3 at 8-15, at 16+ 4 in a handicap else 3; each way 1/4 at 5-7 runners, 1/5 at
8+, 1/4 in handicaps of 12 or more (4 places at 16+).

At each mark (10:00 and 12:00 UK; 60, 15 and 5 minutes before each off) every runner's best back and best lay in
every market: the expected value of a unit back at the best back and of a unit lay at the best lay under each
source (2% commission on winnings; an each-way unit is half win, half place), the size on offer, how many clear 0
and 5%, and the best of them. Where the day's final pass has run, the final books settle the bets the model flags.
SCAN_DAY picks the day (default: today in the UK).
"""

from __future__ import annotations

import gzip
import io
import os
import re
import sys
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from scipy.optimize import linprog  # noqa: E402

from model import race_book as rb  # noqa: E402
from model.ordering import place_probabilities  # noqa: E402
from trading.exchange import Market  # noqa: E402
from trading.matching import attach_ids  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
UK = "Europe/London"
DAY = date.fromisoformat(os.environ.get("SCAN_DAY") or datetime.now(ZoneInfo(UK)).date().isoformat())
C = 0.02                                        # the owner's commission
BANDS = (((2, 7), (0.734, 0.557)), ((8, 11), (0.804, 0.640)), ((12, 15), (0.839, 0.697)), ((16, 99), (0.829, 0.735)))
CLOCK_MARKS = (("10:00", 600), ("12:00", 720))
OFF_MARKS = (60, 15, 5)                         # minutes before the off
KEY = ["market_id", "selection_id"]
CLOSING = rb.load_closing_model("data/models/closing_model_novol.json")   # the trader's, for a feed without volume
ARB_WINDOW = 120                                # minutes before the off in which every poll is searched for arbitrage
SYNC = 90                                       # seconds: books of other markets read with a poll
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")


def _csv(key, **kw):
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
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
    b = _num(b, ["selection_id", "back1", "back1_size", "lay1", "lay1_size", "last_traded", "sp_actual"]).dropna(
        subset=["selection_id"]).copy()
    b["selection_id"] = b.selection_id.astype("int64")
    b["t"] = pd.to_datetime(b.polled_utc, utc=True)
    return b


def catalogue(*tags):
    parts = [_csv(f"betfair_live/{DAY}/markets{'_' + t if t else ''}.csv.gz", dtype={"market_id": str,
                                                                                        "event_id": str})
             for t in tags]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    c = _num(pd.concat(parts, ignore_index=True), ["selection_id"]).dropna(subset=["selection_id"])
    c["selection_id"] = c.selection_id.astype("int64")
    c["start"] = pd.to_datetime(c.market_start_utc, utc=True)
    return c.drop_duplicates(KEY, keep="last")


def model_prices(win_cat):
    """Our model's predicted BSP keyed by (win market, selection), matched as the trader matches it."""
    p = _csv(f"predictions/{DAY}.csv", dtype={"race_time": str, "market_id": str})
    if p.empty or "predicted_bfsp" not in p or win_cat.empty:
        return {}
    markets = [Market(market_id=str(mid), venue=str(g.venue.iloc[0]), country=str(g.country.iloc[0]),
                      start=g.start.iloc[0].to_pydatetime(),
                      runners={int(a): str(b) for a, b in zip(g.selection_id, g.runner_name)})
               for mid, g in win_cat.groupby("market_id")]
    p = attach_ids(p, markets)
    p = _num(p, ["selection_id", "predicted_bfsp"]).dropna(subset=["market_id", "selection_id", "predicted_bfsp"])
    return {(str(r.market_id), int(r.selection_id)): float(r.predicted_bfsp) for r in p.itertuples()
            if r.predicted_bfsp > 1}


def exponents(n):
    for (lo, hi), ge in BANDS:
        if lo <= n <= hi:
            return ge
    return BANDS[-1][1]


def place_chance(p, k):
    """P(first k) for each runner of p (a race's win probabilities, summing to one)."""
    n = len(p)
    if k >= n:
        return np.ones(n)
    g, d = exponents(n)
    return place_probabilities(p, k, gamma=g, delta=d, n_sims=20000, rng=np.random.default_rng(7))


def place_terms(n, handicap):
    if n < 5:
        return 0
    if n <= 7:
        return 2
    if n <= 15:
        return 3
    return 4 if handicap else 3


def each_way_terms(n, handicap):
    """(places, divisor) of the standard each-way terms; (0, None) where none apply."""
    if n < 5:
        return 0, None
    if n <= 7:
        return 2, 4.0
    if handicap and n >= 16:
        return 4, 4.0
    if handicap and n >= 12:
        return 3, 4.0
    return 3, 5.0


def ev_back(p, b):
    return p * (1.0 + (b - 1.0) * (1.0 - C)) - 1.0


def ev_lay(p, lay):
    return (1.0 - p) * (1.0 - C) - p * (lay - 1.0)


def ev_back_ew(pw, pp, price, d):
    """A unit each-way back (half win, half place at 1/d of the odds), per unit staked."""
    return 0.5 * (pw * (1.0 + (price - 1.0) * (1.0 - C)) + pp * (1.0 + (price - 1.0) * (1.0 - C) / d)) - 1.0


def ev_lay_ew(pw, pp, lay, d):
    """A unit each-way lay (the backer's half win, half place), per unit of the backer's stake."""
    placed_only = 0.5 - 0.5 * (lay - 1.0) / d
    placed_only = placed_only * (1.0 - C) if placed_only > 0 else placed_only
    return ((1.0 - pp) * (1.0 - C) + (pp - pw) * placed_only - pw * 0.5 * (lay - 1.0) * (1.0 + 1.0 / d))


def mid(row):
    b, l, lt = row.back1, row.lay1, row.last_traded
    if b > 1 and l > 1:
        return float(np.sqrt(b * l))
    for v in (b, lt, l):
        if v > 1:
            return float(v)
    return np.nan


def nearest(b, mids, t, tol_min):
    """Each market's book at its poll nearest t (within tol_min minutes)."""
    x = b[b.market_id.isin(mids)]
    if x.empty:
        return x
    polls = x.groupby(["market_id", "polled_utc"], as_index=False).agg(t=("t", "first"))
    polls["gap"] = (polls.t - t).abs()
    polls = polls[polls.gap <= pd.Timedelta(minutes=tol_min)].sort_values("gap").drop_duplicates("market_id")
    return x.merge(polls[["market_id", "polled_utc"]], on=["market_id", "polled_utc"])


# ----------------------------------------------------------------------------------------------------------- data
win_cat = catalogue("")
trader_cat = catalogue("trader")
if not trader_cat.empty:                         # the trader's own catalogue call may leave the type blank: it reads win
    trader_cat["market_type"] = trader_cat.market_type.fillna("").replace("", "WIN")
    win_cat = pd.concat([win_cat, trader_cat], ignore_index=True).drop_duplicates(KEY, keep="first")
oth_cat = catalogue("other_place")
oth = books(f"betfair_live/{DAY}/books_other_place.csv.gz")
day = books(f"betfair_live/{DAY}/books.csv.gz")
trader = books(f"betfair_live/{DAY}/books_trader.csv.gz")
prices = model_prices(win_cat[win_cat.market_type.eq("WIN")] if not win_cat.empty else win_cat)
print(f"== {DAY}: catalogues: win/place {win_cat.market_id.nunique() if not win_cat.empty else 0} markets, "
      f"other place {oth_cat.market_id.nunique() if not oth_cat.empty else 0}; books: other place {len(oth):,} rows "
      f"({oth.polled_utc.nunique() if not oth.empty else 0} polls, to "
      f"{oth.t.max().tz_convert(UK):%H:%M} UK)" if not oth.empty else f"== {DAY}: no other-place books yet")
if oth.empty or oth_cat.empty or win_cat.empty:
    print("nothing to scan: the other-place record or a catalogue is missing")
    raise SystemExit(0)
print(f"   day record {len(day):,} rows, the trader's books {len(trader):,} rows; model prices {len(prices)}")
print("   market types:", oth_cat.drop_duplicates("market_id").groupby(["market_type", "market_name"]).size()
      .to_dict())

# the races: every market of an event at one off time
allcat = pd.concat([win_cat, oth_cat], ignore_index=True)
allcat["race"] = allcat.event_id.astype(str) + "|" + allcat.start.astype(str)
mk = allcat.drop_duplicates("market_id").set_index("market_id")
win_book = pd.concat([day, trader], ignore_index=True) if not trader.empty else day


def _last_known(frame, col):
    if frame.empty or col not in frame.columns:
        return {}
    v = frame.assign(_v=pd.to_numeric(frame[col], errors="coerce")).dropna(subset=["_v"])
    return v.groupby("market_id")["_v"].last().to_dict()


RECORDED_PLACES = {**_last_known(day, "number_of_winners"), **_last_known(oth, "number_of_winners")}
RECORDED_DIVISOR = _last_known(oth_cat, "each_way_divisor")
TERMS_USED = {"recorded": 0, "standard": 0}


def market_terms(omid, kname, k, n, handicap):
    """(places, divisor) of a place, each-way or TBP market: its own where the record keeps them, else the standard
    terms (a TBP market's places from its name)."""
    rk, rd = RECORDED_PLACES.get(omid), RECORDED_DIVISOR.get(omid)
    d = None
    if kname == "PLACE":
        k = int(rk) if rk else place_terms(n, handicap)
    elif kname == "EACH_WAY":
        sk, sd = each_way_terms(n, handicap)
        k, d = (int(rk) if rk else sk), (float(rd) if rd else sd)
        rk = rk if rd else None                     # an each-way bet's terms count as recorded only with both
    elif rk:
        k = int(rk)
    TERMS_USED["recorded" if rk else "standard"] += 1
    return k, d


def kind(mid_):
    m = mk.loc[mid_]
    if m.market_type == "WIN":
        return "WIN", 1
    if m.market_type == "PLACE":
        return "PLACE", None
    if m.market_type == "EACH_WAY":
        return "EACH_WAY", None
    hit = re.search(r"(\d)\s*TBP", str(m.market_name), re.I)
    return (f"{hit.group(1)} TBP", int(hit.group(1))) if hit else (str(m.market_name), None)


rows, linked, unlinked = [], 0, 0
for race, g in mk.groupby("race"):
    wins = g.index[g.market_type.eq("WIN")]
    if len(wins) != 1:
        continue
    wmid = wins[0]
    start = g.start.iloc[0]
    handicap = bool(re.search(r"hcap|handicap", str(g.loc[wmid, "market_name"]), re.I))
    others = [m for m in g.index if m != wmid]
    marks = [(name, pd.Timestamp(datetime.combine(DAY, datetime.min.time()), tz=UK) + pd.Timedelta(minutes=m), 8)
             for name, m in CLOCK_MARKS]
    marks += [(f"T-{m}", start - pd.Timedelta(minutes=m), 3 if m > 15 else 2) for m in OFF_MARKS]
    for name, t, tol in marks:
        if t >= start:
            continue
        w = nearest(win_book[win_book.market_id.eq(wmid)], [wmid], t, tol)
        w = w[w.runner_status.eq("ACTIVE")].drop_duplicates("selection_id")
        if len(w) < 2:
            continue
        sel = w.selection_id.to_numpy()
        n = len(sel)
        mp = np.array([prices.get((wmid, int(s)), np.nan) for s in sel])
        mids = np.array([mid(r) for r in w.itertuples()])
        sources = {}
        if np.isfinite(mp).all():
            sources["model"] = (1 / mp) / (1 / mp).sum()
        if np.isfinite(mids).all():
            sources["market"] = (1 / mids) / (1 / mids).sum()
        if np.isfinite(mp).all() and np.isfinite(mids).all():   # the closing model: our price and the market's, a la Benter
            bk, ly = w.back1.to_numpy(float), w.lay1.to_numpy(float)
            now = np.where(np.isfinite(bk) & (bk > 1), rb.market_now(bk, ly), mids)
            _, exp_bsp = rb.race_expected_clv(bk, ly, mp, np.zeros(n), CLOSING, n_draws=4000,
                                              rng=np.random.default_rng(0), market=now)
            if np.isfinite(exp_bsp).all():
                sources["combined"] = (1 / exp_bsp) / (1 / exp_bsp).sum()
        if not sources:
            continue
        books_now = nearest(oth[oth.market_id.isin(others)], others, t, tol)
        place_now = nearest(day[day.market_id.isin(others)], others, t, tol) if not day.empty else day
        books_now = pd.concat([books_now, place_now], ignore_index=True).drop_duplicates(KEY)
        pk_cache = {}
        for omid, ob in books_now.groupby("market_id"):
            kname, k = kind(omid)
            k, d_ew = market_terms(omid, kname, k, n, handicap)
            ew = (k, d_ew) if kname == "EACH_WAY" else None
            if not k:
                continue
            idx = {int(s): i for i, s in enumerate(sel)}
            for r in ob[ob.runner_status.eq("ACTIVE")].itertuples():
                i = idx.get(int(r.selection_id))
                if i is None:
                    unlinked += 1
                    continue
                linked += 1
                rec = dict(race=race, start=start, mark=name, market=kname, market_id=omid, selection_id=int(r.selection_id),
                           runners=n, handicap=handicap, back=r.back1, back_size=r.back1_size, lay=r.lay1,
                           lay_size=r.lay1_size, places=k, divisor=ew[1] if ew else np.nan)
                for src, p in sources.items():
                    if (src, k) not in pk_cache:
                        pk_cache[(src, k)] = place_chance(p, k)
                    pp, pw = pk_cache[(src, k)][i], p[i]
                    rec[f"p_{src}"] = pp
                    if kname == "EACH_WAY":
                        rec[f"evb_{src}"] = ev_back_ew(pw, pp, r.back1, ew[1]) if r.back1 > 1 else np.nan
                        rec[f"evl_{src}"] = ev_lay_ew(pw, pp, r.lay1, ew[1]) if r.lay1 > 1 else np.nan
                    else:
                        rec[f"evb_{src}"] = ev_back(pp, r.back1) if r.back1 > 1 else np.nan
                        rec[f"evl_{src}"] = ev_lay(pp, r.lay1) if r.lay1 > 1 else np.nan
                rows.append(rec)

S = pd.DataFrame(rows)
print(f"   offers read: {len(S):,} (runner x market x mark); runners linked to the win market {linked:,}, "
      f"not linked {unlinked:,}; market terms {TERMS_USED['recorded']:,} recorded, {TERMS_USED['standard']:,} "
      "standard (recorded from 7 Oct)")
if S.empty:
    raise SystemExit(0)
S["spread"] = S.lay / S.back

print("\n== the books: runners priced, the spread (best lay / best back) and the size at each, by market and mark")
print(S.groupby(["market", "mark"]).agg(races=("race", "nunique"), offers=("selection_id", "size"),
                                        spread_median=("spread", "median"), back_size_median=("back_size", "median"),
                                        lay_size_median=("lay_size", "median")).round(2).to_string())

for src in ("model", "combined", "market"):
    if f"evb_{src}" not in S:
        continue
    print(f"\n== expected value under the {src}'s chances (a unit at the best back / best lay, 2% commission)")
    out = []
    for (m, mark), g in S.groupby(["market", "mark"]):
        b, l = g[f"evb_{src}"].dropna(), g[f"evl_{src}"].dropna()
        out.append(dict(market=m, mark=mark, backs=len(b), back_ev_median=b.median(), backs_over_0=(b > 0).sum(),
                        backs_over_5pc=(b > 0.05).sum(), back_ev_mean_over_5pc=b[b > 0.05].mean(),
                        lays=len(l), lay_ev_median=l.median(), lays_over_0=(l > 0).sum(),
                        lays_over_5pc=(l > 0.05).sum(), lay_ev_mean_over_5pc=l[l > 0.05].mean()))
    print(pd.DataFrame(out).round(3).to_string(index=False))

names = allcat.drop_duplicates(KEY).set_index(KEY).runner_name
venue = mk.venue
for side, col, size in (("backs", "evb_combined", "back_size"), ("lays", "evl_combined", "lay_size")):
    if col not in S:
        continue
    top = S[S[col] > 0.05].sort_values(col, ascending=False).head(15).copy()
    if top.empty:
        print(f"\n== no {side} above 5% under the combined chances")
        continue
    top["horse"] = [names.get((m, s), s) for m, s in zip(top.market_id, top.selection_id)]
    top["venue"] = [venue.get(m) for m in top.market_id]
    top["off"] = top.start.dt.tz_convert(UK).dt.strftime("%H:%M")
    cols = ["venue", "off", "mark", "market", "horse", "runners", "back", "back_size", "lay", "lay_size",
            "p_model", "p_combined", "p_market", col.replace("combined", "model"), col, col.replace("combined", "market")]
    print(f"\n== the best {side} under the combined chances (expected value over 5%)")
    print(top[[c for c in cols if c in top]].round(3).to_string(index=False))

# ------------------------------------------------------------------------------------------------ arbitrage
def unit_payoffs(kind, k, d, price, side, n_states):
    """P/L of a unit stake (the backer's stake for a lay) in each state: finishing 1st, 2nd, ... the last state the
    rest. Commission on a market's net winnings: one position a market, so a winning state's P/L carries it."""
    out = np.zeros(n_states)
    for s_ in range(1, n_states + 1):
        if kind == "EACH_WAY":
            if s_ == 1:
                v = 0.5 * (price - 1) * (1 + 1 / d)
                v = v * (1 - C) if side == "back" else -v
            elif s_ <= k:
                v = -0.5 + 0.5 * (price - 1) / d
                v = v if side == "back" else -v
                v = v * (1 - C) if v > 0 else v
            else:
                v = -1.0 if side == "back" else 1 - C
        elif side == "back":
            v = (price - 1) * (1 - C) if s_ <= k else -1.0
        else:
            v = -(price - 1) if s_ <= k else 1 - C
        out[s_ - 1] = v
    return out


def horse_arbitrage(legs, n_states):
    """The most a horse's offers across its markets guarantee whatever its finishing place, at the best prices
    and within the sizes on offer: max z with every state's P/L at least z (a linear programme). legs: (market,
    kind, k, d, back, back size, lay, lay size). Returns (z, money at risk, legs used) or None."""
    cols, ub, desc = [], [], []
    for name, kind, k, d, b, bs, l, ls in legs:
        if b > 1 and bs >= 1:
            cols.append(unit_payoffs(kind, k, d, b, "back", n_states)); ub.append(bs); desc.append((name, "back", b))
        if 1 < l < 1000 and ls >= 1:
            cols.append(unit_payoffs(kind, k, d, l, "lay", n_states)); ub.append(ls); desc.append((name, "lay", l))
    if len(cols) < 2:
        return None
    A = np.array(cols).T
    nv = A.shape[1]
    res = linprog(np.r_[np.zeros(nv), -1.0], A_ub=np.c_[-A, np.ones(n_states)], b_ub=np.zeros(n_states),
                  bounds=[(0, u) for u in ub] + [(None, None)], method="highs")
    if not res.success or -res.fun <= 0.01:
        return None
    x = res.x[:nv]
    risk = sum(xi if side == "back" else xi * (pr - 1) for xi, (_, side, pr) in zip(x, desc))
    used = [(nm, side, pr, round(float(xi), 2)) for xi, (nm, side, pr) in zip(x, desc) if xi > 0.005]
    return float(-res.fun), float(risk), used


def book_arbitrage(ob, k):
    """Back every runner (a stake of 1/price each pays exactly k, whoever fills the k places) or lay every runner:
    the profit locked at the best prices, after commission, as far as the smallest size allows."""
    out = []
    b, bs = ob.back1.to_numpy(float), ob.back1_size.to_numpy(float)
    lay, ls = ob.lay1.to_numpy(float), ob.lay1_size.to_numpy(float)
    if len(b) and (b > 1).all():
        book = (1 / b).sum()
        if book < k:
            lam = float(np.min(b * bs))
            out.append(("back every runner", book, lam * (k - book) * (1 - C)))
    if len(lay) and ((lay > 1) & (lay < 1000)).all():
        book = (1 / lay).sum()
        if book > k:
            lam = float(np.min(lay * ls))
            out.append(("lay every runner", book, lam * (book - k) * (1 - C)))
    return out


arbs, books_found, polls_searched = [], [], 0
for race, g in mk.groupby("race"):
    wins = g.index[g.market_type.eq("WIN")]
    if len(wins) != 1:
        continue
    wmid, start = wins[0], g.start.iloc[0]
    handicap = bool(re.search(r"hcap|handicap", str(g.loc[wmid, "market_name"]), re.I))
    others = [m for m in g.index if m != wmid]
    ob_all = oth[oth.market_id.isin(others)]
    lo = start - pd.Timedelta(minutes=ARB_WINDOW)
    for polled, pb in ob_all[(ob_all.t >= lo) & (ob_all.t < start)].groupby("polled_utc"):
        t = pb.t.iloc[0]
        w = nearest(win_book[win_book.market_id.eq(wmid)], [wmid], t, SYNC / 60)
        w = w[w.runner_status.eq("ACTIVE")].drop_duplicates("selection_id")
        if len(w) < 2:
            continue
        polls_searched += 1
        n = len(w)
        plc = nearest(day[day.market_id.isin(others)], others, t, SYNC / 60) if not day.empty else day
        nowb = pd.concat([pb, plc], ignore_index=True).drop_duplicates(KEY)
        nowb = nowb[nowb.runner_status.eq("ACTIVE")]
        mins_to_off = (start - t).total_seconds() / 60
        terms = {}
        for omid in nowb.market_id.unique():
            kname, k = kind(omid)
            k, d = market_terms(omid, kname, k, n, handicap)
            if k:
                terms[omid] = (kname, k, d)
        # the states: each place up to the most any market pays, then the rest. With a state for 1st-4th and one for
        # 5th or worse, a 5 TBP back (from 7 Oct) won in every state: no risk where it has one
        n_states = min(n, 1 + max([4] + [k for _, k, _ in terms.values()]))
        for omid, (kname, k, d) in terms.items():
            if kname != "EACH_WAY":
                for how, book, prof in book_arbitrage(nowb[nowb.market_id.eq(omid)], k):
                    books_found.append(dict(race=race, start=start, mins_to_off=mins_to_off, market=kname, how=how,
                                            book=book, places=k, profit=prof))
        for r in w.itertuples():
            legs = [("WIN", "WIN", 1, None, r.back1, r.back1_size, r.lay1, r.lay1_size)]
            for q in nowb[nowb.selection_id.eq(r.selection_id)].itertuples():
                if q.market_id in terms:
                    kname, k, d = terms[q.market_id]
                    legs.append((kname, kname, k, d, q.back1, q.back1_size, q.lay1, q.lay1_size))
            if len(legs) < 2:
                continue
            got = horse_arbitrage(legs, n_states)
            if got:
                z, risk, used = got
                arbs.append(dict(race=race, start=start, mins_to_off=round(mins_to_off, 1), selection_id=r.selection_id,
                                 runners=n, profit=z, at_risk=risk, legs=used))

print(f"\n== arbitrage: {polls_searched} race-polls in the last {ARB_WINDOW} minutes before the offs searched "
      f"(every horse across win, place, the TBP markets and each way; books of each market)")
if arbs:
    A_ = pd.DataFrame(arbs)
    A_["return_on_risk"] = A_.profit / A_.at_risk
    A_["pair"] = [" + ".join(sorted(f"{side} {nm}" for nm, side, _, _ in legs)) for legs in A_.legs]
    A_["horse"] = [names.get((mk[(mk.race == rc) & mk.market_type.eq("WIN")].index[0], s_), s_)
                   for rc, s_ in zip(A_.race, A_.selection_id)]
    print(f"   {len(A_)} horse-polls ({A_[['race', 'selection_id']].drop_duplicates().shape[0]} horses, "
          f"{A_.race.nunique()} races) with a guaranteed profit after commission; GBP median "
          f"{A_.profit.median():.2f}, largest {A_.profit.max():.2f}; return on the money at risk median "
          f"{A_.return_on_risk.median():.1%}")
    print(A_.groupby("pair").agg(polls=("profit", "size"), horses=("selection_id", "nunique"),
                                 gbp_median=("profit", "median"), gbp_max=("profit", "max"),
                                 ror_median=("return_on_risk", "median")).sort_values("polls", ascending=False)
          .round(3).head(20).to_string())
    best = A_.sort_values("profit", ascending=False).drop_duplicates(["race", "selection_id"]).head(15)
    best["off"] = best.start.dt.tz_convert(UK).dt.strftime("%H:%M")
    best["venue"] = [mk.loc[mk[(mk.race == rc)].index[0], "venue"] for rc in best.race]
    print(best[["venue", "off", "mins_to_off", "horse", "runners", "profit", "at_risk", "return_on_risk", "legs"]]
          .round(3).to_string(index=False))
else:
    print("   none: no horse's offers guarantee a profit after commission at the best prices")
if books_found:
    B_ = pd.DataFrame(books_found)
    print(f"\n   books: {len(B_)} market-polls where backing or laying every runner locks a profit")
    print(B_.groupby(["market", "how"]).agg(polls=("profit", "size"), gbp_median=("profit", "median"),
                                            gbp_max=("profit", "max"), book_median=("book", "median")).round(3)
          .to_string())
else:
    print("   books: none locks a profit (every back book over its places, every lay book under)")

# ------------------------------------------------------------------------------------------- the close and results
fin_w = day[day.source.eq("final")] if "source" in day else pd.DataFrame()
fin_o = oth[oth.source.eq("final")] if "source" in oth else pd.DataFrame()
if fin_w.empty or fin_o.empty:
    print("\nthe day's final pass has not run: the close (win BSP) and the results come with it")
    raise SystemExit(0)
# the win BSP: from Betfair's price files (named by the day after the racing, in S3 from that evening), else one read
# at the off (source "bsp"), else a final book's. The record's own reads carry none on the delayed application key,
# which returns no Starting Price data (7 Oct: ledger bsp-at-off-1007), so the close comes with the files.


def price_file_bsp(day_):
    import betfair_prices as bp
    parts = []
    for d_ in (date.fromordinal(day_.toordinal() + 1), day_):
        for c_ in ("uk", "ire"):
            name = f"dwbfprices{c_}win{d_:%d%m%Y}.csv"
            try:
                raw = s3.get_object(Bucket=bucket, Key=f"{bp.S3_PREFIX}/{name}")["Body"].read()
            except Exception:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            t_ = bp.parse_file_text(text, name)
            parts.append(t_[t_.race_date == day_.isoformat()])
    if not parts:
        return pd.DataFrame(columns=KEY + ["sp_actual"])
    pf = pd.concat(parts, ignore_index=True).dropna(subset=["bsp", "selection_id"])
    return pd.DataFrame({"market_id": "1." + pf.event_id.astype("Int64").astype(str),
                         "selection_id": pf.selection_id.astype("int64"), "sp_actual": pf.bsp.astype(float)})


at_off = day[day.source.eq("bsp") & (day.sp_actual > 1)]
files = price_file_bsp(DAY)
bsp = (pd.concat([fin_w.loc[fin_w.sp_actual > 1, KEY + ["sp_actual"]], at_off[KEY + ["sp_actual"]], files])
       .drop_duplicates(KEY, keep="last").set_index(KEY).sp_actual)
print(f"   the close: {len(files):,} BSPs in the price files, {len(at_off):,} read at the off, "
      f"{int((fin_w.sp_actual > 1).sum()):,} in final books")
res = pd.concat([fin_o, fin_w]).drop_duplicates(KEY, keep="last").set_index(KEY).runner_status
S["result"] = [res.get((m, s)) for m, s in zip(S.market_id, S.selection_id)]
done = S[S.result.isin(["WINNER", "LOSER"])].copy()
done["won"] = done.result.eq("WINNER")
print(f"\n== settled: {done.race.nunique()} races, {len(done):,} offers")
pb = []
for race, g in done.groupby("race"):
    wmid = mk[(mk.race == race) & mk.market_type.eq("WIN")].index
    if len(wmid) != 1:
        continue
    sel = sorted(set(g.selection_id))
    sp = np.array([bsp.get((wmid[0], s), np.nan) for s in sel])
    if not np.isfinite(sp).all() or (sp <= 1).any():
        continue
    p = (1 / sp) / (1 / sp).sum()
    for k in sorted(set(g.places)):
        pk = place_chance(p, int(k))
        for s, v in zip(sel, pk):
            pb.append(dict(race=race, selection_id=s, places=k, p_bsp=v, pw_bsp=p[sel.index(s)]))
if pb:
    done = done.merge(pd.DataFrame(pb), on=["race", "selection_id", "places"], how="left")
    ew = done.market.eq("EACH_WAY")
    done["evb_bsp"] = np.where(ew, ev_back_ew(done.pw_bsp, done.p_bsp, done.back, done.divisor),
                               ev_back(done.p_bsp, done.back))
    done["evl_bsp"] = np.where(ew, [ev_lay_ew(a, b, c, d) for a, b, c, d in
                                    zip(done.pw_bsp, done.p_bsp, done.lay, done.divisor)], ev_lay(done.p_bsp, done.lay))
flag_b = done[done.get("evb_model", pd.Series(dtype=float)) > 0.05]
flag_l = done[done.get("evl_model", pd.Series(dtype=float)) > 0.05]
summary = []
for side, g in (("back", flag_b), ("lay", flag_l)):
    for (m, mark), h in g.groupby(["market", "mark"]):
        if side == "back":
            pnl = np.where(h.won, (h.back - 1) * (1 - C), -1.0)
        else:
            pnl = np.where(h.won, -(h.lay - 1), 1 - C)
        if (h.market == "EACH_WAY").any():                 # each-way settles on the win too: left to the BSP view
            pnl = np.full(len(h), np.nan)
        summary.append(dict(side=side, market=m, mark=mark, bets=len(h), ev_model=h[f"ev{side[0]}_model"].mean(),
                            ev_at_bsp=h.get(f"ev{side[0]}_bsp", pd.Series(np.nan, index=h.index)).mean(),
                            result_per_unit=np.nanmean(pnl) if np.isfinite(pnl).any() else np.nan))
if summary:
    print("\n== the model's flagged offers (over 5%): expected under the model, expected at the BSP (the close), "
          "and the result")
    print(pd.DataFrame(summary).round(3).to_string(index=False))
