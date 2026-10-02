"""The place rules at 15 minutes before each off, from the recorder's own win and place books: the forward record (read-only).

Scan 2 chose 14 rules on 2010-17 comparing the place SP with the place chance the win SPs imply, plus the place
pocket (win price 5.0 or shorter, expected value above 5% at 2%). place_rules_preplay.py reads them at the price
files' pre-play WAPs. This is the strict version, the one a live trader could run: from 1 Oct the recorder keeps
every GB/IE win and place book each minute in the last hour before the off. At T-15 (the owner's last moment to bet)
each race's win book gives the win chances (the mid of the best back and lay, normalised), scan 2's ordering
exponents give the place chances for the places the place market pays, and the place book gives the price to act at:
the best back for a back, the best lay for a lay. Each bet is settled two ways, at the place SP (an order at the SP
sent at T-15) and at the price shown at T-15, from Betfair's place price file for the day (each runner's place SP and
whether it placed). A day whose file is not yet published waits. A few days decide nothing: this starts the record.
"""

from __future__ import annotations

import gzip
import io
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402
from model.ordering import position_probabilities, simulate_finishing_orders  # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 300)
DAYS = [date(2026, 10, 1), date(2026, 10, 2)]
COMM = 0.02
T_MINUS = 15
BANDS = ((2, 7), (8, 11), (12, 15), (16, 40))
FIT = {(2, 7): (0.734, 0.557), (8, 11): (0.804, 0.640), (12, 15): (0.839, 0.697), (16, 40): (0.829, 0.735)}  # scan 2
CHOSEN = [("1-2", "back", 0.00), ("1-2", "back", 0.02), ("1-2", "back", 0.05), ("2-3", "back", 0.10),
          ("3-5", "lay", 0.02), ("5-8", "back", 0.10), ("5-8", "lay", 0.10), ("8-13", "back", 0.10),
          ("13-21", "back", 0.05), ("13-21", "back", 0.10), ("50+", "lay", 0.00), ("50+", "lay", 0.02),
          ("50+", "lay", 0.05), ("50+", "lay", 0.10)]
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
OUT = Path("out/place_t15_forward")
OUT.mkdir(parents=True, exist_ok=True)


def _get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        print(f"  {key}: {type(exc).__name__}")
        return None


def _csv(key):
    raw = _get(key)
    if raw is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(gzip.decompress(raw) if key.endswith(".gz") else raw),
                       dtype={"market_id": str, "selection_id": "Int64"}, low_memory=False)


def price_file(day, market):
    """Betfair's file for the day's racing in one market (named by the next day), keyed by the exchange's ids."""
    parts = []
    for d in (day + timedelta(days=1), day):
        for c in ("uk", "ire"):
            raw = _get(f"{bp.S3_PREFIX}/dwbfprices{c}{market}{d:%d%m%Y}.csv")
            if raw is None:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            t = bp.parse_file_text(text, f"dwbfprices{c}{market}{d:%d%m%Y}.csv")
            parts.append(t[t.race_date == day.isoformat()])
    if not parts:
        return pd.DataFrame()
    f = pd.concat(parts, ignore_index=True)
    f["market_id"] = "1." + f.event_id.astype("Int64").astype(str)   # the file's EVENT_ID is the market's number
    return f[["market_id", "selection_id", "selection_name", "win_lose", "bsp", "event_dt", "menu_hint"]]


def band_of(n):
    return next((b for b in BANDS if b[0] <= n <= b[1]), BANDS[-1])


def top3(pi, gm, dl):
    s = pi ** gm; s /= s.sum(); t = pi ** dl; t /= t.sum()
    A = pi[:, None] * s[None, :] / (1 - s[:, None]); np.fill_diagonal(A, 0.0)
    B = 1.0 / np.clip(1 - t[:, None] - t[None, :], 1e-12, None); np.fill_diagonal(B, 0.0)
    M = A * B
    return pi, A.sum(0), t * (M.sum() - M.sum(1) - M.sum(0))


def place_q(pi, k, rng):
    gm, dl = FIT[band_of(len(pi))]
    if k <= 3:
        p1, p2, p3 = top3(pi, gm, dl)
        return np.clip(p1 + (p2 if k >= 2 else 0) + (p3 if k >= 3 else 0), 1e-4, 1 - 1e-4)
    o = simulate_finishing_orders(pi, n_sims=4000, gamma=gm, delta=dl, rng=rng)
    return np.clip(position_probabilities(o, len(pi), n_positions=k).sum(axis=1), 1e-4, 1 - 1e-4)


def snapshot(books, mid, at):
    """The market's poll nearest T-15, within five minutes of it."""
    b = books[books.market_id == mid]
    if b.empty:
        return None
    t = b.t.unique()
    near = t[np.argmin(np.abs((pd.to_datetime(t) - at).total_seconds()))]
    if abs((pd.Timestamp(near) - at).total_seconds()) > 300:
        return None
    return b[b.t == near]


rng = np.random.default_rng(0)
bets, path = [], []
MARKS = (60, 30, 15, 5)
for day in DAYS:
    print(f"\n== {day}")
    pf = price_file(day, "place")
    wf = price_file(day, "win")
    if pf.empty or wf.empty:
        print("  Betfair's price files for the day's racing are not in the archive yet: the day waits")
        continue
    books = _csv(f"betfair_live/{day}/books.csv.gz")
    cat = pd.concat([_csv(f"betfair_live/{day}/{n}.csv.gz") for n in ("markets", "markets_trader")], ignore_index=True)
    if books.empty:
        print("  no recorder books for the day")
        continue
    books = books[books.source == "recorder"].copy()
    books["t"] = pd.to_datetime(books.polled_utc, utc=True)
    if not cat.empty:
        print("  the recorder's catalogue, market types:",
              cat.drop_duplicates("market_id").market_type.astype(str).value_counts(dropna=False).to_dict())
    # each horse runs once a day: its SELECTION_ID pairs its win market with its place market in Betfair's files
    pr = (wf[["market_id", "selection_id"]].merge(pf[["market_id", "selection_id"]], on="selection_id",
                                                  suffixes=("_w", "_p"))
          .groupby(["market_id_w", "market_id_p"]).size().reset_index(name="n"))
    pr = pr.sort_values("n").drop_duplicates("market_id_w", keep="last")
    starts = (cat.dropna(subset=["market_start_utc"]).drop_duplicates("market_id").set_index("market_id")
              .market_start_utc.to_dict() if not cat.empty else {})
    venues = (cat.drop_duplicates("market_id").set_index("market_id").venue.to_dict() if not cat.empty else {})
    pairs = pd.DataFrame({"market_id_w": pr.market_id_w, "market_id_p": pr.market_id_p,
                          "market_start_utc": pr.market_id_w.map(starts), "venue_w": pr.market_id_w.map(venues)})
    pairs = pairs.dropna(subset=["market_start_utc"])
    polled = set(books.market_id.unique())
    print(f"  {len(pr)} races paired through the files; {len(pairs)} with a start time; "
          f"{int(pairs.market_id_w.isin(polled).sum())} win and {int(pairs.market_id_p.isin(polled).sum())} place "
          f"markets among the {len(polled)} the recorder polled")
    n_read = 0
    for r in pairs.itertuples(index=False):
        off = pd.Timestamp(r.market_start_utc).tz_convert("UTC")
        at = off - pd.Timedelta(minutes=T_MINUS)
        wb, pbk = snapshot(books, r.market_id_w, at), snapshot(books, r.market_id_p, at)
        if wb is None or pbk is None:
            continue
        w = wb[wb.runner_status.eq("ACTIVE")].copy()
        w["mid"] = np.where(w.back1.notna() & w.lay1.notna(), (w.back1 + w.lay1) / 2, w.back1.fillna(w.lay1))
        w = w[w.mid > 1]
        settle = pf[pf.market_id == r.market_id_p].set_index("selection_id")
        if w.empty or settle.empty or len(w) < 3:
            continue
        k = int(settle.win_lose.sum().round())
        if not 1 <= k < len(w):
            continue
        pi = (1 / w.mid.to_numpy()); pi = pi / pi.sum()
        q = place_q(pi, k, rng)
        pb = pbk.set_index("selection_id")
        n_read += 1
        for mark in MARKS:                                   # the place market's price path to the off
            snap = snapshot(books, r.market_id_p, off - pd.Timedelta(minutes=mark))
            if snap is None:
                continue
            for sr in snap.itertuples(index=False):
                sid = int(sr.selection_id)
                if sid in settle.index:
                    path.append({"day": str(day), "race": r.market_id_w, "selection_id": sid, "mark": mark,
                                 "back": sr.back1, "lay": sr.lay1, "place_bsp": settle.loc[sid].bsp,
                                 "placed": int(settle.loc[sid].win_lose == 1)})
        for sid, wmid, qi in zip(w.selection_id.astype(int), w.mid, q):
            if sid not in pb.index or sid not in settle.index:
                continue
            row = pb.loc[sid]
            back, lay = row.back1, row.lay1
            s = settle.loc[sid]
            bets.append({"day": str(day), "venue": r.venue_w, "off": off.strftime("%H:%M"), "selection_id": sid,
                         "horse": s.selection_name, "win_mid": wmid, "q": qi, "places": k, "runners": len(w),
                         "place_back": back, "place_lay": lay, "place_bsp": s.bsp, "placed": int(s.win_lose == 1),
                         "place_back_size": row.back1_size, "place_lay_size": row.lay1_size})
    print(f"  {n_read} races read at T-{T_MINUS}")
if not bets:
    raise SystemExit("no day could be read")
B = pd.DataFrame(bets)
B["band"] = pd.cut(B.win_mid, [1, 2, 3, 5, 8, 13, 21, 50, 1e9], right=False,
                   labels=["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-50", "50+"]).astype(str)
B["ev_back"] = B.q * (B.place_back - 1) * (1 - COMM) - (1 - B.q)
B["ev_lay"] = (1 - B.q) * (1 - COMM) - B.q * (B.place_lay - 1)


def net(v):
    return np.where(v > 0, v * (1 - COMM), v)


placed = B.placed.to_numpy() == 1
B["back_at_sp"] = net(np.where(placed, B.place_bsp - 1, -1.0))
B["back_at_t15"] = net(np.where(placed, B.place_back - 1, -1.0))
B["lay_at_sp"] = net(np.where(placed, -(B.place_bsp - 1), 1.0))
B["lay_at_t15"] = net(np.where(placed, -(B.place_lay - 1), 1.0))
B.to_csv(OUT / "runners_t15.csv", index=False)

rows = []
rules = [(f"{side} {band} EV>{over:.2f}", side, (B.band == band) & (B[f"ev_{side}"] > over)) for band, side, over in CHOSEN]
rules.append(("place pocket (win <= 5.0, EV>0.05)", "back", (B.win_mid <= 5.0) & (B.ev_back > 0.05)))
for name, side, m in rules:
    for day, g in [("all", B[m])] + [(d, B[m & (B.day == d)]) for d in sorted(B.day.unique())]:
        liab_sp = (g.place_bsp - 1).sum() if side == "lay" else np.nan
        liab_t = (g.place_lay - 1).sum() if side == "lay" else np.nan
        rows.append({"rule": name, "day": day, "bets": len(g), "placed": int(g.placed.sum()),
                     "at_sp_per_1": g[f"{side}_at_sp"].mean() if len(g) else np.nan,
                     "at_t15_per_1": g[f"{side}_at_t15"].mean() if len(g) else np.nan,
                     "at_sp_per_liability": g[f"{side}_at_sp"].sum() / liab_sp if side == "lay" and liab_sp > 0 else np.nan,
                     "size_at_price_med": g[f"place_{side}_size"].median() if len(g) else np.nan})
print("\n== the rules at T-15 (per GBP1 staked for backs, per GBP1 laid for lays)")
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\n== the T-15 place prices against the place SP, and the win mid against the place market (all runners read)")
print(B.groupby("band").apply(lambda g: pd.Series({
    "n": len(g), "back_t15/sp": (g.place_back / g.place_bsp).median(), "lay_t15/sp": (g.place_lay / g.place_bsp).median(),
    "q": g.q.mean(), "placed": g.placed.mean()})).round(3).to_string())

print("\n== the place market's best back and lay before the off against the place SP (median ratio), by win band")
P = pd.DataFrame(path).merge(B[["day", "selection_id", "band"]], on=["day", "selection_id"], how="inner")
if not P.empty:
    P.to_csv(OUT / "place_path.csv", index=False)
    P["lay/sp"], P["back/sp"] = P.lay / P.place_bsp, P.back / P.place_bsp
    print(P.groupby(["band", "mark"]).agg(n=("lay", "size"), lay_sp=("lay/sp", "median"), back_sp=("back/sp", "median"),
                                          placed=("placed", "mean")).round(3).to_string())
    # a lay matched at the price shown stands to the result: +1 unplaced, -(price - 1) placed, per GBP1 laid
    P["lay_fixed"] = net(np.where(P.placed == 1, -(P.lay - 1), 1.0))
    print("\nlay to place at the price shown, win 50+ (per GBP1 laid):")
    print(P[P.band == "50+"].groupby("mark").agg(n=("lay_fixed", "size"), roi=("lay_fixed", "mean"),
                                                 placed=("placed", "sum")).round(4).to_string())
