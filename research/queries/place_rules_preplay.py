"""The place rules of scan 2, read at prices before the off: win and place pre-play WAPs, 2018 to March 2026 (read-only).

Scan 2 (research/queries/done/edge_bsp_bias.py, run 37068154175) chose 14 rules on 2010-17 that compare the place SP
with the place chance the win SPs imply. All 14 were positive on 2018-26Q1, and 12 had their 90% interval above zero.
Every SP is known only at the off, though, so none of those rules can be traded as read. Here the same 14 rules (and
the place pocket) decide on Betfair's pre-play WAP instead:

- the win-implied place chance from the win market's PPWAP (normalised in the race), with scan 2's ordering exponents
  (fitted on 2021-22 finishing orders), and the places the place market paid;
- the expected value at the place market's PPWAP, the win band from the win PPWAP;
- each bet settled two ways: taken at the place SP (an order at the SP placed when the rule fires), and filled at the
  place PPWAP (a price the pre-play market traded at, on average).

The place files' PPWAP is sound (place_fields_check, run 37071718991: never empty, median 0.95 of the SP); their PPMIN,
PPMAX, MORNINGWAP and in-running columns are not read. PPWAP is a volume-weighted average over the pre-play market,
most of it in the last hour, so it stands in for a price some minutes before the off; the recorder's books at T-15
(from 1 Oct) are the strict test. The rules are not re-chosen here: 2018-22 and 2023-26Q1 are both read as out of sample
for rules chosen on 2010-17. 2% commission. Development years only (the holdout from 1 Apr 2026 is not read).
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402
from model.ordering import position_probabilities, simulate_finishing_orders  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 300)
COMM = 0.02
BANDS = ((2, 7), (8, 11), (12, 15), (16, 40))
FIT = {(2, 7): (0.734, 0.557), (8, 11): (0.804, 0.640), (12, 15): (0.839, 0.697), (16, 40): (0.829, 0.735)}  # scan 2
CHOSEN = [("1-2", "back", 0.00), ("1-2", "back", 0.02), ("1-2", "back", 0.05), ("2-3", "back", 0.10),
          ("3-5", "lay", 0.02), ("5-8", "back", 0.10), ("5-8", "lay", 0.10), ("8-13", "back", 0.10),
          ("13-21", "back", 0.05), ("13-21", "back", 0.10), ("50+", "lay", 0.00), ("50+", "lay", 0.02),
          ("50+", "lay", 0.05), ("50+", "lay", 0.10)]          # chosen on 2010-17 at the SP (scan 2, section C)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")

names = []
for name in bp.archive_index(s3, bucket):
    meta = bp.parse_file_name(name)
    if meta and date(2018, 1, 1) <= meta[2] <= date(2026, 4, 1):
        names.append(name)
KEEP = ["country", "market_type", "race_date", "event_id", "event_dt", "menu_hint", "event_name", "selection_id",
        "selection_name", "win_lose", "bsp", "ppwap", "pp_vol"]


def fetch(name):
    raw = s3.get_object(Bucket=bucket, Key=f"{bp.S3_PREFIX}/{name}")["Body"].read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        return bp.parse_file_text(text, name)[KEEP]
    except Exception as exc:
        print(f"  {name}: {type(exc).__name__}")
        return None


with ThreadPoolExecutor(32) as ex:
    d = pd.concat([p for p in ex.map(fetch, names) if p is not None], ignore_index=True)
d = d[(d.race_date >= "2018-01-01") & (d.race_date <= "2026-03-31")]
print(f"{len(names):,} UK/IE files, {len(d):,} rows ({time.time() - t0:.0f}s)")

# each horse's win and place rows of the same day (a horse runs once a day; the file's SELECTION_ID is the exchange's)
w = d[d.market_type == "win"].drop_duplicates(["country", "race_date", "selection_id"])
p = d[d.market_type == "place"].drop_duplicates(["country", "race_date", "selection_id"])
x = w.merge(p[["country", "race_date", "selection_id", "event_id", "event_dt", "win_lose", "bsp", "ppwap", "pp_vol"]],
            on=["country", "race_date", "selection_id"], suffixes=("", "_pl"))
print(f"joined win and place rows: {len(x):,}; the same start time on {(x.event_dt == x.event_dt_pl).mean():.1%}")
x = x[x.event_dt == x.event_dt_pl].copy()
x["raceid"] = x.event_id
x = x[(x.bsp > 1) & (x.ppwap > 1) & (x.bsp_pl > 1) & (x.ppwap_pl > 1)]
# a race is read only whole: every runner of the win market with both markets' WAPs and SPs
n_win = w[w.bsp > 1].groupby("event_id").size()
x["runners"] = x.groupby("raceid").selection_id.transform("size")
x = x[x.runners == x.raceid.map(n_win)]
x["paid"] = x.groupby("raceid").win_lose_pl.transform("sum").round().astype(int)
x = x[(x.paid >= 1) & (x.paid < x.runners) & (x.paid <= 4)].copy()
x["year"] = x.race_date.str[:4]
x["period"] = np.where(x.race_date < "2023-01-01", "2018-22", "2023-26Q1")
print(f"{x.raceid.nunique():,} races whole in both markets, {len(x):,} runners; places paid "
      f"{x.groupby('raceid').paid.first().value_counts().sort_index().to_dict()}")


def band_of(n):
    return next((b for b in BANDS if b[0] <= n <= b[1]), BANDS[-1])


def top3(pi, gm, dl):
    s = pi ** gm; s /= s.sum(); t = pi ** dl; t /= t.sum()
    A = pi[:, None] * s[None, :] / (1 - s[:, None]); np.fill_diagonal(A, 0.0)
    B = 1.0 / np.clip(1 - t[:, None] - t[None, :], 1e-12, None); np.fill_diagonal(B, 0.0)
    M = A * B
    return pi, A.sum(0), t * (M.sum() - M.sum(1) - M.sum(0))


def place_chance(frame, col, rng):
    out = np.full(len(frame), np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index)}
    for _, g in frame.groupby("raceid", sort=False):
        pi = (1 / g[col].to_numpy()); pi = pi / pi.sum()
        k = int(g.paid.iat[0])
        gm, dl = FIT[band_of(len(pi))]
        if k <= 3:
            p1, p2, p3 = top3(pi, gm, dl)
            q = p1 + (p2 if k >= 2 else 0) + (p3 if k >= 3 else 0)
        else:
            o = simulate_finishing_orders(pi, n_sims=4000, gamma=gm, delta=dl, rng=rng)
            q = position_probabilities(o, len(pi), n_positions=k).sum(axis=1)
        out[[pos[i] for i in g.index]] = np.clip(q, 1e-4, 1 - 1e-4)
    return out


rng = np.random.default_rng(0)
x["q_pre"] = place_chance(x, "ppwap", rng)                 # from the win market's pre-play WAP
x["q_sp"] = place_chance(x, "bsp", rng)                    # from the win SP, as scan 2 (for the comparison)
print(f"place chances done ({time.time() - t0:.0f}s)")
x["band_pre"] = pd.cut(x.ppwap, [1, 2, 3, 5, 8, 13, 21, 50, 1e9], right=False,
                       labels=["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-50", "50+"]).astype(str)
x["band_sp"] = pd.cut(x.bsp, [1, 2, 3, 5, 8, 13, 21, 50, 1e9], right=False,
                      labels=["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-50", "50+"]).astype(str)


def net(v):
    return np.where(v > 0, v * (1 - COMM), v)


placed = x.win_lose_pl.to_numpy() == 1
for tag, price in (("sp", x.bsp_pl.to_numpy()), ("pre", x.ppwap_pl.to_numpy())):
    x[f"back_{tag}"] = net(np.where(placed, price - 1, -1.0))                 # per GBP1 staked
    x[f"lay_{tag}"] = net(np.where(placed, -(price - 1), 1.0))                # per GBP1 laid (the backer's stake)
    x[f"liab_{tag}"] = price - 1                                              # the lay's liability per GBP1 laid


def summary(pnl: pd.Series, races: pd.Series, days: int, liab: pd.Series | None = None) -> dict:
    n = len(pnl)
    if n < 30:
        return {"n": n}
    by_race = pnl.groupby(races).sum()
    m = pnl.mean()
    se = by_race.std(ddof=1) * np.sqrt(len(by_race)) / n
    out = {"n": n, "per_day": round(n / max(days, 1), 1), "roi": m, "lo90": m - 1.645 * se, "hi90": m + 1.645 * se,
           "t": m / se}
    if liab is not None:
        out["per_liability"] = pnl.sum() / liab.sum()
    return out


def rule_mask(frame, band, side, over, ev_q, ev_price, band_col):
    if side == "back":
        ev = frame[ev_q] * (frame[ev_price] - 1) * (1 - COMM) - (1 - frame[ev_q])
    else:
        ev = (1 - frame[ev_q]) * (1 - COMM) - frame[ev_q] * (frame[ev_price] - 1)
    return (frame[band_col] == band) & (ev > over)


rows = []
for band, side, over in CHOSEN:
    for period in ("2018-22", "2023-26Q1"):
        f = x[x.period == period]
        base = {"rule": f"{side} {band} EV>{over:.2f}", "period": period}
        # at the SP, as scan 2 chose them (the reference); then decided on the pre-play WAPs
        m_sp = rule_mask(f, band, side, over, "q_sp", "bsp_pl", "band_sp")
        m_pre = rule_mask(f, band, side, over, "q_pre", "ppwap_pl", "band_pre")
        for how, m, col in (("SP rule, at the SP", m_sp, f"{side}_sp"),
                            ("pre-play rule, taken at the SP", m_pre, f"{side}_sp"),
                            ("pre-play rule, filled at the WAP", m_pre, f"{side}_pre")):
            liab = f.loc[m, "liab_" + col.split("_")[1]] if side == "lay" else None
            r = summary(f.loc[m, col], f.loc[m, "raceid"], f.race_date.nunique(), liab)
            rows.append({**base, "how": how, **r,
                         "place_pp_vol_med": float(f.loc[m, "pp_vol_pl"].median()) if m.any() else np.nan})
res = pd.DataFrame(rows)
print("\n== the 14 rules chosen on 2010-17 (ROI per GBP1 staked for backs, per GBP1 laid for lays; per_liability for lays)")
print(res.round(4).to_string(index=False))

print("\n== the place pocket (win price <= 5.0, EV_back > 0.05), by year")
pk = []
for year, f in x.groupby("year"):
    for how, m, col in (("SP rule, at the SP", (f.bsp <= 5) & (f.q_sp * (f.bsp_pl - 1) * (1 - COMM) - (1 - f.q_sp) > 0.05), "back_sp"),
                        ("pre-play, taken at the SP", (f.ppwap <= 5) & (f.q_pre * (f.ppwap_pl - 1) * (1 - COMM) - (1 - f.q_pre) > 0.05), "back_sp"),
                        ("pre-play, filled at the WAP", (f.ppwap <= 5) & (f.q_pre * (f.ppwap_pl - 1) * (1 - COMM) - (1 - f.q_pre) > 0.05), "back_pre")):
        pk.append({"year": year, "how": how, **summary(f.loc[m, col], f.loc[m, "raceid"], f.race_date.nunique())})
print(pd.DataFrame(pk).round(4).to_string(index=False))

print("\n== the pre-play WAPs against the SPs on the runners read (median ratio)")
print(x.groupby("band_sp").apply(lambda g: pd.Series({
    "n": len(g), "win_pre/sp": (g.ppwap / g.bsp).median(), "place_pre/sp": (g.ppwap_pl / g.bsp_pl).median(),
    "q_pre-q_sp": (g.q_pre - g.q_sp).median()})).round(3).to_string())
print(f"\ndone in {time.time() - t0:.0f}s")
