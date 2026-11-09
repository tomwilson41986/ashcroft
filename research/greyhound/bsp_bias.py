"""The SP's bias by segment, as limit-on-close rules (read-only research). Each rule: in a segment (the track and trap,
the track, the trap, the field size, the hour of the race), back at the SP when the SP is at least a threshold, or lay
at the SP when it is at most one. Betfair applies the threshold at the off (a limit-on-close order): nothing is seen
that a bet struck before the off would not see, and the fill is the SP.

Many segments are tried, so the selection is the test's weakest point; it is guarded thus:
- choose on the development years only (``DEV``): a rule is selected when its development return's 90% lower bound
  (normal, from the per-day returns) is above zero on at least ``MIN_BETS`` bets;
- read the selected rules on the holdout years together, as one portfolio (one test, not hundreds), with a bootstrap
  over days; and each on its own;
- report how many rules were tried and selected.

Win and place markets (Betfair's greyhound price files, GB). Commission 2%.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from research.greyhound import place_value as pv

log = logging.getLogger(__name__)

DEV = range(2016, 2022)
BACK_MIN = (1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0)
LAY_MAX = (1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
from greyhound.costs import COMMISSION as COMM   # the account's rate (greyhound/costs.py)
MIN_BETS = 300
Z90 = 1.2816


def frame(store, years=range(2016, 2027), country: str = "GB") -> pd.DataFrame:
    j = pv.load(store, years, country)
    j["year"] = pd.to_datetime(j.race_date).dt.year
    j["race"] = j.race_date + " " + j.race_time.astype(str) + " " + j.track
    j["field"] = j.groupby("race").trap.transform("size")
    j["hour"] = pd.to_numeric(j.race_time.astype(str).str[:2], errors="coerce")
    j["session"] = pd.cut(j.hour, [0, 13, 17, 24], labels=["morning", "afternoon", "evening"], right=False).astype(str)
    return j


SEGMENTS = (("track", "trap"), ("track",), ("trap",), ("field",), ("session",), ("track", "session"))


def _pnl(sp, hit, side):
    return np.where(hit == 1, (sp - 1) * (1 - COMM), -1.0) if side == "back" else np.where(hit == 1, -(sp - 1), 1 - COMM)


def candidates(j: pd.DataFrame, market: str) -> pd.DataFrame:
    """Every rule's development and holdout return, by segment, side and threshold."""
    sp, hit = j[f"bsp_{market[0]}"].to_numpy(float), j[f"win_lose_{market[0]}"].to_numpy(float)
    ok = sp > 1
    dev = j.year.isin(DEV).to_numpy()
    rows = []
    for side, ths in (("back", BACK_MIN), ("lay", LAY_MAX)):
        pnl = _pnl(sp, hit, side)
        for th in ths:
            sel = ok & ((sp >= th) if side == "back" else (sp <= th))
            x = j.loc[sel, ["race_date", "year", *{c for s in SEGMENTS for c in s}]].assign(pnl=pnl[sel],
                                                                                            dev=dev[sel])
            for seg in SEGMENTS:
                for part, m in (("dev", x.dev), ("hold", ~x.dev)):
                    y = x[m]
                    byday = y.groupby(list(seg) + ["race_date"]).pnl.agg(["sum", "size"])
                    g = byday.groupby(level=list(range(len(seg))))
                    agg = pd.DataFrame({"bets": g["size"].sum(), "pnl": g["sum"].sum(), "days": g.size(),
                                        "day_sd": g["sum"].std()})
                    agg[f"roi"] = agg.pnl / agg.bets
                    # the day totals' spread gives the sum's standard error; the lower bound of the return
                    agg["lo90"] = (agg.pnl - Z90 * agg.day_sd * np.sqrt(agg.days)) / agg.bets
                    agg.columns = [f"{part}_{c}" for c in agg.columns]
                    rows.append(agg.reset_index().assign(market=market, side=side, threshold=th,
                                                         segment="+".join(seg), part=part))
    d = pd.concat(rows, ignore_index=True)
    keys = ["market", "side", "threshold", "segment", "track", "trap", "field", "session"]
    for k in keys:
        if k not in d:
            d[k] = np.nan
    d[keys[3:]] = d[keys[3:]].astype(str)
    dv = d[d.part == "dev"].drop(columns=[c for c in d if c.startswith("hold_")] + ["part"])
    ho = d[d.part == "hold"].drop(columns=[c for c in d if c.startswith("dev_")] + ["part"])
    return dv.merge(ho, on=keys, how="left")


def portfolio(j: pd.DataFrame, picks: pd.DataFrame, market: str) -> dict:
    """The selected rules' holdout bets together (a runner taken once per side even if several rules pick it)."""
    sp, hit = j[f"bsp_{market[0]}"].to_numpy(float), j[f"win_lose_{market[0]}"].to_numpy(float)
    hold = ~j.year.isin(DEV).to_numpy()
    out = {}
    for side in ("back", "lay"):
        p = picks[(picks.market == market) & (picks.side == side)]
        if not len(p):
            continue
        take = np.zeros(len(j), bool)
        for r in p.itertuples(index=False):
            m = (sp >= r.threshold) if side == "back" else (sp <= r.threshold)
            for c in r.segment.split("+"):
                v = getattr(r, c)
                col = j[c].astype(str).str.replace(r"\.0$", "", regex=True)
                m &= (col == str(v).replace(".0", "")).to_numpy()
            take |= m
        take &= hold & (sp > 1)
        pnl = _pnl(sp, hit, side)[take]
        t = pd.DataFrame({"day": j.race_date.to_numpy()[take], "pnl": pnl})
        byd = t.groupby("day").agg(pnl=("pnl", "sum"), n=("pnl", "size"))
        lo, hi = pv._boot(byd.pnl, byd.n)
        out[side] = {"rules": len(p), "bets": int(take.sum()), "days": len(byd),
                     "roi%": round(100 * float(pnl.mean()), 2) if len(pnl) else None,
                     "ci90%": [round(100 * lo, 2), round(100 * hi, 2)],
                     "by_year": {int(y): round(100 * float(g.mean()), 2)
                                 for y, g in pd.Series(pnl, index=j.year.to_numpy()[take]).groupby(level=0)}}
    return out


def run(store, country: str = "GB") -> dict:
    j = frame(store, country=country)
    rep = {"runners": len(j), "years": sorted(int(y) for y in j.year.unique())}
    for market in ("win", "place"):
        c = candidates(j, market)
        picks = c[(c.dev_bets >= MIN_BETS) & (c.dev_lo90 > 0)].sort_values("dev_lo90", ascending=False)
        rep[market] = {"rules_tried": len(c[c.dev_bets >= MIN_BETS]), "selected": len(picks),
                       "portfolio": portfolio(j, picks, market),
                       "selected_rules": picks.head(40)[["side", "threshold", "segment", "track", "trap", "field",
                                                         "session", "dev_bets", "dev_roi", "dev_lo90", "hold_bets",
                                                         "hold_roi", "hold_lo90"]].round(4).to_dict("records")}
    return rep


if __name__ == "__main__":
    from sources.common import Store
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(Store()), indent=1, default=str))
