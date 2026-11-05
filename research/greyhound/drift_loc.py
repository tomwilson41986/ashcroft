"""Limit-on-close rules against the morning price (read-only research). A back at the SP limited to at least (1 + d)
times the runner's morning weighted average price is matched only if the dog drifts that far by the off; a lay limited
to at most (1 - d) times it only if the dog shortens that far. Betfair applies the condition at the off, so the rule
sees nothing it could not see in the morning: no look-ahead, no queue.

Data: Betfair's greyhound price files (sources/greyhound_prices.py): the morning WAP (held for 9-44% of GB runners a
year), the SP, the result, win and place markets. Pre-registered: development years ``DEV`` (2016-2021) choose, the
holdout (2022 on) reads; a rule counts only with its holdout 90% lower bound (bootstrap over days) above zero and a
positive development return. Every rule tried is reported. Commission 2% (5% too).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from research.greyhound import place_value as pv

log = logging.getLogger(__name__)

DRIFTS = (0.1, 0.2, 0.3, 0.5, 1.0)
SHORTENS = (0.1, 0.2, 0.3, 0.5)
MIN_VOL = (0.0, 50.0)
COMMS = (0.02, 0.05)
DEV = range(2016, 2022)


def score(x: pd.DataFrame, market: str) -> list[dict]:
    sp = x[f"bsp_{market[0]}"].to_numpy(float)
    morning = x[f"morningwap_{market[0]}"].to_numpy(float)
    hit = x[f"win_lose_{market[0]}"].to_numpy(float)
    vol = x[f"morning_vol_{market[0]}"].fillna(0).to_numpy(float)
    dev = x.year.isin(DEV).to_numpy()
    out = []
    for c in COMMS:
        for v in MIN_VOL:
            base = (morning > 1) & (sp > 1) & (vol >= v)
            for side, params in (("back", DRIFTS), ("lay", SHORTENS)):
                for d in params:
                    if side == "back":
                        sel = base & (sp >= morning * (1 + d))
                        pnl = np.where(hit == 1, (sp - 1) * (1 - c), -1.0)
                        risk = np.ones_like(sp)
                    else:
                        sel = base & (sp <= morning * (1 - d))
                        pnl = np.where(hit == 1, -(sp - 1), 1 - c)
                        risk = sp - 1
                    row = {"market": market, "side": side, "move": d, "min_morning_vol": v, "comm": c}
                    for part, mask in (("dev", dev), ("holdout", ~dev)):
                        s = sel & mask
                        if s.sum() < 50:
                            row[part] = {"bets": int(s.sum())}
                            continue
                        t = pd.DataFrame({"day": x.race_date.to_numpy()[s], "pnl": pnl[s], "risk": risk[s]})
                        byd = t.groupby("day").agg(pnl=("pnl", "sum"), n=("pnl", "size"))
                        lo, hi = pv._boot(byd.pnl, byd.n)
                        row[part] = {"bets": int(s.sum()), "days": len(byd), "mean_sp": round(float(sp[s].mean()), 2),
                                     "roi%": round(100 * t.pnl.mean(), 2),
                                     "ci90%": [round(100 * lo, 2), round(100 * hi, 2)],
                                     "roi_risk%": round(100 * t.pnl.sum() / t.risk.sum(), 2)}
                    h, dv = row.get("holdout", {}), row.get("dev", {})
                    row["passes"] = bool(h.get("ci90%") and h["ci90%"][0] > 0 and dv.get("roi%", -1) > 0)
                    out.append(row)
    return out


def run(store, years=range(2016, 2027), country: str = "GB") -> dict:
    j = pv.load(store, years, country)
    j["year"] = pd.to_datetime(j.race_date).dt.year
    rep = {"runners": len(j), "win_with_morning": int((j.morningwap_w > 1).sum()),
           "place_with_morning": int((j.morningwap_p > 1).sum()), "rules": []}
    for market in ("win", "place"):
        rep["rules"] += score(j, market)
    rep["rules_tried"] = len(rep["rules"])
    rep["passing"] = [r for r in rep["rules"] if r["passes"]]
    return rep


def summary(rep: dict) -> str:
    L = [f"runners {rep['runners']}; with a morning WAP: win {rep['win_with_morning']}, place {rep['place_with_morning']};"
         f" rules tried {rep['rules_tried']}",
         "market | side | move | min vol | comm | dev bets / ROI% [90%] | holdout bets / ROI% [90%] | passes"]
    for r in rep["rules"]:
        dv, h = r.get("dev", {}), r.get("holdout", {})
        L.append(f"{r['market']} | {r['side']} | {r['move']} | {r['min_morning_vol']} | {r['comm']} | {dv.get('bets')} /"
                 f" {dv.get('roi%')} {dv.get('ci90%')} | {h.get('bets')} / {h.get('roi%')} {h.get('ci90%')} | {r['passes']}")
    return "\n".join(L)


if __name__ == "__main__":
    from sources.common import Store
    logging.basicConfig(level=logging.INFO)
    r = run(Store())
    print(summary(r))
    print("PASSING:", json.dumps(r["passing"], indent=1))
