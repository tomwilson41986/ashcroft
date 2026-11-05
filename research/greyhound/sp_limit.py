"""Rules at the Betfair SP with a price limit (read-only research; the owner, 4 Nov 2026: keep working until five
profitable strategies, "fundamental is always just to get the most accurate closing price").

A limit-on-close order is matched at the SP when the SP is at or beyond its limit, and not at all otherwise: no queue,
and the rule sees nothing it would not see before the off (our price, made from the card). Each rule here:

- ``win_back``: back at the win SP when the SP is at least (1 + m) / p, p our chance of winning;
- ``win_lay``: lay at the win SP when the SP is at most (1 - m) / p;
- ``place_back`` / ``place_lay``: the same in the place market at the place SP, p our chance of placing (Harville on
  our win chances, the places the market paid);

for each margin m in ``MARGINS``. Our chances are taken as the model gives them (``raw``) and tempered (p^a normalised
in the race, a fitted on the first year by the winners' likelihood) to undo any spread the model's chances lack.

Pre-registered: every rule is chosen on the development year (``DEV``) and read on the years after it (the holdout);
a rule counts only if its holdout return's 90% lower bound (a bootstrap over days) is above zero AND its development
return was above zero. Every rule tried is reported, so the count of rules is plain (about one in twenty clears a 90%
one-sided bound by chance). Commission 2% on each bet's winnings (5% too). Stakes: a unit stake each bet (backer's
stake on a lay), returns per unit staked and, for lays, per unit of liability.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from research.greyhound import place_value as pv

log = logging.getLogger(__name__)

MARGINS = (0.0, 0.1, 0.2, 0.3, 0.5)
COMMS = (0.02, 0.05)
DEV = (2024,)


def temper_fit(p: np.ndarray, won: np.ndarray, race: np.ndarray) -> float:
    """The exponent a maximising the winners' likelihood of p^a normalised in each race (a grid, then refined)."""
    codes, idx = np.unique(race, return_inverse=True)
    lp = np.log(np.clip(p, 1e-9, 1))

    def nll(a):
        z = np.exp(a * lp)
        s = np.bincount(idx, weights=z)
        return -np.sum(won * (a * lp - np.log(s[idx])))

    grid = np.linspace(0.5, 2.0, 31)
    a0 = grid[int(np.argmin([nll(a) for a in grid]))]
    fine = np.linspace(a0 - 0.05, a0 + 0.05, 21)
    return float(fine[int(np.argmin([nll(a) for a in fine]))])


def temper(p: np.ndarray, race: np.ndarray, a: float) -> np.ndarray:
    z = np.power(np.clip(p, 1e-9, 1), a)
    return z / pd.Series(z).groupby(race).transform("sum").to_numpy()


def place_chance(p: np.ndarray, race: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Harville's chance of finishing in the first k for each runner (k the places the race paid)."""
    codes, idx = np.unique(race, return_inverse=True)
    pos = pd.Series(np.arange(len(race))).groupby(idx).cumcount().to_numpy()
    if pos.max() >= pv.MAXN:
        raise ValueError("a race with more than 8 runners")
    Q = np.zeros((len(codes), pv.MAXN))
    Q[idx, pos] = p
    K = pd.Series(k).groupby(idx).first().reindex(range(len(codes))).to_numpy()
    return pv.harville(Q, K)[idx, pos]


def join_place(pred: pd.DataFrame, store, years) -> pd.DataFrame:
    """Each predicted runner's place SP and place result from the price files (day, time, track, trap)."""
    j = pv.load(store, years)
    j = j[["race_date", "race_time", "track", "trap", "bsp_p", "win_lose_p", "bsp_w", "win_lose_w"]]
    x = pred.copy()
    x["race_time"] = x.race_time.astype(str).str[:5]
    x["trap"] = pd.to_numeric(x.trap, errors="coerce")
    out = x.merge(j, on=["race_date", "race_time", "track", "trap"], how="left")
    log.info("place prices joined: %d of %d runners", int(out.bsp_p.notna().sum()), len(out))
    return out


def _rules(d: pd.DataFrame, p: np.ndarray, sp: np.ndarray, hit: np.ndarray, market: str, label: str) -> list[dict]:
    out = []
    for c in COMMS:
        for m in MARGINS:
            for side in ("back", "lay"):
                if side == "back":
                    sel = sp >= (1 + m) / np.clip(p, 1e-9, None)
                    pnl = np.where(hit == 1, (sp - 1) * (1 - c), -1.0)
                    risk = np.ones_like(sp)
                else:
                    sel = sp <= (1 - m) / np.clip(p, 1e-9, None)
                    pnl = np.where(hit == 1, -(sp - 1), 1 - c)
                    risk = sp - 1
                sel &= np.isfinite(sp) & (sp > 1) & np.isfinite(p)
                row = {"market": market, "side": side, "p": label, "margin": m, "comm": c}
                for part, yrs in (("dev", d.year.isin(DEV)), ("holdout", ~d.year.isin(DEV))):
                    s = sel & yrs.to_numpy()
                    if s.sum() < 50:
                        row[part] = {"bets": int(s.sum())}
                        continue
                    t = pd.DataFrame({"day": d.race_date.to_numpy()[s], "pnl": pnl[s], "risk": risk[s]})
                    byd = t.groupby("day").agg(pnl=("pnl", "sum"), n=("pnl", "size"), risk=("risk", "sum"))
                    lo, hi = pv._boot(byd.pnl, byd.n)
                    row[part] = {"bets": int(s.sum()), "days": len(byd), "mean_sp": round(float(sp[s].mean()), 2),
                                 "roi%": round(100 * t.pnl.mean(), 2), "ci90%": [round(100 * lo, 2), round(100 * hi, 2)],
                                 "roi_risk%": round(100 * t.pnl.sum() / t.risk.sum(), 2)}
                h, dv = row.get("holdout", {}), row.get("dev", {})
                row["passes"] = bool(h.get("ci90%") and h["ci90%"][0] > 0 and dv.get("roi%", -1) > 0)
                out.append(row)
    return out


def run(pred: pd.DataFrame, store=None, place: bool = True) -> dict:
    d = pred.copy()
    d["race_date"] = d.race_date.astype(str)
    d["year"] = pd.to_datetime(d.race_date).dt.year
    d = d.sort_values(["race_id", "trap"]).reset_index(drop=True)
    race = d.race_id.astype(str).to_numpy()
    won = pd.to_numeric(d.won, errors="coerce").fillna(0).to_numpy()
    dev = d.year.isin(DEV).to_numpy()
    a = temper_fit(d.p_model.to_numpy(float)[dev], won[dev], race[dev])
    d["p_raw"] = d.p_model.to_numpy(float)
    d["p_tempered"] = temper(d.p_raw.to_numpy(), race, a)
    rep = {"runners": len(d), "races": int(d.race_id.nunique()), "years": sorted(int(y) for y in d.year.unique()),
           "temper_a_dev": round(a, 3), "rules": []}
    sp = pd.to_numeric(d.get("bsp"), errors="coerce").to_numpy(float)
    for lab in ("p_raw", "p_tempered"):
        rep["rules"] += _rules(d, d[lab].to_numpy(), sp, won, "win", lab)
    if place and store is not None:
        d = join_place(d, store, range(min(rep["years"]), max(rep["years"]) + 1))
        g = d.groupby("race_id")
        whole = g.bsp_p.transform(lambda s: s.notna().all()) & g.trap.transform("size").between(4, pv.MAXN)
        k = g.win_lose_p.transform("sum")
        ok = (whole & k.isin([2, 3]) & (k < g.trap.transform("size"))).to_numpy()
        x = d[ok].reset_index(drop=True)
        rep["place_runners"] = len(x)
        rep["place_races"] = int(x.race_id.nunique())
        for lab in ("p_raw", "p_tempered"):
            pp = place_chance(x[lab].to_numpy(), x.race_id.astype(str).to_numpy(),
                              x.groupby("race_id").win_lose_p.transform("sum").to_numpy())
            rep["rules"] += _rules(x, pp, x.bsp_p.to_numpy(float), x.win_lose_p.to_numpy(), "place", lab)
    rep["passing"] = [r for r in rep["rules"] if r["passes"]]
    rep["rules_tried"] = len(rep["rules"])
    return rep


def summary(rep: dict) -> str:
    L = [f"runners {rep['runners']}, races {rep['races']}, years {rep['years']}, temper a (dev) {rep['temper_a_dev']}",
         f"place: {rep.get('place_runners')} runners, {rep.get('place_races')} races; rules tried {rep['rules_tried']}",
         "market | side | p | margin | comm | dev bets / ROI% | holdout bets / ROI% [90%] | passes"]
    for r in rep["rules"]:
        dv, h = r.get("dev", {}), r.get("holdout", {})
        L.append(f"{r['market']} | {r['side']} | {r['p']} | {r['margin']} | {r['comm']} | {dv.get('bets')} / "
                 f"{dv.get('roi%')} | {h.get('bets')} / {h.get('roi%')} {h.get('ci90%')} | {r['passes']}")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    import sys
    from sources.common import Store
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--no-place", action="store_true")
    a = ap.parse_args()
    r = run(pd.read_parquet(a.pred), Store(), place=not a.no_place)
    print(summary(r))
    print(json.dumps(r["passing"], indent=1))
    sys.exit(0)
