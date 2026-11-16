"""Is a trap's chance missing from the Betfair SP, and from our model? (the owner, 16 Nov 2026: "Measure our model's
error against the Betfair SP by trap, to check the trap 4 finding"; ledger greyhound-trap-residual-1116).

Butler, Butler and Gaine (2025, Review of Behavioral Economics; https://hdl.handle.net/10468/18155), on 78,595 GB runs
of 2018, found trap 4 a significant disadvantage "not reflected in market odds" (the bookmakers' SP). Two readings:

1. **The market** (Betfair's price files, GB win markets, every race with six runners and a BSP for each): a runner's
   chance at the BSP (1/BSP over the race's sum, the overround taken out) against its result, by trap: the actual
   over expected winners (A/E), the level-stake return backing every runner at the BSP after commission, each with a
   90% interval from a bootstrap over days; by year and by track for trap 4.
2. **Our model** (a fast-loop arm's saved out-of-sample predictions: race_id, dog_id, race_date, bsp, won, pred_bsp,
   p_model; the trap from GBGB's results): the error of log(BSP) on log(the model's BSP) by trap (bias and RMSE),
   and the model's win chance against the result by trap (A/E).

Read-only, S3 only. Writes out/trap_residual/{report.json, report.md}.

    python -m research.greyhound.trap_residual --years 2019-2026 --pred greyhound/research/pred/stack_mls_live_oos.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COMMISSION = 0.02
DRAWS = 2000
MIN_TRACK = 1000


def market_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """One row a runner of a six-runner race where every runner has a BSP above 1 and the race one winner."""
    p = prices.copy()
    p["bsp"] = pd.to_numeric(p.bsp, errors="coerce")
    p["trap"] = pd.to_numeric(p.trap, errors="coerce")
    p["won"] = pd.to_numeric(p.win_lose, errors="coerce")
    p = p.dropna(subset=["bsp", "trap", "won"])
    p = p[(p.bsp > 1) & p.trap.between(1, 6)]
    key = ["race_date", "race_time", "track"]
    g = p.groupby(key)
    ok = (g.trap.transform("size") == 6) & (g.trap.transform("nunique") == 6) & (g.won.transform("sum") == 1)
    p = p[ok].copy()
    inv = 1.0 / p.bsp
    p["p_mkt"] = inv / inv.groupby([p[k] for k in key]).transform("sum")
    p["ret"] = np.where(p.won == 1, (p.bsp - 1.0) * (1.0 - COMMISSION), -1.0)
    p["day"] = p.race_date.astype(str)
    p["year"] = p.day.str[:4]
    return p


def _boot(day: np.ndarray, num: np.ndarray, den: np.ndarray, draws: int = DRAWS, seed: int = 0) -> tuple[float, float]:
    """90% interval of sum(num)/sum(den) from a bootstrap over days."""
    d = pd.DataFrame({"d": day, "n": num, "e": den}).groupby("d")[["n", "e"]].sum()
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (draws, len(d)))
    n, e = d.n.to_numpy()[idx].sum(1), d.e.to_numpy()[idx].sum(1)
    r = n / e
    return float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))


def by_trap(m: pd.DataFrame, group: list[str] | None = None) -> list[dict]:
    rows = []
    keys = (group or []) + ["trap"]
    for k, x in m.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        ae = float(x.won.sum() / x.p_mkt.sum())
        lo, hi = _boot(x.day.to_numpy(), x.won.to_numpy(float), x.p_mkt.to_numpy())
        roi = float(x.ret.mean())
        rlo, rhi = _boot(x.day.to_numpy(), x.ret.to_numpy(), np.ones(len(x)))
        rows.append({**dict(zip(keys, [str(v) if not isinstance(v, (int, float)) else int(v) for v in k])),
                     "runners": int(len(x)), "wins": int(x.won.sum()), "win%": round(100 * float(x.won.mean()), 2),
                     "mkt%": round(100 * float(x.p_mkt.mean()), 2), "ae": round(ae, 4),
                     "ae_90": [round(lo, 4), round(hi, 4)], "roi_bsp%": round(100 * roi, 2),
                     "roi_90": [round(100 * rlo, 2), round(100 * rhi, 2)]})
    return rows


def model_by_trap(pred: pd.DataFrame, traps: pd.DataFrame) -> list[dict]:
    d = pred.merge(traps, on=["race_id", "dog_id"], how="inner")
    d = d[(d.bsp > 1) & (d.pred_bsp > 1) & d.trap.between(1, 6)].copy()
    d["day"] = d.race_date.astype(str)
    d["err"] = np.log(d.bsp.astype(float)) - np.log(d.pred_bsp.astype(float))
    d["p"] = d.p_model.astype(float)
    rows = []
    for t, x in d.groupby("trap"):
        rng = np.random.default_rng(int(t))
        g = x.groupby("day").err.agg(["sum", "size"])
        idx = rng.integers(0, len(g), (DRAWS, len(g)))
        bias = g["sum"].to_numpy()[idx].sum(1) / g["size"].to_numpy()[idx].sum(1)
        lo, hi = _boot(x.day.to_numpy(), x.won.to_numpy(float), x.p.to_numpy())
        rows.append({"trap": int(t), "runners": int(len(x)), "bias_log_bsp": round(float(x.err.mean()), 4),
                     "bias_90": [round(float(np.quantile(bias, 0.05)), 4), round(float(np.quantile(bias, 0.95)), 4)],
                     "rmse": round(float(np.sqrt((x.err ** 2).mean())), 4),
                     "model_ae": round(float(x.won.sum() / x.p.sum()), 4), "model_ae_90": [round(lo, 4), round(hi, 4)]})
    return rows


def report_md(out: dict) -> str:
    L = ["# Trap against the Betfair SP", "", f"Market: {out['market']['races']:,} six-runner races, "
         f"{out['market']['from']} to {out['market']['to']}; A/E is winners over the BSP's chance (overround out); "
         f"the return backs every runner at the BSP after {COMMISSION:.0%} commission. 90% intervals by day.", "",
         "| Trap | Runners | Win % | BSP chance % | A/E | 90% | Return % | 90% |", "|---|---|---|---|---|---|---|---|"]
    for r in out["market"]["all"]:
        L.append(f"| {r['trap']} | {r['runners']:,} | {r['win%']} | {r['mkt%']} | {r['ae']} | {r['ae_90']} | "
                 f"{r['roi_bsp%']} | {r['roi_90']} |")
    L += ["", "Trap 4 by year:", "", "| Year | Runners | A/E | 90% | Return % |", "|---|---|---|---|---|"]
    for r in out["market"]["by_year"]:
        if r["trap"] == 4:
            L.append(f"| {r['year']} | {r['runners']:,} | {r['ae']} | {r['ae_90']} | {r['roi_bsp%']} |")
    L += ["", f"Trap 4 by track (tracks with {MIN_TRACK:,}+ trap-4 runners):", "",
          "| Track | Runners | A/E | 90% | Return % |", "|---|---|---|---|---|"]
    for r in out["market"]["trap4_by_track"]:
        L.append(f"| {r['track']} | {r['runners']:,} | {r['ae']} | {r['ae_90']} | {r['roi_bsp%']} |")
    if out.get("model"):
        L += ["", f"Model ({out['model_source']}): error of log(BSP) on log(model BSP) by trap (above 0: the BSP is "
              "longer than the model said), RMSE, and the model's win chance against the result.", "",
              "| Trap | Runners | Bias | 90% | RMSE | Model A/E | 90% |", "|---|---|---|---|---|---|---|"]
        for r in out["model"]:
            L.append(f"| {r['trap']} | {r['runners']:,} | {r['bias_log_bsp']} | {r['bias_90']} | {r['rmse']} | "
                     f"{r['model_ae']} | {r['model_ae_90']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from greyhound.model import load_prices
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", default="2019-2026")
    ap.add_argument("--pred", default="greyhound/research/pred/stack_mls_live_oos.parquet")
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/trap_residual")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    m = market_frame(load_prices(store, a.years))
    log.info("market: %d runners, %d races", len(m), len(m) // 6)
    big = m[m.trap == 4].groupby("track").size()
    t4 = by_trap(m[(m.trap == 4) & m.track.isin(big[big >= MIN_TRACK].index)], ["track"])
    out = {"market": {"runners": int(len(m)), "races": int(len(m) // 6), "from": str(m.day.min()), "to": str(m.day.max()),
                      "all": by_trap(m), "by_year": by_trap(m, ["year"]),
                      "trap4_by_track": sorted(t4, key=lambda r: r["ae"])}}
    pred = store.get_parquet(a.pred) if a.pred else None
    if pred is not None and len(pred):
        from greyhound.data import load_runs
        years = sorted({int(str(d)[:4]) for d in pred.race_date.unique()})
        runs = load_runs(store, years=years)
        traps = runs[["race_id", "dog_id", "trap"]].assign(trap=lambda x: pd.to_numeric(x.trap, errors="coerce"))
        out["model"] = model_by_trap(pred, traps.drop_duplicates(["race_id", "dog_id"]))
        out["model_source"] = f"{a.pred}, {pred.race_date.min()} to {pred.race_date.max()}"
    else:
        out["model_note"] = "no data found: no saved predictions at " + str(a.pred)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "report.json").write_text(json.dumps(out, indent=1, default=str))
    md = report_md(out)
    (o / "report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
