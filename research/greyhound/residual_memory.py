"""The residual memory: a layer on top of the greyhound price that remembers, dog by dog, how the market has differed
from the model.

The owner, 11 Nov 2026: "identify new forms of modelling that haven't been discovered yet". The model prices a dog
from its form. The Betfair SP then says what the market made of it. The difference - the model's miss on that dog,
log(BSP) less log(our price), against the rest of its race - is not noise from one race to the next: on the stack's
out-of-sample prices of 1 Jun to 5 Nov 2026 a dog's miss correlates 0.146 with its previous one, and the mean of
its earlier misses 0.14 with the next (scratch probe, 11 Nov; batch 24's stack3_all4). The market knows something
lasting about the dog that the form does not show, or the model is wrong about it in a lasting way; either way the
miss can be carried forward. A memory built without the model's own prices (the SP against the expected time) adds
nothing (correlation with the stack's miss -0.02): the model already has the market's past view of the dog. What it
does not have is ITS OWN running error.

The layer, for a runner on day D:

    m = (n / (n + K)) x mean of the dog's misses on days before D        (0 with none)
    log price' = log price + b x (m less the race's mean m): the race's mean log price stays
    chance'    proportional to chance x exp(-c x m), normalised in the race

``K`` (runs of shrinkage), ``b`` (for the price) and ``c`` (for the chance) are fitted on the FIRST part of a
window of out-of-sample prices and the layer is scored on the rest, each runner reading only misses of earlier days
(the fit period's among them: they are the past by then). Nothing here is fitted on the days it is scored on.

Live, the misses are the forward record's: each day's served prices against the BSP that followed.

    python -m research.greyhound.residual_memory --pred out/pred_f_stack.parquet --split 2026-09-01 --out out/memory.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

KS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0)


def misses(pred: pd.DataFrame) -> pd.DataFrame:
    """The frame with each runner's miss (log BSP less log price, less the race's mean of it; NaN where the runner or
    its race has no BSP) and, from the dog's EARLIER days, the sum and the count of its misses."""
    p = pred.copy()
    ok = (p.bsp > 1) & (p.pred_bsp > 1)
    e = (np.log(p.bsp.where(ok)) - np.log(p.pred_bsp.where(ok)))
    whole = ok.groupby(p.race_id).transform("all")
    p["miss"] = (e - e.groupby(p.race_id).transform("mean")).where(whole)
    p["_d"] = pd.to_datetime(p.race_date)
    # one row a dog-day (a dog rarely runs twice in a day; both runs then read the same earlier days)
    day = p.groupby(["dog_id", "_d"], as_index=False).agg(s=("miss", "sum"), n=("miss", "count")).sort_values(["dog_id", "_d"])
    g = day.groupby("dog_id")
    day["sum_before"] = g.s.cumsum() - day.s
    day["n_before"] = g.n.cumsum() - day.n
    p = p.merge(day[["dog_id", "_d", "sum_before", "n_before"]], on=["dog_id", "_d"], how="left")
    return p.drop(columns="_d")


def memory(p: pd.DataFrame, k: float) -> np.ndarray:
    n = p.n_before.to_numpy(float)
    return np.where(n > 0, p.sum_before.to_numpy(float) / (n + k), 0.0)


def _centre(x: np.ndarray, race: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return (s - s.groupby(race).transform("mean")).to_numpy()


def fit(p: pd.DataFrame) -> dict:
    """K and b by least squares of the miss on the memory (both race-centred), c by the winners' likelihood, on ``p``."""
    race = p.race_id.to_numpy()
    y = p.miss.to_numpy(float)
    have = np.isfinite(y)
    best = None
    for k in KS:
        x = _centre(memory(p, k), race)
        b = float((x[have] * y[have]).sum() / max((x[have] ** 2).sum(), 1e-12))
        sse = float(((y[have] - b * x[have]) ** 2).mean())
        if best is None or sse < best["mse"]:
            best = {"k": k, "b": round(b, 4), "mse": sse}
    from scipy.optimize import minimize_scalar
    m = memory(p, best["k"])
    lp = np.log(np.clip(p.p_model.to_numpy(float), 1e-12, 1.0))
    won = p.won.to_numpy(float)

    def nll(c: float) -> float:
        z = pd.Series(lp - c * m)
        mx = z.groupby(race).transform("max")
        lse = np.log(np.exp(z - mx).groupby(race).transform("sum")) + mx
        return float(-(won * (z - lse)).sum())

    c = float(minimize_scalar(nll, bounds=(-1.0, 2.0), method="bounded").x)
    return {"k": best["k"], "b": best["b"], "c": round(c, 4), "fit_runners": int(have.sum()),
            "fit_runners_with_a_memory": int((have & (p.n_before.to_numpy() > 0)).sum())}


def apply(p: pd.DataFrame, prm: dict) -> pd.DataFrame:
    """The frame with the layer's price and chance (``pred_bsp_m``, ``p_model_m``): the race's book kept."""
    out = p.copy()
    race = out.race_id.to_numpy()
    m = memory(out, prm["k"])
    z = np.log(np.clip(out.p_model.to_numpy(float), 1e-12, 1.0)) - prm["c"] * m
    s = pd.Series(np.exp(z - pd.Series(z).groupby(race).transform("max").to_numpy()))
    out["p_model_m"] = (s / s.groupby(race).transform("sum")).to_numpy()
    # the price: the log price moved by b x the memory against the race's mean memory (the race's mean log price stays)
    out["pred_bsp_m"] = np.exp(np.log(out.pred_bsp.to_numpy(float)) + prm["b"] * _centre(m, race))
    return out


def score(p: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """RMSE of log(BSP) and log-loss with and without the layer, paired by day (bootstrap over days)."""
    ok = (p.bsp > 1) & (p.pred_bsp > 1)
    d = p[ok]
    sb = (np.log(d.pred_bsp) - np.log(d.bsp)) ** 2
    se = (np.log(d.pred_bsp_m) - np.log(d.bsp)) ** 2
    w = d.won.astype(float)
    ll = lambda q: -(w * np.log(np.clip(q, 1e-12, 1)) + (1 - w) * np.log(np.clip(1 - q, 1e-12, 1)))  # noqa: E731
    day = pd.DataFrame({"d": d.race_date.astype(str), "e": se, "b": sb, "lle": ll(d.p_model_m), "llb": ll(d.p_model)}) \
        .groupby("d").agg(e=("e", "sum"), b=("b", "sum"), lle=("lle", "sum"), llb=("llb", "sum"), n=("e", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(day), (n_boot, len(day)))
    E, B, LE, LB, N = (day[c].to_numpy()[idx].sum(1) for c in ("e", "b", "lle", "llb", "n"))
    dr = np.sqrt(E / N) - np.sqrt(B / N)
    dl = (LE - LB) / N
    rb, re = float(np.sqrt(day.b.sum() / day.n.sum())), float(np.sqrt(day.e.sum() / day.n.sum()))
    lb, le = float(day.llb.sum() / day.n.sum()), float(day.lle.sum() / day.n.sum())
    return {"runners": int(day.n.sum()), "days": int(len(day)), "with_a_memory": round(float((d.n_before > 0).mean()), 4),
            "rmse": round(rb, 4), "rmse_layer": round(re, 4), "rmse_diff": round(re - rb, 4),
            "rmse_z": round((re - rb) / float(dr.std()), 1) if dr.std() > 0 else None,
            "logloss": round(lb, 5), "logloss_layer": round(le, 5), "logloss_diff": round(le - lb, 5),
            "logloss_z": round((le - lb) / float(dl.std()), 1) if dl.std() > 0 else None}


def run(pred: pd.DataFrame, split: str) -> dict:
    p = misses(pred)
    first = p[p.race_date.astype(str) < split]
    later = p[p.race_date.astype(str) >= split]
    prm = fit(first)
    out = {"split": split, "from": str(p.race_date.min()), "to": str(p.race_date.max()), "fit": prm,
           "scored": score(apply(later, prm))}
    after = later[later.race_date.astype(str) > "2026-11-05"]
    if len(after):
        out["scored_after_5_nov"] = score(apply(after, prm))
    h = later.merge(p[["race_id", "dog_id"]], on=["race_id", "dog_id"])
    a = apply(h, prm)
    for lab, mask in (("no miss behind it", a.n_before == 0), ("1 to 4", a.n_before.between(1, 4)), ("5 or more", a.n_before >= 5)):
        if mask.sum() >= 500:
            out.setdefault("by_memory", {})[lab] = score(a[mask])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="2026-09-01")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = run(pd.read_parquet(a.pred), a.split)
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
