"""The model's final layer updated each day on the days before it (the owner, 15 Nov 2026: online learning; ledger
greyhound-online-recal-1115), on a fast-loop arm's saved predictions (race_id, dog_id, race_date, track, bsp, won,
pred_bsp, p_model).

Two layers, each fitted on rows of earlier days only and applied to one day:
- **price**: log(BSP) = a + b log(pred_bsp) + a per-track intercept (the track's mean residual shrunk by n / (n + K_TRACK));
- **win**: p proportional to p_model ^ tau within the race, tau by the races' log-likelihood (a grid).

Three versions scored on the same days: ``raw`` (the arm as saved), ``static`` (the layer fitted once on the warm-up
month and frozen) and ``online_K`` (refitted each day on the K days before it). Paired by day against ``static``, 2,000
bootstrap draws, by period.

    python -m research.greyhound.online_recal --pred <pred_arm.parquet> --out out/online.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

K_TRACK = 200.0
TAUS = np.round(np.arange(0.70, 1.41, 0.02), 2)
WARMUP_END = "2026-06-30"
PERIODS = (("Jul-Aug", "2026-07-01", "2026-08-31"), ("Sep-11Nov", "2026-09-01", "2026-11-11"),
           ("all", "2026-07-01", "2026-11-11"))


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = d[(d.bsp > 1) & (d.pred_bsp > 1) & d.p_model.between(1e-9, 1 - 1e-9)].copy()
    d["day"] = d.race_date.astype(str)
    d["y"], d["x"] = np.log(d.bsp.astype(float)), np.log(d.pred_bsp.astype(float))
    d["lp"] = np.log(d.p_model.astype(float))
    return d.sort_values(["day", "race_id"]).reset_index(drop=True)


def fit_price(tr: pd.DataFrame) -> dict:
    b, a = np.polyfit(tr.x, tr.y, 1)
    r = tr.y - (a + b * tr.x)
    g = r.groupby(tr.track)
    shift = (g.sum() / (g.size() + K_TRACK)).to_dict()
    return {"a": a, "b": b, "shift": shift}


def apply_price(m: dict, d: pd.DataFrame) -> np.ndarray:
    return m["a"] + m["b"] * d.x.to_numpy() + d.track.map(m["shift"]).fillna(0.0).to_numpy()


def _race_ll(d: pd.DataFrame, tau: float) -> np.ndarray:
    z = tau * d.lp.to_numpy()
    rid = d.race_id.to_numpy()
    s = pd.Series(np.exp(z - pd.Series(z).groupby(rid).transform("max").to_numpy())).groupby(rid).transform("sum").to_numpy()
    zmax = pd.Series(z).groupby(rid).transform("max").to_numpy()
    p = np.exp(z - zmax) / s
    w = d.won.to_numpy(float)
    return -(w * np.log(np.clip(p, 1e-12, 1)) + (1 - w) * np.log(np.clip(1 - p, 1e-12, 1)))


def fit_tau(tr: pd.DataFrame) -> float:
    return float(TAUS[int(np.argmin([_race_ll(tr, t).mean() for t in TAUS]))])


def run(d: pd.DataFrame, ks=(14, 28, 56)) -> pd.DataFrame:
    """Per runner of the test days: squared log error and log-loss under raw, static and online_K."""
    days = sorted(d.day.unique())
    warm = d[d.day <= WARMUP_END]
    sp, st = fit_price(warm), fit_tau(warm)
    out = []
    for day in [x for x in days if x > WARMUP_END]:
        t = d[d.day == day]
        row = pd.DataFrame({"day": day, "race_id": t.race_id.to_numpy()})
        row["se_raw"] = (t.x - t.y).to_numpy() ** 2
        row["ll_raw"] = _race_ll(t, 1.0) if False else -(t.won * np.log(t.p_model) + (1 - t.won) * np.log(1 - t.p_model)).to_numpy()
        row["se_static"] = (apply_price(sp, t) - t.y.to_numpy()) ** 2
        row["ll_static"] = _race_ll(t, st)
        for k in ks:
            lo = (pd.Timestamp(day) - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            tr = d[(d.day >= lo) & (d.day < day)]
            row[f"se_online_{k}"] = (apply_price(fit_price(tr), t) - t.y.to_numpy()) ** 2
            row[f"ll_online_{k}"] = _race_ll(t, fit_tau(tr))
        out.append(row)
    return pd.concat(out, ignore_index=True)


def paired(r: pd.DataFrame, a: str, b: str, start: str, end: str, draws: int = 2000, seed: int = 0) -> dict:
    x = r[(r.day >= start) & (r.day <= end)]
    g = x.groupby("day").agg(sa=(f"se_{a}", "sum"), sb=(f"se_{b}", "sum"), la=(f"ll_{a}", "sum"), lb=(f"ll_{b}", "sum"),
                             n=(f"se_{a}", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (draws, len(g)))
    SA, SB, LA, LB, N = (g[c].to_numpy()[idx].sum(1) for c in ("sa", "sb", "la", "lb", "n"))
    dr = np.sqrt(SA / N) - np.sqrt(SB / N)
    dl = (LA - LB) / N
    r_a, r_b = np.sqrt(g.sa.sum() / g.n.sum()), np.sqrt(g.sb.sum() / g.n.sum())
    ll = (g.la.sum() - g.lb.sum()) / g.n.sum()
    return {"days": int(len(g)), "runners": int(g.n.sum()), "rmse": round(float(r_a), 4), "rmse_vs": round(float(r_b), 4),
            "diff": round(float(r_a - r_b), 4), "z": round(float((r_a - r_b) / dr.std()), 1) if dr.std() > 0 else None,
            "ll_diff": round(float(ll), 5), "ll_z": round(float(ll / dl.std()), 1) if dl.std() > 0 else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="out/online.json")
    a = ap.parse_args(argv)
    d = prepare(pd.read_parquet(a.pred))
    r = run(d)
    res = {}
    for arm in ["raw", "online_14", "online_28", "online_56"]:
        res[arm] = {p: paired(r, arm, "static", s, e) for p, s, e in PERIODS}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
