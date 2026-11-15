"""The probability of backtest overfitting (Bailey, Borwein, Lopez de Prado and Zhu 2017, combinatorially symmetric
cross-validation) across the fast loop's saved arm predictions (the owner, 15 Nov 2026; ledger greyhound-pbo-cscv-1115).

Each arm's ``pred_<arm>.parquet`` (race_id, dog_id, race_date, bsp, won, pred_bsp, p_model). On the runners every arm
in the family shares, within the window, a day's score per arm (minus the RMSE of log(BSP) that day, or minus the win
log-loss). The days are cut into ``S`` contiguous blocks; for every way of taking S/2 blocks in-sample, the arm best
in-sample and its relative rank out-of-sample (0 worst, 1 best); its logit; PBO = the share of splits where that rank
is at or below the median. Also the chosen arm's out-of-sample score against its in-sample score (degradation), and how
often it loses to the out-of-sample median arm.

    python -m research.greyhound.pbo --root <folder with the batch folders> --out out/pbo.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

KEY = ["race_id", "dog_id"]


def load(root: str, start: str, end: str, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Every arm's saved predictions under ``root`` (the first file seen for an arm name), cut to the window."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "pred_*.parquet"), recursive=True)):
        name = os.path.basename(p)[5:-8]
        if name in out or (names and name not in names):
            continue
        d = pd.read_parquet(p, columns=KEY + ["race_date", "bsp", "won", "pred_bsp", "p_model"])
        d = d[(d.race_date.astype(str) >= start) & (d.race_date.astype(str) <= end)]
        if len(d):
            out[name] = d
    return out


def day_scores(arms: dict[str, pd.DataFrame], min_cover: float = 0.97) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Per day and arm: minus the RMSE of log(BSP) and minus the mean win log-loss, on the runners every kept arm
    shares. An arm covering less than ``min_cover`` of the most common runner set is left out (and named)."""
    sizes = {k: len(v) for k, v in arms.items()}
    top = max(sizes.values())
    dropped = [k for k, s in sizes.items() if s < min_cover * top]
    keep = [k for k in arms if k not in dropped]
    common = None
    for k in keep:
        s = arms[k][KEY].drop_duplicates()
        common = s if common is None else common.merge(s, on=KEY)
    base = arms[keep[0]].merge(common, on=KEY)[KEY + ["race_date", "bsp", "won"]]
    base = base[base.bsp > 1]
    rm, ll = {}, {}
    for k in keep:
        m = base.merge(arms[k][KEY + ["pred_bsp", "p_model"]], on=KEY)
        m = m[m.pred_bsp > 1]
        se = (np.log(m.pred_bsp) - np.log(m.bsp)) ** 2
        p = m.p_model.clip(1e-12, 1 - 1e-12)
        l = -(m.won * np.log(p) + (1 - m.won) * np.log(1 - p))
        g = pd.DataFrame({"d": m.race_date.astype(str), "se": se, "ll": l}).groupby("d")
        rm[k] = -np.sqrt(g.se.mean())
        ll[k] = -g.ll.mean()
    return pd.DataFrame(rm).dropna(), pd.DataFrame(ll).dropna(), dropped


def cscv(M: pd.DataFrame, S: int = 16) -> dict:
    """PBO by combinatorially symmetric cross-validation on ``M`` (rows days in order, columns arms, higher better)."""
    X = M.sort_index().to_numpy()
    T, N = X.shape
    blocks = np.array_split(np.arange(T), S)
    logits, degr, loss = [], [], []
    for ins in itertools.combinations(range(S), S // 2):
        i_rows = np.concatenate([blocks[b] for b in ins])
        o_rows = np.concatenate([blocks[b] for b in range(S) if b not in ins])
        is_perf, oos_perf = X[i_rows].mean(0), X[o_rows].mean(0)
        best = int(np.argmax(is_perf))
        rank = (oos_perf < oos_perf[best]).sum() / (N - 1)          # 0 worst, 1 best
        w = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
        degr.append((is_perf[best], oos_perf[best]))
        loss.append(oos_perf[best] < np.median(oos_perf))
    logits = np.array(logits)
    d = np.array(degr)
    slope = float(np.polyfit(d[:, 0], d[:, 1], 1)[0]) if len(d) > 2 else float("nan")
    return {"arms": N, "days": T, "blocks": S, "splits": len(logits), "pbo": round(float((logits <= 0).mean()), 4),
            "logit_median": round(float(np.median(logits)), 3),
            "chosen_oos_below_median_share": round(float(np.mean(loss)), 4),
            "degradation_slope": round(slope, 3),
            "chosen_is_mean": round(float(d[:, 0].mean()), 5), "chosen_oos_mean": round(float(d[:, 1].mean()), 5)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-11-05")
    ap.add_argument("--family", default=None, help="comma-separated arm names (default every arm)")
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--out", default="out/pbo.json")
    a = ap.parse_args(argv)
    names = [x.strip() for x in a.family.split(",")] if a.family else None
    arms = load(a.root, a.start, a.end, names)
    rm, ll, dropped = day_scores(arms)
    out = {"window": [a.start, a.end], "dropped_low_cover": dropped, "arms_used": list(rm.columns),
           "rmse": cscv(rm, a.blocks), "logloss": cscv(ll, a.blocks),
           "best_full_window_rmse": rm.mean().sort_values(ascending=False).head(5).round(5).to_dict()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
