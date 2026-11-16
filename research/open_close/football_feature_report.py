"""What the football move model's features are worth, measured on Betfair's own moves; and two checks of the target.

racing² football research, 10 Nov 2026 (the owner: "show me the feature set we have and feature importance";
the review's step 0). Read-only over the lab matrix. The arm is the expected-value model's
(``team_all+home_adv+team_news+form_short+x_short``), refitted on the same-book moves of 2013/14-2024/25 so that
Betfair's 2025/26 moves are out of sample (2025/26 was read by earlier rules: development evidence, not a holdout).

1. **Gain share** by feature and by family (LightGBM, home and away fits).
2. **Permutation importance on Betfair 2025/26:** each family's columns shuffled together within the league, the
   stack's (ridge + LightGBM, the mean) loss of skill against "no move" on Betfair's home and away moves, overall and
   by league tier; and the same for the top single features by gain.
3. **Leave a family out:** the stack refitted without each family, its skill on the same rows.
4. **Calibration:** the forecast's tenths against the mean move that followed, and the slope.
5. **The target (review A4):** the margin removed three ways (proportional, power, Shin) from the same book's early and
   closing 1X2; under each, the mean change in chance by early-chance band. A fair price is a martingale, so a band
   whose chance drifts on average is the margin's removal showing, not a move.
6. **The target's scale (review A5):** the spread of the move by early-chance band in log-chance and in logit units.

    python -m research.open_close.football_feature_report --cache <lab cache> --out-json out/football_feature_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = "team_all+home_adv+team_news+form_short+x_short"
TRAIN = (2013, 2024)
BANDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
PREFIXES = (("fs_", "short form (fs_)"), ("d_", "team windows (d_)"), ("x_", "xG (x_)"), ("tn_", "team news (tn_)"),
            ("ha_", "home advantage (ha_)"))


def family_of(col: str) -> str:
    for p, name in PREFIXES:
        if col.startswith(p):
            return name
    return col


# ---------------------------------------------------------------------------
# the margin, removed three ways
# ---------------------------------------------------------------------------

def devig_proportional(prices: np.ndarray) -> np.ndarray:
    inv = 1.0 / prices
    return inv / inv.sum(1, keepdims=True)


def devig_power(prices: np.ndarray) -> np.ndarray:
    """q_i = (1/price_i)^k, k per row such that the chances sum to one (bisection; k > 1 for an overround)."""
    inv = 1.0 / prices
    lo, hi = np.full(len(inv), 0.5), np.full(len(inv), 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        over = (inv ** k[:, None]).sum(1) > 1
        lo, hi = np.where(over, k, lo), np.where(over, hi, k)
    q = inv ** ((lo + hi) / 2)[:, None]
    return q / q.sum(1, keepdims=True)


def devig_shin(prices: np.ndarray) -> np.ndarray:
    """Shin (1993): q_i = (sqrt(z^2 + 4 (1 - z) pi_i^2 / B) - z) / (2 (1 - z)), B the booksum, z per row such that the
    chances sum to one (bisection). An underround row falls back to the proportional split."""
    inv = 1.0 / prices
    B = inv.sum(1)

    def q(z):
        z = z[:, None]
        return (np.sqrt(z ** 2 + 4 * (1 - z) * inv ** 2 / B[:, None]) - z) / (2 * (1 - z))
    lo, hi = np.zeros(len(inv)), np.full(len(inv), 0.6)
    for _ in range(60):
        z = (lo + hi) / 2
        over = q(z).sum(1) > 1
        lo, hi = np.where(over, z, lo), np.where(over, hi, z)
    out = q((lo + hi) / 2)
    out = out / out.sum(1, keepdims=True)
    return np.where((B <= 1)[:, None], devig_proportional(prices), out)


DEVIG = {"proportional": devig_proportional, "power": devig_power, "shin": devig_shin}


def band_table(qo: np.ndarray, x: np.ndarray) -> dict:
    """Mean and standard error of ``x`` by early-chance band."""
    out = {}
    b = np.digitize(qo, BANDS[1:-1])
    for i in range(len(BANDS) - 1):
        v = x[(b == i) & np.isfinite(x)]
        if len(v) >= 30:
            out[f"{BANDS[i]:.1f}-{BANDS[i + 1]:.1f}"] = {"n": int(len(v)), "mean": float(v.mean()),
                                                         "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                                                         "sd": float(v.std(ddof=1))}
    return out


def target_checks(d: pd.DataFrame) -> dict:
    """The same-book 1X2 rows' early and closing prices under each way of removing the margin."""
    need = ["open_h", "open_d", "open_a", "close_h", "close_d", "close_a"]
    if not set(need) <= set(d.columns):
        return {"status": "no data found: the matrix holds no raw early and closing 1X2 prices"}
    m = d[d["same_book"] & d[need].gt(1).all(axis=1)]
    if len(m) < 1000:
        return {"status": f"no data found: {len(m)} same-book rows with raw prices"}
    po, pc = m[need[:3]].to_numpy(float), m[need[3:]].to_numpy(float)
    res = {"rows": int(len(m)), "overround_early": float((1 / po).sum(1).mean()),
           "overround_close": float((1 / pc).sum(1).mean()), "drift_of_chance_by_early_band": {}, "bias_score": {}}
    for name, f in DEVIG.items():
        qo, qc = f(po), f(pc)
        t = band_table(qo.ravel(), (qc - qo).ravel())
        res["drift_of_chance_by_early_band"][name] = t
        # one number a method: the root mean square of the bands' mean drift, weighted by their rows
        n = np.array([v["n"] for v in t.values()], float)
        mu = np.array([v["mean"] for v in t.values()])
        res["bias_score"][name] = float(np.sqrt((n * mu ** 2).sum() / n.sum())) if len(n) else None
    qo, qc = devig_proportional(po), devig_proportional(pc)
    logit = lambda q: np.log(q / (1 - q))        # noqa: E731
    res["spread_of_move_by_early_band"] = {
        "log_chance": {k: round(v["sd"], 5) for k, v in band_table(qo.ravel(), np.log(qc / qo).ravel()).items()},
        "logit": {k: round(v["sd"], 5) for k, v in band_table(qo.ravel(), (logit(qc) - logit(qo)).ravel()).items()}}
    return res


# ---------------------------------------------------------------------------
# the fitted stack, as a predictor that can be asked again on shuffled rows
# ---------------------------------------------------------------------------

class Stack:
    """The lab's stack (football.lab: ridge on the standardised columns + LightGBM with the league, the mean)."""

    def __init__(self, tr: pd.DataFrame, cols: list[str], y: np.ndarray):
        import lightgbm as lgb
        from football import lab
        self.tr, self.cols = tr, cols
        Z, _ = lab._z(tr, tr.iloc[:1], cols)
        self.ybar = float(y.mean())
        self.beta = np.linalg.solve(Z.T @ Z + lab.RIDGE_ALPHA * np.eye(Z.shape[1]), Z.T @ (y - self.ybar))
        self.use = cols + ["league"]
        self.g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
                                   importance_type="gain")
        self.g.fit(tr[self.use], y)

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        from football import lab
        _, Zt = lab._z(self.tr, te, self.cols)
        return (self.ybar + Zt @ self.beta + self.g.predict(te[self.use])) / 2

    def gain(self) -> pd.Series:
        s = pd.Series(self.g.feature_importances_, index=self.use)
        return s / s.sum()


def skill(y: np.ndarray, p: np.ndarray) -> float:
    """Skill against 'no move': 1 - SSE(forecast) / SSE(zero)."""
    return 1 - float(np.sum((y - p) ** 2)) / float(np.sum(y ** 2))


def shuffled(te: pd.DataFrame, cols: list[str], rng) -> pd.DataFrame:
    """``cols`` shuffled together (one permutation of the rows) within each league."""
    out = te.copy()
    idx = np.arange(len(te))
    for _, pos in pd.Series(idx).groupby(te["league"].astype(str).to_numpy()):
        p = pos.to_numpy()
        idx[p] = rng.permutation(p)
    num = [c for c in cols if c != "league"]
    if num:
        out[num] = te[num].to_numpy()[idx]
    if "league" in cols:                                   # the league itself: shuffled across the whole test set
        out["league"] = pd.Categorical(te["league"].to_numpy()[rng.permutation(len(te))],
                                       categories=te["league"].cat.categories)
    return out


def permutation(models: dict, te: pd.DataFrame, ys: dict, groups: dict, tiers: np.ndarray, repeats: int = 5,
                seed: int = 0) -> dict:
    """Each group's loss of skill on the test rows when its columns are shuffled (mean over sides and repeats), overall
    and by tier."""
    rng = np.random.default_rng(seed)
    base = {s: (ys[s] - models[s].predict(te)) ** 2 for s in models}
    zero = {s: ys[s] ** 2 for s in models}

    def drop(se_new, mask):
        return float(np.mean([(se_new[s][mask].sum() - base[s][mask].sum()) / zero[s][mask].sum() for s in models]))
    out = {}
    for name, cols in groups.items():
        runs = []
        for _ in range(repeats):
            sh = shuffled(te, cols, rng)
            runs.append({s: (ys[s] - models[s].predict(sh)) ** 2 for s in models})
        allm = np.ones(len(te), bool)
        r = {"columns": len(cols), "skill_lost": float(np.mean([drop(x, allm) for x in runs])),
             "sd": float(np.std([drop(x, allm) for x in runs]))}
        for t in np.unique(tiers):
            r[f"skill_lost[{t}]"] = float(np.mean([drop(x, tiers == t) for x in runs]))
        out[name] = r
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["skill_lost"]))


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    t = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").agg(forecast=("p", "mean"), move=("y", "mean"),
                                                             n=("y", "size"))
    b = float(np.cov(p, y)[0, 1] / np.var(p, ddof=1))
    return {"slope": b, "intercept": float(y.mean() - b * p.mean()), "corr": float(np.corrcoef(p, y)[0, 1]),
            "tenths": [{"forecast": round(float(r.forecast), 5), "move": round(float(r.move), 5), "n": int(r.n)}
                       for r in t.itertuples()]}


def run(cache: str, spec: str = SPEC, top: int = 20) -> dict:
    from research.open_close import football_improve as fi
    from research.open_close.football_model_diagnostics import tier
    from research.open_close.football_stack_betfair_read import betfair_moves, betfair_rows
    d, cols = fi.load(cache, spec)
    tr = d[d["same_book"] & d["season"].between(*TRAIN)]
    te = d[d["test_season"] == 2025]
    te = te[betfair_rows(te)]
    mv = betfair_moves(te)
    ok = np.isfinite(mv["y_h"]) & np.isfinite(mv["y_a"])
    te, mv = te[ok], mv[ok]
    res = {"spec": spec, "features": len(cols), "train_rows": int(len(tr)), "train_seasons": list(TRAIN),
           "test": "Betfair's own early-to-close moves, 2025/26 to date (development evidence)",
           "test_rows": int(len(te)), "target": target_checks(d)}
    if len(te) < 500 or len(tr) < 5000:
        res["status"] = "no data found: too few rows to fit or to score"
        return res
    ys = {"h": mv["y_h"].to_numpy(float), "a": mv["y_a"].to_numpy(float)}
    models = {s: Stack(tr, cols, tr[f"y_{s}"].to_numpy(float)) for s in ("h", "a")}
    pred = {s: models[s].predict(te) for s in models}
    res["skill_vs_no_move"] = {s: skill(ys[s], pred[s]) for s in models}
    res["calibration"] = {s: calibration(ys[s], pred[s]) for s in models}
    gain = pd.concat([models[s].gain() for s in models], axis=1).mean(axis=1).sort_values(ascending=False)
    res["gain_share"] = {k: round(float(v), 4) for k, v in gain.items()}
    fam = gain.groupby([family_of(c) for c in gain.index]).sum().sort_values(ascending=False)
    res["gain_share_by_family"] = {k: round(float(v), 4) for k, v in fam.items()}
    tiers = tier(te["league"].astype(str))
    groups: dict = {}
    for c in cols + ["league"]:
        groups.setdefault(family_of(c), []).append(c)
    res["permutation_by_family"] = permutation(models, te, ys, groups, tiers)
    res["permutation_top_features"] = permutation(models, te, ys, {c: [c] for c in gain.index[:top]}, tiers, repeats=3)
    # leave a family out: refitted
    left = {}
    for name, g in groups.items():
        if name == "league":
            continue
        keep = [c for c in cols if c not in g]
        m2 = {s: Stack(tr, keep, tr[f"y_{s}"].to_numpy(float)) for s in ("h", "a")}
        sk = {s: skill(ys[s], m2[s].predict(te)) for s in m2}
        left[name] = {"skill_h": sk["h"], "skill_a": sk["a"],
                      "change": float(np.mean([sk[s] - res["skill_vs_no_move"][s] for s in sk]))}
    res["leave_family_out"] = dict(sorted(left.items(), key=lambda kv: kv[1]["change"]))
    res["tier_rows"] = {str(k): int(v) for k, v in pd.Series(tiers).value_counts().items()}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    res = run(a.cache, a.spec)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    brief = {k: res.get(k) for k in ("features", "train_rows", "test_rows", "skill_vs_no_move", "gain_share_by_family",
                                     "permutation_by_family", "leave_family_out")}
    brief["bias_score"] = (res.get("target") or {}).get("bias_score")
    print(json.dumps(brief, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
