"""Does a family of team features add to the market? Two tests, walk-forward by season, a holdout read once.

1. **The outcome test (stage 2).** A softmax over home, draw and away whose utilities are a weight on the log of the
   de-vigged opening price, a weight on the log of the Dixon-Coles price, outcome intercepts and, for a family, a
   weight on each of its home-minus-away features (+ for home, - for away, none for the draw; standardised on the
   training rows, missing set to the training mean). Fitted by maximum likelihood on every season before the scored
   one. A family adds if its log-loss is below the open-plus-model arm's, the 90% interval of the gain (bootstrap
   by match day, paired) above zero.
2. **The move test.** Least squares of the market's move from the open to the close, in implied log supremacy
   (log(home rate / away rate) at the close less at the open), on the family's features; out-of-sample R2 against
   no change and against the training mean, beside a floor from the same fit on permuted training moves. Only matches
   whose open and close come from the same book (Pinnacle to Pinnacle, or Betfair to Betfair), so the move is not a
   change of venue.

Development: each season from ``DEV_FROM`` to ``DEV_TO`` scored by a fit on the seasons before it (from
``TRAIN_FROM``). Holdout: the seasons from ``HOLDOUT_FROM``, scored once by a fit on every development season. Rows: both
sides with at least ``MIN_PREV`` earlier matches, every baseline input present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRAIN_FROM, DEV_FROM, DEV_TO, HOLDOUT_FROM = 2013, 2016, 2024, 2025
MIN_PREV = 5
WINS = ("l5", "l10", "ewm", "season")

FAMILIES = {
    "strength_market": ["r_diff", "d_own_r_d5"],
    "schedule": [f"d_opp_r_{w}" for w in WINS],
    "form_raw": [f"d_{c}_{w}" for c in ("pts", "gd", "sotd") for w in WINS],
    "added_vs_model": [f"d_{c}_{w}" for c in ("pae_dc", "gae_dc") for w in WINS],
    "added_vs_market": [f"d_{c}_{w}" for c in ("pae_mkt", "gae_mkt") for w in WINS],
    "disagreement": ["dis_open"],
}
FAMILIES["all"] = sorted({c for v in FAMILIES.values() for c in v})


def prepare(f: pd.DataFrame) -> pd.DataFrame:
    d = f.copy()
    d["r_diff"] = d["r_h"] - d["r_a"]
    gd = d["ft_home"] - d["ft_away"]
    d["y"] = np.select([gd > 0, gd == 0], [0, 1], 2)                 # 0 home, 1 draw, 2 away
    for k, c in (("h", "p_h"), ("d", "p_d"), ("a", "p_a")):
        d[f"lq_{k}"] = np.log(np.clip(d[f"qo_{k}"], 1e-6, 1))
        d[f"lp_{k}"] = np.log(np.clip(d[c], 1e-6, 1))
    base_ok = d[["lq_h", "lq_d", "lq_a", "lp_h", "lp_d", "lp_a"]].notna().all(axis=1)
    d = d[base_ok & (d["n_prev_min"] >= MIN_PREV)].copy()
    d["move"] = np.log(d["mclam_h"] / d["mclam_a"]) - np.log(d["molam_h"] / d["molam_a"])
    src_o, src_c = d.get("open_1x2_src", pd.Series("", index=d.index)).astype(str), \
        d.get("close_1x2_src", pd.Series("", index=d.index)).astype(str)
    d["same_book"] = ((src_o == "PS") & (src_c == "PSC")) | ((src_o == "BFE") & (src_c == "BFEC"))
    return d


class Stage2:
    """The softmax of the outcome test. ``use_model``: the Dixon-Coles term; ``feats``: the family's columns."""

    def __init__(self, feats: list[str] | None = None, use_model: bool = True):
        self.feats, self.use_model = list(feats or []), use_model

    def _x(self, d: pd.DataFrame) -> np.ndarray:
        if not self.feats:
            return np.zeros((len(d), 0))
        z = (d[self.feats].to_numpy(float) - self.mu) / self.sd
        return np.nan_to_num(z, nan=0.0)

    def _probs(self, theta, lq, lp, z):
        a, b = theta[0], theta[1] if self.use_model else 0.0
        ah, aa = theta[2], theta[3]
        c = theta[4:]
        s = z @ c if len(c) else 0.0
        u = np.c_[ah + a * lq[:, 0] + b * lp[:, 0] + s, a * lq[:, 1] + b * lp[:, 1], aa + a * lq[:, 2] + b * lp[:, 2] - s]
        u -= u.max(axis=1, keepdims=True)
        e = np.exp(u)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, d: pd.DataFrame) -> "Stage2":
        if self.feats:
            v = d[self.feats].to_numpy(float)
            self.mu, self.sd = np.nanmean(v, axis=0), np.nanstd(v, axis=0)
            self.sd[~np.isfinite(self.sd) | (self.sd == 0)] = 1.0
            self.mu[~np.isfinite(self.mu)] = 0.0
        lq, lp = d[["lq_h", "lq_d", "lq_a"]].to_numpy(), d[["lp_h", "lp_d", "lp_a"]].to_numpy()
        z, y = self._x(d), d["y"].to_numpy()

        def nll(th):
            p = self._probs(th, lq, lp, z)
            return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))) + 1e-4 * np.sum(th[4:] ** 2)
        th0 = np.r_[1.0, 0.0, 0.0, 0.0, np.zeros(z.shape[1])]
        self.theta = minimize(nll, th0, method="L-BFGS-B").x
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self._probs(self.theta, d[["lq_h", "lq_d", "lq_a"]].to_numpy(), d[["lp_h", "lp_d", "lp_a"]].to_numpy(),
                           self._x(d))

    def coefs(self) -> dict:
        out = {"w_open": float(self.theta[0])}
        if self.use_model:
            out["w_model"] = float(self.theta[1])
        out.update({f: float(c) for f, c in zip(self.feats, self.theta[4:])})
        return out


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def boot_gain(base_ll: np.ndarray, arm_ll: np.ndarray, day: np.ndarray, n: int = 1000, seed: int = 0) -> tuple:
    """The mean log-loss gain (base - arm) and its 90% interval, resampling match days."""
    df = pd.DataFrame({"g": base_ll - arm_ll, "day": day}).groupby("day")["g"].agg(["sum", "size"])
    s, c = df["sum"].to_numpy(), df["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    draws = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))


def outcome_test(d: pd.DataFrame, families: dict = FAMILIES, scored: str = "dev") -> dict:
    """Every arm's log-loss on the scored seasons (dev: walk-forward; holdout: once), gains against open + model."""
    if scored == "dev":
        plan = [(s, d[(d["season"] >= TRAIN_FROM) & (d["season"] < s)], d[d["season"] == s])
                for s in range(DEV_FROM, DEV_TO + 1)]
    else:
        plan = [("holdout", d[(d["season"] >= TRAIN_FROM) & (d["season"] <= DEV_TO)], d[d["season"] >= HOLDOUT_FROM])]
    arms = {"open": Stage2(use_model=False), "open+model": Stage2()}
    arms.update({k: Stage2(v) for k, v in families.items()})
    lls = {k: [] for k in arms}
    test_rows = []
    for _s, tr, te in plan:
        if len(tr) < 1000 or te.empty:
            continue
        test_rows.append(te[["date"]])
        for k, arm in arms.items():
            mod = type(arm)(arm.feats, arm.use_model).fit(tr)
            lls[k].append(_ll(mod.predict(te), te["y"].to_numpy()))
    day = pd.concat(test_rows)["date"].dt.strftime("%Y-%m-%d").to_numpy()
    ll = {k: np.concatenate(v) for k, v in lls.items()}
    out = {"rows": int(len(day)), "log_loss": {k: float(v.mean()) for k, v in ll.items()}, "gain_vs_open_model": {},
           "gain_vs_open": {}}
    for k in ll:
        if k not in ("open", "open+model"):
            out["gain_vs_open_model"][k] = boot_gain(ll["open+model"], ll[k], day)
        if k != "open":
            out["gain_vs_open"][k] = boot_gain(ll["open"], ll[k], day)
    return out


def coefficients(d: pd.DataFrame, families: dict = FAMILIES, n_boot: int = 100, seed: int = 0) -> dict:
    """Each family's weights fitted on every development season, with 90% intervals (bootstrap by match day)."""
    dev = d[(d["season"] >= TRAIN_FROM) & (d["season"] <= DEV_TO)]
    days = dev["date"].dt.strftime("%Y-%m-%d")
    uniq = days.unique()
    rng = np.random.default_rng(seed)
    out = {}
    for k, feats in {"open+model": [], **families}.items():
        point = Stage2(feats).fit(dev).coefs()
        draws = []
        for _ in range(n_boot):
            pick = pd.Series(rng.choice(uniq, size=len(uniq), replace=True))
            rows = dev.set_index(days).loc[pick].reset_index(drop=True)
            draws.append(Stage2(feats).fit(rows).coefs())
        dr = pd.DataFrame(draws)
        out[k] = {c: {"coef": point[c], "lo90": float(dr[c].quantile(0.05)), "hi90": float(dr[c].quantile(0.95))}
                  for c in point}
    return out


def move_test(d: pd.DataFrame, families: dict = FAMILIES, scored: str = "dev", seed: int = 0) -> dict:
    """Out-of-sample R2 of the open-to-close move on each family, against no change and the training mean, with the
    permuted-training floor."""
    m = d[d["same_book"] & d["move"].notna()]
    if scored == "dev":
        plan = [(m[(m["season"] >= TRAIN_FROM) & (m["season"] < s)], m[m["season"] == s]) for s in range(DEV_FROM, DEV_TO + 1)]
    else:
        plan = [(m[(m["season"] >= TRAIN_FROM) & (m["season"] <= DEV_TO)], m[m["season"] >= HOLDOUT_FROM])]
    rng = np.random.default_rng(seed)
    res = {}
    for k, feats in families.items():
        ys, ps, pm, pp = [], [], [], []
        for tr, te in plan:
            if len(tr) < 500 or te.empty:
                continue
            X = tr[feats].to_numpy(float)
            mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
            sd[~np.isfinite(sd) | (sd == 0)] = 1.0
            mu[~np.isfinite(mu)] = 0.0
            Z = np.c_[np.ones(len(tr)), np.nan_to_num((X - mu) / sd)]
            Zt = np.c_[np.ones(len(te)), np.nan_to_num((te[feats].to_numpy(float) - mu) / sd)]
            y = tr["move"].to_numpy(float)
            beta = np.linalg.lstsq(Z, y, rcond=None)[0]
            bperm = np.linalg.lstsq(Z, rng.permutation(y), rcond=None)[0]
            ys.append(te["move"].to_numpy(float))
            ps.append(Zt @ beta)
            pp.append(Zt @ bperm)
            pm.append(np.full(len(te), y.mean()))
        if not ys:
            continue
        y, p, q, mbar = (np.concatenate(x) for x in (ys, ps, pp, pm))
        sse = lambda f: float(np.sum((y - f) ** 2))                     # noqa: E731
        res[k] = {"rows": int(len(y)), "r2_vs_no_change": 1 - sse(p) / sse(np.zeros_like(y)),
                  "r2_vs_mean": 1 - sse(p) / sse(mbar), "r2_permuted_vs_mean": 1 - sse(q) / sse(mbar),
                  "sign_right": float(np.mean(np.sign(p - mbar) == np.sign(y - mbar)))}
    return res


def _json(o) -> str:
    import json

    def default(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x) if np.isfinite(x) else None
        return str(x)
    return json.dumps(o, indent=1, default=default)


def main(argv=None) -> int:
    import argparse
    import logging
    from pathlib import Path

    from football.data import load_gold
    from football.model import walk_forward
    from football.team_features import build
    from sources.common import Store
    ap = argparse.ArgumentParser(description="Team features against the market: the outcome and move tests")
    ap.add_argument("--root", default=None, help="a local sources folder in place of S3 (the gold table)")
    ap.add_argument("--dc-cache", default=None, help="parquet of the Dixon-Coles walk-forward prices (kept)")
    ap.add_argument("--features-out", default=None, help="parquet of the match features (kept)")
    ap.add_argument("--holdout", action="store_true", help="also score the holdout seasons, once")
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    gold = load_gold(Store(root=a.root))
    if a.dc_cache and Path(a.dc_cache).exists():
        dc = pd.read_parquet(a.dc_cache)
    else:
        dc = walk_forward(gold, "2012-07-01")[["match_id", "lam_h", "lam_a", "p_h", "p_d", "p_a", "p_o25"]]
        if a.dc_cache:
            dc.to_parquet(a.dc_cache, index=False)
    f, t = build(gold, dc)
    if a.features_out:
        f.to_parquet(a.features_out, index=False)
    d = prepare(f)
    res = {"rows": {"gold_finished": int((gold["status"] == "FINISHED").sum()), "feature_rows": int(len(f)),
                    "evaluated": int(len(d)), "by_season": d.groupby("season").size().to_dict(),
                    "same_book_moves": int(d["same_book"].sum())},
           "null_share": {c: float(d[c].isna().mean()) for c in FAMILIES["all"]},
           "dev": {"outcome": outcome_test(d), "move": move_test(d), "coefficients": coefficients(d, n_boot=a.n_boot)}}
    if a.holdout:
        res["holdout"] = {"outcome": outcome_test(d, scored="holdout"), "move": move_test(d, scored="holdout")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(_json(res) + "\n")
    print(_json({k: v for k, v in res.items() if k != "null_share"})[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
