"""The stacked move rule for the football paper trial: choose its threshold, freeze its models, score it forward at
the recorded Betfair books (reports/preregistration_football_stack.md).

The rule: the early-to-close move of each side's chance (home and away), forecast by the mean of a ridge regression and
a LightGBM model on the team features, arm ``team_all`` (reports/preregistration_football_stack.md). The models are
fitted ONCE on every same-book move of 2013/14-2024/25 and frozen; their files and SHA-256 are in the pre-registration.
In the matrix those moves are Pinnacle's only: 2024/25's stored close is Pinnacle's, so its 488 same-book matches are
Pinnacle to Pinnacle, and the "2024/25 fold" the arm was chosen on is those 488 matches, not Betfair's
(reports/preregistration_football_stack_errata.md; the rule itself is unchanged). Forward, each match is scored from the lab matrix rebuilt on the data as it then stands: every
feature is built from what was known before the match's collection day (tests/test_football_lab.py), so a
retrospective score equals a live one.

    python -m research.open_close.football_stack_trial sweep  --cache <lab cache>                 # threshold, 2024/25
    python -m research.open_close.football_stack_trial freeze --cache <lab cache> --out models/football/stack_v1
    python -m research.open_close.football_stack_trial score  --cache <lab cache> --models models/football/stack_v1 \\
        --root <sources> --out out/football_stack_trial.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ARM = "team_all"
THRESHOLDS = (0.01, 0.02, 0.03, 0.04, 0.05)
TRAIN_FROM, TRAIN_TO = 2013, 2024
PREREG = "reports/preregistration_football_stack.md"
GATE = {"min_bets": 300, "min_days": 90, "roi_lo90_above": 0, "clv_mean_above": 0}
TOP5 = ("E0", "D1", "I1", "SP1", "F1")


def _matrix(cache: str) -> tuple[pd.DataFrame, list[str]]:
    from football import lab
    m = lab.latest(cache)
    m, bfeat = lab.add_blocks(m, [])
    d = lab.prepared(m)
    cols = lab.resolve(ARM, lab.families(m), bfeat, d.columns)
    return d, cols


def sweep(cache: str) -> dict:
    """The threshold, chosen on 2024/25 only (models fitted on the seasons before it): the best 90% lower bound of
    the return at a symmetric book among thresholds with at least 300 bets in takeable books."""
    from football import lab
    d, cols = _matrix(cache)
    rows = {}
    for th in THRESHOLDS:
        r = lab.betfair_costed(d, cols, "stack", 2024, TRAIN_FROM, threshold=th)
        rows[th] = r
        log.info("threshold %.2f: %s", th, r.get("symmetric_book"))
    ok = {th: r["symmetric_book"] for th, r in rows.items()
          if (r.get("symmetric_book") or {}).get("bets", 0) >= 300 and r["symmetric_book"].get("roi_lo90") is not None}
    chosen = max(ok, key=lambda th: ok[th]["roi_lo90"]) if ok else None
    return {"arm": ARM, "season": 2024, "rule": "best symmetric-book 90% lower bound, >= 300 bets", "chosen": chosen,
            "by_threshold": {str(k): v for k, v in rows.items()}}


# ---------------------------------------------------------------------------
# freeze and predict
# ---------------------------------------------------------------------------


def freeze(cache: str, out: str) -> dict:
    """Fit the home and away models on every same-book move of TRAIN_FROM-TRAIN_TO and write them with their hashes."""
    import lightgbm as lgb
    from football import lab
    d, cols = _matrix(cache)
    tr = d[d["same_book"] & d["season"].between(TRAIN_FROM, TRAIN_TO)]
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    X = tr[cols].to_numpy(float)
    keep = np.mean(np.isfinite(X), axis=0) >= 0.5
    kept = [c for c, k in zip(cols, keep) if k]
    mu, sd = np.nanmean(X[:, keep], 0), np.nanstd(X[:, keep], 0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    Z = np.clip(np.nan_to_num((X[:, keep] - mu) / sd), -lab.CLIP, lab.CLIP)
    spec = {"arm": ARM, "features": cols, "ridge_features": kept, "mu": mu.tolist(), "sd": sd.tolist(),
            "clip": lab.CLIP, "ridge_alpha": lab.RIDGE_ALPHA, "train": [TRAIN_FROM, TRAIN_TO], "rows": int(len(tr)),
            "leagues": sorted(tr["fd_code"].astype(str).unique()), "sides": {}}
    for side, ycol in (("h", "y_h"), ("a", "y_a")):
        y = tr[ycol].to_numpy(float)
        beta = np.linalg.solve(Z.T @ Z + lab.RIDGE_ALPHA * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
        g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        Xg = tr[cols + ["league"]].copy()
        Xg["league"] = pd.Categorical(Xg["league"].astype(str), categories=spec["leagues"])
        g.fit(Xg, y)
        path = o / f"gbm_{side}.txt"
        g.booster_.save_model(str(path))
        spec["sides"][side] = {"ridge_intercept": float(y.mean()), "ridge_beta": beta.tolist(), "gbm": path.name}
    (o / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(o.iterdir()) if p.is_file()}
    (o / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items() if n != "SHA256SUMS"))
    return {"out": str(o), "rows": spec["rows"], "features": len(cols), "ridge_features": len(kept), "sha256": hashes}


def verify_models(models: str, expected: dict) -> None:
    """Refuse to score unless every frozen file's SHA-256 is the one the pre-registration records."""
    o = Path(models)
    for name, want in expected.items():
        f = o / name
        got = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
        if got != want:
            raise SystemExit(f"{f}: SHA-256 {got} is not the pre-registered {want}")


def predict(models: str, d: pd.DataFrame) -> pd.DataFrame:
    """Each match's forecast chance moves (home, away) from the frozen models: the mean of ridge and LightGBM."""
    import lightgbm as lgb
    o = Path(models)
    spec = json.loads((o / "spec.json").read_text())
    X = d[spec["ridge_features"]].to_numpy(float)
    Z = np.clip(np.nan_to_num((X - np.array(spec["mu"])) / np.array(spec["sd"])), -spec["clip"], spec["clip"])
    Xg = d[spec["features"] + ["league"]].copy()
    Xg["league"] = pd.Categorical(Xg["league"].astype(str), categories=spec["leagues"])
    out = pd.DataFrame({"match_id": d["match_id"].to_numpy()})
    for side, s in spec["sides"].items():
        ridge = s["ridge_intercept"] + Z @ np.array(s["ridge_beta"])
        gbm = lgb.Booster(model_file=str(o / s["gbm"])).predict(Xg)
        out[f"chance_{side}"] = (ridge + gbm) / 2
    return out


# ---------------------------------------------------------------------------
# the forward score
# ---------------------------------------------------------------------------


def signals(d: pd.DataFrame, pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """The trial's signal rows (the paper-trial scorer's format): one a side with a forecast; the price forecast is
    minus the chance forecast; back when the price is forecast shorter by more than ``threshold``, lay when longer."""
    from research.open_close import football_paper_trial as pt
    from research.open_close.football_exchange import width_label
    m = d.merge(pred, on="match_id")
    rows = []
    for leg, k in (("H", "h"), ("A", "a")):
        yhat = -m[f"chance_{k}"].to_numpy(float)
        rows.append(pd.DataFrame({
            "event_id": m["match_id"].to_numpy(), "market_type": "MATCH_ODDS", "selection_id": leg, "yhat": yhat,
            "side": np.where(yhat < -threshold, 1, np.where(yhat > threshold, -1, 0)),
            "fd_code": m["fd_code"].astype(str).to_numpy(),
            "tier": np.where(m["fd_code"].isin(TOP5), "top five", "rest"),
            "match_day": m["match_date"].astype(str).to_numpy(),
            "book_width": width_label(1 / m["bfe_open_h"] + 1 / m["bfe_open_d"] + 1 / m["bfe_open_a"]),
            "fit_month": "frozen"}))
    s = pd.concat(rows, ignore_index=True)
    coll = [pt.collection_of(x) for x in s["match_day"]]
    s["collection"] = [c.isoformat() for c, _ in coll]
    s["moment_utc"] = [mo for _, mo in coll]
    return s


def score(cache: str, models: str, root: str | None, live_root: str | None, start: str | None,
          as_of: str | None, prereg: str = PREREG, n_boot: int = 2000) -> dict:
    from datetime import date, timedelta

    from football.data import load_gold
    from football.reference import load_aliases
    from research.open_close import football_paper_trial as pt
    from research.open_close.football_exchange import read_prereg
    from sources.common import Store
    rule, digest = read_prereg(prereg)
    verify_models(models, rule["model_sha256"])
    read_day = pt.recorder_reader(live_root)
    today = date.today()
    as_of_d = date.fromisoformat(as_of) if as_of else today - timedelta(days=1)
    if start:
        start_d = date.fromisoformat(start)
    else:
        first = pt.first_recorded_day(read_day, max(date.fromisoformat(pt.RECORDER_FROM),
                                                    date.fromisoformat(rule["written_utc"][:10])), as_of_d)
        if first is None:
            return {"status": "no football book record found", "prereg_sha256": digest}
        start_d = pt.trial_start(first)
    if start_d > as_of_d:                  # the record has begun, the trial's first collection has not
        return {"status": "not started", "start": start_d.isoformat(), "as_of": as_of_d.isoformat(),
                "prereg_sha256": digest, "rule": rule}
    d, _cols = _matrix(cache)
    d = d[(d["match_date"] >= start_d.isoformat()) & (d["match_date"] <= as_of_d.isoformat())
          & d["fd_code"].isin(rule["leagues_fd"])]
    if d.empty:
        return {"status": "no matches in the window yet", "start": start_d.isoformat(), "as_of": as_of_d.isoformat(),
                "prereg_sha256": digest, "rule": rule}
    sig = signals(d, predict(models, d), rule["threshold"])
    store = Store(root=root)
    t = SimpleNamespace(gold=load_gold(store), read_day=read_day, tolerance_min=pt.TOLERANCE_MIN,
                        notional=pt.NOTIONAL_GBP,
                        aliases=load_aliases(extra=store.get_parquet("football/gold/team_aliases.parquet")))
    tr, sk = pt.trade_signals(t, sig, rule["book_width"]) if (sig["side"] != 0).any() else (pd.DataFrame(), pd.DataFrame())
    w = pt.window(tr, start_d, as_of_d, GATE["min_bets"], GATE["min_days"])
    inw = tr[pd.to_datetime(tr["day"]).dt.date <= date.fromisoformat(w["end"])] if len(tr) else tr
    res = {"prereg": prereg, "prereg_sha256": digest, "rule": rule, "start": start_d.isoformat(),
           "as_of": as_of_d.isoformat(), "forecasts": int(len(sig)), "signals": int((sig["side"] != 0).sum()),
           "window": w, "headline": pt.scored(inw, rule["commission"], n_boot),
           "skipped": sk["reason"].value_counts().to_dict() if len(sk) else {}}
    res["halves"] = (pt.halves(inw, start_d, date.fromisoformat(w["end"]), rule["commission"]) if len(inw)
                     else {"first": {"bets": 0}, "second": {"bets": 0}})
    for key in ("book_width", "band", "tier", "fd_code", "month"):
        res[f"by_{key}"] = pt.by_key(inw, key, rule["commission"])
    res["overfit"] = pt.overfit(inw, rule["commission"], pt.trials_considered(rule, GATE))
    res["gate"] = pt.gate(res, GATE)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--cache", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--cache", required=True)
    f.add_argument("--out", default="models/football/stack_v1")
    c = sub.add_parser("score")
    c.add_argument("--cache", required=True)
    c.add_argument("--models", default="models/football/stack_v1")
    c.add_argument("--root", default=None)
    c.add_argument("--live-root", default=None)
    c.add_argument("--start", default=None)
    c.add_argument("--as-of", default=None)
    c.add_argument("--prereg", default=PREREG)
    for p in (s, f, c):
        p.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    res = (sweep(a.cache) if a.cmd == "sweep" else freeze(a.cache, a.out) if a.cmd == "freeze"
           else score(a.cache, a.models, a.root, a.live_root, a.start, a.as_of, a.prereg))
    text = json.dumps(res, indent=1, default=str)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(text + "\n")
    print(text[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
