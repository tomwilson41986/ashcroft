"""The football lab: a cached feature matrix and a one-minute experiment runner (docs/FOOTBALL_FAST_LOOP.md).

    python -m football.lab cache --root <sources> --history-root <api-football tables> --cache .football_cache
    python -m football.lab run --cache .football_cache --arm base=team_all --arm xg=team_all+xg_all --model stack

``cache`` builds the matrix in stages, each stored as parquet under ``--cache`` with a key: the sha256 of the source of
the modules the stage runs, its inputs' fingerprints and the upstream stages' keys. A stage whose key is unchanged is
read, not rebuilt: editing a feature module rebuilds that stage and those after it, and nothing else. Stages:
Dixon-Coles prices (collection-aligned) -> team features -> the early bookmaker prices (football-data's silver) ->
xG features -> player ratings' line-up surprise -> the joined matrix.

``run`` reads the matrix, adds the feature blocks (``football/blocks``, computed in seconds), and scores each arm (a
named feature set: families, blocks and columns joined by ``+``) with each model, walk-forward by season on the
early-to-close move of the home side's chance (Pinnacle to Pinnacle or Betfair to Betfair). It reports R2 against the
training mean, a permuted floor, the move caught by the top tenth of forecasts and its right-way share, each fold's R2,
and, on Betfair's own 2024/25 prices, the trade costed two ways. Every run is appended to the ledger
(``reports/football_runs/lab_ledger.jsonl``). The seasons from 2025/26 were read for earlier rules and are left out
unless ``--include-spent`` is given; a result that uses them is development evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "reports" / "football_runs" / "lab_ledger.jsonl"
SPENT_FROM = 2025
CLIP, RIDGE_ALPHA = 4.0, 50.0

# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


def _sha(paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode())
        h.update((ROOT / p).read_bytes())
    return h.hexdigest()[:16]


def _fp(path: Path) -> str:
    """A file's fingerprint: size and sha256 of its bytes (inputs are tens of MB: seconds)."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"{path.stat().st_size}:{h}"


STAGE_CODE = {
    "dc": ["football/team_features.py", "football/model.py", "football/ratings.py"],
    "team": ["football/team_features.py"],
    "books": ["football/lab.py"],
    "xg": ["football/xg_features.py", "football/team_features.py", "football/player_ratings.py"],
    "players": ["football/player_ratings.py", "football/team_features.py"],
}


class Cache:
    def __init__(self, cache: str, root: str | None = None, history_root: str | None = None):
        self.dir = Path(cache)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.root, self.history_root = root, history_root
        self.keys: dict[str, str] = {}
        self.timings: dict[str, float] = {}

    def _store(self, root):
        from sources.common import Store
        return Store(root=root)

    def inputs(self) -> dict:
        r = Path(self.root) if self.root else None
        hr = Path(self.history_root) if self.history_root else None
        return {"gold": _fp(r / "football/gold/matches.parquet") if r else "s3",
                "silver": _fp(r / "football/matches.parquet") if r else "s3",
                "history": "|".join(_fp(hr / f"api_football/history_{x}.parquet")
                                    for x in ("matches", "players", "events", "stats")) if hr else "none"}

    def key(self, stage: str, upstream: list[str], inputs: list[str]) -> str:
        ins = self.inputs()
        parts = [stage, _sha(STAGE_CODE[stage])] + [self.keys[u] for u in upstream] + [ins[i] for i in inputs]
        self.keys[stage] = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        return self.keys[stage]

    def stage(self, stage: str, upstream: list[str], inputs: list[str], fn) -> pd.DataFrame:
        k = self.key(stage, upstream, inputs)
        path = self.dir / f"{stage}_{k}.parquet"
        if path.exists():
            log.info("cache: %s %s (read)", stage, k)
            return pd.read_parquet(path)
        t0 = time.time()
        log.info("cache: %s %s (building)", stage, k)
        df = fn()
        df.to_parquet(path, index=False)
        self.timings[stage] = round(time.time() - t0, 1)
        log.info("cache: %s built in %.0f s", stage, self.timings[stage])
        return df

    def build(self) -> Path:
        from football.data import load_gold
        from football.team_features import build as team_build, dc_collection_aligned
        gs = self._store(self.root)
        gold = load_gold(gs)
        dc = self.stage("dc", [], ["gold"], lambda: dc_collection_aligned(gold, "2012-07-01"))
        team = self.stage("team", ["dc"], ["gold"], lambda: team_build(gold, dc)[0])
        books = self.stage("books", [], ["gold", "silver"], lambda: early_books(gs, gold))
        parts = [team, books]
        if self.history_root:
            hs = self._store(self.history_root)
            hm, hp, he, st = (hs.get_parquet(f"api_football/history_{x}.parquet")
                              for x in ("matches", "players", "events", "stats"))
            from football.player_ratings import build_all, link_matches
            from football.reference import load_aliases
            from football.xg_features import build as xg_build
            aliases = load_aliases(extra=gs.get_parquet("football/gold/team_aliases.parquet"))

            def xg():
                f = xg_build(team, link_matches(hm, gold, aliases), st, hm, he)
                return f[["match_id"] + [c for c in f.columns if c.startswith("x_")
                                         or c in ("r_xg_diff", "xg_gap_vs_market", "n_xg_prev_min")]]

            def players():
                f, _m, _g = build_all(gold, team, hm, hp, he, aliases)
                return f[["match_id"] + [c for c in f.columns if c.startswith(("mpm_", "gpm_"))]]
            parts += [self.stage("xg", ["team"], ["gold", "history"], xg),
                      self.stage("players", ["team"], ["gold", "history"], players)]
        key = hashlib.sha256("|".join(self.keys[s] for s in self.keys).encode()).hexdigest()[:16]
        out = self.dir / f"matrix_{key}.parquet"
        if not out.exists():
            m = parts[0]
            for p in parts[1:]:
                m = m.merge(p.drop_duplicates("match_id"), on="match_id", how="left")
            m.to_parquet(out, index=False)
        (self.dir / "LATEST").write_text(out.name + "\n")
        (self.dir / "manifest.json").write_text(json.dumps(
            {"matrix": out.name, "keys": self.keys, "built_s": self.timings, "inputs": self.inputs(),
             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))
        return out


def early_books(gs, gold: pd.DataFrame) -> pd.DataFrame:
    """The early bookmaker prices of football-data's silver rows (market average, best, Bet365, Pinnacle; 1X2 and
    over/under 2.5), joined to the canonical match_id."""
    from football.data import _with_ids
    s = gs.get_parquet("football/matches.parquet")
    ids = _with_ids(s, gold)
    cols = {"AvgH": "bk_avg_h", "AvgD": "bk_avg_d", "AvgA": "bk_avg_a", "MaxH": "bk_max_h", "MaxD": "bk_max_d",
            "MaxA": "bk_max_a", "B365H": "bk_b365_h", "B365D": "bk_b365_d", "B365A": "bk_b365_a",
            "PSH": "bk_ps_h", "PSD": "bk_ps_d", "PSA": "bk_ps_a", "Avg>2.5": "bk_avg_o", "Avg<2.5": "bk_avg_u",
            "Max>2.5": "bk_max_o", "Max<2.5": "bk_max_u"}
    out = pd.DataFrame({"match_id": ids["match_id"]})
    for src, dst in cols.items():
        v = pd.to_numeric(ids[src], errors="coerce") if src in ids.columns else np.nan
        out[dst] = np.where(np.asarray(v, float) > 1.0, v, np.nan)
    return out.drop_duplicates("match_id")


def latest(cache: str) -> pd.DataFrame:
    name = (Path(cache) / "LATEST").read_text().strip()
    return pd.read_parquet(Path(cache) / name)


# ---------------------------------------------------------------------------
# arms and blocks
# ---------------------------------------------------------------------------


def families(m: pd.DataFrame) -> dict:
    from football.feature_eval import FAMILIES
    fam = dict(FAMILIES)
    fam["team_all"] = FAMILIES["all"]
    try:
        from football.xg_features import families as xg_fam
        for k, v in xg_fam().items():
            if not k.startswith("team"):
                fam[k] = [c for c in v if c in m.columns]
    except Exception:                                    # no xG stage in this cache
        pass
    fam["players"] = [c for c in ("mpm_surprise", "gpm_surprise", "mpm_xi_diff", "gpm_xi_diff") if c in m.columns]
    return fam


def add_blocks(m: pd.DataFrame, names: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """The matrix with the named blocks' columns joined (all blocks when ``names`` is None); {block: FEATURES}."""
    from football.blocks import available
    blocks = available()
    use = blocks if names is None else {n: blocks[n] for n in names}
    out, feats = m, {}
    for name, mod in use.items():
        t0 = time.time()
        b = mod.build(m)
        out = out.merge(b[["match_id"] + mod.FEATURES].drop_duplicates("match_id"), on="match_id", how="left")
        feats[name] = list(mod.FEATURES)
        log.info("block %s: %d features in %.1f s", name, len(mod.FEATURES), time.time() - t0)
    return out, feats


def resolve(spec: str, fam: dict, blocks: dict, columns) -> list[str]:
    """``team_all+xg_core+schedule+r_diff`` -> the column list (families, blocks or single columns)."""
    cols: list[str] = []
    for part in spec.split("+"):
        part = part.strip()
        if part in fam:
            cols += fam[part]
        elif part in blocks:
            cols += blocks[part]
        elif part in columns:
            cols.append(part)
        else:
            raise SystemExit(f"unknown arm part {part!r}: not a family, block or column")
    return list(dict.fromkeys(cols))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def _z(tr, te, cols):
    X, Xt = tr[cols].to_numpy(float), te[cols].to_numpy(float)
    keep = np.mean(np.isfinite(X), axis=0) >= 0.5
    X, Xt = X[:, keep], Xt[:, keep]
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    f = lambda A: np.clip(np.nan_to_num((A - mu) / sd), -CLIP, CLIP)       # noqa: E731
    return f(X), f(Xt)


def ridge(tr, te, cols, y):
    Z, Zt = _z(tr, te, cols)
    beta = np.linalg.solve(Z.T @ Z + RIDGE_ALPHA * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return y.mean() + Zt @ beta


def gbm(tr, te, cols, y, mkt=("league",)):
    import lightgbm as lgb
    use = cols + [c for c in mkt if c in tr.columns]
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    g.fit(tr[use], y)
    return g.predict(te[use])


MODELS = {"ridge": lambda tr, te, c, y: ridge(tr, te, c, y),
          "gbm": lambda tr, te, c, y: gbm(tr, te, c, y),
          "stack": lambda tr, te, c, y: (ridge(tr, te, c, y) + gbm(tr, te, c, y)) / 2}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def prepared(m: pd.DataFrame) -> pd.DataFrame:
    from football.feature_eval import prepare
    d = prepare(m)
    d["league"] = pd.Categorical(d["fd_code"].astype(str))
    d["y_h"] = np.log(d["qc_h"] / d["qo_h"])
    d["y_a"] = np.log(d["qc_a"] / d["qo_a"])
    return d


def folds(d, first, last, train_from):
    for s in range(first, last + 1):
        tr, te = d[(d["season"] >= train_from) & (d["season"] < s)], d[d["season"] == s]
        if len(tr) >= 500 and len(te):
            yield s, tr, te


def score_arm(d, cols, model, first, last, train_from, seed=0) -> dict:
    """Walk-forward on the home chance's move, same-book rows."""
    m = d[d["same_book"]]
    rng = np.random.default_rng(seed)
    fit = MODELS[model]
    ys, ps, pp, mb, per = [], [], [], [], {}
    for s, tr, te in folds(m, first, last, train_from):
        y = tr["y_h"].to_numpy(float)
        p = fit(tr, te, cols, y)
        q = ridge(tr, te, cols, rng.permutation(y))
        yt = te["y_h"].to_numpy(float)
        ys.append(yt), ps.append(p), pp.append(q), mb.append(np.full(len(te), y.mean()))
        per[int(s)] = 1 - float(np.sum((yt - p) ** 2)) / float(np.sum((yt - y.mean()) ** 2))
    if not ys:
        return {"rows": 0}
    y, p, q, b = (np.concatenate(x) for x in (ys, ps, pp, mb))
    sse = lambda f: float(np.sum((y - f) ** 2))          # noqa: E731
    top = np.abs(p) >= np.quantile(np.abs(p), 0.9)
    caught = np.sign(p[top]) * y[top]
    r = np.array(list(per.values()))
    return {"rows": int(len(y)), "r2": 1 - sse(p) / sse(b), "r2_floor": 1 - sse(q) / sse(b),
            "r2_fold_mean": float(r.mean()), "r2_fold_se": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else None,
            "top_caught": float(caught.mean()), "top_right": float((caught > 0).mean()), "fold_r2": per}


def betfair_costed(d, cols, model, season, train_from, threshold=0.03) -> dict:
    """The arm traded on Betfair's own prices of ``season`` (home and away sides; takeable books, 1.05 or under),
    fitted on the Pinnacle moves of the seasons before it, costed a tick off the fair price and with a lay from a
    book symmetric about fair (research/open_close/football_team_move.py, team_move_cost_check.json)."""
    from research.open_close.backtest import hedged_trades, summarise
    from research.open_close.football_exchange import width_label
    from trading.exchange import tick_up
    pin = d[d["same_book"] & (d["open_1x2_src"] == "PS") & (d["season"] >= train_from) & (d["season"] < season)]
    b = d[(d["season"] == season) & d[["bfe_open_h", "bfe_open_d", "bfe_open_a", "bfe_close_h", "bfe_close_d",
                                       "bfe_close_a"]].gt(1).all(axis=1)]
    if b.empty or len(pin) < 500:
        return {}
    over_o = 1 / b["bfe_open_h"] + 1 / b["bfe_open_d"] + 1 / b["bfe_open_a"]
    over_c = 1 / b["bfe_close_h"] + 1 / b["bfe_close_d"] + 1 / b["bfe_close_a"]
    rows = []
    for side, k in (("h", "y_h"), ("a", "y_a")):
        pred = -MODELS[model](pin, b, cols, pin[k].to_numpy(float))           # chance up = price down
        r = pd.DataFrame({"market_id": (b["match_id"] + ":MO").to_numpy(), "day": b["match_date"].astype(str).to_numpy(),
                          "month": b["match_date"].astype(str).str[:7].to_numpy(), "sport": "football",
                          "country": b["country"].to_numpy(), "market_type": "MATCH_ODDS",
                          "open_raw": b[f"bfe_open_{side}"].to_numpy(float), "close_raw": b[f"bfe_close_{side}"].to_numpy(float),
                          "over": over_o.to_numpy(float), "pred": pred})
        r["open_price"] = r["open_raw"] * r["over"]
        r["close_price"] = r["close_raw"] * over_c.to_numpy(float)
        r["y"] = np.log(r["close_price"] / r["open_price"])
        rows.append(r)
    r = pd.concat(rows, ignore_index=True)
    r = r[np.isin(width_label(r["over"]), ["tight <1.02", "medium 1.02-1.05"])].reset_index(drop=True)
    out = {"season": season, "threshold": threshold}
    t = hedged_trades(r, "pred", threshold, spread=True)
    out["tick_off_fair"] = _brief(summarise(t, n_boot=500)) if len(t) else {"bets": 0}
    t = hedged_trades(r, "pred", threshold)
    if len(t):
        idx = t["row"].to_numpy()
        ob, cb = r["open_raw"].to_numpy(float)[idx], r["close_raw"].to_numpy(float)[idx]
        of, cf = r["open_price"].to_numpy(float)[idx], r["close_price"].to_numpy(float)[idx]
        sym = lambda raw, fair: np.array([tick_up(v - 1e-9) for v in fair ** 2 / raw])   # noqa: E731
        back = t["side"].to_numpy() > 0
        o = np.where(back, ob, sym(ob, of))
        c = np.where(back, sym(cb, cf), cb)
        t["profit"] = np.where(back, o / c - 1, 1 - o / c)
        out["symmetric_book"] = _brief(summarise(t, n_boot=500))
        out["mid"] = _brief(summarise(hedged_trades(r, "pred", threshold), n_boot=0))
    return out


def _brief(s: dict) -> dict:
    return {k: s.get(k) for k in ("bets", "roi", "roi_lo90", "roi_hi90", "right_way")}


def run(cache: str, arms: dict, models: list[str], first: int, last: int, train_from: int, subset: str | None,
        blocks: list[str] | None, include_spent: bool, costed: bool, note: str = "") -> dict:
    t0 = time.time()
    m = latest(cache)
    m, bfeat = add_blocks(m, blocks)
    fam = families(m)
    d = prepared(m)
    if subset == "xg":
        d = d[d.get("n_xg_prev_min", pd.Series(0, index=d.index)).fillna(0) >= 3]
    if not include_spent:
        last = min(last, SPENT_FROM - 1)
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": note,
           "matrix": (Path(cache) / "LATEST").read_text().strip(), "git": _git(), "folds": [first, last],
           "train_from": train_from, "subset": subset, "include_spent": include_spent, "arms": {}}
    for name, spec in arms.items():
        cols = resolve(spec, fam, bfeat, d.columns)
        for model in models:
            t1 = time.time()
            r = score_arm(d, cols, model, first, last, train_from)
            if costed and first <= 2024 <= max(last, 2024):
                r["betfair_2024"] = betfair_costed(d, cols, model, 2024, train_from)
            r.update({"spec": spec, "n_features": len(cols), "seconds": round(time.time() - t1, 1)})
            res["arms"][f"{name}:{model}"] = r
            log.info("%s:%s  R2 %.2f%% (floor %.2f%%)  top %.2f%%  %.0fs", name, model, 100 * r.get("r2", 0),
                     100 * r.get("r2_floor", 0), 100 * r.get("top_caught", 0), r["seconds"])
    res["seconds"] = round(time.time() - t0, 1)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(res, default=_default) + "\n")
    return res


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def _default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    return str(x)


def table(res: dict) -> str:
    lines = [f"{'arm:model':34} {'rows':>6} {'R2':>7} {'floor':>7} {'fold se':>8} {'top':>7} {'right':>6} "
             f"{'BF sym':>8} {'BF tick':>8} {'s':>5}"]
    for k, r in res["arms"].items():
        bf = r.get("betfair_2024", {})
        sym = (bf.get("symmetric_book") or {}).get("roi")
        tick = (bf.get("tick_off_fair") or {}).get("roi")
        f = lambda v: f"{100 * v:+.2f}%" if isinstance(v, (int, float)) and v == v else "-"      # noqa: E731
        lines.append(f"{k:34} {r.get('rows', 0):>6} {f(r.get('r2')):>7} {f(r.get('r2_floor')):>7} "
                     f"{f(r.get('r2_fold_se')):>8} {f(r.get('top_caught')):>7} "
                     f"{(r.get('top_right') or 0):>6.3f} {f(sym):>8} {f(tick):>8} {r.get('seconds', 0):>5}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--root", default=None)
    c.add_argument("--history-root", default=None)
    c.add_argument("--cache", default=".football_cache")
    r = sub.add_parser("run")
    r.add_argument("--cache", default=".football_cache")
    r.add_argument("--arm", action="append", required=True, help="name=family+block+column (repeatable)")
    r.add_argument("--model", action="append", default=None, choices=list(MODELS))
    r.add_argument("--from", dest="first", type=int, default=2016)
    r.add_argument("--to", dest="last", type=int, default=2024)
    r.add_argument("--train-from", type=int, default=2013)
    r.add_argument("--subset", choices=["xg"], default=None)
    r.add_argument("--blocks", default=None, help="comma-separated blocks to add (default: all)")
    r.add_argument("--include-spent", action="store_true", help="score the seasons from 2025/26 (read before)")
    r.add_argument("--no-cost", action="store_true", help="skip the Betfair 2024/25 costing")
    r.add_argument("--note", default="")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if a.cmd == "cache":
        print(Cache(a.cache, a.root, a.history_root).build())
        return 0
    arms = dict(x.split("=", 1) for x in a.arm)
    res = run(a.cache, arms, a.model or ["ridge"], a.first, a.last, a.train_from, a.subset,
              a.blocks.split(",") if a.blocks else None, a.include_spent, not a.no_cost, a.note)
    print(table(res))
    print(f"total {res['seconds']} s; ledger {LEDGER.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
