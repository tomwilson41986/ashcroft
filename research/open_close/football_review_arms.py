"""The model review's repairs and new blocks, one change at a time against the expected-value model's arm.

racing² football research, 10 Nov 2026 (the owner: improve the closing-price model; the review's steps 1b and 2).
Each arm is the base arm (``team_all+home_adv+team_news+form_short+x_short``, the stack of ridge and LightGBM) with one
change; ``all`` carries every change that is a feature. Two reads, both development evidence (every season to date was
read by earlier rules; the holdout is the matches from 10 Nov 2026, not read here):

- **Pinnacle, walk-forward 2016/17-2024/25** (each season fitted on the same-book moves from 2013/14 before it), the
  home chance's move: skill against "no move";
- **Betfair's own moves, 2025/26 to date** (fitted on the same-book moves 2013/14-2024/25), home and away.

Against the base, paired: the change in skill, with an interval from 1,000 resamples of match days (the same days
drawn for both arms). A change keeps its place only if the interval on Betfair's moves excludes zero.

Arms:
- ``price``: the early chance (log) of both sides added (review A5);
- ``logit``: fitted on the move in logit units and turned back into the log-chance move (A5);
- ``no_league``: LightGBM without the league category;
- ``league_fx``: the shrunk league effects in its place (F);
- ``move_memory``, ``kalman_form``: the new blocks added;
- ``kalman_for_windows``: the Kalman block in place of the raw window ladder's points and goal-difference columns (A6);
- ``shrunk_ratings``: the xG and Massey ratings with the estimated penalty added (A1);
- ``all``.

    python -m research.open_close.football_review_arms --cache <lab cache> --out-json out/football_review_arms.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE = "team_all+home_adv+team_news+form_short+x_short"
BLOCKS = ["home_adv", "form_short", "league_fx", "move_memory", "kalman_form"]
TRAIN_FROM, DEV = 2013, (2016, 2024)
RAW_LADDER = ("pts", "gd")                 # the unadjusted measures the Kalman block stands in for

ARMS = {
    "base": {},
    "price": {"add": ["lq_h", "lq_a"]},
    "logit": {"target": "logit"},
    "price+logit": {"add": ["lq_h", "lq_a"], "target": "logit"},
    "no_league": {"league": False},
    "league_fx": {"add": "league_fx", "league": False},
    "league_fx+league": {"add": "league_fx"},
    "move_memory": {"add": "move_memory"},
    "kalman_form": {"add": "kalman_form"},
    "kalman_for_windows": {"add": "kalman_form", "drop_raw": True},
    "shrunk_ratings": {"add": ["r_xgs_diff", "xgs_gap_vs_market", "x_masseys_diff", "x_masseys_vs_open"]},
    "all": {"add": ["lq_h", "lq_a", "league_fx", "move_memory", "kalman_form", "r_xgs_diff", "xgs_gap_vs_market",
                    "x_masseys_diff", "x_masseys_vs_open"]},
}


def columns(base: list[str], arm: dict, bfeat: dict, have) -> list[str]:
    cols = list(base)
    add = arm.get("add", [])
    for a in ([add] if isinstance(add, str) else add):
        cols += bfeat[a] if a in bfeat else ([a] if a in have else [])
    if arm.get("drop_raw"):
        cols = [c for c in cols if not any(c.startswith((f"d_{m}_", f"fs_{m}_")) for m in RAW_LADDER)]
    return list(dict.fromkeys(cols))


def logit(q):
    q = np.clip(np.asarray(q, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], side: str, arm: dict, qo_te: np.ndarray):
    """The stack's forecast of the side's log-chance move on ``te``. ``qo_te``: the side's early chance there (to
    turn a logit forecast back)."""
    import lightgbm as lgb
    from football import lab
    if arm.get("target") == "logit":
        y = logit(tr[f"qc_{side}"]) - logit(tr[f"qo_{side}"])
    else:
        y = tr[f"y_{side}"].to_numpy(float)
    use = cols + (["league"] if arm.get("league", True) else [])
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    g.fit(tr[use], y)
    p = (lab.ridge(tr, te, cols, y) + g.predict(te[use])) / 2
    if arm.get("target") == "logit":
        qc = 1 / (1 + np.exp(-(logit(qo_te) + p)))
        p = np.log(qc / np.clip(qo_te, 1e-6, 1))
    return p


def paired(days: np.ndarray, se_base: np.ndarray, se_arm: np.ndarray, zero: np.ndarray, n_boot: int = 1000,
           seed: int = 0) -> dict:
    """The arm's skill less the base's (both against 'no move'), with a 95% interval from resampled days."""
    d = pd.DataFrame({"day": days, "b": se_base, "a": se_arm, "z": zero}).groupby("day").sum()
    b, a, z = d["b"].to_numpy(), d["a"].to_numpy(), d["z"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = (b[idx].sum(1) - a[idx].sum(1)) / z[idx].sum(1)
    return {"skill": float(1 - a.sum() / z.sum()), "vs_base": float((b.sum() - a.sum()) / z.sum()),
            "lo95": float(np.quantile(boot, 0.025)), "hi95": float(np.quantile(boot, 0.975)), "days": int(len(d))}


def run(cache: str, only: list[str] | None = None) -> dict:
    from football import lab
    from research.open_close.football_stack_betfair_read import betfair_rows
    m = lab.latest(cache)
    m, bfeat = lab.add_blocks(m, BLOCKS)
    d = lab.prepared(m)
    base = lab.resolve(BASE, lab.families(m), bfeat, d.columns)
    same = d[d["same_book"] & np.isfinite(d["y_h"]) & np.isfinite(d["y_a"])]
    bf = d[(d["season"] >= 2025) & betfair_rows(d)].copy()
    oo = 1 / bf[["bfe_open_h", "bfe_open_d", "bfe_open_a"]].to_numpy(float)
    cc = 1 / bf[["bfe_close_h", "bfe_close_d", "bfe_close_a"]].to_numpy(float)
    qo, qc = oo / oo.sum(1, keepdims=True), cc / cc.sum(1, keepdims=True)
    bf_q = {"h": qo[:, 0], "a": qo[:, 2]}
    bf_y = {"h": np.log(qc[:, 0] / qo[:, 0]), "a": np.log(qc[:, 2] / qo[:, 2])}
    bf_tr = same[same["season"].between(TRAIN_FROM, 2024)]
    res = {"base": BASE, "blocks": {k: v for k, v in bfeat.items() if k in ("league_fx", "move_memory", "kalman_form")},
           "pinnacle": {"folds": list(DEV), "rows": 0}, "betfair": {"rows": int(len(bf)), "train_rows": int(len(bf_tr))},
           "arms": {}, "note": "development evidence: every season here was read by earlier rules"}
    if len(bf) < 500 or len(bf_tr) < 5000:
        res["status"] = "no data found: too few rows to fit or to score"
        return res
    store = {}
    for name, arm in ARMS.items():
        if only and name not in only and name != "base":
            continue
        cols = columns(base, arm, bfeat, d.columns)
        # Pinnacle walk-forward, the home move
        days, se, zero = [], [], []
        for s in range(DEV[0], DEV[1] + 1):
            tr, te = same[(same["season"] >= TRAIN_FROM) & (same["season"] < s)], same[same["season"] == s]
            if len(tr) < 500 or te.empty:
                continue
            p = fit_predict(tr, te, cols, "h", arm, te["qo_h"].to_numpy(float))
            y = te["y_h"].to_numpy(float)
            days.append(te["match_date"].astype(str).to_numpy()), se.append((y - p) ** 2), zero.append(y ** 2)
        pin = tuple(np.concatenate(x) for x in (days, se, zero))
        # Betfair's own moves, both sides
        bse = {s: (bf_y[s] - fit_predict(bf_tr, bf, cols, s, arm, bf_q[s])) ** 2 for s in ("h", "a")}
        bfd = bf["match_date"].astype(str).to_numpy()
        store[name] = (pin, bse)
        b_pin, b_bf = store["base"]
        r = {"features": len(cols),
             "pinnacle_home": paired(pin[0], b_pin[1], pin[1], pin[2]),
             "betfair": paired(np.r_[bfd, bfd], np.r_[b_bf["h"], b_bf["a"]], np.r_[bse["h"], bse["a"]],
                               np.r_[bf_y["h"] ** 2, bf_y["a"] ** 2]),
             "betfair_home_skill": float(1 - bse["h"].sum() / (bf_y["h"] ** 2).sum()),
             "betfair_away_skill": float(1 - bse["a"].sum() / (bf_y["a"] ** 2).sum())}
        r["keeps_its_place"] = bool(name != "base" and r["betfair"]["lo95"] > 0)
        res["arms"][name] = r
        res["pinnacle"]["rows"] = int(len(pin[0]))
        log.info("%-20s feats %3d  pinnacle %.2f%% (%+.2f)  betfair %.2f%% (%+.2f, %+.2f to %+.2f)", name, len(cols),
                 100 * r["pinnacle_home"]["skill"], 100 * r["pinnacle_home"]["vs_base"], 100 * r["betfair"]["skill"],
                 100 * r["betfair"]["vs_base"], 100 * r["betfair"]["lo95"], 100 * r["betfair"]["hi95"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--only", default=None, help="comma-separated arms (the base always runs)")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run(a.cache, a.only.split(",") if a.only else None)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res.get("arms", res), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
