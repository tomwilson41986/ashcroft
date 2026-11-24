"""The 24 Nov feature batch: eight new blocks, each against the best model so far, paired by match day.

racing² football research, 24 Nov 2026 (the owner: add more features; ideate and discover). Registered as
``football-feature-batch-1124`` before its run. The base is the expected-value model's arm with the three blocks the
10 Nov review kept (league effects, move memory, Kalman form). Each arm adds one new block; ``all_new`` adds all eight.
The method is ``football_review_arms``' unchanged: the stack of ridge and LightGBM, Pinnacle walk-forward 2016/17 to
2024/25 and Betfair's own moves 2025/26, skill against "no move", the change against the base with a 95% interval
from 1,000 resamples of match days. A block keeps its place only if its interval on Betfair's moves is above zero.

    python -m research.open_close.football_feature_batch --cache <lab cache> --out-json out/football_feature_batch.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.open_close import football_review_arms as ra

KEPT = ["league_fx", "move_memory", "kalman_form"]
NEW = ["mkt_xg", "decay_form", "sos_split", "venue_fx", "norm_rank", "style", "referee", "sim"]
#: {batch: (ledger tag, blocks in the base, new blocks)}. ``tactics`` (the owner, 24 Nov: "build the match simulator
#: and formation matchups"): the base also carries the two blocks of the first batch that held on both books
BATCHES = {"first": ("football-feature-batch-1124", KEPT, NEW),
           "tactics": ("football-tactics-batch-1124", KEPT + ["sim", "style"], ["formation", "match_sim"]),
           "sportmonks": ("football-sportmonks-xg-1124", KEPT + ["sim", "style"], ["sm_xg"]),
           "fd_lines": ("football-fd-lines-1124", KEPT + ["sim", "style"], ["fd_lines"]),
           "sm_pred": ("football-sportmonks-pred-1124", KEPT + ["sim", "style"], ["sm_pred"]),
           "eu_form": ("football-eu-form-1124", KEPT + ["sim", "style"], ["eu_form"])}
#: {batch: (first training season, (first, last) Pinnacle test season)}: Sportmonks' xG starts in 2024/25, so its
#: batch fits both arms on 2024/25 alone and tests 2025/26 (the other batches keep the review arms' settings)
WINDOWS = {}
#: the batches whose data starts in 2024/25 (Sportmonks, European form). Their first design (fit on the 2024/25
#: same-book moves) found 488 training rows - the bookmakers' same-book pairs are thin in 2024/25 - and no Pinnacle
#: rows in 2025/26 (run 45373794175, "no data found"). They are read instead on Betfair's own moves alone, walk-forward
#: by month from August 2024: each month predicted by models fitted on the earlier months (both arms the same way)
MONTHLY = {"sportmonks", "sm_pred", "eu_form"}
MONTHLY_FROM, MONTHLY_MIN_TRAIN = "2024-08", 3000
EV_ARM = ra.BASE
TRAIN_FROM0, DEV0 = ra.TRAIN_FROM, ra.DEV


def configure(batch: str = "first") -> None:
    """Point the review-arms runner at a batch (its method is not changed)."""
    _tag, kept, new = BATCHES[batch]
    ra.TRAIN_FROM, ra.DEV = WINDOWS.get(batch, (TRAIN_FROM0, DEV0))
    ra.BASE = "+".join([EV_ARM] + kept)
    ra.BLOCKS = ["home_adv", "form_short"] + kept + new
    ra.ARMS = {"base": {}, **{b: {"add": b} for b in new}, "all_new": {"add": list(new)}}
    ra.COSTED = ("base", "all_new")


def coverage(cache: str, new: list[str]) -> dict:
    """The share of rows with a value, a new feature: the seasons the models are fitted on, and 2025/26."""
    from football import lab
    m, bfeat = lab.add_blocks(lab.latest(cache), new)
    d = lab.prepared(m)
    fit, test = d[d["season"].between(ra.TRAIN_FROM, 2024)], d[d["season"] >= 2025]
    return {b: {f: {"fitted_seasons": round(float(fit[f].notna().mean()), 4),
                    "from_2025": round(float(test[f].notna().mean()), 4)} for f in bfeat[b]} for b in new}


def fit_predict(tr, te, cols: list[str], y: np.ndarray) -> np.ndarray:
    """The review arms' stack (ridge and LightGBM averaged, the league as a category) fitted on ``y``."""
    import lightgbm as lgb
    from football import lab
    use = cols + ["league"]
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    g.fit(tr[use], y)
    return (lab.ridge(tr, te, cols, y) + g.predict(te[use])) / 2


def run_monthly(cache: str, batch: str) -> dict:
    """Betfair's own moves, walk-forward by month from ``MONTHLY_FROM``: the base and each arm fitted on the earlier
    months' rows, the month predicted, the change in skill paired by match day (``ra.paired``)."""
    from football import lab
    from football.blocks import available
    from research.open_close.football_stack_betfair_read import betfair_moves, betfair_rows, costed
    tag, kept, new = BATCHES[batch]
    configure(batch)
    m, bfeat = lab.add_blocks(lab.latest(cache), ra.BLOCKS)
    d = lab.prepared(m)
    base = lab.resolve(ra.BASE, lab.families(m), bfeat, d.columns)
    bf = d[betfair_rows(d)].copy()
    mv = betfair_moves(bf)
    bf["yb_h"], bf["yb_a"] = mv["y_h"], mv["y_a"]
    bf["month"] = bf["match_date"].astype(str).str[:7]
    bf = bf[np.isfinite(bf["yb_h"]) & np.isfinite(bf["yb_a"]) & (bf["month"] >= MONTHLY_FROM)]
    months = sorted(bf["month"].unique())
    arms = {"base": base, **{b: list(dict.fromkeys(base + bfeat[b])) for b in new}}
    if len(new) > 1:
        arms["all_new"] = list(dict.fromkeys(base + [c for b in new for c in bfeat[b]]))
    pred = {a: {"h": [], "a": []} for a in arms}
    tested = []
    for mo in months:
        tr, te = bf[bf["month"] < mo], bf[bf["month"] == mo]
        if len(tr) < MONTHLY_MIN_TRAIN or te.empty:
            continue
        for a, cols in arms.items():
            for side in ("h", "a"):
                pred[a][side].append(fit_predict(tr, te, cols, tr[f"yb_{side}"].to_numpy(float)))
        tested.append(te)
    res = {"tag": tag, "design": "betfair_monthly", "base": ra.BASE, "kept_in_base": kept,
           "blocks": {b: list(available()[b].FEATURES) for b in new}, "rows": int(len(bf)),
           "months": months, "arms": {}, "note": "development evidence: Betfair's own moves, walk-forward by month"}
    if not tested:
        res["status"] = "no data found: too few Betfair rows to fit"
        return res
    te = pd.concat(tested)
    res["test_months"] = sorted(te["month"].unique())
    res["test_rows"] = int(len(te))
    day = te["match_date"].astype(str).to_numpy()
    y = {s: te[f"yb_{s}"].to_numpy(float) for s in ("h", "a")}
    se = {a: {s: (y[s] - np.concatenate(pred[a][s])) ** 2 for s in ("h", "a")} for a in arms}
    for a in arms:
        r = {"features": len(arms[a]),
             "betfair": ra.paired(np.r_[day, day], np.r_[se["base"]["h"], se["base"]["a"]],
                                  np.r_[se[a]["h"], se[a]["a"]], np.r_[y["h"] ** 2, y["a"] ** 2])}
        r["keeps_its_place"] = bool(a != "base" and r["betfair"]["lo95"] > 0)
        ph, pa = np.concatenate(pred[a]["h"]), np.concatenate(pred[a]["a"])
        r["costed"] = {str(th): costed(te, ph, pa, th) for th in ra.THRESHOLDS}
        res["arms"][a] = r
    res["coverage"] = {b: {f: {"tested_rows": round(float(te[f].notna().mean()), 4)} for f in bfeat[b]} for b in new}
    return res


def run(cache: str, only: list[str] | None = None, batch: str = "first") -> dict:
    from football.blocks import available
    if batch in MONTHLY:
        return run_monthly(cache, batch)
    configure(batch)
    res = ra.run(cache, only)
    tag, kept, new = BATCHES[batch]
    mods = available()
    res["blocks"] = {b: list(mods[b].FEATURES) for b in new}
    res["kept_in_base"] = kept
    res["coverage"] = coverage(cache, new)
    res["tag"] = tag
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--only", default=None, help="comma-separated arms (the base always runs)")
    ap.add_argument("--batch", default="first", choices=sorted(BATCHES))
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run(a.cache, a.only.split(",") if a.only else None, a.batch)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res.get("arms", res), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
