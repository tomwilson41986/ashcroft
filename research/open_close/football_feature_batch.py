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

from research.open_close import football_review_arms as ra

KEPT = ["league_fx", "move_memory", "kalman_form"]
NEW = ["mkt_xg", "decay_form", "sos_split", "venue_fx", "norm_rank", "style", "referee", "sim"]
#: {batch: (ledger tag, blocks in the base, new blocks)}. ``tactics`` (the owner, 24 Nov: "build the match simulator
#: and formation matchups"): the base also carries the two blocks of the first batch that held on both books
BATCHES = {"first": ("football-feature-batch-1124", KEPT, NEW),
           "tactics": ("football-tactics-batch-1124", KEPT + ["sim", "style"], ["formation", "match_sim"]),
           "sportmonks": ("football-sportmonks-xg-1124", KEPT + ["sim", "style"], ["sm_xg"]),
           "fd_lines": ("football-fd-lines-1124", KEPT + ["sim", "style"], ["fd_lines"]),
           "sm_pred": ("football-sportmonks-pred-1124", KEPT + ["sim", "style"], ["sm_pred"])}
#: {batch: (first training season, (first, last) Pinnacle test season)}: Sportmonks' xG starts in 2024/25, so its
#: batch fits both arms on 2024/25 alone and tests 2025/26 (the other batches keep the review arms' settings)
WINDOWS = {"sportmonks": (2024, (2025, 2025)), "sm_pred": (2024, (2025, 2025))}
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


def run(cache: str, only: list[str] | None = None, batch: str = "first") -> dict:
    from football.blocks import available
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
