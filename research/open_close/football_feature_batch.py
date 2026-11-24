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


def configure() -> None:
    """Point the review-arms runner at this batch (its method is not changed)."""
    ra.BASE = "+".join([ra.BASE.split("+league_fx")[0]] + KEPT)
    ra.BLOCKS = ["home_adv", "form_short"] + KEPT + NEW
    ra.ARMS = {"base": {}, **{b: {"add": b} for b in NEW}, "all_new": {"add": list(NEW)}}
    ra.COSTED = ("base", "all_new")


def run(cache: str, only: list[str] | None = None) -> dict:
    from football.blocks import available
    configure()
    res = ra.run(cache, only)
    mods = available()
    res["blocks"] = {b: list(mods[b].FEATURES) for b in NEW}
    res["kept_in_base"] = KEPT
    res["tag"] = "football-feature-batch-1124"
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
