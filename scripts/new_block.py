#!/usr/bin/env python3
"""Start a drop-in feature block from a working template.

    python scripts/new_block.py going_pref --prefix gp_

writes model/blocks/going_pref.py: one per-horse window (the horse's earlier
days) and one shrunk cell statistic (a course's earlier days), both lag-safe,
which already passes tests/test_feature_blocks.py. Replace the two examples with
the idea, keep the pattern, then

    pytest tests/test_feature_blocks.py -k going_pref       # lag and card rules, seconds
    # research/loop.json: "recipe": "quick", "blocks": "going_pref", push   (~15 min)

See model/blocks/__init__.py for the contract.
"""

from __future__ import annotations

import argparse
import keyword
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TEMPLATE = '''"""{title}

What it measures, in a sentence or two, and why the price might miss it.

Earlier days only: every statistic reads the horse's (or the cell's) earlier
racing days, so a row's own race, and anything else on its day, never enters
its features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.freshness_features import _Days, _col
from model.race_shape import _codes, asof_decayed_mean, day_index, field_size, race_key
from model.shrinkage import shrunk_mean

FEATURES = ["{p}nfp_l1", "{p}course_nfp"]
#: Per-run readings left on the frame that describe the race itself (refused as inputs).
POST_RACE: set[str] = set()


def _num(df, name):
    return pd.to_numeric(_col(df, name), errors="coerce").to_numpy(dtype=float)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    n = field_size(df).to_numpy(dtype=float)
    pos = _num(df, "placing_numerical")
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp = np.clip((n - pos) / np.where(n > 1, n - 1, np.nan), 0.0, 1.0)   # a run's own result

    # 1. a per-horse window: the horse's previous racing day's value
    horse = _codes(df["horse_name"].fillna("").astype(str).str.strip().str.lower())
    H = _Days(horse, day)
    v = H.per_block(nfp)
    blocks = np.arange(len(H.u))
    last = np.where(H.valid & (H.pos >= 1), v[np.clip(blocks - 1, 0, None)], np.nan)

    # 2. a cell statistic: the course's runners over earlier days (half-life a year),
    #    shrunk toward the average runner (0.5) with the weight of 50 runs
    course = _codes(_col(df, "track").fillna("").astype(str).str.lower())
    mean, n_eff = asof_decayed_mean(course, day, nfp, course, day, halflife_days=365.0)
    course_nfp = shrunk_mean(np.nan_to_num(mean) * n_eff, n_eff, 0.5, 50.0)

    new = pd.DataFrame({{"{p}nfp_l1": H.to_rows(last), "{p}course_nfp": course_nfp}}, index=df.index)
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new[FEATURES]], axis=1)
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="module name, e.g. going_pref")
    ap.add_argument("--prefix", default=None, help="feature prefix (default: the name's initials + _)")
    ap.add_argument("--title", default=None, help="the docstring's first line")
    args = ap.parse_args()
    name = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or keyword.iskeyword(name):
        raise SystemExit(f"{name!r}: lower-case letters, digits and underscores, starting with a letter")
    path = REPO / "model" / "blocks" / f"{name}.py"
    if path.exists():
        raise SystemExit(f"{path.relative_to(REPO)} exists")
    prefix = args.prefix or "".join(w[0] for w in name.split("_")) + "x_"
    title = args.title or f"{name.replace('_', ' ').capitalize()}: a drop-in feature block."
    path.write_text(TEMPLATE.format(title=title, p=prefix))
    print(f"wrote {path.relative_to(REPO)} (features {prefix}*)")
    print(f"next:  pytest tests/test_feature_blocks.py -k {name}")
    print(f'then:  research/loop.json  "recipe": "quick", "blocks": "{name}"  and push')


if __name__ == "__main__":
    sys.exit(main())
