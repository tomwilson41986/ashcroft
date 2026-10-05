#!/usr/bin/env python3
"""Sale prices from Timeform's horse files -> data/external/timeform_sales.csv.gz, for the sales block.

Timeform's production comment opens with the horse's sales before it ran: "€42,000Y",
"180,000Y, €400,000 2-y-o", "11,000F", and over jumps "€8,500 3-y-o" (a store). The
Blandford scraper keeps two snapshots of the horse file (daily-horse-scraper:
Horses_Data_Part1-4.csv, about Feb 2025, every horse; Horses_Data_03072025_part1/2.csv,
3 Jul 2025, foaled 2018 on). The later snapshot wins for a horse in both.

Only sales that come before a horse races are kept: foal, yearling and two-year-old
(breeze-up) sales, and three- and four-year-old stores from the jumps comment. A Flat
three-year-old sale, and any later one, is usually a horses-in-training sale priced on
form the horse had already shown; it would carry later results back into its earlier
races, so it is left out. Only the opening clause (before the first colon) is read: the
rest of the comment describes the horse's racing.

One row per (name, foaling year); a name and year two horses share is dropped, both.

    python scripts/build_timeform_sales.py --horses /home/user/daily-horse-scraper/Horses_Data_*.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from model.blocks.sales import norm_name  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "external", "timeform_sales.csv.gz")
# Pounds per unit, round 2024-25 rates; a guinea is 1.05 pounds by definition.
FX = {"gns": 1.05, "£": 1.0, "€": 0.85, "$": 0.78}
SALE = re.compile(r"(?P<cur>[€£$])?\s?(?P<amt>\d{1,3}(?:,\d{3})+|\d{3,})\s?(?:gns\s?)?(?P<kind>Y\b|F\b|(?P<age>[1-9])-y-o)")
KINDS = {"foal": 0, "yearling": 1, "breeze": 2, "store": 3}


def sales_in(comment, jumps: bool) -> dict:
    """{kind: price in pounds} for the sales a comment's opening clause names before racing (last one wins)."""
    if not isinstance(comment, str):
        return {}
    out = {}
    for m in SALE.finditer(comment.split(":", 1)[0]):
        k = m.group("kind")
        if k == "F":
            kind = "foal"
        elif k == "Y":
            kind = "yearling"
        else:
            age = int(m.group("age"))
            kind = "breeze" if age <= 2 else ("store" if jumps and age <= 4 else None)
        if kind is None:
            continue
        cur = m.group("cur") or "gns"            # Timeform writes Tattersalls prices bare, in guineas
        out[kind] = float(m.group("amt").replace(",", "")) * FX[cur]
        out["_last"] = kind
    return out


def build(paths: list[str]) -> pd.DataFrame:
    cols = ["horseCode", "horseName", "foalingDate", "productionCommentFlat", "productionCommentJump"]
    # The later snapshot last, so it wins a horse in both
    frames = [pd.read_csv(p, usecols=cols, dtype=str, low_memory=False) for p in sorted(paths, key=_snapshot_rank)]
    hd = pd.concat(frames, ignore_index=True)
    hd["horseCode"] = hd["horseCode"].str.lstrip("0")
    hd = hd.drop_duplicates("horseCode", keep="last")
    hd["foal_year"] = pd.to_datetime(hd["foalingDate"], errors="coerce").dt.year
    hd = hd[hd["foal_year"].between(2003, 2030)].copy()
    rows = []
    for name, fy, flat, jump in hd[["horseName", "foal_year", "productionCommentFlat", "productionCommentJump"]] \
            .itertuples(index=False):
        s = sales_in(flat, jumps=False)
        s.pop("_last", None)
        for k, v in sales_in(jump, jumps=True).items():
            if k != "_last" and (k not in s or k == "store"):
                s[k] = v
        # The last sale is the latest stage the horse was sold at: foal, yearling, breeze-up, store
        last = max(s, key=KINDS.get) if s else None
        rows.append((norm_name(name), int(fy), s.get("foal"), s.get("yearling"), s.get("breeze"), s.get("store"),
                     s.get(last) if last else None, KINDS.get(last, np.nan),
                     int(isinstance(flat, str) or isinstance(jump, str))))
    t = pd.DataFrame(rows, columns=["name_norm", "foal_year", "foal_gbp", "yearling_gbp", "breeze_gbp", "store_gbp",
                                    "last_gbp", "last_kind", "has_comment"])
    t = t[t["name_norm"] != ""]
    shared = t.duplicated(["name_norm", "foal_year"], keep=False)
    return t[~shared].sort_values(["name_norm", "foal_year"]).reset_index(drop=True)


def _snapshot_rank(path: str) -> int:
    return 1 if "03072025" in os.path.basename(path) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horses", nargs="+", required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    t = build(args.horses)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t.to_csv(args.out, index=False, float_format="%.0f", compression="gzip")
    sold = t["last_gbp"].notna()
    print(f"{len(t):,} horses foaled 2003 on, {sold.sum():,} sold before racing "
          f"(foal {t['foal_gbp'].notna().sum():,}, yearling {t['yearling_gbp'].notna().sum():,}, "
          f"breeze-up {t['breeze_gbp'].notna().sum():,}, store {t['store_gbp'].notna().sum():,}); "
          f"median last price GBP {t.loc[sold, 'last_gbp'].median():,.0f} -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
