#!/usr/bin/env python3
"""Does what a horse sold for at auction know something the market does not?

The sale price comes from Timeform's production comment, the paragraph that
opens with the horse's sales history ("€42,000Y", "180,000Y, €400,000 2-y-o",
"11,000F"): the horse master files the Blandford scraper keeps
(daily-horse-scraper/Horses_Data_03072025_part1/2.csv, horses foaled 2018 on).
Only the opening clause is read, the sales before the horse ran; the rest of
the comment describes its racing and is not used.

The race sample is where public form is thinnest and the served model's error
largest: Flat and all-weather maidens and novice races, August 2024 to March
2026 (nothing from the locked holdout). Runners are young horses (foaled 2022
on, so every run they made is in the window) with at most two earlier runs.
The tests are the sectionals screen's: the winner beyond Betfair SP (and beyond
our model's probability), and the within-race error of our log-BSP forecast.

Usage
    python scripts/external_sales_screen.py \\
        --horses /home/user/daily-horse-scraper/Horses_Data_03072025_part1.csv \\
                 /home/user/daily-horse-scraper/Horses_Data_03072025_part2.csv \\
        --oos data/oos_predictions.csv --out reports/external_sales_screen.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from betfair_prices import normalise_horse  # noqa: E402
from external_sectionals_screen import build_sample, design_present, parts_for  # noqa: E402
from residual_screen import MoveScreen, OutcomeScreen, gain_summary  # noqa: E402

# Pounds per unit, round 2024-25 rates: a guinea is 1.05 pounds by definition.
FX = {"gns": 1.05, "£": 1.0, "€": 0.85, "$": 0.78}
SALE = re.compile(r"(?P<cur>[€£$])?\s?(?P<amt>\d{1,3}(?:,\d{3})+|\d{3,})\s?(?P<gns>gns)?\s?"
                  r"(?P<kind>Y\b|F\b|2-y-o|3-y-o)")


def parse_sale(comment) -> tuple[float, str] | tuple[None, None]:
    """The last sale named in the opening clause (before the first colon): its price in pounds and kind."""
    if not isinstance(comment, str):
        return None, None
    head = comment.split(":", 1)[0]
    found = list(SALE.finditer(head))
    if not found:
        return None, None
    m = found[-1]
    amt = float(m.group("amt").replace(",", ""))
    cur = m.group("cur") or "gns"            # Timeform writes Tattersalls prices bare, in guineas
    kind = {"Y": "yearling", "F": "foal", "2-y-o": "2yo", "3-y-o": "3yo"}[m.group("kind")]
    return amt * FX[cur], kind


def load_sales(paths: list[str]) -> pd.DataFrame:
    hd = pd.concat([pd.read_csv(p, dtype=str, low_memory=False) for p in paths], ignore_index=True)
    hd["foal_year"] = pd.to_datetime(hd["foalingDate"], errors="coerce").dt.year
    hd = hd[hd["foal_year"] >= 2022].copy()
    parsed = hd["productionCommentFlat"].map(parse_sale)
    hd["price_gbp"] = [p[0] for p in parsed]
    hd["sale_kind"] = [p[1] for p in parsed]
    hd["horse_norm"] = hd["horseName"].map(normalise_horse)
    hd["has_comment"] = hd["productionCommentFlat"].notna()
    dup = hd["horse_norm"].duplicated(keep=False)          # two young horses, one name: read neither
    return hd.loc[~dup, ["horse_norm", "foal_year", "price_gbp", "sale_kind", "has_comment"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--horses", nargs="+", required=True)
    ap.add_argument("--oos", default="data/oos_predictions.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sales = load_sales(args.horses)
    s = build_sample(args.oos)
    s["prior_runs"] = s.groupby("horse_norm").cumcount()     # runs in the window before this one (sorted by date)
    s = s.merge(sales, on="horse_norm", how="left")
    s = s[(s["race_type"].isin(["Maiden", "Novices"])) & (s["date"] >= "2024-08-01")].copy()
    young = (s["foal_year"] >= 2022) & (s["prior_runs"] <= 2)
    cov = young.groupby(s["race"]).transform("mean") >= 0.8
    s = s[cov].sort_values(["date", "race"], kind="stable").reset_index(drop=True)

    sold = s["price_gbp"].notna()
    lp = np.log(s["price_gbp"].where(sold))
    s["ln_price"] = lp
    s["ln_price_rel"] = lp - lp.groupby(s["race"]).transform("mean")
    s["price_rank"] = lp.groupby(s["race"]).rank(pct=True)
    s["sold"] = sold.astype(float)
    s["breeze_up"] = (s["sale_kind"] == "2yo").astype(float)
    s["debut"] = (s["prior_runs"] == 0).astype(float)
    s["debut_x_price_rel"] = s["debut"] * s["ln_price_rel"].fillna(0)
    feats = ["ln_price", "ln_price_rel", "price_rank", "sold", "breeze_up", "debut_x_price_rel"]
    rep = {"rows": int(len(s)), "races": int(s["race"].nunique()), "share_sold": float(sold.mean()),
           "share_in_horse_file": float(s["foal_year"].notna().mean()),
           "median_price_gbp": float(s["price_gbp"].median())}

    month = s["date"].dt.year.to_numpy() * 12 + s["date"].dt.month.to_numpy()
    even, odd = month % 2 == 0, month % 2 == 1
    tr_fwd = (s["date"] < "2025-06-01").to_numpy()
    vals = {c: s[c].to_numpy(float) for c in feats}
    for split_name, splits, trm in (("parity", [parts_for(s, even, odd), parts_for(s, odd, even)], even | odd),
                                    ("forward", [parts_for(s, tr_fwd, ~tr_fwd)], tr_fwd)):
        for base_name, base_extra in (("market", None), ("market+model", s["ln_model"].to_numpy(float))):
            scr = OutcomeScreen(splits)
            res = {"market_R2": scr.market_r2(), "per_feature": {}}
            for f in feats:
                F = design_present(vals[f], trm, quadratic=f not in ("sold", "breeze_up"))
                res["per_feature"][f] = gain_summary(scr.gains(F, base_extra=base_extra), scr.null)
            F = np.column_stack([design_present(vals[f], trm, quadratic=False) for f in feats])
            best = None
            for ridge in (1.0, 10.0, 100.0):
                g = gain_summary(scr.gains(F, ridge=ridge, base_extra=base_extra), scr.null)
                g["ridge"] = ridge
                best = g if best is None or g["dll_mnats"] > best["dll_mnats"] else best
            res["joint"] = best
            rep[f"{split_name}|{base_name}"] = res
        fsplits = ([parts_for(s, even, odd, "ln_pred_bsp"), parts_for(s, odd, even, "ln_pred_bsp")]
                   if split_name == "parity" else [parts_for(s, tr_fwd, ~tr_fwd, "ln_pred_bsp")])
        ms = MoveScreen(fsplits, s["ln_bsp"].to_numpy(float))
        F = np.column_stack([design_present(vals[f], trm, quadratic=False) for f in feats])
        g = ms.gains(F, ridge=10.0)
        base_sse = sum(float(b.sum()) for b, _ in ms.base)
        rep[f"{split_name}|forecast"] = {"sse_cut_pct": 100 * float(g.sum()) / base_sse,
                                         "t": float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))}

    txt = json.dumps(rep, indent=1, default=float)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
