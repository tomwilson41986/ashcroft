"""Today's predictions priced from Betfair's card against the morning's from HRB's (card-fallback-check.yml; read-only).

Joins s3 <prefix>/<day>.csv to predictions/<day>.csv on track, race time and the horse's name without its country
suffix, and prints: races and runners on each, the predicted BFSP's log difference (median and 90th percentile of
its size, the share within 5% and 10%), the rank correlation within races, and the races whose favourite differs.

Pass for the fallback (set before the run): runners on both at least 95% of HRB's; the median absolute log difference
at most 0.05; the median rank correlation within races at least 0.90.
"""
from __future__ import annotations

import argparse
import io
import os
import re
from datetime import date

import numpy as np
import pandas as pd


def _norm(s) -> str:
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\s*\((?:[A-Z]{2,3})\)\s*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def compare(hrb: pd.DataFrame, bf: pd.DataFrame) -> dict:
    for d in (hrb, bf):
        d["race"] = d.venue.map(_norm) + "|" + d.race_time.astype(str).str.strip()
        d["k"] = d.race + "|" + d.runner_name.map(_norm)
    both = hrb.merge(bf, on="k", suffixes=("_hrb", "_bf"))
    both = both[(both.predicted_bfsp_hrb > 1) & (both.predicted_bfsp_bf > 1)]
    lo = np.log(both.predicted_bfsp_bf / both.predicted_bfsp_hrb)
    rho = both.groupby("race_hrb").apply(
        lambda g: g.predicted_bfsp_hrb.rank().corr(g.predicted_bfsp_bf.rank()) if len(g) > 2 else np.nan)
    fav = both.groupby("race_hrb").apply(
        lambda g: g.loc[g.predicted_bfsp_hrb.idxmin(), "k"] == g.loc[g.predicted_bfsp_bf.idxmin(), "k"])
    out = {"races_hrb": hrb.race.nunique(), "races_bf": bf.race.nunique(), "runners_hrb": len(hrb),
           "runners_bf": len(bf), "runners_both": len(both), "share_both": len(both) / max(len(hrb), 1),
           "mad_log": float(lo.abs().median()), "p90_abs_log": float(lo.abs().quantile(0.9)),
           "mean_log": float(lo.mean()), "within_5pc": float((lo.abs() <= np.log(1.05)).mean()),
           "within_10pc": float((lo.abs() <= np.log(1.10)).mean()), "rank_corr_median": float(rho.median()),
           "same_favourite": float(fav.mean())}
    out["pass"] = bool(out["share_both"] >= 0.95 and out["mad_log"] <= 0.05 and out["rank_corr_median"] >= 0.90)
    worst = both.assign(lo=lo).reindex(lo.abs().sort_values(ascending=False).index).head(15)
    out["_worst"] = worst[["venue_hrb", "race_time_hrb", "runner_name_hrb", "predicted_bfsp_hrb",
                           "predicted_bfsp_bf", "lo"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="predictions_betfair_card")
    ap.add_argument("--day", default=date.today().isoformat())
    args = ap.parse_args()
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION") or "eu-west-2")
    bucket = os.environ.get("ULTRA_BETTING_S3_BUCKET") or "ashcroft"

    def read(prefix):
        raw = s3.get_object(Bucket=bucket, Key=f"{prefix}/{args.day}.csv")["Body"].read()
        return pd.read_csv(io.BytesIO(raw), dtype={"race_time": str})

    r = compare(read("predictions"), read(args.prefix))
    worst = r.pop("_worst")
    print(f"## Card fallback check, {args.day}: predictions/ (HRB's card) against {args.prefix}/ (Betfair's)\n")
    print("```")
    for k, v in r.items():
        print(f"{k:18s} {v:.4f}" if isinstance(v, float) else f"{k:18s} {v}")
    print("\nthe 15 largest differences:")
    print(worst.round(3).to_string(index=False))
    print("```")


if __name__ == "__main__":
    main()
