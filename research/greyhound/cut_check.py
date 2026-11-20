"""Does any feature read a later day? The cache against a build cut short (the owner, 20 Nov 2026: the discovery
round, build 6; ledger greyhound-cut-check-1120). Descriptive, 0 trials; read-only.

Build 6's residual on the BSP returned 5% at the SP on years it never saw: too good to accept without proof that no
feature of a race knows anything from after it. ``greyhound.live --check-card`` proves a day's features do not read
that day's results (the card build against the results build). This proves the other half: the fast-loop cache is one
build through its last day; here the engine is run again on the runs and prices up to a cut day only, and every
card-safe feature of the cut's last ``days`` days is compared, runner by runner, with the cache's value for the same
runner. A feature that agrees was computed from the cut and earlier alone.

    python -m research.greyhound.cut_check --cut 2025-06-30 --days 30 --out out/study
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-cut-check-1120"
TOL = 1e-6


def compare(cache: pd.DataFrame, cut: pd.DataFrame, feats: list[str]) -> dict:
    """Each feature's share of runners where the cache and the cut build agree (both missing, or within TOL)."""
    k = ["race_id", "dog_id"]
    j = cache[k + feats].merge(cut[k + feats], on=k, suffixes=("_cache", "_cut"))
    agree, worst = {}, {}
    for f in feats:
        x = pd.to_numeric(j[f"{f}_cache"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(j[f"{f}_cut"], errors="coerce").to_numpy(float)
        same = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= TOL * np.maximum(1.0, np.abs(x)))
        agree[f] = round(float(same.mean()), 5) if len(j) else None
        if len(j) and not same.all():
            d = np.abs(x - y)[~same]
            worst[f] = {"agree": agree[f], "differ": int((~same).sum()),
                        "median_abs_gap": None if not np.isfinite(d).any() else round(float(np.nanmedian(d)), 6)}
    bad = dict(sorted(worst.items(), key=lambda kv: kv[1]["agree"]))
    return {"runners_compared": int(len(j)), "cache_rows": int(len(cache)), "cut_rows": int(len(cut)),
            "features": len(feats), "agree_all": int(sum(1 for v in agree.values() if v == 1.0)),
            "agree_99": int(sum(1 for v in agree.values() if v is not None and v >= 0.99)), "differ": bad}


def main(argv=None) -> int:
    from greyhound import fastloop as fl
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices, modelling_rows, slim
    from greyhound.track import runs_through
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cut", default="2025-06-30")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="out/cut_check")
    a = ap.parse_args(argv)
    store = Store()
    cut = date.fromisoformat(a.cut)
    lo = f"{cut - timedelta(days=a.days - 1)}"
    df, meta = fl.load_cache(store)
    feats = [f for f in fl.feature_set(meta, "card_safe") if f in df.columns and f not in fl.LABELS]
    day = pd.to_datetime(df.t).dt.strftime("%Y-%m-%d")
    cache = df[(day >= lo) & (day <= a.cut)][["race_id", "dog_id"] + feats].copy()
    del df
    log.info("cache %s: %d rows %s..%s, %d card-safe features", meta.get("hash"), len(cache), lo, a.cut, len(feats))
    runs = runs_through(store, cut)
    runs = runs[runs.race_date.astype(str) <= a.cut]                       # nothing after the cut day reaches the engine
    prices = load_prices(store, fl.PRICE_YEARS)
    prices = prices[prices.race_date.astype(str) <= a.cut]
    eng = GreyhoundMetricsEngine(blocks=fl.BLOCKS)
    b = eng.calculate_all(runs, prices=prices)
    b = modelling_rows(slim(b, eng.features + list(fl.EXTRA_COLS)))
    bday = pd.to_datetime(b.t).dt.strftime("%Y-%m-%d")
    b = b[(bday >= lo) & (bday <= a.cut)]
    res = compare(cache, b, [f for f in feats if f in b.columns])
    res.update({"tag": TAG, "cut": a.cut, "from": lo, "cache_hash": meta.get("hash"), "cache_last_day": meta.get("last_day"),
                "features_missing_from_the_cut_build": [f for f in feats if f not in b.columns]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    md = [f"## The cache against a build cut at {a.cut} ({TAG})", "",
          f"{res['runners_compared']:,} runners of {lo} to {a.cut} compared on {res['features']} card-safe features: "
          f"{res['agree_all']} agree on every runner, {res['agree_99']} on at least 99%.", ""]
    if res["differ"]:
        md += ["| Feature | Agree | Runners that differ | Median gap |", "|---|---|---|---|"]
        md += [f"| {f} | {v['agree']} | {v['differ']} | {v['median_abs_gap']} |" for f, v in list(res["differ"].items())[:60]]
    else:
        md += ["No feature differs: every value in the window was computed from the cut day and earlier."]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
