"""The greyhound data audit (the owner, 18 Nov 2026: the improve-validate-test brief, Phase 0; read-only).

Per source and per year, every number from the stored tables, nothing estimated:

1. **GBGB results** (``gbgb/runs_<year>.parquet``, the training data): rows in; duplicate (race_id, dog_id) and
   (race_id, trap); nulls in the key fields; the date range; then ``greyhound.data.clean``'s drops by reason (no
   dog/race id or date, no position on a non-card run, no parseable off time, a race with no winner - void), the rows
   out, trials, non-finishers, the field-size spread (short fields: vacant traps), and handicaps.
2. **Betfair price files** (``greyhound_prices/prices_<year>.parquet``, GB win): rows, markets, BSP null share,
   duplicates on (date, off, track, trap); the GBGB runs matched to a BSP (``greyhound.parity.attach_bsp``) and the
   races whose whole field is priced (the model's target set).
3. **Recorded books** (``betfair_live/<day>/books_greyhound.csv.gz``, the recorder, delayed key; and
   ``books_greyhound_live.csv.gz``, the live key): per day rows, markets, removed runners (non-runners: Betfair's
   reduction factor applies, Rule 4's exchange form), settled markets with a BSP.
4. **Prices served** (``greyhound/live/predictions/<day>.csv``) and the out-of-sample prices of the live recipe: days
   held.

    python -m research.greyhound.data_audit --years 2014-2026 --out out/audit
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEY = ["dog_id", "race_id", "race_date", "race_time", "track", "trap", "position", "sp_decimal"]


def gbgb_year(raw: pd.DataFrame) -> dict:
    from greyhound.data import clean
    out = {"rows_in": int(len(raw)), "races_in": int(raw.race_id.nunique()),
           "date_min": str(raw.race_date.min()), "date_max": str(raw.race_date.max()),
           "dup_race_dog": int(raw.duplicated(["race_id", "dog_id"]).sum()),
           "dup_race_trap": int(raw.dropna(subset=["trap"]).duplicated(["race_id", "trap"]).sum()),
           "nulls": {c: int(raw[c].isna().sum()) for c in KEY if c in raw}}
    # clean()'s drops, step by step as it makes them
    card = raw["is_card"].fillna(False).astype(bool) if "is_card" in raw else pd.Series(False, index=raw.index)
    no_id = raw.dog_id.isna() | raw.race_id.isna() | raw.race_date.isna()
    no_pos = ~no_id & raw.position.isna() & ~card
    a = raw[~no_id & ~no_pos]
    t = pd.to_datetime(a.race_date.astype(str) + " " + a.race_time.astype(str).str[:5], errors="coerce")
    no_time = int(t.isna().sum())
    a = a[t.notna()]
    won = (a.position == 1).groupby(a.race_id).transform("any")
    void_rows, void_races = int((~won).sum()), int(a.loc[~won, "race_id"].nunique())
    c = clean(raw)
    out.update({"drop_no_id_or_date": int(no_id.sum()), "drop_no_position": int(no_pos.sum()),
                "drop_no_off_time": no_time, "drop_void_race_rows": void_rows, "drop_void_races": void_races,
                "rows_out": int(len(c)), "races_out": int(c.race_id.nunique()),
                "trial_rows": int(c.is_trial.sum()), "non_finisher_rows": int(c.dnf.sum()),
                "handicap_rows": int(c.handicap.sum()),
                "field_size": {str(k): int(v) for k, v in c.drop_duplicates("race_id").field.value_counts().sort_index().items()},
                "sp_missing_rows": int(c.sp_decimal.isna().sum()) if "sp_decimal" in c else None})
    return out, c


def prices_year(p: pd.DataFrame | None, runs: pd.DataFrame) -> dict:
    from greyhound.parity import attach_bsp
    if p is None or not len(p):
        return {"note": "no data found: no price file for this year"}
    w = p[(p.market == "win") & (p.country == "GB")]
    out = {"rows": int(len(w)), "markets": int(w.drop_duplicates(["race_date", "race_time", "track"]).shape[0]),
           "bsp_null_share": round(float(w.bsp.isna().mean()), 4),
           "dup_date_off_track_trap": int(w.dropna(subset=["trap"]).duplicated(["race_date", "race_time", "track", "trap"]).sum())}
    r = attach_bsp(runs[~runs.is_trial], w)
    out.update({"gbgb_runs_non_trial": int(len(r)), "runs_with_bsp": int((r.bsp > 1).sum()),
                "runs_bsp_share": round(float((r.bsp > 1).mean()), 4) if len(r) else None,
                "races_fully_priced": int(r[r.bsp_p_norm.notna()].race_id.nunique()),
                "races": int(r.race_id.nunique())})
    return out


def books_days(store) -> dict:
    """Every recorded day's book files, read from S3 (betfair_recorder's day prefix)."""
    from betfair_recorder import PREFIX, bucket, s3_client
    s3 = s3_client()
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=f"{PREFIX}/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    out = {}
    for name in ("books_greyhound.csv.gz", "books_greyhound_live.csv.gz"):
        rows = []
        for k in sorted(x for x in keys if x.endswith("/" + name)):
            day = k.split("/")[-2]
            try:
                b = pd.read_csv(io.BytesIO(gzip.decompress(s3.get_object(Bucket=bucket(), Key=k)["Body"].read())),
                                dtype={"market_id": str}, on_bad_lines="skip", low_memory=False)
            except Exception as exc:
                rows.append({"day": day, "note": f"unreadable ({exc})"[:200]})
                continue
            fin = b[b.source.astype(str) == "final"] if "source" in b else b.iloc[0:0]
            st = b.runner_status.astype(str) if "runner_status" in b else pd.Series("", index=b.index)
            rows.append({"day": day, "rows": int(len(b)), "markets": int(b.market_id.nunique()),
                         "removed_runners": int(b.loc[st == "REMOVED", ["market_id", "selection_id"]].drop_duplicates().shape[0]),
                         "settled_markets": int(fin.market_id.nunique()),
                         "settled_with_bsp": int(fin[pd.to_numeric(fin.get("bsp"), errors="coerce") > 1].market_id.nunique())
                         if "bsp" in fin else 0})
        out[name] = {"days": len(rows), "first": rows[0]["day"] if rows else None,
                     "last": rows[-1]["day"] if rows else None, "per_day": rows}
    return out


def main(argv=None) -> int:
    from greyhound.model import load_prices
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", default="2014-2026")
    ap.add_argument("--out", default="out/audit")
    a = ap.parse_args(argv)
    store = Store()
    lo, hi = (int(x) for x in a.years.split("-"))
    res = {"gbgb": {}, "prices": {}}
    held = sorted(int(re.search(r"(\d{4})", k).group(1)) for k in store.listing("gbgb/")
                  if re.search(r"gbgb/runs_\d{4}\.parquet$", k))
    for y in range(lo, hi + 1):
        if y not in held:
            res["gbgb"][str(y)] = {"note": "no data found: no GBGB file"}
            continue
        raw = pd.read_parquet(io.BytesIO(store.get(f"gbgb/runs_{y}.parquet")))
        res["gbgb"][str(y)], runs = gbgb_year(raw)
        res["prices"][str(y)] = prices_year(load_prices(store, f"{y}-{y}"), runs)
        log.info("%s: %s | %s", y, {k: v for k, v in res["gbgb"][str(y)].items() if k != "field_size"},
                 res["prices"][str(y)])
        del raw, runs
    res["books"] = books_days(store)
    pred = sorted(k.rsplit("/", 1)[-1][:10] for k in store.listing("greyhound/live/predictions/") if k.endswith(".csv"))
    res["served_predictions"] = {"days": len(pred), "first": pred[0] if pred else None, "last": pred[-1] if pred else None}
    for pfx in ("greyhound/live/oos_prices_stack_mlsitd", "greyhound/live/oos_prices_stack_mlsitd_mem"):
        d = sorted(k.rsplit("/", 1)[-1][:10] for k in store.listing(f"{pfx}/") if k.endswith(".parquet"))
        res[pfx] = {"days": len(d), "first": d[0] if d else None, "last": d[-1] if d else None}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "books"}, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_day"} for k, v in res["books"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
