"""The shadow price: the morning's card priced again by the previous served blend (the owner's review of 20 Nov 2026,
item M1; the owner, 21 Nov: "continue all research and development").

When the served model changes, the question "is the new one doing worse?" can only be answered if the old one priced
the same cards. data/models/shadow.json names the commit whose data/models held the previous blend and the last day to
shadow it:

    {"commit": "<sha>", "until": "2026-12-05", "note": "..."}

Each morning, after the 03:10 UTC predictions (predict-shadow.yml):

1. read the card the morning's run kept, as fetched (s3 racecards/<day>_<HHMM>.csv, the earliest of the day), so
   both blends price exactly the same runners, and nothing asks horseracebase.com again;
2. load the previous blend from that commit (git show into a folder; the gate and members as they were);
3. price the card with it on the same code path as the live run (predict_bfsp_today.prepare_and_predict, history from
   its own training start), and write s3 predictions_shadow/<day>.csv in the live file's columns.

Read-only: nothing here writes predictions/, which the trader reads. No shadow.json, or a day past `until`: nothing
to do, exit 0. The comparison is research/queries (each day's two files against the BSP of Betfair's own record).

The two files are paired only when both priced the same database. horse_racing.db is replaced in S3 by its writers
(blandford-sync.yml at 06:45 UTC, daily-results.yml at 21:30, blandford-sectionals.yml at 00:20) and the bucket keeps
no versions, so the 04:25 run reads what the 03:00 run read, and a run after 06:45 does not (the first check, 21 Nov
2026, run at 09:30: 18 of 46 races identical, the rest moved by the 06:47 upload). Each row carries the database's
S3 LastModified (`db_modified`) and the trader's file's (`live_modified`); a database newer than the trader's file is
logged as not paired.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
log = logging.getLogger("shadow_predict")
MODEL_FILES = ("bfsp_model.lgb", "bfsp_model.lgb.xz", "bfsp_model_meta.json", "bfsp_training_summary.json",
               "bfsp_ensemble.json")
MEMBERS = ("", "members/dml", "members/rx")


def config(path: Path = ROOT / "data" / "models" / "shadow.json") -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def active(cfg: dict | None, day: date) -> bool:
    return bool(cfg and cfg.get("commit") and day <= date.fromisoformat(str(cfg.get("until", "1970-01-01"))))


def materialise(commit: str, out: Path) -> Path:
    """The previous blend's files at `commit`, written under `out` as data/models lays them out."""
    subprocess.run(["git", "fetch", "--quiet", "--depth", "1", "origin", commit], cwd=ROOT, check=False)
    written = 0
    for sub in MEMBERS:
        for name in MODEL_FILES:
            rel = f"data/models/{sub + '/' if sub else ''}{name}"
            r = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True)
            if r.returncode != 0:
                continue
            dest = out / (sub or ".") / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.stdout)
            written += 1
    if written == 0:
        raise SystemExit(f"no model files found at {commit}")
    log.info("previous blend from %s: %d files into %s", commit[:10], written, out)
    return out


def morning_card(day: date):
    """The earliest card the morning's run kept for `day` (racecards/<day>_<HHMM>.csv), as the run fetched it."""
    import boto3
    import pandas as pd
    from ultra_betting.config import S3_BUCKET
    s3 = boto3.client("s3")
    keys = sorted(o["Key"] for o in s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"racecards/{day}_").get("Contents", []))
    if not keys:
        return None, None
    body = s3.get_object(Bucket=S3_BUCKET, Key=keys[0])["Body"].read()
    card = pd.read_csv(io.BytesIO(body), dtype={"race_time": str, "time": str})
    return card, keys[0]


def price(model_dir: Path, card, day: date):
    from predict_bfsp_today import load_bfsp_model, load_historical, prepare_and_predict, with_model_rank
    from ultra_betting.config import DB_PATH
    from ultra_betting.model.predict import training_start
    model, cols, vocab = load_bfsp_model(str(model_dir))
    hist = load_historical(str(DB_PATH), start_date=training_start(model_dir))
    before = hist[hist["race_date"].dt.date < day].copy()
    del hist
    return with_model_rank(prepare_and_predict(before, align(card, before), model, cols, day, vocab))


def align(card, history):
    """The card as the live run holds it in memory: a column the history holds as numbers made numbers (a CSV read
    gives text where a value was blank or odd, and one text column on the card turns the history's column to Python
    objects when the two are joined: a million boxed values, the first shadow run's lost runner, 21 Nov 2026)."""
    import pandas as pd
    out = card.copy()
    for c in out.columns:
        if c in history.columns and c != "race_date":
            if pd.api.types.is_numeric_dtype(history[c]) and not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def s3_modified(bucket: str, key: str) -> str:
    """The object's S3 LastModified (UTC, ISO), or '' when it cannot be read."""
    import boto3
    try:
        return boto3.client("s3").head_object(Bucket=bucket, Key=key)["LastModified"].isoformat()
    except Exception as e:  # noqa: BLE001  a record, never a reason to stop
        log.warning("no LastModified for s3://%s/%s (%s)", bucket, key, type(e).__name__)
        return ""


def paired(db_modified: str, live_modified: str) -> bool:
    """True when the database was last written before the trader's file: both priced the same database."""
    return bool(db_modified and live_modified and db_modified <= live_modified)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default=None)
    ap.add_argument("--models", default=str(ROOT / ".shadow_models"))
    ap.add_argument("--dry-run", action="store_true", help="price, print the summary, write nothing to S3")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    day = date.fromisoformat(args.date) if args.date else date.today()
    cfg = config()
    if not active(cfg, day):
        log.info("no shadow for %s (data/models/shadow.json: %s)", day, cfg)
        return 0
    card, key = morning_card(day)
    if card is None:
        log.warning("no data found: no morning card kept for %s (s3 racecards/%s_*)", day, day)
        return 0
    log.info("the morning card %s: %d runners", key, len(card))
    from pipeline.shared import ensure_database
    from ultra_betting.config import S3_BUCKET, S3_DB_BUCKET, S3_DB_KEY
    db_modified = s3_modified(S3_DB_BUCKET, S3_DB_KEY)
    live_modified = s3_modified(S3_BUCKET, f"predictions/{day}.csv")
    if paired(db_modified, live_modified):
        log.info("paired: the database (%s) is the one the trader's file (%s) was priced on", db_modified, live_modified)
    else:
        log.warning("NOT PAIRED: the database was written %s, the trader's file %s: a newer database prices the "
                    "shadow, so its difference from the trader's file is not the models' alone", db_modified or "?",
                    live_modified or "?")
    ensure_database()
    out = price(materialise(cfg["commit"], Path(args.models)), card, day)
    from predict_bfsp_today import OUTPUT_COLS
    out = out[[c for c in OUTPUT_COLS if c in out.columns]].copy()
    out["shadow_of"] = cfg["commit"]
    out["db_modified"], out["live_modified"] = db_modified, live_modified
    log.info("shadow priced %d runners in %d races", len(out), out["race_id"].nunique() if "race_id" in out else -1)
    if args.dry_run:
        print(out.head(20).to_string())
        return 0
    from ultra_betting.data.s3 import write_csv
    write_csv("predictions_shadow", day, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
