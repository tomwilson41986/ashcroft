"""The Professor's record in S3 and on disk: keys under professor/, the day's files, the post log, the STOP switch.

Everything the tipster writes lives under s3://$ULTRA_BETTING_S3_BUCKET/professor/ (the trading bucket, 'ashcroft' by
default), and a copy of the day's files under out/professor/<day>/ for dry runs and the workflow artifact. A dry run
(PROFESSOR_DRY=1 or --dry-run) writes only to disk.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

BUCKET = os.getenv("ULTRA_BETTING_S3_BUCKET", "ashcroft")
REGION = os.getenv("PRED_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-west-2"
PREFIX = "professor"
OUT = Path(os.getenv("PROFESSOR_OUT") or "out/professor")
_s3 = None


def s3():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client("s3", region_name=REGION)
    return _s3


def dry() -> bool:
    return os.getenv("PROFESSOR_DRY", "") not in ("", "0", "no", "false")


def key(day: date | str, *parts: str) -> str:
    return "/".join([PREFIX, "tips", str(day)] + list(parts))


def out_dir(day: date | str) -> Path:
    p = OUT / str(day)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_bytes(k: str, bucket: str | None = None) -> bytes | None:
    """An object's body, or None when it is not there (or S3 is unreachable: logged, never raised)."""
    try:
        return s3().get_object(Bucket=bucket or BUCKET, Key=k)["Body"].read()
    except Exception as e:  # noqa: BLE001
        if "NoSuchKey" not in str(e) and "Not Found" not in str(e):
            log.warning("s3 get %s: %s", k, str(e)[:120])
        return None


def read_json(k: str):
    raw = get_bytes(k)
    return None if raw is None else json.loads(raw.decode())


def read_csv(k: str, **kw) -> pd.DataFrame | None:
    raw = get_bytes(k)
    if raw is None:
        return None
    if k.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), low_memory=False, **kw)


def put_bytes(k: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    if dry():
        return
    s3().put_object(Bucket=BUCKET, Key=k, Body=body, ContentType=content_type)


def write_json(k: str, data, local: Path | None = None) -> None:
    body = json.dumps(data, indent=2, default=str).encode()
    if local is not None:
        local.write_bytes(body)
    put_bytes(k, body, "application/json")


def write_csv(k: str, df: pd.DataFrame, local: Path | None = None) -> None:
    body = df.to_csv(index=False).encode()
    if local is not None:
        local.write_bytes(body)
    put_bytes(k, body, "text/csv")


def write_text(k: str, text: str, local: Path | None = None, content_type: str = "text/plain") -> None:
    if local is not None:
        local.write_text(text)
    put_bytes(k, text.encode(), content_type)


def stopped() -> bool:
    """No publishing while professor/STOP or trading/STOP exists (the owner's switches)."""
    for k in (f"{PREFIX}/STOP", "trading/STOP"):
        try:
            s3().head_object(Bucket=BUCKET, Key=k)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def post_log(day: date | str) -> dict:
    """Every post of the day by its key: {"<channel>:<kind>:<slug>": {"at": utc, ...the API's reply}}."""
    local = out_dir(day) / "posts.json"
    if local.exists():
        return json.loads(local.read_text())
    return read_json(key(day, "posts.json")) or {}


def record_post(day: date | str, post_key: str, result: dict) -> dict:
    log_ = post_log(day)
    log_[post_key] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result}
    write_json(key(day, "posts.json"), log_, out_dir(day) / "posts.json")
    return log_


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
