"""S3 read/write helpers for the ultra-betting pipeline."""

import io
import json
import logging
from datetime import date

import boto3
import pandas as pd

from ultra_betting.config import S3_BUCKET

log = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3")
    return _client


def _key(prefix: str, dt: date | str, ext: str = "csv") -> str:
    """Build an S3 key like predictions/2026-03-16.csv."""
    return f"{prefix}/{dt}.{ext}"


# ── Write helpers ──────────────────────────────────────────────────

def write_csv(prefix: str, dt: date | str, df: pd.DataFrame) -> str:
    """Write a DataFrame as CSV to S3."""
    key = _key(prefix, dt)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    _get_client().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    log.info(f"Wrote s3://{S3_BUCKET}/{key} ({len(df)} rows)")
    return key


def write_json(key: str, data: dict | list) -> str:
    """Write JSON to S3."""
    body = json.dumps(data, indent=2, default=str)
    _get_client().put_object(Bucket=S3_BUCKET, Key=key, Body=body)
    log.info(f"Wrote s3://{S3_BUCKET}/{key}")
    return key


def write_html(prefix: str, dt: date | str, html: str) -> str:
    """Write HTML to S3."""
    key = f"{prefix}/{dt}.html"
    _get_client().put_object(
        Bucket=S3_BUCKET, Key=key, Body=html, ContentType="text/html"
    )
    log.info(f"Wrote s3://{S3_BUCKET}/{key}")
    return key


# ── Read helpers ──────────────────────────────────────────────────

def read_csv(prefix: str, dt: date | str) -> pd.DataFrame:
    """Read a CSV from S3 into a DataFrame."""
    key = _key(prefix, dt)
    try:
        obj = _get_client().get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_csv(io.StringIO(obj["Body"].read().decode()))
    except _get_client().exceptions.NoSuchKey:
        log.warning(f"No file at s3://{S3_BUCKET}/{key}")
        return pd.DataFrame()
    except Exception as e:
        log.warning(f"Failed to read s3://{S3_BUCKET}/{key}: {e}")
        return pd.DataFrame()


def read_json(key: str) -> dict | list | None:
    """Read JSON from S3."""
    try:
        obj = _get_client().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode())
    except Exception as e:
        log.warning(f"Failed to read s3://{S3_BUCKET}/{key}: {e}")
        return None


def append_to_json_list(key: str, item: dict) -> None:
    """Append an item to a JSON list in S3 (read-modify-write)."""
    existing = read_json(key)
    if not isinstance(existing, list):
        existing = []
    existing.append(item)
    write_json(key, existing)
