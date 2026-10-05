"""What every source shares: the store (S3, or a local folder for tests and a dry run), a polite HTTP session, a clock."""

from __future__ import annotations

import gzip
import io
import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

PREFIX = "sources"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


class Store:
    """Files under ``sources/<source>/`` in the capture bucket, or under a local folder (``root``) when one is given.
    Keys are relative to ``sources/``, e.g. ``gbgb/raw/2024/2024-05-01.json.gz``."""

    def __init__(self, root: str | Path | None = None, s3=None, bucket: str | None = None):
        self.root = Path(root) if root else None
        self._s3 = s3
        self._bucket = bucket

    @property
    def s3(self):
        if self._s3 is None:
            from betfair_recorder import s3_client
            self._s3 = s3_client()
        return self._s3

    @property
    def bucket(self) -> str:
        if self._bucket is None:
            from betfair_recorder import bucket
            self._bucket = bucket()
        return self._bucket

    def where(self, key: str = "") -> str:
        return str(self.root / key) if self.root else f"s3://{self.bucket}/{PREFIX}/{key}"

    def put(self, key: str, data: bytes) -> None:
        if self.root:
            p = self.root / key
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(p)
            return
        self.s3.put_object(Bucket=self.bucket, Key=f"{PREFIX}/{key}", Body=data)

    def get(self, key: str) -> bytes | None:
        if self.root:
            p = self.root / key
            return p.read_bytes() if p.exists() else None
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=f"{PREFIX}/{key}")["Body"].read()
        except Exception as exc:
            if "NoSuchKey" in str(exc) or "404" in str(exc):
                return None
            raise

    def listing(self, prefix: str) -> dict[str, int]:
        """Every key under ``prefix`` (relative to ``sources/``) with its size."""
        out: dict[str, int] = {}
        if self.root:
            base = self.root / prefix
            if base.exists():
                for p in base.rglob("*"):
                    if p.is_file() and not p.name.endswith(".part"):
                        out[str(p.relative_to(self.root))] = p.stat().st_size
            return out
        full = f"{PREFIX}/{prefix}"
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=full):
            for o in page.get("Contents", []):
                out[o["Key"][len(PREFIX) + 1:]] = o["Size"]
        return out

    def put_parquet(self, key: str, df) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.put(key, buf.getvalue())

    def get_parquet(self, key: str):
        import pandas as pd
        raw = self.get(key)
        return None if raw is None else pd.read_parquet(io.BytesIO(raw))


def gz(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def gunzip(data: bytes) -> bytes:
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return s


def get(s: requests.Session, url: str, params=None, timeout: int = 60, tries: int = 4, pause: float = 0.0,
        ok_missing: bool = True) -> requests.Response | None:
    """One polite GET: retried on a fault or a 429/5xx with a growing wait; None for a 404 (a file not published)."""
    wait = 5.0
    for attempt in range(tries):
        try:
            r = s.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == tries - 1:
                raise
            log.warning("%s: %s; again in %.0f s", url, exc, wait)
            time.sleep(wait)
            wait *= 2
            continue
        if r.status_code == 404 and ok_missing:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == tries - 1:
                r.raise_for_status()
            time.sleep(float(r.headers.get("Retry-After") or wait))
            wait *= 2
            continue
        r.raise_for_status()
        if pause:
            time.sleep(pause)
        return r
    return None


class Budget:
    """A deadline: ``left()`` is False once ``minutes`` have passed (None: no limit)."""

    def __init__(self, minutes: float | None, clock=time.monotonic):
        self.clock = clock
        self.end = None if minutes is None else clock() + 60.0 * float(minutes)

    def left(self) -> bool:
        return self.end is None or self.clock() < self.end


def env(name: str) -> str | None:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else None
