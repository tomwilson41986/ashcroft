"""Bronze: the raw Betfair Historic Data files, kept exactly as served, as a table.

A bundle (the zip or tar the owner downloads with scripts/betfair_historic_download.ps1, or the tar the Historic
Data website builds) is read member by member; each market file becomes one row of ``bronze/raw`` in the partition
of its own path date (``.../2026/Jan/29/<event>/<market>.bz2`` -> year=2026/month=01), with its bytes and SHA-256.
``bronze/index`` holds the same rows without the bytes (small: what the builds and checks read).

Idempotent: a file already held with the same hash is skipped; the same path with other bytes is kept only if it
is longer (a market downloaded again after it settled), and counted. Nested zips (a month zipped inside the year's
zip) are read through; a member that is not a market file is counted and left.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import tarfile
import zipfile
from typing import Iterable, Iterator

import pyarrow as pa

from football.warehouse.schema import BRONZE_RAW, TableSpec
from football.warehouse.store_io import (fingerprint, list_parts, part_name, read_manifest, read_part, utc_now,
                                         write_manifest, write_part)

log = logging.getLogger(__name__)

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
_MON = "|".join(MONTHS)
FULL_RE = re.compile(rf"(?:^|/)(?:(BASIC|ADVANCED|PRO)/)?(\d{{4}})/({_MON})/(\d{{1,2}})/(\d+)/(\d+\.\d+)\.bz2$")
SHORT_RE = re.compile(rf"(?:^|/)({_MON})/(\d{{1,2}})/(\d+)/(\d+\.\d+)\.bz2$")
YEAR_RE = re.compile(r"(?:^|/)(\d{4})(?:/|$)")

BRONZE_INDEX = TableSpec(
    "bronze/index", "bronze/raw without the bytes: one row a raw file. What builds and checks read.",
    tuple(c for c in BRONZE_RAW.columns if c.name != "data"), key=("plan_path",), sort=("plan_path",))

FLUSH_BYTES = 1_500_000_000        # buffered file bytes before the largest partition is written out
URLS_ENV = "FOOTBALL_BUNDLE_URLS"   # {"<bundle name>": "<download link>"} as JSON: never in the repository


def plan_path(member: str, container: str | None = None, plan: str = "BASIC") -> dict | None:
    """A member's identity from its path: {'plan_path', 'year', 'month', 'day', 'event_id', 'market_id'}, or None
    when the member is not a market file. ``container``: the archive member a nested archive came from, which may
    carry the year its own members lack ('2026/Oct.zip' holding 'Oct/1/<event>/<market>.bz2')."""
    name = member.replace("\\", "/")
    m = FULL_RE.search(name)
    if m:
        p, year, mon, day, event, market = m.group(1) or plan, int(m.group(2)), m.group(3), int(m.group(4)), \
            int(m.group(5)), m.group(6)
    else:
        m = SHORT_RE.search(name)
        y = YEAR_RE.search((container or "").replace("\\", "/"))
        if not m or not y:
            return None
        p, year, mon, day, event, market = plan, int(y.group(1)), m.group(1), int(m.group(2)), int(m.group(3)), \
            m.group(4)
    return {"plan_path": f"{p}/{year}/{mon}/{day}/{event}/{market}.bz2", "year": year, "month": MONTHS[mon],
            "day": day, "event_id": event, "market_id": market}


def iter_zip(path) -> Iterator[tuple[str, bytes, str | None]]:
    """(member name, bytes, container) for every file in a zip, reading through zips nested inside it."""
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if info.filename.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(z.read(info))) as inner:
                    for sub in inner.infolist():
                        if not sub.is_dir():
                            yield sub.filename, inner.read(sub), info.filename
            else:
                yield info.filename, z.read(info), None


def iter_tar(path) -> Iterator[tuple[str, bytes, str | None]]:
    with tarfile.open(path) as t:
        for m in t:
            if m.isfile():
                f = t.extractfile(m)
                if f is not None:
                    yield m.name, f.read(), None


def iter_bundle(path) -> Iterator[tuple[str, bytes, str | None]]:
    return iter_zip(path) if zipfile.is_zipfile(path) else iter_tar(path)


class Ingest:
    """Files on their way into bronze, buffered by partition."""

    def __init__(self, store, source: str, flush_bytes: int = FLUSH_BYTES):
        self.store, self.source, self.flush_bytes = store, source, flush_bytes
        self.buf: dict[tuple[int, int], dict[str, dict]] = {}
        self.buffered = 0
        self.report = {"source": source, "members": 0, "market_files": 0, "not_market_files": 0,
                       "duplicates_in_bundle": 0, "new": 0, "same": 0, "replaced": 0, "kept_longer_existing": 0,
                       "partitions": {}}

    def add(self, member: str, data: bytes, container: str | None = None) -> None:
        self.report["members"] += 1
        ident = plan_path(member, container)
        if ident is None:
            self.report["not_market_files"] += 1
            return
        self.report["market_files"] += 1
        part = self.buf.setdefault((ident["year"], ident["month"]), {})
        row = {"plan_path": ident["plan_path"], "event_id": ident["event_id"], "market_id": ident["market_id"],
               "day": ident["day"], "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "source": self.source, "data": data}
        old = part.get(row["plan_path"])
        if old is not None:
            self.report["duplicates_in_bundle"] += 1
            if old["size"] >= row["size"]:
                return
            self.buffered -= old["size"]
        part[row["plan_path"]] = row
        self.buffered += len(data)
        while self.buffered > self.flush_bytes and self.buf:
            self.flush(max(self.buf, key=lambda k: sum(r["size"] for r in self.buf[k].values())))

    def flush(self, key: tuple[int, int]) -> None:
        rows = self.buf.pop(key, {})
        if not rows:
            return
        self.buffered -= sum(r["size"] for r in rows.values())
        year, month = key
        held = read_part(self.store, BRONZE_RAW.name, year, month)
        merged: dict[str, dict] = {r["plan_path"]: r for r in held.to_pylist()} if held is not None else {}
        counts = {"new": 0, "same": 0, "replaced": 0, "kept_longer_existing": 0}
        for path, row in rows.items():
            old = merged.get(path)
            if old is None:
                merged[path] = row
                counts["new"] += 1
            elif old["sha256"] == row["sha256"]:
                counts["same"] += 1
            elif row["size"] > old["size"]:
                merged[path] = row
                counts["replaced"] += 1
            else:
                counts["kept_longer_existing"] += 1
        name = part_name(year, month)
        fp = fingerprint(f"{r['plan_path']}:{r['sha256']}" for r in merged.values())
        man = read_manifest(self.store, BRONZE_RAW.name)
        # the partition, its index and its manifest entry are three writes: one that died between them is put
        # right here (the manifest does not describe what is held), whether or not this flush adds a file
        stale = (man["parts"].get(name) or {}).get("fingerprint") != fp
        if counts["new"] or counts["replaced"] or held is None or stale:
            table = pa.Table.from_pylist(list(merged.values()), schema=BRONZE_RAW.schema)
            wrote = write_part(self.store, BRONZE_RAW, year, month, table)
            write_part(self.store, BRONZE_INDEX, year, month, table.drop_columns(["data"]))
            man = read_manifest(self.store, BRONZE_RAW.name)
            sources = sorted(set(r["source"] for r in merged.values()))
            man["parts"][name] = {"rows": wrote["rows"], "bytes": wrote["bytes"], "sources": sources,
                                  "fingerprint": fp, "written_utc": utc_now()}
            write_manifest(self.store, BRONZE_RAW.name, man)
        for k, v in counts.items():
            self.report[k] += v
        p = self.report["partitions"].setdefault(name, {"new": 0, "same": 0, "replaced": 0,
                                                        "kept_longer_existing": 0, "held": 0})
        for k, v in counts.items():
            p[k] += v
        p["held"] = len(merged)
        log.info("bronze %s: %d held (%d new, %d same, %d replaced)", name, len(merged), counts["new"],
                 counts["same"], counts["replaced"])

    def close(self) -> dict:
        for key in sorted(self.buf):
            self.flush(key)
        return self.report


def ingest(store, members: Iterable[tuple[str, bytes, str | None]], source: str,
           flush_bytes: int = FLUSH_BYTES) -> dict:
    """Every market file of ``members`` into bronze; the report (members, market files, new, same, replaced, the
    count held per partition)."""
    ing = Ingest(store, source, flush_bytes)
    for member, data, container in members:
        ing.add(member, data, container)
    return ing.close()


def ingest_bundle(store, path, source: str) -> dict:
    return ingest(store, iter_bundle(path), source)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def ingest_queue(store, queue: str | None = None, dest: str = ".", fetch=None) -> dict:
    """Every bundle listed in ``football/warehouse/bundles.json`` ([{name, bytes, sha256, sport, note}]) into
    bronze, each once: a bundle the store notes as ingested (``bronze/bundles/<name>.json``) is not downloaded
    again. A bundle's link comes from the environment, ``FOOTBALL_BUNDLE_URLS`` = {"<name>": "<url>"} as JSON (a
    repository secret in the workflows): the data is licensed to its buyer and the repository is public, so the
    list need not carry it. An entry may still give a ``url`` itself (as sources/betfair_imports.json does); a
    bundle with no link in either place is reported as waiting, not as failed.

    The download is curl's (retried; ``fetch(url, path)`` in its place for tests). A download is only read if it
    is the bundle: its size and SHA-256 must be the entry's where the entry gives them (a share link can serve a
    web page, or a file cut short, with no error), and it must hold at least one market file. A bundle that fails
    is reported and not noted as ingested, so the next run tries again; the downloaded file is removed either
    way, and nothing stored or printed carries the link."""
    import json
    import os
    import subprocess
    from pathlib import Path

    from football.warehouse import ROOT
    path = Path(queue) if queue else Path(__file__).with_name("bundles.json")
    items = json.loads(path.read_text()) if path.exists() else []
    # the secret must be what it is meant to be: a bare link pasted in, or a map under another bundle's name, would
    # otherwise read as "no link yet" and pass for ever
    given = os.environ.get(URLS_ENV) or ""
    try:
        links = json.loads(given or "{}")
    except ValueError:
        links = None
    broken = not isinstance(links, dict)
    links = {} if broken else links
    out = {}
    for it in items:
        name = it["name"]
        marker = f"{ROOT}/bronze/bundles/{name}.json"
        if store.get(marker) is not None:
            out[name] = "already ingested"
            continue
        url = links.get(name) or it.get("url")
        if not url and broken:
            out[name] = {"failed": f"{URLS_ENV} is set and is not a JSON object of bundle name to link"}
            continue
        if not url and links:
            out[name] = {"failed": f"{URLS_ENV} has links, none under this bundle's name"}
            continue
        if not url:
            out[name] = f"waiting for its link ({URLS_ENV})"
            continue
        local = Path(dest) / f"{name}.bundle"
        try:
            local.unlink(missing_ok=True)            # never resume onto what an earlier run left behind
            if fetch is not None:
                fetch(url, str(local))
            else:
                subprocess.run(["curl", "-sSLf", "--retry", "5", "-o", str(local), url], check=True,
                               stderr=subprocess.DEVNULL)        # curl's errors can quote the link
            size = local.stat().st_size
            if it.get("bytes") is not None and size != int(it["bytes"]):
                raise ValueError(f"the download is {size} bytes, the bundle is {it['bytes']}")
            if it.get("sha256") and file_sha256(local) != it["sha256"]:
                raise ValueError("the download's SHA-256 is not the bundle's")
            rep = ingest_bundle(store, str(local), name)
            if not rep["market_files"]:
                raise ValueError("no market file in the download")
            rep["bytes"] = size
            noted = {k: v for k, v in it.items() if k != "url"}
            store.put(marker, (json.dumps({**noted, **rep, "ingested_utc": utc_now()}, indent=1) + "\n").encode())
            out[name] = {k: rep[k] for k in ("members", "market_files", "not_market_files", "duplicates_in_bundle",
                                             "new", "same", "replaced", "bytes")}
        except Exception as exc:                     # noqa: BLE001 - one bundle's failure must not hide the others'
            why = "the download failed" if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            log.error("bundle %s failed: %s: %s", name, type(exc).__name__, why.replace(url, "<link>"))
            out[name] = {"failed": f"{type(exc).__name__}: {why}".replace(url, "<link>")[:300]}
        finally:
            local.unlink(missing_ok=True)
    return out


def index(store, parts: list[tuple[int, int]] | None = None) -> pa.Table | None:
    """bronze/index for the partitions asked (all by default), with year and month."""
    from football.warehouse.store_io import read_table
    return read_table(store, BRONZE_INDEX.name, parts=parts, with_partition=True)


def partitions(store) -> list[tuple[int, int]]:
    return list_parts(store, BRONZE_RAW.name)
