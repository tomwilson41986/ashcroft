"""What tote.co.uk serves in public, so a Tote record can be built (the owner's ask of 6 Oct 2026; read-only).

The owner wants the Tote's horse-racing pools (win, place, exacta, trifecta, swinger) beside the Betfair record,
to back on the Tote against a lay on Betfair where the prices part. tote.co.uk answers "Unavailable in Region"
outside the UK, so this runs on the owner's UK server (tote-probe.yml). It logs in to nothing and posts nothing.
It reads robots.txt first and keeps to it, then the home, racecards and results pages and the race pages they
link to on the same site, at most --max-requests GETs two seconds apart, and looks in each page for the data it
is built from: JSON embedded in the page (Next.js __NEXT_DATA__, Nuxt and window state objects) and the API
addresses its scripts name. A few of those addresses that read as data (no placeholders) are asked for JSON.
Every page and JSON answer is saved under --out with summary.json: each request's status and size, whether the
region block answered, the embedded data's keys that name pools, dividends, races or runners, and the API
addresses found. With CAPTURE_BUCKET set the same files go to s3://$CAPTURE_BUCKET/tote/probe/<UTC time>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("tote_probe")

BASE = "https://tote.co.uk"
START = ["/", "/racecards", "/results"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
PAUSE = 2.0                                     # seconds between requests
MAX_BYTES = 5_000_000                           # a larger answer is cut when saved
EMBEDDED = [
    ("next_data", re.compile(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)),
    ("nuxt", re.compile(r"window\.__NUXT__\s*=\s*(\{.*?\})\s*;?\s*</script>", re.S)),
    ("state", re.compile(r"window\.__(?:INITIAL|PRELOADED|APOLLO|REDUX)_STATE__\s*=\s*(\{.*?\})\s*;?\s*</script>",
                         re.S)),
    ("ld_json", re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.S)),
]
QUOTED = re.compile(r"""["'`]([^"'`\s<>]{4,300})["'`]""")
HREF = re.compile(r"""href=["']([^"'#]+)["']""", re.I)
SCRIPT_SRC = re.compile(r"""<script[^>]+src=["']([^"']+)["']""", re.I)
TITLE = re.compile(r"<title>(.*?)</title>", re.I | re.S)
PAGE_WORDS = ("racecard", "result", "race", "meeting", "pool", "exacta", "trifecta", "swinger", "placepot")
DATA_WORDS = ("pool", "dividend", "payout", "approx", "willpay", "odds", "price", "race", "meeting", "runner",
              "selection", "exacta", "trifecta", "swinger", "placepot", "result")


def embedded_json(html: str) -> dict[str, object]:
    """The JSON a page carries for its scripts, by kind (the first of each kind that parses)."""
    out: dict[str, object] = {}
    for kind, pattern in EMBEDDED:
        for m in pattern.finditer(html):
            try:
                out[kind] = json.loads(m.group(1))
                break
            except ValueError:
                continue
    return out


def interesting_keys(obj, words=DATA_WORDS, path: str = "", depth: int = 0, limit: int = 200) -> list[str]:
    """Paths (a.b[0].c) to the keys whose names hold any of the words, depth-first, at most limit of them."""
    found: list[str] = []

    def walk(o, p, d):
        if len(found) >= limit or d > 12:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                q = f"{p}.{k}" if p else str(k)
                if any(w in str(k).lower() for w in words):
                    found.append(q)
                walk(v, q, d + 1)
        elif isinstance(o, list) and o:
            walk(o[0], f"{p}[0]", d + 1)        # the first item stands for the list

    walk(obj, path, depth)
    return found[:limit]


def api_candidates(text: str, base: str = BASE) -> list[str]:
    """Addresses in a page or script that read as data: an api host or path, graphql, or a data word in the path.
    Relative paths are joined to base; addresses with template placeholders are kept, marked by their braces."""
    out = []
    for s in QUOTED.findall(text):
        low = s.lower()
        if low.startswith(("data:", "mailto:", "tel:", "javascript:")) or low.endswith(
                (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".css", ".woff", ".woff2", ".ico", ".map")):
            continue
        absolute = low.startswith(("http://", "https://", "//"))
        if not absolute and not s.startswith("/"):
            continue
        u = urllib.parse.urljoin(base, s)
        host = urllib.parse.urlparse(u).netloc.lower()
        path = urllib.parse.urlparse(u).path.lower()
        if (host.startswith("api") or "/api" in path or "graphql" in low
                or (absolute and any(w in path for w in ("pool", "dividend", "racecard", "meeting", "result")))):
            out.append(u)
    return sorted(set(out))


def site_links(html: str, base: str = BASE) -> list[str]:
    """Links on the same site whose path names racing, in page order, without repeats."""
    host = urllib.parse.urlparse(base).netloc.lower().removeprefix("www.")
    seen, out = set(), []
    for h in HREF.findall(html):
        u = urllib.parse.urljoin(base, h)
        p = urllib.parse.urlparse(u)
        if p.netloc.lower().removeprefix("www.") != host or not any(w in p.path.lower() for w in PAGE_WORDS):
            continue
        u = urllib.parse.urlunparse(p._replace(query="", fragment=""))
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def blocked(html: str) -> bool:
    m = TITLE.search(html or "")
    return bool(m) and "unavailable in region" in m.group(1).lower()


def safe_name(url: str, suffix: str) -> str:
    p = urllib.parse.urlparse(url)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (p.netloc + p.path).strip("/"))[:150] or "root"
    return f"{name}{suffix}"


class Probe:
    def __init__(self, out: Path, max_requests: int, pause: float = PAUSE, base: str = BASE):
        self.out, self.max_requests, self.pause, self.base = out, max_requests, pause, base
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept-Language": "en-GB,en;q=0.9"})
        self.robots = urllib.robotparser.RobotFileParser()
        self.requests: list[dict] = []
        self.out.mkdir(parents=True, exist_ok=True)

    def allowed(self, url: str) -> bool:
        return self.robots.can_fetch("*", url)

    def get(self, url: str, accept: str = "text/html,application/xhtml+xml,*/*;q=0.8") -> requests.Response | None:
        if len(self.requests) >= self.max_requests:
            return None
        if self.requests:
            time.sleep(self.pause)
        rec = {"url": url, "accept": accept.split(",")[0]}
        try:
            r = self.session.get(url, headers={"Accept": accept}, timeout=25, allow_redirects=True)
            rec.update(status=r.status_code, bytes=len(r.content), type=r.headers.get("content-type", ""),
                       final_url=r.url)
            self.requests.append(rec)
            return r
        except requests.RequestException as exc:
            rec.update(status=None, error=f"{type(exc).__name__}: {exc}"[:300])
            self.requests.append(rec)
            return None

    def save(self, name: str, content: bytes) -> str:
        path = self.out / name
        path.write_bytes(content[:MAX_BYTES])
        return name

    def run(self) -> dict:
        summary: dict = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "base": self.base,
                         "pages": [], "api": {}, "embedded": {}, "region_blocked": False}
        r = self.get(urllib.parse.urljoin(self.base, "/robots.txt"), accept="text/plain,*/*")
        robots_text = r.text if (r is not None and r.ok and "text/plain" in r.headers.get("content-type", "")) else ""
        self.save("robots.txt", robots_text.encode())
        self.robots.parse(robots_text.splitlines())
        if r is not None and r.status_code in (401, 403):
            self.robots.disallow_all = True     # as urllib.robotparser reads a refused robots.txt
        summary["robots_txt"] = bool(robots_text)
        summary["robots_disallow_all"] = self.robots.disallow_all

        queue = [urllib.parse.urljoin(self.base, p) for p in START]
        seen: set[str] = set()
        scripts: list[str] = []
        apis: set[str] = set()
        while queue and len(self.requests) < self.max_requests:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            if not self.allowed(url):
                summary["pages"].append({"url": url, "skipped": "robots.txt"})
                continue
            r = self.get(url)
            if r is None:
                summary["pages"].append({"url": url, "status": None})
                continue
            html = r.text
            page = {"url": url, "status": r.status_code, "bytes": len(r.content), "title": None,
                    "file": self.save(safe_name(url, ".html"), r.content)}
            m = TITLE.search(html)
            page["title"] = m.group(1).strip()[:200] if m else None
            if blocked(html):
                summary["region_blocked"] = True
                page["region_blocked"] = True
            data = embedded_json(html)
            for kind, obj in data.items():
                name = self.save(safe_name(url, f".{kind}.json"), json.dumps(obj).encode())
                summary["embedded"][name] = interesting_keys(obj)
            apis.update(api_candidates(html, self.base))
            for s in SCRIPT_SRC.findall(html):
                u = urllib.parse.urljoin(url, s)
                if u not in scripts:
                    scripts.append(u)
            links = site_links(html, self.base)
            page["links"] = links[:40]
            summary["pages"].append(page)
            if len(seen) < 12:                  # the start pages and the first race pages they link to
                queue.extend(u for u in links[:8] if u not in seen)

        bundles = []
        for u in scripts[:10]:                  # the page's scripts name the API it calls
            if len(self.requests) >= self.max_requests:
                break
            if urllib.parse.urlparse(u).netloc.lower().removeprefix("www.") in ("tote.co.uk",) and not self.allowed(u):
                continue
            r = self.get(u, accept="application/javascript,*/*")
            if r is None or not r.ok:
                continue
            found = api_candidates(r.text, self.base)
            apis.update(found)
            bundles.append({"url": u, "bytes": len(r.content), "api": found[:50]})
        summary["scripts"] = bundles

        tried = 0
        for u in sorted(apis):
            summary["api"][u] = {"tried": False}
            if "{" in u or "${" in u or tried >= 10 or len(self.requests) >= self.max_requests:
                continue
            if urllib.parse.urlparse(u).netloc.lower().removeprefix("www.") == "tote.co.uk" and not self.allowed(u):
                summary["api"][u]["skipped"] = "robots.txt"
                continue
            tried += 1
            r = self.get(u, accept="application/json,*/*;q=0.5")
            rec = {"tried": True, "status": None if r is None else r.status_code}
            if r is not None and r.ok and "json" in r.headers.get("content-type", ""):
                try:
                    obj = r.json()
                    rec["file"] = self.save(safe_name(u, ".json"), json.dumps(obj).encode())
                    rec["keys"] = interesting_keys(obj, limit=60)
                except ValueError:
                    pass
            summary["api"][u] = rec
        summary["requests"] = self.requests
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1))
        return summary


def upload(out: Path, bucket: str, region: str, stamp: str) -> int:
    import boto3
    s3 = boto3.client("s3", region_name=region)
    n = 0
    for f in sorted(out.iterdir()):
        if f.is_file():
            s3.upload_file(str(f), bucket, f"tote/probe/{stamp}/{f.name}")
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="tote_probe_out")
    ap.add_argument("--max-requests", type=int, default=40)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    s = Probe(out, args.max_requests, base=args.base).run()
    pages = [p for p in s["pages"] if p.get("status")]
    log.info("%d requests; region blocked: %s; robots.txt: %s", len(s["requests"]), s["region_blocked"],
             s["robots_txt"])
    for p in s["pages"]:
        log.info("page %s %s %s %s", p.get("status"), p.get("bytes"), p["url"], p.get("title") or p.get("skipped", ""))
    for name, keys in s["embedded"].items():
        log.info("embedded %s: %d data keys, e.g. %s", name, len(keys), keys[:8])
    for u, rec in s["api"].items():
        if rec.get("tried"):
            log.info("api %s %s %s", rec.get("status"), u, (rec.get("keys") or [])[:6])
    log.info("%d API addresses named (%d asked)", len(s["api"]), sum(1 for r in s["api"].values() if r.get("tried")))
    bucket = os.environ.get("CAPTURE_BUCKET")
    if bucket:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            n = upload(out, bucket, os.environ.get("CAPTURE_REGION") or "eu-west-2", stamp)
            log.info("%d files to s3://%s/tote/probe/%s/", n, bucket, stamp)
        except Exception as exc:                # the artifact keeps them either way
            log.warning("S3 upload failed: %s", exc)
    return 0 if pages else 1


if __name__ == "__main__":
    raise SystemExit(main())
