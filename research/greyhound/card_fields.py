"""The fields of the greyhounds.today cards we already record (the owner, 14 Nov 2026: "Yes do these").

greyhound/today_cards.py writes each track's card JSON raw to ``sources/greyhounds_today/<day>/<HHMMSS>Z_<track>_<file>.json``
(from 7 Oct 2026). This reads ``--n`` of those files (default 40), spread across days and tracks (the last version of
each card a day first: the fullest, weights filled in), walks every JSON path in each and reports, per path: the files
and records that have it, the fill rate (records with a non-empty value), the value types and a few example values
(long strings cut). Paths that look like the fields the greyhound model lacks are flagged: seeding (railer, middle,
wide; W/M/R codes and any other short code, e.g. the rails-to-middle pilot code), a bitch's season date, bend
positions or running lines, comments, going, reserves and non-runners, weights. A flag is a guess from the key's name
and its values: read the examples before using a field.

Read-only: nothing is written to the store. Outputs ``<out>.md`` and ``<out>.json`` (default out/card_fields).

    python -m research.greyhound.card_fields --n 40 --out out/card_fields
    python -m research.greyhound.card_fields --store /tmp/sources --from 2026-10-07 --to 2026-11-14
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PREFIX = "greyhounds_today"
FILE_RE = re.compile(r"(?:^|/)greyhounds_today/(\d{4}-\d{2}-\d{2})/(\d{6})Z_(.+)\.json$")
TRACKS = ("Central Park", "Dunstall Park", "Hove", "Kinsley", "Monmore", "Newcastle", "Nottingham", "Romford",
          "Sheffield", "Sunderland", "Towcester", "Yarmouth")            # greyhounds.today /api/tracks, 14 Nov 2026
EXAMPLES = 5
CUT = 60
DISTINCT_KEEP = 40                     # distinct values kept a path (for the code lists of short-coded fields)
REDACT_KEY = re.compile(r"e-?mail|phone|mobile|address|postcode|password|token", re.I)

#: a flag: (the key's name, any component of the path) or (the values)
FLAG_KEYS = {
    "seeding": re.compile(r"seed|railer|rails?$|^wide|middle|runstyle|run_style|runningstyle|^style$|pilot|^rmw$", re.I),
    "season": re.compile(r"season|seas$|inseason|heat", re.I),
    "bends_running_line": re.compile(r"bend|^bnd|split|sectional|^sec$|runline|running_?line|^rl$|calc|^posn?s?$|"
                                     r"positions|^pos\d|^b\d$", re.I),
    "comments": re.compile(r"comment|remark|^rmks?$|^notes?$|^cmt$|closeup|close_up|summary|spotlight|verdict", re.I),
    "going": re.compile(r"going|^ga$|^gng$", re.I),
    "reserves_non_runners": re.compile(r"reserve|^res$|non_?runner|^nr$|withdrawn|vacant|scratch|^status$|^void", re.I),
    "weights": re.compile(r"weight|^wt$|^wgt$|^kg$", re.I),
}
SEED_CODE = re.compile(r"^\(?\s*(?:R|M|W|RM|MW|RW|WM|MR|R/?M|M/?W|R-M|M-W|R2M|RtoM|RTM)\s*\)?$", re.I)
BENDS_LIKE = re.compile(r"^\d{3,6}$|^\d(?:[-/ ]\d){2,5}$")


def parse_key(key: str) -> tuple[str | None, str | None]:
    """(day, track) of a stored card file's key; the track from the known list where its name leads the file part."""
    m = FILE_RE.search(key)
    if not m:
        return None, None
    rest = m.group(3)
    for t in TRACKS:
        safe = t.replace(" ", "_")
        if rest.startswith(safe + "_") or rest == safe:
            return m.group(1), t
    return m.group(1), rest.split("_")[0]


def choose(keys, n: int, first: str | None = None, last: str | None = None) -> list[str]:
    """Up to ``n`` card files spread across days and tracks: the last file of each day and track first, taken evenly
    over the (day, track) groups, then the other files the same way while there is room."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for k in keys:
        day, track = parse_key(k)
        if day is None or (first and day < first) or (last and day > last):
            continue
        groups[(day, track)].append(k)
    if not groups or n <= 0:
        return []
    order = sorted(groups)
    tops = [sorted(groups[g])[-1] for g in order]

    def spread(items: list[str], m: int) -> list[str]:
        if m >= len(items):
            return list(items)
        if m == 1:
            return [items[-1]]
        step = (len(items) - 1) / (m - 1)
        return [items[round(i * step)] for i in range(m)]

    picked = spread(tops, n)
    if len(picked) < n:
        rest = [k for g in order for k in sorted(groups[g])[:-1]]
        picked += spread(rest, n - len(picked))
    return picked


def walk(x, path: str = "$"):
    """Every (path, value) in a JSON document; list positions written ``[]``, so a runner's field is one path. A
    container is yielded too (its path, the container) so an empty list or object counts as present."""
    yield path, x
    if isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(x, list):
        for v in x:
            yield from walk(v, f"{path}[]")


def _empty(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, (list, dict)) and not v)


def _tname(v) -> str:
    return {dict: "object", list: "array", str: "string", bool: "bool", int: "int", float: "float"}.get(type(v), "null") \
        if v is not None else "null"


def _show(v) -> str:
    s = v if isinstance(v, str) else json.dumps(v)
    return s if len(s) <= CUT else s[:CUT - 3] + "..."


def last_key(path: str) -> str:
    return path.rsplit(".", 1)[-1].replace("[]", "")


def flags(path: str, values: list) -> list[str]:
    """The flags a path earns from its name and its (scalar) values."""
    out = []
    key = last_key(path)
    parts = [p.replace("[]", "") for p in path.split(".")[1:]]
    strs = [str(v).strip() for v in values if not _empty(v) and not isinstance(v, (dict, list))]
    for name, rx in FLAG_KEYS.items():
        if rx.search(key) or (name in ("bends_running_line", "seeding") and any(rx.search(p) for p in parts[:-1])):
            out.append(name)
    if "seeding" not in out and strs and sum(bool(SEED_CODE.match(s)) for s in strs) >= 0.8 * len(strs):
        out.append("seeding")
    if "season" not in out and strs and re.search(r"bitch|dog", path, re.I) and re.search(r"season", " ".join(strs), re.I):
        out.append("season")
    if "bends_running_line" not in out and strs and sum(bool(BENDS_LIKE.match(s)) for s in strs) >= 0.8 * len(strs) \
            and re.search(r"pos|bnd|bend|line|run", path, re.I):
        out.append("bends_running_line")
    return out


def profile(docs: dict[str, object]) -> dict:
    """The path table of ``docs`` ({file key: parsed JSON})."""
    stat: dict[str, dict] = {}
    for fk, doc in docs.items():
        seen = set()
        for path, v in walk(doc):
            s = stat.setdefault(path, {"files": 0, "records": 0, "filled": 0, "types": defaultdict(int),
                                       "distinct": {}, "redacted": bool(REDACT_KEY.search(last_key(path)))})
            if path not in seen:
                seen.add(path)
                s["files"] += 1
            s["records"] += 1
            s["types"][_tname(v)] += 1
            if not _empty(v):
                s["filled"] += 1
                if not isinstance(v, (dict, list)) and len(s["distinct"]) < DISTINCT_KEEP:
                    s["distinct"].setdefault(v if not isinstance(v, str) else v.strip(), 0)
            if not isinstance(v, (dict, list)) and not _empty(v):
                key = v if not isinstance(v, str) else v.strip()
                if key in s["distinct"]:
                    s["distinct"][key] += 1
    n = len(docs)
    rows = []
    for path in sorted(stat):
        s = stat[path]
        vals = list(s["distinct"])
        ex = ["<redacted>"] if s["redacted"] and vals else [_show(v) for v in vals[:EXAMPLES]]
        rows.append({
            "path": path, "files": s["files"], "file_share": round(s["files"] / n, 3) if n else 0.0,
            "records": s["records"], "filled": s["filled"],
            "fill_rate": round(s["filled"] / s["records"], 3) if s["records"] else 0.0,
            "types": dict(s["types"]), "examples": ex,
            "distinct_seen": len(vals), "distinct_capped": len(vals) >= DISTINCT_KEEP,
            "flags": flags(path, vals),
        })
    for r in rows:                                   # a short-coded flagged field: its codes and counts, all of them
        if r["flags"] and not stat[r["path"]]["redacted"]:
            d = stat[r["path"]]["distinct"]
            if d and all(isinstance(k, str) and len(k) <= 8 for k in d):
                r["codes"] = dict(sorted(d.items(), key=lambda kv: -kv[1]))
    return {"paths": rows}


def run(store, n: int = 40, first: str | None = None, last: str | None = None) -> dict:
    keys = [k for k in store.listing(f"{PREFIX}/") if k.endswith(".json")]
    chosen = choose(keys, n, first, last)
    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": store.where(f"{PREFIX}/"), "card_files_held": len(keys), "asked": n,
           "from": first, "to": last, "files": [], "unreadable": []}
    docs = {}
    for k in chosen:
        raw = store.get(k)
        try:
            docs[k] = json.loads(raw)
        except (TypeError, ValueError) as e:
            out["unreadable"].append({"key": k, "error": str(e)[:200]})
            continue
        day, track = parse_key(k)
        out["files"].append({"key": k, "day": day, "track": track, "bytes": len(raw)})
    out["read"] = len(docs)
    out["days"] = sorted({f["day"] for f in out["files"]})
    out["tracks"] = sorted({f["track"] for f in out["files"]})
    out.update(profile(docs) if docs else {"paths": []})
    out["flagged"] = {name: [r["path"] for r in out["paths"] if name in r["flags"]] for name in FLAG_KEYS}
    return out


def markdown(r: dict) -> str:
    L = ["# greyhounds.today card fields", "",
         f"Generated {r['generated_utc']} from `{r['source']}` (research/greyhound/card_fields.py; read-only).", ""]
    if not r["read"]:
        L += [f"No data found: {r['card_files_held']} card files held under the prefix"
              + (f" between {r['from']} and {r['to']}" if r.get("from") or r.get("to") else "")
              + "; none read." + (f" Unreadable: {len(r['unreadable'])}." if r["unreadable"] else ""), ""]
        return "\n".join(L)
    L += [f"- Card files held: {r['card_files_held']}; read: {r['read']} of {r['asked']} asked"
          + (f"; unreadable: {len(r['unreadable'])}" if r["unreadable"] else ""),
          f"- Days: {len(r['days'])} ({r['days'][0]} to {r['days'][-1]}); tracks: {', '.join(r['tracks'])}",
          f"- Paths: {len(r['paths'])} (list positions written `[]`)", "",
          "## Flagged paths (a guess from the name and the values: read the examples)", ""]
    for name, paths in r["flagged"].items():
        L.append(f"- **{name}**: " + (", ".join(f"`{p}`" for p in paths) if paths else "none found"))
    coded = [p for p in r["paths"] if p.get("codes")]
    if coded:
        L += ["", "## Codes of the flagged short-coded fields (value: records)", ""]
        for p in coded:
            L.append(f"- `{p['path']}`: " + ", ".join(f"{k}: {v}" for k, v in list(p["codes"].items())[:DISTINCT_KEEP])
                     + (" (first 40 distinct only)" if p["distinct_capped"] else ""))
    L += ["", "## Every path", "",
          "| path | files | records | fill rate | types | examples | flags |", "|---|---:|---:|---:|---|---|---|"]
    for p in r["paths"]:
        ex = "; ".join(e.replace("|", "/").replace("\n", " ") for e in p["examples"])
        ty = ", ".join(f"{k} {v}" for k, v in p["types"].items())
        L.append(f"| `{p['path']}` | {p['files']} | {p['records']} | {p['fill_rate']:.3f} | {ty} | {ex} | "
                 f"{', '.join(p['flags'])} |")
    L += ["", "## Files read", ""] + [f"- {f['day']} {f['track']}: `{f['key']}`" for f in r["files"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=None, help="the sources store root (default S3)")
    ap.add_argument("--n", type=int, default=40, help="card files to read")
    ap.add_argument("--from", dest="first", default=None, help="first day, YYYY-MM-DD")
    ap.add_argument("--to", dest="last", default=None, help="last day, YYYY-MM-DD")
    ap.add_argument("--out", default="out/card_fields")
    a = ap.parse_args(argv)
    r = run(Store(root=a.store), a.n, a.first, a.last)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(r, indent=1, default=str))
    Path(a.out + ".md").write_text(markdown(r))
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
