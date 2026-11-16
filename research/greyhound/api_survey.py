"""What GreyhoundAPI holds, field by field (the owner, 16 Nov 2026: "What other data from the api can we use? Research
it and include in modelling").

The documentation page refuses automated reads (HTTP 403), so the inventory is taken from the data itself:

1. **Stored raw data** (``sources/greyhoundapi/raw/``, written by sources/greyhoundapi.py): every key path of the
   results (one month a year a region, plus the latest), the full runner lists (``/races/<id>/runners``), the latest
   card snapshot, and the lists (tracks, sires, track stats), each with its fill share, type and up to three examples.
2. **The runner tables** (``runs_<region>_<year>.parquet``): each column's fill share by region and year.
3. **The API itself** (``--probe``, with ``GREYHOUNDAPI_KEY``): its own schema if it serves one (``openapi.json``),
   and a short list of likely endpoints called once each with real ids from the latest data: the status code and the
   key paths of what comes back. About 40 calls, against the plan's 600,000 a month.

Read-only to the API and the store; writes out/api_survey/{survey.json, report.md} and, with ``--save``, the survey
and any schema to ``sources/greyhoundapi/survey/``. The key is read from the environment and never written.

    python -m research.greyhound.api_survey --probe --save --out out/api_survey
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PREFIX = "greyhoundapi"
ROOT = "https://api.greyhoundapi.com"
EXAMPLES = 3
#: endpoints tried once each in ``--probe`` (``{dog}``, ``{trainer}``, ``{race}``, ``{track}``, ``{sire}``,
#: ``{meeting}`` filled from the latest stored results)
PROBE = ["/dogs/{dog}", "/dogs/{dog}/form", "/dogs/{dog}/runs", "/dogs/{dog}/history", "/dogs/{dog}/results",
         "/dogs/{dog}/stats", "/dogs/{dog}/ratings", "/dogs/{dog}/trials", "/dogs/{dog}/weights",
         "/dogs/{dog}/sectionals", "/dogs?search={dogname}", "/trainers", "/trainers/{trainer}",
         "/trainers/{trainer}/stats", "/trainers/{trainer}/runners", "/races/{race}", "/races/{race}/runners",
         "/races/{race}/market", "/races/{race}/comments", "/races/{race}/sectionals", "/races/{race}/result",
         "/meetings", "/meetings/{meeting}", "/tracks/{track}", "/tracks/{track}/stats", "/tracks/{track}/records",
         "/tracks/{track}/standards", "/sires/{sire}", "/sires/{sire}/progeny", "/dams", "/owners", "/trials",
         "/ratings", "/predictions", "/tips", "/stats", "/insights", "/analysis", "/racecards/today", "/usage"]
SCHEMA_PATHS = ["/v1/openapi.json", "/openapi.json", "/v1/swagger.json", "/swagger.json", "/v1/docs", "/docs"]


def walk(obj, path: str, acc: dict) -> None:
    """Each key path (``a.b[].c``) under ``obj``: seen, non-null, types, a few examples."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}" if path else str(k), acc)
        return
    if isinstance(obj, list):
        if obj and all(not isinstance(x, (dict, list)) for x in obj):
            _note(acc, f"{path}[]", obj[:3])
            return
        for x in obj:
            walk(x, f"{path}[]", acc)
        return
    _note(acc, path, obj)


def _note(acc: dict, path: str, v) -> None:
    e = acc.setdefault(path, {"seen": 0, "filled": 0, "types": set(), "examples": []})
    e["seen"] += 1
    if v is None or v == "" or v == []:
        return
    e["filled"] += 1
    e["types"].add(type(v).__name__)
    s = str(v)[:60]
    if len(e["examples"]) < EXAMPLES and s not in e["examples"]:
        e["examples"].append(s)


def table(acc: dict, n: int | None = None) -> list[dict]:
    """The paths as rows: fill share of the times the path's parent was seen (``seen``), types, examples."""
    rows = []
    for p, e in sorted(acc.items()):
        rows.append({"path": p, "seen": e["seen"], "fill": round(e["filled"] / e["seen"], 3) if e["seen"] else 0.0,
                     "types": sorted(e["types"]), "examples": e["examples"]})
    return rows


def _load(store, key):
    raw = store.get(key)
    return json.loads(gzip.decompress(raw)) if raw else None


def stored(store, regions=("GB", "IE", "AU")) -> dict:
    out: dict = {"results": {}, "runners": {}, "cards": None, "lists": {}}
    for reg in regions:
        keys = sorted(k for k in store.listing(f"{PREFIX}/raw/results/{reg}/") if k.endswith(".json.gz"))
        by_year: dict = {}
        for k in keys:
            by_year.setdefault(k.rsplit("/", 1)[1][:4], []).append(k)
        pick = sorted({v[len(v) // 2] for v in by_year.values()} | ({keys[-1]} if keys else set()))
        per_year = {}
        for k in pick:
            races = _load(store, k) or []
            acc: dict = {}
            for r in races:
                walk(r, "", acc)
            per_year[k.rsplit("/", 1)[1][:7]] = {"races": len(races), "paths": table(acc)}
        out["results"][reg] = per_year
        rk = sorted(k for k in store.listing(f"{PREFIX}/raw/runners/{reg}/") if k.endswith(".json.gz"))
        if rk:
            acc = {}
            n = 0
            for k in rk[-2:]:
                for v in (_load(store, k) or {}).values():
                    walk(v, "", acc)
                    n += 1
            out["runners"][reg] = {"months_held": len(rk), "months_read": [k.rsplit("/", 1)[1][:7] for k in rk[-2:]],
                                   "races": n, "paths": table(acc)}
    ck = sorted(k for k in store.listing(f"{PREFIX}/raw/cards/") if k.endswith(".json.gz"))
    if ck:
        acc = {}
        snap = _load(store, ck[-1]) or {}
        for v in snap.values():
            walk(v, "", acc)
        out["cards"] = {"snapshot": ck[-1].split("/raw/cards/")[-1], "snapshots_held": len(ck), "paths": table(acc)}
    for k in sorted(store.listing(f"{PREFIX}/raw/lists/")):
        if not k.endswith(".json.gz"):
            continue
        v = _load(store, k)
        acc = {}
        walk(v[:500] if isinstance(v, list) else v, "", acc)
        out["lists"][k.rsplit("/", 1)[1]] = {"items": len(v) if isinstance(v, (list, dict)) else None,
                                            "paths": table(acc)}
    return out


def card_weights(store) -> dict:
    """In every held card snapshot: for races not yet resulted, the runners' weight status and the weight's fill, by
    hours before the scheduled off (is the on-the-day weight on the card before the race?)."""
    from datetime import datetime, timezone
    out: dict = {}
    for k in sorted(k for k in store.listing(f"{PREFIX}/raw/cards/") if k.endswith(".json.gz")):
        m = re.search(r"cards/(\d{4}-\d{2}-\d{2})/(\d{2})(\d{2})\.json\.gz$", k)
        if not m:
            continue
        snap = datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:00+00:00")
        for part in (_load(store, k) or {}).values():
            for r in part or []:
                if r.get("status") in ("complete", "results_pending") or (r.get("result") or {}).get("positions"):
                    continue
                start = (r.get("scheduled_start") or {}).get("utc")
                try:
                    hrs = (datetime.fromisoformat(str(start).replace("Z", "+00:00")) - snap).total_seconds() / 3600
                except ValueError:
                    continue
                band = "<1h" if hrs < 1 else "1-3h" if hrs < 3 else "3-12h" if hrs < 12 else "12h+"
                for p in r.get("runners") or []:
                    w = p.get("weight") or {}
                    e = out.setdefault(f"{r.get('region')}|{band}", {"runners": 0, "kg": 0, "status": {}})
                    e["runners"] += 1
                    e["kg"] += int((w.get("kg") if isinstance(w, dict) else None) is not None or p.get("weight_kg") is not None)
                    st = w.get("status") if isinstance(w, dict) else None
                    e["status"][str(st)] = e["status"].get(str(st), 0) + 1
    return out


def tables_fill(store, regions=("GB", "IE", "AU")) -> dict:
    out = {}
    for reg in regions:
        for k in sorted(store.listing(f"{PREFIX}/")):
            m = re.search(rf"{PREFIX}/runs_{reg}_(\d{{4}})\.parquet$", k)
            if not m:
                continue
            t = store.get_parquet(k)
            if t is None or not len(t):
                continue
            out.setdefault(reg, {})[m.group(1)] = {"rows": int(len(t)),
                                                   "fill": {c: round(float(t[c].notna().mean()), 3) for c in t.columns}}
    return out


def ids(store) -> dict:
    """Real ids from the latest GB results month for the probe."""
    keys = sorted(k for k in store.listing(f"{PREFIX}/raw/results/GB/") if k.endswith(".json.gz"))
    races = _load(store, keys[-1]) if keys else []
    for r in reversed(races or []):
        pos = (r.get("result") or {}).get("positions") or []
        if pos and r.get("race_id"):
            p = pos[0]
            return {"race": r.get("race_id"), "meeting": r.get("meeting_id"), "track": (r.get("track") or {}).get("track_id"),
                    "dog": p.get("dog_id"), "dogname": p.get("dog_name"), "trainer": (p.get("trainer") or {}).get("trainer_id"),
                    "sire": None}
    return {}


def probe(store, client=None, session=None) -> dict:
    import requests
    from sources.greyhoundapi import Client
    key = os.environ.get("GREYHOUNDAPI_KEY", "")
    if not key:
        return {"note": "no GREYHOUNDAPI_KEY: probe skipped"}
    s = session or requests.Session()
    out: dict = {"schema": {}, "endpoints": {}}
    for p in SCHEMA_PATHS:
        try:
            r = s.get(f"{ROOT}{p}", headers={"X-API-Key": key}, timeout=60)
        except requests.RequestException as exc:
            out["schema"][p] = {"error": type(exc).__name__}
            continue
        ct = r.headers.get("content-type", "")
        out["schema"][p] = {"status": r.status_code, "content_type": ct, "bytes": len(r.content)}
        if r.status_code == 200 and "json" in ct:
            out["schema_doc"] = r.json()
            out["schema"][p]["paths"] = sorted((out["schema_doc"].get("paths") or {}).keys())
            break
        if r.status_code == 200 and "html" in ct:
            out["schema"][p]["openapi_links"] = sorted(set(re.findall(r"[\w/.-]*openapi[\w/.-]*\.json", r.text)))[:10]
    c = client or Client(key=key)
    got = ids(store)
    sires = _load(store, f"{PREFIX}/raw/lists/sires.json.gz") or []
    got["sire"] = (sires[0] or {}).get("sire_id") if sires and isinstance(sires[0], dict) else None
    out["ids"] = {k: (str(v) if v is not None else None) for k, v in got.items()}
    for p in PROBE:
        need = re.findall(r"\{(\w+)\}", p)
        if any(got.get(n) is None for n in need):
            out["endpoints"][p] = {"skipped": "no id"}
            continue
        path = p.format(**{n: got[n] for n in need})
        try:
            j = c.get(path, tries=2)
            acc: dict = {}
            walk(j, "", acc)
            out["endpoints"][p] = {"status": 200, "paths": table(acc)}
        except RuntimeError as exc:
            m = re.search(r"HTTP (\d+)", str(exc))
            out["endpoints"][p] = {"status": int(m.group(1)) if m else None, "error": str(exc)[:160]}
    out["calls"] = c.calls
    return out


def report_md(sv: dict) -> str:
    L = ["# GreyhoundAPI field survey", ""]
    pr = sv.get("probe") or {}
    if pr:
        L += ["## The API's own schema", ""]
        for p, v in (pr.get("schema") or {}).items():
            L.append(f"- `{p}`: {v.get('status', v.get('error'))} {v.get('content_type', '')} "
                     f"{('paths: ' + ', '.join(v['paths'])) if v.get('paths') else ''} "
                     f"{('links: ' + ', '.join(v['openapi_links'])) if v.get('openapi_links') else ''}")
        L += ["", "## Endpoints probed (once each)", "", "| Endpoint | Status | Fields (path: fill) |", "|---|---|---|"]
        for p, v in (pr.get("endpoints") or {}).items():
            f = "; ".join(f"{r['path']}: {r['fill']}" for r in (v.get("paths") or [])[:60])
            L.append(f"| `{p}` | {v.get('status', v.get('skipped'))} | {f or v.get('error', '')} |")
        L.append("")
    st = sv.get("stored") or {}
    for reg, per in (st.get("results") or {}).items():
        L += [f"## Results, {reg}: fill share by sampled month", ""]
        months = sorted(per)
        paths = sorted({r["path"] for m in months for r in per[m]["paths"]})
        L += ["| Field | " + " | ".join(months) + " |", "|---|" + "---|" * len(months)]
        for p in paths:
            cells = []
            for m in months:
                hit = next((r for r in per[m]["paths"] if r["path"] == p), None)
                cells.append(f"{hit['fill']:.2f}" if hit else "-")
            L.append(f"| `{p}` | " + " | ".join(cells) + " |")
        L.append("")
    for name, block in (("Runner lists", st.get("runners") or {}),):
        for reg, v in block.items():
            L += [f"## {name}, {reg} ({v['months_held']} months held; read {', '.join(v['months_read'])}; "
                  f"{v['races']:,} races)", "", "| Field | Fill | Examples |", "|---|---|---|"]
            L += [f"| `{r['path']}` | {r['fill']} | {'; '.join(r['examples'])} |" for r in v["paths"]]
            L.append("")
    if st.get("cards"):
        v = st["cards"]
        L += [f"## Card snapshot {v['snapshot']} ({v['snapshots_held']} held)", "", "| Field | Fill | Examples |",
              "|---|---|---|"]
        L += [f"| `{r['path']}` | {r['fill']} | {'; '.join(r['examples'])} |" for r in v["paths"]]
        L.append("")
    for name, v in (st.get("lists") or {}).items():
        L += [f"## List {name} ({v['items']} items)", "", "| Field | Fill | Examples |", "|---|---|---|"]
        L += [f"| `{r['path']}` | {r['fill']} | {'; '.join(r['examples'])} |" for r in v["paths"][:80]]
        L.append("")
    cw = sv.get("card_weights") or {}
    if cw:
        L += ["## Cards before the off: is the weight there?", "", "| Region, hours to off | Runners | Weight held | Status |",
              "|---|---|---|---|"]
        for k, v in sorted(cw.items()):
            L.append(f"| {k} | {v['runners']:,} | {v['kg'] / max(v['runners'], 1):.2f} | {v['status']} |")
        L.append("")
    tf = sv.get("tables") or {}
    for reg, per in tf.items():
        years = sorted(per)
        cols = sorted({c for y in years for c in per[y]["fill"]})
        L += [f"## runs_{reg}: column fill by year", "", "| Column | " + " | ".join(years) + " |",
              "|---|" + "---|" * len(years)]
        L.append("| rows | " + " | ".join(f"{per[y]['rows']:,}" for y in years) + " |")
        for c in cols:
            L.append(f"| `{c}` | " + " | ".join(f"{per[y]['fill'].get(c, 0):.2f}" for y in years) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/api_survey")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    sv = {"stored": stored(store), "tables": tables_fill(store), "card_weights": card_weights(store)}
    if a.probe:
        sv["probe"] = probe(store)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    schema = (sv.get("probe") or {}).pop("schema_doc", None)
    body = json.dumps(sv, indent=1, default=str)
    (o / "survey.json").write_text(body)
    md = report_md(sv)
    (o / "report.md").write_text(md)
    if schema is not None:
        (o / "openapi.json").write_text(json.dumps(schema, indent=1))
    if a.save:
        store.put(f"{PREFIX}/survey/survey.json", body.encode())
        if schema is not None:
            store.put(f"{PREFIX}/survey/openapi.json", json.dumps(schema).encode())
    print(md[:60000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
