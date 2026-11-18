"""The experiment log as a table: a view of reports/research_ledger.jsonl (the owner's model brief, 18 Nov 2026:
"Log every experiment (including failures) to experiments/log.csv: id, hypothesis, change, folds, metrics,
decision").

The ledger stays the one record (append-only, checked by `python -m desk.trials --check`); this file is rebuilt from
it whole on every run, so it is idempotent and never edited by hand.

    python scripts/ledger_to_csv.py [--ledger reports/research_ledger.jsonl] [--out experiments/log.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

COLUMNS = ["id", "when_utc", "kind", "sport", "hypothesis", "change", "folds", "metrics", "decision", "line"]


def _text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def rows(ledger: str):
    with open(ledger, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield {
                "id": d.get("tag", ""),
                "when_utc": d.get("when", ""),
                "kind": d.get("kind", ""),
                "sport": d.get("sport", ""),
                "hypothesis": _text(d.get("question")),
                "change": _text(d.get("design") or d.get("features") or d.get("blocks")),
                "folds": _text(d.get("window") or d.get("run")),
                "metrics": _text(d.get("result")),
                "decision": _text(d.get("verdict") or d.get("decision") or d.get("reading")),
                "line": n,
            }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", default="reports/research_ledger.jsonl")
    ap.add_argument("--out", default="experiments/log.csv")
    a = ap.parse_args(argv)
    out = list(rows(a.ledger))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    os.replace(tmp, a.out)
    print(f"{len(out)} ledger lines -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
