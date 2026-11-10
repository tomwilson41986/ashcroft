"""The research ledger (reports/research_ledger.jsonl) kept honest by code, not by convention.

- Append-only: every line on the base branch is still there, unchanged and in order. A result cannot be rewritten
  and a registration cannot be loosened after its run (an amendment is a new line).
- New lines are checked: ``when`` (UTC, minutes), ``kind``, a ``tag`` no earlier line used, and what each kind needs.
  A preregistration carries its question, its design and its reading before the result; a desk trial carries its
  strategy, its sport and the hash of the gate it was registered with (desk/registry.py gate_hash).
- Trials are counted for the deflated Sharpe (research/overfit.py): a sport's count is its own trial lines plus every
  earlier line with no sport, so more search raises the bar and none is forgotten.

    python -m desk.trials --check --base origin/<base branch>
    python -m desk.trials --count
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from desk import SPORTS

LEDGER = Path(__file__).resolve().parents[1] / "reports" / "research_ledger.jsonl"
WHEN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$")
TRIAL_KINDS = {"research", "replay", "scan", "model", "dev", "preregistration", "preregistered_test", "query",
               "gate", "trial"}
NEEDS = {
    "preregistration": ("question", "design", "reading"),
    "trial": ("question", "design", "reading", "strategy", "sport", "gate_sha256"),
    "result": ("of", "run", "result", "verdict"),
}


def parse(text: str) -> tuple[list[dict], list[str]]:
    lines, errs = [], []
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as exc:
            errs.append(f"line {n}: not JSON ({exc})")
            continue
        if not isinstance(d, dict):
            errs.append(f"line {n}: not an object")
            continue
        lines.append(d)
    return lines, errs


def check(new_text: str, base_text: str | None) -> list[str]:
    """Every error: the base's lines changed or gone, and every new line that does not pass."""
    new_raw = [ln for ln in new_text.splitlines() if ln.strip()]
    base_raw = [ln for ln in (base_text or "").splitlines() if ln.strip()]
    errs = []
    if new_raw[:len(base_raw)] != base_raw:
        first = next((i for i, (a, b) in enumerate(zip(base_raw, new_raw)) if a != b), min(len(base_raw), len(new_raw)))
        errs.append(f"the ledger is append-only: line {first + 1} of the base changed or is gone")
        return errs
    base, _ = parse("\n".join(base_raw))
    added, perr = parse("\n".join(new_raw[len(base_raw):]))
    errs += [f"new {e}" for e in perr]
    seen = {str(d.get("tag")) for d in base}
    for i, d in enumerate(added, len(base_raw) + 1):
        where = f"line {i} ({d.get('tag', '?')})"
        for k in ("when", "kind", "tag"):
            if not d.get(k):
                errs.append(f"{where}: missing {k}")
        if d.get("when") and not WHEN.match(str(d["when"])):
            errs.append(f"{where}: when {d['when']!r} is not UTC as 2026-11-10T09:30Z")
        if d.get("tag") and str(d["tag"]) in seen:
            errs.append(f"{where}: tag already used")
        seen.add(str(d.get("tag")))
        for k in NEEDS.get(str(d.get("kind")), ()):
            if d.get(k) in (None, ""):
                errs.append(f"{where}: a {d['kind']} line needs {k}")
        if d.get("sport") is not None and d["sport"] not in SPORTS:
            errs.append(f"{where}: sport {d['sport']!r} not one of {SPORTS}")
    return errs


def count_trials(lines: list[dict], sport: str | None = None) -> int:
    """Trials counted against a sport: its own trial lines and every trial line with no sport (conservative)."""
    return sum(1 for d in lines if str(d.get("kind")) in TRIAL_KINDS
               and (sport is None or d.get("sport") in (None, sport)))


def _git_show(ref: str, path: Path) -> str | None:
    rel = path.resolve().relative_to(Path(__file__).resolve().parents[1])
    try:
        return subprocess.run(["git", "show", f"{ref}:{rel.as_posix()}"], capture_output=True, text=True,
                              check=True).stdout
    except subprocess.CalledProcessError:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--base", default=None, help="git ref holding the ledger to compare against (append-only)")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    a = ap.parse_args(argv)
    text = a.ledger.read_text()
    rc = 0
    if a.check:
        base = _git_show(a.base, a.ledger) if a.base else None
        if a.base and base is None:
            print(f"no ledger at {a.base}: every line checked as new", file=sys.stderr)
        errs = check(text, base)
        for e in errs:
            print("ERROR", e, file=sys.stderr)
        print(f"ledger: {len(text.splitlines())} lines, {len(errs)} errors")
        rc = 1 if errs else 0
    if a.count or not a.check:
        lines, _ = parse(text)
        for s in SPORTS:
            print(f"{s:13} {count_trials(lines, s)} trials")
    return rc


if __name__ == "__main__":
    sys.exit(main())
