"""The algorithm inventory: every strategy and every trader that can place an order, one JSON file each under
``desk/strategies/<sport>/<id>.json``.

A strategy's file names what it is (hypothesis, mechanism, markets), where it runs (entry point, workflow, the switch
that makes it real), how its orders are known on the account (the customerOrderRef prefix), how it is stopped (the S3
STOP objects it honours) and, for every strategy the desk itself runs, its pre-registered gate and the gate's hash.

Stages:
    legacy    a trader the owner put live before the desk existed: run as it is, governed from outside (the warden)
    research  registered, scored on history and on forward days; no money
    paper     scored on the live feed, forward days only; no money
    pilot     real money at the pilot budget (desk.envelope)
    scaled    real money, one stake step at a time
    retired   no orders

The gate is pinned when a strategy is registered: ``gate_sha256`` is the hash of its gate, so a gate edited after
results are seen no longer matches and the strategy cannot move (an amendment is a new strategy id).

    python -m desk.registry --check          # validate every file; exit 1 on any error
    python -m desk.registry --list           # the inventory as a table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from desk import SPORTS

ROOT = Path(__file__).resolve().parent
STRATEGIES = ROOT / "strategies"
STAGES = ("legacy", "research", "paper", "pilot", "scaled", "retired")
MONEY_STAGES = ("pilot", "scaled")
BUCKETS = ("pred", "capture")          # ULTRA_BETTING_S3_BUCKET (the horse trader's) or CAPTURE_BUCKET (the rest)
LEGACY_PREFIX = "ash"                  # the live horse trader's: no desk strategy may start with it (session.py sync)
REQUIRED = ("id", "sport", "name", "stage", "hypothesis", "entry_point", "workflow", "order_ref_prefix",
            "stop_keys", "kill_switch_fails")
GATE_FIELDS = ("metric", "min_days", "min_bets", "interval", "dsr", "read")
MIN_DSR = 0.95                         # CLAUDE.md, the owner 5 and 9 Nov 2026: research/overfit.py DSR_THRESHOLD
MIN_INTERVAL = 0.90
MIN_DAYS = 30                          # research/overfit.py MIN_OBS: no Sharpe worth deflating on fewer (tests hold them equal;
                                       # not imported, so the warden's slim install can read the registry)
LIMIT_FIELDS = ("max_order", "max_day_turnover", "max_day_loss", "max_market_exposure", "max_orders_per_minute",
                "price_collar", "max_book_age_s", "min_stake")


@dataclass(frozen=True)
class Strategy:
    id: str
    sport: str
    name: str
    stage: str
    hypothesis: str
    entry_point: str
    workflow: str
    order_ref_prefix: str | None
    stop_keys: tuple
    kill_switch_fails: str
    bucket: str = "capture"
    live_switch: str | None = None
    dashboard_ids: tuple = ()
    mechanism: str = ""
    markets: tuple = ()
    evidence: tuple = ()
    approved: dict | None = None
    gate: dict | None = None
    gate_sha256: str | None = None
    limits: dict | None = None
    ledger_tag: str | None = None
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def desk_run(self) -> bool:
        return self.stage != "legacy"

    @property
    def params(self) -> dict:
        """A desk strategy's frozen parameters (its algo's constructor arguments), as registered."""
        return dict(self.raw.get("params") or {})


def gate_hash(gate: dict) -> str:
    """The pinned hash of a gate: canonical JSON (sorted keys, no spaces), SHA-256."""
    return hashlib.sha256(json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _strategy(d: dict) -> Strategy:
    return Strategy(
        id=d["id"], sport=d["sport"], name=d["name"], stage=d["stage"], hypothesis=d["hypothesis"],
        entry_point=d["entry_point"], workflow=d["workflow"], order_ref_prefix=d.get("order_ref_prefix"),
        stop_keys=tuple(d.get("stop_keys") or ()), kill_switch_fails=d["kill_switch_fails"],
        bucket=d.get("bucket", "capture"), live_switch=d.get("live_switch"), dashboard_ids=tuple(d.get("dashboard_ids") or ()),
        mechanism=d.get("mechanism", ""), markets=tuple(d.get("markets") or ()),
        evidence=tuple(json.dumps(e, sort_keys=True) for e in d.get("evidence") or ()), approved=d.get("approved"),
        gate=d.get("gate"), gate_sha256=d.get("gate_sha256"), limits=d.get("limits"), ledger_tag=d.get("ledger_tag"),
        raw=d)


def check_one(d: dict, path: Path | None = None) -> list[str]:
    """Every error in one strategy's file; [] when it is sound on its own."""
    where = str(path) if path else d.get("id", "?")
    errs = [f"{where}: missing {k}" for k in REQUIRED if k not in d]
    if errs:
        return errs
    if path is not None and path.stem != d["id"]:
        errs.append(f"{where}: the file name is not the id {d['id']!r}")
    if path is not None and path.parent.name != d["sport"]:
        errs.append(f"{where}: filed under {path.parent.name!r}, sport is {d['sport']!r}")
    if d["sport"] not in SPORTS:
        errs.append(f"{where}: sport {d['sport']!r} not one of {SPORTS}")
    if d["stage"] not in STAGES:
        errs.append(f"{where}: stage {d['stage']!r} not one of {STAGES}")
    if d.get("bucket", "capture") not in BUCKETS:
        errs.append(f"{where}: bucket {d.get('bucket')!r} not one of {BUCKETS}")
    if d["kill_switch_fails"] not in ("open", "safe"):
        errs.append(f"{where}: kill_switch_fails must be 'open' or 'safe'")
    if not d["stop_keys"] or not all(str(k).endswith("STOP") for k in d["stop_keys"]):
        errs.append(f"{where}: stop_keys must name the S3 STOP objects it honours")
    for e in d.get("evidence") or ():
        if not (isinstance(e, dict) and e.get("claim") and e.get("source")):
            errs.append(f"{where}: every evidence item needs a claim and its source")
    prefix = d["order_ref_prefix"]
    if d["stage"] == "legacy":
        a = d.get("approved") or {}
        if not (a.get("by") and a.get("on") and a.get("source")):
            errs.append(f"{where}: a legacy trader needs approved.by, approved.on and approved.source")
        return errs
    # a strategy the desk runs
    if d["kill_switch_fails"] != "safe":
        errs.append(f"{where}: a desk strategy's kill switch must fail safe")
    if not prefix:
        errs.append(f"{where}: a desk strategy needs its own order_ref_prefix")
    elif str(prefix).startswith(LEGACY_PREFIX):
        errs.append(f"{where}: prefix {prefix!r} starts with {LEGACY_PREFIX!r}: the live horse trader would adopt it")
    if not d.get("ledger_tag"):
        errs.append(f"{where}: a desk strategy needs the research ledger tag of its registration")
    gate = d.get("gate")
    if not isinstance(gate, dict) or any(k not in gate for k in GATE_FIELDS):
        errs.append(f"{where}: the gate needs {GATE_FIELDS}")
    else:
        if float(gate["dsr"]) < MIN_DSR:
            errs.append(f"{where}: gate dsr {gate['dsr']} under {MIN_DSR}")
        if float(gate["interval"]) < MIN_INTERVAL:
            errs.append(f"{where}: gate interval {gate['interval']} under {MIN_INTERVAL}")
        if int(gate["min_days"]) < MIN_DAYS:
            errs.append(f"{where}: gate min_days {gate['min_days']} under {MIN_DAYS} (the deflated Sharpe's minimum)")
        if gate["read"] != "forward":
            errs.append(f"{where}: the gate reads forward days only (history earns paper, never money)")
        if d.get("gate_sha256") != gate_hash(gate):
            errs.append(f"{where}: gate_sha256 does not match the gate: a gate changed after registration")
    if d["stage"] in MONEY_STAGES:
        lim = d.get("limits") or {}
        missing = [k for k in LIMIT_FIELDS if k not in lim]
        if missing:
            errs.append(f"{where}: stage {d['stage']} needs limits {missing}")
    return errs


def load(root: Path = STRATEGIES) -> tuple[list[Strategy], list[str]]:
    """Every strategy under root and every error: per file, then across files (ids and prefixes unique)."""
    out, errs = [], []
    for p in sorted(root.glob("*/*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            errs.append(f"{p}: not JSON ({exc})")
            continue
        e = check_one(d, p)
        errs += e
        if not e:
            out.append(_strategy(d))
    ids = [s.id for s in out]
    errs += [f"duplicate id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    prefixes = [s.order_ref_prefix for s in out if s.order_ref_prefix]
    for a in prefixes:
        for b in prefixes:
            if a is not b and a != b and b.startswith(a):
                errs.append(f"prefix {b!r} starts with {a!r}: their orders could not be told apart")
    errs += [f"duplicate prefix {x!r}" for x in sorted({x for x in prefixes if prefixes.count(x) > 1})]
    return out, errs


def owner_of(ref: str | None, strategies: list[Strategy]) -> Strategy | None:
    """The strategy an order belongs to: the longest registered prefix its customerOrderRef starts with."""
    ref = str(ref or "")
    best = None
    for s in strategies:
        p = s.order_ref_prefix
        if p and ref.startswith(p) and (best is None or len(p) > len(best.order_ref_prefix)):
            best = s
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    strategies, errs = load()
    if a.list or not a.check:
        print(f"{'id':32} {'sport':13} {'stage':8} {'prefix':7} {'switch':22} workflow")
        for s in strategies:
            print(f"{s.id:32} {s.sport:13} {s.stage:8} {str(s.order_ref_prefix or '-'):7} "
                  f"{str(s.live_switch or '-'):22} {s.workflow}")
    for e in errs:
        print("ERROR", e, file=sys.stderr)
    print(f"{len(strategies)} strategies, {len(errs)} errors")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
