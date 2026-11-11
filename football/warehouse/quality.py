"""The warehouse's data-quality report: what must hold, checked on what is stored.

``report(store)`` reads the manifests and the tables (key columns only; never the ticks themselves) and returns
every check with its value; ``ok`` is false if a hard check fails, and ``python -m football.warehouse check`` exits 1
then. Hard checks are what the layers promise each other:

- bronze -> silver: every bronze partition is built from its current fingerprint; files in = files read + files
  that could not be read + files superseded by a longer copy; each partition's own reconciliation passed; no
  traded price outside Betfair's range (1.01 to 1000);
- keys: every table's key is unique (market ids and event ids across partitions too) and every gold table's rows
  refer to a market or an event that exists;
- point in time: no checkpoint carries a price published after its moment;
- arithmetic: a complete book's normalised chances sum to one; implied totals and supremacies are in range.

Soft checks are reported, not enforced: the share of files that could not be read, result coverage, and the closing
1X2 chances against the results (the market should be calibrated in aggregate).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from football.warehouse import bronze, gold
from football.warehouse.schema import BRONZE_RAW, SILVER, SILVER_MARKET
from football.warehouse.silver import MANIFEST as SILVER_MANIFEST
from football.warehouse.store_io import list_parts, part_name, read_manifest, read_part, read_single, read_table


def _unique(t: pa.Table, key: list[str]) -> bool:
    return t.num_rows == 0 or t.group_by(key).aggregate([]).num_rows == t.num_rows


def _check(out: list, name: str, ok: bool, value=None, hard: bool = True, detail: str | None = None) -> None:
    out.append({"check": name, "ok": bool(ok), "hard": hard, "value": value, **({"detail": detail} if detail else {})})


def report(store) -> dict:
    checks: list[dict] = []
    parts = bronze.partitions(store)
    bman = read_manifest(store, BRONZE_RAW.name)
    sman = read_manifest(store, SILVER_MANIFEST)

    # -- bronze -> silver ---------------------------------------------------------------------------------------
    stale, unrec, failed_checks = [], [], []
    files_in = files_failed = superseded = 0
    outside = invalid = 0
    for y, m in parts:
        name = part_name(y, m)
        b, s = bman["parts"].get(name) or {}, sman["parts"].get(name) or {}
        if not s or s.get("bronze_fingerprint") != b.get("fingerprint"):
            stale.append(name)
            continue
        files_in += s.get("files_in", 0)
        files_failed += s.get("files_failed", 0)
        superseded += s.get("superseded", 0)
        if s.get("files_in") != s.get("files_read", 0) + s.get("files_failed", 0) + s.get("superseded", 0):
            unrec.append(name)
        if any(k.endswith("_ok") and not v for k, v in (s.get("checks") or {}).items()):
            failed_checks.append(name)
        outside += (s.get("checks") or {}).get("ticks_outside_1.01_1000", 0)
        invalid += (s.get("checks") or {}).get("ltp_invalid_left_out", 0)
    _check(checks, "bronze partitions held", bool(parts), len(parts))
    _check(checks, "silver built from the current bronze", not stale, stale or "all")
    _check(checks, "files in = read + unreadable + superseded", not unrec, unrec or "all")
    _check(checks, "each silver partition's reconciliation", not failed_checks, failed_checks or "all")
    _check(checks, "traded prices within 1.01 to 1000", outside == 0, outside)
    _check(checks, "price updates outside Betfair's range, left out of silver/tick", True, invalid, hard=False)
    _check(checks, "share of files that could not be read", files_failed <= 0.001 * max(files_in, 1),
           {"files_in": files_in, "unreadable": files_failed, "superseded": superseded}, hard=False)

    # -- keys across partitions ---------------------------------------------------------------------------------
    market = read_table(store, SILVER_MARKET.name, ["market_id", "event_id", "market_time"], with_partition=True)
    n_market = market.num_rows if market is not None else 0
    market_ids = pc.unique(market.column("market_id")) if market is not None else pa.array([], pa.string())
    _check(checks, "silver/market: market_id unique across partitions", len(market_ids) == n_market, n_market)
    for spec in SILVER:
        if spec.name in (SILVER_MARKET.name, "silver/tick"):
            continue
        bad = []
        for y, m in parts:
            t = read_part(store, spec.name, y, m, ["market_id"])
            known = t is None or not t.num_rows or pc.all(pc.is_in(t.column("market_id"), value_set=market_ids)).as_py()
            if not known:
                bad.append(part_name(y, m))
        _check(checks, f"{spec.name}: every row's market exists", not bad, bad or "all")
    if market is not None and n_market:
        in_month = pc.sum(pc.and_(pc.equal(pc.year(market.column("market_time")), market.column("year")),
                                  pc.equal(pc.month(market.column("market_time")), market.column("month")))).as_py()
        _check(checks, "markets whose kick-off month is their partition's", True,
               {"share": round(in_month / n_market, 4),
                "kickoff_min": str(pc.min(market.column("market_time")).as_py()),
                "kickoff_max": str(pc.max(market.column("market_time")).as_py())}, hard=False)

    # -- gold ---------------------------------------------------------------------------------------------------
    events = None
    rows: dict[str, int] = {}
    for mod in gold.order(gold.load()):
        for spec in mod.SPECS:
            key = list(spec.key)
            if mod.SCOPE == "all":
                t = read_single(store, spec.name, key)
                if t is None:
                    _check(checks, f"{spec.name}: built", False, "missing", hard=False)
                    continue
                rows[spec.name] = t.num_rows
                _check(checks, f"{spec.name}: key unique", _unique(t, key), t.num_rows)
                continue
            have = list_parts(store, spec.name)
            if not have:
                _check(checks, f"{spec.name}: built", False, "missing", hard=False)
                continue
            dup, orphan, n = [], [], 0
            ref = "market_id" if "market_id" in spec.names else "event_id" if "event_id" in spec.names else None
            for y, m in have:
                t = read_part(store, spec.name, y, m, key + ([ref] if ref and ref not in key else []))
                n += t.num_rows
                if not _unique(t, key):
                    dup.append(part_name(y, m))
                if ref == "market_id" and t.num_rows and not pc.all(
                        pc.is_in(t.column("market_id"), value_set=market_ids)).as_py():
                    orphan.append(part_name(y, m))
            rows[spec.name] = n
            _check(checks, f"{spec.name}: key unique in every partition", not dup, dup or n)
            if ref == "market_id":
                _check(checks, f"{spec.name}: every row's market exists", not orphan, orphan or "all")
            missing = [part_name(y, m) for y, m in parts if (y, m) not in have]
            _check(checks, f"{spec.name}: every partition built", not missing, missing or len(have), hard=False)

    events = read_table(store, "gold/event", ["event_id", "went_in_play", "void", "result", "close_legs", "p_home",
                                              "p_draw", "p_away", "close_age_s"])
    if events is not None:
        n_ev = events.num_rows
        _check(checks, "gold/event: event_id unique across partitions",
               len(pc.unique(events.column("event_id"))) == n_ev, n_ev)
        if market is not None:
            _check(checks, "gold/event: one row for every event in silver/market",
                   n_ev == len(pc.unique(market.column("event_id"))), n_ev)
        e = events.to_pandas()
        played = e[e["went_in_play"] & ~e["void"]]
        cov = float(played["result"].notna().mean()) if len(played) else None
        _check(checks, "gold/event: matches played that have a result", cov is None or cov >= 0.995,
               None if cov is None else round(cov, 5), hard=False)
        fresh = e[(e["close_legs"] == 3) & e["result"].notna() & (e["close_age_s"] < 600)]
        if len(fresh) >= 1000:
            cal = {k: (round(float(fresh[f"p_{k}"].mean()), 4), round(float((fresh["result"] == r).mean()), 4))
                   for k, r in (("home", "H"), ("draw", "D"), ("away", "A"))}
            gap = max(abs(a - b) for a, b in cal.values())
            _check(checks, "closing 1X2 chances against results (mean chance, rate)", gap < 0.01,
                   {"matches": int(len(fresh)), **cal}, hard=False)

    cp_bad_time, cp_bad_sum, cp_rows = [], [], 0
    for y, m in list_parts(store, "gold/checkpoint"):
        t = read_part(store, "gold/checkpoint", y, m, ["market_id", "cp_min", "line", "age_s", "p_norm",
                                                         "book_complete"])
        cp_rows += t.num_rows
        if t.num_rows and pc.min(t.column("age_s")).as_py() < 0:
            cp_bad_time.append(part_name(y, m))
        full = t.filter(t.column("book_complete"))
        if full.num_rows:
            s = full.group_by(["market_id", "cp_min", "line"]).aggregate([("p_norm", "sum")]).column("p_norm_sum")
            if pc.max(pc.abs(pc.subtract(s, 1.0))).as_py() > 1e-9:
                cp_bad_sum.append(part_name(y, m))
    if cp_rows:
        _check(checks, "gold/checkpoint: no price published after its checkpoint", not cp_bad_time,
               cp_bad_time or "all")
        _check(checks, "gold/checkpoint: a complete book's chances sum to one", not cp_bad_sum, cp_bad_sum or "all")

    implied = read_table(store, "gold/implied", ["total", "supremacy", "p_home", "p_draw", "p_away"])
    if implied is not None and implied.num_rows:
        i = implied.to_pandas()
        r = i[i["total"].notna() & i["supremacy"].notna()]
        ok = bool(((r["total"] > 0) & (r["supremacy"].abs() < r["total"])).all())
        s3 = i[["p_home", "p_draw", "p_away"]].dropna().sum(axis=1)
        _check(checks, "gold/implied: totals positive, supremacy inside the total", ok, int(len(r)))
        _check(checks, "gold/implied: 1X2 chances sum to one", bool(((s3 - 1).abs() < 1e-9).all()), int(len(s3)))

    hard_failed = [c["check"] for c in checks if c["hard"] and not c["ok"]]
    soft_failed = [c["check"] for c in checks if not c["hard"] and not c["ok"]]
    return {"ok": not hard_failed, "hard_failed": hard_failed, "soft_failed": soft_failed, "rows": rows,
            "partitions": [part_name(y, m) for y, m in parts], "checks": checks}


def markdown(rep: dict) -> str:
    """The report as a table for a run summary."""
    lines = [f"**Warehouse quality: {'PASS' if rep['ok'] else 'FAIL'}** "
             f"({len(rep['checks'])} checks; hard failures: {len(rep['hard_failed'])}; soft: "
             f"{len(rep['soft_failed'])})", "", "| Check | Result | Value |", "|---|---|---|"]
    for c in rep["checks"]:
        mark = "pass" if c["ok"] else ("FAIL" if c["hard"] else "warn")
        lines.append(f"| {c['check']} | {mark} | {str(c['value'])[:160]} |")
    return "\n".join(lines) + "\n"
