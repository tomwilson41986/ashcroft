"""The warden: the account seen whole, every trader at once, against the owner's envelope and the registry.

Read-only on Betfair (the delayed key): the account's current orders, today's settled bets and the funds. It finds
    - an order whose reference no registered strategy owns (a trader the registry does not know),
    - the day's settled loss over the desk's day loss, or over a sport's share of it,
    - the account's exposure over the bankroll,
    - a market whose worst case over every order on it is over the desk's market exposure (reported),
and reports orders placed by hand (no reference) and settled P&L by strategy from Betfair's own record.

It writes nothing to S3 unless run with ``--enforce``: then a breach writes the STOP objects the traders already
honour (a sport's own switches, or ``trading/STOP`` in both buckets for the whole account). It never cancels or places
an order: ``READS`` is the only Betfair surface it has. Without the envelope it checks the registry alone.

    python -m desk.warden --out out/warden.json              # report; exit 2 on a breach
    python -m desk.warden --enforce                          # and write the STOP objects
    python -m desk.warden --issue                            # and tell the owner on a GitHub issue

The two calls without market ids (listCurrentOrders for the whole account, listClearedOrders by settledDateRange) are
the first in this repository made that way; the first dispatched run is their check, and an error is reported, not
treated as a breach.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from desk import account, envelope as envmod, registry

UK = ZoneInfo("Europe/London")
READS = ("listCurrentOrders", "listClearedOrders")
GLOBAL_STOP = "trading/STOP"


@dataclass
class Finding:
    level: str                      # breach | warn | info | error
    kind: str
    message: str
    stops: list = field(default_factory=list)       # [(bucket kind, key)] written with --enforce


def _sport_stops(sport: str, strategies) -> list[tuple[str, str]]:
    """The switches that stop a sport: each of its strategies' own key; a strategy whose only switch is the
    account-wide trading/STOP can be stopped only by the account-wide stop."""
    out = set()
    for s in strategies:
        if s.sport != sport or not s.order_ref_prefix:
            continue
        own = [k for k in s.stop_keys if k != GLOBAL_STOP]
        out |= {(s.bucket, k) for k in own} if own else {("pred", GLOBAL_STOP), ("capture", GLOBAL_STOP)}
    return sorted(out)


ALL_STOP = [("pred", GLOBAL_STOP), ("capture", GLOBAL_STOP)]


def assess(strategies, env: envmod.Envelope | None, env_reason: str, orders: list[dict], cleared: list[dict],
           funds: dict | None) -> tuple[list[Finding], dict]:
    """Every finding and the day's figures, from Betfair's record. Pure: tested without the network."""
    f: list[Finding] = []
    by = account.by_strategy(orders, strategies)
    settled = account.settled_by_strategy(cleared, strategies)
    if by.get(account.UNKNOWN):
        refs = sorted({str(o.get("customerOrderRef")) for o in by[account.UNKNOWN]})[:10]
        f.append(Finding("breach", "unknown-reference",
                         f"{len(by[account.UNKNOWN])} open orders under references no registered strategy owns: {refs}",
                         ALL_STOP))
    if settled.get(account.UNKNOWN):
        f.append(Finding("breach", "unknown-reference-settled",
                         f"{settled[account.UNKNOWN]['bets']} bets settled today under references no registered "
                         f"strategy owns (GBP{settled[account.UNKNOWN]['profit']:+,.2f})", ALL_STOP))
    if by.get(account.UNATTRIBUTED):
        f.append(Finding("info", "by-hand", f"{len(by[account.UNATTRIBUTED])} open orders with no reference "
                                            "(placed by hand): counted in the account's figures"))
    total = round(sum(v["profit"] for v in settled.values()), 2)
    sport_of = {s.id: s.sport for s in strategies}
    by_sport = defaultdict(float)
    for sid, v in settled.items():
        if sid in sport_of:
            by_sport[sport_of[sid]] += v["profit"]
    worst = account.market_worst(orders)
    exposure = None if funds is None else abs(float(funds.get("exposure") or 0.0))
    if funds is None:
        f.append(Finding("error", "funds", "the account's funds could not be read"))
    if env is None:
        f.append(Finding("info", "no-envelope", env_reason + "; the account is checked against the registry only"))
    else:
        if total <= -env.daily_loss:
            f.append(Finding("breach", "day-loss", f"settled today GBP{total:+,.2f}, the desk's day loss is "
                                                   f"GBP{env.daily_loss:,.2f}", ALL_STOP))
        for sport, pnl in sorted(by_sport.items()):
            cap = env.sport_daily_loss(sport)
            if pnl <= -cap:
                f.append(Finding("breach", "sport-day-loss", f"{sport} settled today GBP{pnl:+,.2f}, its share of the "
                                                             f"day loss is GBP{cap:,.2f}", _sport_stops(sport, strategies)))
        if exposure is not None and exposure > env.bankroll:
            f.append(Finding("breach", "exposure", f"the account's exposure GBP{exposure:,.2f} is over the bankroll "
                                                   f"GBP{env.bankroll:,.2f}", ALL_STOP))
        over = {m: w for m, w in worst.items() if -w > env.max_market_exposure}
        if over:
            f.append(Finding("warn", "market-exposure",
                             f"{len(over)} markets' worst case over GBP{env.max_market_exposure:,.2f} (every order "
                             f"on them, the owner's traders included): " +
                             ", ".join(f"{m} GBP{w:,.2f}" for m, w in sorted(over.items(), key=lambda x: x[1])[:5])))
    figures = {"settled_total": total, "settled_by_strategy": settled,
               "settled_by_sport": {k: round(v, 2) for k, v in by_sport.items()},
               "open_orders_by_strategy": {k: len(v) for k, v in by.items()},
               "markets_open": len(worst), "worst_market": round(min(worst.values()), 2) if worst else None,
               "exposure": exposure, "available": None if funds is None else funds.get("availableToBetBalance")}
    return f, figures


# --- I/O -------------------------------------------------------------------------------------------------------------

def _paged(client, method: str, params: dict, key: str) -> list[dict]:
    if method not in READS:
        raise ValueError(f"{method}: the warden only reads")
    out, start = [], 0
    while True:
        reply = client._api_call(method, {**params, "fromRecord": start, "recordCount": 1000}) or {}
        got = reply.get(key, []) or []
        out += got
        if not reply.get("moreAvailable") or not got:
            return out
        start += len(got)


def read_account(client, now: datetime) -> tuple[list[dict], list[dict], dict | None, list[str]]:
    errs = []
    day0 = datetime(now.astimezone(UK).year, now.astimezone(UK).month, now.astimezone(UK).day, tzinfo=UK)
    span = {"from": day0.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    orders = cleared = []
    funds = None
    try:
        orders = _paged(client, "listCurrentOrders", {"orderProjection": "ALL"}, "currentOrders")
    except Exception as exc:
        errs.append(f"listCurrentOrders: {str(exc)[:200]}")
    try:
        cleared = _paged(client, "listClearedOrders", {"betStatus": "SETTLED", "settledDateRange": span},
                         "clearedOrders")
    except Exception as exc:
        errs.append(f"listClearedOrders: {str(exc)[:200]}")
    try:
        funds = client.account_funds()
    except Exception as exc:
        errs.append(f"getAccountFunds: {str(exc)[:200]}")
    return orders, cleared, funds, errs


def bucket_for(kind: str) -> str:
    """The bucket each trader reads its switch from: the horse trader's (auto_trade.py BUCKET, morning/live.py
    bucket) or the capture bucket (betfair_recorder.bucket)."""
    if kind == "pred":
        return os.getenv("ULTRA_BETTING_S3_BUCKET") or os.getenv("PRED_BUCKET") or "ashcroft"
    return os.getenv("CAPTURE_BUCKET") or os.getenv("ULTRA_BETTING_S3_BUCKET") or "ashcroft"


def write_stops(stops, reason: str, put=None) -> list[str]:
    if put is None:
        import boto3
        s3 = boto3.client("s3")

        def put(bucket, key, body):
            s3.put_object(Bucket=bucket, Key=key, Body=body)
    done = []
    for kind, key in sorted(set(stops)):
        b = bucket_for(kind)
        if f"{b}/{key}" in done:
            continue
        put(b, key, reason.encode())
        done.append(f"{b}/{key}")
    return done


def report_md(findings: list[Finding], fig: dict, day: str, enforced: list[str], errs: list[str]) -> str:
    lines = [f"## Desk warden, {day}", ""]
    for x in findings:
        lines.append(f"- **{x.level}** {x.kind}: {x.message}")
    for e in errs:
        lines.append(f"- **error** read: {e}")
    if not findings and not errs:
        lines.append("- nothing found")
    unread = any(e.startswith(("login", "listClearedOrders")) for e in errs)
    total = "no data found" if unread else f"GBP{fig['settled_total']:+,.2f}"
    lines += ["", f"Settled today (Betfair's listClearedOrders, profit before commission): {total}",
              "", "| strategy | bets | profit |", "|---|---|---|"]
    for k, v in sorted(fig["settled_by_strategy"].items()):
        lines.append(f"| {k} | {v['bets']} | {v['profit']:+,.2f} |")
    if not fig["settled_by_strategy"]:
        lines.append("| no data found | | |")
    lines += ["", f"Open orders by strategy: {fig['open_orders_by_strategy'] or 'no data found'}",
              f"Account exposure: {fig['exposure'] if fig['exposure'] is not None else 'no data found'}"]
    lines.append("STOP written: " + (", ".join(enforced) if enforced else "none"))
    return "\n".join(lines) + "\n"


def tell_owner(day: str, text: str) -> str:
    """Open the day's issue ("Desk warden: <day>") or comment on it, as scripts/session_watchdog.py tells the owner."""
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    repo, token = os.environ["GITHUB_REPOSITORY"], os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")

    def gh(method, path, body=None):
        req = urllib.request.Request(f"{base}/repos/{repo}/{path}", method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
        return json.loads(raw) if raw else {}
    title = f"Desk warden: {day}"
    mine = [i for i in gh("GET", "issues?state=open&per_page=50") or [] if i.get("title") == title]
    if not mine:
        gh("POST", "issues", {"title": title, "body": text})
        return "opened the issue"
    seen = [mine[0].get("body") or ""] + [c.get("body") or "" for c in
                                          gh("GET", f"issues/{mine[0]['number']}/comments?per_page=100") or []]
    key = text.splitlines()[-1]
    if any(key in b for b in seen):
        return "already told today"
    gh("POST", f"issues/{mine[0]['number']}/comments", {"body": text})
    return "commented on the issue"


def alert_key(findings: list[Finding], errs: list[str]) -> str:
    """One alert a day for each set of problems: a breach or a read error that persists is not repeated every run."""
    kinds = sorted({f"{x.level}:{x.kind}" for x in findings if x.level in ("breach", "error")} |
                   {"error:" + e.split(":")[0] for e in errs})
    return f"<!-- desk-warden {','.join(kinds)} -->"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--enforce", action="store_true", help="write the STOP objects on a breach")
    ap.add_argument("--issue", action="store_true", help="open or comment on the day's GitHub issue on a breach")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    strategies, reg_errs = registry.load()
    env, why = envmod.from_env()
    now = datetime.now(timezone.utc)
    day = now.astimezone(UK).date().isoformat()
    from betfair_client import BetfairClient
    client = BetfairClient()
    errs = list(f"registry: {e}" for e in reg_errs)
    try:
        client.login()
        orders, cleared, funds, read_errs = read_account(client, now)
        errs += read_errs
    except Exception as exc:
        orders, cleared, funds = [], [], None
        errs.append(f"login: {str(exc)[:200]}")
    findings, fig = assess(strategies, env, why, orders, cleared, funds)
    breaches = [x for x in findings if x.level == "breach"]
    enforced = []
    if breaches and a.enforce:
        enforced = write_stops([s for x in breaches for s in x.stops],
                               f"desk warden {now:%Y-%m-%dT%H:%MZ}: " + "; ".join(x.kind for x in breaches))
    text = report_md(findings, fig, day, enforced, errs)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"at": now.isoformat(), "findings": [asdict(x) for x in findings],
                                     "figures": fig, "errors": errs, "stops_written": enforced}, indent=2, default=str))
    if (breaches or errs) and a.issue:
        try:
            print(tell_owner(day, text + alert_key(findings, errs) + "\n"))
        except Exception as exc:
            print(f"the issue could not be written: {exc}", file=sys.stderr)
    return 2 if breaches else (3 if errs else 0)


if __name__ == "__main__":
    sys.exit(main())
