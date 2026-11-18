"""Betfair's own description of every GB/IE ante-post win market open now: one listMarketCatalogue call with
MARKET_DESCRIPTION (the rules text, the BSP flag, the suspend time, turn in play). Read-only, the delayed key; no
order call. Kept apart from phase0.py so the UK runner needs requests and python-dotenv only (antepost-rules.yml).

    python -m antepost.rules --out out/antepost_rules
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FIELDS = ("bspMarket", "turnInPlayEnabled", "persistenceEnabled", "marketBaseRate", "discountAllowed", "bettingType",
          "suspendTime", "marketTime", "settleTime", "rulesHasDate", "clarifications", "raceType", "eachWayDivisor")


def strip_html(s: str) -> str:
    s = re.sub(r"(?i)<br\s*/?>|</p>|</li>", "\n", s or "")
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def summarise(cat: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    rows, texts = [], {}
    for m in cat or []:
        d, ev = m.get("description") or {}, m.get("event") or {}
        texts.setdefault(strip_html(d.get("rules") or ""), []).append(m.get("marketId"))
        rows.append({"market_id": m.get("marketId"), "market_name": m.get("marketName"), "event": ev.get("name"),
                     "country": ev.get("countryCode"), "market_start": m.get("marketStartTime"),
                     "runners": len(m.get("runners") or []), "total_matched": m.get("totalMatched"),
                     **{k: d.get(k) for k in FIELDS}})
    return rows, texts


def read_rules(out: Path, client=None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if client is None:
        from betfair_client import BetfairClient
        client = BetfairClient()
        client.login()
    now = datetime.now(timezone.utc)
    try:
        cat = client._api_call("listMarketCatalogue", {
            "filter": {"eventTypeIds": ["7"], "marketTypeCodes": ["ANTEPOST_WIN"], "marketCountries": ["GB", "IE"],
                       "marketStartTime": {"from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                           "to": (now + timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%SZ")}},
            "marketProjection": ["EVENT", "MARKET_START_TIME", "MARKET_DESCRIPTION", "RUNNER_DESCRIPTION"],
            "maxResults": 200, "sort": "FIRST_TO_START"}) or []
    finally:
        try:
            client.logout()
        except Exception:
            pass
    rows, texts = summarise(cat)
    rep = {"read_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": len(rows), "markets_list": rows,
           "distinct_rules_texts": [{"markets": len(v), "example_market": v[0], "text": k[:20000]}
                                    for k, v in sorted(texts.items(), key=lambda kv: -len(kv[1]))]}
    (out / "antepost_rules.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out/antepost_rules")
    a = ap.parse_args(argv)
    print(json.dumps(read_rules(Path(a.out)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
