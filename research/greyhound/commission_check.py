"""What Betfair charges this account on greyhound markets (read-only; the owner, 6 Nov 2026: "check betfair to see
what commission we pay on greyhounds. Then set this as standard").

Three readings, each from Betfair itself:
- the account (Accounts API getAccountDetails): its commission discount rate and points balance (no name or other
  personal detail is printed);
- the markets (listMarketCatalogue, MARKET_DESCRIPTION): the market base rate Betfair quotes on every greyhound WIN
  and PLACE market in the next 24 hours, by country, and whether a discount applies;
- the account's settled greyhound markets (listClearedOrders, SETTLED, grouped by market, the last 90 days, all
  Betfair keeps): the commission Betfair actually took on each market against the market's profit, so the rate paid
  is read off the money rather than assumed.

No order method is called. Writes out/commission/report.md and report.json.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

GREYHOUNDS = "4339"
COUNTRIES = ["GB", "IE", "AU", "NZ", "US"]


def _account(client, method: str, params: dict) -> dict:
    from betfair_client import ACCOUNT_URL
    headers = {"X-Application": client.app_key, "X-Authentication": client.session_token,
               "Content-Type": "application/json", "Accept": "application/json"}
    r = client._session.post(f"{ACCOUNT_URL}{method}/", json=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def base_rates(client, per_call: int = 100) -> pd.DataFrame:
    """The next 24 hours' greyhound WIN and PLACE markets with their descriptions, a country at a time and at most
    ``per_call`` markets a call: Betfair refused 1,000 descriptions in one call (TOO_MUCH_DATA, run 42207915048)."""
    now = datetime.now(timezone.utc)
    raw = []
    for country in COUNTRIES:
        raw += client._api_call("listMarketCatalogue", {
            "filter": {"eventTypeIds": [GREYHOUNDS], "marketCountries": [country], "marketTypeCodes": ["WIN", "PLACE"],
                       "marketStartTime": {"from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                           "to": (now + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")}},
            "marketProjection": ["EVENT", "MARKET_DESCRIPTION"], "maxResults": per_call}) or []
    rows = []
    for m in raw:
        d, ev = m.get("description", {}) or {}, m.get("event", {}) or {}
        rows.append({"market": m.get("marketId"), "country": ev.get("countryCode"), "venue": ev.get("venue"),
                     "type": d.get("marketType"), "base_rate": d.get("marketBaseRate"),
                     "discount_allowed": d.get("discountAllowed")})
    return pd.DataFrame(rows)


def cleared_markets(client, days: int = 90) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    out, start = [], 0
    while True:
        reply = client._api_call("listClearedOrders", {
            "betStatus": "SETTLED", "eventTypeIds": [GREYHOUNDS], "groupBy": "MARKET", "includeItemDescription": True,
            "settledDateRange": {"from": (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "to": now.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "fromRecord": start, "recordCount": 1000}) or {}
        got = reply.get("clearedOrders", []) or []
        out += got
        if not reply.get("moreAvailable") or not got:
            break
        start += len(got)
    rows = []
    for o in out:
        it = o.get("itemDescription", {}) or {}
        rows.append({"market": o.get("marketId"), "settled": o.get("settledDate"), "bets": o.get("betCount"),
                     "profit": o.get("profit"), "commission": o.get("commission"),
                     "event": it.get("eventDesc"), "market_desc": it.get("marketDesc"),
                     "market_type": it.get("marketType")})
    df = pd.DataFrame(rows)
    for c in ("profit", "commission"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarise(acct: dict, rates: pd.DataFrame, cleared: pd.DataFrame) -> dict:
    rep = {"read_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "account": {k: acct.get(k) for k in ("discountRate", "pointsBalance", "currencyCode", "region",
                                                "countryCode")}}
    if len(rates):
        g = rates.groupby(["country", "type"]).agg(markets=("market", "size"),
                                                   base_rates=("base_rate", lambda x: dict(Counter(x))),
                                                   discount_allowed=("discount_allowed", lambda x: dict(Counter(x))))
        rep["market_base_rates"] = [{"country": c, "type": t, **r} for (c, t), r in g.to_dict("index").items()]
    else:
        rep["market_base_rates"] = "no data found: no greyhound markets in the next 24 hours"
    if len(cleared):
        win = cleared[(cleared.profit > 0)]
        paid = win[win.commission.fillna(0) > 0]
        rep["settled_markets"] = {
            "markets": int(len(cleared)), "from": str(cleared.settled.min()), "to": str(cleared.settled.max()),
            "net_profit": round(float(cleared.profit.sum()), 2), "commission": round(float(cleared.commission.sum()), 2),
            "markets_won": int(len(win)), "markets_charged": int(len(paid)),
            # the rate on a market's winnings, read both ways (Betfair's profit before or after its commission)
            "rate_on_profit": round(float(paid.commission.sum() / paid.profit.sum()), 5) if len(paid) else None,
            "rate_on_profit_plus_commission": round(float(paid.commission.sum() /
                                                          (paid.profit + paid.commission).sum()), 5) if len(paid) else None,
            "per_market_rate_on_profit": (paid.commission / paid.profit).round(4).value_counts().head(10).to_dict()
            if len(paid) else {},
            "markets_won_not_charged": int(len(win) - len(paid)),
        }
    else:
        rep["settled_markets"] = "no data found: no settled greyhound markets in the last 90 days"
    return rep


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from betfair_client import BetfairClient
    c = BetfairClient()
    c.login()
    errors = {}

    def attempt(name, fn, empty):
        """Each reading on its own: one Betfair refusal leaves the others standing (and is reported)."""
        try:
            return fn()
        except Exception as exc:
            errors[name] = str(exc)[:300]
            log.warning("%s not read (%s)", name, errors[name])
            return empty
    acct = attempt("account", lambda: _account(c, "getAccountDetails", {}), {})
    rates = attempt("market_base_rates", lambda: base_rates(c), pd.DataFrame())
    cleared = attempt("settled_markets", lambda: cleared_markets(c), pd.DataFrame())
    rep = summarise(acct, rates, cleared)
    if errors:
        rep["errors"] = errors
    out = Path("out/commission")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    if len(cleared):
        cleared.to_csv(out / "settled_markets.csv", index=False)
    md = ["## Betfair commission on greyhounds (read-only)", "", "```", json.dumps(rep, indent=1, default=str), "```"]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
