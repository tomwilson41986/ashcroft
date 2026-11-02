"""Betfair horse racing by country: every country Betfair lists racing in, its win markets, venues and the money matched
so far (read-only, the delayed key). The owner's ask of 1 Nov 2026: whether Hong Kong, Japan and the rest trade at all
before their form data is built (reports/international_racing_parse.md).

Kept apart from the live horse system (the owner, 2 Nov 2026: nothing in the live UK/IE model changes for any other
sport or country): this module only reads Betfair and imports the recorder's bucket and S3 helpers; it changes
nothing in betfair_recorder.py, and it runs in paper-models.yml on its own runner, never in live-trade.yml or
live-record.yml.

    python -m sources.betfair_census --hours 168          # the week ahead; saved to betfair_live/<day>/census_racing_<HHMM>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)
HORSE_RACING = "7"


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def census_racing_countries(data, hours: float = 24.0, top: int = 200, now: datetime | None = None) -> dict:
    """One listCountries, then one catalogue of each country's ``top`` win markets by matched money. The money is what
    has matched by the reading, not the final total, so a reading taken while a country races is the one to quote."""
    start = now or datetime.now(timezone.utc)
    end = start + timedelta(hours=hours)
    window = {"marketStartTime": {"from": _iso(start), "to": _iso(end)}}
    out: dict = {"read_utc": _iso(start), "hours": hours, "countries": []}
    for c in data._read("listCountries", {"filter": {**window, "eventTypeIds": [HORSE_RACING]}}) or []:
        cc = c.get("countryCode")
        row = {"country": cc, "markets": c.get("marketCount")}
        try:
            cat = data._read("listMarketCatalogue", {
                "filter": {**window, "eventTypeIds": [HORSE_RACING], "marketCountries": [cc], "marketTypeCodes": ["WIN"]},
                "maxResults": top, "sort": "MAXIMUM_TRADED", "marketProjection": ["EVENT", "MARKET_START_TIME"]}) or []
            matched = sorted((float(m.get("totalMatched") or 0.0) for m in cat), reverse=True)
            row["win_markets"] = len(cat)
            row["win_matched_total"] = round(sum(matched), 2)
            row["win_matched_median"] = round(matched[len(matched) // 2], 2) if matched else 0.0
            row["venues"] = sorted({(m.get("event") or {}).get("venue") or (m.get("event") or {}).get("name") or ""
                                    for m in cat} - {""})
            row["first_off_utc"] = min((m.get("marketStartTime") for m in cat if m.get("marketStartTime")), default=None)
        except Exception as exc:                          # one country unread: the census goes on
            row["error"] = str(exc)[:200]
        out["countries"].append(row)
    out["countries"].sort(key=lambda r: -(r.get("win_matched_total") or 0.0))
    return out


def save(result: dict, day: date, s3=None, root: Path | str | None = None, name: str | None = None) -> str:
    """Beside the day's record, under its own name: betfair_live/<day>/census_racing_<HHMM>.json, locally and in S3."""
    from betfair_recorder import PREFIX, ROOT, bucket, s3_client
    name = name or f"census_racing_{datetime.now(timezone.utc):%H%M}.json"
    body = json.dumps(result, indent=1, default=str).encode()
    path = Path(root or ROOT) / f"{day:%Y-%m-%d}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    key = f"{PREFIX}/{day:%Y-%m-%d}/{name}"
    try:
        (s3 or s3_client()).put_object(Bucket=bucket(), Key=key, Body=body)
    except Exception as exc:
        log.warning("census not copied to S3 (%s); kept at %s", exc, path)
        return str(path)
    return f"s3://{bucket()}/{key}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=24.0, help="the window ahead, hours")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from trading.exchange import BetfairData
    data = BetfairData()
    data.login()
    got = census_racing_countries(data, hours=a.hours)
    for r in got["countries"]:
        print(r.get("country"), {k: r.get(k) for k in ("markets", "win_markets", "win_matched_total",
                                                       "win_matched_median")}, (r.get("venues") or [])[:8])
    print("saved:", save(got, datetime.now(timezone.utc).date()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
