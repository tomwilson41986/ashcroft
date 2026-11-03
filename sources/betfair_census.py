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


#: The calls the read-only client permits (trading/exchange.py BetfairData._read); the census uses only these.
READ_CALLS = ("listMarketCatalogue",)


def census_racing_countries(data, hours: float = 24.0, top: int = 1000, now: datetime | None = None) -> dict:
    """Every country with Betfair horse racing win markets in the next ``hours``: one listMarketCatalogue a day of the
    window (the read-only client permits no listCountries; 2 Nov 2026: the first census failed on it), each the
    ``top`` win markets by matched money (Betfair's cap is 1,000), grouped by the event's country. The money is what
    has matched by the reading, not the final total, so a reading taken while a country races is the one to quote."""
    start = now or datetime.now(timezone.utc)
    end = start + timedelta(hours=hours)
    out: dict = {"read_utc": _iso(start), "hours": hours, "countries": [], "calls": 0, "capped_days": 0}
    by: dict[str, dict] = {}
    t = start
    while t < end:
        t2 = min(end, t + timedelta(hours=24))
        try:
            cat = data._read("listMarketCatalogue", {
                "filter": {"eventTypeIds": [HORSE_RACING], "marketTypeCodes": ["WIN"],
                           "marketStartTime": {"from": _iso(t), "to": _iso(t2)}},
                "maxResults": top, "sort": "MAXIMUM_TRADED", "marketProjection": ["EVENT", "MARKET_START_TIME"]}) or []
        except Exception as exc:                          # one day unread: the census goes on
            out.setdefault("errors", []).append(f"{_iso(t)}: {str(exc)[:160]}")
            cat = []
        out["calls"] += 1
        out["capped_days"] += int(len(cat) >= top)
        for m in cat:
            ev = m.get("event") or {}
            cc = ev.get("countryCode") or "?"
            r = by.setdefault(cc, {"country": cc, "matched": [], "venues": set(), "first": None})
            r["matched"].append(float(m.get("totalMatched") or 0.0))
            v = ev.get("venue") or ev.get("name")
            if v:
                r["venues"].add(v)
            st = m.get("marketStartTime")
            if st and (r["first"] is None or st < r["first"]):
                r["first"] = st
        t = t2
    for cc, r in by.items():
        matched = sorted(r["matched"], reverse=True)
        out["countries"].append({"country": cc, "win_markets": len(matched),
                                 "win_matched_total": round(sum(matched), 2),
                                 "win_matched_median": round(matched[len(matched) // 2], 2) if matched else 0.0,
                                 "venues": sorted(r["venues"]), "first_off_utc": r["first"]})
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
        print(r.get("country"), {k: r.get(k) for k in ("win_markets", "win_matched_total", "win_matched_median")},
              (r.get("venues") or [])[:8])
    print({k: got.get(k) for k in ("calls", "capped_days", "errors")})
    print("saved:", save(got, datetime.now(timezone.utc).date()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
