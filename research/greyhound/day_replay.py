"""A whole day of the early lay, replayed on the recorded books (read-only; the owner, 15 Nov 2026: "How would the
early lays have performed today?", then "Yes do this").

The forward test's lays (greyhound/laytest.py: first traded book, the forecast drifters, laid at 80% of the forecast
SP) priced by the paper trader's live model (the prices the live early lay trades on), on each set of books the day
has: the market recorder's (re-timed) and the live key's (greyhound/book_record.py, every 5 s near the off). For each:

- the fill: ``traded`` (matched once the price trades) and ``through`` (matched once a later book offers our price:
  the fill the live reconciliation trusts);
- the close: ``last`` (greened at the best back price in the last book before the off) and ``sp`` (a back at the SP);
- ``spclose3``: through, closed at the SP, no lay under 3; and the short-priced gate (mid under 3) at 80% and 90%.

Each race's net after the account's commission on its positive net. Descriptive: no gate is read and nothing is
traded. Writes out/day_replay/{report.json, report.md}.

    python -m research.greyhound.day_replay --date 2026-11-15
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


def arms() -> list[tuple[str, dict, float | None]]:
    from greyhound.laytest import PRIMARY
    return [
        ("traded fill, greened (the running table's rule)", dict(PRIMARY), None),
        ("through fill, greened (the live rule)", dict(PRIMARY, fill="through"), None),
        ("through fill, closed at the SP (spclose)", dict(PRIMARY, fill="through", exit="sp"), None),
        ("through fill, closed at the SP, no lay under 3 (spclose3)", dict(PRIMARY, fill="through", exit="sp"), 3.0),
        ("through fill, greened, 90% of the forecast (m10)", dict(PRIMARY, fill="through", param=0.1), None),
        ("short-priced, 80%, through, greened (short20)", dict(PRIMARY, fill="through", gate="short"), None),
        ("short-priced, 90%, through, greened (short10)", dict(PRIMARY, fill="through", gate="short", param=0.1), None),
    ]


def run(store, day: date) -> dict:
    from greyhound.laytest import intraday
    out = {"day": f"{day:%Y-%m-%d}", "prices": "the paper trader's live model", "books": {}}
    for src in ("recorder", "live"):
        res = {}
        for name, rule, mp in arms():
            try:
                r = intraday(store, day, rule=rule, source=src, live_model=True, min_price=mp)
            except Exception as exc:                       # one arm's failure never hides the others
                res[name] = {"note": f"not scored ({str(exc)[:160]})"}
                continue
            if not r.get("markets"):
                res[name] = {"note": r.get("note", "no data found")}
                continue
            res[name] = {k: r[k] for k in ("races", "lays_placed", "matched", "liability", "pnl", "roi%", "source")}
            ms = r["markets"]
            res[name]["laid_dogs_won"] = int(sum(m["laid_dogs_won"] for m in ms))
            res[name]["low"] = min(m["running"] for m in ms)
            res[name]["high"] = max(m["running"] for m in ms)
            res[name]["last_race"] = f"{ms[-1]['race_time']} {ms[-1]['track']}"
        out["books"][src] = res
    return out


def report_md(out: dict) -> str:
    L = [f"# Early lay replay, {out['day']}", "",
         f"Prices: {out['prices']}. Each race's net after commission. Descriptive only.", ""]
    for src, res in out["books"].items():
        L += [f"## Books: {'the market recorder, re-timed' if src == 'recorder' else 'the live key'}", "",
              "| Version | Races | Lays | Matched | Laid dogs won | Liability | P&L | ROI % | Low | High | Last race |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            if "note" in r:
                L.append(f"| {name} | {r['note']} | | | | | | | | | |")
                continue
            L.append(f"| {name} | {r['races']} | {r['lays_placed']} | {r['matched']} | {r['laid_dogs_won']} | "
                     f"{r['liability']:,.2f} | {r['pnl']:+,.2f} | {r['roi%']} | {r['low']:+.2f} | {r['high']:+.2f} | "
                     f"{r['last_race']} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default=None)
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/day_replay")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    out = run(Store(root=a.store), day)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "report.json").write_text(json.dumps(out, indent=1, default=str))
    md = report_md(out)
    (o / "report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
