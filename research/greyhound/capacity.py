"""How much the early-lay forward test's lays could get matched as the stake grows (read-only; the owner, 8 Nov 2026:
"How much could we have got matched in the 'closed at the SP' model ... yes run the stake ladder").

The forward test's lays for each recorded day (research.greyhound.stream_lab.forward_test_orders: the time, the runner,
the limit price) are replayed through the recorded Exchange Stream (research.greyhound.stream_replay) at a fixed
liability a lay, ``LIABILITIES``, under two fill readings:
- ``full``: the replay's usual rule, a trade beyond our price fills the rest of the order at once;
- ``capped``: a trade beyond our price fills no more than the money that crossed it (the conservative reading, which
  matters as the order grows).
Our own orders' effect on the book (others reacting to a bigger lay) is not modelled. Each run reports, a liability
and a reading: lays placed and matched, the liability offered and matched, the share matched, and the P&L and return
on matched liability closed 5 s before the off, at the SP, and left to run to the result (the account's commission).

    python -m research.greyhound.capacity --out out/capacity [--days 2026-11-05,2026-11-06]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from research.greyhound import stream_lab, stream_replay as sr

log = logging.getLogger(__name__)

LIABILITIES = (20.0, 50.0, 100.0, 250.0)
READINGS = {"full": False, "capped": True}


def at_liability(orders: list, liability: float) -> list:
    """The same lays, each sized to ``liability`` (stake x (price - 1)), unplaced."""
    return [dataclasses.replace(o, size=round(liability / (o.price - 1.0), 2), tag=f"L{int(liability)}")
            for o in orders if o.price > 1.01]


def summarise(df: pd.DataFrame, orders: list) -> dict:
    offered = sum(o.size * (o.price - 1.0) for o in orders)
    m = df[df.matched > 0]
    liab = float((m.matched * (m.avg - 1.0)).sum()) if len(m) else 0.0

    def tot(col):
        v = pd.to_numeric(m.get(col), errors="coerce") if len(m) and col in m else pd.Series(dtype=float)
        return float(v.sum()) if len(v) else 0.0

    out = {"lays": len(orders), "matched_lays": int(len(m)), "liability_offered": round(offered, 2),
           "liability_matched": round(liab, 2), "share_matched": round(liab / offered, 4) if offered else None}
    for col, lab in (("pnl", "green"), ("pnl_sp", "sp"), ("pnl_result", "result")):
        p = tot(col)
        out[f"pnl_{lab}"] = round(p, 2)
        out[f"roi_{lab}%"] = round(100 * p / liab, 2) if liab > 0 else None
    return out


def ladder(lines, store, day: date, comm: float) -> list[dict]:
    base = stream_lab.forward_test_orders(store, day)
    if not base:
        return []
    rows = []
    for L in LIABILITIES:
        for name, cap in READINGS.items():
            orders = at_liability(base, L)
            df = sr.replay(lines, orders, vol_factor=1.0, comm=comm, through_cap=cap)
            rows.append({"day": f"{day}", "liability": L, "reading": name, **summarise(df, orders)})
            log.info("%s L%d %s: %s", day, L, name, rows[-1])
    return rows


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", default="", help="days, comma-separated (default: every complete recorded day)")
    ap.add_argument("--out", default="out/capacity")
    a = ap.parse_args(argv)
    from betfair_recorder import bucket, s3_client
    from greyhound.costs import commission
    from sources.common import Store
    s3, bk, store = s3_client(), bucket(), Store()

    def listing(prefix):
        out, tok = [], None
        while True:
            r = s3.list_objects_v2(Bucket=bk, Prefix=prefix, **({"ContinuationToken": tok} if tok else {}))
            out += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return out
            tok = r["NextContinuationToken"]

    today = datetime.now(timezone.utc).date().isoformat()
    days = [d.strip() for d in a.days.split(",") if d.strip()] or sorted(
        d for d in {k.split("/")[2] for k in listing(f"{sr.PREFIX}/") if k.count("/") >= 3} if d < today)
    rows = []
    for d in days:
        lines = sr.load_lines(lambda k: s3.get_object(Bucket=bk, Key=k)["Body"].read(), listing, date.fromisoformat(d))
        rows += ladder(lines, store, date.fromisoformat(d), commission())
    res = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "capacity.csv", index=False)
    md = ["## The early lay's lays at larger stakes, replayed through the stream (read-only)", "",
          f"Days: {', '.join(days) or 'none'}. A lay's liability fixed at each level; 'full': a trade beyond our price "
          "fills the rest, 'capped': only what crossed. Our own orders' effect on the book is not modelled. Returns on "
          "matched liability.", ""]
    if len(res):
        g = res.groupby(["liability", "reading"]).agg(
            days=("day", "nunique"), lays=("lays", "sum"), matched=("matched_lays", "sum"),
            offered=("liability_offered", "sum"), liab=("liability_matched", "sum"), green=("pnl_green", "sum"),
            sp=("pnl_sp", "sum"), result=("pnl_result", "sum")).reset_index()
        md += ["| Liability a lay | Reading | Days | Lays | Matched | Liability offered | Liability matched | Share | "
               "Closed 5 s out | At the SP | To the result |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples(index=False):
            pc = lambda v: f"GBP{v:+,.0f} ({100 * v / r.liab:+.2f}%)" if r.liab > 0 else "-"
            md.append(f"| GBP{r.liability:.0f} | {r.reading} | {r.days} | {r.lays} | {r.matched} | GBP{r.offered:,.0f} | "
                      f"GBP{r.liab:,.0f} | {r.liab / r.offered:.0%} | {pc(r.green)} | {pc(r.sp)} | {pc(r.result)} |")
        md += ["", "By day:", "", "```", res.to_string(index=False), "```"]
    else:
        md.append("no data found: no forward-test lays on the days asked")
    (out / "report.md").write_text("\n".join(md) + "\n")
    (out / "capacity.json").write_text(json.dumps(rows, indent=1, default=str))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
